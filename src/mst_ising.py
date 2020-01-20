import sys

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterable, Tuple

from tqdm import trange


class GenerativeModel(nn.Module, ABC):
    """A simple super class for a PyTorch generative model."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log probability for x."""
        pass


class MSTIsingModel(GenerativeModel):
    """A learned maximum-spanning tree approximation of the Ising model."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the model.

        The params argument is a dictionary with the following keys:

        "shape": The size of the grid.
        "dtype": The torch datatype to be used.
        "device": The device to be used with torch.
        "sigma_init": The standard deviation for initializing the weights.
        "sigma_weights": The standard deviation for the weights to be used for caluclating the MST.

        :param params: The parameter dictionary.
        """
        super().__init__()
        self.params = params
        self.image_shape = params['shape']
        self.point_wise = nn.Parameter(torch.randn(self.image_shape, dtype=params['dtype']) * params['sigma_init'])
        self.horizontal_links = nn.Parameter(torch.randn((self.image_shape[0],
                                                          self.image_shape[1] - 1),
                                                         dtype=params['dtype']) * params['sigma_init'])
        self.vertical_links = nn.Parameter(torch.randn((self.image_shape[0] - 1,
                                                        self.image_shape[1]),
                                                       dtype=params['dtype']) * params['sigma_init'])
        self.horizontal_mu = nn.Parameter(torch.zeros((self.image_shape[0], self.image_shape[1] - 1),
                                                      dtype=params['dtype']))
        self.vertical_mu = nn.Parameter(torch.zeros((self.image_shape[0] - 1, self.image_shape[1]),
                                                    dtype=params['dtype']))
        self.horizontal_sampler = dist.Normal(loc=self.horizontal_mu, scale=params['sigma_weights'])
        self.vertical_sampler = dist.Normal(loc=self.vertical_mu, scale=params['sigma_weights'])
        self.edge_list = []
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1] - 1):
                self.edge_list.append(((i, j), (i, j + 1)))
        for j in range(self.image_shape[1]):
            for i in range(self.image_shape[0] - 1):
                self.edge_list.append(((i, j), (i + 1, j)))
        self.edge_list = np.array(self.edge_list)
        self._curr_tree = None
        self._curr_log_denominator = None

    @property
    def tree(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the MST based on the current weight means.
        :return: The tree as a tuple of two matrices, one for the horizontal edges, the other for the vertical.
        """
        if self._curr_tree is not None:
            return self._curr_tree
        else:
            return self._kruskal(self.horizontal_mu, self.vertical_mu)

    @property
    def _log_denominator(self) -> torch.Tensor:
        """
        Get the log-denominator based on the current horizontal_mu and vertical_mu
        :return: The log-denominator as a tensor (possibly fetching from cache)
        """
        if self._curr_log_denominator is not None:
            return self._curr_log_denominator
        else:
            return self._calculate_log_denom(self.tree)

    def train(self, mode=True):
        """Invalidates/builds the tree cache."""
        super().train(mode)
        if mode:
            self._curr_tree = None
            self._curr_log_denominator = None
        else:
            self._curr_tree = self._kruskal(self.horizontal_mu, self.vertical_mu)
            self._curr_log_denominator = self._calculate_log_denom(self._curr_tree)
        return self

    def forward(self, x: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the average log-probability for the batch and the weights for each weight sample.

        :param x: The batch (batch x *image_shape)
        :param n_samples: The number of weight samples to be averaged
        :return: A pair of tensors representing the data log-probability (batch_size, n_samples) and the weights
                 log-probability (n_samples, ).
        """
        conditioned_logprobs = []
        tree_logprobs = []
        for i in range(n_samples):
            horizontal_sample = self.horizontal_sampler.sample()
            vertical_sample = self.vertical_sampler.sample()
            h_log_prob = self.horizontal_sampler.log_prob(horizontal_sample).sum()
            v_log_prob = self.vertical_sampler.log_prob(vertical_sample).sum()
            tree = self._kruskal(horizontal_sample, vertical_sample)
            denominator = self._calculate_log_denom(tree)
            numerator = self._calculate_log_num(x, tree)
            conditioned_logprobs.append(numerator - denominator)
            tree_logprobs.append(h_log_prob + v_log_prob)
        return torch.stack(conditioned_logprobs).t(), torch.stack(tree_logprobs)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of images, get the average log-probability using the learned tree.
        :param x: The batch of images.
        :return: The average log-probability (batch_size, )
        """
        tree = self.tree
        denominator = self._log_denominator
        numerator = self._calculate_log_num(x, tree)
        return numerator - denominator

    def _get_order_gen(self) -> Generator[Tuple[int, int], None, None]:
        """Get the point generator for Gibbs sampling."""
        all_points = [(x, y) for x in range(self.image_shape[0]) for y in range(self.image_shape[1])]
        while True:
            yield all_points[np.random.choice(len(all_points))]

    @torch.no_grad()
    def sample(self, burn_in: int = 100000, n_iters: int = 10000) -> np.ndarray:
        """
        Generate samples using Gibbs sampling.

        :param burn_in: The number of initial samples to discard.
        :param n_iters: The number of iterations after burn-in.
        :return: A numpy array of n_iters samples
        """
        point_iterator = self._get_order_gen()
        tree = self.tree
        log_denominator = self._log_denominator.item()
        bar = trange(burn_in, file=sys.stdout, desc=f'Burn-in log-prob: NaN', leave=False)
        curr_guess = (np.random.random(self.image_shape) < 0.5).astype(np.int64)
        for _ in bar:
            curr_guess, new_lp = self._gibbs_step(curr_guess, next(point_iterator), tree, log_denominator)
            bar.set_description(f'Burn-in log-prob: {new_lp}')
        samples = []
        bar = trange(n_iters, file=sys.stdout, desc=f'Gibbs-chain log-prob: NaN', leave=False)
        for _ in bar:
            new_guess, new_lp = self._gibbs_step(curr_guess, next(point_iterator), tree, log_denominator)
            samples.append(new_guess)
            bar.set_description(f'Gibbs-chain log-prob: {new_lp}')
            curr_guess = new_guess
        return np.array(samples)

    def _gibbs_step(self, curr_guess: np.ndarray, point: Tuple[int, int],
                    tree: Tuple[torch.Tensor, torch.Tensor], log_denominator: float) -> Tuple[np.ndarray, float]:
        """Run a single Gibbs step, returning the new sample and the log-probability."""
        batch = torch.tensor(curr_guess[None, :, :], device=self.params['device'], dtype=torch.int64)
        new_guess = curr_guess.copy()
        batch[0][point] = 1
        log_num_one = self._calculate_log_num(batch, tree)
        batch[0][point] = 0
        log_num_zero = self._calculate_log_num(batch, tree)
        total = torch.logsumexp(torch.stack([log_num_one, log_num_zero]), dim=0)
        if np.random.random() < np.exp(log_num_one - total):
            new_guess[point] = 1
            new_lp = log_num_one - log_denominator
        else:
            new_guess[point] = 0
            new_lp = log_num_zero - log_denominator
        return new_guess, new_lp.item()

    def _kruskal(self, horizontal_weights: torch.Tensor,
                 vertical_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the maximum spanning tree using Kruskal's algorithm.
        :param horizontal_weights: The weights for the horizontal links.
        :param vertical_weights: The weights for the vertical links
        :return: The generated tree as a tuple of active horizontal and vertical links.
        """
        all_weights = torch.cat([horizontal_weights.view(-1), vertical_weights.view(-1)])
        indices = torch.argsort(all_weights, descending=True)
        edges = self.edge_list[indices.cpu()]
        horizontal_edges = torch.zeros_like(horizontal_weights, dtype=torch.int64)
        vertical_edges = torch.zeros_like(vertical_weights, dtype=torch.int64)
        union_find = {(x, y): (x, y) for y in range(self.image_shape[1]) for x in range(self.image_shape[0])}

        def _find(p: Tuple[int, int]) -> Tuple[int, int]:
            """Find operation with path-compression."""
            curr = p
            while union_find[curr] != curr:
                curr = union_find[curr]
            union_find[p] = curr
            return curr

        def _union(p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
            """Union of two sets."""
            s1 = _find(p1)
            s2 = _find(p2)
            if s1 != s2:
                union_find[s1] = s2

        for p1, p2 in edges:
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            if not (_find(p1) == _find(p2)):
                _union(p1, p2)
                if p1[0] == p2[0]:
                    horizontal_edges[p1] = 1
                else:
                    vertical_edges[p1] = 1
        return horizontal_edges, vertical_edges

    def _calculate_log_num(self, x: torch.Tensor, tree: Tuple[torch.Tensor, torch.Tensor]):
        """Get the numerator for the log-probability."""
        dtype = self.params['dtype']
        point_wise = (x.type(dtype) * self.point_wise).sum(dim=(-1, -2))
        horizontal = ((x[:, :, :-1] == x[:, :, 1:]).type(dtype) * tree[0].float() *
                      self.horizontal_links).sum(dim=(-1, -2))
        vertical = ((x[:, :-1, :] == x[:, 1:, :]).type(dtype) * tree[1].float() *
                    self.vertical_links).sum(dim=(-1, -2))
        return point_wise + horizontal + vertical

    def _calculate_log_denom(self, tree: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Get the denominator for the log-probability calculated using recursion."""
        visited = np.zeros(self.image_shape, dtype=np.uint8)

        def _func(curr: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
            """The recursive function that calculates the log-probability for every node below 'curr'."""
            zero_lp, one_lp = torch.tensor(0., dtype=self.params['dtype'],
                                           device=self.params['device']), self.point_wise[curr]
            visited[curr] = 1
            if curr[0] > 0 and not visited[(curr[0] - 1, curr[1])] and tree[1][curr[0] - 1, curr[1]]:
                zero_next, one_next = _func((curr[0] - 1, curr[1]))
                zero_lp = zero_lp + torch.logsumexp(torch.stack([one_next,
                                                                 zero_next + self.vertical_links[(curr[0] - 1,
                                                                                                  curr[1])]]), dim=0)
                one_lp = one_lp + torch.logsumexp(torch.stack([zero_next,
                                                               one_next + self.vertical_links[(curr[0] - 1,
                                                                                               curr[1])]]), dim=0)
            if curr[0] < self.image_shape[0] - 1 and not visited[(curr[0] + 1, curr[1])] and tree[1][curr[0], curr[1]]:
                zero_next, one_next = _func((curr[0] + 1, curr[1]))
                zero_lp = zero_lp + torch.logsumexp(torch.stack([one_next,
                                                                 zero_next + self.vertical_links[(curr[0],
                                                                                                  curr[1])]]), dim=0)
                one_lp = one_lp + torch.logsumexp(torch.stack([zero_next,
                                                               one_next + self.vertical_links[(curr[0],
                                                                                               curr[1])]]), dim=0)
            if curr[1] > 0 and not visited[(curr[0], curr[1] - 1)]  and tree[0][curr[0], curr[1] - 1]:
                zero_next, one_next = _func((curr[0], curr[1] - 1))
                zero_lp = zero_lp + torch.logsumexp(torch.stack([one_next,
                                                                 zero_next + self.horizontal_links[(curr[0],
                                                                                                    curr[1] - 1)]]),
                                                    dim=0)
                one_lp = one_lp + torch.logsumexp(torch.stack([zero_next,
                                                               one_next + self.horizontal_links[(curr[0],
                                                                                                 curr[1] - 1)]]),
                                                  dim=0)
            if curr[1] < self.image_shape[1] - 1 and not visited[(curr[0], curr[1] + 1)] and tree[0][curr[0], curr[1]]:
                zero_next, one_next = _func((curr[0], curr[1] + 1))
                zero_lp = zero_lp + torch.logsumexp(torch.stack([one_next,
                                                                 zero_next + self.horizontal_links[(curr[0],
                                                                                                    curr[1])]]), dim=0)
                one_lp = one_lp + torch.logsumexp(torch.stack([zero_next,
                                                               one_next + self.horizontal_links[(curr[0],
                                                                                                 curr[1])]]), dim=0)
            return zero_lp, one_lp

        return torch.logsumexp(torch.stack(_func((0, 0))), dim=0)


class Classifier(nn.Module):
    """A classifier that takes in a list of generative models, one for each class and uses them to predict classes."""

    def __init__(self, generative_models: Iterable[GenerativeModel]) -> None:
        """
        Initialize the classifier
        :param generative_models: A list of PyTorch generative models.
        """
        super().__init__()
        self.generative_models = nn.ModuleList(generative_models)
        self.n_classes = len(self.generative_models)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """
        Predict the class for the input x
        :param x: The input x
        :return: The class index and the log-probabilities (batch_size, n_classes)
        """
        log_probs = [model.log_prob(x) for model in self.generative_models]
        conditional_lps = torch.log_softmax(torch.stack(log_probs), dim=0)
        return conditional_lps.t()
