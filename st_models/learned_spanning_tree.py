import torch
import torch.distributions as dist
import torch.nn as nn

from typing import Any, Dict, Tuple

from ._spanning_tree_ising import SpanningTreeIsing


class LearnedSpanningTreeModel(SpanningTreeIsing):
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
        super().__init__(params)
        self.horizontal_mu = nn.Parameter(torch.zeros((self.image_shape[0], self.image_shape[1] - 1),
                                                      dtype=params['dtype']))
        self.vertical_mu = nn.Parameter(torch.zeros((self.image_shape[0] - 1, self.image_shape[1]),
                                                    dtype=params['dtype']))
        self.horizontal_sampler = dist.Normal(loc=self.horizontal_mu, scale=params['sigma_weights'])
        self.vertical_sampler = dist.Normal(loc=self.vertical_mu, scale=params['sigma_weights'])
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

    def train(self, mode: bool = True):
        """Invalidates/builds the tree and denominator caches apart from setting the training status."""
        super().train(mode)
        if mode:
            self._curr_tree = None
            self._curr_log_denominator = None
        else:
            self._curr_tree = self.tree
            self._curr_log_denominator = self._log_denominator
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
