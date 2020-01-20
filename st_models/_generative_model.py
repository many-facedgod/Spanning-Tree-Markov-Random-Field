import sys

import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Tuple

from tqdm import trange


class BinaryGenerativeModel(nn.Module, ABC):
    """An abstract super class for a PyTorch generative model for 2D binary images."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the model with the parameters. Must have the following keys:

        "shape": The size of the grid.
        "dtype": The torch datatype to be used.
        "device": The device to be used with torch.

        :param params: The parameters dictionary.
        """
        nn.Module.__init__(self)
        self.params = params
        self.image_shape = params['shape']

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log probability for x."""
        pass

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
        bar = trange(burn_in, file=sys.stdout, desc=f'Burn-in log-prob: NaN', leave=False)
        curr_guess = (np.random.random(self.image_shape) < 0.5).astype(np.int64)
        for _ in bar:
            curr_guess, new_lp = self._gibbs_step(curr_guess, next(point_iterator))
            bar.set_description(f'Burn-in log-prob: {new_lp}')
        samples = []
        bar = trange(n_iters, file=sys.stdout, desc=f'Gibbs-chain log-prob: NaN', leave=False)
        for _ in bar:
            new_guess, new_lp = self._gibbs_step(curr_guess, next(point_iterator))
            samples.append(new_guess)
            bar.set_description(f'Gibbs-chain log-prob: {new_lp}')
            curr_guess = new_guess
        return np.array(samples)

    def _gibbs_step(self, curr_guess: np.ndarray, point: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Run a single Gibbs step, returning the new sample and the log-probability."""
        batch = torch.tensor(curr_guess[None, :, :], device=self.params['device'], dtype=torch.int64)
        new_guess = curr_guess.copy()
        batch[0][point] = 1
        log_p_one = self.log_prob(batch)
        batch[0][point] = 0
        log_p_zero = self.log_prob(batch)
        total = torch.logsumexp(torch.stack([log_p_one, log_p_zero]), dim=0)
        if np.random.random() < np.exp(log_p_one - total):
            new_guess[point] = 1
            new_lp = log_p_one
        else:
            new_guess[point] = 0
            new_lp = log_p_zero
        return new_guess, new_lp.item()
