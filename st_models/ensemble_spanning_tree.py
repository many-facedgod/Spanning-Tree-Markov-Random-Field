import numpy as np
import torch
import torch.nn as nn

from typing import Any, Dict, List

from ._spanning_tree_ising import SpanningTreeIsing


class EnsembleSpanningTreeModel(SpanningTreeIsing):
    """An ensemble of spanning tree approximations for an Ising model."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the model.

        The params argument is a dictionary with the following keys:

        "shape": The size of the grid.
        "dtype": The torch datatype to be used.
        "device": The device to be used with torch.
        "n_trees": The number of trees in the ensemble.

        :param params: The parameter dictionary.
        """
        super().__init__(params)
        device = params['device']
        self.n_trees = params['n_trees']
        self.prior_logits = nn.Parameter(torch.randn((self.n_trees, ), dtype=params['dtype']))
        self.trees = [self._kruskal(torch.randn((self.image_shape[0], self.image_shape[1] - 1), device=device),
                                    torch.randn((self.image_shape[0] - 1, self.image_shape[1]), device=device))
                      for _ in range(self.n_trees)]
        self._curr_log_denominators = None

    @property
    def _log_denominators(self) -> List[torch.Tensor]:
        """The log denominators based on the current weights. Possibly cached."""
        if self._curr_log_denominators is None:
            return [self._calculate_log_denom(t) for t in self.trees]
        else:
            return self._curr_log_denominators

    def train(self, mode: bool = True):
        """Invalidates/builds the log denominator cache apart from setting the training status."""
        super().train(mode)
        if mode:
            self._curr_log_denominators = None
        else:
            self._curr_log_denominators = self._log_denominators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The same as the call to log_prob."""
        return self.log_prob(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log-probabilities for a batch of inputs."""
        weights = torch.log_softmax(self.prior_logits, dim=0)
        log_probs = []
        for denom, tree in zip(self._log_denominators, self.trees):
            num = self._calculate_log_num(x, tree)
            log_probs.append(num - denom)
        weighted_probs = weights.view(-1, 1) + torch.stack(log_probs)
        return torch.logsumexp(weighted_probs - np.log(self.n_trees), dim=0)
