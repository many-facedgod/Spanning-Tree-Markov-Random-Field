import numpy as np
import torch
import torch.nn as nn

from abc import ABC
from typing import Any, Dict, Tuple

from ._generative_model import BinaryGenerativeModel


class SpanningTreeIsing(BinaryGenerativeModel, ABC):
    """Abstract class for sppanning tree approximation of the Ising model."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the model.

        The params argument is a dictionary with the following keys:

        "shape": The size of the grid.
        "dtype": The torch datatype to be used.
        "device": The device to be used with torch.
        "sigma_init": The standard deviation for initializing the weights.

        :param params: The parameter dictionary.
        """
        BinaryGenerativeModel.__init__(self, params)
        self.point_wise = nn.Parameter(torch.randn(self.image_shape, dtype=params['dtype']) * params['sigma_init'])
        self.horizontal_links = nn.Parameter(torch.randn((self.image_shape[0],
                                                          self.image_shape[1] - 1),
                                                         dtype=params['dtype']) * params['sigma_init'])
        self.vertical_links = nn.Parameter(torch.randn((self.image_shape[0] - 1,
                                                        self.image_shape[1]),
                                                       dtype=params['dtype']) * params['sigma_init'])
        self.edge_list = []
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1] - 1):
                self.edge_list.append(((i, j), (i, j + 1)))
        for j in range(self.image_shape[1]):
            for i in range(self.image_shape[0] - 1):
                self.edge_list.append(((i, j), (i + 1, j)))
        self.edge_list = np.array(self.edge_list)

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
