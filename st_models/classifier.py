import torch
import torch.nn as nn

from typing import Iterable

from ._generative_model import BinaryGenerativeModel


class Classifier(nn.Module):
    """A classifier that takes in a list of generative models, one for each class and uses them to predict classes."""

    def __init__(self, generative_models: Iterable[BinaryGenerativeModel]) -> None:
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
