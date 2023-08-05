"""This module consists of functions that are specific to the PyTorch
library, encapsulating most of the repetitive tasks that I typically
had to replicate across projects.
"""

from typing import Dict, Any
from pprint import pprint
import torch
from torch import nn
from torch.optim import Optimizer
from .types import Metric


# pylint: disable=too-many-arguments
def train(inputs: torch.Tensor,
          labels: torch.Tensor,
          model: nn.Module,
          optimizer: Optimizer,
          loss_fn: nn.Module | None,
          epochs: int = 3,
          device: str = "cuda",
          model_kwargs: Dict[str, Any] | None = None): 
    """Basic training loop in PyTorch which should work with any basic
    classifier.

    Args:
        inputs (torch.Tensor): input tensor
        labels (torch.Tensor): labels for each input sample
        model (nn.Module): model instance to be used
        optimizer (Optimizer): optimizer
        loss_fn (nn.Module | None): loss function
        epochs (int, optional): number of epochs to train for. Defaults to 3.
        device (str, optional): which device to train on. Defaults to "cuda".
        model_kwargs (Dict[str, Any], optional): additional keyword args
                                                 to be fed to the model.
                                                 Defaults to an empty dict.
    """

    if model_kwargs is None:
        model_kwargs = dict()

    inputs = inputs.to(device)
    labels = labels.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs, **model_kwargs)
        # TODO: add basic metrics collection
        if loss_fn is None:
            # assume loss is given to us as part of the outputs,
            # that's what Huggingface transformers API does for example
            loss = outputs.loss
        else:
            loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        pprint(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
