from typing import Callable, Final, List

import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Optimizer


class _SingleStepTrainer:
    """Runs a single training step on a single batch."""

    @staticmethod
    def train(
        device: torch.device,
        optimizer: Optimizer,
        training_xs: torch.Tensor,
        training_ys: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        per_batch_callbacks: List[Callable],
    ):
        optimizer.zero_grad()
        training_xs = training_xs.to(device=device)
        training_ys = training_ys.to(device=device)
        training_xs = training_xs.view(training_xs.shape[0], -1)
        preds = model(training_xs)
        loss = criterion(preds, training_ys)
        loss.backward()
        optimizer.step()
        if per_batch_callbacks is not None:
            for cb in per_batch_callbacks:
                cb()
