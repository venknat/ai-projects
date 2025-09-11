from typing import Final, List, Callable

import torch
import torch.nn as nn
import torchmetrics

from torch.optim import Optimizer


class _SingleStepTrainer:
    """Runs a single training step on a single batch."""

    @staticmethod
    def train(
        optimizer: Optimizer,
        training_xs: torch.Tensor,
        training_ys: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        metric: torchmetrics.Metric,
        per_batch_callbacks: List[Callable],
    ):
        optimizer.zero_grad()
        training_xs = training_xs.view(training_xs.shape[0], -1)
        preds = model(training_xs)
        loss = criterion(preds, training_ys)
        metric.update(preds, training_ys)
        loss.backward()
        optimizer.step()
        if per_batch_callbacks is not None:
            for cb in per_batch_callbacks:
                cb()
