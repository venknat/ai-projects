from typing import NamedTuple
from collections import namedtuple

import torch
import torchmetrics

from torch import nn
from torch.utils.data import DataLoader

ModelResult = namedtuple("TrainResult", ["loss", "metric"])


class ModelUtil:
    @staticmethod
    def evaluate_model(
        device: torch.device,
        data: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        metric: torchmetrics.Metric,
    ) -> ModelResult:
        loss = None if criterion is None else 0.0
        metric_val = None if metric is None else 0.0
        with torch.no_grad():
            for examples, labels in data:
                examples = examples.to(device=device)
                labels = labels.to(device=device)
                examples = examples.view(examples.shape[0], -1)
                preds = model(examples)
                if criterion is not None:
                    loss += criterion(preds, labels).item() * examples.size(0)
                if metric is not None:
                    metric.update(preds, labels)
        if loss is not None:
            loss /= len(data.dataset)
        if metric is not None:
            metric_val = metric.compute()

        return ModelResult(loss=loss, metric=metric_val)
