import math
from collections import OrderedDict, namedtuple
from typing import List, Union

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
        """
        Evaluates a model on the given loss and metric on the given labeled data.

        Args:
            device: the device on which the entities are (needed to know where to move the
            underlying data to.
            data: the data to evaluate
            model: the model to evaluate
            criterion: the loss function
            metric: the metric
        """
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
