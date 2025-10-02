import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, random_split

from pytorch_util.simple_trainer import SimpleTrainer, SimpleTrainerResult


@dataclass
class GridsearchResult:
    best_hyperparams: Tuple[Any]
    results: Dict[Tuple[Any], SimpleTrainerResult]


# For now, this only supports learning rate and weight decay
def gridsearch_model(
    device: torch.device,
    model_factory,  # should be a function that creates a model
    training_data,  # Something that can be random_split and loaded into a training dataloader
    num_classes,
    train_valid_split,
    batch_size: int,
    num_epochs: int,
    learning_rates: List[int],
    weight_decays: List[int],
    print_progress: bool = False,
):
    result = GridsearchResult(best_hyperparams=None, results={})
    best_hyperparams = None
    best_metric = None
    for learning_rate, weight_decay in itertools.product(learning_rates, weight_decays):
        model = model_factory().to(device)
        train_dataset, validation_dataset = random_split(
            training_data, train_valid_split
        )
        training_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
        optimizer = create_optimizer("adam", model, learning_rate, weight_decay)
        criterion = nn.CrossEntropyLoss()
        train_metric = create_metric("accuracy", "multiclass", num_classes).to(device)
        validation_metric = create_metric("accuracy", "multiclass", num_classes).to(
            device
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        r = SimpleTrainer.train(
            device,
            model,
            training_dataloader,
            validation_dataloader,
            optimizer,
            criterion,
            num_epochs,
            train_metric,
            validation_metric,
            per_epoch_callbacks=[lambda: scheduler.step()],
            print_progress=print_progress,
        )
        if best_hyperparams is None or r.best_metric > best_metric:
            best_metric = r.best_metric
            best_hyperparams = (learning_rate, weight_decay)
        print(
            f"learning rate: {learning_rate}, weight_decay: {weight_decay}, metric: {r.best_metric} train:{r.train_metrics[-1]}"
        )
        result.results[(learning_rate, weight_decay)] = r

    result.best_hyperparams = best_hyperparams
    return result


def create_optimizer(
    optimizer_name: str, model: nn.Module, lr: float, weight_decay: float, **kwargs
):
    if optimizer_name == "adam":
        return Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer requested")


def create_metric(metric_name: str, task: str, num_classes: int):
    if metric_name == "accuracy":
        return torchmetrics.Accuracy(task=task, num_classes=num_classes)
    else:
        raise ValueError("Unsupported metric requested")


def create_loss(loss_name: str):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss requested")
