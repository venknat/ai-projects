# src/pytorch_util/simple_trainer.py

"""A utility to train a model with specified parameters."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List

import torch
import torch.nn as nn
import torchmetrics
from IPython.display import clear_output
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ._single_step_trainer import _SingleStepTrainer
from .model_util import ModelUtil


@dataclass
class SimpleTrainerResult:
    best_metric: float
    best_model_state: Dict[Any, Any]
    train_losses: List[float]
    train_metrics: List[float]
    validation_losses: List[float]
    validation_metrics: List[float]


# TODO: Make this more generalizable as I start doing cross-validation, etc, as well as allow
# more general models
#
# There are many undesirable restrictions here:
class SimpleTrainer:
    @staticmethod
    def train(
        device: torch.device,
        model: nn.Module,
        training_data: DataLoader,
        validation_data: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        train_metric: torchmetrics.Metric = None,
        validation_metric: torchmetrics.Metric = None,  # required if validation_data is given
        per_batch_callbacks: List[Callable] = None,
        per_epoch_callbacks: List[Callable] = None,
        print_progress: bool = False,
    ) -> SimpleTrainerResult:
        """
        A utility to train a sequential model with specified parameters.  Note that this does not
        attempt to do cross-validation or hyperparameter searching.  This is simply meant to be a
        reusable training loop.

        Args:
            model: nn.Module: The underlying model to train on
            training_data: DataLoader: The training data to train on
            validation_data: Dataloader: If given, will provide validation losses/metrics as well.
            optimizer: Optimizer: The optimizer used for training
            criterion: nn.Module: The loss function
            num_epochs: int: The number of epochs to train for
            metric: torchmetrics.Metric: The metric used for computing metrics. Optional
            per_batch_callbacks: List[Callable]: Optional code to be run with each batch
            per_epoch_callbacks: List[Callable]: Optional code to be run after each epoch
            print_progress: whether to print progress during training
            live_plot: boolean: whether to live-plot data as training is done.  Intended for use
                        in notebooks only.
        Returns:
            A Dictionary with keys "losses" giving the loss after each epoch, and "metrics" giving the
            computed metrics after each epoch.  Returns the best metric and model according to that
            metric.

            ***ASSUMES HIGHER METRIC VALUES ARE BETTER***
        """
        last_messages = []
        window: Final = 5
        best_metric = -float("inf")
        best_model_state = None
        result = SimpleTrainerResult(
            best_metric=-float("inf"),
            best_model_state=None,
            train_losses=[],
            train_metrics=[],
            validation_losses=[] if validation_data is not None else None,
            validation_metrics=[] if validation_data is not None else None,
        )

        for epoch in range(1, num_epochs + 1):
            if train_metric is not None:
                train_metric.reset()
            for examples, labels in training_data:
                _SingleStepTrainer.train(
                    device,
                    optimizer,
                    examples,
                    labels,
                    model,
                    criterion,
                    per_batch_callbacks,
                )

            (train_loss, train_metric_val) = ModelUtil.evaluate_model(
                device, training_data, model, criterion, train_metric
            )

            result.train_losses.append(train_loss)
            result.train_metrics.append(train_metric_val)
            if validation_data is not None:
                (validation_loss, validation_metric_val) = ModelUtil.evaluate_model(
                    device, validation_data, model, criterion, validation_metric
                )
                if validation_metric > best_metric:
                    best_model_state = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                    best_metric = validation_metric_val
                result.validation_losses.append(validation_loss)
                result.validation_metrics.append(validation_metric_val)
            else:
                if train_metric > best_metric:
                    best_model_state = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                    best_metric = train_metric_val

            if print_progress:
                # TODO: We should say something more descriptive than "metric"
                quality_message = f"Epoch {epoch}: loss: {train_loss}"
                if train_metric is not None:
                    quality_message += f", train metric: {train_metric_val}"
                if validation_metric is not None:
                    quality_message += f", validation metric: {validation_metric_val}"
                last_messages.append(quality_message)
                if len(last_messages) > window:
                    last_messages.pop(0)
                clear_output(wait=True)
                print("\n".join(last_messages), flush=True)
            if per_epoch_callbacks is not None:
                for cb in per_epoch_callbacks:
                    cb()
        result.best_model_state = best_model_state
        result.best_metric = best_metric
        return result
