# src/pytorch_util/simple_trainer.py

"""A utility to train a model with specified parameters."""

from typing import Final, List, Callable, Dict

import torch
import torch.nn as nn
import torchmetrics

from ._single_step_trainer import _SingleStepTrainer

from torch.utils.data import DataLoader
from IPython.display import clear_output
from torch.optim import Optimizer

from .model_util import ModelUtil


# TODO: Make this more generalizable as I start doing cross-validation, etc, as well as allow
# more general models
#
# There are many undesirable restrictions here:
class SimpleTrainer:
    """A utility to train a sequential model with specified parameters.  Note that this does not
    attempt to do cross-validation or hyperparameter searching.  This is simply meant to be a
    reusable training loop."""

    @staticmethod
    def train(
        model: nn.Module,
        training_data: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        metric: torchmetrics.Metric = None,
        per_batch_callbacks: List[Callable] = None,
        per_epoch_callbacks: List[Callable] = None,
        print_progress: bool = False,
        live_plot: bool = False,
    ) -> Dict[str, List[float]]:
        """

        Args:
            model: nn.Module: The underlying model to train on
            training_data: DataLoader: The training data to train on
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
            computed metrics after each epoch.
        """
        last_messages = []
        window: Final = 5
        best_loss = float("inf")
        best_model_state = None
        result = {"losses": []}
        if metric is not None:
            result["metrics"] = []

        for epoch in range(1, num_epochs + 1):
            if metric is not None:
                metric.reset()
            for examples, labels in training_data:
                _SingleStepTrainer.train(
                    optimizer,
                    examples,
                    labels,
                    model,
                    criterion,
                    per_batch_callbacks,
                )

            train_loss = 0.0
            # After each epoch, output losses.  TODO: Add in accuracy.
            (train_loss, metric_val) = ModelUtil.evaluate_model(
                training_data, model, criterion, metric
            )

            result["losses"].append(train_loss)
            result["metrics"].append(metric_val)

            if train_loss < best_loss:
                best_model_state = model.state_dict()
                best_loss = train_loss

            if print_progress:
                # TODO: We should say something more descriptive than "metric"
                last_messages.append(
                    f"Epoch {epoch}: loss: {train_loss}, metric: {metric.compute()}"
                )
                if len(last_messages) > window:
                    last_messages.pop(0)
                clear_output(wait=True)
                print("\n".join(last_messages), flush=True)
            if per_epoch_callbacks is not None:
                for cb in per_epoch_callbacks:
                    cb()
        return result
