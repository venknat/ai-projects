# src/pytorch_util/simple_trainer.py

"""A utility to train a model with specified parameters."""

from typing import Final, List, Callable

import torch
import torch.nn as nn
import torchmetrics

from torch.utils.data import DataLoader
from IPython.display import clear_output
from torch.optim import Optimizer


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
        live_plot: bool = False,
    ) -> List[float]:
        """

        Args:
            model: nn.Module: The underlying model to train on
            training_data: DataLoader: The training data to train on
            optimizer: Optimizer: The optimizer used for training
            criterion: nn.Module: The loss function
            num_epochs: int: The number of epochs to train for
            metric: torchmetrics.Metric: The metric used for computing metrics. Optional
            live_plot: boolean: whether to live-plot data as training is done.  Intended for use
                       in notebooks only.
        Returns:
            TBD
        """
        print("Training...")
        last_messages = []
        window: Final = 5
        best_loss = float("inf")
        best_model_state = None
        losses = []

        for epoch in range(1, num_epochs + 1):
            if metric is not None:
                metric.reset()
            for iteration_num, (examples, labels) in enumerate(training_data):
                optimizer.zero_grad()
                examples = examples.view(examples.shape[0], -1)
                preds = model(examples)
                loss = criterion(preds, labels)
                metric.update(preds, labels)
                loss.backward()
                optimizer.step()
                if per_batch_callbacks is not None:
                    for cb in per_batch_callbacks:
                        cb()

            train_loss = 0.0
            # After each epoch, output losses.  TODO: Add in accuracy.
            with torch.no_grad():
                for examples, labels in training_data:
                    examples = examples.view(examples.shape[0], -1)
                    preds = model(examples)
                    train_loss += criterion(preds, labels).item() * examples.size(0)

            train_loss /= len(training_data.dataset)
            losses.append(train_loss)
            if train_loss < best_loss:
                best_model_state = model.state_dict()
                best_loss = train_loss
            last_messages.append(
                f"Epoch {epoch}: loss: {train_loss}, accuracy: {metric.compute()}"
            )
            if len(last_messages) > window:
                last_messages.pop(0)
            clear_output(wait=True)
            print("\n".join(last_messages), flush=True)
            if per_epoch_callbacks is not None:
                for cb in per_epoch_callbacks:
                    cb()
        return losses
