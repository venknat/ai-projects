from unittest.mock import Mock

import pytest
import torchmetrics

from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_util._single_step_trainer import _SingleStepTrainer
from pytorch_util.simple_trainer import SimpleTrainer


class TestSimpleTrainer:
    def test_simple_trainer(self):
        num_epochs = 5
        num_batches = 8
        batch_callback1 = Mock()
        batch_callback2 = Mock()

        per_epoch_callback1 = Mock()
        per_epoch_callback2 = Mock()

        dataloader, model, criterion, metric, optimizer = self._setup_test_params(
            num_batches, True, True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = SimpleTrainer.train(
            model,
            dataloader,
            optimizer,
            criterion,
            num_epochs,
            metric,
            [batch_callback1, batch_callback2],
            [per_epoch_callback1, per_epoch_callback2],
            print_progress=True,
        )

        assert "losses" in result
        assert "metrics" in result
        assert len(result["losses"]) == num_epochs
        assert len(result["metrics"]) == num_epochs
        assert batch_callback1.call_count == num_epochs * num_batches
        assert batch_callback2.call_count == num_epochs * num_batches
        assert per_epoch_callback1.call_count == num_epochs
        assert per_epoch_callback2.call_count == num_epochs

    def _setup_test_params(self, num_batches, create_loss, create_metric):
        batch_size = 4
        num_classes = 10
        image_shape = (1, 28, 28)
        model = nn.Sequential(
            nn.Linear(image_shape[1] * image_shape[2], num_classes),
        )
        images = torch.rand(batch_size * num_batches, *image_shape)
        labels = torch.randint(0, num_classes, (batch_size * num_batches,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss() if create_loss else None
        metric = (
            torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            if create_metric
            else None
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        return (dataloader, model, criterion, metric, optimizer)
