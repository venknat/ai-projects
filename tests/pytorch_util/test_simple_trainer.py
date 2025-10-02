from unittest.mock import Mock

import pytest
import test_util
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_util.simple_trainer import SimpleTrainer


class TestSimpleTrainer:
    @pytest.mark.parametrize(
        "use_gpu,use_validation",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_simple_trainer(self, use_gpu, use_validation):
        num_epochs = 6
        num_train_batches = 8
        num_validation_batches = 4
        batch_callback1 = Mock()
        batch_callback2 = Mock()

        per_epoch_callback1 = Mock()
        per_epoch_callback2 = Mock()
        image_shape = (1, 28, 28)
        num_classes = 10

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_shape[1] * image_shape[2], num_classes),
        )
        (
            device,
            train_dataloader,
            validation_dataloader,
            model,
            criterion,
            train_metric,
            validation_metric,
            optimizer,
        ) = test_util.create_simple_trainer_params(
            num_train_batches, num_validation_batches, model, True, True, use_gpu
        )

        result = SimpleTrainer.train(
            device,
            model,
            train_dataloader,
            validation_dataloader if use_validation else None,
            optimizer,
            criterion,
            num_epochs,
            train_metric,
            validation_metric if use_validation else None,
            [batch_callback1, batch_callback2],
            [per_epoch_callback1, per_epoch_callback2],
            print_progress=True,
        )

        assert len(result.train_losses) == num_epochs
        assert len(result.train_metrics) == num_epochs
        if use_validation:
            assert len(result.validation_losses) == num_epochs
            assert len(result.validation_metrics) == num_epochs
        else:
            assert result.validation_losses is None
            assert result.validation_metrics is None
        assert batch_callback1.call_count == num_epochs * num_train_batches
        assert batch_callback2.call_count == num_epochs * num_train_batches
        assert per_epoch_callback1.call_count == num_epochs
        assert per_epoch_callback2.call_count == num_epochs
