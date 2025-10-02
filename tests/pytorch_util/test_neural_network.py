from unittest.mock import Mock

import pytest
import test_util
import torch
import torchmetrics
from test_util import create_categorical_dataloader
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_util.neural_network import NeuralNetwork
from pytorch_util.simple_trainer import SimpleTrainer


class TestNeuralNetwork:
    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_neural_network(self, use_gpu):
        num_epochs = 6
        num_train_batches = 8
        num_validation_batches = 2
        batch_callback1 = Mock()
        batch_callback2 = Mock()
        image_shape = (1, 28, 28)

        per_epoch_callback1 = Mock()
        per_epoch_callback2 = Mock()
        model = NeuralNetwork(
            input_shape=image_shape,
            num_output_classes=10,
            hidden_layer_sizes=[256, 512, 256],
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = SimpleTrainer.train(
            device,
            model,
            train_dataloader,
            validation_dataloader,
            optimizer,
            criterion,
            num_epochs,
            train_metric,
            validation_metric,
            [batch_callback1, batch_callback2],
            [per_epoch_callback1, per_epoch_callback2],
            print_progress=True,
        )

        assert len(result.train_losses) == num_epochs
        assert len(result.train_metrics) == num_epochs
        assert len(result.validation_losses) == num_epochs
        assert len(result.validation_metrics) == num_epochs
        assert batch_callback1.call_count == num_epochs * num_train_batches
        assert batch_callback2.call_count == num_epochs * num_train_batches
        assert per_epoch_callback1.call_count == num_epochs
        assert per_epoch_callback2.call_count == num_epochs
