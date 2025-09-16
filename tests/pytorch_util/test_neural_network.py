from unittest.mock import Mock

import pytest
import test_util
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_util.neural_network import NeuralNetwork
from pytorch_util.simple_trainer import SimpleTrainer


class TestNeuralNetwork:
    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_neural_network(self, use_gpu):
        num_epochs = 6
        num_batches = 8
        batch_callback1 = Mock()
        batch_callback2 = Mock()

        per_epoch_callback1 = Mock()
        per_epoch_callback2 = Mock()

        device, dataloader, model, criterion, metric, optimizer = (
            self._setup_test_params(num_batches, True, True, use_gpu)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = SimpleTrainer.train(
            device,
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

    @staticmethod
    def _setup_test_params(num_batches, create_loss, create_metric, use_gpu):
        device = test_util.get_available_gpu() if use_gpu else torch.device("cpu")
        batch_size = 4
        num_classes = 10
        image_shape = (1, 28, 28)
        model = NeuralNetwork(
            input_shape=image_shape,
            num_output_classes=10,
            hidden_layer_sizes=[256, 512, 256],
        ).to(device=device)
        images = torch.rand(batch_size * num_batches, *image_shape)
        labels = torch.randint(0, num_classes, (batch_size * num_batches,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss() if create_loss else None
        metric = (
            torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            if create_metric
            else None
        ).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        return device, dataloader, model, criterion, metric, optimizer
