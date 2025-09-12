import pytest
import torchmetrics

from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_util.model_util import ModelUtil


class TestModelUtil:
    def test_evaluate_model(self):
        dataloader, model, criterion, metric = self._setup_test_params(
            create_loss=True, create_metric=True
        )
        (loss, accuracy) = ModelUtil.evaluate_model(
            dataloader, model, criterion, metric
        )
        assert loss >= 0
        assert accuracy >= 0
        assert accuracy <= 1

    def test_evaluate_model_no_loss(self):
        dataloader, model, criterion, metric = self._setup_test_params(
            create_loss=False, create_metric=True
        )
        (loss, accuracy) = ModelUtil.evaluate_model(
            dataloader, model, criterion, metric
        )
        assert loss is None
        assert accuracy >= 0
        assert accuracy <= 1

    def test_evaluate_model_no_metric(self):
        dataloader, model, criterion, metric = self._setup_test_params(
            create_loss=True, create_metric=False
        )
        (loss, accuracy) = ModelUtil.evaluate_model(
            dataloader, model, criterion, metric
        )
        assert loss > 0
        assert accuracy is None

    def _setup_test_params(self, create_loss, create_metric):
        batch_size = 32
        num_classes = 10
        image_shape = (1, 28, 28)
        model = nn.Sequential(
            nn.Linear(image_shape[1] * image_shape[2], num_classes),
        )
        images = torch.rand(batch_size, *image_shape)
        labels = torch.randint(0, num_classes, (batch_size,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss() if create_loss else None
        metric = (
            torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            if create_metric
            else None
        )
        return (dataloader, model, criterion, metric)
