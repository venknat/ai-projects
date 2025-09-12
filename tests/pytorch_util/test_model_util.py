import pytest
import torchmetrics

from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_util.model_util import ModelUtil
import test_util


class TestModelUtil:

    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_evaluate_model(self, use_gpu):
        device, dataloader, model, criterion, metric = TestModelUtil._setup_test_params(
            create_loss=True, create_metric=True, use_gpu=use_gpu
        )
        (loss, accuracy) = ModelUtil.evaluate_model(
            device, dataloader, model, criterion, metric
        )
        assert loss >= 0
        assert accuracy >= 0
        assert accuracy <= 1

    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_evaluate_model_no_loss(self, use_gpu):
        device, dataloader, model, criterion, metric = self._setup_test_params(
            create_loss=False, create_metric=True, use_gpu=use_gpu
        )
        (loss, accuracy) = ModelUtil.evaluate_model(
            device, dataloader, model, criterion, metric
        )
        assert loss is None
        assert accuracy >= 0
        assert accuracy <= 1

    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_evaluate_model_no_metric(self, use_gpu):
        device, dataloader, model, criterion, metric = self._setup_test_params(
            create_loss=True, create_metric=False, use_gpu=use_gpu
        )
        (loss, accuracy) = ModelUtil.evaluate_model(
            device, dataloader, model, criterion, metric
        )
        assert loss > 0
        assert accuracy is None

    @staticmethod
    def _setup_test_params(create_loss, create_metric, use_gpu):
        device = test_util.get_available_gpu() if use_gpu else torch.device("cpu")
        batch_size = 32
        num_classes = 10
        image_shape = (1, 28, 28)
        model = nn.Sequential(
            nn.Linear(image_shape[1] * image_shape[2], num_classes),
        ).to(device=device)
        images = torch.rand(batch_size, *image_shape)
        labels = torch.randint(0, num_classes, (batch_size,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss() if create_loss else None
        metric = (
            torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
                device=device
            )
            if create_metric
            else None
        )

        return device, dataloader, model, criterion, metric
