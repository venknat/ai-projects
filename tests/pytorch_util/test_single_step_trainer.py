import pytest

# Example function in your src/myml/utils.py (replace with real code later)

import torch
import torch.nn as nn
import torchmetrics

from unittest.mock import Mock

from pytorch_util._single_step_trainer import _SingleStepTrainer


class TestSingleStepTrainer:
    def test_simple_trainer(self):
        batch_size = 32
        num_classes = 10
        image_shape = (1, 28, 28)
        criterion = nn.CrossEntropyLoss()
        model = nn.Sequential(
            nn.Linear(image_shape[1] * image_shape[2], num_classes),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        callback1 = Mock()
        callback2 = Mock()

        images = torch.rand(batch_size, *image_shape)
        labels = torch.randint(0, num_classes, (batch_size,))

        _SingleStepTrainer.train(
            optimizer, images, labels, model, criterion, [callback1, callback2]
        )
        callback1.assert_called_once()
        callback2.assert_called_once()
