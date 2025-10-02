import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset


def get_available_gpu() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


@staticmethod
def create_simple_trainer_params(
    num_train_batches,
    num_validation_batches,
    model,
    create_loss,
    create_metric,
    use_gpu,
):
    device = get_available_gpu() if use_gpu else torch.device("cpu")
    batch_size = 4
    num_classes = 10
    image_shape = (1, 28, 28)
    model = model.to(device=device)

    train_dataloader = create_categorical_dataloader(
        image_shape, batch_size, num_train_batches, num_classes
    )
    validation_dataloader = create_categorical_dataloader(
        image_shape, batch_size, num_validation_batches, num_classes
    )

    criterion = nn.CrossEntropyLoss() if create_loss else None
    train_metric = (
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        if create_metric
        else None
    ).to(device=device)
    validation_metric = (
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        if create_metric
        else None
    ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    return (
        device,
        train_dataloader,
        validation_dataloader,
        model,
        criterion,
        train_metric,
        validation_metric,
        optimizer,
    )


def create_categorical_dataloader(
    data_shape, batch_size, num_batches, num_classes, random_seed=None
):
    data = torch.rand(batch_size * num_batches, *data_shape)
    print(data.shape)
    labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
