"""Additional checks for held-out evaluation and frozen model state."""

import copy

import pytest
import torch
import torch.nn as nn

from train import train


def test_test_labels_cannot_change_training(blobs):
    X, y = blobs
    first = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 3))
    second = copy.deepcopy(first)

    torch.manual_seed(7)
    train(first, X, y, X, y, epochs=2, lr=0.01, batch_size=17)
    torch.manual_seed(7)
    train(second, X, y, X, (y + 1) % 3,
          epochs=2, lr=0.01, batch_size=17)

    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), (
            "Changing only the test labels must not change the trained model."
        )


def test_frozen_batchnorm_keeps_running_statistics(blobs):
    X, y = blobs
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 3))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    before = copy.deepcopy(model[0].state_dict())

    train(model, X, y, X, y, epochs=2, lr=0.01, batch_size=17)

    for name, value in model[0].state_dict().items():
        assert torch.equal(value.cpu(), before[name]), (
            "Training the last layer must preserve the frozen backbone's state."
        )


def test_test_loss_counts_every_image_equally(blobs):
    X, y = blobs
    model = nn.Linear(3, 3)
    # The small last batch has a different class from the first full batch.
    X_test, y_test = X[[0, 1, 2, 3, 4, 80, 81]], y[[0, 1, 2, 3, 4, 80, 81]]
    history = train(model, X, y, X_test, y_test,
                    epochs=1, lr=0.01, batch_size=5)
    device = next(model.parameters()).device
    with torch.no_grad():
        expected = nn.CrossEntropyLoss()(
            model(torch.from_numpy(X_test).to(device)),
            torch.from_numpy(y_test).to(device),
        ).item()
    assert history["test_loss"][0] == pytest.approx(expected, abs=1e-6)
