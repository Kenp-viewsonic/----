"""Tests for the stage-wise new-class separation loss."""

import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.separation_loss import NewClassSeparationLoss


def test_separation_loss_positive_when_new_class_overlaps_old_cluster():
    loss_fn = NewClassSeparationLoss(margin=0.2)

    cls_hidden = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.95, 0.05],
        [0.85, 0.15],
    ], dtype=torch.float32)
    labels = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    loss = loss_fn(cls_hidden, labels, old_num_classes=1)
    assert loss.item() > 0.0


def test_separation_loss_zero_when_new_class_is_well_separated():
    loss_fn = NewClassSeparationLoss(margin=0.1)

    cls_hidden = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [-1.0, 0.0],
        [-0.9, -0.1],
    ], dtype=torch.float32)
    labels = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    loss = loss_fn(cls_hidden, labels, old_num_classes=1)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


if __name__ == "__main__":
    test_separation_loss_positive_when_new_class_overlaps_old_cluster()
    test_separation_loss_zero_when_new_class_is_well_separated()
    print("All separation loss tests passed.")