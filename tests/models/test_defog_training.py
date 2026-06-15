import os
import sys

os.environ['TL_BACKEND'] = 'torch'

import torch

EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog'))
sys.path.insert(0, EXAMPLE_DIR)

from defog_training import SeededRandomSampler, batch_global_features
from train_metrics import LossMetric, TrainLossDiscrete


class Batch:
    pass


def test_seeded_sampler_is_reproducible_per_epoch():
    first = SeededRandomSampler(range(8), seed=7)
    second = SeededRandomSampler(range(8), seed=7)
    assert list(first) == list(second)
    assert list(first) == list(second)


def test_batch_global_features_preserves_device_and_shape():
    batch = Batch()
    batch.y = torch.arange(6, dtype=torch.float64).reshape(2, 1, 3)
    y = batch_global_features(batch, 2, True)
    assert y.shape == (2, 3)
    assert y.dtype == torch.float32
    assert y.device == batch.y.device
    assert batch_global_features(batch, 2, False).shape == (2, 0)


def test_masked_cross_entropy_ignores_invalid_logits():
    loss_fn = TrainLossDiscrete()
    targets = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    mask = targets.abs().sum(dim=-1) > 0
    first = torch.tensor([
        [3.0, -1.0],
        [-2.0, 2.0],
        [100.0, -100.0],
    ])
    second = first.clone()
    second[2] = torch.tensor([-100.0, 100.0])

    first_loss = loss_fn._masked_loss(
        LossMetric(), first, targets, mask
    )
    second_loss = loss_fn._masked_loss(
        LossMetric(), second, targets, mask
    )

    expected = torch.nn.functional.cross_entropy(
        first[:2], torch.tensor([0, 1])
    )
    assert torch.allclose(first_loss, expected)
    assert torch.allclose(second_loss, expected)


if __name__ == '__main__':
    test_seeded_sampler_is_reproducible_per_epoch()
    test_batch_global_features_preserves_device_and_shape()
    test_masked_cross_entropy_ignores_invalid_logits()
    print('DeFoG training utility tests passed')
