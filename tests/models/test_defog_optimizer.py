import os
import sys

import torch


EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from defog_optimizer import TorchAdamW


def test_matches_torch_adamw_update():
    actual = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    expected = torch.nn.Parameter(actual.detach().clone())
    adapter = TorchAdamW(lr=0.01, weight_decay=0.1, amsgrad=True)
    reference = torch.optim.AdamW(
        [expected], lr=0.01, weight_decay=0.1, amsgrad=True
    )

    actual_loss = actual.square().sum()
    adapter.gradient(actual_loss, [actual])
    adapter.apply_gradients()

    reference.zero_grad()
    expected.square().sum().backward()
    reference.step()
    assert torch.allclose(actual, expected)


def test_gradient_clipping_and_non_finite_loss():
    weight = torch.nn.Parameter(torch.tensor([10.0]))
    adapter = TorchAdamW(lr=0.01, grad_clip_norm=0.5)
    adapter.gradient(weight.square().sum(), [weight])
    assert weight.grad.norm() <= 0.500001
    adapter.apply_gradients()

    before = weight.detach().clone()
    grads = adapter.gradient(weight.sum() * torch.tensor(float('nan')), [weight])
    adapter.apply_gradients()
    assert torch.equal(weight, before)
    assert torch.equal(grads[0], torch.zeros_like(weight))


def test_non_finite_gradients_are_sanitized():
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    adapter = TorchAdamW(lr=0.01)
    weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
    adapter.gradient(weight.square().sum(), [weight])
    assert torch.equal(weight.grad, torch.zeros_like(weight))


if __name__ == '__main__':
    test_matches_torch_adamw_update()
    test_gradient_clipping_and_non_finite_loss()
    test_non_finite_gradients_are_sanitized()
