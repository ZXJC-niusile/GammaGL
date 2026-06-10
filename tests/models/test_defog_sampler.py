import os
import sys
from unittest import mock

import numpy as np

os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from sampler import compute_step_probs


def _reference_step_probs(rates, states, dt):
    step = rates * dt
    current = states.argmax(axis=-1)
    rows = np.indices(current.shape)
    step[(*rows, current)] = 0.0
    stay = np.clip(1.0 - step.sum(axis=-1), a_min=0.0, a_max=None)
    step[(*rows, current)] = stay
    return step


def test_compute_step_probs_matches_numpy_reference_without_sync():
    rates_x_np = np.array(
        [[[0.4, 0.2, 0.1], [0.3, 0.5, 0.2]]], dtype=np.float32
    )
    rates_e_np = np.array(
        [[[[0.4, 0.2], [0.1, 0.3]], [[0.2, 0.5], [0.4, 0.1]]]],
        dtype=np.float32,
    )
    x_t_np = np.array([[[0, 1, 0], [0, 0, 1]]], dtype=np.float32)
    e_t_np = np.array(
        [[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]], dtype=np.float32
    )
    expected_x = _reference_step_probs(rates_x_np.copy(), x_t_np, 0.1)
    expected_e = _reference_step_probs(rates_e_np.copy(), e_t_np, 0.1)

    rates_x = tlx.convert_to_tensor(rates_x_np)
    rates_e = tlx.convert_to_tensor(rates_e_np)
    x_t = tlx.convert_to_tensor(x_t_np)
    e_t = tlx.convert_to_tensor(e_t_np)

    with mock.patch.object(
        tlx, 'convert_to_numpy', side_effect=AssertionError('device sync')
    ):
        prob_x, prob_e = compute_step_probs(
            rates_x, rates_e, x_t, e_t, 0.1
        )

    assert np.allclose(prob_x.detach().cpu().numpy(), expected_x)
    assert np.allclose(prob_e.detach().cpu().numpy(), expected_e)
    assert np.allclose(prob_x.detach().cpu().sum(dim=-1).numpy(), 1.0)
    assert np.allclose(prob_e.detach().cpu().sum(dim=-1).numpy(), 1.0)


if __name__ == '__main__':
    test_compute_step_probs_matches_numpy_reference_without_sync()
    print('sampler numerical test passed')
