import os
import sys
from unittest import mock

import numpy as np
import torch

os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from extra_features import ExtraMolecularFeatures


def _numpy_reference(x, e, valencies, atom_weights, max_weight):
    dx = x.shape[-1]
    de = e.shape[-1]
    bond_orders = np.array(
        [0, 1, 2, 3, 1.5] if de == 5 else [0, 1, 2, 3],
        dtype=np.float32,
    ).reshape(1, 1, 1, -1)
    current = np.argmax(e * bond_orders, axis=-1).sum(axis=-1)

    valencies = np.pad(valencies, (0, max(0, dx - len(valencies))))
    normal = np.argmax(x * valencies.reshape(1, 1, -1), axis=-1)
    charge = (normal - current).astype(np.float32)

    atom_weights = np.pad(
        atom_weights, (0, max(0, dx - len(atom_weights)))
    )
    weight = atom_weights[np.argmax(x, axis=-1)].sum(axis=1, keepdims=True)
    return charge, current.astype(np.float32), weight / max_weight


def _run_case(dx, de, valencies, atom_weights, device):
    rng = np.random.default_rng(7 + dx + de)
    node_labels = rng.integers(0, dx, size=(2, 4))
    edge_labels = rng.integers(0, de, size=(2, 4, 4))
    x = np.eye(dx, dtype=np.float32)[node_labels]
    e = np.eye(de, dtype=np.float32)[edge_labels]

    expected = _numpy_reference(
        x, e, np.asarray(valencies), np.asarray(atom_weights), 500.0
    )
    noisy = {
        'X_t': torch.tensor(x, device=device),
        'E_t': torch.tensor(e, device=device),
    }
    features = ExtraMolecularFeatures({
        'valencies': valencies,
        'atom_weights': atom_weights,
        'max_weight': 500.0,
    })

    with mock.patch.object(
        tlx, 'convert_to_numpy', side_effect=AssertionError('device sync')
    ):
        result = features(noisy)

    assert result.X.device.type == device.type
    assert result.E.device.type == device.type
    assert result.y.device.type == device.type
    assert result.X.shape == (2, 4, 2)
    assert result.E.shape == (2, 4, 4, 0)
    assert np.allclose(result.X[..., 0].cpu().numpy(), expected[0])
    assert np.allclose(result.X[..., 1].cpu().numpy(), expected[1])
    assert np.allclose(result.y.cpu().numpy(), expected[2])


def test_qm9_molecular_features_match_numpy_reference():
    _run_case(
        dx=5,
        de=5,
        valencies=[4, 3, 2, 1, 1],
        atom_weights=[12.011, 14.007, 15.999, 18.998, 1.008],
        device=torch.device('cpu'),
    )


def test_zinc_molecular_features_match_numpy_reference():
    _run_case(
        dx=9,
        de=4,
        valencies=[4, 3, 2, 1, 3, 2, 1, 1, 1],
        atom_weights=[12.011, 14.007, 15.999, 18.998, 30.974, 32.06, 35.45, 79.904, 126.90],
        device=torch.device('cpu'),
    )


def test_molecular_features_preserve_cuda_device():
    if not torch.cuda.is_available():
        return
    _run_case(
        dx=9,
        de=4,
        valencies=[4, 3, 2, 1, 3, 2, 1, 1, 1],
        atom_weights=[12.011, 14.007, 15.999, 18.998, 30.974, 32.06, 35.45, 79.904, 126.90],
        device=torch.device('cuda'),
    )


if __name__ == '__main__':
    test_qm9_molecular_features_match_numpy_reference()
    test_zinc_molecular_features_match_numpy_reference()
    test_molecular_features_preserve_cuda_device()
    print('molecular feature tests passed')
