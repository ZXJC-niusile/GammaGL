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

from defog_utils import to_dense


def test_defog_dense_uses_device_ops_and_preserves_encoding():
    x = tlx.convert_to_tensor(
        [[1, 0], [0, 1], [1, 0]], dtype=tlx.float32
    )
    edge_index = tlx.convert_to_tensor([[0, 1], [1, 0]], dtype=tlx.int64)
    edge_attr = tlx.convert_to_tensor([[0, 1], [0, 1]], dtype=tlx.float32)
    batch = tlx.convert_to_tensor([0, 0, 1], dtype=tlx.int64)

    with mock.patch.object(
        tlx, 'convert_to_numpy', side_effect=AssertionError('device sync')
    ):
        dense, node_mask = to_dense(x, edge_index, edge_attr, batch)

    dense_x = dense.X.detach().cpu().numpy()
    dense_e = dense.E.detach().cpu().numpy()
    mask = node_mask.detach().cpu().numpy()

    assert dense_x.shape == (2, 2, 2)
    assert np.array_equal(mask, [[True, True], [True, False]])
    assert np.allclose(dense_e[0, 0, 1], [0, 1])
    assert np.allclose(dense_e[0, 1, 0], [0, 1])
    assert np.allclose(dense_e[:, 0, 0], 0)
    assert np.allclose(dense_e[1, 0, 1], 0)
    assert np.allclose(dense_e[1, 1], 0)


if __name__ == '__main__':
    test_defog_dense_uses_device_ops_and_preserves_encoding()
    print('DeFoG dense tests passed')
