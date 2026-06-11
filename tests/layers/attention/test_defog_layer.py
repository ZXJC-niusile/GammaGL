import os
os.environ['TL_BACKEND'] = 'torch'
import numpy as np
from unittest import mock
import tensorlayerx as tlx
from gammagl.layers.attention.defog_layer import XEyTransformerLayer, masked_softmax


def test_masked_softmax_normalizes_valid_entries_without_sync():
    logits = tlx.convert_to_tensor([[1.0, 2.0, 3.0]])
    mask = tlx.convert_to_tensor([[True, False, True]])

    with mock.patch.object(
        tlx, 'convert_to_numpy', side_effect=AssertionError('device sync')
    ):
        output = masked_softmax(logits, mask)

    output = output.detach().cpu().numpy()
    assert np.allclose(output[0, 1], 0.0)
    assert np.allclose(output.sum(axis=-1), 1.0)


def test_masked_softmax_all_masked_returns_zero():
    logits = tlx.convert_to_tensor([[1.0, 2.0, 3.0]])
    mask = tlx.convert_to_tensor([[False, False, False]])
    output = masked_softmax(logits, mask).detach().cpu().numpy()

    assert np.allclose(output, 0.0)


def test_masked_softmax_broadcasts_over_attention_features():
    logits = tlx.ones((2, 3, 3, 4, 2))
    mask = tlx.convert_to_tensor(
        [
            [[[True] * 4, [True] * 4, [False] * 4]] * 3,
            [[[False] * 4, [False] * 4, [False] * 4]] * 3,
        ]
    )
    output = masked_softmax(logits, mask, dim=2)
    output = output.detach().cpu().numpy()

    assert np.allclose(output[0].sum(axis=1), 1.0)
    assert np.allclose(output[0, :, 2], 0.0)
    assert np.allclose(output[1], 0.0)


def test_defog_layer_forward_has_no_host_sync_and_masks_padding():
    layer = XEyTransformerLayer(16, 8, 4, 4)
    X = tlx.ones((1, 3, 16))
    E = tlx.ones((1, 3, 3, 8))
    y = tlx.ones((1, 4))
    node_mask = tlx.convert_to_tensor([[True, True, False]])

    with mock.patch.object(
        tlx, 'convert_to_numpy', side_effect=AssertionError('device sync')
    ):
        out_X, out_E, _ = layer(X, E, y, node_mask)

    out_X = out_X.detach().cpu().numpy()
    out_E = out_E.detach().cpu().numpy()
    assert np.allclose(out_X[:, 2], 0.0)
    assert np.allclose(out_E[:, 2], 0.0)
    assert np.allclose(out_E[:, :, 2], 0.0)

def test_defog_layer():
    dx, de, dy = 16, 8, 4
    n_head = 4
    layer = XEyTransformerLayer(dx, de, dy, n_head)
    
    bs, n = 2, 5
    X = tlx.ones((bs, n, dx))
    E = tlx.ones((bs, n, n, de))
    y = tlx.ones((bs, dy))
    node_mask = tlx.ones((bs, n))
    
    out_X, out_E, out_y = layer(X, E, y, node_mask)
    
    assert out_X.shape == (bs, n, dx)
    assert out_E.shape == (bs, n, n, de)
    assert out_y.shape == (bs, dy)

def test_defog_layer_edge_cases():
    dx, de, dy = 16, 8, 4
    n_head = 4
    layer = XEyTransformerLayer(dx, de, dy, n_head)
    
    # 1. Batch size 1
    bs, n = 1, 3
    X = tlx.ones((bs, n, dx))
    E = tlx.ones((bs, n, n, de))
    y = tlx.ones((bs, dy))
    node_mask = tlx.ones((bs, n))
    out_X, out_E, out_y = layer(X, E, y, node_mask)
    assert out_X.shape == (bs, n, dx)
    
    # 2. Empty graph (n=1)
    bs, n = 2, 1
    X = tlx.ones((bs, n, dx))
    E = tlx.ones((bs, n, n, de))
    y = tlx.ones((bs, dy))
    node_mask = tlx.ones((bs, n))
    out_X, out_E, out_y = layer(X, E, y, node_mask)
    assert out_X.shape == (bs, n, dx)

if __name__ == '__main__':
    test_masked_softmax_normalizes_valid_entries_without_sync()
    test_masked_softmax_all_masked_returns_zero()
    test_masked_softmax_broadcasts_over_attention_features()
    test_defog_layer_forward_has_no_host_sync_and_masks_padding()
    test_defog_layer()
    test_defog_layer_edge_cases()
