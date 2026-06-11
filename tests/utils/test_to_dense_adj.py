import numpy as np
import tensorlayerx as tlx

from gammagl.utils.to_dense_adj import to_dense_adj


def test_to_dense_adj_keeps_empty_graph_slots():
    edge_index = tlx.convert_to_tensor([[0, 1], [1, 0]], dtype=tlx.int64)
    batch = tlx.convert_to_tensor([0, 0, 1], dtype=tlx.int64)
    edge_attr = tlx.convert_to_tensor(
        [[0.25, 0.75], [1.5, 2.5]], dtype=tlx.float32
    )

    adj = to_dense_adj(
        edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=2
    )
    adj_np = tlx.convert_to_numpy(adj)

    assert adj.shape == (2, 2, 2, 2)
    assert adj_np.dtype == np.float32
    assert np.allclose(adj_np[0, 0, 1], [0.25, 0.75])
    assert np.allclose(adj_np[0, 1, 0], [1.5, 2.5])
    assert np.allclose(adj_np[1], 0.0)


def test_to_dense_adj_without_edge_features():
    edge_index = tlx.convert_to_tensor([[0, 1, 3], [1, 2, 4]], dtype=tlx.int64)
    batch = tlx.convert_to_tensor([0, 0, 0, 1, 1], dtype=tlx.int64)

    adj = tlx.convert_to_numpy(to_dense_adj(edge_index, batch=batch))

    assert adj.shape == (2, 3, 3)
    assert adj[0, 0, 1] == 1
    assert adj[0, 1, 2] == 1
    assert adj[1, 0, 1] == 1


if __name__ == '__main__':
    test_to_dense_adj_keeps_empty_graph_slots()
    test_to_dense_adj_without_edge_features()
    print('to_dense_adj tests passed')
