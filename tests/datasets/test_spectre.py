import os

os.environ['TL_BACKEND'] = 'torch'

import numpy as np

from gammagl.datasets import (
    Comm20GraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
    SpectreGraphDataset,
    TreeGraphDataset,
)
from gammagl.datasets.spectre import _adj_to_edge_index_and_attr


def test_spectre_dataset_exports():
    assert issubclass(PlanarGraphDataset, SpectreGraphDataset)
    assert issubclass(TreeGraphDataset, SpectreGraphDataset)
    assert issubclass(SBMGraphDataset, SpectreGraphDataset)
    assert Comm20GraphDataset.__name__ == 'Comm20GraphDataset'


def test_adjacency_conversion_removes_diagonal_and_encodes_edges():
    adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    edge_index, edge_attr = _adj_to_edge_index_and_attr(adjacency)
    assert edge_index.shape == (2, 4)
    assert np.all(edge_index[0] != edge_index[1])
    assert np.allclose(edge_attr[:, 0], 0)
    assert np.allclose(edge_attr[:, 1], 1)


if __name__ == '__main__':
    test_spectre_dataset_exports()
    test_adjacency_conversion_removes_diagonal_and_encodes_edges()
    print('SPECTRE dataset tests passed')
