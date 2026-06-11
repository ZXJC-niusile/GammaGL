import os
import sys

os.environ['TL_BACKEND'] = 'torch'

import torch

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from defog_utils import encode_no_edge, to_dense


def _assert_no_edge_encoding(device):
    edges = torch.zeros((1, 3, 3, 3), device=device)
    edges[0, 0, 1, 1] = 1
    edges[0, 1, 0, 1] = 1
    encoded = encode_no_edge(edges)

    assert torch.equal(encoded[0, 0, 1], torch.tensor([0., 1., 0.], device=device))
    assert torch.equal(encoded[0, 0, 2], torch.tensor([1., 0., 0.], device=device))
    assert torch.equal(encoded[0, 2, 0], torch.tensor([1., 0., 0.], device=device))
    diagonal = encoded[0, torch.arange(3), torch.arange(3)]
    assert torch.equal(diagonal, torch.zeros_like(diagonal))
    assert torch.equal(encoded, encoded.transpose(1, 2))


def test_encode_no_edge_cpu():
    _assert_no_edge_encoding('cpu')


def test_encode_no_edge_cuda():
    if torch.cuda.is_available():
        _assert_no_edge_encoding('cuda')


def test_to_dense_keeps_padding_zero_instead_of_no_edge():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.tensor([[0., 1.], [0., 1.]])
    batch = torch.tensor([0, 0, 1])

    dense, node_mask = to_dense(x, edge_index, edge_attr, batch)
    assert node_mask.tolist() == [[True, True], [True, False]]
    assert torch.equal(dense.E[0, 0, 1], torch.tensor([0., 1.]))
    assert torch.equal(dense.E[1, 0, 1], torch.zeros(2))
    assert torch.equal(dense.E[1, 1], torch.zeros((2, 2)))


if __name__ == '__main__':
    test_encode_no_edge_cpu()
    test_encode_no_edge_cuda()
    test_to_dense_keeps_padding_zero_instead_of_no_edge()
    print('DeFoG no-edge tests passed')
