import os

os.environ['TL_BACKEND'] = 'torch'

import torch

from gammagl.layers.attention.defog_layer import NodeEdgeBlock, masked_softmax
from gammagl.models.defog import DeFoGModel


def _model():
    return DeFoGModel(
        n_layers=2,
        input_dims={'X': 5, 'E': 4, 'y': 2},
        hidden_mlp_dims={'X': 16, 'E': 8, 'y': 16},
        hidden_dims={
            'dx': 16, 'de': 8, 'dy': 16, 'n_head': 2,
            'dim_ffX': 32, 'dim_ffE': 16, 'dim_ffy': 32,
        },
        output_dims={'X': 5, 'E': 4, 'y': 0},
    )


def _inputs(device):
    torch.manual_seed(5)
    x = torch.randn(2, 5, 5, device=device)
    e = torch.randn(2, 5, 5, 4, device=device)
    e = (e + e.transpose(1, 2)) / 2
    y = torch.tensor([[0.2, 0.4], [0.7, 0.6]], device=device)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]],
        device=device,
    )
    return x, e, y, mask


def _assert_model_invariants(device):
    model = _model().to(device)
    model.set_eval()
    x, e, y, mask = _inputs(device)
    with torch.no_grad():
        out_x, out_e, out_y = model(x, e, y, mask)

    x_mask = mask.unsqueeze(-1)
    e_mask = x_mask.unsqueeze(2) & x_mask.unsqueeze(1)
    diagonal = torch.eye(e.shape[1], dtype=torch.bool, device=device)[None, :, :, None]
    masked_x = out_x.masked_select(~x_mask)
    assert torch.equal(masked_x, torch.zeros_like(masked_x))
    masked_e = out_e.masked_select(~e_mask)
    assert torch.equal(masked_e, torch.zeros_like(masked_e))
    diagonal_e = out_e.masked_select(diagonal)
    assert torch.equal(diagonal_e, torch.zeros_like(diagonal_e))
    assert torch.allclose(out_e, out_e.transpose(1, 2), atol=1e-6)
    assert torch.isfinite(out_x).all() and torch.isfinite(out_e).all()
    assert out_y.shape == (2, 0)

    noisy_x, noisy_e = x.clone(), e.clone()
    noisy_x[~mask] = 1e4 * torch.randn_like(noisy_x[~mask])
    invalid_edges = ~(mask[:, :, None] & mask[:, None, :])
    noisy_e[invalid_edges] = 1e4 * torch.randn_like(noisy_e[invalid_edges])
    with torch.no_grad():
        noisy_out_x, noisy_out_e, _ = model(noisy_x, noisy_e, y, mask)
    valid_x = x_mask.expand_as(out_x)
    assert torch.allclose(noisy_out_x[valid_x], out_x[valid_x])
    valid_e = e_mask.expand_as(out_e)
    assert torch.allclose(noisy_out_e[valid_e], out_e[valid_e])


def test_masked_softmax_handles_fully_masked_rows():
    logits = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    output = masked_softmax(logits, mask, dim=-1)
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isfinite(output).all()


def test_attention_block_masks_nodes_and_edges():
    torch.manual_seed(3)
    block = NodeEdgeBlock(dx=8, de=4, dy=8, n_head=2)
    x = torch.randn(1, 4, 8)
    e = torch.randn(1, 4, 4, 4)
    y = torch.randn(1, 8)
    mask = torch.tensor([[True, True, False, False]])
    out_x, out_e, _ = block(x, e, y, mask)
    x_mask = mask.unsqueeze(-1)
    e_mask = x_mask.unsqueeze(2) & x_mask.unsqueeze(1)
    masked_x = out_x.masked_select(~x_mask)
    assert torch.equal(masked_x, torch.zeros_like(masked_x))
    masked_e = out_e.masked_select(~e_mask)
    assert torch.equal(masked_e, torch.zeros_like(masked_e))


def test_model_mask_and_symmetry_cpu():
    _assert_model_invariants('cpu')


def test_model_mask_and_symmetry_cuda():
    if torch.cuda.is_available():
        _assert_model_invariants('cuda')


if __name__ == '__main__':
    test_masked_softmax_handles_fully_masked_rows()
    test_attention_block_masks_nodes_and_edges()
    test_model_mask_and_symmetry_cpu()
    test_model_mask_and_symmetry_cuda()
