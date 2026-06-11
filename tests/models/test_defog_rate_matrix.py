import os
import sys

import torch
import torch.nn.functional as F


EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from defog_utils import PlaceHolder
from flow_matching import RateMatrixDesigner, dt_p_xt_g_x1, p_xt_g_x1
from sampler import compute_step_probs


def _inputs():
    limit = PlaceHolder(
        X=torch.tensor([0.5, 0.3, 0.2]),
        E=torch.tensor([0.7, 0.3]),
        y=torch.empty(0),
    )
    x_labels = torch.tensor([[0, 1, 2]])
    e_labels = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    x_t = F.one_hot(x_labels, 3).float()
    e_t = F.one_hot(e_labels, 2).float()
    pred_x = torch.tensor([[[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.1, 0.3, 0.6]]])
    pred_e = torch.tensor([[[[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
                            [[0.3, 0.7], [0.9, 0.1], [0.2, 0.8]],
                            [[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]]]])
    return limit, x_t, e_t, pred_x, pred_e, torch.ones(1, 3, dtype=torch.bool)


def test_flow_probabilities_and_derivatives_are_normalized():
    limit, x_t, e_t, _, _, _ = _inputs()
    x_labels = x_t.argmax(-1)
    e_labels = e_t.argmax(-1)
    prob_x, prob_e = p_xt_g_x1(x_labels, e_labels, torch.tensor([[0.4]]), limit)
    dx, de = dt_p_xt_g_x1(x_labels, e_labels, limit)
    assert torch.allclose(prob_x.sum(-1), torch.ones_like(prob_x[..., 0]))
    assert torch.allclose(prob_e.sum(-1), torch.ones_like(prob_e[..., 0]))
    assert torch.allclose(dx.sum(-1), torch.zeros_like(dx[..., 0]), atol=1e-6)
    assert torch.allclose(de.sum(-1), torch.zeros_like(de[..., 0]), atol=1e-6)


def test_rate_designs_produce_valid_symmetric_step_probabilities():
    limit, x_t, e_t, pred_x, pred_e, node_mask = _inputs()
    designs = [('general', 'max_marginal'), ('marginal', 'max_marginal')]
    designs += [('column', criterion) for criterion in (
        'max_marginal', 'x_t', 'abs_state', 'p_x1_g_xt',
        'x_1', 'p_xt_g_x1', 'xhat_t',
    )]
    designs += [('entry', criterion) for criterion in ('abs_state', 'first')]

    for rdb, criterion in designs:
        torch.manual_seed(7)
        designer = RateMatrixDesigner(
            rdb=rdb, rdb_crit=criterion, eta=0.5, omega=0.1,
            limit_dist=limit,
        )
        rate_x, rate_e = designer.compute_graph_rate_matrix(
            torch.tensor([[0.4]]), node_mask, (x_t, e_t), (pred_x, pred_e)
        )
        assert rate_x.shape == pred_x.shape
        assert rate_e.shape == pred_e.shape
        assert torch.isfinite(rate_x).all() and torch.isfinite(rate_e).all()
        assert (rate_x >= 0).all() and (rate_e >= 0).all()
        assert torch.allclose(rate_e, rate_e.transpose(1, 2))

        max_rate = torch.maximum(rate_x.max(), rate_e.max()).item()
        dt = min(0.01, 0.5 / max(max_rate, 1.0))
        prob_x, prob_e = compute_step_probs(rate_x, rate_e, x_t, e_t, dt)
        assert (prob_x >= 0).all() and (prob_e >= 0).all()
        assert torch.allclose(prob_x.sum(-1), torch.ones_like(prob_x[..., 0]))
        assert torch.allclose(prob_e.sum(-1), torch.ones_like(prob_e[..., 0]))
        assert torch.allclose(prob_e, prob_e.transpose(1, 2))


def test_non_default_designs_preserve_cuda_device():
    if not torch.cuda.is_available():
        return
    limit, x_t, e_t, pred_x, pred_e, node_mask = _inputs()
    limit = PlaceHolder(limit.X.cuda(), limit.E.cuda(), limit.y.cuda())
    tensors = [value.cuda() for value in (x_t, e_t, pred_x, pred_e, node_mask)]
    x_t, e_t, pred_x, pred_e, node_mask = tensors
    for rdb, criterion in (('column', 'max_marginal'), ('entry', 'first')):
        torch.manual_seed(7)
        designer = RateMatrixDesigner(
            rdb=rdb, rdb_crit=criterion, eta=0.5, omega=0.1,
            limit_dist=limit,
        )
        rate_x, rate_e = designer.compute_graph_rate_matrix(
            torch.tensor([[0.4]], device='cuda'),
            node_mask,
            (x_t, e_t),
            (pred_x, pred_e),
        )
        assert rate_x.is_cuda and rate_e.is_cuda


if __name__ == '__main__':
    test_flow_probabilities_and_derivatives_are_normalized()
    test_rate_designs_produce_valid_symmetric_step_probabilities()
    test_non_default_designs_preserve_cuda_device()
