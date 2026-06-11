import os
import sys

os.environ['TL_BACKEND'] = 'torch'

import torch

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from sampler import sample_in_batches


def _indexed_sampler(batch_size, cond_labels=None, num_nodes=None, **kwargs):
    del kwargs
    labels = [-1] * batch_size if cond_labels is None else cond_labels.tolist()
    nodes = [-1] * batch_size if num_nodes is None else list(num_nodes)
    return list(zip(labels, nodes))


def test_chunked_sampling_matches_single_batch_order_and_slices():
    labels = torch.arange(7)
    nodes = [3, 4, 5, 6, 7, 8, 9]
    whole = sample_in_batches(
        _indexed_sampler, 7, 7, cond_labels=labels, num_nodes=nodes
    )
    chunked = sample_in_batches(
        _indexed_sampler, 7, 3, cond_labels=labels, num_nodes=nodes
    )
    assert chunked == whole
    assert chunked == list(zip(range(7), nodes))


def test_chunked_sampling_handles_remainder_and_empty_request():
    calls = []

    def sampler(batch_size, **kwargs):
        calls.append(batch_size)
        return [len(calls)] * batch_size

    result = sample_in_batches(sampler, total_size=8, batch_size=3)
    assert calls == [3, 3, 2]
    assert len(result) == 8
    assert sample_in_batches(sampler, total_size=0, batch_size=3) == []


def test_chunked_sampling_rejects_invalid_batch_size():
    try:
        sample_in_batches(_indexed_sampler, total_size=2, batch_size=0)
    except ValueError as exc:
        assert 'batch_size' in str(exc)
    else:
        raise AssertionError('invalid batch size was accepted')


if __name__ == '__main__':
    test_chunked_sampling_matches_single_batch_order_and_slices()
    test_chunked_sampling_handles_remainder_and_empty_request()
    test_chunked_sampling_rejects_invalid_batch_size()
    print('DeFoG sampler batching tests passed')
