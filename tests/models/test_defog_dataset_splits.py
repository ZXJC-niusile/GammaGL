import os
import sys

import numpy as np

os.environ['TL_BACKEND'] = 'torch'

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from defog_config import get_dataset_preset
from defog_datasets.qm9_dataset import build_qm9_split_ids
from defog_datasets.zinc250k_dataset import build_zinc_split_indices


def test_qm9_defog_split_is_deterministic_disjoint_and_complete():
    first = build_qm9_split_ids(133885)
    second = build_qm9_split_ids(133885)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert len(first['train']) == 100000
    assert len(first['test']) == 13388
    assert len(first['val']) == 20497
    parts = [set(values.tolist()) for values in first.values()]
    assert parts[0].isdisjoint(parts[1])
    assert parts[0].isdisjoint(parts[2])
    assert parts[1].isdisjoint(parts[2])
    assert set.union(*parts) == set(range(133885))


def test_qm9_preset_uses_original_split():
    assert get_dataset_preset('qm9')['use_defog_split'] is True


def test_zinc_split_matches_original_held_out_protocol():
    split = build_zinc_split_indices(10, [7, 2, 7])
    assert split['train'] == [0, 1, 3, 4, 5, 6, 8, 9]
    assert split['val'] == [2, 7]
    assert split['test'] == [2, 7]
    assert set(split['train']).isdisjoint(split['test'])


def test_zinc_split_rejects_out_of_range_indices():
    try:
        build_zinc_split_indices(3, [3])
    except ValueError as exc:
        assert 'outside' in str(exc)
    else:
        raise AssertionError('invalid ZINC split index was accepted')


if __name__ == '__main__':
    test_qm9_defog_split_is_deterministic_disjoint_and_complete()
    test_qm9_preset_uses_original_split()
    test_zinc_split_matches_original_held_out_protocol()
    test_zinc_split_rejects_out_of_range_indices()
    print('DeFoG dataset split tests passed')
