import os
import sys
import tempfile

os.environ['TL_BACKEND'] = 'torch'

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from evaluator import smiles_cache_paths


def test_reference_caches_are_isolated_by_evaluation_split():
    with tempfile.TemporaryDirectory() as cache_dir:
        train_val, ref_val = smiles_cache_paths(
            cache_dir, 'zinc250k', 'val_200', {}
        )
        train_test, ref_test = smiles_cache_paths(
            cache_dir, 'zinc250k', 'test_full', {}
        )
    assert train_val == train_test
    assert ref_val != ref_test
    assert ref_val.endswith('ref_smiles_cache_zinc250k_val_200.pkl')
    assert ref_test.endswith('ref_smiles_cache_zinc250k_test_full.pkl')


def test_qm9_caches_are_isolated_by_hydrogen_configuration():
    with tempfile.TemporaryDirectory() as cache_dir:
        no_h = smiles_cache_paths(
            cache_dir, 'qm9', 'test_full', {'remove_h': True}
        )
        with_h = smiles_cache_paths(
            cache_dir, 'qm9', 'test_full', {'remove_h': False}
        )
    assert no_h != with_h
    assert all('qm9_no_h' in path for path in no_h)
    assert all('qm9_with_h' in path for path in with_h)


def test_reference_cache_tag_cannot_escape_cache_directory():
    with tempfile.TemporaryDirectory() as cache_dir:
        _, reference = smiles_cache_paths(
            cache_dir, 'qm9', '../test full', {'remove_h': True}
        )
        assert os.path.dirname(reference) == cache_dir
        assert '..' not in os.path.basename(reference)


if __name__ == '__main__':
    test_reference_caches_are_isolated_by_evaluation_split()
    test_qm9_caches_are_isolated_by_hydrogen_configuration()
    test_reference_cache_tag_cannot_escape_cache_directory()
    print('DeFoG FCD cache tests passed')
