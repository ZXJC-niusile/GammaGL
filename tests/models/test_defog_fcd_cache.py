import os
import sys
import tempfile
import types

os.environ['TL_BACKEND'] = 'torch'

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from evaluator import smiles_cache_paths
import defog_fcd
from rdkit_functions import compute_fcd


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


def test_fcd_evaluation_uses_cpu():
    calls = {}
    fake_fcd = types.ModuleType('fcd')

    def get_fcd(smiles1, smiles2, device=None):
        calls['device'] = device
        return 0.25

    fake_fcd.get_fcd = get_fcd
    previous = sys.modules.get('fcd')
    sys.modules['fcd'] = fake_fcd
    try:
        assert defog_fcd._compute_fcd_cpu(['C'], ['C']) == 0.25
    finally:
        if previous is None:
            del sys.modules['fcd']
        else:
            sys.modules['fcd'] = previous
    assert calls['device'] == 'cpu'


def test_rdkit_compute_fcd_uses_isolated_helper():
    calls = {}
    original = defog_fcd.compute_fcd_cpu_isolated

    def fake_compute(generated_smiles, reference_smiles):
        calls['generated'] = generated_smiles
        calls['reference'] = reference_smiles
        return 0.5

    defog_fcd.compute_fcd_cpu_isolated = fake_compute
    try:
        assert compute_fcd(['C', None], ['C']) == 0.5
    finally:
        defog_fcd.compute_fcd_cpu_isolated = original

    assert calls == {'generated': ['C'], 'reference': ['C']}


if __name__ == '__main__':
    test_reference_caches_are_isolated_by_evaluation_split()
    test_qm9_caches_are_isolated_by_hydrogen_configuration()
    test_reference_cache_tag_cannot_escape_cache_directory()
    test_fcd_evaluation_uses_cpu()
    test_rdkit_compute_fcd_uses_isolated_helper()
    print('DeFoG FCD cache tests passed')
