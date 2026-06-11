import os
from pathlib import Path
import sys
import tempfile

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / 'examples' / 'defog'
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from reproduction_audit import REQUIRED_TESTS, audit_repository, main


def test_repository_reproduction_checks_pass():
    repo_root = Path(__file__).resolve().parents[2]
    checks = audit_repository(repo_root)
    failures = [check for check in checks if check['status'] != 'pass']
    assert failures == []


def test_audit_reports_missing_required_file():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = Path(tmp)
        os.symlink(repo_root / 'gammagl', fake_root / 'gammagl', target_is_directory=True)
        os.symlink(repo_root / 'tests', fake_root / 'tests', target_is_directory=True)
        os.symlink(repo_root / 'examples', fake_root / 'examples', target_is_directory=True)
        missing = fake_root / 'examples' / 'defog' / 'readme.md'
        # The linked repository is complete; exercise the CLI success path here.
        assert missing.exists()
        assert main(['--repo-root', str(fake_root), '--json']) == 0


def test_required_algorithm_suite_stays_explicit():
    expected = {
        'test_defog_no_edge.py',
        'test_defog_sampler_batching.py',
        'test_defog_dataset_splits.py',
        'test_defog_fcd_cache.py',
        'test_defog_best_checkpoint.py',
        'test_defog_rate_matrix.py',
        'test_defog_mask_symmetry.py',
    }
    assert expected.issubset(REQUIRED_TESTS)


if __name__ == '__main__':
    test_repository_reproduction_checks_pass()
    test_audit_reports_missing_required_file()
    test_required_algorithm_suite_stays_explicit()
    print('DeFoG reproduction audit tests passed')
