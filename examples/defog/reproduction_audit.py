"""Reproduction readiness checks for the DeFoG GammaGL example."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REQUIRED_TESTS = (
    'test_defog.py',
    'test_defog_best_checkpoint.py',
    'test_defog_checkpoint.py',
    'test_defog_config.py',
    'test_defog_dataset_package.py',
    'test_defog_dataset_splits.py',
    'test_defog_dense.py',
    'test_defog_fcd_cache.py',
    'test_defog_feature_placeholder.py',
    'test_defog_mask_symmetry.py',
    'test_defog_molecular_features.py',
    'test_defog_no_edge.py',
    'test_defog_optimizer.py',
    'test_defog_rate_matrix.py',
    'test_defog_sampler.py',
    'test_defog_sampler_batching.py',
)

MANUAL_CHECKS = (
    'Run the paper-scale datasets with the documented seeds and checkpoints.',
    'Report mean and standard deviation over the paper evaluation folds.',
    'Compare metrics with the matching paper protocol and sampling steps.',
    'Resolve known metric gaps before claiming full reproduction.',
)


def _result(name, passed, detail):
    return {'name': name, 'status': 'pass' if passed else 'fail', 'detail': detail}


def audit_repository(repo_root):
    repo_root = Path(repo_root).resolve()
    example = repo_root / 'examples' / 'defog'
    tests = repo_root / 'tests' / 'models'
    checks = []

    required_paths = (
        repo_root / 'gammagl' / 'models' / 'defog.py',
        repo_root / 'gammagl' / 'layers' / 'attention' / 'defog_layer.py',
        example / 'defog_trainer.py',
        example / 'defog_config.py',
        example / 'defog_checkpoint.py',
        example / 'defog_optimizer.py',
        example / 'sampler.py',
        example / 'evaluator.py',
        example / 'readme.md',
    )
    missing_paths = [str(path.relative_to(repo_root)) for path in required_paths if not path.exists()]
    checks.append(_result('required files', not missing_paths, ', '.join(missing_paths) or 'complete'))

    model_exports = (repo_root / 'gammagl' / 'models' / '__init__.py').read_text()
    checks.append(_result(
        'model export', 'DeFoGModel' in model_exports,
        'DeFoGModel is exported from gammagl.models',
    ))

    trainer_text = (example / 'defog_trainer.py').read_text()
    checks.append(_result(
        'GammaGL training API',
        'WithLoss' in trainer_text and 'TrainOneStep' in trainer_text,
        'trainer uses WithLoss and TrainOneStep',
    ))

    dataset_dir = example / 'defog_datasets'
    checks.append(_result(
        'dataset package namespace',
        dataset_dir.is_dir() and not (example / 'datasets').is_dir(),
        'example package does not shadow Hugging Face datasets',
    ))

    missing_tests = [name for name in REQUIRED_TESTS if not (tests / name).exists()]
    checks.append(_result(
        'algorithm test inventory', not missing_tests,
        ', '.join(missing_tests) or f'{len(REQUIRED_TESTS)} required tests present',
    ))

    sys.path.insert(0, str(example))
    try:
        from defog_config import get_dataset_preset
        qm9 = get_dataset_preset('qm9')
        zinc = get_dataset_preset('zinc250k')
        sbm = get_dataset_preset('sbm')
    finally:
        sys.path.pop(0)
    preset_ok = (
        qm9.get('use_defog_split') is True
        and qm9.get('sample_steps') == 500
        and zinc.get('sample_steps') == 1000
        and sbm.get('transition') == 'absorbfirst'
    )
    checks.append(_result(
        'paper presets', preset_ok,
        'QM9 split/steps, ZINC steps, and SBM transition are pinned',
    ))

    requirements = (example / 'requirements.txt').read_text().lower()
    deps = ('rdkit', 'fcd', 'graph-tool', 'networkx', 'scipy', 'pyemd')
    missing_deps = [dep for dep in deps if dep not in requirements]
    checks.append(_result(
        'evaluation dependencies', not missing_deps,
        ', '.join(missing_deps) or 'optional evaluation dependencies documented',
    ))

    readme = (example / 'readme.md').read_text()
    checks.append(_result(
        'source attribution',
        'arxiv.org/abs/2410.04263' in readme and 'manuelmlmadeira/DeFoG' in readme,
        'paper and original implementation are linked',
    ))
    return checks


def run_test_modules(repo_root):
    repo_root = Path(repo_root).resolve()
    env = os.environ.copy()
    env['TL_BACKEND'] = 'torch'
    env['PYTHONPATH'] = str(repo_root)
    failures = []
    for name in REQUIRED_TESTS:
        path = repo_root / 'tests' / 'models' / name
        result = subprocess.run([sys.executable, str(path)], cwd=repo_root, env=env)
        if result.returncode:
            failures.append(name)
    return failures


def run_smoke(repo_root):
    repo_root = Path(repo_root).resolve()
    env = os.environ.copy()
    env['TL_BACKEND'] = 'torch'
    env['PYTHONPATH'] = str(repo_root)
    test = repo_root / 'tests' / 'models' / 'test_defog_smoke.py'
    return subprocess.run([sys.executable, str(test)], cwd=repo_root, env=env).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default=Path(__file__).resolve().parents[2])
    parser.add_argument('--tests', action='store_true', help='run direct DeFoG test modules')
    parser.add_argument('--smoke', action='store_true', help='run the synthetic training smoke test')
    parser.add_argument('--json', action='store_true', help='emit machine-readable output')
    args = parser.parse_args(argv)

    checks = audit_repository(args.repo_root)
    test_failures = run_test_modules(args.repo_root) if args.tests else []
    smoke_failed = bool(run_smoke(args.repo_root)) if args.smoke else False
    payload = {
        'checks': checks,
        'tests': {'status': 'fail' if test_failures else 'pass', 'failures': test_failures}
        if args.tests else {'status': 'not-run'},
        'smoke': {'status': 'fail' if smoke_failed else 'pass'}
        if args.smoke else {'status': 'not-run'},
        'manual_checks': list(MANUAL_CHECKS),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for check in checks:
            print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        if args.tests:
            print(f"[{'FAIL' if test_failures else 'PASS'}] direct test modules")
        if args.smoke:
            print(f"[{'FAIL' if smoke_failed else 'PASS'}] synthetic smoke")
        print(f'[MANUAL] {len(MANUAL_CHECKS)} paper-scale checks remain experimental')

    failed = any(check['status'] == 'fail' for check in checks)
    return int(failed or test_failures or smoke_failed)


if __name__ == '__main__':
    raise SystemExit(main())
