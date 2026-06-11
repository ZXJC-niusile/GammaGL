import os
import subprocess
import sys


def test_defog_dataset_package_does_not_shadow_huggingface_datasets():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    example_dir = os.path.join(repo_root, 'examples', 'defog')
    script = f"""
import importlib.util
import os
import sys

example_dir = {example_dir!r}
hf_spec = importlib.util.find_spec('datasets')
if hf_spec is not None:
    import datasets as hf_datasets
    hf_path = os.path.abspath(hf_datasets.__file__)
    assert 'examples/defog' not in hf_path.replace(os.sep, '/')

import types
fake_hf_datasets = types.ModuleType('datasets')
fake_hf_datasets.__file__ = '/fake/site-packages/datasets/__init__.py'
sys.modules['datasets'] = fake_hf_datasets

sys.path.insert(0, example_dir)
import defog_datasets
from dataset_utils import create_synthetic_dataset
from tls_metrics import compute_tls_metrics

assert sys.modules['datasets'] is fake_hf_datasets
assert defog_datasets.__name__ == 'defog_datasets'
assert callable(create_synthetic_dataset)
assert callable(compute_tls_metrics)
print(defog_datasets.__file__)
"""
    env = os.environ.copy()
    env['TL_BACKEND'] = 'torch'
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'defog_datasets' in result.stdout


if __name__ == '__main__':
    test_defog_dataset_package_does_not_shadow_huggingface_datasets()
    print('DeFoG dataset package test passed')
