import argparse
import hashlib
import json
import os
import sys


EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from defog_config import (
    _BASE_EXPERIMENT_PRESET,
    _DATASET_PRESETS,
    apply_dataset_preset,
    get_dataset_preset,
)


def test_presets_match_pre_extraction_values():
    payload = {
        '_BASE_EXPERIMENT_PRESET': _BASE_EXPERIMENT_PRESET,
        '_DATASET_PRESETS': _DATASET_PRESETS,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    assert digest == '3fa21953c958cdaae17e32a7d71278adf138fd292bd6748facfa6e5795f54e44'
    assert set(_DATASET_PRESETS) == {
        'planar', 'tree', 'sbm', 'qm9', 'zinc250k',
        'guacamol', 'moses', 'tls', 'comm20',
    }


def test_explicit_cli_values_override_dataset_preset():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--omega', type=float, default=0.0)
    parser.add_argument('--sample_steps', type=int, default=10)
    argv = ['--dataset', 'zinc250k', '--batch_size', '17']
    args = apply_dataset_preset(parser.parse_args(argv), parser, argv)

    assert args.batch_size == 17
    assert args.n_epochs == 300
    assert args.omega == 0.1
    assert args.sample_steps == 1000
    assert get_dataset_preset('zinc250k')['batch_size'] == 256


def test_synthetic_configuration_is_unchanged():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic')
    parser.add_argument('--batch_size', type=int, default=40)
    args = parser.parse_args([])
    assert apply_dataset_preset(args, parser, []) is args
    assert args.batch_size == 40


if __name__ == '__main__':
    test_presets_match_pre_extraction_values()
    test_explicit_cli_values_override_dataset_preset()
    test_synthetic_configuration_is_unchanged()
