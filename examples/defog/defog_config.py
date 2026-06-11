"""Dataset-specific experiment presets for the DeFoG example."""

import sys

_BASE_EXPERIMENT_PRESET = {
    'transition': 'marginal',
    'extra_features': 'rrwp',
    'rrwp_steps': 12,
    'n_layers': 5,
    'hidden_mlp_X': 256,
    'hidden_mlp_E': 128,
    'hidden_mlp_y': 128,
    'dx': 256,
    'de': 64,
    'dy': 64,
    'n_head': 8,
    'dim_ffX': 256,
    'dim_ffE': 128,
    'dim_ffy': 128,
    'n_epochs': 1000,
    'batch_size': 512,
    'lr': 2e-4,
    'weight_decay': 1e-12,
    'ema_decay': 0.0,
    'train_distortion': 'identity',
    'sample_steps': 1000,
    'sample_distortion': 'identity',
    'eta': 0.0,
    'omega': 0.0,
    'rdb': 'general',
    'rdb_crit': 'max_marginal',
    'num_sample_fold': 1,
    'sample_every_val': 4,
    'check_val_every_n_epochs': 5,
    'val_num_samples': 512,
}

_DATASET_PRESETS = {
    'planar': {
        'n_layers': 10,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 100000,
        'batch_size': 64,
        'sample_distortion': 'polydec',
        'omega': 0.05,
        'eta': 50.0,
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'tree': {
        'n_layers': 10,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 100000,
        'batch_size': 64,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'sbm': {
        'transition': 'absorbfirst',
        'rrwp_steps': 20,
        'n_layers': 8,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'de': 64,
        'dy': 64,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 50000,
        'batch_size': 32,
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'qm9': {
        'n_layers': 9,
        'n_epochs': 1000,
        'batch_size': 1024,
        'sample_steps': 500,
        'sample_distortion': 'polydec',
        'sample_every_val': 1,
        'check_val_every_n_epochs': 50,
        'val_num_samples': 512,
    },
    'zinc250k': {
        'rrwp_steps': 20,
        'n_layers': 12,
        'hidden_mlp_X': 256,
        'hidden_mlp_E': 128,
        'hidden_mlp_y': 256,
        'de': 64,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 300,
        'batch_size': 256,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_steps': 1000,
        'omega': 0.1,
        'eta': 300.0,
        'sample_every_val': 2,
        'check_val_every_n_epochs': 4,
        'val_num_samples': 256,
    },
    'guacamol': {
        'rrwp_steps': 20,
        'n_layers': 12,
        'hidden_mlp_X': 256,
        'hidden_mlp_E': 128,
        'hidden_mlp_y': 256,
        'de': 64,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 1000,
        'batch_size': 64,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_steps': 1000,
        'omega': 0.1,
        'eta': 300.0,
        'sample_every_val': 2,
        'check_val_every_n_epochs': 2,
        'val_num_samples': 500,
    },
    'moses': {
        'rrwp_steps': 20,
        'n_layers': 12,
        'hidden_mlp_X': 256,
        'hidden_mlp_E': 128,
        'hidden_mlp_y': 256,
        'de': 64,
        'dy': 128,
        'dim_ffX': 256,
        'dim_ffE': 128,
        'dim_ffy': 256,
        'n_epochs': 300,
        'batch_size': 256,
        'train_distortion': 'polydec',
        'sample_distortion': 'polydec',
        'sample_steps': 1000,
        'omega': 0.5,
        'eta': 200.0,
        'sample_every_val': 4,
        'check_val_every_n_epochs': 1,
        'val_num_samples': 256,
    },
    'tls': {
        'n_layers': 10,
        'rrwp_steps': 20,
        'hidden_mlp_X': 128,
        'hidden_mlp_E': 64,
        'hidden_mlp_y': 128,
        'dim_ffE': 64,
        'dim_ffy': 256,
        'n_epochs': 100000,
        'batch_size': 64,
        'sample_distortion': 'polydec',
        'omega': 0.05,
        'eta': 0.0,
        'sample_every_val': 1,
        'check_val_every_n_epochs': 2000,
        'val_num_samples': 40,
    },
    'comm20': {
        'n_layers': 8,
        'n_epochs': 1000000,
        'batch_size': 256,
        'sample_every_val': 10,
        'check_val_every_n_epochs': 1000,
        'val_num_samples': 20,
    },
}


def get_dataset_preset(dataset):
    preset = dict(_BASE_EXPERIMENT_PRESET)
    preset.update(_DATASET_PRESETS.get(dataset, {}))
    return preset


def _get_explicit_cli_dests(parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv or any(arg.startswith(opt + '=') for arg in argv):
                explicit.add(action.dest)
                break
    return explicit


def apply_dataset_preset(args, parser, argv=None):
    dataset = getattr(args, 'dataset', None)
    if dataset in (None, 'synthetic'):
        return args

    preset = get_dataset_preset(dataset)
    explicit = _get_explicit_cli_dests(parser, argv)
    applied = []

    for key, value in preset.items():
        if hasattr(args, key) and key not in explicit:
            setattr(args, key, value)
            applied.append(key)

    if applied:
        preview = ', '.join(applied[:8])
        suffix = ' ...' if len(applied) > 8 else ''
        print(f"Applied original DeFoG preset for {dataset}: {preview}{suffix}")

    return args
