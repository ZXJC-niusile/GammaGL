"""Checkpoint I/O for the DeFoG example."""

import json
import math
import os

from defog_utils import EMA


def _paths(save_dir, prefix):
    base = os.path.join(save_dir, prefix)
    return f'{base}_model.npz', f'{base}_ema.pkl'


def find_snapshot(save_dir, prefixes):
    for prefix in prefixes:
        model_path, ema_path = _paths(save_dir, prefix)
        if os.path.exists(model_path):
            return model_path, ema_path
    return None, None


def save_snapshot(model, ema, save_dir, prefix, output_dims=None):
    model_path, ema_path = _paths(save_dir, prefix)
    model.save_weights(model_path, format='npz_dict')
    if ema is not None:
        with open(ema_path, 'wb') as f:
            f.write(ema.state_dict())
    elif os.path.exists(ema_path):
        os.remove(ema_path)

    if output_dims is not None:
        with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
            json.dump({'output_dims': output_dims}, f, indent=2)
    return model_path, ema_path


def delete_snapshot(save_dir, prefix):
    for path in _paths(save_dir, prefix):
        if os.path.exists(path):
            os.remove(path)


def load_snapshot(model, save_dir, prefixes, ema=None, ema_decay=0.0,
                  swap_ema=False, required=True):
    model_path, ema_path = find_snapshot(save_dir, prefixes)
    if model_path is None:
        if required:
            expected = ' or '.join(f'{name}_model.npz' for name in prefixes)
            raise FileNotFoundError(
                f'No checkpoint found in {save_dir}. Expected {expected}'
            )
        return None, ema

    model.load_weights(model_path, format='npz_dict')
    if os.path.exists(ema_path) and (ema is not None or swap_ema):
        ema = ema or EMA(model, decay=max(float(ema_decay), 0.999))
        with open(ema_path, 'rb') as f:
            ema.load_state_dict(f.read())
        if swap_ema:
            ema.swap_in(model)
    return model_path, ema


def save_training_state(save_dir, best_score, best_epoch):
    with open(os.path.join(save_dir, 'training_state.json'), 'w') as f:
        json.dump({'best_score': best_score, 'best_epoch': best_epoch}, f, indent=2)


def load_training_state(save_dir):
    path = os.path.join(save_dir, 'training_state.json')
    if not os.path.exists(path):
        return float('-inf'), None
    with open(path, 'r') as f:
        state = json.load(f)
    return float(state.get('best_score', float('-inf'))), state.get('best_epoch')


def save_best_if_improved(model, ema, save_dir, score, epoch,
                          best_score, best_epoch, output_dims=None):
    r"""Persist a new best snapshot only for a finite strict improvement."""
    score = float(score)
    if not math.isfinite(score) or score <= best_score:
        return best_score, best_epoch, False

    save_snapshot(model, ema, save_dir, 'best', output_dims)
    save_training_state(save_dir, score, epoch)
    return score, epoch, True
