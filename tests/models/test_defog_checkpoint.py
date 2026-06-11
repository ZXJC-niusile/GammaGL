import json
import os
import sys
import tempfile


EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

import defog_checkpoint as checkpoint


class FakeModel:
    def __init__(self):
        self.saved = []
        self.loaded = []

    def save_weights(self, path, format=None):
        self.saved.append((path, format))
        with open(path, 'wb') as f:
            f.write(b'model')

    def load_weights(self, path, format=None):
        self.loaded.append((path, format))


class FakeEMA:
    def __init__(self):
        self.loaded = None
        self.swapped = False

    def state_dict(self):
        return b'ema-state'

    def load_state_dict(self, value):
        self.loaded = value

    def swap_in(self, model):
        self.swapped = True


def test_snapshot_priority_and_ema_restore():
    with tempfile.TemporaryDirectory() as save_dir:
        model = FakeModel()
        ema = FakeEMA()
        checkpoint.save_snapshot(model, ema, save_dir, 'last', {'X': 2, 'E': 2, 'y': 0})
        checkpoint.save_snapshot(model, ema, save_dir, 'best')

        path, _ = checkpoint.find_snapshot(save_dir, ('best', 'last'))
        assert path.endswith('best_model.npz')

        loaded_path, loaded_ema = checkpoint.load_snapshot(
            model, save_dir, ('last', 'best'), ema=FakeEMA(), swap_ema=True
        )
        assert loaded_path.endswith('last_model.npz')
        assert loaded_ema.loaded == b'ema-state'
        assert loaded_ema.swapped

        _, disabled_ema = checkpoint.load_snapshot(
            model, save_dir, ('last', 'best'), ema=None, swap_ema=False
        )
        assert disabled_ema is None

        with open(os.path.join(save_dir, 'model_config.json')) as f:
            assert json.load(f)['output_dims']['X'] == 2

        checkpoint.delete_snapshot(save_dir, 'best')
        assert not os.path.exists(os.path.join(save_dir, 'best_model.npz'))
        assert not os.path.exists(os.path.join(save_dir, 'best_ema.pkl'))


def test_optional_missing_snapshot_and_training_state():
    with tempfile.TemporaryDirectory() as save_dir:
        model = FakeModel()
        assert checkpoint.load_snapshot(model, save_dir, ('last', 'best'), required=False) == (None, None)
        assert checkpoint.load_training_state(save_dir) == (float('-inf'), None)

        checkpoint.save_training_state(save_dir, 0.75, 12)
        assert checkpoint.load_training_state(save_dir) == (0.75, 12)


if __name__ == '__main__':
    test_snapshot_priority_and_ema_restore()
    test_optional_missing_snapshot_and_training_state()
