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

    def save_weights(self, path, format=None):
        self.saved.append((path, format))
        with open(path, 'wb') as file:
            file.write(b'model')


def test_best_checkpoint_updates_only_on_strict_finite_improvement():
    with tempfile.TemporaryDirectory() as save_dir:
        model = FakeModel()
        best_score, best_epoch, improved = checkpoint.save_best_if_improved(
            model, None, save_dir, 0.7, 4, float('-inf'), None, {'X': 2}
        )
        assert improved
        assert (best_score, best_epoch) == (0.7, 4)
        assert checkpoint.load_training_state(save_dir) == (0.7, 4)
        assert len(model.saved) == 1

        for score in (0.7, 0.6, float('nan'), float('inf'), float('-inf')):
            new_score, new_epoch, improved = checkpoint.save_best_if_improved(
                model, None, save_dir, score, 5, best_score, best_epoch
            )
            assert not improved
            assert (new_score, new_epoch) == (best_score, best_epoch)
        assert len(model.saved) == 1
        assert checkpoint.load_training_state(save_dir) == (0.7, 4)


def test_later_improvement_replaces_snapshot_and_state():
    with tempfile.TemporaryDirectory() as save_dir:
        model = FakeModel()
        best_score, best_epoch, _ = checkpoint.save_best_if_improved(
            model, None, save_dir, -2.0, 2, float('-inf'), None
        )
        best_score, best_epoch, improved = checkpoint.save_best_if_improved(
            model, None, save_dir, -1.5, 7, best_score, best_epoch
        )
        assert improved
        assert (best_score, best_epoch) == (-1.5, 7)
        assert len(model.saved) == 2
        assert checkpoint.load_training_state(save_dir) == (-1.5, 7)


if __name__ == '__main__':
    test_best_checkpoint_updates_only_on_strict_finite_improvement()
    test_later_improvement_replaces_snapshot_and_state()
    print('DeFoG best checkpoint tests passed')
