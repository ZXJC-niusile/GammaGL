import ast
import os
import sys

import torch

os.environ['TL_BACKEND'] = 'torch'

EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'defog')
)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

import extra_features as features


def test_feature_placeholder_has_single_definition():
    source_path = os.path.join(EXAMPLE_DIR, 'extra_features.py')
    with open(source_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read())
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'DenseFeaturePlaceHolder'
    ]
    assert len(definitions) == 1


def test_all_feature_families_return_same_placeholder_type():
    noisy = {
        'X_t': torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        'E_t': torch.tensor(
            [[
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]]
        ),
        'node_mask': torch.tensor([[True, True]]),
    }

    dummy = features.DummyExtraFeatures()(noisy)
    structural = features.ExtraFeatures(
        extra_features_type='none', dataset_info={'max_n_nodes': 2}
    )(noisy)
    molecular = features.ExtraMolecularFeatures({
        'valencies': [4, 3],
        'atom_weights': [12.0, 14.0],
    })(noisy)

    assert type(dummy) is features.DenseFeaturePlaceHolder
    assert type(structural) is features.DenseFeaturePlaceHolder
    assert type(molecular) is features.DenseFeaturePlaceHolder


if __name__ == '__main__':
    test_feature_placeholder_has_single_definition()
    test_all_feature_families_return_same_placeholder_type()
    print('feature placeholder tests passed')
