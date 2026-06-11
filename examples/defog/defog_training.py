"""Training data utilities for the DeFoG example."""

import numpy as np
import tensorlayerx as tlx

from gammagl.loader import DataLoader
from gammagl.loader.dataloader import Collater
from tensorlayerx.dataflow import BatchSampler


class SeededRandomSampler:
    """Shuffle deterministically with ``seed + epoch``."""

    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = np.arange(len(self.data_source))
        np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)


def create_training_loader(graphs, batch_size, seed, num_workers=8):
    sampler = SeededRandomSampler(graphs, seed=seed)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    collater = Collater(follow_batch=None, exclude_keys=None)
    try:
        from torch.utils.data import DataLoader as TorchDataLoader
        loader = TorchDataLoader(
            graphs, batch_sampler=batch_sampler, collate_fn=collater,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context='spawn' if num_workers > 0 else None,
        )
        return loader, f'PyTorch, num_workers={num_workers}'
    except Exception as error:
        loader = DataLoader(
            graphs, batch_sampler=batch_sampler, collate_fn=collater
        )
        return loader, f'GammaGL fallback ({error})'


def batch_global_features(batch, batch_size, conditional):
    if not conditional or not hasattr(batch, 'y') or batch.y is None:
        return tlx.zeros([batch_size, 0], dtype=tlx.float32)
    y = batch.y
    if len(y.shape) == 1:
        y = tlx.reshape(y, [batch_size, -1])
    elif len(y.shape) > 2:
        y = tlx.reshape(y, [batch_size, -1])
    if y.shape[-1] == 0:
        return tlx.zeros([batch_size, 0], dtype=tlx.float32)
    return tlx.cast(y, tlx.float32)
