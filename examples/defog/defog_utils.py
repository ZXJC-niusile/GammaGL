import numpy as np
import torch
import tensorlayerx as tlx
from gammagl.utils import to_dense_adj, to_dense_batch

class PlaceHolder:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = y

    def mask(self, node_mask):
        X, E = apply_node_mask(self.X, self.E, node_mask)
        return PlaceHolder(X=X, E=E, y=self.y)

    def split(self, node_mask):
        r"""Split a batched PlaceHolder into a list of individual graphs."""
        bs = node_mask.shape[0]
        n_nodes = tlx.reduce_sum(tlx.cast(node_mask, tlx.int64), axis=1)
        n_nodes = tlx.convert_to_numpy(n_nodes)
        graphs = []
        X_np = tlx.convert_to_numpy(self.X)
        E_np = tlx.convert_to_numpy(self.E)
        for i in range(bs):
            n = int(n_nodes[i])
            xi = tlx.convert_to_tensor(X_np[i, :n])
            ei = tlx.convert_to_tensor(E_np[i, :n, :n])
            graphs.append((xi, ei))
        return graphs

    def __repr__(self):
        x_shape = self.X.shape if hasattr(self.X, 'shape') else self.X
        e_shape = self.E.shape if hasattr(self.E, 'shape') else self.E
        y_shape = self.y.shape if (self.y is not None and hasattr(self.y, 'shape')) else self.y
        return f"PlaceHolder(X={x_shape}, E={e_shape}, y={y_shape})"


# ============================================================
# Dense conversion utilities
# ============================================================

def apply_node_mask(X, E, node_mask):
    """Zero-out features of padded nodes (node_mask=False)."""
    x_mask = tlx.expand_dims(tlx.cast(node_mask, X.dtype), axis=-1)
    e_mask = tlx.expand_dims(x_mask, axis=2) * tlx.expand_dims(x_mask, axis=1)
    
    X_masked = X * x_mask
    E_masked = E * e_mask
    return X_masked, E_masked


def to_dense(x, edge_index, edge_attr, batch):
    r"""Convert sparse graph to dense representation.

    Parameters
    ----------
    x : tensor
        Node features ``(N_total, dx)``.
    edge_index : tensor
        Edge indices ``(2, E_total)``.
    edge_attr : tensor
        Edge attributes ``(E_total, de)``.
    batch : tensor
        Batch assignment vector ``(N_total,)``.

    Returns
    -------
    PlaceHolder, node_mask
        Dense graph data and boolean node mask.
    """
    X, node_mask = to_dense_batch(x, batch)

    max_num_nodes = X.shape[1]

    # Remove self-loops
    src = edge_index[0]
    dst = edge_index[1]
    mask = src != dst
    edge_index_clean = edge_index[:, mask]
    edge_attr_clean = edge_attr[mask] if edge_attr is not None else None

    E = to_dense_adj(
        edge_index_clean,
        batch=batch,
        edge_attr=edge_attr_clean,
        max_num_nodes=max_num_nodes,
    )

    if len(E.shape) == 3:
        E = tlx.expand_dims(E, axis=-1)

    E = encode_no_edge(E)

    # Apply node_mask to zero out padding positions
    X, E = apply_node_mask(X, E, node_mask)

    node_mask = tlx.cast(node_mask, tlx.bool)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    r"""Encode 'no-edge' as the first channel (index 0).

    Parameters
    ----------
    E : tensor
        Edge features ``(bs, n, n, de)``.

    Returns
    -------
    tensor
        Modified E with ``E[:,:,:,0] = 1`` where no edge exists.
    """
    if len(E.shape) != 4:
        return E
    if E.shape[-1] == 0:
        return E

    no_edge = tlx.cast(
        tlx.reduce_sum(E, axis=-1, keepdims=True) == 0,
        E.dtype,
    )
    E = tlx.concat([E[..., :1] + no_edge, E[..., 1:]], axis=-1)

    n = E.shape[1]
    diagonal = torch.eye(n, dtype=E.dtype, device=E.device).reshape(
        1, n, n, 1
    )
    return E * (1.0 - diagonal)


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    r"""Exponential Moving Average of model parameters.

    Maintains shadow copies of all trainable parameters and updates them
    after each training step: ``shadow = decay * shadow + (1 - decay) * param``.

    During sampling/evaluation, the EMA weights are swapped in place of the
    original weights, then restored afterwards.

    Parameters
    ----------
    model : tlx.nn.Module
        The model whose parameters to track.
    decay : float
        EMA decay factor (e.g. 0.999). Higher = slower update.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow_params = {}
        self._backup_params = None
        # Initialize shadow parameters as clones of model params
        for name, param in model.named_parameters():
            self.shadow_params[name] = param.clone().detach()

    def update(self, model):
        """Update shadow parameters after a training step."""
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                new_val = self.decay * self.shadow_params[name] + (1.0 - self.decay) * param.detach()
                self.shadow_params[name] = new_val
            else:
                self.shadow_params[name] = param.clone().detach()

    def swap_in(self, model):
        """Replace model parameters with EMA shadow parameters."""
        self._backup_params = {}
        for name, param in model.named_parameters():
            self._backup_params[name] = param.clone().detach()
            param.data.copy_(self.shadow_params[name])

    def swap_out(self, model):
        """Restore original model parameters from backup."""
        if self._backup_params is not None:
            for name, param in model.named_parameters():
                param.data.copy_(self._backup_params[name])
            self._backup_params = None

    def state_dict(self):
        """Return state for saving."""
        import pickle, io
        buf = io.BytesIO()
        shadow_np = {k: tlx.convert_to_numpy(v) for k, v in self.shadow_params.items()}
        pickle.dump({'shadow_params': shadow_np, 'decay': self.decay}, buf)
        return buf.getvalue()

    def load_state_dict(self, state_bytes):
        """Load state from saved bytes."""
        import pickle, io
        buf = io.BytesIO(state_bytes)
        data = pickle.loads(buf.read())

        if isinstance(data, dict) and 'shadow_params' in data:
            self.decay = data.get('decay', self.decay)
            shadow_params = data['shadow_params']
        else:
            shadow_params = data

        for k, v in shadow_params.items():
            self.shadow_params[k] = tlx.convert_to_tensor(v)



