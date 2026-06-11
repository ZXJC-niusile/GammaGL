import torch


class TorchAdamW:
    """Expose ``torch.optim.AdamW`` through the TLX optimizer protocol."""

    def __init__(self, lr, weight_decay=0.0, amsgrad=True, grad_clip_norm=None):
        self.options = dict(lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        self.grad_clip_norm = grad_clip_norm
        self.optimizer = None

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError('Parameter train_weights must be entered.')
        weights = list(weights)
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(weights, **self.options)
        self.optimizer.zero_grad()
        if not torch.isfinite(loss):
            print('[warn:optim] Non-finite loss; skipping step', flush=True)
            return [torch.zeros_like(w) for w in weights] if return_grad else None

        loss.backward()
        for weight in weights:
            if weight.grad is not None and not torch.isfinite(weight.grad).all():
                torch.nan_to_num_(weight.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if self.grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(weights, self.grad_clip_norm)
            if not torch.isfinite(norm):
                for weight in weights:
                    if weight.grad is not None:
                        weight.grad.zero_()
        return [weight.grad for weight in weights] if return_grad else None

    def apply_gradients(self, grads_and_vars=None, closure=None):
        if self.optimizer is None:
            raise AttributeError('Call gradient() first.')
        return self.optimizer.step(closure) if closure else self.optimizer.step()
