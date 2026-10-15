"""Learning rate schedulers tailored for Transformer optimization."""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLRScheduler(_LRScheduler):
    """Noam learning rate schedule from 'Attention Is All You Need'.

    learning_rate = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 256,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]


class CosineWarmupLRScheduler(_LRScheduler):
    """Cosine annealing learning rate schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 2000,
        total_steps: int = 50000,
        min_lr_ratio: float = 0.05,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / max(1, self.warmup_steps)
            return [base_lr * progress for base_lr in self.base_lrs]

        # Cosine decay
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
        return [base_lr * decay_factor for base_lr in self.base_lrs]
