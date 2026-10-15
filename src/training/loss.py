"""Loss functions with label smoothing for Sequence-to-Sequence training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Cross-Entropy Loss with label smoothing.

    Prevents model overconfidence on target token probabilities:
        q(k) = (1 - epsilon) * 1(k == y) + epsilon / K
    """

    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross entropy loss.

        Args:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size) or (N, vocab_size)
            targets: Target tensor of shape (batch_size, seq_len) or (N,)

        Returns:
            Scalar loss tensor.
        """
        # Reshape to (N, vocab_size) and (N,)
        if logits.dim() == 3:
            vocab_size = logits.size(-1)
            logits = logits.contiguous().view(-1, vocab_size)
            targets = targets.contiguous().view(-1)

        # PyTorch F.cross_entropy natively supports label_smoothing in >=1.10
        return F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )
