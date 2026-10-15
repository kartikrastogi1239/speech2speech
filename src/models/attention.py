"""Multi-Head Attention and Scaled Dot-Product Attention from scratch."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism.

    Computes:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + Mask) * V
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for scaled dot-product attention.

        Args:
            query: Tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
            key: Tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
            value: Tensor of shape (batch_size, num_heads, seq_len_v, head_dim)
            mask: Optional broadcastable mask tensor. Positions with True or 1
                  (or negative infinity for additive masks) will be masked out.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context: Tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
                - attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        head_dim = query.size(-1)
        # Raw attention scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            # Handle both boolean masks and additive float masks
            if mask.dtype == torch.bool:
                # True indicates positions to be masked out
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                # Additive mask (e.g., 0.0 for keep, -inf for mask)
                scores = scores + mask

        attention_weights = F.softmax(scores, dim=-1)

        # Handle edge cases with all -inf (e.g. completely padded rows)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module as described in 'Attention Is All You Need'.

    Projects queries, keys, and values into `num_heads` lower-dimensional spaces,
    performs scaled dot-product attention in parallel, and projects back to `d_model`.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear projection
        self.w_o = nn.Linear(d_model, d_model)

        # Attention computation
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden dimension into (num_heads, head_dim) and transpose.

        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and combine heads back into hidden dimension.

        Input: (batch_size, num_heads, seq_len, head_dim)
        Output: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Tensor of shape (batch_size, seq_len_q, d_model)
            key: Tensor of shape (batch_size, seq_len_k, d_model)
            value: Tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k),
                  (1, 1, seq_len_q, seq_len_k), or (batch_size, 1, 1, seq_len_k).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Tensor of shape (batch_size, seq_len_q, d_model)
                - attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # Linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention across all heads
        context, attn_weights = self.attention(q, k, v, mask=mask)

        # Concatenate heads and project to d_model
        output = self._combine_heads(context)
        output = self.w_o(output)

        return output, attn_weights
