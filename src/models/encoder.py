"""Transformer Encoder and Encoder Layer implemented from scratch."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer Encoder.

    Consists of:
    1. Multi-Head Self-Attention + Residual connection + Layer Normalization.
    2. Position-wise Feed-Forward Network + Residual connection + Layer Normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        # Multi-Head Self Attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed Forward Network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for an encoder layer.

        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            src_mask: Optional mask for padding or restricted attention.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        if self.norm_first:
            # Pre-LN architecture
            norm_x = self.norm1(src)
            attn_out, attn_weights = self.self_attn(
                query=norm_x,
                key=norm_x,
                value=norm_x,
                mask=src_mask,
            )
            src = src + self.dropout1(attn_out)

            norm_x = self.norm2(src)
            ff_out = self.feed_forward(norm_x)
            src = src + self.dropout2(ff_out)
        else:
            # Post-LN architecture (Vaswani et al.)
            attn_out, attn_weights = self.self_attn(
                query=src,
                key=src,
                value=src,
                mask=src_mask,
            )
            src = self.norm1(src + self.dropout1(attn_out))

            ff_out = self.feed_forward(src)
            src = self.norm2(src + self.dropout2(ff_out))

        return src, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of N Transformer Encoder Layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass through the encoder stack.

        Args:
            src: Tensor of shape (batch_size, seq_len, d_model)
            src_mask: Optional source padding mask.
            return_all_attentions: If True, returns attention weights from all layers.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
                - Final encoder representations of shape (batch_size, seq_len, d_model)
                - List of attention weights from each layer if return_all_attentions=True, else None
        """
        all_attentions = [] if return_all_attentions else None

        x = src
        for layer in self.layers:
            x, attn = layer(x, src_mask=src_mask)
            if return_all_attentions:
                all_attentions.append(attn)

        x = self.final_norm(x)
        return x, all_attentions
