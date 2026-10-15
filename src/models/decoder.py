"""Transformer Decoder and Decoder Layer implemented from scratch."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """A single layer of the Transformer Decoder.

    Consists of:
    1. Masked Multi-Head Self-Attention + Residual connection + Layer Normalization.
    2. Multi-Head Cross-Attention (attending to encoder memory) + Residual + LayerNorm.
    3. Position-wise Feed-Forward Network + Residual + LayerNorm.
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

        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 2. Encoder-Decoder Cross-Attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 3. Feed Forward Network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a decoder layer.

        Args:
            tgt: Target tensor of shape (batch_size, seq_len_tgt, d_model)
            memory: Encoder representations of shape (batch_size, seq_len_src, d_model)
            tgt_mask: Mask for target self-attention (causal and/or padding mask)
            memory_mask: Mask for encoder memory padding

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, seq_len_tgt, d_model)
                - Self-attention weights
                - Cross-attention weights
        """
        if self.norm_first:
            # Pre-LN architecture
            norm_tgt = self.norm1(tgt)
            self_out, self_attn = self.self_attn(
                query=norm_tgt,
                key=norm_tgt,
                value=norm_tgt,
                mask=tgt_mask,
            )
            tgt = tgt + self.dropout1(self_out)

            norm_tgt = self.norm2(tgt)
            cross_out, cross_attn = self.cross_attn(
                query=norm_tgt,
                key=memory,
                value=memory,
                mask=memory_mask,
            )
            tgt = tgt + self.dropout2(cross_out)

            norm_tgt = self.norm3(tgt)
            ff_out = self.feed_forward(norm_tgt)
            tgt = tgt + self.dropout3(ff_out)
        else:
            # Post-LN architecture (Vaswani et al.)
            self_out, self_attn = self.self_attn(
                query=tgt,
                key=tgt,
                value=tgt,
                mask=tgt_mask,
            )
            tgt = self.norm1(tgt + self.dropout1(self_out))

            cross_out, cross_attn = self.cross_attn(
                query=tgt,
                key=memory,
                value=memory,
                mask=memory_mask,
            )
            tgt = self.norm2(tgt + self.dropout2(cross_out))

            ff_out = self.feed_forward(tgt)
            tgt = self.norm3(tgt + self.dropout3(ff_out))

        return tgt, self_attn, cross_attn


class TransformerDecoder(nn.Module):
    """Stack of N Transformer Decoder Layers."""

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
                TransformerDecoderLayer(
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
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, List[torch.Tensor]]]]:
        """Forward pass through the decoder stack.

        Args:
            tgt: Target tensor of shape (batch_size, seq_len_tgt, d_model)
            memory: Encoder representations of shape (batch_size, seq_len_src, d_model)
            tgt_mask: Target self-attention mask (causal + padding)
            memory_mask: Memory padding mask
            return_all_attentions: If True, returns dictionary of self and cross attention maps

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, List[torch.Tensor]]]]:
                - Output tensor of shape (batch_size, seq_len_tgt, d_model)
                - Dictionary containing self_attentions and cross_attentions if requested
        """
        self_attentions = [] if return_all_attentions else None
        cross_attentions = [] if return_all_attentions else None

        x = tgt
        for layer in self.layers:
            x, s_attn, c_attn = layer(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            if return_all_attentions:
                self_attentions.append(s_attn)
                cross_attentions.append(c_attn)

        x = self.final_norm(x)

        attentions = None
        if return_all_attentions:
            attentions = {
                "self_attentions": self_attentions,
                "cross_attentions": cross_attentions,
            }

        return x, attentions
