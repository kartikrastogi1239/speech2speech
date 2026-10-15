"""End-to-end Sequence-to-Sequence Transformer architecture implemented from scratch."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .positional_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding


class AudioFrontendAdapter(nn.Module):
    """Subsampling convolution and linear projection adapter for audio feature sequences.

    Downsamples acoustic features (e.g., 80-channel log-mel filterbanks)
    along the time dimension to match the acoustic frame rate to textual token rates.
    """

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        subsample_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsample_factor = subsample_factor

        if subsample_factor == 4:
            # 2 layers of Conv1d with stride 2 = 4x temporal downsampling
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif subsample_factor == 2:
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            # Linear projection without temporal subsampling
            self.conv = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio frontend.

        Args:
            x: Audio features of shape (batch_size, time_steps, input_dim)

        Returns:
            Projected features of shape (batch_size, downsampled_time_steps, d_model)
        """
        if isinstance(self.conv, nn.Sequential):
            # Conv1d expects (batch_size, channels, time)
            x = x.transpose(1, 2)
            x = self.conv(x)
            # Transpose back to (batch_size, time, d_model)
            x = x.transpose(1, 2)
        else:
            x = self.conv(x)
        return x


class Seq2SeqTransformer(nn.Module):
    """Sequence-to-Sequence Transformer with custom Encoder, Decoder, and Multi-Head Attention.

    Supports:
    - Text-to-Text translation (Source Hindi text -> Target Spanish text)
    - Speech-to-Text translation (Source Hindi audio features -> Target Spanish text)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_length: int = 1024,
        audio_input_dim: Optional[int] = None,
        audio_subsample_factor: int = 4,
        positional_encoding_type: str = "sinusoidal",
        norm_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.audio_input_dim = audio_input_dim

        # Source Embeddings & Adapters
        self.src_tok_embedding = (
            nn.Embedding(src_vocab_size, d_model) if src_vocab_size > 0 else None
        )
        if audio_input_dim is not None:
            self.audio_adapter = AudioFrontendAdapter(
                input_dim=audio_input_dim,
                d_model=d_model,
                subsample_factor=audio_subsample_factor,
                dropout=dropout,
            )
        else:
            self.audio_adapter = None

        # Target Embeddings
        self.tgt_tok_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        if positional_encoding_type == "sinusoidal":
            self.src_pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=max_seq_length, dropout=dropout
            )
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=max_seq_length, dropout=dropout
            )
        else:
            self.src_pos_encoding = LearnablePositionalEncoding(
                d_model=d_model, max_len=max_seq_length, dropout=dropout
            )
            self.tgt_pos_encoding = LearnablePositionalEncoding(
                d_model=d_model, max_len=max_seq_length, dropout=dropout
            )

        # Encoder & Decoder Stacks
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        # Final projection to target vocabulary logits
        self.output_head = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with Xavier Uniform for weights and zeroes for biases."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate upper triangular boolean causal mask to prevent look-ahead in decoder.

        Positions with True will be masked out (disallowed from attending).

        Args:
            sz: Sequence length of target tokens.
            device: Target torch device.

        Returns:
            Tensor of shape (sz, sz) where upper triangle above diagonal is True.
        """
        # torch.triu(..., diagonal=1) returns 1s strictly above the diagonal
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    @staticmethod
    def create_padding_mask(
        seq: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:
        """Generate boolean mask where True indicates padded tokens.

        Args:
            seq: Integer tensor of shape (batch_size, seq_len)
            pad_idx: The integer index of padding token.

        Returns:
            Boolean tensor of shape (batch_size, 1, 1, seq_len) ready for broadcasting.
        """
        # (batch_size, 1, 1, seq_len)
        return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        is_audio: bool = False,
    ) -> torch.Tensor:
        """Encode source sequence into memory representations.

        Args:
            src: Token indices (batch_size, seq_len) OR audio features (batch_size, time, dim)
            src_mask: Optional source padding mask
            is_audio: Whether the input is continuous audio features

        Returns:
            Encoded representations of shape (batch_size, enc_seq_len, d_model)
        """
        if is_audio:
            if self.audio_adapter is None:
                raise ValueError("audio_adapter is not configured for this model.")
            x = self.audio_adapter(src)
        else:
            if self.src_tok_embedding is None:
                raise ValueError("src_tok_embedding is not configured for text inputs.")
            x = self.src_tok_embedding(src)

        x = self.src_pos_encoding(x)
        memory, _ = self.encoder(x, src_mask=src_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target tokens attending to encoder memory representations.

        Args:
            tgt: Target token tensor of shape (batch_size, seq_len_tgt)
            memory: Encoded representations of shape (batch_size, seq_len_src, d_model)
            tgt_mask: Target causal/padding mask
            memory_mask: Memory padding mask

        Returns:
            Decoded representations of shape (batch_size, seq_len_tgt, d_model)
        """
        x = self.tgt_tok_embedding(tgt)
        x = self.tgt_pos_encoding(x)
        out, _ = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return out

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_audio: bool = False,
    ) -> torch.Tensor:
        """Complete forward pass: Encode source -> Decode target -> Project to vocabulary.

        Args:
            src: Source tokens (batch_size, src_len) or audio features (batch_size, time, dim)
            tgt: Target tokens (batch_size, tgt_len)
            src_mask: Mask for source padding
            tgt_mask: Mask for target (causal and/or padding)
            memory_mask: Mask for encoder memory padding
            is_audio: Whether src represents continuous audio frames

        Returns:
            Logits of shape (batch_size, tgt_len, tgt_vocab_size)
        """
        memory = self.encode(src=src, src_mask=src_mask, is_audio=is_audio)
        decoded = self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        logits = self.output_head(decoded)
        return logits
