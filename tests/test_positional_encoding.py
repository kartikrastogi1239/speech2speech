"""Unit tests for positional encoding modules."""

import pytest
import torch
from src.models.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
)


def test_sinusoidal_positional_encoding_shapes():
    """Test output shapes of sinusoidal positional encoding."""
    d_model = 64
    max_len = 100
    batch_size = 4
    seq_len = 25

    pe_module = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    x = torch.zeros(batch_size, seq_len, d_model)

    out = pe_module(x)
    assert out.shape == (batch_size, seq_len, d_model)

    # Check that registered buffer is not updated by optimizers
    assert "pe" in dict(pe_module.named_buffers())
    assert len(list(pe_module.parameters())) == 0


def test_sinusoidal_positional_encoding_overflow():
    """Test that requesting sequence longer than max_len raises ValueError."""
    pe_module = SinusoidalPositionalEncoding(d_model=32, max_len=50)
    x = torch.randn(2, 60, 32)
    with pytest.raises(ValueError):
        pe_module(x)


def test_learnable_positional_encoding_shapes():
    """Test learnable positional encoding output shapes and parameter presence."""
    d_model = 64
    max_len = 100
    batch_size = 4
    seq_len = 20

    lpe = LearnablePositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    x = torch.randn(batch_size, seq_len, d_model)

    out = lpe(x)
    assert out.shape == (batch_size, seq_len, d_model)
    assert len(list(lpe.parameters())) == 1  # pos_embedding.weight
