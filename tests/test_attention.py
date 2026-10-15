"""Unit tests for ScaledDotProductAttention and MultiHeadAttention."""

import pytest
import torch
from src.models.attention import ScaledDotProductAttention, MultiHeadAttention


def test_scaled_dot_product_attention_shapes():
    """Verify output shapes and probability normalization of scaled dot product attention."""
    attn = ScaledDotProductAttention(dropout=0.0)
    batch_size = 4
    num_heads = 8
    seq_len_q = 16
    seq_len_k = 20
    head_dim = 32

    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len_k, head_dim)

    context, weights = attn(q, k, v)

    # Check shapes
    assert context.shape == (batch_size, num_heads, seq_len_q, head_dim)
    assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    # Attention weights along the key dimension must sum to 1.0 (softmax property)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_scaled_dot_product_attention_causal_mask():
    """Verify that causal masking zeroes out future attention weights."""
    attn = ScaledDotProductAttention(dropout=0.0)
    batch_size = 2
    num_heads = 4
    seq_len = 6
    head_dim = 16

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Upper triangular mask: True for positions strictly above diagonal
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    _, weights = attn(q, k, v, mask=causal_mask)

    # Masked positions must have weight 0.0
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert torch.all(weights[:, :, i, j] == 0.0)


def test_multi_head_attention_shapes():
    """Verify that MultiHeadAttention preserves input sequence shapes."""
    d_model = 128
    num_heads = 4
    batch_size = 8
    seq_len_q = 10
    seq_len_kv = 15

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)

    q = torch.randn(batch_size, seq_len_q, d_model)
    k = torch.randn(batch_size, seq_len_kv, d_model)
    v = torch.randn(batch_size, seq_len_kv, d_model)

    output, attn_weights = mha(query=q, key=k, value=v)

    assert output.shape == (batch_size, seq_len_q, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)


def test_multi_head_attention_invalid_dim():
    """Verify that an exception is raised when d_model is not divisible by num_heads."""
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=100, num_heads=3)
