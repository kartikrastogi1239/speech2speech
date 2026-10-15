"""Unit tests for the complete Seq2SeqTransformer, Encoder, Decoder, and Causal Masking."""

import pytest
import torch
from src.models.encoder import TransformerEncoderLayer, TransformerEncoder
from src.models.decoder import TransformerDecoderLayer, TransformerDecoder
from src.models.transformer import Seq2SeqTransformer, AudioFrontendAdapter


def test_encoder_layer_and_stack_shapes():
    """Verify shapes through EncoderLayer and stacked TransformerEncoder."""
    d_model = 64
    num_heads = 4
    dim_feedforward = 128
    batch_size = 2
    seq_len = 10

    encoder = TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_layers=2,
        dropout=0.0,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    out, attentions = encoder(x, return_all_attentions=True)

    assert out.shape == (batch_size, seq_len, d_model)
    assert len(attentions) == 2
    assert attentions[0].shape == (batch_size, num_heads, seq_len, seq_len)


def test_decoder_layer_and_stack_shapes():
    """Verify shapes through DecoderLayer and stacked TransformerDecoder."""
    d_model = 64
    num_heads = 4
    dim_feedforward = 128
    batch_size = 2
    seq_len_tgt = 8
    seq_len_src = 12

    decoder = TransformerDecoder(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_layers=2,
        dropout=0.0,
    )

    tgt = torch.randn(batch_size, seq_len_tgt, d_model)
    memory = torch.randn(batch_size, seq_len_src, d_model)

    out, attns = decoder(tgt=tgt, memory=memory, return_all_attentions=True)

    assert out.shape == (batch_size, seq_len_tgt, d_model)
    assert len(attns["self_attentions"]) == 2
    assert len(attns["cross_attentions"]) == 2
    assert attns["self_attentions"][0].shape == (batch_size, num_heads, seq_len_tgt, seq_len_tgt)
    assert attns["cross_attentions"][0].shape == (batch_size, num_heads, seq_len_tgt, seq_len_src)


def test_seq2seq_transformer_text_forward():
    """Test full forward pass with text token inputs."""
    src_vocab_size = 200
    tgt_vocab_size = 300
    d_model = 64
    batch_size = 3
    src_len = 14
    tgt_len = 10

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )

    src_tokens = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt_tokens = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    causal_mask = Seq2SeqTransformer.generate_square_subsequent_mask(tgt_len)
    src_pad_mask = Seq2SeqTransformer.create_padding_mask(src_tokens, pad_idx=0)

    logits = model(
        src=src_tokens,
        tgt=tgt_tokens,
        src_mask=src_pad_mask,
        tgt_mask=causal_mask,
        is_audio=False,
    )

    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size)


def test_seq2seq_transformer_audio_forward():
    """Test full forward pass with continuous audio feature inputs (e.g. log-mel)."""
    tgt_vocab_size = 300
    d_model = 64
    audio_dim = 80
    batch_size = 2
    audio_time_frames = 64
    tgt_len = 12

    model = Seq2SeqTransformer(
        src_vocab_size=0,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        audio_input_dim=audio_dim,
        audio_subsample_factor=4,
        dropout=0.0,
    )

    audio_features = torch.randn(batch_size, audio_time_frames, audio_dim)
    tgt_tokens = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    causal_mask = Seq2SeqTransformer.generate_square_subsequent_mask(tgt_len)

    logits = model(
        src=audio_features,
        tgt=tgt_tokens,
        tgt_mask=causal_mask,
        is_audio=True,
    )

    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size)


def test_transformer_causal_masking_invariant():
    """Crucial test: Changing token at position t+1 must NOT alter predictions at position t."""
    torch.manual_seed(42)
    model = Seq2SeqTransformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    model.eval()

    src = torch.randint(1, 100, (1, 8))
    tgt_a = torch.tensor([[10, 20, 30, 40, 50]])
    tgt_b = torch.tensor([[10, 20, 30, 99, 88]])  # tokens at index 3 and 4 are changed

    causal_mask = Seq2SeqTransformer.generate_square_subsequent_mask(5)

    with torch.no_grad():
        out_a = model(src, tgt_a, tgt_mask=causal_mask)
        out_b = model(src, tgt_b, tgt_mask=causal_mask)

    # Predictions at indices 0, 1, 2 must be identical because decoder cannot look ahead
    assert torch.allclose(out_a[:, :3, :], out_b[:, :3, :], atol=1e-5)


def test_backward_gradient_flow():
    """Verify that end-to-end backpropagation computes gradients for all layers."""
    model = Seq2SeqTransformer(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=32,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )

    src = torch.randint(1, 50, (2, 6))
    tgt = torch.randint(1, 50, (2, 5))
    target_labels = torch.randint(0, 50, (2, 5))

    causal_mask = Seq2SeqTransformer.generate_square_subsequent_mask(5)
    logits = model(src, tgt, tgt_mask=causal_mask)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, 50), target_labels.view(-1))
    loss.backward()

    # Check that key parameters received gradients
    assert model.encoder.layers[0].self_attn.w_q.weight.grad is not None
    assert model.decoder.layers[0].cross_attn.w_q.weight.grad is not None
    assert model.output_head.weight.grad is not None
