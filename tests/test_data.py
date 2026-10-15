"""Unit tests for tokenizer and data collator."""

import torch
from src.data.tokenizer import SimpleCharSubwordTokenizer
from src.data.collate import SpeechTranslationCollator


def test_tokenizer_encode_decode():
    """Verify that tokenizer encodes and decodes text faithfully."""
    tokenizer = SimpleCharSubwordTokenizer()
    sample_corpus = [
     "नमस्ते दुनिया",  # Hindi
     "hola mundo",      # Spanish
    ]
    tokenizer.fit_on_texts(sample_corpus)

    test_str = "hola mundo"
    token_ids = tokenizer.encode(test_str, add_special_tokens=True)

    # Check BOS and EOS
    assert token_ids[0] == tokenizer.bos_id
    assert token_ids[-1] == tokenizer.eos_id

    # Decode and verify reconstruction
    decoded_str = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded_str == test_str


def test_speech_translation_collator_padding_and_masks():
    """Verify dynamic padding and causal mask creation in collator."""
    collator = SpeechTranslationCollator(pad_token_id=0)

    # Create synthetic variable-length batch
    batch_raw = [
        {
            "audio_features": torch.randn(20, 80),
            "src_tokens": torch.tensor([1, 10, 20, 2]),
            "tgt_tokens": torch.tensor([1, 5, 6, 7, 2]),  # length 5
        },
        {
            "audio_features": torch.randn(35, 80),
            "src_tokens": torch.tensor([1, 15, 2]),
            "tgt_tokens": torch.tensor([1, 8, 2]),        # length 3
        },
    ]

    batch = collator(batch_raw)

    # Audio padding: max length is 35
    assert batch["audio_features"].shape == (2, 35, 80)
    assert batch["audio_lengths"].tolist() == [20, 35]

    # Target teacher forcing splits:
    # Max target tokens is 5 -> tgt_input has length 4, tgt_label has length 4
    assert batch["tgt_input"].shape == (2, 4)
    assert batch["tgt_label"].shape == (2, 4)

    # Verify causal mask shape: (4, 4)
    assert batch["tgt_causal_mask"].shape == (4, 4)
    assert batch["tgt_causal_mask"].dtype == torch.bool
    assert batch["tgt_causal_mask"][0, 1].item() is True  # Future position masked
    assert batch["tgt_causal_mask"][1, 0].item() is False # Past position not masked
