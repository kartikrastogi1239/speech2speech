# Hindi-to-Spanish Speech-to-Speech Translation (S2ST)

A modular, extensible Speech-to-Speech Translation system from Hindi (`hi`) to Spanish (`es`) built with Python and PyTorch.

This project features:
1. **From-Scratch Transformer Architecture**: Core Multi-Head Attention, Scaled Dot-Product Attention, Positional Encodings, Feed-Forward Networks, and Encoder-Decoder Stacks implemented directly in PyTorch without relying on high-level black-box wrappers.
2. **Audio Frontend & Pretrained Audio Components**: Native Log-Mel spectrogram feature extraction and temporal subsampling adapter, coupled with Meta's Massively Multilingual Speech (MMS-TTS) for natural Spanish audio generation.
3. **Google Colab & GPU Ready**: Native mixed-precision training (`torch.amp`), gradient accumulation, configurable device selection (`cuda`, `mps`, `cpu`), and dedicated Colab runner notebooks.
4. **Rigorous Test Suite**: Comprehensive tests verifying mathematical invariants (causal mask isolation, attention softmax normalization, gradient backpropagation, and tensor dimensional transitions).

---

## 1. Architectural Research & Dataset Realities

### Speech-to-Speech Translation (S2ST) Paradigms
1. **Direct Speech-to-Unit Translation (S2UT / Discrete Acoustic Units)**:
   - Speech $\to$ Discrete Acoustic Tokens (e.g. EnCodec, HuBERT k-means) $\to$ Neural Vocoder.
   - *Limitation*: Requires massive parallel speech-to-speech recordings.
2. **Cascaded / Hybrid Modular Translation (Speech-to-Text $\to$ Pretrained Neural TTS)**:
   - **Frontend**: Hindi speech waveforms converted to Log-Mel Spectrogram features (80-channel filterbanks).
   - **Temporal Subsampling Adapter**: 2D/1D strided convolutions reducing acoustic frame rate by 4x to align with linguistic token distributions.
   - **Core Transformer**: Custom Seq2Seq Transformer encoding Hindi acoustic representations and autoregressively decoding Spanish text tokens.
   - **Backend TTS**: Pretrained Spanish neural vocoder (Meta MMS-TTS `facebook/mms-tts-spa` based on VITS) generating natural Spanish speech waveforms.

### Dataset Availability Analysis for Hindi $\to$ Spanish
- **Human Parallel Audio Reality**: There are **no public, human-recorded parallel Hindi-to-Spanish speech-to-speech datasets** (existing benchmarks like CVSS / CoVoST-2 cover English-centric pairs: `hi -> en` and `en -> es`).
- **Valid Strategy**:
  1. *Two-stage Cascade*: Hindi Speech $\to$ Hindi ASR (or Common Voice Hindi audio) $\to$ Custom Transformer (Hindi text $\to$ Spanish text) $\to$ Pretrained Spanish MMS-TTS.
  2. *Synthetic Paired Audio*: Common Voice Hindi or Samanantar Hindi text spoken via TTS, paired with Spanish translations synthesized via Spanish MMS-TTS.

---

## 2. Directory Structure

```
dl_a1/
├── .gitignore                     # Excludes datasets, checkpoints, and venv
├── requirements.txt               # Pinned PyTorch, audio, and NLP dependencies
├── README.md                      # Architecture documentation & setup guide
├── configs/
│   └── default_config.yaml        # Model, training, audio, and device settings
├── data/                          # Kept out of Git
│   ├── raw/                       # Raw audio (.wav) and transcripts
│   └── processed/                 # Tokenizer vocabs and precomputed features
├── checkpoints/                   # Kept out of Git (model weights & optimizer states)
├── notebooks/
│   └── colab_runner.ipynb         # Interactive Google Colab notebook for GPU training
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py   # Native Log-Mel spectrogram extractor & CMVN
│   │   └── vocoder.py             # Spanish MMS-TTS wrapper & WAV exporter
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py           # Subword/Char tokenizer with special tokens
│   │   ├── dataset.py             # SpeechTranslationDataset for audio & text
│   │   └── collate.py             # Dynamic batch padding & causal mask generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py           # ScaledDotProductAttention & MultiHeadAttention
│   │   ├── positional_encoding.py # Sinusoidal & Learnable Positional Encodings
│   │   ├── feed_forward.py        # PositionwiseFeedForward network (ReLU / GELU)
│   │   ├── encoder.py             # TransformerEncoderLayer & TransformerEncoder
│   │   ├── decoder.py             # TransformerDecoderLayer & TransformerDecoder
│   │   └── transformer.py         # Seq2SeqTransformer & AudioFrontendAdapter
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lr_scheduler.py        # Noam & Cosine Warmup learning rate schedulers
│   │   ├── loss.py                # Label-smoothed cross-entropy loss
│   │   └── trainer.py             # Mixed-precision S2STTrainer with grad scaling
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── translator.py          # Greedy and Beam Search autoregressive decoders
│   │   └── pipeline.py            # End-to-end Hindi Audio -> Spanish Audio pipeline
│   └── utils/
│       ├── __init__.py
│       ├── device.py              # Device selection (CUDA/MPS/CPU) & AMP autocast
│       └── config.py              # YAML configuration loader & saver
└── tests/
    ├── __init__.py
    ├── test_attention.py          # Shape, normalization, and causal masking tests
    ├── test_positional_encoding.py# Sequence length and buffer tests
    ├── test_transformer.py        # Full Seq2Seq, audio frontend, and gradient tests
    └── test_audio_features.py     # Mel filterbank and CMVN normalization tests
```

---

## 3. Setup Instructions

### Local Development
```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests
pytest tests/ -v
```

### Google Colab GPU Workflow
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU** (or A100 if available).
3. Upload or clone this repository into Colab:
   ```bash
   git clone <YOUR_REPO_URL> dl_a1
   cd dl_a1
   pip install -r requirements.txt
   ```
4. Open and run [`notebooks/colab_runner.ipynb`](file:///Users/kartikrastogi/Desktop/untitled%20folder/dl_a1/notebooks/colab_runner.ipynb).

---

## 4. Model Architecture & Invariants Verified by Tests

The custom Transformer implementation satisfies strict theoretical properties verified in [`tests/`](file:///Users/kartikrastogi/Desktop/untitled%20folder/dl_a1/tests):
1. **Softmax Normalization**: $\sum_{j} \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)_{i, j} = 1.0$.
2. **Causal Mask Isolation**: For target autoregressive decoding, predictions at index $t$ are mathematically invariant to token alterations at indices $t' > t$.
3. **Gradient Flow**: End-to-end backpropagation updates query, key, value, feed-forward, and projection layers without vanishing or disconnected graphs.
4. **Audio Subsampling**: 80-dim log-mel features are downsampled by $4\times$ along the time dimension via strided 1D convolutions to match linguistic representation scale.

---

## 5. Next Steps

- [x] Phase 1: Architecture research, modular skeleton, and from-scratch Transformer implementation.
- [x] Phase 2: Comprehensive unit test suite for attention, shapes, and gradients.
- [ ] Phase 3: Tokenizer corpus training on Hindi-Spanish parallel data (e.g. Samanantar or IITB).
- [ ] Phase 4: Google Colab mixed-precision training run and checkpoint evaluation.
- [ ] Phase 5: BLEU / chrF evaluation and Spanish TTS audio synthesis benchmarking.
