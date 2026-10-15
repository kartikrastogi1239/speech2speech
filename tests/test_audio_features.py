"""Unit tests for MelSpectrogramExtractor and CMVN."""

import torch
from src.audio.feature_extractor import MelSpectrogramExtractor, compute_cmvn


def test_mel_spectrogram_extractor_shapes():
    """Verify log-mel feature extraction output shapes."""
    sample_rate = 16000
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    batch_size = 2
    duration_secs = 1.0
    num_samples = int(sample_rate * duration_secs)

    extractor = MelSpectrogramExtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    waveform = torch.randn(batch_size, num_samples)
    features = extractor(waveform)

    # Expected time frames: floor(num_samples / hop_length) + 1
    expected_frames = num_samples // hop_length + 1
    assert features.dim() == 3
    assert features.shape[0] == batch_size
    assert features.shape[1] == expected_frames
    assert features.shape[2] == n_mels


def test_compute_cmvn_normalization():
    """Verify CMVN produces approximately zero mean and unit variance."""
    batch_size = 4
    time_frames = 100
    dim = 80

    features = torch.randn(batch_size, time_frames, dim) * 5.0 + 10.0
    normalized = compute_cmvn(features)

    # Mean along time axis should be close to 0
    means = normalized.mean(dim=1)
    stds = normalized.std(dim=1)

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-3)
