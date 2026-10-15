"""Audio preprocessing, Log-Mel spectrogram extraction, and normalization."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class MelSpectrogramExtractor(nn.Module):
    """PyTorch-native Log-Mel Spectrogram extractor.

    Converts raw 1D audio waveforms into 2D time-frequency log-mel representations.
    Compatible with CPU, CUDA, and Apple MPS.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.eps = eps

        # Create Hann window buffer
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        # Build triangular Mel filterbank matrix of shape (n_fft // 2 + 1, n_mels)
        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_filterbank", mel_fb)

    def _hz_to_mel(self, hz: float) -> float:
        """Convert frequency in Hertz to Mel scale."""
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        """Convert frequency in Mel scale to Hertz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Compute triangular Mel filterbank matrix."""
        num_freq_bins = self.n_fft // 2 + 1
        min_mel = self._hz_to_mel(self.f_min)
        max_mel = self._hz_to_mel(self.f_max)

        mel_points = torch.linspace(min_mel, max_mel, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()

        filterbank = torch.zeros(num_freq_bins, self.n_mels)
        for i in range(self.n_mels):
            left = bin_points[i].item()
            center = bin_points[i + 1].item()
            right = bin_points[i + 2].item()

            for f in range(left, center):
                if center > left:
                    filterbank[f, i] = (f - left) / (center - left)
            for f in range(center, right):
                if right > center:
                    filterbank[f, i] = (right - f) / (right - center)

        return filterbank

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram features from audio waveform.

        Args:
            waveform: Audio tensor of shape (batch_size, num_samples) or (num_samples,)

        Returns:
            Log-mel spectrogram of shape (batch_size, time_frames, n_mels)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # STFT computation: (batch_size, freq_bins, time_frames, 2) or complex tensor
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # Magnitude spectrogram: (batch_size, freq_bins, time_frames)
        magnitude = torch.abs(stft)

        # Mel spectrogram: (batch_size, time_frames, n_mels)
        # magnitude is (batch, freq, time) -> transpose to (batch, time, freq)
        magnitude = magnitude.transpose(1, 2)
        mel_spec = torch.matmul(magnitude, self.mel_filterbank)

        # Log compression with numerical stability
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=self.eps))
        return log_mel_spec


def compute_cmvn(
    features: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Apply Cepstral Mean and Variance Normalization (CMVN) along the time axis.

    Args:
        features: Feature tensor of shape (batch_size, time_frames, feature_dim)

    Returns:
        Normalized features with zero mean and unit variance.
    """
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    return (features - mean) / (std + eps)
