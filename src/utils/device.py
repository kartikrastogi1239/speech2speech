"""Device selection and mixed-precision acceleration utilities."""

from contextlib import nullcontext
from typing import Optional, Union
import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Select the best available compute device or use the preferred option.

    Args:
        preferred: One of "auto", "cuda", "mps", or "cpu".

    Returns:
        torch.device: The selected compute device.
    """
    preferred = preferred.lower().strip()
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Warning] CUDA requested but not available. Falling back to auto.")
    elif preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[Warning] MPS requested but not available. Falling back to auto.")
    elif preferred == "cpu":
        return torch.device("cpu")

    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: str = "float16",
):
    """Return appropriate torch.amp.autocast context manager based on device.

    Args:
        device: Target torch device.
        use_amp: Whether to enable automatic mixed precision.
        amp_dtype: 'float16' or 'bfloat16'.

    Returns:
        A context manager for mixed precision or nullcontext.
    """
    if not use_amp or device.type == "cpu":
        # CPU amp is supported in newer PyTorch but often not desired for standard training
        return nullcontext()

    target_dtype = torch.bfloat16 if amp_dtype.lower() == "bfloat16" else torch.float16

    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=target_dtype)
    if device.type == "mps":
        # PyTorch MPS supports float16 autocast in recent versions
        if hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast(device_type="mps", dtype=torch.float16)
            except Exception:
                return nullcontext()
        return nullcontext()

    return nullcontext()


class GradScalerManager:
    """Wrapper around PyTorch GradScaler for mixed-precision training stability."""

    def __init__(self, enabled: bool = True, device: Optional[torch.device] = None):
        self.device = device or get_device()
        # GradScaler is primarily needed for CUDA float16; bfloat16 doesn't strictly need scaling
        self.enabled = enabled and (self.device.type == "cuda")
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss value if gradient scaling is enabled."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer through scaler or directly."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
