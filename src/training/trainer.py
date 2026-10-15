"""Trainer class supporting mixed-precision, gradient accumulation, and checkpointing."""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_device, get_autocast_context, GradScalerManager


class S2STTrainer:
    """Trainer for Speech-to-Speech / Speech-to-Text translation models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.config = config or {}
        system_cfg = self.config.get("system", {})
        training_cfg = self.config.get("training", {})

        # Device selection & configuration
        preferred_dev = system_cfg.get("device", "auto")
        self.device = get_device(preferred_dev) if device is None else torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler

        # Mixed precision setup
        self.use_amp = system_cfg.get("use_amp", True)
        self.amp_dtype = system_cfg.get("amp_dtype", "float16")
        self.scaler_manager = GradScalerManager(enabled=self.use_amp, device=self.device)

        # Training hyperparameters
        self.accumulate_grad_batches = training_cfg.get("accumulate_grad_batches", 1)
        self.clip_grad_norm = training_cfg.get("clip_grad_norm", 1.0)
        self.checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self, is_audio: bool = False) -> float:
        """Run one training epoch over train_dataloader."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move batch tensors to device
            tgt_input = batch["tgt_input"].to(self.device)
            tgt_label = batch["tgt_label"].to(self.device)
            tgt_causal_mask = batch.get("tgt_causal_mask")
            if tgt_causal_mask is not None:
                tgt_causal_mask = tgt_causal_mask.to(self.device)
            tgt_pad_mask = batch.get("tgt_pad_mask")
            if tgt_pad_mask is not None:
                tgt_pad_mask = tgt_pad_mask.to(self.device)

            if is_audio:
                src = batch["audio_features"].to(self.device)
                src_mask = None
            else:
                src = batch["src_tokens"].to(self.device)
                src_mask = batch.get("src_pad_mask")
                if src_mask is not None:
                    src_mask = src_mask.to(self.device)

            # Combined target mask
            tgt_mask = tgt_causal_mask
            if tgt_pad_mask is not None and tgt_causal_mask is not None:
                tgt_mask = tgt_causal_mask | tgt_pad_mask

            # Forward pass with mixed precision
            with get_autocast_context(self.device, use_amp=self.use_amp, amp_dtype=self.amp_dtype):
                logits = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    is_audio=is_audio,
                )
                loss = self.criterion(logits, tgt_label)
                # Scale loss by accumulation steps
                loss_scaled = loss / self.accumulate_grad_batches

            # Backward pass with gradient scaling
            self.scaler_manager.scale(loss_scaled).backward()

            # Optimizer step every accumulate_grad_batches or end of epoch
            if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (batch_idx + 1) == num_batches:
                if self.clip_grad_norm > 0:
                    self.scaler_manager.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )

                self.scaler_manager.step(self.optimizer)
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.global_step += 1

            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, is_audio: bool = False) -> float:
        """Evaluate model on validation dataset."""
        if self.val_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            tgt_input = batch["tgt_input"].to(self.device)
            tgt_label = batch["tgt_label"].to(self.device)
            tgt_causal_mask = batch.get("tgt_causal_mask")
            if tgt_causal_mask is not None:
                tgt_causal_mask = tgt_causal_mask.to(self.device)

            if is_audio:
                src = batch["audio_features"].to(self.device)
                src_mask = None
            else:
                src = batch["src_tokens"].to(self.device)
                src_mask = batch.get("src_pad_mask")
                if src_mask is not None:
                    src_mask = src_mask.to(self.device)

            with get_autocast_context(self.device, use_amp=self.use_amp, amp_dtype=self.amp_dtype):
                logits = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_mask=src_mask,
                    tgt_mask=tgt_causal_mask,
                    is_audio=is_audio,
                )
                loss = self.criterion(logits, tgt_label)

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, path: Union[str, Path], extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights, optimizer state, and training metadata."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "extra_meta": extra_meta or {},
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved model checkpoint to: {ckpt_path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model weights and optimizer state from checkpoint."""
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        print(f"[Checkpoint] Successfully loaded checkpoint from {ckpt_path}")
