"""Training loop with early stopping, cosine scheduling, and checkpoint management."""
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .model import VisuoMotorLSTM, build_model


# ── Loss ──────────────────────────────────────────────────────────────────────

class RobotActionLoss(nn.Module):
    """
    Compound loss for robot action prediction.

    Motion  (dims 0-5): MSE — continuous end-effector delta
    Gripper (dim   6):  BCE — binary open/close command
    """

    def __init__(self, action_w: float = 1.0, gripper_w: float = 2.0):
        super().__init__()
        self.action_w = action_w
        self.gripper_w = gripper_w
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        pred: torch.Tensor,   # (B, T, action_dim)
        target: torch.Tensor, # (B, T, action_dim)
    ) -> Dict[str, torch.Tensor]:
        motion_pred = pred[..., :-1]
        motion_tgt = target[..., :-1]
        gripper_pred = pred[..., -1]
        gripper_tgt = target[..., -1]

        motion_loss = self.mse(motion_pred, motion_tgt)
        gripper_loss = self.bce(gripper_pred, gripper_tgt)
        total = self.action_w * motion_loss + self.gripper_w * gripper_loss

        return {
            "loss": total,
            "motion_loss": motion_loss.detach(),
            "gripper_loss": gripper_loss.detach(),
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def gripper_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Binary accuracy for gripper command."""
    pred_bin = (pred[..., -1] > 0.5).float()
    tgt_bin = (target[..., -1] > 0.5).float()
    return (pred_bin == tgt_bin).float().mean().item()


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: VisuoMotorLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
    ):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

        self.device = self._resolve_device(train_cfg.device)
        self.model = model.to(self.device)

        self.loss_fn = RobotActionLoss(
            action_w=train_cfg.action_loss_weight,
            gripper_w=train_cfg.gripper_loss_weight,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.ckpt_dir = Path(train_cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_gripper_acc": [], "val_gripper_acc": [],
        }
        self.best_val_loss = float("inf")
        self.best_checkpoints: list = []

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _build_scheduler(self):
        cfg = self.train_cfg
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 1e-2
            )
        if cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
            )
        return None

    def _warmup_lr(self, epoch: int):
        if epoch < self.train_cfg.warmup_epochs:
            scale = (epoch + 1) / self.train_cfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.train_cfg.learning_rate * scale

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = total_motion = total_gripper = total_acc = 0.0
        n = 0

        for batch in self.train_loader:
            frames = batch["frames"].to(self.device, non_blocking=True)
            actions = batch["actions"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            pred, _ = self.model(frames)
            losses = self.loss_fn(pred, actions)
            losses["loss"].backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_cfg.grad_clip
            )
            self.optimizer.step()

            b = frames.size(0)
            total_loss += losses["loss"].item() * b
            total_motion += losses["motion_loss"].item() * b
            total_gripper += losses["gripper_loss"].item() * b
            total_acc += gripper_accuracy(pred.detach(), actions) * b
            n += b

        return {
            "loss": total_loss / n,
            "motion_loss": total_motion / n,
            "gripper_loss": total_gripper / n,
            "gripper_acc": total_acc / n,
        }

    @torch.no_grad()
    def eval_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_acc = 0.0
        n = 0

        for batch in self.val_loader:
            frames = batch["frames"].to(self.device, non_blocking=True)
            actions = batch["actions"].to(self.device, non_blocking=True)
            pred, _ = self.model(frames)
            losses = self.loss_fn(pred, actions)

            b = frames.size(0)
            total_loss += losses["loss"].item() * b
            total_acc += gripper_accuracy(pred, actions) * b
            n += b

        return {"loss": total_loss / n, "gripper_acc": total_acc / n}

    def save_checkpoint(self, epoch: int, val_loss: float):
        path = self.ckpt_dir / f"epoch_{epoch:03d}_val{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "model_config": self.model_cfg,
                "history": self.history,
            },
            path,
        )
        self.best_checkpoints.append((val_loss, path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        # Prune old checkpoints
        while len(self.best_checkpoints) > self.train_cfg.keep_best:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

    def run(self) -> Dict[str, list]:
        cfg = self.train_cfg
        print(f"Training on {self.device} — {self.model.num_parameters:,} parameters")
        print(f"  Epochs: {cfg.epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.learning_rate}")
        print(f"  Train batches: {len(self.train_loader)}  |  Val batches: {len(self.val_loader)}")

        for epoch in range(cfg.epochs):
            t0 = time.time()
            self._warmup_lr(epoch)

            train_m = self.train_epoch()
            val_m = self.eval_epoch()

            if self.scheduler and epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_gripper_acc"].append(train_m["gripper_acc"])
            self.history["val_gripper_acc"].append(val_m["gripper_acc"])

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:3d}/{cfg.epochs}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"gripper_acc={val_m['gripper_acc']:.3f}  "
                f"lr={lr:.2e}  ({elapsed:.1f}s)"
            )

            is_best = val_m["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m["loss"]
                self.save_checkpoint(epoch + 1, val_m["loss"])
                print(f"  → New best checkpoint saved (val_loss={self.best_val_loss:.4f})")
            elif (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(epoch + 1, val_m["loss"])

        # Save final history
        hist_path = self.ckpt_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete. History saved to {hist_path}")
        return self.history
