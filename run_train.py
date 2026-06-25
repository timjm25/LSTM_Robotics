"""
Entry point for training the VisuoMotor LSTM model.

Quick start (synthetic data):
    python run_train.py

With real data directory:
    python run_train.py --data_dir /path/to/episodes --epochs 100

With pretrained MobileNetV2 backbone:
    python run_train.py --backbone mobilenetv2 --epochs 50

Evaluate a trained checkpoint:
    python run_train.py --eval --checkpoint checkpoints/best.pt
"""
import argparse
import os
import random

import numpy as np
import torch

from lstm_robotics.config import ModelConfig, TrainConfig
from lstm_robotics.dataset import make_dataloaders
from lstm_robotics.model import build_model
from lstm_robotics.train import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train VisuoMotor ConvLSTM")
    # Data
    parser.add_argument("--data_dir", default="data/demo")
    # Model
    parser.add_argument("--backbone", default="lightweight",
                        choices=["lightweight", "mobilenetv2"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--action_dim", type=int, default=7)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint", help="Checkpoint to load for eval")
    args = parser.parse_args()

    set_seed(args.seed)

    model_cfg = ModelConfig(
        backbone=args.backbone,
        image_height=args.image_size,
        image_width=args.image_size,
        sequence_length=args.seq_len,
        action_dim=args.action_dim,
    )
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    if args.eval:
        if not args.checkpoint:
            parser.error("--checkpoint required with --eval")
        from lstm_robotics.evaluate import run_evaluation
        run_evaluation(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            device_str="cpu",
        )
        return

    print("=" * 60)
    print("  VisuoMotor ConvLSTM  —  Robot Vision Training")
    print("=" * 60)
    print(f"  Backbone     : {model_cfg.backbone}")
    print(f"  Image size   : {model_cfg.image_height}×{model_cfg.image_width}")
    print(f"  Sequence len : {model_cfg.sequence_length}")
    print(f"  Action dim   : {model_cfg.action_dim}")
    print(f"  Data dir     : {train_cfg.data_dir}")
    print("=" * 60)

    train_loader, val_loader, test_loader = make_dataloaders(train_cfg, model_cfg)
    model = build_model(model_cfg)
    trainer = Trainer(model, train_loader, val_loader, train_cfg, model_cfg)
    trainer.run()

    print("\nRunning final test evaluation...")
    from lstm_robotics.evaluate import evaluate, print_metrics
    device = trainer.device
    metrics = evaluate(model, test_loader, device)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
