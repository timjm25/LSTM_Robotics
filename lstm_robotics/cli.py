"""
Command line: python -m lstm_robotics <command>

    template      write an example trajectory log (CSV + frames) to copy for your data
    validate      check a trajectory log CSV against the schema, print a summary
    train         train on a CSV log, an episode dir / .h5, or --demo synthetic data
    evaluate      metrics (+ plot) for a checkpoint on a dataset's test split
    run           drive the configured robot (simulated by default) with a checkpoint
    check-robot   connect to the configured robot and grab one frame (sends no motion)

`python run_train.py ...` is kept as an alias of `python -m lstm_robotics train ...`.
"""
import argparse
import random
import sys
from typing import List, Optional

import numpy as np


def _set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _floats(text: Optional[str]) -> Optional[List[float]]:
    if not text:
        return None
    try:
        return [float(v) for v in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_template(args) -> int:
    from .logs import write_template
    path = write_template(args.out_dir, n_episodes=args.episodes, episode_len=args.length,
                          size=args.size)
    print(f"wrote SYNTHETIC example log: {path}")
    print(f"replace rows/frames with your robot's data, then: "
          f"python -m lstm_robotics validate {path}")
    return 0


def cmd_validate(args) -> int:
    from .logs import load_trajectory_csv
    log = load_trajectory_csv(args.csv, action_scale=_floats(args.action_scale),
                              min_length=args.seq_len)
    lengths = [len(e) for e in log.episodes]
    print(f"OK: {len(log.episodes)} episode(s), {log.n_steps} steps "
          f"(min/max length {min(lengths)}/{max(lengths)}), action mode '{log.action_mode}'")
    print(f"  action_scale (dx..dyaw): {[round(float(v), 6) for v in log.action_scale]}")
    if log.sensor_names:
        print(f"  sensor channels: {log.sensor_names}")
    for w in log.warnings:
        print(f"  warning: {w}")
    return 0


def cmd_train(args) -> int:
    if args.eval:
        if not args.checkpoint:
            raise SystemExit("error: --checkpoint required with --eval")
        args.no_plot = False
        return cmd_evaluate(args)

    from .config import ModelConfig, TrainConfig
    from .dataset import dataset_action_scale, make_dataloaders
    from .evaluate import evaluate, print_metrics
    from .model import build_model
    from .train import Trainer

    _set_seed(args.seed)
    model_cfg = ModelConfig(
        backbone=args.backbone, image_height=args.image_size, image_width=args.image_size,
        sequence_length=args.seq_len, action_dim=args.action_dim,
    )
    if args.small:
        model_cfg.backbone_out_channels = 32
        model_cfg.convlstm_hidden_channels = [32, 16]
        model_cfg.action_hidden_dim = 64
    train_cfg = TrainConfig(
        data_dir=args.data_dir, data_csv=args.data_csv or "", use_synthetic=args.demo,
        synthetic_episodes=args.demo_episodes, action_scale=_floats(args.action_scale),
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        device=args.device, checkpoint_dir=args.checkpoint_dir, seed=args.seed,
        num_workers=args.num_workers, warmup_epochs=min(3, max(1, args.epochs // 3)),
    )
    print("=" * 60)
    print("  VisuoMotor ConvLSTM  —  Robot Vision Training")
    print("=" * 60)
    source = (train_cfg.data_csv or ("synthetic demo" if args.demo and not
                                      _has_data(train_cfg.data_dir) else train_cfg.data_dir))
    print(f"  Data         : {source}")
    print(f"  Backbone     : {model_cfg.backbone}   Image: {model_cfg.image_height}px   "
          f"Seq: {model_cfg.sequence_length}")
    print("=" * 60)

    train_loader, val_loader, test_loader = make_dataloaders(train_cfg, model_cfg)
    scale = dataset_action_scale(train_loader)
    metadata = {"data_source": str(source), "action_scale": scale}
    model = build_model(model_cfg)
    trainer = Trainer(model, train_loader, val_loader, train_cfg, model_cfg, metadata=metadata)
    trainer.run()

    print("\nRunning final test evaluation...")
    metrics = evaluate(model, test_loader, trainer.device, action_scale=scale)
    print_metrics(metrics)
    return 0


def _has_data(d: str) -> bool:
    from pathlib import Path
    p = Path(d)
    return p.exists() and (p.is_file() or any(p.iterdir()))


def cmd_evaluate(args) -> int:
    from .evaluate import run_evaluation
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir or "",
        data_csv=getattr(args, "data_csv", "") or "",
        use_synthetic=getattr(args, "demo", False),
        output_dir=getattr(args, "output_dir", "eval_output"),
        device_str="cpu" if getattr(args, "device", "cpu") == "auto" else args.device,
        plot=not getattr(args, "no_plot", False),
        num_workers=getattr(args, "num_workers", 0),
    )
    return 0


def cmd_run(args) -> int:
    from .predict import Predictor
    from .robot import ActionLimiter, max_delta_from_env, robot_from_env, run_policy
    import os

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    meta = predictor.metadata
    scale = _floats(args.action_scale) or meta.get("action_scale")
    if scale is None:
        print("warning: checkpoint has no action_scale (trained on non-CSV data); "
              "sending normalised actions x 1.0")
        scale = [1.0] * 6
    limiter = ActionLimiter(_floats(args.max_delta) or max_delta_from_env(os.environ))
    robot = robot_from_env(image_size=predictor.cfg.image_height)
    print(f"robot: {type(robot).__name__}   steps: {args.steps}   "
          f"max |delta| per step: {limiter.max_delta.tolist()}")
    sent = run_policy(predictor, robot, args.steps, scale, limiter=limiter,
                      log_csv=args.log, save_frames_dir=args.save_frames)
    print(f"sent {len(sent)} actions ({limiter.n_clipped} clipped by the limiter)")
    if args.log:
        print(f"run log: {args.log}")
    return 0


def cmd_check_robot(args) -> int:
    from .robot import robot_from_env
    robot = robot_from_env()
    robot.connect()
    try:
        frame = robot.get_frame()
        name = getattr(robot, "name", "")
        print(f"OK: {type(robot).__name__} {name} frame {frame.shape}")
    finally:
        robot.close()
    return 0


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m lstm_robotics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="write an example trajectory log")
    t.add_argument("out_dir")
    t.add_argument("--episodes", type=int, default=3)
    t.add_argument("--length", type=int, default=12)
    t.add_argument("--size", type=int, default=32)
    t.set_defaults(func=cmd_template)

    v = sub.add_parser("validate", help="check a trajectory log CSV")
    v.add_argument("csv")
    v.add_argument("--seq_len", type=int, default=8)
    v.add_argument("--action_scale", help="6 comma-separated values (default: auto)")
    v.set_defaults(func=cmd_validate)

    def data_args(p):
        p.add_argument("--data_csv", default="", help="trajectory log CSV (see template)")
        p.add_argument("--data_dir", default="data/demo", help="episode dir or folder with .h5")
        p.add_argument("--demo", action="store_true",
                       help="use synthetic demo data when no real data is given")
        p.add_argument("--num_workers", type=int, default=0)
        p.add_argument("--device", default="auto")

    tr = sub.add_parser("train", help="train a policy (also: python run_train.py)")
    data_args(tr)
    tr.add_argument("--backbone", default="lightweight", choices=["lightweight", "mobilenetv2"])
    tr.add_argument("--seq_len", type=int, default=8)
    tr.add_argument("--image_size", type=int, default=224, help="multiple of 16")
    tr.add_argument("--action_dim", type=int, default=7)
    tr.add_argument("--small", action="store_true", help="narrow model for quick CPU runs")
    tr.add_argument("--action_scale", help="6 values dx..dyaw for CSV logs (default: auto)")
    tr.add_argument("--demo_episodes", type=int, default=200)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--checkpoint_dir", default="checkpoints")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--eval", action="store_true", help="evaluate --checkpoint instead")
    tr.add_argument("--checkpoint", help="checkpoint for --eval")
    tr.add_argument("--output_dir", default="eval_output")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="evaluate a checkpoint")
    data_args(ev)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--output_dir", default="eval_output")
    ev.add_argument("--no_plot", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("run", help="run a checkpoint on the configured robot (sim default)")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--device", default="cpu")
    r.add_argument("--action_scale", help="override the checkpoint's action scale")
    r.add_argument("--max_delta", help="override LSTM_ROBOT_MAX_DELTA (1 or 6 values)")
    r.add_argument("--log", help="write the run as a trajectory log CSV")
    r.add_argument("--save_frames", help="folder for the run's camera frames")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check-robot", help="connect + grab one frame (no motion)")
    c.set_defaults(func=cmd_check_robot)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    from .logs import LogValidationError
    from .robot import RobotError, RobotUnavailableError

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except LogValidationError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 2
    except RobotUnavailableError as exc:
        print(f"robot unavailable: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError, RobotError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
