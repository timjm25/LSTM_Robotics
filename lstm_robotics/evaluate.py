"""
Evaluation and visualisation utilities.

Computes per-axis metrics and generates prediction plots.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import VisuoMotorLSTM, load_checkpoint

ACTION_NAMES = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]


@torch.no_grad()
def evaluate(
    model: VisuoMotorLSTM,
    loader: DataLoader,
    device: torch.device,
    action_scale: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute MAE and RMSE per action dimension.

    With ``action_scale`` (CSV logs), also reports motion MAE in physical units
    (``mae_phys_<axis>``), i.e. the same units as the imported log.
    """
    model.eval()
    preds_list: List[np.ndarray] = []
    tgts_list: List[np.ndarray] = []

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"]
        pred, _ = model(frames)
        preds_list.append(pred.cpu().numpy())
        tgts_list.append(actions.numpy())

    preds = np.concatenate(preds_list, axis=0)  # (N, T, 7)
    tgts = np.concatenate(tgts_list, axis=0)
    flat_pred = preds.reshape(-1, preds.shape[-1])
    flat_tgt = tgts.reshape(-1, tgts.shape[-1])

    mae = np.abs(flat_pred - flat_tgt).mean(axis=0)   # (7,)
    rmse = np.sqrt(((flat_pred - flat_tgt) ** 2).mean(axis=0))
    gripper_acc = (
        (flat_pred[:, -1] > 0.5) == (flat_tgt[:, -1] > 0.5)
    ).mean()

    metrics: Dict[str, float] = {}
    for i, name in enumerate(ACTION_NAMES):
        metrics[f"mae_{name}"] = float(mae[i])
        metrics[f"rmse_{name}"] = float(rmse[i])
    if action_scale is not None:
        scale = np.asarray(action_scale, dtype=np.float64)
        for i, name in enumerate(ACTION_NAMES[:-1]):
            metrics[f"mae_phys_{name}"] = float(mae[i] * scale[i])
    metrics["overall_mae"] = float(mae.mean())
    metrics["overall_rmse"] = float(rmse.mean())
    metrics["gripper_accuracy"] = float(gripper_acc)
    return metrics


def print_metrics(metrics: Dict[str, float]):
    print("\n── Evaluation Results ─────────────────────────────────")
    print(f"  Overall MAE  : {metrics['overall_mae']:.4f}")
    print(f"  Overall RMSE : {metrics['overall_rmse']:.4f}")
    print(f"  Gripper Acc  : {metrics['gripper_accuracy']:.4f}")
    print("\n  Per-axis MAE:")
    for name in ACTION_NAMES:
        print(f"    {name:10s}: {metrics[f'mae_{name}']:.4f}")
    print("────────────────────────────────────────────────────────\n")


def plot_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    out_path: Optional[str] = None,
    n_samples: int = 4,
):
    """
    Plot predicted vs actual action trajectories.
    Requires matplotlib — gracefully skips if not installed.
    """
    try:
        import matplotlib
        if out_path:
            matplotlib.use("Agg")  # headless: write the file, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print("[evaluate] matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(
        n_samples, len(ACTION_NAMES),
        figsize=(3 * len(ACTION_NAMES), 2.5 * n_samples),
    )
    if n_samples == 1:
        axes = [axes]

    for row, (pred, tgt) in enumerate(zip(preds[:n_samples], targets[:n_samples])):
        for col, name in enumerate(ACTION_NAMES):
            ax = axes[row][col]
            ax.plot(tgt[:, col], label="target", color="steelblue")
            ax.plot(pred[:, col], label="predicted", color="tomato", linestyle="--")
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Seq {row}", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.suptitle("Predicted vs Actual Robot Actions", y=1.01, fontsize=12)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=120)
        print(f"[evaluate] Plot saved to {out_path}")
    else:
        plt.show()
    plt.close()


def run_evaluation(
    checkpoint_path: str,
    data_dir: str = "",
    output_dir: str = "eval_output",
    device_str: str = "cpu",
    data_csv: str = "",
    use_synthetic: bool = False,
    plot: bool = True,
    num_workers: int = 0,
):
    """
    Full evaluation pipeline: load checkpoint → evaluate on the test split → save
    metrics.json (+ predictions.png). CSV logs are re-normalised with the action_scale
    stored in the checkpoint so metrics are comparable with training.
    """
    from .config import TrainConfig
    from .dataset import make_dataloaders

    device = torch.device(device_str)
    model, ckpt = load_checkpoint(checkpoint_path, device_str)
    model = model.to(device)

    model_cfg = ckpt["model_config"]
    meta = ckpt.get("metadata", {}) or {}
    action_scale = meta.get("action_scale")
    train_cfg = TrainConfig(
        data_dir=data_dir or "data/demo", data_csv=data_csv, use_synthetic=use_synthetic,
        action_scale=action_scale if (data_csv or data_dir.endswith(".csv")) else None,
        num_workers=num_workers,
    )
    _, _, test_loader = make_dataloaders(train_cfg, model_cfg)

    metrics = evaluate(model, test_loader, device, action_scale=action_scale)
    print_metrics(metrics)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if plot:
        preds_list, tgts_list = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                frames = batch["frames"].to(device)
                pred, _ = model(frames)
                preds_list.append(pred.cpu().numpy())
                tgts_list.append(batch["actions"].numpy())
                if len(preds_list) >= 4:
                    break
        preds_np = np.concatenate(preds_list)
        tgts_np = np.concatenate(tgts_list)
        plot_predictions(preds_np, tgts_np, out_path=str(out / "predictions.png"),
                         n_samples=min(4, len(preds_np)))

    return metrics
