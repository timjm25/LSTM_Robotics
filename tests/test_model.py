"""Unit tests for model architecture and dataset utilities."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from lstm_robotics.config import ModelConfig, TrainConfig
from lstm_robotics.model import (
    ConvLSTMCell,
    ConvLSTM,
    LightweightBackbone,
    VisuoMotorLSTM,
    build_model,
)
from lstm_robotics.dataset import SyntheticRobotDataset, _numpy_to_tensor
from lstm_robotics.train import RobotActionLoss, gripper_accuracy


# ── fixtures ──────────────────────────────────────────────────────────────────

def _small_cfg() -> ModelConfig:
    """Tiny config for fast CPU tests."""
    return ModelConfig(
        image_height=32,
        image_width=32,
        sequence_length=4,
        backbone_out_channels=16,
        convlstm_hidden_channels=[16, 8],
        action_hidden_dim=32,
    )


# ── ConvLSTMCell tests ────────────────────────────────────────────────────────

def test_convlstm_cell_output_shape():
    cell = ConvLSTMCell(in_channels=8, hidden_channels=16, kernel_size=3)
    x = torch.randn(2, 8, 14, 14)
    h, (h2, c2) = cell(x)
    assert h.shape == (2, 16, 14, 14)
    assert h2.shape == h.shape
    assert c2.shape == h.shape


def test_convlstm_cell_state_passed():
    cell = ConvLSTMCell(in_channels=4, hidden_channels=8)
    x = torch.randn(1, 4, 7, 7)
    h0 = torch.zeros(1, 8, 7, 7)
    c0 = torch.zeros(1, 8, 7, 7)
    h1, (h1b, c1) = cell(x, (h0, c0))
    assert not torch.equal(c1, c0), "cell state should update"


# ── ConvLSTM tests ────────────────────────────────────────────────────────────

def test_convlstm_sequence_output():
    net = ConvLSTM(in_channels=8, hidden_channels=[16, 8])
    x = torch.randn(2, 5, 8, 7, 7)  # (B, T, C, H, W)
    out, states = net(x)
    assert out.shape == (2, 5, 8, 7, 7)
    assert len(states) == 2


# ── Backbone tests ────────────────────────────────────────────────────────────

def test_lightweight_backbone_stride():
    bb = LightweightBackbone(in_channels=3, out_channels=16)
    x = torch.randn(1, 3, 64, 64)
    out = bb(x)
    assert out.shape == (1, 16, 4, 4)  # 64 / 16 = 4


# ── Full model tests ──────────────────────────────────────────────────────────

def test_model_forward_shape():
    cfg = _small_cfg()
    model = build_model(cfg)
    x = torch.randn(2, cfg.sequence_length, 3, cfg.image_height, cfg.image_width)
    actions, states = model(x)
    assert actions.shape == (2, cfg.sequence_length, cfg.action_dim)
    assert len(states) == len(cfg.convlstm_hidden_channels)


def test_model_predict_one():
    cfg = _small_cfg()
    model = build_model(cfg)
    frame = torch.randn(3, cfg.image_height, cfg.image_width)  # (C, H, W)
    action, states = model.predict_one(frame)
    assert action.shape == (cfg.action_dim,)


def test_model_streaming_states():
    """States from t=0 should propagate to t=1 call."""
    cfg = _small_cfg()
    model = build_model(cfg)
    f1 = torch.randn(3, cfg.image_height, cfg.image_width)  # (C, H, W)
    f2 = torch.randn(3, cfg.image_height, cfg.image_width)
    _, states1 = model.predict_one(f1, states=None)
    _, states2 = model.predict_one(f2, states=states1)
    # States should have changed
    for (h1, c1), (h2, c2) in zip(states1, states2):
        assert not torch.equal(h1, h2)


def test_model_parameter_count():
    model = build_model(_small_cfg())
    assert model.num_parameters > 0
    print(f"  Small model: {model.num_parameters:,} parameters")


def test_default_model_parameter_count():
    model = build_model(ModelConfig())
    print(f"  Default model: {model.num_parameters:,} parameters")
    # Sanity: should be > 1M but < 100M
    assert 1_000_000 < model.num_parameters < 100_000_000


# ── Dataset tests ─────────────────────────────────────────────────────────────

def test_synthetic_dataset_length():
    cfg = _small_cfg()
    ds = SyntheticRobotDataset(n_episodes=10, episode_len=20, cfg=cfg, seed=0)
    expected = 10 * (20 - cfg.sequence_length + 1)
    assert len(ds) == expected


def test_synthetic_dataset_item_shapes():
    cfg = _small_cfg()
    ds = SyntheticRobotDataset(n_episodes=5, episode_len=15, cfg=cfg, seed=1)
    item = ds[0]
    assert item["frames"].shape == (cfg.sequence_length, 3, cfg.image_height, cfg.image_width)
    assert item["actions"].shape == (cfg.sequence_length, cfg.action_dim)


def test_synthetic_dataset_action_range():
    cfg = _small_cfg()
    ds = SyntheticRobotDataset(n_episodes=5, episode_len=15, cfg=cfg, seed=2)
    for i in range(min(20, len(ds))):
        item = ds[i]
        assert (item["actions"][:, :-1].abs() <= 1.0).all(), "motion out of [-1,1]"
        assert (item["actions"][:, -1] >= 0).all()
        assert (item["actions"][:, -1] <= 1).all()


def test_numpy_to_tensor_shape():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t = _numpy_to_tensor(img, 32, 32)
    assert t.shape == (3, 32, 32)
    assert t.dtype == torch.float32


# ── Loss tests ────────────────────────────────────────────────────────────────

def test_robot_action_loss():
    loss_fn = RobotActionLoss()
    pred = torch.rand(2, 4, 7)
    pred[..., -1] = torch.sigmoid(pred[..., -1])  # gripper to [0,1]
    pred[..., :-1] = torch.tanh(pred[..., :-1])   # motion to [-1,1]
    target = torch.rand(2, 4, 7)
    target[..., -1] = (target[..., -1] > 0.5).float()
    target[..., :-1] = target[..., :-1] * 2 - 1
    losses = loss_fn(pred, target)
    assert "loss" in losses
    assert losses["loss"] > 0
    assert losses["motion_loss"] >= 0
    assert losses["gripper_loss"] >= 0


def test_gripper_accuracy_perfect():
    pred = torch.zeros(2, 4, 7)
    pred[..., -1] = 0.9  # predict close
    target = torch.zeros(2, 4, 7)
    target[..., -1] = 1.0  # true = close
    acc = gripper_accuracy(pred, target)
    assert acc == 1.0


def test_gripper_accuracy_all_wrong():
    pred = torch.zeros(2, 4, 7)
    pred[..., -1] = 0.1   # predict open
    target = torch.zeros(2, 4, 7)
    target[..., -1] = 1.0  # true = close
    acc = gripper_accuracy(pred, target)
    assert acc == 0.0


# ── Gradient flow test ────────────────────────────────────────────────────────

def test_gradients_flow():
    cfg = _small_cfg()
    model = build_model(cfg)
    model.train()
    x = torch.randn(1, cfg.sequence_length, 3, cfg.image_height, cfg.image_width)
    target = torch.zeros(1, cfg.sequence_length, cfg.action_dim)
    target[..., -1] = 0.5

    loss_fn = RobotActionLoss()
    pred, _ = model(x)
    loss = loss_fn(pred, target)["loss"]
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No gradients computed"
    assert all(not np.isnan(g) for g in grad_norms), "NaN gradients detected"
    assert any(g > 0 for g in grad_norms), "All-zero gradients"


# ── Checkpoint round-trip test ────────────────────────────────────────────────

def test_checkpoint_save_load(tmp_path):
    import torch
    from lstm_robotics.model import load_checkpoint

    cfg = _small_cfg()
    model = build_model(cfg)
    path = str(tmp_path / "test.pt")
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 0.5,
        "model_config": cfg,
        "history": {},
    }, path)

    loaded_model, ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 1
    # Outputs should match (eval mode: dropout off; load_checkpoint returns an eval model)
    model.eval()
    assert not loaded_model.training
    x = torch.randn(1, cfg.sequence_length, 3, cfg.image_height, cfg.image_width)
    with torch.no_grad():
        out_orig, _ = model(x)
        out_load, _ = loaded_model(x)
    assert torch.allclose(out_orig, out_load)


if __name__ == "__main__":
    tests = [
        test_convlstm_cell_output_shape,
        test_convlstm_cell_state_passed,
        test_convlstm_sequence_output,
        test_lightweight_backbone_stride,
        test_model_forward_shape,
        test_model_predict_one,
        test_model_streaming_states,
        test_model_parameter_count,
        test_default_model_parameter_count,
        test_synthetic_dataset_length,
        test_synthetic_dataset_item_shapes,
        test_synthetic_dataset_action_range,
        test_numpy_to_tensor_shape,
        test_robot_action_loss,
        test_gripper_accuracy_perfect,
        test_gripper_accuracy_all_wrong,
        test_gradients_flow,
    ]
    import tempfile, pathlib
    for fn in tests:
        name = fn.__name__
        try:
            if "checkpoint" in name:
                fn(tmp_path=pathlib.Path(tempfile.mkdtemp()))
            else:
                fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            raise
    print(f"\nAll {len(tests)} tests passed.")
