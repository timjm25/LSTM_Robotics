"""End-to-end: CSV log -> train -> best.pt -> load -> evaluate -> run on robots (tiny, CPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from lstm_robotics.cli import main
from lstm_robotics.config import ModelConfig, TrainConfig
from lstm_robotics.dataset import CSVTrajectoryDataset, make_dataloaders
from lstm_robotics.logs import load_trajectory_csv, write_template
from lstm_robotics.model import load_checkpoint

TINY = ["--image_size", "32", "--seq_len", "4", "--small", "--epochs", "2",
        "--batch_size", "4", "--device", "cpu"]


def _cfg():
    return ModelConfig(image_height=32, image_width=32, sequence_length=4,
                       backbone_out_channels=16, convlstm_hidden_channels=[16, 8],
                       action_hidden_dim=32)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    csv_path = write_template(str(root / "logs"), n_episodes=4, episode_len=8, size=32)
    ckpt_dir = root / "ck"
    assert main(["train", "--data_csv", str(csv_path), "--checkpoint_dir", str(ckpt_dir),
                 *TINY]) == 0
    return root, csv_path, ckpt_dir


def test_csv_dataset_items_and_episode_split(tmp_path):
    csv_path = write_template(str(tmp_path), n_episodes=4, episode_len=6, size=16)
    ds = CSVTrajectoryDataset.from_csv(str(csv_path), _cfg())
    item = ds[0]
    assert item["frames"].shape == (4, 3, 32, 32)
    assert item["actions"].shape == (4, 7)
    assert item["sensors"].shape == (4, 2)
    tr, va, te = make_dataloaders(TrainConfig(data_csv=str(csv_path), num_workers=0,
                                              batch_size=2), _cfg())
    ids = [{e.episode_id for e in dl.dataset.episodes} for dl in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_no_data_without_demo_flag_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="--demo"):
        make_dataloaders(TrainConfig(data_dir=str(tmp_path / "none")), _cfg())


def test_demo_flag_uses_synthetic(tmp_path):
    cfg = TrainConfig(data_dir=str(tmp_path / "none"), use_synthetic=True, synthetic_episodes=3,
                      num_workers=0)
    tr, _, _ = make_dataloaders(cfg, _cfg())
    assert type(tr.dataset.dataset).__name__ == "SyntheticRobotDataset"


def test_train_writes_best_last_and_metadata(trained):
    _, csv_path, ckpt_dir = trained
    assert (ckpt_dir / "best.pt").exists() and (ckpt_dir / "last.pt").exists()
    model, ckpt = load_checkpoint(str(ckpt_dir / "best.pt"))
    assert not model.training
    scale = ckpt["metadata"]["action_scale"]
    expected = load_trajectory_csv(str(csv_path)).action_scale
    np.testing.assert_allclose(scale, expected, rtol=1e-6)
    assert isinstance(ckpt["model_config"], ModelConfig)
    hist = json.loads((ckpt_dir / "history.json").read_text())
    assert len(hist["train_loss"]) == 2


def test_evaluate_command(trained, tmp_path):
    _, csv_path, ckpt_dir = trained
    out = tmp_path / "ev"
    assert main(["evaluate", "--checkpoint", str(ckpt_dir / "best.pt"), "--data_csv",
                 str(csv_path), "--output_dir", str(out), "--no_plot"]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert "mae_phys_Δx" in metrics and "gripper_accuracy" in metrics


def test_run_on_simulated_robot_and_reimport(trained, tmp_path, monkeypatch):
    _, _, ckpt_dir = trained
    monkeypatch.delenv("LSTM_ROBOT_DRIVER", raising=False)
    log = tmp_path / "run" / "log.csv"
    assert main(["run", "--checkpoint", str(ckpt_dir / "best.pt"), "--steps", "5",
                 "--log", str(log), "--save_frames", str(tmp_path / "run" / "frames")]) == 0
    back = load_trajectory_csv(str(log))
    assert back.n_steps == 5


def test_legacy_checkpoint_layout_still_loads(tmp_path):
    from lstm_robotics.model import build_model
    cfg = _cfg()
    model = build_model(cfg)
    p = tmp_path / "legacy.pt"
    torch.save({"epoch": 3, "model_state_dict": model.state_dict(), "model_config": cfg}, p)
    loaded, ckpt = load_checkpoint(str(p))
    assert ckpt["metadata"] == {} and loaded.cfg.image_height == 32


def test_bad_csv_gives_clean_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("episode_id,timestep\nA,0\n")
    assert main(["validate", str(bad)]) == 2
    assert "missing column" in capsys.readouterr().err
