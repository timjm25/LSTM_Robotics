"""Tests for importing real trajectory / sensor logs from CSV."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from lstm_robotics.logs import (
    DELTA_COLUMNS,
    LogValidationError,
    denormalise_actions,
    load_trajectory_csv,
    write_template,
)

TEMPLATES = Path(__file__).parent.parent / "data" / "templates" / "trajectory_log"


def _rows(p):
    with open(p, newline="") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def _write(p, header, rows):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def log_csv(tmp_path):
    return write_template(str(tmp_path), n_episodes=2, episode_len=6, size=16)


def test_shipped_templates_validate():
    delta = load_trajectory_csv(str(TEMPLATES / "trajectory_log_template.csv"), min_length=4)
    pose = load_trajectory_csv(str(TEMPLATES / "trajectory_log_pose_example.csv"), min_length=4)
    assert delta.action_mode == "delta" and pose.action_mode == "pose"
    assert delta.sensor_names == ["sensor_force_z", "sensor_gripper_width_m"]
    assert all(e.episode_id.startswith("SYNTHETIC_SAMPLE") for e in delta.episodes)
    # The pose example integrates the delta template: same physical motion (except last step).
    for d_ep, p_ep in zip(delta.episodes, pose.episodes):
        d_phys = denormalise_actions(d_ep.actions, delta.action_scale)[:-1, :6]
        p_phys = denormalise_actions(p_ep.actions, pose.action_scale)[:-1, :6]
        np.testing.assert_allclose(d_phys, p_phys, atol=2e-5)


def test_actions_normalised_and_scale_inverts(log_csv):
    log = load_trajectory_csv(str(log_csv))
    acts = np.concatenate([e.actions for e in log.episodes])
    assert acts.shape[1] == 7 and np.abs(acts[:, :6]).max() == pytest.approx(1.0, abs=1e-6)
    header, rows = _rows(log_csv)
    raw = np.array([[float(r[c]) for c in DELTA_COLUMNS] for r in rows])
    back = denormalise_actions(acts, log.action_scale)[:, :6]
    np.testing.assert_allclose(back, raw, atol=1e-6)
    assert log.episodes[0].sensors.shape == (6, 2)


def test_explicit_action_scale(log_csv):
    log = load_trajectory_csv(str(log_csv), action_scale=[1.0] * 6)
    assert np.abs(log.episodes[0].actions[:, :6]).max() < 0.1
    with pytest.raises(LogValidationError, match="exceed action_scale"):
        load_trajectory_csv(str(log_csv), action_scale=[1e-6] * 6)
    with pytest.raises(ValueError, match="6 positive"):
        load_trajectory_csv(str(log_csv), action_scale=[1.0, 2.0])


def test_missing_columns_reported(log_csv):
    header, rows = _rows(log_csv)
    _write(log_csv, [h for h in header if h not in ("dz", "gripper")], rows)
    with pytest.raises(LogValidationError) as ei:
        load_trajectory_csv(str(log_csv))
    assert "gripper" in str(ei.value) and "dx,dy,dz" in str(ei.value)


def test_row_errors_collected_with_line_numbers(log_csv):
    header, rows = _rows(log_csv)
    rows[0]["dx"] = "abc"
    rows[1]["gripper"] = "1.5"
    rows[2]["frame_path"] = "frames/missing.png"
    rows[3]["timestep"] = "x"
    _write(log_csv, header, rows)
    with pytest.raises(LogValidationError) as ei:
        load_trajectory_csv(str(log_csv))
    text = str(ei.value)
    assert "line 2: dx='abc'" in text
    assert "line 3: gripper=1.5" in text
    assert "line 4: frame not found" in text
    assert "line 5: timestep" in text


def test_gap_and_duplicate_timesteps(log_csv):
    header, rows = _rows(log_csv)
    rows[2]["timestep"] = "40"
    rows[7]["timestep"] = rows[6]["timestep"]
    _write(log_csv, header, rows)
    with pytest.raises(LogValidationError) as ei:
        load_trajectory_csv(str(log_csv))
    assert "contiguous" in str(ei.value) and "duplicate" in str(ei.value)


def test_timestamps_must_increase(log_csv):
    header, rows = _rows(log_csv)
    rows[2]["timestamp_s"] = "0.0"
    _write(log_csv, header, rows)
    with pytest.raises(LogValidationError, match="strictly increase"):
        load_trajectory_csv(str(log_csv))


def test_short_episodes_skipped_with_warning(log_csv):
    log = load_trajectory_csv(str(log_csv), min_length=6)
    assert len(log.episodes) == 2 and not log.warnings
    with pytest.raises(LogValidationError, match="no episode has at least"):
        load_trajectory_csv(str(log_csv), min_length=7)


def test_pose_mode_wraps_angles(tmp_path):
    write_template(str(tmp_path), n_episodes=1, episode_len=3, size=16)
    rows = [
        ["e", 0, "frames/SYNTHETIC_SAMPLE_00_0000.png", 0, 0, 0, 0, 0, 3.1, 0],
        ["e", 1, "frames/SYNTHETIC_SAMPLE_00_0001.png", 0.01, 0, 0, 0, 0, -3.1, 1],
        ["e", 2, "frames/SYNTHETIC_SAMPLE_00_0002.png", 0.02, 0, 0, 0, 0, -3.0, 1],
    ]
    p = tmp_path / "pose.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "timestep", "frame_path", "x", "y", "z", "roll", "pitch",
                    "yaw", "gripper"])
        w.writerows(rows)
    log = load_trajectory_csv(str(p))
    phys = denormalise_actions(log.episodes[0].actions, log.action_scale)
    assert phys[0, 5] == pytest.approx(2 * np.pi - 6.2, abs=1e-6)  # wrapped, not -6.2
    assert phys[0, 0] == pytest.approx(0.01) and np.all(phys[-1, :6] == 0)


def test_both_action_groups_rejected(log_csv):
    header, rows = _rows(log_csv)
    _write(log_csv, header + ["x", "y", "z", "roll", "pitch", "yaw"], rows)
    with pytest.raises(LogValidationError, match="not both"):
        load_trajectory_csv(str(log_csv))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_trajectory_csv("/nonexistent/log.csv")
