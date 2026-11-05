"""
Import real robot trajectory / sensor logs from CSV (torch-free).

One row per control timestep. Template: ``data/templates/trajectory_log_template.csv``.

Required columns
----------------
    episode_id   name of the demonstration; rows with the same id form one episode
    timestep     integer step index, contiguous within an episode (any start value)
    frame_path   camera frame for this step, relative to the CSV's folder
                 (.png / .jpg / .bmp via Pillow, or .npy holding an (H, W, 3) uint8 array)
    gripper      gripper command in [0, 1] (0 = open, 1 = closed)

plus ONE of these action groups:

    dx, dy, dz, droll, dpitch, dyaw    commanded end-effector deltas per step  ("delta" mode)
    x, y, z, roll, pitch, yaw          absolute end-effector pose per step     ("pose" mode)

In pose mode the action at step t is pose[t+1] - pose[t] (angles wrapped to [-pi, pi]); the
last step of each episode gets a zero motion delta. Use one unit system throughout (we suggest
metres and radians); the same units come back out of the robot adapter.

Optional columns
----------------
    timestamp_s  seconds; must increase within an episode
    sensor_*     any number of numeric sensor channels (e.g. sensor_force_z, sensor_joint_3).
                 Validated and returned with each training window as ``sensors``; the current
                 model does not consume them.

Normalisation
-------------
The policy's motion head outputs [-1, 1] (tanh). Physical deltas are divided by a per-axis
``action_scale`` (6 values). By default it is the largest |delta| per axis in the file, so the
data spans [-1, 1]; pass your own (e.g. the robot's max step) for consistent units across
datasets. The scale is stored in every checkpoint and used to convert predictions back.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DELTA_COLUMNS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
POSE_COLUMNS = ("x", "y", "z", "roll", "pitch", "yaw")
BASE_COLUMNS = ("episode_id", "timestep", "frame_path", "gripper")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
_ANGLE_IDX = (3, 4, 5)


class LogValidationError(ValueError):
    """The CSV failed validation. ``problems`` holds every issue found (with line numbers)."""

    def __init__(self, source: str, problems: Sequence[str]):
        self.problems = list(problems)
        shown = "\n  - ".join(self.problems[:25])
        more = (f"\n  ... and {len(self.problems) - 25} more"
                if len(self.problems) > 25 else "")
        super().__init__(f"{source}: {len(self.problems)} problem(s):\n  - {shown}{more}")


@dataclass
class LoggedEpisode:
    episode_id: str
    frame_paths: List[Path]
    actions: np.ndarray            # (T, 7) float32, motion normalised to [-1, 1], gripper [0, 1]
    sensors: np.ndarray            # (T, n_sensors) float32 (n_sensors may be 0)
    timestamps: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.frame_paths)


@dataclass
class TrajectoryLog:
    source: Path
    episodes: List[LoggedEpisode]
    action_scale: np.ndarray       # (6,) physical units per normalised unit
    action_mode: str               # "delta" | "pose"
    sensor_names: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return sum(len(e) for e in self.episodes)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _parse_float(value: str, col: str, line: int, problems: List[str]) -> float:
    try:
        v = float(value)
    except ValueError:
        problems.append(f"line {line}: {col}={value!r} is not a number")
        return 0.0
    if not math.isfinite(v):
        problems.append(f"line {line}: {col} is NaN/inf")
        return 0.0
    return v


def load_trajectory_csv(
    path: str,
    action_scale: Optional[Sequence[float]] = None,
    check_frames: bool = True,
    min_length: int = 1,
) -> TrajectoryLog:
    """Parse and validate a trajectory log CSV. Raises LogValidationError listing all issues."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trajectory log not found: {p}")
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        raw_rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()}
                    for r in reader]

    problems: List[str] = []
    missing = [c for c in BASE_COLUMNS if c not in header]
    has_delta = all(c in header for c in DELTA_COLUMNS)
    has_pose = all(c in header for c in POSE_COLUMNS)
    if not has_delta and not has_pose:
        missing.append("either " + ",".join(DELTA_COLUMNS) + " or " + ",".join(POSE_COLUMNS))
    if has_delta and has_pose:
        problems.append("give delta columns OR pose columns, not both")
    if missing:
        problems.append(f"missing column(s): {missing}")
    if problems:
        raise LogValidationError(str(p), problems + [f"header found: {header}"])
    mode = "delta" if has_delta else "pose"
    motion_cols = DELTA_COLUMNS if has_delta else POSE_COLUMNS
    sensor_names = [h for h in header if h.startswith("sensor_")]
    has_ts = "timestamp_s" in header

    grouped: Dict[str, List[dict]] = {}
    for line, row in enumerate(raw_rows, start=2):
        if not any(row.values()):
            continue
        ep = row["episode_id"]
        if not ep:
            problems.append(f"line {line}: episode_id is empty")
            continue
        try:
            ts = int(row["timestep"])
        except ValueError:
            problems.append(f"line {line}: timestep={row['timestep']!r} is not an integer")
            continue
        motion = [_parse_float(row[c], c, line, problems) for c in motion_cols]
        grip = _parse_float(row["gripper"], "gripper", line, problems)
        if not 0.0 <= grip <= 1.0:
            problems.append(f"line {line}: gripper={grip} must be in [0, 1]")
        sensors = [_parse_float(row[c], c, line, problems) for c in sensor_names]
        t_s = _parse_float(row["timestamp_s"], "timestamp_s", line, problems) if has_ts else None
        frame = row["frame_path"]
        fpath = p.parent / frame
        if not frame:
            problems.append(f"line {line}: frame_path is empty")
        elif fpath.suffix.lower() not in IMAGE_EXT | {".npy"}:
            problems.append(f"line {line}: unsupported frame type {fpath.suffix!r}")
        elif check_frames and not fpath.exists():
            problems.append(f"line {line}: frame not found: {frame}")
        grouped.setdefault(ep, []).append(dict(line=line, ts=ts, motion=motion, grip=grip,
                                               sensors=sensors, t_s=t_s, frame=fpath))

    episodes_raw = []
    warnings: List[str] = []
    for ep, rows in grouped.items():
        rows.sort(key=lambda r: r["ts"])
        steps = [r["ts"] for r in rows]
        if len(set(steps)) != len(steps):
            problems.append(f"episode {ep!r}: duplicate timestep values")
            continue
        if steps != list(range(steps[0], steps[0] + len(steps))):
            problems.append(f"episode {ep!r}: timesteps must be contiguous, got gaps in "
                            f"{steps[0]}..{steps[-1]}")
            continue
        if has_ts:
            t = np.array([r["t_s"] for r in rows])
            if np.any(np.diff(t) <= 0):
                problems.append(f"episode {ep!r}: timestamp_s must strictly increase")
        if len(rows) < min_length:
            warnings.append(f"episode {ep!r} skipped: {len(rows)} steps < sequence length "
                            f"{min_length}")
            continue
        motion = np.array([r["motion"] for r in rows], dtype=np.float64)
        if mode == "pose":
            d = np.zeros_like(motion)
            d[:-1] = np.diff(motion, axis=0)
            d[:, list(_ANGLE_IDX)] = _wrap(d[:, list(_ANGLE_IDX)])
            motion = d
        episodes_raw.append((ep, rows, motion))
    if not grouped and not problems:
        problems.append("no data rows")
    if grouped and not episodes_raw and not problems:
        problems.append(f"no episode has at least {min_length} steps; lower the sequence "
                        "length (--seq_len) or log longer episodes")
    if problems:
        raise LogValidationError(str(p), problems)

    all_motion = np.concatenate([m for _, _, m in episodes_raw], axis=0)
    if action_scale is None:
        scale = np.abs(all_motion).max(axis=0)
        scale[scale == 0] = 1.0
    else:
        scale = np.asarray(action_scale, dtype=np.float64).reshape(-1)
        if scale.shape != (6,) or np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("action_scale needs 6 positive numbers (dx..dyaw)")
        over = np.abs(all_motion) > scale * (1 + 1e-6)
        if over.any():
            axes = [DELTA_COLUMNS[i] for i in np.where(over.any(axis=0))[0]]
            raise LogValidationError(str(p), [
                f"{int(over.sum())} delta value(s) exceed action_scale on axes {axes}; "
                "increase the scale or omit it to auto-scale"])

    episodes = []
    for ep, rows, motion in episodes_raw:
        actions = np.zeros((len(rows), 7), dtype=np.float32)
        actions[:, :6] = np.clip(motion / scale, -1.0, 1.0)
        actions[:, 6] = [r["grip"] for r in rows]
        sensors = np.array([r["sensors"] for r in rows], dtype=np.float32).reshape(
            len(rows), len(sensor_names))
        ts = np.array([r["t_s"] for r in rows], dtype=np.float64) if has_ts else None
        episodes.append(LoggedEpisode(ep, [r["frame"] for r in rows], actions, sensors, ts))
    return TrajectoryLog(p, episodes, scale.astype(np.float32), mode, sensor_names, warnings)


def load_frame(path: Path) -> np.ndarray:
    """Read one frame as (H, W, 3) uint8 RGB."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{path.name}: .npy frames must be (H, W, 3), got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr
    from PIL import Image
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def denormalise_actions(actions: np.ndarray, action_scale: Sequence[float]) -> np.ndarray:
    """Model output ([-1, 1] motion, [0, 1] gripper) -> physical deltas + gripper."""
    out = np.array(actions, dtype=np.float64, copy=True)
    out[..., :6] = out[..., :6] * np.asarray(action_scale, dtype=np.float64)
    return out


def write_template(out_dir: str, n_episodes: int = 2, episode_len: int = 6,
                   size: int = 32, seed: int = 0) -> Path:
    """Write a SYNTHETIC example log (CSV + PNG frames) in the delta schema; returns CSV path."""
    from PIL import Image

    out = Path(out_dir)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    csv_path = out / "trajectory_log.csv"
    sensors = ["sensor_force_z", "sensor_gripper_width_m"]
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "timestep", "timestamp_s", "frame_path", *DELTA_COLUMNS,
                    "gripper", *sensors])
        for e in range(n_episodes):
            ep = f"SYNTHETIC_SAMPLE_{e:02d}"
            switch = episode_len // 2
            for t in range(episode_len):
                img = np.zeros((size, size, 3), dtype=np.uint8)
                cx = int(size * (0.2 + 0.6 * t / max(1, episode_len - 1)))
                img[size // 2 - 3: size // 2 + 3, max(0, cx - 3): cx + 3] = (200, 80, 40)
                name = f"{ep}_{t:04d}.png"
                Image.fromarray(img).save(frames / name)
                d = rng.normal(0, 0.005, 6).round(5)
                grip = 1.0 if t >= switch else 0.0
                w.writerow([ep, t, f"{t * 0.04:.2f}", f"frames/{name}", *d, grip,
                            f"{rng.normal(2.0, 0.1):.3f}", f"{0.08 * (1 - grip):.3f}"])
    return csv_path
