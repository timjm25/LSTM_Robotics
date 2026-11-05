"""
Robot video dataset loaders.

Supports four data formats:
  1. CSV trajectory / sensor log — real robot logs (schema in lstm_robotics/logs.py)
  2. HDF5 episodes  — standard robotics research format (e.g. RoboNet, RT-X)
  3. Directory of episodes — folders of images + actions.json
  4. Synthetic generator — optional demo / smoke test (use_synthetic=True / --demo)
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import ModelConfig, TrainConfig
from .logs import TrajectoryLog, load_frame, load_trajectory_csv


# ── Transforms ────────────────────────────────────────────────────────────────

def _default_transform(cfg: ModelConfig):
    """Normalisation matching ImageNet statistics (used by Google MobileNet)."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((cfg.image_height, cfg.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    except ImportError:
        return None


def _numpy_to_tensor(img: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Convert (H, W, C) uint8 numpy array to normalised (C, H, W) tensor."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    # Resize via simple nearest-neighbour (avoids PIL dependency)
    if img.shape[:2] != (height, width):
        h, w = img.shape[:2]
        rh = np.linspace(0, h - 1, height, dtype=int)
        rw = np.linspace(0, w - 1, width, dtype=int)
        img = img[np.ix_(rh, rw)]
    img = img.transpose(2, 0, 1)  # (C, H, W)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    img = (img - mean) / std
    return torch.tensor(img, dtype=torch.float32)


# ── Episode directory dataset ─────────────────────────────────────────────────

class EpisodeDirectoryDataset(Dataset):
    """
    Reads robot episodes from a directory structure:

        data/
          episode_000/
            frames/
              0000.png  0001.png  ...
            actions.json   # list of action vectors [{x,y,z,roll,pitch,yaw,gripper}]
          episode_001/
            ...

    Each dataset item is a fixed-length window of (sequence_length) consecutive
    frames with the corresponding action labels.
    """

    def __init__(
        self,
        data_dir: str,
        cfg: ModelConfig,
        augment: bool = False,
    ):
        self.cfg = cfg
        self.augment = augment
        self.episodes: List[Dict] = []

        root = Path(data_dir)
        for ep_dir in sorted(root.iterdir()):
            if not ep_dir.is_dir():
                continue
            frames_dir = ep_dir / "frames"
            actions_file = ep_dir / "actions.json"
            if not frames_dir.exists() or not actions_file.exists():
                continue
            frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
            if len(frame_paths) < cfg.sequence_length:
                continue
            with open(actions_file) as f:
                actions = json.load(f)  # list of 7-element lists
            if len(actions) < len(frame_paths):
                continue
            self.episodes.append({
                "frames": frame_paths,
                "actions": np.array(actions, dtype=np.float32),
            })

        # Build flat index of (episode_idx, start_frame)
        self.index: List[Tuple[int, int]] = []
        for ep_i, ep in enumerate(self.episodes):
            n = len(ep["frames"]) - cfg.sequence_length
            for start in range(n + 1):
                self.index.append((ep_i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_i, start = self.index[idx]
        ep = self.episodes[ep_i]
        T = self.cfg.sequence_length

        frames = []
        for t in range(T):
            img = self._load_image(ep["frames"][start + t])
            frames.append(img)

        frames_tensor = torch.stack(frames, dim=0)   # (T, C, H, W)
        actions_tensor = torch.tensor(
            ep["actions"][start: start + T], dtype=torch.float32
        )

        if self.augment:
            frames_tensor = self._augment(frames_tensor)

        return {"frames": frames_tensor, "actions": actions_tensor}

    def _load_image(self, path: Path) -> torch.Tensor:
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))
        except ImportError:
            img = np.zeros(
                (self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8
            )
        return _numpy_to_tensor(img, self.cfg.image_height, self.cfg.image_width)

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        """Consistent spatial augmentation applied across all frames in sequence."""
        # Random horizontal flip (consistent across T)
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])
        # Random colour jitter (mild; consistent across T)
        brightness = random.uniform(0.8, 1.2)
        frames = (frames * brightness).clamp(-3.0, 3.0)
        return frames


# ── CSV trajectory-log dataset (real robot logs) ─────────────────────────────

class CSVTrajectoryDataset(Dataset):
    """
    Fixed-length windows over episodes imported from a CSV trajectory log.

    Items: {"frames": (T, C, H, W), "actions": (T, 7)} plus "sensors": (T, n) when the log
    has sensor_* columns. Actions are normalised with the log's action_scale.
    """

    def __init__(self, log: TrajectoryLog, cfg: ModelConfig, augment: bool = False,
                 episode_ids: Optional[List[str]] = None):
        self.cfg = cfg
        self.augment = augment
        self.log = log
        self.action_scale = log.action_scale
        keep = None if episode_ids is None else set(episode_ids)
        self.episodes = [e for e in log.episodes if keep is None or e.episode_id in keep]
        T = cfg.sequence_length
        self.index: List[Tuple[int, int]] = [
            (i, s) for i, ep in enumerate(self.episodes) for s in range(len(ep) - T + 1)
        ]

    @classmethod
    def from_csv(cls, path: str, cfg: ModelConfig, action_scale=None, **kw) -> "CSVTrajectoryDataset":
        log = load_trajectory_csv(path, action_scale=action_scale,
                                  min_length=cfg.sequence_length)
        for w in log.warnings:
            print(f"[dataset] warning: {w}")
        return cls(log, cfg, **kw)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_i, start = self.index[idx]
        ep = self.episodes[ep_i]
        T = self.cfg.sequence_length
        frames = torch.stack([
            _numpy_to_tensor(load_frame(ep.frame_paths[start + t]),
                             self.cfg.image_height, self.cfg.image_width)
            for t in range(T)
        ])
        if self.augment and random.random() < 0.5:
            frames = (frames * random.uniform(0.8, 1.2)).clamp(-3.0, 3.0)
        item = {
            "frames": frames,
            "actions": torch.tensor(ep.actions[start: start + T], dtype=torch.float32),
        }
        if ep.sensors.shape[1]:
            item["sensors"] = torch.tensor(ep.sensors[start: start + T], dtype=torch.float32)
        return item


# ── HDF5 dataset (RoboNet / Open X-Embodiment format) ────────────────────────

class HDF5Dataset(Dataset):
    """
    Loads episodes from a single HDF5 file.

    Expected schema:
        /episode_NNN/
            observations/images   — (T, H, W, 3)  uint8
            actions               — (T, action_dim) float32
    """

    def __init__(self, hdf5_path: str, cfg: ModelConfig):
        import h5py
        self.cfg = cfg
        self.index: List[Tuple] = []
        self._h5 = h5py.File(hdf5_path, "r")
        for ep_key in self._h5.keys():
            ep = self._h5[ep_key]
            T = ep["observations/images"].shape[0]
            for start in range(T - cfg.sequence_length + 1):
                self.index.append((ep_key, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_key, start = self.index[idx]
        T = self.cfg.sequence_length
        ep = self._h5[ep_key]
        imgs = ep["observations/images"][start: start + T]   # (T, H, W, 3)
        acts = ep["actions"][start: start + T]               # (T, action_dim)
        frames = torch.stack([
            _numpy_to_tensor(imgs[t], self.cfg.image_height, self.cfg.image_width)
            for t in range(T)
        ])
        return {"frames": frames, "actions": torch.tensor(acts.astype(np.float32), dtype=torch.float32)}


# ── Synthetic dataset (smoke-test / demo) ────────────────────────────────────

class SyntheticRobotDataset(Dataset):
    """
    Generates random video sequences with correlated synthetic actions.
    Used for architecture smoke-tests and CI without requiring real data.

    The synthetic episodes model a simplified pick-and-place task:
      - Gripper moves toward a random object colour region
      - Action trajectory is smooth (low-pass filtered noise)
    """

    def __init__(
        self,
        n_episodes: int = 100,
        episode_len: int = 30,
        cfg: Optional[ModelConfig] = None,
        seed: int = 42,
    ):
        self.cfg = cfg or ModelConfig()
        self.n_episodes = n_episodes
        self.episode_len = episode_len
        rng = np.random.default_rng(seed)

        T = self.cfg.sequence_length
        self.index: List[Tuple[int, int]] = [
            (ep, t)
            for ep in range(n_episodes)
            for t in range(episode_len - T + 1)
        ]
        self._rng = np.random.default_rng(seed + 1)
        self._episodes = self._generate_all(rng, n_episodes, episode_len)

    def _generate_all(self, rng, n_ep, ep_len):
        H, W = self.cfg.image_height, self.cfg.image_width
        episodes = []
        for _ in range(n_ep):
            # Smooth action trajectory via AR(1) process
            actions = np.zeros((ep_len, self.cfg.action_dim), dtype=np.float32)
            for d in range(self.cfg.action_dim - 1):
                noise = rng.standard_normal(ep_len)
                for t in range(1, ep_len):
                    actions[t, d] = 0.8 * actions[t - 1, d] + 0.2 * noise[t]
            # Gripper: binary-ish, switches once mid-episode
            switch = rng.integers(ep_len // 3, 2 * ep_len // 3)
            actions[:switch, -1] = 0.0
            actions[switch:, -1] = 1.0
            # Clip to [-1, 1] and [0, 1]
            actions[:, :-1] = np.clip(actions[:, :-1], -1, 1)
            actions[:, -1] = np.clip(actions[:, -1], 0, 1)

            # Synthetic frames: random coloured blobs on dark background
            obj_color = rng.uniform(0.3, 1.0, (3,)).astype(np.float32)
            frames = []
            for t in range(ep_len):
                img = np.zeros((H, W, 3), dtype=np.float32)
                # Moving object blob
                cx = int(W * (0.3 + 0.4 * (t / ep_len)))
                cy = int(H * 0.5)
                r = max(H, W) // 8
                ys, xs = np.ogrid[:H, :W]
                mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
                img[mask] = obj_color
                # Gripper indicator
                gx = int(W * (0.1 + 0.8 * actions[t, 0]))
                gy = int(H * (0.1 + 0.8 * actions[t, 1]))
                img[
                    max(0, gy - 4): gy + 4,
                    max(0, gx - 4): gx + 4,
                ] = [0.9, 0.9, 0.9]
                frames.append(img)

            episodes.append({
                "frames": np.stack(frames).astype(np.float32),  # (T, H, W, 3) float32 [0,1]
                "actions": actions,
            })
        return episodes

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_i, start = self.index[idx]
        T = self.cfg.sequence_length
        ep = self._episodes[ep_i]
        frames_np = ep["frames"][start: start + T]       # (T, H, W, 3)

        frames = []
        for t in range(T):
            img = (frames_np[t] * 255).astype(np.uint8)
            frames.append(_numpy_to_tensor(img, self.cfg.image_height, self.cfg.image_width))

        return {
            "frames": torch.stack(frames),               # (T, C, H, W)
            "actions": torch.tensor(ep["actions"][start: start + T], dtype=torch.float32),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def _split_by_episode(ds: "CSVTrajectoryDataset", train_cfg: TrainConfig, model_cfg: ModelConfig):
    """Episode-level split so overlapping windows never leak between train/val/test."""
    ids = [e.episode_id for e in ds.episodes]
    order = np.random.default_rng(train_cfg.seed).permutation(len(ids))
    n_test = max(1, int(round(len(ids) * train_cfg.test_split)))
    n_val = max(1, int(round(len(ids) * train_cfg.val_split)))
    test_ids = [ids[i] for i in order[:n_test]]
    val_ids = [ids[i] for i in order[n_test:n_test + n_val]]
    train_ids = [ids[i] for i in order[n_test + n_val:]]
    mk = lambda sel, aug: CSVTrajectoryDataset(ds.log, model_cfg, augment=aug,  # noqa: E731
                                               episode_ids=sel)
    return mk(train_ids, True), mk(val_ids, False), mk(test_ids, False)


def build_dataset(train_cfg: TrainConfig, model_cfg: ModelConfig) -> Dataset:
    """Pick the data source from the config (CSV log > data_dir > synthetic demo)."""
    if train_cfg.data_csv:
        return CSVTrajectoryDataset.from_csv(train_cfg.data_csv, model_cfg,
                                             action_scale=train_cfg.action_scale)
    data_path = Path(train_cfg.data_dir)
    if data_path.suffix.lower() == ".csv":
        return CSVTrajectoryDataset.from_csv(str(data_path), model_cfg,
                                             action_scale=train_cfg.action_scale)
    if data_path.exists() and data_path.is_dir() and any(data_path.iterdir()):
        if any(data_path.glob("*.h5")) or any(data_path.glob("*.hdf5")):
            h5_file = next(data_path.glob("*.h5*"))
            return HDF5Dataset(str(h5_file), model_cfg)
        return EpisodeDirectoryDataset(str(data_path), model_cfg, augment=True)
    if train_cfg.use_synthetic:
        print("[dataset] Using SyntheticRobotDataset (demo data, not real robot data).")
        return SyntheticRobotDataset(n_episodes=train_cfg.synthetic_episodes, cfg=model_cfg,
                                     seed=train_cfg.seed)
    raise FileNotFoundError(
        f"no training data at {data_path!s}. Pass --data_csv <trajectory_log.csv> (see "
        "data/templates/), --data_dir <episodes dir or .h5>, or --demo for synthetic data."
    )


def make_dataloaders(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build (train, val, test) DataLoaders from the configured data source.

    CSV logs with >= 3 episodes are split by episode; other sources split by window.
    Synthetic demo data is used only when train_cfg.use_synthetic is True.
    """
    full_ds = build_dataset(train_cfg, model_cfg)
    if len(full_ds) == 0:
        raise ValueError("dataset has no complete windows; lower --seq_len or add longer episodes")

    if isinstance(full_ds, CSVTrajectoryDataset) and len(full_ds.episodes) >= 3:
        train_ds, val_ds, test_ds = _split_by_episode(full_ds, train_cfg, model_cfg)
    else:
        n = len(full_ds)
        if n < 3:
            raise ValueError(f"only {n} training window(s); need at least 3 for train/val/test")
        n_test = max(1, int(n * train_cfg.test_split))
        n_val = max(1, int(n * train_cfg.val_split))
        n_train = n - n_val - n_test

        g = torch.Generator()
        g.manual_seed(train_cfg.seed)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test], generator=g
        )

    kw = dict(
        batch_size=train_cfg.batch_size,
        num_workers=min(train_cfg.num_workers, os.cpu_count() or 1),
        pin_memory=train_cfg.pin_memory and torch.cuda.is_available(),
    )
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def dataset_action_scale(loader: DataLoader) -> Optional[List[float]]:
    """The physical action scale behind a loader (CSV logs), or None for other sources."""
    ds = loader.dataset
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    scale = getattr(ds, "action_scale", None)
    return None if scale is None else [float(v) for v in scale]
