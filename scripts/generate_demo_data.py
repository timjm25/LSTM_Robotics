"""
Generate synthetic demo episodes in the EpisodeDirectory format.

Creates data/demo/ with N episodes, each containing:
  - frames/  (PNG images of a simulated pick-and-place workspace)
  - actions.json

Usage:
    python scripts/generate_demo_data.py --n_episodes 50 --episode_len 30
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[warn] Pillow not installed. Frames will be 1x1 placeholder PNGs.")
    print("       Install with: pip install Pillow")


def _make_frame(t: int, ep_len: int, obj_color: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Render a simple 224×224 RGB pick-and-place workspace frame."""
    H, W = 224, 224
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Table surface
    img[H // 2:, :] = [40, 35, 30]

    # Moving target object (coloured sphere projection)
    cx = int(W * (0.2 + 0.6 * (t / ep_len)))
    cy = int(H * 0.55)
    r = 18
    ys, xs = np.ogrid[:H, :W]
    mask_obj = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    img[mask_obj] = (obj_color * 255).astype(np.uint8)

    # Gripper end-effector position
    gx = int(W * (0.1 + 0.8 * max(0, min(1, 0.5 + action[0]))))
    gy = int(H * (0.1 + 0.4 * max(0, min(1, 0.5 + action[1]))))
    gripper_open = action[-1] < 0.5
    col_gripper = np.array([200, 200, 200]) if gripper_open else np.array([60, 200, 60])
    img[
        max(0, gy - 5): min(H, gy + 5),
        max(0, gx - 5): min(W, gx + 5),
    ] = col_gripper

    # Approach indicator bar
    progress = t / ep_len
    bar_len = int(W * progress)
    img[5:10, :bar_len] = [80, 160, 240]

    return img


def _smooth_actions(ep_len: int, action_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth action trajectory via AR(1) process."""
    actions = np.zeros((ep_len, action_dim), dtype=np.float32)
    for d in range(action_dim - 1):
        noise = rng.standard_normal(ep_len).astype(np.float32)
        for t in range(1, ep_len):
            actions[t, d] = 0.85 * actions[t - 1, d] + 0.15 * noise[t]
        actions[:, d] = np.clip(actions[:, d], -1, 1)

    # Gripper: open for first half, close for second half (+ some noise)
    switch = rng.integers(ep_len // 3, 2 * ep_len // 3)
    actions[:switch, -1] = 0.0
    actions[switch:, -1] = 1.0
    return actions


def generate(n_episodes: int, episode_len: int, out_dir: str, seed: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        ep_dir = out / f"episode_{ep:04d}"
        frames_dir = ep_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        obj_color = rng.uniform(0.3, 1.0, (3,)).astype(np.float32)
        actions = _smooth_actions(episode_len, 7, rng)

        for t in range(episode_len):
            frame = _make_frame(t, episode_len, obj_color, actions[t])
            img_path = frames_dir / f"{t:04d}.png"
            if PIL_AVAILABLE:
                Image.fromarray(frame).save(img_path)
            else:
                # Write minimal 1×1 PNG placeholder
                import struct, zlib
                def _png_1x1(r, g, b):
                    raw = bytes([0, r, g, b])
                    compressed = zlib.compress(raw)
                    def chunk(name, data):
                        c = name + data
                        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
                    sig = b'\x89PNG\r\n\x1a\n'
                    ihdr = chunk(b'IHDR', struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                    idat = chunk(b'IDAT', compressed)
                    iend = chunk(b'IEND', b'')
                    return sig + ihdr + idat + iend
                img_path.write_bytes(_png_1x1(frame[0, 0, 0], frame[0, 0, 1], frame[0, 0, 2]))

        with open(ep_dir / "actions.json", "w") as f:
            json.dump(actions.tolist(), f)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Generated episode {ep + 1}/{n_episodes}")

    print(f"\nDemo data written to: {out.resolve()}")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Episode len : {episode_len} frames")
    print(f"  Total frames: {n_episodes * episode_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic robot demo data")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--episode_len", type=int, default=30)
    parser.add_argument("--out_dir", default="data/demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.n_episodes, args.episode_len, args.out_dir, args.seed)
