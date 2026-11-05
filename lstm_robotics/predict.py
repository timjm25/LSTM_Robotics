"""
Online inference — stream video frames to the model and receive robot actions.

Usage
-----
Single image:
    python -m lstm_robotics.predict --checkpoint checkpoints/best.pt --image frame.png

Video file:
    python -m lstm_robotics.predict --checkpoint checkpoints/best.pt --video episode.mp4

ROS 2 bridge example (pseudocode):
    predictor = Predictor.from_checkpoint("checkpoints/best.pt")
    for frame in camera_stream:
        action = predictor.step(frame)
        robot.send_action(action)
"""
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .dataset import _numpy_to_tensor
from .model import VisuoMotorLSTM, load_checkpoint


class Predictor:
    """
    Stateful online predictor.

    Maintains ConvLSTM hidden states between calls so the model has memory of
    the full trajectory history without buffering all past frames.
    """

    def __init__(self, model: VisuoMotorLSTM, device: str = "cpu",
                 metadata: Optional[dict] = None):
        self.model = model.to(device)
        # From the checkpoint, e.g. {"action_scale": [...]} to convert actions to robot units
        self.metadata = dict(metadata or {})
        self.model.eval()
        self.device = torch.device(device)
        self.states = None  # ConvLSTM hidden states
        self.cfg = model.cfg

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "Predictor":
        model, ckpt = load_checkpoint(path, device)
        return cls(model, device, metadata=ckpt.get("metadata", {}))

    def reset(self):
        """Reset hidden states — call at the start of each new episode."""
        self.states = None

    def step(self, frame_np: np.ndarray) -> np.ndarray:
        """
        Process one frame and return the predicted action.

        Args:
            frame_np: (H, W, 3) uint8 numpy array (RGB)

        Returns:
            action: (action_dim,) numpy array
                [0:3] Δ(x, y, z)          — normalised [-1, 1]
                [3:6] Δ(roll, pitch, yaw)  — normalised [-1, 1]
                [6]   gripper              — [0, 1] (0=open, 1=close)
        """
        frame_t = _numpy_to_tensor(
            frame_np, self.cfg.image_height, self.cfg.image_width
        ).to(self.device)  # (C, H, W)

        action_t, self.states = self.model.predict_one(frame_t, self.states)
        return action_t.cpu().numpy()

    def step_sequence(self, frames_np: np.ndarray) -> np.ndarray:
        """
        Process a batch of frames (no state persistence between sequences).

        Args:
            frames_np: (T, H, W, 3) uint8 numpy array

        Returns:
            actions: (T, action_dim)
        """
        T = frames_np.shape[0]
        frames = torch.stack([
            _numpy_to_tensor(frames_np[t], self.cfg.image_height, self.cfg.image_width)
            for t in range(T)
        ]).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        with torch.no_grad():
            actions, _ = self.model(frames)
        return actions[0].cpu().numpy()  # (T, action_dim)


def _predict_image(predictor: Predictor, image_path: str):
    try:
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
    except ImportError:
        print("[predict] PIL not installed. Install Pillow to load images.")
        return

    predictor.reset()
    action = predictor.step(img)
    print("\nPredicted action for frame:")
    names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    for name, val in zip(names, action):
        print(f"  {name:10s}: {val:+.4f}")
    gripper_state = "CLOSE" if action[-1] > 0.5 else "OPEN"
    print(f"\n  Gripper: {gripper_state} (raw={action[-1]:.4f})")


def _predict_video(predictor: Predictor, video_path: str, out_path: Optional[str] = None):
    try:
        import cv2
    except ImportError:
        print("[predict] opencv-python not installed. Install it to process videos.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[predict] Cannot open video: {video_path}")
        return

    predictor.reset()
    actions = []
    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        action = predictor.step(frame_rgb)
        actions.append(action)
        frame_count += 1

    cap.release()
    elapsed = time.time() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0
    actions_np = np.array(actions)

    print(f"\nProcessed {frame_count} frames at {fps:.1f} fps")
    print("\nAction statistics (mean ± std):")
    names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    for i, name in enumerate(names):
        print(f"  {name:10s}: {actions_np[:, i].mean():+.4f} ± {actions_np[:, i].std():.4f}")

    if out_path:
        np.save(out_path, actions_np)
        print(f"\nActions saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisuoMotor LSTM online inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--image", help="Single image path for one-shot prediction")
    parser.add_argument("--video", help="Video file for sequential prediction")
    parser.add_argument("--output", help="Output path (.npy) for video predictions")
    parser.add_argument("--device", default="cpu", help="Device: cpu | cuda | mps")
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)

    if args.image:
        _predict_image(predictor, args.image)
    elif args.video:
        _predict_video(predictor, args.video, out_path=args.output)
    else:
        print("Provide --image or --video. Run with --help for usage.")
