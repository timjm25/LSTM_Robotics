"""
Robot interface: where the trained policy meets a robot (simulated by default).

    RobotInterface         the contract a robot adapter implements
    SimulatedRobot         default; renders frames from its own state, no hardware
    TCPBridgeRobot         real robots via a small bridge process (JSON lines over TCP)
                           + an OpenCV camera; configured by LSTM_ROBOT_* env vars
    ActionLimiter          per-step safety clip applied before any action is sent
    robot_from_env()       pick the adapter from the environment
    run_policy()           closed-loop control: frame -> policy -> limiter -> robot

Environment variables (see .env.example)
----------------------------------------
    LSTM_ROBOT_DRIVER      "sim" (default) or "tcp"
    LSTM_ROBOT_HOST        bridge host (tcp)               e.g. 127.0.0.1
    LSTM_ROBOT_PORT        bridge port (tcp)               e.g. 5555
    LSTM_ROBOT_CAMERA      OpenCV camera: device index ("0") or stream URL / video path (tcp)
    LSTM_ROBOT_TIMEOUT_S   socket timeout, default 2.0
    LSTM_ROBOT_MAX_DELTA   per-step |delta| cap in physical units: one value or six
                           (dx,dy,dz,droll,dpitch,dyaw); default 0.01

Bridge protocol (TCPBridgeRobot <-> your robot-side process), one JSON object per line:

    -> {"cmd": "hello"}                                   <- {"ok": true, "name": "..."}
    -> {"cmd": "action", "delta": [dx,dy,dz,droll,dpitch,dyaw], "gripper": g}
                                                          <- {"ok": true}
    -> {"cmd": "stop"}                                    <- {"ok": true}
    any failure                                           <- {"ok": false, "error": "..."}

Write the bridge on the robot side with your vendor SDK or ROS 2 (``lstm_robotics.bridge``
is a dry-run reference implementation). Deltas are in the units of the CSV logs the policy
was trained on (we suggest metres / radians), expressed in the end-effector frame you logged.
"""
from __future__ import annotations

import csv
import json
import os
import socket
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .logs import DELTA_COLUMNS


class RobotUnavailableError(RuntimeError):
    """The configured robot / camera cannot be reached. The message says what to check."""


class RobotError(RuntimeError):
    """The robot (bridge) answered with an error."""


@runtime_checkable
class RobotInterface(Protocol):
    def connect(self) -> None: ...
    def get_frame(self) -> np.ndarray:
        """Latest camera frame, (H, W, 3) uint8 RGB."""
        ...
    def send_action(self, delta: np.ndarray, gripper: float) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class ActionLimiter:
    """Clip each motion component to +/- max_delta and the gripper to [0, 1]."""

    def __init__(self, max_delta: Sequence[float] | float = 0.01):
        md = np.broadcast_to(np.asarray(max_delta, dtype=np.float64), (6,)).copy()
        if np.any(md <= 0) or not np.all(np.isfinite(md)):
            raise ValueError("max_delta must be positive and finite")
        self.max_delta = md
        self.n_clipped = 0

    def __call__(self, delta: np.ndarray, gripper: float):
        d = np.asarray(delta, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(d)) or not np.isfinite(gripper):
            raise ValueError(f"non-finite action {d}, gripper={gripper}")
        clipped = np.clip(d, -self.max_delta, self.max_delta)
        if not np.array_equal(clipped, d):
            self.n_clipped += 1
        return clipped, float(np.clip(gripper, 0.0, 1.0))


class SimulatedRobot:
    """No-hardware robot: integrates deltas into a pose and draws a frame from it."""

    def __init__(self, image_size: int = 224, workspace: float = 1.0, seed: int = 0):
        self.image_size = image_size
        self.workspace = workspace
        self.pose = np.zeros(6, dtype=np.float64)
        self.gripper = 0.0
        self.sent: List[np.ndarray] = []
        self.connected = False
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None:
        self.connected = True

    def get_frame(self) -> np.ndarray:
        s = self.image_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        img[s // 2:, :] = (40, 35, 30)  # table
        u = np.clip(0.5 + self.pose[:2] / (2 * self.workspace), 0.0, 1.0)
        gx, gy = int(u[0] * (s - 1)), int(u[1] * (s - 1))
        colour = (60, 200, 60) if self.gripper > 0.5 else (200, 200, 200)
        img[max(0, gy - 4): gy + 4, max(0, gx - 4): gx + 4] = colour
        return img

    def send_action(self, delta: np.ndarray, gripper: float) -> None:
        if not self.connected:
            raise RobotError("SimulatedRobot.send_action before connect()")
        d = np.asarray(delta, dtype=np.float64).reshape(6)
        self.pose[:3] = np.clip(self.pose[:3] + d[:3], -self.workspace, self.workspace)
        self.pose[3:] = (self.pose[3:] + d[3:] + np.pi) % (2 * np.pi) - np.pi
        self.gripper = float(gripper)
        self.sent.append(np.concatenate([d, [gripper]]))

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.connected = False


def _opencv_camera(source: str) -> Callable[[], np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RobotUnavailableError("opencv-python is needed for LSTM_ROBOT_CAMERA "
                                    "(pip install opencv-python)") from exc
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RobotUnavailableError(f"cannot open camera {source!r}; check LSTM_ROBOT_CAMERA "
                                    "and that the camera is plugged in")

    def grab() -> np.ndarray:
        ok, bgr = cap.read()
        if not ok:
            raise RobotError(f"camera {source!r} returned no frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return grab


class TCPBridgeRobot:
    """Real-robot adapter: actions to a bridge over TCP, frames from a camera callable."""

    def __init__(self, host: str, port: int, frame_source: Callable[[], np.ndarray],
                 timeout_s: float = 2.0):
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self._frame_source = frame_source
        self._sock: Optional[socket.socket] = None
        self._rfile = None
        self.name = ""

    def connect(self) -> None:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        except OSError as exc:
            raise RobotUnavailableError(
                f"cannot reach robot bridge at {self.host}:{self.port} ({exc}). Is the bridge "
                "running and the robot on? Check LSTM_ROBOT_HOST / LSTM_ROBOT_PORT."
            ) from exc
        self._rfile = self._sock.makefile("r", encoding="utf-8")
        self.name = str(self._request({"cmd": "hello"}).get("name", ""))

    def _request(self, msg: Dict) -> Dict:
        if self._sock is None or self._rfile is None:
            raise RobotError("not connected; call connect() first")
        try:
            self._sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            line = self._rfile.readline()
        except OSError as exc:
            raise RobotError(f"bridge connection failed during {msg.get('cmd')!r}: {exc}") from exc
        if not line:
            raise RobotError(f"bridge closed the connection during {msg.get('cmd')!r}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RobotError(f"bridge sent invalid JSON: {line[:200]!r}") from exc
        if not reply.get("ok", False):
            raise RobotError(f"bridge rejected {msg.get('cmd')!r}: {reply.get('error', reply)}")
        return reply

    def get_frame(self) -> np.ndarray:
        frame = np.asarray(self._frame_source())
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RobotError(f"camera frame must be (H, W, 3), got {frame.shape}")
        return frame.astype(np.uint8, copy=False)

    def send_action(self, delta: np.ndarray, gripper: float) -> None:
        d = [float(v) for v in np.asarray(delta, dtype=np.float64).reshape(6)]
        self._request({"cmd": "action", "delta": d, "gripper": float(gripper)})

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._request({"cmd": "stop"})
            except RobotError:
                pass

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
        self._sock = self._rfile = None


def max_delta_from_env(env: Mapping[str, str]) -> np.ndarray:
    raw = env.get("LSTM_ROBOT_MAX_DELTA", "0.01").strip() or "0.01"
    try:
        vals = [float(v) for v in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"LSTM_ROBOT_MAX_DELTA must be numbers, got {raw!r}") from exc
    if len(vals) not in (1, 6):
        raise ValueError("LSTM_ROBOT_MAX_DELTA takes one value or six (dx..dyaw)")
    return np.broadcast_to(np.asarray(vals, dtype=np.float64), (6,)).copy()


def robot_from_env(
    env: Optional[Mapping[str, str]] = None,
    image_size: int = 224,
    frame_source: Optional[Callable[[], np.ndarray]] = None,
) -> RobotInterface:
    """Build the configured robot (simulated unless LSTM_ROBOT_DRIVER=tcp)."""
    e = os.environ if env is None else env
    driver = e.get("LSTM_ROBOT_DRIVER", "sim").strip().lower() or "sim"
    if driver == "sim":
        return SimulatedRobot(image_size=image_size)
    if driver != "tcp":
        raise ValueError(f"LSTM_ROBOT_DRIVER must be 'sim' or 'tcp', got {driver!r}")
    missing = [k for k in ("LSTM_ROBOT_HOST", "LSTM_ROBOT_PORT") if not e.get(k, "").strip()]
    if frame_source is None and not e.get("LSTM_ROBOT_CAMERA", "").strip():
        missing.append("LSTM_ROBOT_CAMERA")
    if missing:
        raise RobotUnavailableError(f"LSTM_ROBOT_DRIVER=tcp needs {', '.join(missing)} "
                                    "(see .env.example)")
    try:
        port = int(e["LSTM_ROBOT_PORT"])
        timeout = float(e.get("LSTM_ROBOT_TIMEOUT_S", "2.0") or 2.0)
    except ValueError as exc:
        raise ValueError(f"invalid LSTM_ROBOT_PORT / LSTM_ROBOT_TIMEOUT_S: {exc}") from exc
    source = frame_source or _opencv_camera(e["LSTM_ROBOT_CAMERA"].strip())
    return TCPBridgeRobot(e["LSTM_ROBOT_HOST"].strip(), port, source, timeout_s=timeout)


def run_policy(
    predictor,
    robot: RobotInterface,
    steps: int,
    action_scale: Sequence[float],
    limiter: Optional[ActionLimiter] = None,
    log_csv: Optional[str] = None,
    save_frames_dir: Optional[str] = None,
    episode_id: str = "run_000",
) -> np.ndarray:
    """
    Closed loop for ``steps`` ticks: frame -> predictor.step -> denormalise -> limit -> send.

    ``predictor`` is an ``lstm_robotics.predict.Predictor`` (anything with reset()/step()).
    Returns the (steps, 7) physical actions sent. On ANY error the robot is told to stop
    before the exception propagates. With ``log_csv`` the run is written in the trajectory
    log schema (with ``save_frames_dir`` it can be re-imported as training data).
    """
    limiter = limiter or ActionLimiter()
    scale = np.asarray(action_scale, dtype=np.float64).reshape(6)
    sent: List[np.ndarray] = []
    rows: List[list] = []
    frames_dir = Path(save_frames_dir) if save_frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    robot.connect()
    try:
        predictor.reset()
        for t in range(steps):
            frame = robot.get_frame()
            a = np.asarray(predictor.step(frame), dtype=np.float64)
            delta, grip = limiter(a[:6] * scale, float(a[6]))
            robot.send_action(delta, grip)
            sent.append(np.concatenate([delta, [grip]]))
            frame_ref = ""
            if frames_dir is not None:
                from PIL import Image
                name = f"{episode_id}_{t:04d}.png"
                Image.fromarray(frame).save(frames_dir / name)
                frame_ref = str(frames_dir / name)
            rows.append([episode_id, t, frame_ref, *delta.tolist(), grip])
    except BaseException:
        robot.stop()
        raise
    finally:
        if log_csv:
            p = Path(log_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["episode_id", "timestep", "frame_path", *DELTA_COLUMNS, "gripper"])
                for r in rows:
                    if r[2] and frames_dir is not None:
                        r[2] = os.path.relpath(r[2], p.parent)
                    w.writerow(r)
        robot.close()
    return np.array(sent, dtype=np.float64).reshape(-1, 7)
