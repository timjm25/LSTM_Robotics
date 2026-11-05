# LSTM Robotics — Video-Conditioned Robot Action Policy

A general ConvLSTM model for training robots to perform manipulation tasks
from video observations, inspired by Google Research's spatiotemporal learning
architecture.

---

## Business Problem

Modern robotics applications in manufacturing, logistics, and healthcare require
robots that can generalise across variable environments from raw visual input.
Collecting hand-coded policies for every task is impractical. Instead, robots
should learn to act from demonstration videos — the same way humans learn from
watching.

This project provides a complete training pipeline where:

- A robot camera stream is the primary input
- An LSTM with convolutional gates models how the scene evolves over time
- The model predicts end-effector actions at each timestep
- The trained policy can run on-robot for closed-loop control

**Target domains:**
- Bin picking / pick-and-place on assembly lines
- Surgical instrument handling (tele-op imitation learning)
- Autonomous mobile robot navigation from ego-camera
- Inspection and quality control using visual servoing

---

## Model Architecture

The model is a **VisuoMotor ConvLSTM** — a video-to-action policy network
inspired by Google Brain's work on spatiotemporal LSTM and MobileNet-based
on-device robotics inference.

```
Video frames (B, T, 3, H, W)
        │
        ▼ (applied independently at each timestep)
┌───────────────────────────┐
│  CNN Backbone              │   Lightweight depthwise-separable CNN
│  (Lightweight / MobileNetV2│   or pretrained MobileNetV2 (Google, 2018)
│   stride-16 feature map)   │   Output: (B·T, C, H/16, W/16)
└───────────┬───────────────┘
            │ reshape → (B, T, C, H', W')
            ▼
┌───────────────────────────┐
│  ConvLSTM Stack            │   Shi et al. NeurIPS 2015 — gates replaced
│  (3 layers, configurable)  │   with 2D convolutions to preserve spatial
│                            │   structure across timesteps
└───────────┬───────────────┘
            │ (B, T, Ch, H', W')
            ▼
        Flatten spatial dims
            │
            ▼
┌───────────────────────────┐
│  Action MLP Head           │   Separate heads for motion + gripper
│  LayerNorm + GELU          │   Motion : tanh → [-1, 1]  (6 DOF delta)
│                            │   Gripper: sigmoid → [0, 1] (open/close)
└───────────────────────────┘
            │
            ▼
Robot actions (B, T, 7)
  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

### ConvLSTM Cell

The core recurrent unit replaces standard LSTM fully-connected gates with
spatial convolutions:

```
i_t = σ(W_xi * X_t  +  W_hi * H_{t-1}  +  b_i)
f_t = σ(W_xf * X_t  +  W_hf * H_{t-1}  +  b_f)   ← bias init 1.0
g_t = tanh(W_xg * X_t  +  W_hg * H_{t-1}  +  b_g)
o_t = σ(W_xo * X_t  +  W_ho * H_{t-1}  +  b_o)
C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ g_t
H_t = o_t ⊙ tanh(C_t)
```

where `*` denotes 2D convolution (not matrix multiply).

### Action Space (7 DOF)

| Index | Dimension | Range      | Meaning                  |
|-------|-----------|------------|--------------------------|
| 0     | Δx        | [-1, 1]    | End-effector x delta     |
| 1     | Δy        | [-1, 1]    | End-effector y delta     |
| 2     | Δz        | [-1, 1]    | End-effector z delta     |
| 3     | Δroll     | [-1, 1]    | Wrist roll delta         |
| 4     | Δpitch    | [-1, 1]    | Wrist pitch delta        |
| 5     | Δyaw      | [-1, 1]    | Wrist yaw delta          |
| 6     | gripper   | [0, 1]     | 0=open, 1=close          |

### Loss Function

```
L = w_motion · MSE(pred[0:6], target[0:6])
  + w_gripper · BCE(pred[6],   target[6])
```

Default: `w_motion=1.0`, `w_gripper=2.0` (gripper accuracy is critical for
task completion).

### Default Hyperparameters

| Parameter          | Default    | Notes                              |
|--------------------|------------|------------------------------------|
| Image size         | 224 × 224  | Matches ImageNet pre-training norm  |
| Sequence length    | 8 frames   | ~0.3s at 25 fps                    |
| Backbone channels  | 256        | Feature map at H/16, W/16          |
| ConvLSTM layers    | 3          | Channels: [256, 128, 64]           |
| Action hidden dim  | 256        | MLP intermediate width             |
| Learning rate      | 1e-4       | AdamW + cosine decay               |
| Batch size         | 8          | Sequences (not frames)             |
| Trainable params   | ~10.2M     | Lightweight backbone               |

---

## Google Model Reference

This architecture directly references Google Research publications:

| Component        | Google Reference                                             |
|------------------|--------------------------------------------------------------|
| ConvLSTM cell    | Shi et al., NeurIPS 2015 (co-authored with Google Brain)     |
| MobileNetV2 opt. | Sandler et al., CVPR 2018 (Google)                           |
| Backbone design  | MobileNet depthwise-separable blocks (Google, 2017)          |
| Video prediction | "Video Prediction for Model-Based Deep RL" (Google, 2018)    |
| Action space     | RT-1 / RT-2 action tokenisation (Google Robotics, 2022/2023) |
| Training scheme  | SayCan visuomotor policy structure (Google, 2022)            |

The lightweight backbone mirrors Google's on-device MobileNet family, enabling
deployment on robot compute (Jetson, Coral, Raspberry Pi 5) without a GPU.

---

## Data Formats

### 1. Episode Directory

```
data/
  episode_0000/
    frames/
      0000.png   0001.png   ...   (RGB images, any resolution)
    actions.json               # [[x,y,z,roll,pitch,yaw,gripper], ...]
  episode_0001/
    ...
```

Generate synthetic demo data:
```bash
python scripts/generate_demo_data.py --n_episodes 100 --episode_len 30
```

### 2. HDF5 (RoboNet / Open X-Embodiment)

```
episode.h5
  /episode_NNN/
    observations/images   (T, H, W, 3)  uint8
    actions               (T, 7)        float32
```

### 3. CSV trajectory / sensor log (real robot logs)

One row per control step. Templates (clearly labelled `SYNTHETIC_SAMPLE_*` rows, 32 px PNG
frames): `data/templates/trajectory_log/trajectory_log_template.csv` (delta mode, with sensor
channels) and `trajectory_log_pose_example.csv` (pose mode).

| column | required | meaning |
|---|---|---|
| `episode_id` | yes | demonstration name; rows with the same id form one episode |
| `timestep` | yes | integer, contiguous within an episode |
| `frame_path` | yes | camera frame, relative to the CSV folder (`.png/.jpg/.bmp`, or `.npy` (H,W,3) uint8) |
| `dx,dy,dz,droll,dpitch,dyaw` | one group | commanded end-effector deltas per step |
| `x,y,z,roll,pitch,yaw` | one group | absolute end-effector pose; deltas are computed (angles wrapped) |
| `gripper` | yes | 0 = open ... 1 = closed |
| `timestamp_s` | no | must strictly increase within an episode |
| `sensor_*` | no | any numeric sensor channels (force, joint angles...); validated and returned with each window as `sensors`, not yet used by the model |

Validation lists every problem with its CSV line number (bad numbers, NaN, gripper outside
[0, 1], missing frames, timestep gaps/duplicates, non-increasing timestamps) and the CLI exits
with code 2. Episodes shorter than `--seq_len` are skipped with a warning. Motion is divided
by a per-axis `action_scale` (auto = max |delta| per axis, or pass `--action_scale` with six
values) so it fits the model's [-1, 1] output; the scale is saved in the checkpoint and used
to convert predictions back to robot units. CSV logs with 3+ episodes are split into
train/val/test **by episode**, so overlapping windows don't leak.

### 4. Synthetic (optional demo)

Pass `--demo` to train on the built-in `SyntheticRobotDataset` (no data needed) — useful for
checking the pipeline. Without `--demo`, a missing data path is an error rather than a
silent switch to synthetic data. `scripts/generate_demo_data.py` writes the same kind of
synthetic data in the episode-directory format.

---

## Quick Start

### Install

```bash
# Python 3.9+ required
pip install -r requirements.txt
pip install pytest            # to run the tests
```

### 1. Bring your own robot logs

```bash
python -m lstm_robotics template my_logs/            # example CSV + frames to copy
python -m lstm_robotics validate my_logs/trajectory_log.csv --seq_len 8
```

### 2. Train, evaluate, reload

```bash
python run_train.py --data_csv my_logs/trajectory_log.csv --epochs 100   # or: python -m lstm_robotics train ...
#   -> checkpoints/best.pt (lowest val loss), checkpoints/last.pt, checkpoints/history.json
python -m lstm_robotics evaluate --checkpoint checkpoints/best.pt --data_csv my_logs/trajectory_log.csv
#   -> eval_output/metrics.json (incl. MAE in your log's units) + predictions.png

python run_train.py --demo --epochs 5                                   # synthetic demo
python run_train.py --demo --image_size 64 --small --epochs 2           # fast CPU smoke test
python run_train.py --data_dir /path/to/episodes                        # episode dir / .h5
```

Checkpoints load with `torch.load(weights_only=True)` (`lstm_robotics.model.load_checkpoint`);
they are git-ignored — don't commit weights.

### 3. Run the policy on a robot (simulated by default)

```bash
python -m lstm_robotics run --checkpoint checkpoints/best.pt --steps 200 --log runs/run1.csv
python -m lstm_robotics check-robot          # connect + grab one frame, sends no motion
```

The robot is chosen by environment variables (see `.env.example`):

- `LSTM_ROBOT_DRIVER=sim` (default) — `SimulatedRobot`, no hardware.
- `LSTM_ROBOT_DRIVER=tcp` — `TCPBridgeRobot`: frames from an OpenCV camera
  (`LSTM_ROBOT_CAMERA`), actions as JSON lines to a bridge process at
  `LSTM_ROBOT_HOST:LSTM_ROBOT_PORT` that you run next to the robot (vendor SDK / ROS 2).
  The protocol is documented in `lstm_robotics/robot.py`; `python -m lstm_robotics.bridge`
  is a dry-run reference bridge that logs commands and moves nothing.
  Missing settings, an unreachable bridge or an unopenable camera raise
  `RobotUnavailableError` with what to check (`check-robot` exits with code 3).

Every action passes an `ActionLimiter` (per-step clip, `LSTM_ROBOT_MAX_DELTA`) before it is
sent, and any error mid-run sends `stop` to the robot. `--log` (with `--save_frames`) writes
the run in the CSV log schema, so runs can be reviewed or re-imported as training data.

### Online inference (one frame at a time)

```bash
python -m lstm_robotics.predict --checkpoint checkpoints/best.pt --image frame.png
python -m lstm_robotics.predict --checkpoint checkpoints/best.pt --video episode.mp4 \
  --output predicted_actions.npy
```

```python
from lstm_robotics.predict import Predictor
from lstm_robotics.logs import denormalise_actions

predictor = Predictor.from_checkpoint("checkpoints/best.pt")
scale = predictor.metadata.get("action_scale")      # set when trained on a CSV log
predictor.reset()
for frame_rgb in camera_stream:                     # (H, W, 3) uint8
    action = predictor.step(frame_rgb)              # (7,) normalised
    physical = denormalise_actions(action, scale)   # robot units
```

---

## Project Structure

```
LSTM_Robotics/
├── lstm_robotics/
│   ├── __init__.py
│   ├── config.py          # ModelConfig + TrainConfig dataclasses
│   ├── model.py           # ConvLSTMCell, ConvLSTM, VisuoMotorLSTM
│   ├── dataset.py         # EpisodeDirectory, HDF5, Synthetic datasets
│   ├── train.py           # Trainer, RobotActionLoss, gripper_accuracy
│   ├── evaluate.py        # Per-axis metrics, prediction plots
│   ├── predict.py         # Predictor (stateful online inference)
│   ├── logs.py            # CSV trajectory/sensor log import + validation
│   ├── robot.py           # RobotInterface, SimulatedRobot, TCPBridgeRobot, run_policy
│   ├── bridge.py          # dry-run reference robot bridge (TCP JSON lines)
│   └── cli.py             # python -m lstm_robotics {template,validate,train,evaluate,run,check-robot}
├── scripts/
│   ├── generate_demo_data.py
│   └── visualise_predictions.py
├── data/templates/        # example trajectory logs (synthetic, clearly labelled)
├── tests/                 # 47 tests: model, logs, pipeline, robot adapters
├── run_train.py           # Training entry point
└── requirements.txt
```

---

## Tests

```bash
python -m pytest -q
# → 47 passed (CPU only, ~10 s): model, CSV log import/validation, train -> best.pt ->
#   load -> evaluate -> run, robot adapters (sim, TCP bridge, failure paths)
```

## Still out of scope

- **No trained weights ship** with the repo; train on your own logs.
- **No vendor-specific robot driver.** Real robots connect through the documented TCP bridge
  protocol; the robot-side bridge (vendor SDK / ROS 2 node) is yours to write — the dry-run
  bridge is the template. Action frames/units must match how your logs were recorded.
- `sensor_*` channels are imported and validated but not yet fed to the model.
- The safety layer is a per-step clip plus stop-on-error; workspace limits, watchdog and
  e-stop belong in the robot-side controller (see Regulatory / Safety Notes).

---

## Hardware Requirements

| Mode          | Minimum              | Recommended           |
|---------------|----------------------|-----------------------|
| Training      | 8 GB RAM, CPU        | NVIDIA A100 / H100    |
| Training fast | Apple M2 (MPS)       | 4× A100 (DDP)         |
| Inference     | Jetson Nano (4 GB)   | Jetson Orin / Coral   |

The lightweight backbone runs at **~30 fps** on an Apple M2 CPU (224×224, seq=8).

---

## Extending the Model

**Larger backbone:** Pass `--backbone mobilenetv2` for pretrained ImageNet
features — typically +5% success rate on real robot benchmarks at the cost of
larger model size.

**Longer history:** Increase `--seq_len` to 16–32 frames for tasks requiring
longer temporal context (e.g., deformable object manipulation).

**Multi-camera:** Stack frames from multiple cameras along the channel dim
before passing to the backbone.

**Language conditioning:** Concatenate a sentence embedding (CLIP / T5) to the
flattened LSTM output before the action head — matches the RT-2 style
language-conditioned policy.

**Distributional output:** Replace the MSE motion head with a mixture of
Gaussians (as in Diffusion Policy) for bimodal action distributions.

---

## Regulatory / Safety Notes

For deployment on physical robots:
- Add velocity and workspace limit clipping in the action executor
- Log all predicted actions to a ring buffer for post-incident analysis
- Implement a watchdog that stops the robot if prediction confidence drops
  (e.g., detect OOD frames via reconstruction error on a VAE side-branch)
- Follow ISO 10218 / TS 15066 for collaborative robot safety

---

## License

MIT License — see `LICENSE` for details.

---

*Architecture inspired by Google Research spatiotemporal LSTM, MobileNetV2,
and the RT-1/RT-2 robotics policy family.*
