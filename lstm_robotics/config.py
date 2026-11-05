"""Training and model hyperparameters."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    # Input
    image_height: int = 224
    image_width: int = 224
    image_channels: int = 3
    sequence_length: int = 8          # number of frames per sequence

    # CNN backbone
    backbone: str = "lightweight"     # "lightweight" | "mobilenetv2"
    backbone_out_channels: int = 256  # feature map channels after backbone

    # ConvLSTM stack
    convlstm_hidden_channels: List[int] = field(default_factory=lambda: [256, 128, 64])
    convlstm_kernel_size: int = 3

    # Action head
    action_dim: int = 7               # [x, y, z, roll, pitch, yaw, gripper]
    action_hidden_dim: int = 256

    # Dropout
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # Data — exactly one source is used, in this priority:
    #   data_csv (trajectory log, see lstm_robotics/logs.py) > data_dir (episode dir / .h5)
    #   > synthetic demo (only when use_synthetic=True)
    data_dir: str = "data/demo"
    data_csv: str = ""
    use_synthetic: bool = False
    synthetic_episodes: int = 200
    # Per-axis physical scale for dx..dyaw (CSV logs only). None = auto from the data.
    action_scale: Optional[List[float]] = None
    val_split: float = 0.15
    test_split: float = 0.10

    # Training
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_epochs: int = 3

    # Scheduler
    scheduler: str = "cosine"        # "cosine" | "step" | "none"
    lr_step_size: int = 20
    lr_gamma: float = 0.5

    # Loss weights
    action_loss_weight: float = 1.0
    gripper_loss_weight: float = 2.0  # gripper accuracy is critical

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    keep_best: int = 3

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "auto"             # "auto" | "cuda" | "mps" | "cpu"
    num_workers: int = 4
    pin_memory: bool = True
