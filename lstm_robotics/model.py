"""
VisuoMotor ConvLSTM — video-conditioned robot action policy.

Architecture (inspired by Google Research's spatiotemporal LSTM work):

  Video frames  →  CNN Backbone  →  ConvLSTM Stack  →  Temporal Pool
                                                            ↓
                                                      Action MLP  →  Robot Actions

References
----------
- Shi et al., "Convolutional LSTM Network" (NeurIPS 2015) — ConvLSTM cell design
- Google Research, "Video Prediction for Model-Based Deep RL" (2018)
- RT-1: Robotics Transformer (Google 2022) — action tokenisation inspiration
- Google Brain MobileNetV2 backbone option
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ── ConvLSTM cell ─────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al., NeurIPS 2015).

    Replaces the FC gates of a standard LSTM with 2-D convolutions so that
    spatial structure in the feature maps is preserved across timesteps.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # Four gates concatenated: input, forget, cell, output
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        # Initialise forget-gate bias to 1 to encourage remembering early on
        nn.init.ones_(self.gates.bias[self.hidden_channels: 2 * self.hidden_channels])

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_channels, H, W)
            c = x.new_zeros(B, self.hidden_channels, H, W)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


# ── Stacked ConvLSTM ──────────────────────────────────────────────────────────

class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM that processes a temporal sequence of feature maps."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        ch_in = in_channels
        for ch_out in hidden_channels:
            self.layers.append(ConvLSTMCell(ch_in, ch_out, kernel_size))
            ch_in = ch_out
        self.out_channels = hidden_channels[-1]
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, C, H, W) — batch of video sequences
            initial_states: optional list of (h, c) per layer

        Returns:
            out: (B, T, hidden[-1], H, W) — all hidden states
            final_states: list of (h, c) for each layer at last timestep
        """
        B, T, C, H, W = x.shape
        states = initial_states or [None] * len(self.layers)
        final_states = []

        # Run each layer over the full time axis
        layer_input = x  # (B, T, C, H, W)
        for layer, state in zip(self.layers, states):
            outputs = []
            for t in range(T):
                h, state = layer(layer_input[:, t], state)
                outputs.append(h)
            layer_input = self.dropout(
                torch.stack(outputs, dim=1)  # (B, T, C_out, H, W)
            )
            final_states.append(state)

        return layer_input, final_states  # layer_input is now (B, T, out_ch, H, W)


# ── Lightweight CNN backbone ──────────────────────────────────────────────────

class LightweightBackbone(nn.Module):
    """
    Four-stage depthwise-separable CNN backbone.
    Outputs a spatial feature map at 1/16 the input resolution.
    Inspired by MobileNet-style blocks used in Google's on-device robotics models.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()

        def dw_sep(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU6(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = dw_sep(32, 64, stride=2)
        self.stage2 = dw_sep(64, 128, stride=2)
        self.stage3 = dw_sep(128, out_channels, stride=2)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


# ── MobileNetV2 backbone (pretrained) ────────────────────────────────────────

class MobileNetV2Backbone(nn.Module):
    """Pretrained MobileNetV2 feature extractor (Google, 2018)."""

    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v2(weights=weights)
        # Use all features up to layer 14 (stride 16 total)
        self.features = nn.Sequential(*list(base.features)[:14])
        backbone_out = 96  # MobileNetV2 layer 14 output channels
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_out, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


# ── Action head ───────────────────────────────────────────────────────────────

class ActionHead(nn.Module):
    """
    MLP that maps a pooled spatiotemporal feature vector to robot actions.

    Output layout (action_dim=7):
      [0:3]  end-effector Δ(x, y, z)   — continuous, tanh-normalised
      [3:6]  end-effector Δ(roll, pitch, yaw) — continuous, tanh-normalised
      [6]    gripper command            — continuous 0→1 (0=open, 1=close)
    """

    def __init__(self, in_features: int, hidden_dim: int = 256, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim
        self.gripper_idx = action_dim - 1

        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        # Separate heads so gripper loss can be weighted differently
        self.motion_head = nn.Linear(hidden_dim // 2, action_dim - 1)
        self.gripper_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        motion = torch.tanh(self.motion_head(h))  # bounded [-1, 1]
        gripper = torch.sigmoid(self.gripper_head(h))  # bounded [0, 1]
        return torch.cat([motion, gripper], dim=-1)


# ── Full model ────────────────────────────────────────────────────────────────

class VisuoMotorLSTM(nn.Module):
    """
    Video-conditioned robot action policy.

    Encodes each frame with a CNN backbone, passes the spatial feature
    sequence through a stacked ConvLSTM, temporally pools the output, then
    decodes to robot actions via an MLP head.

    Input  : (B, T, C, H, W)  — batch of video sequences
    Output : (B, T, action_dim) — action prediction at every timestep
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Backbone ────────────────────────────────────────────────
        if cfg.backbone == "mobilenetv2":
            self.backbone = MobileNetV2Backbone(
                out_channels=cfg.backbone_out_channels
            )
        else:
            self.backbone = LightweightBackbone(
                in_channels=cfg.image_channels,
                out_channels=cfg.backbone_out_channels,
            )

        # ── ConvLSTM ────────────────────────────────────────────────
        self.convlstm = ConvLSTM(
            in_channels=cfg.backbone_out_channels,
            hidden_channels=cfg.convlstm_hidden_channels,
            kernel_size=cfg.convlstm_kernel_size,
            dropout=cfg.dropout,
        )

        # Spatial feature map size after backbone at 1/16 resolution
        feat_h = cfg.image_height // 16
        feat_w = cfg.image_width // 16
        pooled_dim = self.convlstm.out_channels * feat_h * feat_w

        # ── Temporal context aggregation ────────────────────────────
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.convlstm.out_channels, 1),
            nn.Flatten(start_dim=-2),
        )

        # ── Action head ─────────────────────────────────────────────
        self.action_head = ActionHead(
            in_features=pooled_dim,
            hidden_dim=cfg.action_hidden_dim,
            action_dim=cfg.action_dim,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply backbone to each frame independently.

        Args:
            frames: (B, T, C, H, W)
        Returns:
            features: (B, T, backbone_out, H', W')
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        feats = self.backbone(flat)
        _, Cf, Hf, Wf = feats.shape
        return feats.view(B, T, Cf, Hf, Wf)

    def forward(
        self,
        frames: torch.Tensor,
        states: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Args:
            frames : (B, T, C, H, W)
            states : optional initial ConvLSTM states (for online inference)

        Returns:
            actions      : (B, T, action_dim)
            final_states : ConvLSTM hidden states at t=T (for streaming)
        """
        # Spatial encoding — backbone shared across time
        features = self.encode_frames(frames)            # (B, T, Cf, Hf, Wf)

        # Spatiotemporal modelling
        lstm_out, final_states = self.convlstm(features, states)
        # lstm_out: (B, T, Ch, Hf, Wf)

        B, T, Ch, Hf, Wf = lstm_out.shape

        # Flatten spatial dims → (B, T, Ch*Hf*Wf)
        pooled = lstm_out.view(B, T, Ch * Hf * Wf)

        # Action prediction at every timestep
        actions = self.action_head(pooled)               # (B, T, action_dim)

        return actions, final_states

    @torch.no_grad()
    def predict_one(
        self,
        frame: torch.Tensor,
        states: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List]:
        """Online single-frame inference for deployment.

        Args:
            frame  : (C, H, W) — single camera frame (no batch/time dims)
            states : previous ConvLSTM hidden states (None for first frame)

        Returns:
            action       : (action_dim,) tensor
            final_states : updated states for next call
        """
        self.eval()
        # frame: (C, H, W) → (1, 1, C, H, W) for model forward
        if frame.dim() == 3:
            x = frame.unsqueeze(0).unsqueeze(0)
        else:
            x = frame.unsqueeze(0)  # assume (1, C, H, W) → (1, 1, C, H, W) if needed
        actions, final_states = self.forward(x, states)
        return actions[0, 0], final_states

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: Optional[ModelConfig] = None) -> VisuoMotorLSTM:
    if cfg is None:
        cfg = ModelConfig()
    return VisuoMotorLSTM(cfg)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[VisuoMotorLSTM, dict]:
    """Load model + metadata from a checkpoint file (returned model is in eval mode).

    Uses ``torch.load(weights_only=True)``: only tensors, primitives and ModelConfig are
    accepted, so a tampered checkpoint cannot execute code on load.
    """
    import dataclasses
    from pathlib import Path

    if not Path(path).exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with torch.serialization.safe_globals([ModelConfig]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a VisuoMotorLSTM checkpoint (no model_state_dict)")
    cfg = ckpt.get("model_config", ModelConfig())
    if isinstance(cfg, dict):
        known = {f.name for f in dataclasses.fields(ModelConfig)}
        cfg = ModelConfig(**{k: v for k, v in cfg.items() if k in known})
    ckpt["model_config"] = cfg
    ckpt.setdefault("metadata", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
