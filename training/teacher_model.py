"""Teacher and Student Neural Network Architectures for Ultimate Tic-Tac-Toe.

Defines:
- Teacher50MNet: 13.27M parameter Deep ResNet with residual blocks and layer normalization.
- StudentNet: 296k parameter compact 2-layer MLP suitable for in-browser ONNX inference.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + res)

class Teacher50MNet(nn.Module):
    """Deep 13.27M Parameter ResNet Teacher for deep MCTS self-play exploration."""

    def __init__(self, in_features: int = 280):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
        )
        self.block1 = ResBlock(2048)
        self.transition = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )
        self.block2 = ResBlock(1024)

        self.policy_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 81),
        )
        self.value_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(state)
        x = self.block1(x)
        x = self.transition(x)
        x = self.block2(x)
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value

class StudentNet(nn.Module):
    """Compact 296k parameter network for ultra-fast browser inference (<1.5 MB ONNX)."""

    def __init__(self, in_features: int = 280, hidden_widths: Tuple[int, int] = (512, 256)):
        super().__init__()
        w1, w2 = hidden_widths
        self.trunk = nn.Sequential(
            nn.Linear(in_features, w1),
            nn.ReLU(),
            nn.Linear(w1, w2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(w2, 81)
        self.value_head = nn.Linear(w2, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy_logits, value
