"""
NetPulse — Temporal Convolutional Network (TCN)
================================================
Dilated causal convolutions for sequence modeling on IAT signals.
Architecture: Input projection → N residual blocks (exp. dilations) → classifier
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1-D convolution that cannot peek into the future."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=self.pad, dilation=dilation)
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad else out


class ResidualBlock(nn.Module):
    """Two causal convolutions + residual connection."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) + x


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for congestion prediction.

    Args:
        input_channels: number of input features per timestep (1 for raw IATs)
        num_classes: output classes (3 = green/yellow/red)
        hidden: hidden channel width
        kernel_size: convolution kernel size
        num_blocks: number of residual blocks (dilations = 2^0 … 2^(n-1))
        dropout: dropout probability
    """

    def __init__(self, input_channels=1, num_classes=3, hidden=64,
                 kernel_size=7, num_blocks=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden, 1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, kernel_size, 2 ** i, dropout)
              for i in range(num_blocks)]
        )
        self.classifier = nn.Linear(hidden, num_classes)
        self.receptive_field = kernel_size * (2 ** num_blocks)

    def forward(self, x):
        """x: (batch, 1, seq_len) → logits (batch, num_classes)"""
        h = self.input_proj(x)
        h = self.blocks(h)
        h = h[:, :, -1]          # take last timestep
        return self.classifier(h)

    def predict_proba(self, x):
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


def build_tcn(config=None):
    """Factory function with optional config dict override."""
    defaults = dict(input_channels=1, num_classes=3, hidden=64,
                    kernel_size=7, num_blocks=4, dropout=0.2)
    if config:
        defaults.update(config)
    return TCNModel(**defaults)
