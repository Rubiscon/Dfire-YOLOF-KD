"""Dilated + Deformable Conv (DCN) blocks for YOLOF-style heads.

This module provides a compact implementation that first applies a dilated
conv to provide a large receptive field, then refines features using a
DeformConv2d from torchvision. Offsets are predicted from the intermediate
features with a small conv. The block aims to combine dilation priors with
deformable fine-tuning of sampling locations.

Note: requires torchvision.ops.DeformConv2d (torchvision >= 0.8+).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .conv import Conv

try:
    from torchvision.ops import DeformConv2d
except Exception:  # pragma: no cover - runtime dependency
    DeformConv2d = None


class DilatedDCNBlock(nn.Module):
    """Dilated Residual Block followed by a Deformable Conv refinement.

    Args:
        c1: input channels
        c2: mainly ignored (kept for compatibility with existing blocks)
        d: dilation rate for the dilated conv
    """

    def __init__(self, c1, c2, d=1):
        super().__init__()
        if DeformConv2d is None:
            raise ImportError("DeformConv2d not available. Please install torchvision with ops support.")

        c = c1
        c_mid = c // 2

        # Dilated convolution to provide a large receptive field prior
        self.cv1 = Conv(c, c_mid, k=1, s=1)
        self.cv2 = Conv(c_mid, c_mid, k=3, s=1, d=d)

        # Offsets predictor for DeformConv2d: 2 * kH * kW
        self.offset_conv = nn.Conv2d(c_mid, 2 * 3 * 3, kernel_size=3, padding=1)

        # Deformable conv refines locations (output channels back to c)
        self.dcn = DeformConv2d(c_mid, c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.last_feature = None  # cached residual output for knowledge distillation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.cv2(y)
        # predict offsets and apply deform conv
        offsets = self.offset_conv(y)
        y = self.dcn(y, offsets)
        y = self.act(self.bn(y))
        out = x + y
        self.last_feature = out
        return out

    def __deepcopy__(self, memo):
        # last_feature caches a non-leaf graph tensor for KD; null it so deepcopy
        # (ModelEMA init / checkpoint saving) doesn't fail on non-leaf tensors.
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = None if k == "last_feature" else copy.deepcopy(v, memo)
        return new
