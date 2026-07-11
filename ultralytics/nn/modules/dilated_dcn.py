# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dilated + Deformable convolutional block: combines a dilation prior with a deformable
convolution refinement. Falls back to torchvision DeformConv2d or regular dilated conv.

Signature compatible with parse_model: DilatedDeformBlock(c1, c2, d=1, mid_factor=2, use_dcn=True)
"""
import copy
from typing import Optional

import torch
import torch.nn as nn

from .conv import Conv


try:
    # Prefer local DCNv4 implementation if present (user-provided)
    from dcn_v4 import DeformConv2d  # type: ignore
    DCN_AVAILABLE = True
except Exception:
    try:
        from torchvision.ops import DeformConv2d  # type: ignore

        DCN_AVAILABLE = True
    except Exception:
        DCN_AVAILABLE = False


class DilatedDeformBlock(nn.Module):
    def __init__(self, c1: int, c2: int, d: int = 1, mid_factor: int = 2, use_dcn: bool = True):
        """Dilated conv + optional DCN refinement block.

        Args:
            c1: input channels
            c2: output channels
            d: dilation rate
            mid_factor: factor to reduce channels in bottleneck (mid = c2 // mid_factor)
            use_dcn: whether to use deformable conv (falls back if not available)
        """
        super().__init__()
        mid = max(1, c2 // max(1, mid_factor))
        self.cv1 = Conv(c1, mid, k=1, s=1, p=0)  # reduce
        self.d = int(d)
        self.use_dcn = bool(use_dcn) and DCN_AVAILABLE

        if self.use_dcn:
            # offset conv predicts 2*k*k offsets (k=3)
            self.offset = nn.Conv2d(mid, 2 * 3 * 3, kernel_size=3, padding=self.d, dilation=self.d)
            # Deformable conv (padding respects dilation)
            self.dcn = DeformConv2d(mid, mid, kernel_size=3, padding=self.d, dilation=self.d)
            nn.init.constant_(self.offset.weight, 0.0)
            nn.init.constant_(self.offset.bias, 0.0)
        else:
            # fallback to regular dilated conv
            self.conv_dil = nn.Conv2d(mid, mid, kernel_size=3, padding=self.d, dilation=self.d, bias=False)
            self.bn = nn.BatchNorm2d(mid)
            self.act = nn.ReLU(inplace=True)

        self.cv3 = Conv(mid, c2, k=1, s=1, p=0)  # expand
        self.last_feature = None  # cached residual output for knowledge distillation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        if self.use_dcn:
            off = self.offset(y)
            y = self.dcn(y, off)
        else:
            y = self.conv_dil(y)
            y = self.bn(y)
            y = self.act(y)
        y = self.cv3(y)
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


__all__ = ["DilatedDeformBlock"]
