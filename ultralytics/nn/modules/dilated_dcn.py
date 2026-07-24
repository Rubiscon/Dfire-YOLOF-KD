# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dilated + Deformable block aligned to CrisReport Figure 3 (Dilated Deformable Block).

Figure 3 structure:
  Conv 1x1 → DeDi(x) 3x3 → (+ residual around DeDi) → Conv 1x1
  DeDi(x) = DeformConv 3x3 (dilated ratio x) → BatchNorm2d → Activation

Falls back to torchvision DeformConv2d or regular dilated conv when DCNv4 is absent.

Signature compatible with parse_model: DilatedDeformBlock(c1, c2, d=1, mid_factor=2, use_dcn=True)
"""
import copy

import torch
import torch.nn as nn

from .conv import Conv


try:
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
        """Proposal Fig. 3 dilated deformable residual block.

        Args:
            c1: input channels
            c2: output channels
            d: dilation rate (``dilated ratio x``)
            mid_factor: bottleneck width = c2 // mid_factor
            use_dcn: use deformable conv when available
        """
        super().__init__()
        mid = max(1, c2 // max(1, mid_factor))
        self.cv1 = Conv(c1, mid, k=1, s=1, p=0)  # first 1x1
        self.d = int(d)
        self.use_dcn = bool(use_dcn) and DCN_AVAILABLE
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)

        if self.use_dcn:
            self.offset = nn.Conv2d(mid, 2 * 3 * 3, kernel_size=3, padding=self.d, dilation=self.d)
            self.dcn = DeformConv2d(mid, mid, kernel_size=3, padding=self.d, dilation=self.d)
            nn.init.constant_(self.offset.weight, 0.0)
            nn.init.constant_(self.offset.bias, 0.0)
            self.conv_dil = None
        else:
            self.offset = None
            self.dcn = None
            self.conv_dil = nn.Conv2d(mid, mid, kernel_size=3, padding=self.d, dilation=self.d, bias=False)

        self.cv3 = Conv(mid, c2, k=1, s=1, p=0)  # final 1x1
        self.last_feature = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fig. 3: 1x1 → DeDi → (+ skip around DeDi) → 1x1
        y = self.cv1(x)
        if self.use_dcn:
            off = self.offset(y)
            m = self.dcn(y, off)
        else:
            m = self.conv_dil(y)
        m = self.act(self.bn(m))
        y = y + m
        out = self.cv3(y)
        self.last_feature = out
        return out

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = None if k == "last_feature" else copy.deepcopy(v, memo)
        return new


__all__ = ["DilatedDeformBlock"]
