# ultralytics/nn/modules/yolof.py
"""YOLOF dilated residual block"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

class DilatedResBlock(nn.Module):
    def __init__(self, c1, c2, d=1):
        super().__init__()
        c = c1
        c_mid = c // 2
        self.cv1 = Conv(c, c_mid, 1, 1)
        self.cv2 = Conv(c_mid, c_mid, 3, 1, d=d)
        self.cv3 = Conv(c_mid, c, 1, 1)
        self.last_feature = None

    def forward(self, x):
        x = x + self.cv3(self.cv2(self.cv1(x)))
        self.last_feature = x
        return x

    def __deepcopy__(self, memo):
        # last_feature caches a non-leaf graph tensor for KD; null it so deepcopy
        # (ModelEMA init / checkpoint saving) doesn't fail on non-leaf tensors.
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = None if k == "last_feature" else copy.deepcopy(v, memo)
        return new


class FeatureProjector(nn.Module):
    """Project YOLOF features to teacher FPN feature dimensions and spatial sizes."""

    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, 1, 1)
        self.out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)

    def forward(self, x):
        x = self.conv(x)
        if x.shape[-2:] != tuple(self.out_size):
            x = nn.functional.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class DeconvNet(nn.Module):
    """Proposal "project module": learned upsampling via stacked ConvTranspose2d.

    Aligns a YOLOF feature map (in_channel, in_size) to a teacher FPN level
    (out_channel, out_size). Each transpose-conv layer doubles the spatial size;
    when no upsampling is needed (out_size == in_size) it degenerates to a 1x1
    channel projection. A final bilinear resize guards against non-power-of-2
    scale factors so the output always matches the teacher spatially.

    Args:
        in_channel (int): channels of the YOLOF feature.
        out_channel (int): channels of the target FPN feature.
        in_size (int): spatial size (H==W) of the YOLOF feature.
        out_size (int): spatial size (H==W) of the target FPN feature.
    """

    def __init__(self, in_channel, out_channel, in_size, out_size):
        super().__init__()
        in_size = int(in_size[0] if isinstance(in_size, (tuple, list)) else in_size)
        out_size = int(out_size[0] if isinstance(out_size, (tuple, list)) else out_size)
        self.out_size = (out_size, out_size)

        scale = max(out_size // max(in_size, 1), 1)
        num_up = max(int(round(math.log2(scale))), 0)
        hidden_channel = max(in_channel, out_channel, 64)

        layers = []
        if num_up == 0:
            # No spatial change needed: project channels only.
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=1))
        else:
            c_in = in_channel
            for k in range(num_up):
                last = k == num_up - 1
                c_out = out_channel if last else hidden_channel
                layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2))
                if not last:
                    layers.append(nn.BatchNorm2d(c_out))
                    layers.append(nn.ReLU(inplace=True))
                c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        if x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class DictionaryModule(nn.Module):
    """Proposal 'dictionary module' for backbone distillation (CrisReport fig. 2).

    Matches every student backbone channel (query) to its closest teacher early-feature
    channel (key) via a correlation matrix of pooled channel tokens, then reorganizes the
    teacher feature by the matched index so it can serve as a per-channel pseudo ground
    truth for the student feature:

        key   K = flatten(avgpool(BN(Conv(x_t))))   (B, Ct, d)
        query Q = flatten(avgpool(BN(Conv(n_s))))   (B, Cs, d)
        M = Q K^T (B, Cs, Ct);  index = M.argmax(dim=2)
        x_c = x_t[index]                            (B, Cs, Ht, Wt)

    The student feature is projected (DeconvNet) to the same channel/spatial size as
    ``x_c`` for the weighted align loss and the attention restriction loss. Note the
    index selection is non-differentiable (hard argmax per the proposal), so the key /
    query encoders act as fixed random projections; gradients flow to the student only
    through the projector.

    Args:
        c_t (int): teacher feature channels.
        c_s (int): student feature channels.
        t_size (int): teacher feature spatial size (H == W) at trace time.
        s_size (int): student feature spatial size (H == W) at trace time.
        grid (int): pooled token grid; token dim d = grid * grid.
    """

    def __init__(self, c_t: int, c_s: int, t_size: int, s_size: int, grid: int = 4):
        super().__init__()
        # Conv + BN eliminate the feature distribution discrepancy before pooling (proposal III.a).
        self.key_enc = nn.Sequential(
            nn.Conv2d(c_t, c_t, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_t),
        )
        self.query_enc = nn.Sequential(
            nn.Conv2d(c_s, c_s, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_s),
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = DeconvNet(c_s, c_s, s_size, t_size)

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor):
        """Return (projected student feature, reorganized teacher feature), both (B, Cs, Ht, Wt)."""
        b, _, h, w = t_feat.shape
        k = self.pool(self.key_enc(t_feat)).flatten(2)  # (B, Ct, d)
        q = self.pool(self.query_enc(s_feat)).flatten(2)  # (B, Cs, d)
        m = q @ k.transpose(1, 2)  # correlation matrix (B, Cs, Ct)
        index = m.argmax(dim=2)  # closest teacher channel per student channel (B, Cs)
        t_reorg = torch.gather(t_feat, 1, index[:, :, None, None].expand(-1, -1, h, w))
        s_proj = self.proj(s_feat)
        if s_proj.shape[-2:] != t_feat.shape[-2:]:  # multi-scale / rect batches
            s_proj = F.interpolate(s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        return s_proj, t_reorg
