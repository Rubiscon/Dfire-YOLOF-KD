# ultralytics/nn/modules/yolof.py
"""YOLOF dilated residual block and dictionary distillation modules."""
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
    """Early-stage backbone dictionary module (student n10 ↔ teacher x6 / x10).

    Matches every student backbone channel (query) to teacher early-feature channels
    (keys) via a correlation matrix of pooled channel tokens, then reorganizes the
    teacher feature so it can serve as a per-channel pseudo ground truth:

        key   K = flatten(avgpool(BN(Conv(x_t))))   (B, Ct, d)
        query Q = flatten(avgpool(BN(Conv(n_s))))   (B, Cs, d)
        M = Q K^T (B, Cs, Ct)

    Matching modes (``dict_match``):
      - ``soft`` (default): M → softmax → soft channel gather (differentiable cross-attention;
        key/query encoders and the student backbone receive gradients — closer to the
        proposal's mutual-information / cross-attention intent).
      - ``hard``: index = argmax(M); non-differentiable gather (legacy). Encoders act as
        fixed projections; freeze their params after init when hard is selected upstream.

    The student feature is projected (DeconvNet) to the same channel/spatial size as
    the reorganized teacher feature for weighted align + attention restriction losses.

    Args:
        c_t (int): teacher feature channels.
        c_s (int): student feature channels.
        t_size (int): teacher feature spatial size (H == W) at trace time.
        s_size (int): student feature spatial size (H == W) at trace time.
        grid (int): pooled token grid; token dim d = grid * grid.
        match (str): ``soft`` or ``hard``.
        temperature (float): softmax temperature for soft matching.
    """

    def __init__(
        self,
        c_t: int,
        c_s: int,
        t_size: int,
        s_size: int,
        grid: int = 4,
        match: str = "soft",
        temperature: float = 0.07,
    ):
        super().__init__()
        # Teacher key path: Conv+BN then pool to ~1/16 spatial size (proposal early-feature encoder).
        self.key_enc = nn.Sequential(
            nn.Conv2d(c_t, c_t, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_t),
        )
        # Student query path: same encoder family without downsampling conv stride.
        self.query_enc = nn.Sequential(
            nn.Conv2d(c_s, c_s, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_s),
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = DeconvNet(c_s, c_s, s_size, t_size)
        self.match = str(match).lower()
        self.temperature = float(temperature)

    def freeze_encoders(self) -> None:
        """Freeze key/query encoders (use after init when ``match=hard``)."""
        for p in self.key_enc.parameters():
            p.requires_grad = False
        for p in self.query_enc.parameters():
            p.requires_grad = False
        self.key_enc.eval()
        self.query_enc.eval()

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor):
        """Return (s_proj, t_reorg, commit_loss).

        ``t_reorg`` is the dictionary-reorganized teacher feature (B, Cs, Ht, Wt).
        ``commit_loss`` pulls query tokens toward their soft-matched keys so encoders
        learn under stopgrad(teacher) distillation (0 for hard matching).
        """
        _, _, h, w = t_feat.shape
        k = self.pool(self.key_enc(t_feat)).flatten(2)  # (B, Ct, d)
        q = self.pool(self.query_enc(s_feat)).flatten(2)  # (B, Cs, d)
        # Normalize tokens so correlation is cosine-like (stable soft matching).
        k = F.normalize(k, dim=2)
        q = F.normalize(q, dim=2)
        m = q @ k.transpose(1, 2)  # (B, Cs, Ct)
        commit = t_feat.new_zeros(())

        if self.match == "hard":
            index = m.argmax(dim=2)  # (B, Cs)
            t_reorg = torch.gather(t_feat, 1, index[:, :, None, None].expand(-1, -1, h, w))
        else:
            # Soft cross-attention gather: each student channel is a mixture of teacher channels.
            w_soft = F.softmax(m / max(self.temperature, 1e-6), dim=2)  # (B, Cs, Ct)
            t_reorg = torch.einsum("bsc,bchw->bshw", w_soft, t_feat)
            # Commitment: queries should agree with the teacher keys they attend to.
            k_hat = torch.einsum("bsc,bcd->bsd", w_soft.detach(), k.detach())
            commit = (1.0 - F.cosine_similarity(q, k_hat, dim=2)).mean()

        s_proj = self.proj(s_feat)
        if s_proj.shape[-2:] != t_feat.shape[-2:]:  # multi-scale / rect batches
            s_proj = F.interpolate(s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        return s_proj, t_reorg, commit
