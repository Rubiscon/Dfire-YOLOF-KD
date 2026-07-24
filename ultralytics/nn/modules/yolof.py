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
    """Backbone dictionary module (CrisReport Fig. 2): student n10 ↔ teacher x^e (e.g. x6/x10).

    Matches every student channel (query) to teacher channels (keys), reorganizes the
    teacher feature as per-channel pseudo-GT, then projects the student for
    the proposal weighted-align loss (Fig. 2):

        key   K = flatten(avgpool(BN(Conv(x^e))))   (B, Ct, d)
        query Q = flatten(avgpool(BN(Conv(n))))     (B, Cs, d)
        M = Q K^T (B, Cs, Ct)
        hard (proposal): index = argmax(M, dim=Ct); x_org = gather(x^e, index)

    Pool target is H/4 × W/4 so d = (H/4)·(W/4) (proposal: resolution ↓16× in area).

    Matching modes (``dict_match``):
      - ``hard`` (proposal default): ``torch.max`` / argmax gather.
      - ``soft`` (ablation): softmax gather + commitment loss.

    Args:
        c_t, c_s: teacher / student channels.
        t_size, s_size: spatial size (H==W) at build time.
        grid: pooled token side length (proposal: ~H_e/4).
        match: ``hard`` or ``soft``.
        temperature: softmax temperature for soft matching.
    """

    def __init__(
        self,
        c_t: int,
        c_s: int,
        t_size: int,
        s_size: int,
        grid: int = 4,
        match: str = "hard",
        temperature: float = 0.07,
    ):
        super().__init__()
        # Proposal: Conv + BN, then average-pool (no stride-conv downsample).
        self.key_enc = nn.Sequential(
            nn.Conv2d(c_t, c_t, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_t),
        )
        self.query_enc = nn.Sequential(
            nn.Conv2d(c_s, c_s, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_s),
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = DeconvNet(c_s, c_s, s_size, t_size)
        self.match = str(match).lower()
        self.temperature = float(temperature)
        if self.match == "hard":
            # Hard argmax has no derivative with respect to Q/K, so randomly initialized
            # encoders would remain random forever. Start from an identity channel transform;
            # BatchNorm still calibrates the two feature distributions as stated in Fig. 2.
            self._init_identity_encoders()

    def _init_identity_encoders(self) -> None:
        """Initialize proposal Conv+BN encoders as channel-wise identity transforms."""
        for encoder in (self.key_enc, self.query_enc):
            conv, bn = encoder
            with torch.no_grad():
                conv.weight.zero_()
                channels = min(conv.in_channels, conv.out_channels)
                idx = torch.arange(channels, device=conv.weight.device)
                conv.weight[idx, idx, 1, 1] = 1.0
                bn.weight.fill_(1.0)
                bn.bias.zero_()
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0)

    def freeze_encoders(self) -> None:
        """Freeze hard-match encoder parameters; BN running statistics still calibrate."""
        for p in self.key_enc.parameters():
            p.requires_grad = False
        for p in self.query_enc.parameters():
            p.requires_grad = False

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor):
        """Return (s_proj, t_reorg, commit_loss).

        ``t_reorg`` is the dictionary-reorganized teacher feature (B, Cs, Ht, Wt).
        ``commit_loss`` is 0 for hard matching (proposal); soft match only otherwise.
        """
        _, _, h, w = t_feat.shape
        # Proposal: M = Q K^T without token L2-normalization.
        k = self.pool(self.key_enc(t_feat)).flatten(2)  # (B, Ct, d)
        q = self.pool(self.query_enc(s_feat)).flatten(2)  # (B, Cs, d)
        m = q @ k.transpose(1, 2)  # (B, Cs, Ct)
        commit = t_feat.new_zeros(())

        if self.match == "hard":
            # Proposal: index = torch.max(M, dim=1)[1]  (argmax over teacher channels).
            index = m.argmax(dim=2)  # (B, Cs)
            t_reorg = torch.gather(t_feat, 1, index[:, :, None, None].expand(-1, -1, h, w))
        else:
            w_soft = F.softmax(m / max(self.temperature, 1e-6), dim=2)  # (B, Cs, Ct)
            t_reorg = torch.einsum("bsc,bchw->bshw", w_soft, t_feat)
            k_n = F.normalize(k, dim=2)
            q_n = F.normalize(q, dim=2)
            k_hat = torch.einsum("bsc,bcd->bsd", w_soft.detach(), k_n.detach())
            commit = (1.0 - F.cosine_similarity(q_n, k_hat, dim=2)).mean()

        s_proj = self.proj(s_feat)
        if s_proj.shape[-2:] != t_feat.shape[-2:]:
            s_proj = F.interpolate(s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        return s_proj, t_reorg, commit
