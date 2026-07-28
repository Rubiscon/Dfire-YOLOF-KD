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
        match_norm: str = "l2",
        match_init: str = "default",
        infomax_marginal_weight: float = 1.0,
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
        match_aliases = {"st": "straight_through", "ste": "straight_through", "hard_st": "straight_through"}
        self.match = match_aliases.get(str(match).lower(), str(match).lower())
        if self.match not in {"hard", "soft", "straight_through"}:
            raise ValueError(f"Unknown dictionary match={match!r}; expected hard, soft, or straight_through")
        self.temperature = float(temperature)
        self.match_norm = str(match_norm).lower()
        if self.match_norm not in {"l2", "none"}:
            raise ValueError(f"Unknown dictionary match_norm={match_norm!r}; expected l2 or none")
        self.match_init = str(match_init).lower()
        self.infomax_marginal_weight = float(infomax_marginal_weight)
        self.last_match_stats = None
        self._previous_dominant_assignment = None
        if self.match_init == "identity":
            self._init_identity_encoders()
        elif self.match_init not in {"default", "random", "kaiming"}:
            raise ValueError(f"Unknown dictionary match_init={match_init!r}; expected default or identity")

    def _init_identity_encoders(self) -> None:
        """Start Q/K projections as channel-wise identities to avoid arbitrary initial permutations."""
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
        """Freeze key/query encoders (use after init when ``match=hard``)."""
        for p in self.key_enc.parameters():
            p.requires_grad = False
        for p in self.query_enc.parameters():
            p.requires_grad = False
        self.key_enc.eval()
        self.query_enc.eval()

    @property
    def differentiable_assignment(self) -> bool:
        """Whether assignment-specific losses can update Q/K and the student query path."""
        return self.match in {"soft", "straight_through"}

    @staticmethod
    def infomax_loss(assignment: torch.Tensor, marginal_weight: float = 1.0):
        """Return H(T|S)-lambda*H(T), conditional entropy, and marginal entropy."""
        eps = torch.finfo(assignment.dtype).eps
        conditional_entropy = -(assignment * assignment.clamp_min(eps).log()).sum(dim=2).mean()
        teacher_marginal = assignment.mean(dim=1)  # (B, Ct), 1/Cs sum_s A_st
        marginal_entropy = -(
            teacher_marginal * teacher_marginal.clamp_min(eps).log()
        ).sum(dim=1).mean()
        return (
            conditional_entropy - float(marginal_weight) * marginal_entropy,
            conditional_entropy,
            marginal_entropy,
        )

    @staticmethod
    def spatial_entropy_weight(
        query_feat: torch.Tensor,
        value_feat: torch.Tensor,
        grid_size: tuple[int, int],
        temperature: float = 0.1,
        floor: float = 0.1,
        inverse: bool = False,
        return_entropy: bool = False,
    ):
        """Build a positive spatial weight from cross-feature attention entropy.

        The independently projected student and teacher features are pooled into
        spatial tokens ``Q`` and ``V``. With ``Nq`` query and ``Nv`` value tokens:

            A = softmax(Q V^T / temperature, dim=2),  A: (B, Nq, Nv)
            H_i = -sum_j A_ij log(A_ij),              H: (B, Nq)

        ``dim=2`` is the row-normalization and row-entropy axis for the batched
        matrix. This is the positive-entropy correction of the mentor's
        double-negative expression. Dividing by ``log(Nv)`` keeps the map in
        [0, 1] across grid sizes; ``floor`` prevents confident rows from turning
        off align supervision entirely. ``inverse=True`` is an opt-in confidence
        weighting ablation. The returned map (and optional normalized row-entropy
        map) is always detached so align cannot optimize its own spatial weights.
        """
        if query_feat.ndim != 4 or value_feat.ndim != 4:
            raise ValueError("Entropy weighting expects BCHW query/value features")
        if query_feat.shape[0] != value_feat.shape[0] or query_feat.shape[1] != value_feat.shape[1]:
            raise ValueError(
                f"Entropy query/value must share batch and channel dimensions, got "
                f"{tuple(query_feat.shape)} and {tuple(value_feat.shape)}"
            )
        temperature = max(float(temperature), 1e-6)
        floor = min(max(float(floor), 0.0), 1.0)
        gh, gw = max(int(grid_size[0]), 1), max(int(grid_size[1]), 1)

        q = F.adaptive_avg_pool2d(query_feat.float(), (gh, gw)).flatten(2).transpose(1, 2)
        v = F.adaptive_avg_pool2d(value_feat.float(), (gh, gw)).flatten(2).transpose(1, 2)
        q = F.normalize(q, dim=2)
        v = F.normalize(v, dim=2)
        attention = F.softmax((q @ v.transpose(1, 2)) / temperature, dim=2)
        entropy = -(attention * attention.clamp_min(1e-10).log()).sum(dim=2)

        num_values = attention.shape[2]
        if num_values > 1:
            entropy = entropy / math.log(num_values)
        else:
            entropy = torch.ones_like(entropy)
        entropy = entropy.clamp(0.0, 1.0)
        weight_signal = 1.0 - entropy if inverse else entropy
        weight = floor + (1.0 - floor) * weight_signal
        weight = weight.reshape(query_feat.shape[0], 1, gh, gw).detach()
        entropy = entropy.reshape(query_feat.shape[0], 1, gh, gw).detach()
        return (weight, entropy) if return_entropy else weight

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor, collect_diagnostics: bool = False):
        """Return (s_proj, t_reorg, commit_loss, infomax_loss).

        ``t_reorg`` is the dictionary-reorganized teacher feature (B, Cs, Ht, Wt).
        ``commit_loss`` pulls query tokens toward their soft-matched keys so encoders
        learn under stopgrad(teacher) distillation (0 for hard matching).
        ``infomax_loss`` is H(T|S)-lambda*H(T), computed from the soft assignment.

        Straight-through matching has an exactly hard forward assignment while its
        backward pass follows the soft assignment. Callers detach ``t_reorg`` for
        align/AT; commit and InfoMax are the intended Q/K training signals.
        """
        _, _, h, w = t_feat.shape
        k = self.pool(self.key_enc(t_feat)).flatten(2)  # (B, Ct, d)
        q = self.pool(self.query_enc(s_feat)).flatten(2)  # (B, Cs, d)
        if self.match_norm == "l2":
            k = F.normalize(k, dim=2)
            q = F.normalize(q, dim=2)
        m = q @ k.transpose(1, 2)  # (B, Cs, Ct)
        commit = t_feat.new_zeros(())
        infomax = t_feat.new_zeros(())
        assignment_soft = F.softmax(m.float() / max(self.temperature, 1e-6), dim=2)
        assignment_index = assignment_soft.argmax(dim=2)

        if self.match == "hard":
            t_reorg = torch.gather(
                t_feat, 1, assignment_index[:, :, None, None].expand(-1, -1, h, w)
            )
        else:
            assignment = assignment_soft
            if self.match == "straight_through":
                assignment_hard = F.one_hot(assignment_index, num_classes=m.shape[2]).to(assignment.dtype)
                assignment = assignment_hard + assignment - assignment.detach()
            t_reorg = torch.einsum("bsc,bchw->bshw", assignment.to(t_feat.dtype), t_feat)
            # Commitment: queries should agree with the teacher keys they attend to.
            k_n = F.normalize(k, dim=2)
            q_n = F.normalize(q, dim=2)
            k_hat = torch.einsum("bsc,bcd->bsd", assignment_soft.detach(), k_n.detach())
            commit = (1.0 - F.cosine_similarity(q_n, k_hat, dim=2)).mean()

            infomax, conditional_entropy, marginal_entropy = self.infomax_loss(
                assignment_soft, self.infomax_marginal_weight
            )

        if collect_diagnostics:
            with torch.no_grad():
                counts = F.one_hot(assignment_index, num_classes=m.shape[2]).sum(dim=(0, 1)).float()
                top2 = assignment_soft.topk(min(2, assignment_soft.shape[2]), dim=2).values
                margin = (
                    (top2[:, :, 0] - top2[:, :, 1]).mean()
                    if top2.shape[2] > 1
                    else assignment_soft.new_zeros(())
                )
                dominant = assignment_index.mode(dim=0).values
                churn = assignment_soft.new_zeros(())
                if (
                    self._previous_dominant_assignment is not None
                    and self._previous_dominant_assignment.shape == dominant.shape
                ):
                    churn = (dominant != self._previous_dominant_assignment.to(dominant.device)).float().mean()
                self._previous_dominant_assignment = dominant.detach().cpu()
                _, conditional_entropy, marginal_entropy = self.infomax_loss(
                    assignment_soft, self.infomax_marginal_weight
                )
                self.last_match_stats = {
                    "used_teacher_ratio": (counts > 0).float().mean().detach(),
                    "max_teacher_share": (counts.max() / counts.sum().clamp_min(1.0)).detach(),
                    "match_margin": margin.detach(),
                    "assignment_churn": churn.detach(),
                    "channel_conditional_entropy": conditional_entropy.detach(),
                    "channel_marginal_entropy": marginal_entropy.detach(),
                    "channel_effective_teacher_channels": marginal_entropy.exp().detach(),
                    "channel_infomax_loss": infomax.detach(),
                }
        else:
            self.last_match_stats = None

        s_proj = self.proj(s_feat)
        if s_proj.shape[-2:] != t_feat.shape[-2:]:  # multi-scale / rect batches
            s_proj = F.interpolate(s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
        return s_proj, t_reorg, commit, infomax
