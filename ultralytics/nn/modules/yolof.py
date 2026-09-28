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


# Dictionary encoder forms. ``full`` is the original 3x3 Conv+BN; the depthwise
# forms follow MobileNetV2: ``depthwise`` is a per-channel 3x3 with no cross-channel
# mixing, ``depthwise_separable`` appends the 1x1 pointwise that restores it.
# Depthwise forms require in_channels == out_channels.
_ENCODER_FORMS = ("full", "depthwise", "depthwise_separable")


class ChannelIdentityProjection(nn.Module):
    """Parameter-free, channel-local projection: output channel c' is student channel c'.

    The shipped ``DeconvNet`` projection mixes all student channels into every output
    channel, and its first layer is a ``ConvTranspose2d(Cs, Cs, k=2, s=2)`` whose output
    channel c' has its OWN weight vector. Because of that, reassigning the target of
    student channel c' from one teacher channel to another only relabels a set of
    independent sub-problems: measured by fitting the projection to four different target
    maps, the per-channel loss multiset for a map and for that map with the student
    channels permuted agreed to 0.0006, while changing which teacher channels are used at
    all was worth 0.04. In other words the learned projection ABSORBS the pairing, so the
    correspondence cannot influence the objective no matter how accurate it is.

    This projection removes that escape route: it has no parameters, so output channel c'
    can only ever be student channel c', and the assignment therefore decides which teacher
    channel that student channel is aligned to. The only operation performed is a spatial
    resize to the teacher tap's resolution. Measured on real features, the pairing is then
    worth 13.4% of the align loss instead of 0.03%.
    """

    def __init__(self, out_size):
        super().__init__()
        size = int(out_size[0] if isinstance(out_size, (tuple, list)) else out_size)
        self.out_size = (size, size)

    def forward(self, x):
        if x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class ChannelLocalProjection(nn.Module):
    """Channel-local projection WITH parameters: a depthwise conv, identity-initialised.

    Purpose. The parameter-free ``ChannelIdentityProjection`` above keeps the pairing visible
    (13.4% of the align loss instead of 0.03%) but gives the model nothing to learn, so the
    pairing has to carry the whole benefit by itself. This form keeps the pairing visible
    while restoring a small amount of learnable freedom -- one spatial filter per channel.

    Why depthwise (``groups=C``). Channel-locality is what keeps the pairing visible, and
    ``groups=C`` guarantees it *structurally*: output channel c depends only on input channel
    c, whatever the weights are. A mixing projection cannot preserve the pairing even in
    principle, and a plain ``Conv2d(C, C, 1)`` (per-channel affine) does not work either --
    see below.

    Why a k x k kernel and not a per-channel affine. The align loss standardises each channel
    over space (zero mean, unit variance). A per-channel affine ``a*S + b`` is EXACTLY
    cancelled by that step -- ``standardize(a*S + b) = sign(a)*standardize(S)`` -- so its
    parameters receive no gradient (measured: 1.4e-05 against 2.0e-02 for the k=3 form, and
    the residual is pure ``eps`` leakage). Such a projection would silently sit at its
    initialisation for the whole run and measure nothing. A k x k depthwise filter is not
    cancelled, because standardisation removes only the per-channel mean and scale, not the
    spatial reweighting.

    Initialisation is the identity (centre tap = 1, bias = 0), so training starts from the
    configuration whose behaviour is already known, and note that the identity form passes
    about 4.7x more align gradient to the student tap than ``DeconvNet`` does at the same
    ``dict_align_loss`` -- the align weight may need to come down accordingly.
    """

    def __init__(self, channels, out_size, kernel: int = 3, init_seed: int | None = None):
        super().__init__()
        channels = int(channels)
        kernel = int(kernel)
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("channel-local kernel must be a positive odd number, got %r" % (kernel,))

        def build():
            conv = nn.Conv2d(channels, channels, kernel, stride=1, padding=kernel // 2,
                             groups=channels, bias=True)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, 0, kernel // 2, kernel // 2] = 1.0   # centre tap = identity
                conv.bias.zero_()
            return conv

        self.kernel_size = kernel
        if init_seed is None:
            self.conv = build()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(init_seed))
                self.conv = build()
        size = int(out_size[0] if isinstance(out_size, (tuple, list)) else out_size)
        self.out_size = (size, size)

    def forward(self, x):
        x = self.conv(x)
        if x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x



def _normalize_encoder_form(form: str) -> str:
    raw = str(form or "full").lower().replace("-", "_")
    aliases = {
        "plain": "full",
        "conv": "full",
        "dw": "depthwise",
        "separable": "depthwise_separable",
        "dw_pw": "depthwise_separable",
        "mobilenetv2": "depthwise_separable",
    }
    raw = aliases.get(raw, raw)
    if raw not in _ENCODER_FORMS:
        raise ValueError(f"Unknown dictionary encoder form={form!r}; expected one of {list(_ENCODER_FORMS)}")
    return raw


def _build_encoder(channels: int, form: str) -> nn.Sequential:
    """Build a dictionary key/query encoder with the requested form."""
    c = int(channels)
    if form == "full":
        return nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
    if c <= 0:
        raise ValueError("depthwise encoder needs channels > 0")
    if form == "depthwise":
        return nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
        )
    if form == "depthwise_separable":
        return nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
        )
    raise ValueError(f"Unknown dictionary encoder form={form!r}; expected one of {list(_ENCODER_FORMS)}")


def _init_identity_conv(conv: nn.Conv2d) -> None:
    """Turn a conv into a channel-wise identity at its centre tap (requires in == out).

    Covers the ordinary 3x3 (diagonal kernel), the depthwise 3x3 (weight shape
    ``(C, 1, k, k)``) and the 1x1 pointwise used by ``depthwise_separable``.
    """
    if conv.in_channels != conv.out_channels:
        raise ValueError("identity init requires in_channels == out_channels")
    kh, kw = conv.kernel_size
    with torch.no_grad():
        conv.weight.zero_()
        if conv.groups == conv.out_channels:
            conv.weight[:, 0, kh // 2, kw // 2] = 1.0
        else:
            idx = torch.arange(conv.out_channels, device=conv.weight.device)
            conv.weight[idx, idx, kh // 2, kw // 2] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


def _init_identity_stack(stack: nn.Sequential) -> None:
    """Identity-init a key/query encoder stack ((Conv, BN) pairs)."""
    for layer in stack:
        if isinstance(layer, nn.Conv2d):
            _init_identity_conv(layer)
        elif isinstance(layer, nn.BatchNorm2d):
            with torch.no_grad():
                layer.weight.fill_(1.0)
                layer.bias.zero_()
                layer.running_mean.zero_()
                layer.running_var.fill_(1.0)


class DictionaryModule(nn.Module):
    """Early-stage backbone dictionary module (student n10 ↔ teacher x6 / x10).

    Matches every student backbone channel (query) to teacher early-feature channels
    (keys) via a correlation matrix of pooled channel tokens, then reorganizes the
    teacher feature so it can serve as a per-channel pseudo ground truth:

        key   K = flatten(avgpool(encoder(x_t)))   (B, Ct, d)
        query Q = flatten(avgpool(encoder(n_s)))   (B, Cs, d)
        M = Q K^T (B, Cs, Ct)

    Matching modes (``dict_match``):
      - ``soft`` (default): M → softmax → soft channel gather (differentiable cross-attention;
        key/query encoders and the student backbone receive gradients — closer to the
        proposal's mutual-information / cross-attention intent).
      - ``hard``: index = argmax(M); non-differentiable gather (legacy). Encoders act as
        fixed projections; freeze their params after init when hard is selected upstream.
      - ``straight_through``: hard forward assignment, soft backward. Forward matches
        Figure 2 exactly while still letting the encoders receive gradient.

    Encoder variants (ablation knobs):
      - ``teacher_encoder=False`` removes the teacher-side encoder entirely, so
        ``K = flatten(avgpool(x_t))`` on the raw teacher tap (no parameters).
      - ``student_encoder="depthwise"`` replaces the student-side 3x3 conv with a
        per-channel depthwise conv (MobileNetV2 style, ``groups == out_channels``;
        requires ``in_channels == out_channels``, which holds for the query path).
      - ``student_encoder="depthwise_separable"`` adds the 1x1 pointwise conv that
        restores cross-channel mixing.

    The student feature is projected (DeconvNet) to the same channel/spatial size as
    the reorganized teacher feature for weighted align + attention restriction losses.

    Args:
        c_t (int): teacher feature channels.
        c_s (int): student feature channels.
        t_size (int): teacher feature spatial size (H == W) at trace time.
        s_size (int): student feature spatial size (H == W) at trace time.
        grid (int): pooled token grid; token dim d = grid * grid.
        match (str): ``soft``, ``hard``, or ``straight_through``.
        temperature (float): softmax temperature for soft matching.
        teacher_encoder (bool): build the teacher-side (key) encoder.
        student_encoder (str): ``full``, ``depthwise``, or ``depthwise_separable``.
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
        teacher_encoder: bool = True,
        student_encoder: str = "full",
        init_seed: int | None = None,
        proj_form: str = "deconv",
        proj_kernel: int = 3,
        fixed_map=None,
    ):
        super().__init__()
        match_aliases = {
            "st": "straight_through",
            "ste": "straight_through",
            "hard_st": "straight_through",
            "batch": "batch_corr",
            "oracle": "batch_corr",
            "batch_correlation": "batch_corr",
            "index": "index",
            "index_map": "index",
            "identity_map": "index",
            "naive": "index",
            "fixed": "fixed",
            "fixed_map": "fixed",
            "external": "fixed",
        }
        self.match = match_aliases.get(str(match).lower(), str(match).lower())
        if self.match not in {"hard", "soft", "straight_through", "batch_corr", "index", "fixed"}:
            raise ValueError(
                "Unknown dictionary match=%r; expected hard, soft, straight_through, "
                "batch_corr, index, or fixed" % (match,)
            )
        if self.match == "fixed":
            if fixed_map is None:
                raise ValueError("match='fixed' requires fixed_map (per-student-channel teacher index)")
            fm = torch.as_tensor(fixed_map, dtype=torch.long).flatten()
            if fm.numel() != int(c_s):
                raise ValueError(
                    "fixed_map has %d entries but the student tap has %d channels" % (fm.numel(), int(c_s))
                )
            if int(fm.min()) < 0 or int(fm.max()) >= int(c_t):
                raise ValueError(
                    "fixed_map entries must lie in [0, %d); got [%d, %d]"
                    % (int(c_t), int(fm.min()), int(fm.max()))
                )
            # Registered (not plain-assigned) so it follows .to(device) with the module.
            self.register_buffer("fixed_map", fm, persistent=False)
        else:
            self.fixed_map = None
        # An externally supplied assignment is not derived from features, so neither encoder
        # is needed; building them would only add parameters that cannot influence anything.
        self.has_teacher_encoder = bool(teacher_encoder) and self.match != "fixed"
        self.key_enc = _build_encoder(c_t, "full") if self.has_teacher_encoder else None
        # Student query path: same encoder family without downsampling conv stride.
        self.student_encoder = _normalize_encoder_form(student_encoder)
        self.query_enc = (
            None if self.match == "fixed" else _build_encoder(c_s, self.student_encoder)
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)
        # The projection. Under a frozen-encoder assignment (hard / index / fixed) this is
        # the only randomly initialised parameter in the module that training can still
        # move, so its starting point matters. Because it is built after the encoders, any
        # change to the encoder shapes shifts the RNG stream and silently re-rolls proj:
        # measured under an identical seed, removing the teacher encoder and switching the
        # student encoder to depthwise moves proj's weights by up to 0.062. That is larger
        # than the effects being measured, and with one run per arm it is not controlled
        # for. Passing init_seed pins proj's initialisation, so configurations that differ
        # in their encoders can still be compared without that confound.
        self.proj_form = str(proj_form or "deconv").lower()
        if self.proj_form not in {"deconv", "identity", "channel_local"}:
            raise ValueError("Unknown dictionary proj_form=%r; expected deconv, identity or "
                             "channel_local" % (proj_form,))
        if self.proj_form == "identity":
            self.proj = ChannelIdentityProjection(t_size)
        elif self.proj_form == "channel_local":
            # Channel-local but learnable: depthwise conv, identity-initialised. Keeps the
            # correspondence visible (groups=C pins output channel c to input channel c) while
            # giving the projection a small amount of freedom. See the class docstring for why
            # the kernel must be k>1 rather than a per-channel affine.
            self.proj = ChannelLocalProjection(c_s, t_size, kernel=proj_kernel, init_seed=init_seed)
        elif init_seed is None:
            self.proj = DeconvNet(c_s, c_s, s_size, t_size)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(init_seed))
                self.proj = DeconvNet(c_s, c_s, s_size, t_size)
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
        """Start Q/K projections as channel-wise identities to avoid arbitrary initial permutations.

        Works for the ordinary, depthwise and depthwise-separable forms, and skips the
        teacher path when its encoder was removed.
        """
        for encoder in (self.key_enc, self.query_enc):
            if encoder is not None:
                _init_identity_stack(encoder)

    def encoders(self) -> list:
        """Key/query encoder stacks that exist (the teacher side may be removed)."""
        return [enc for enc in (self.key_enc, self.query_enc) if enc is not None]

    def freeze_encoders(self, teacher: bool = True, student: bool = True) -> tuple[str, ...]:
        """Freeze the key/query encoder stacks, optionally per side.

        Teacher and student sides are separable on purpose: a distillation setup may want a stable
        teacher target while still letting the student query path adapt, and the two must be
        testable independently. A side that does not exist (the teacher encoder can be removed, and
        ``match='fixed'`` builds neither) is simply skipped.

        Params are marked ``_kd_frozen`` so the trainer's DDP/optimizer re-enable pass leaves them
        alone; the trainer also keeps these BNs in eval mode.

        Args:
            teacher: freeze ``key_enc`` (the teacher-side encoder).
            student: freeze ``query_enc`` (the student-side encoder).

        Returns:
            The side names that existed and were frozen, e.g. ``("key_enc", "query_enc")``.
        """
        wanted = (("key_enc", bool(teacher)), ("query_enc", bool(student)))
        frozen: list[str] = []
        for side, want in wanted:
            encoder = getattr(self, side)
            if encoder is None or not want:
                continue
            for p in encoder.parameters():
                p.requires_grad = False
                p._kd_frozen = True
            encoder.eval()
            frozen.append(side)
        return tuple(frozen)

    @property
    def differentiable_assignment(self) -> bool:
        """Whether assignment-specific losses can update Q/K and the student query path."""
        return self.match in {"soft", "straight_through"}
    def _batch_correlation_assignment(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """Channel correspondence estimated from per-batch activation statistics.

        Each channel is summarised by its per-image spatial mean, giving one series per
        channel across the batch. The correspondence is the argmax of the correlation
        between a student channel's series and each teacher channel's series, so it is
        estimated from the data instead of from two independently projected token spaces.
        Returns one teacher index per student channel, shared by every image of the batch.

        Diagnostic role: this is an oracle-ish matcher under the "channels that fire on
        the same images correspond" criterion. It exists to settle whether matching
        accuracy is the binding constraint, so it deliberately ignores ``key_enc`` and
        ``query_enc``.
        """
        with torch.no_grad():
            t_series = t_feat.float().mean(dim=(2, 3))  # (B, Ct)
            s_series = s_feat.float().mean(dim=(2, 3))  # (B, Cs)
            if t_series.shape[0] < 2:
                return torch.zeros(s_series.shape[1], dtype=torch.long, device=s_series.device)
            t_c = t_series - t_series.mean(dim=0, keepdim=True)
            s_c = s_series - s_series.mean(dim=0, keepdim=True)
            t_n = t_c / t_c.norm(dim=0, keepdim=True).clamp_min(1e-6)
            s_n = s_c / s_c.norm(dim=0, keepdim=True).clamp_min(1e-6)
            corr = (s_n.transpose(0, 1) @ t_n) / t_series.shape[0]  # (Cs, Ct)
            return corr.argmax(dim=1)

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
        n_teachers = t_feat.shape[1]
        commit = t_feat.new_zeros(())
        infomax = t_feat.new_zeros(())
        if self.match in {"batch_corr", "index", "fixed"}:
            if self.match == "fixed":
                # Externally prescribed correspondence: student channel s is aligned to the
                # teacher channel fixed_map[s]. This is the only mode whose assignment can be
                # set to a known, high-accuracy map, which is what makes "does an accurate
                # correspondence help?" testable.
                idx1d = self.fixed_map.to(device=t_feat.device)
            elif self.match == "index":
                # No matching at all: student channel s is aligned to teacher channel
                # s mod Ct. This is the "naive per-channel alignment" control that
                # isolates what the argmax matching contributes on top of it.
                idx1d = torch.arange(s_feat.shape[1], device=t_feat.device) % n_teachers
            else:
                # Data-adaptive correspondence from batch statistics; encoders bypassed.
                idx1d = self._batch_correlation_assignment(t_feat, s_feat)
            assignment_index = idx1d.unsqueeze(0).expand(t_feat.shape[0], -1).contiguous()
            assignment_soft = (
                F.one_hot(idx1d, num_classes=n_teachers).to(t_feat.dtype).unsqueeze(0)
                .expand(t_feat.shape[0], -1, -1).contiguous()
            )
            t_reorg = torch.gather(
                t_feat, 1, assignment_index[:, :, None, None].expand(-1, -1, h, w)
            )
        else:
            k_src = self.key_enc(t_feat) if self.key_enc is not None else t_feat
            k = self.pool(k_src).flatten(2)  # (B, Ct, d)
            q = self.pool(self.query_enc(s_feat)).flatten(2)  # (B, Cs, d)
            if self.match_norm == "l2":
                k = F.normalize(k, dim=2)
                q = F.normalize(q, dim=2)
            m = q @ k.transpose(1, 2)  # (B, Cs, Ct)
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
                counts = F.one_hot(assignment_index, num_classes=n_teachers).sum(dim=(0, 1)).float()
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
