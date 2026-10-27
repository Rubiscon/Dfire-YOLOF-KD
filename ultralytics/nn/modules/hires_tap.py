# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Narrow high-resolution tap for single-scale (YOLOF-style) encoders.

Motivation
----------
A YOLOF encoder widens the receptive field on ``C5`` (stride 32) with dilated residual
blocks, but it never adds sampling points. At ``imgsz=640`` the detection lattice is
20x20, so no function of that grid can resolve objects below roughly one cell. The
receptive field is sufficient; the sampling density is not. That is a hard floor on
``AP_S`` that no feature- or response-matching distillation objective can lift, because
the information was never sampled in the first place.

``HighResTap`` restores one finer lattice. It consumes a backbone feature that is already
available at a lower stride (typically stride 8), compresses it to a narrow bottleneck, and
mixes locally. The bottleneck is what keeps the pathway cheap; ``c2`` is kept separately
configurable so the produced level can match the width the detection head expects.

Cost note
---------
In Ultralytics the Detect head's hidden width is derived from the *first* level's channel
count (``c2 = max(16, ch[0] // 4, reg_max * 4)``, ``c3 = max(ch[0], min(nc, 100))``). A tap
that emits few channels therefore narrows every head branch, not just its own. Set ``c2``
to the existing level width for an experiment that changes only the added pathway, and to a
small value for a cost-curve point.

The block caches its output in ``last_feature`` so the distillation trainer can tap it the
same way it taps the dilated residual blocks.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("HighResTap",)


class HighResTap(nn.Module):
    """``1x1`` compress to ``mid``, a local depthwise mix, then ``1x1`` expand to ``c2``.

    Signature compatible with ``parse_model``: ``HighResTap(c1, c2, mid=None, k=3)``.

    Attributes:
        reduce (Conv): Pointwise compression from ``c1`` to the ``mid`` bottleneck.
        refine (Conv): Depthwise local mixing at ``mid`` channels.
        expand (Conv): Pointwise expansion from ``mid`` to ``c2``.
        last_feature (torch.Tensor | None): Cached output for knowledge distillation.
    """

    def __init__(self, c1: int, c2: int, mid: Optional[int] = None, k: int = 3):
        """Initialise the tap.

        Args:
            c1: Input channels (the backbone tap width, e.g. stride-8 features).
            c2: Output width consumed by the detection head.
            mid: Bottleneck width. Defaults to a quarter of ``c2``, floored at 8.
            k: Spatial kernel of the local mixing convolution.
        """
        super().__init__()
        mid = int(mid) if mid else max(c2 // 4, 8)
        self.reduce = Conv(c1, mid, k=1, s=1, p=0, act=True)
        self.refine = Conv(mid, mid, k=k, s=1, p=k // 2, g=mid, act=True)
        self.expand = Conv(mid, c2, k=1, s=1, p=0, act=True)
        self.last_feature: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress, mix locally, expand, caching the result for distillation."""
        y = self.expand(self.refine(self.reduce(x)))
        self.last_feature = y
        return y

    def __deepcopy__(self, memo):
        # ``last_feature`` caches a non-leaf graph tensor for KD; null it so deepcopy
        # (ModelEMA init / checkpoint saving) does not fail on non-leaf tensors.
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = None if k == "last_feature" else copy.deepcopy(v, memo)
        return new
