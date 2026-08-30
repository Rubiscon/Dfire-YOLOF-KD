"""Guards that the shared parity kernel reproduces BaseTrainer's protocol exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from dfire_parity import build_args, build_optimizer, lr_lambda, scaled_weight_decay, warmup_iterations
from dfire_parity.engine import UltralyticsParityEngine
from ultralytics.engine.trainer import BaseTrainer

REFERENCE_IMAGES = 14122


def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, bias=True),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 8, 1, bias=False),
    )


def test_controlled_args_match_full_aug_protocol():
    args = build_args(data="dfire.yaml", model="models/yolo26n-DCN.yaml")
    assert (args.epochs, args.batch, args.nbs) == (200, 112, 112)
    assert (args.lr0, args.lrf, args.momentum, args.weight_decay) == (0.01, 0.01, 0.9, 0.0005)
    assert args.optimizer == "SGD" and not args.cos_lr and args.amp and not args.rect
    assert args.mosaic == 1.0 and args.close_mosaic == 10
    assert args.mixup == args.cutmix == args.copy_paste == 0.0
    assert (args.hsv_h, args.hsv_s, args.hsv_v) == (0.015, 0.7, 0.4)
    assert (args.translate, args.scale) == (0.1, 0.5)
    assert args.degrees == args.shear == args.perspective == 0.0
    assert (args.fliplr, args.flipud, args.bgr) == (0.5, 0.0, 0.0)
    assert args.warmup_bias_lr == 0.0 and args.warmup_momentum == 0.9


def test_weight_decay_is_unscaled_when_batch_equals_nbs():
    assert scaled_weight_decay(build_args()) == pytest.approx(0.0005)


def test_warmup_iterations_match_reference_run():
    args = build_args()
    nb = math.ceil(REFERENCE_IMAGES / args.batch)
    assert nb == 127
    assert warmup_iterations(args, nb) == 101


def test_lr_lambda_matches_reference_curve():
    args = build_args()
    lf = lr_lambda(args)
    # Values transcribed from Log/8.15_results.csv (lr/pg0 column).
    assert args.lr0 * lf(0) == pytest.approx(0.01)
    assert args.lr0 * lf(84) == pytest.approx(0.005842, abs=5e-7)
    assert args.lr0 * lf(199) == pytest.approx(0.00014950, abs=5e-9)


def test_optimizer_groups_match_base_trainer():
    args = build_args()
    model = _tiny_model()
    ours = build_optimizer(model, args)
    theirs = BaseTrainer.build_optimizer(
        BaseTrainer.__new__(BaseTrainer),
        model=model,
        name="SGD",
        lr=args.lr0,
        momentum=args.momentum,
        decay=scaled_weight_decay(args),
    )
    assert [g["param_group"] for g in ours.param_groups] == ["weight", "bn", "bias"]
    reference = {g["param_group"]: g for g in theirs.param_groups}
    for group in ours.param_groups:
        ref = reference[group["param_group"]]
        assert len(group["params"]) == len(ref["params"])
        assert group["weight_decay"] == ref.get("weight_decay", 0.0)
        assert group["lr"] == ref["lr"] and group["nesterov"] is ref["nesterov"]


def test_norm_bias_lands_in_bias_group():
    optimizer = build_optimizer(_tiny_model(), build_args())
    counts = {g["param_group"]: len(g["params"]) for g in optimizer.param_groups}
    # conv weight x2 decayed; bn weight no-decay; conv bias + bn bias in the bias group.
    assert counts == {"weight": 2, "bn": 1, "bias": 2}


def test_warmup_ramps_lr_from_zero_and_reaches_target():
    args = build_args()
    engine = UltralyticsParityEngine(_tiny_model(), args, steps_per_epoch=127, enable_ema=False)
    engine.before_train_iter(0)
    assert all(g["lr"] == pytest.approx(0.0) for g in engine.optimizer.param_groups)
    engine.before_train_iter(engine.nw)
    assert all(g["lr"] == pytest.approx(args.lr0) for g in engine.optimizer.param_groups)
    assert all(g["momentum"] == pytest.approx(args.momentum) for g in engine.optimizer.param_groups)


def test_ema_tracks_model_and_is_the_evaluated_weight():
    args = build_args()
    model = _tiny_model()
    engine = UltralyticsParityEngine(model, args, steps_per_epoch=127)
    before = engine.ema.ema.state_dict()["0.weight"].clone()
    with torch.no_grad():
        model[0].weight.add_(1.0)
    engine.ema.update(model)
    after = engine.ema.ema.state_dict()["0.weight"]
    assert engine.ema.updates == 1
    assert not torch.allclose(before, after)
    assert engine.eval_model() is engine.ema.ema


def test_scheduler_advances_once_per_epoch():
    args = build_args()
    engine = UltralyticsParityEngine(_tiny_model(), args, steps_per_epoch=127, enable_ema=False)
    for _ in range(85):
        engine.on_epoch_end()
    assert engine.epoch == 85
    assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * lr_lambda(args)(85))
