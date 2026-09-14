"""Reusable pieces of the Ultralytics training loop for external frameworks.

MMDetection (Structural KD) and Detectron2 (GID) call into this module so that the
data pipeline, optimizer parameter grouping, LR/momentum warmup, LR schedule, EMA,
AMP and gradient clipping are literally the same code as ``BaseTrainer``, not a
re-implementation that can silently drift.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Callable

import numpy as np
import torch
from torch import nn, optim

from .protocol import (
    EMA_DECAY,
    EMA_TAU,
    GRAD_CLIP_MAX_NORM,
    GRAD_CLIP_NORM_TYPE,
    STRIDE,
    VAL_BATCH_MULTIPLIER,
)


def val_batch_size(args) -> int:
    """Validation batch the reference run used: twice the training batch."""
    return int(args.batch) * VAL_BATCH_MULTIPLIER


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def resolve_dataset(data_yaml: str) -> dict[str, Any]:
    """Resolve a YOLO ``data.yaml`` through the fork's own checker."""
    from ultralytics.data.utils import check_det_dataset

    return check_det_dataset(str(data_yaml))


def build_dataset(args, data: dict[str, Any], mode: str = "train", stride: int = STRIDE):
    """Build the fork's ``YOLODataset`` exactly as ``DetectionTrainer`` does.

    ``mode='train'`` applies the protocol augmentations (mosaic/HSV/...).
    ``mode='val'`` uses ``rect=True``, matching the reference run's per-epoch val.
    """
    from ultralytics.data.build import build_yolo_dataset

    if mode not in {"train", "val"}:
        raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
    img_path = data["train"] if mode == "train" else (data.get("val") or data.get("test"))
    return build_yolo_dataset(
        args, img_path, args.batch, data, mode=mode, rect=mode == "val", stride=stride
    )


def build_loader(args, dataset, mode: str = "train", rank: int = -1,
                 pin_memory: bool | None = None):
    """Wrap a dataset in the fork's ``InfiniteDataLoader`` with matching worker counts.

    ``pin_memory`` defaults to ``args.pin_memory`` (controlled protocol: False) so
    host trainers do not inherit Ultralytics' True default, which has crashed
    long CanKD runs inside the pin_memory worker thread.
    """
    from ultralytics.data.build import build_dataloader

    if pin_memory is None:
        pin_memory = bool(getattr(args, "pin_memory", False))
    return build_dataloader(
        dataset,
        batch=args.batch if mode == "train" else val_batch_size(args),
        workers=args.workers if mode == "train" else args.workers * 2,
        shuffle=mode == "train",
        rank=rank,
        pin_memory=pin_memory,
    )


def preprocess_batch(batch: dict[str, Any], device,
                     non_blocking: bool | None = None) -> dict[str, Any]:
    """Move a collated batch to ``device`` and scale images to ``[0, 1]``.

    ``non_blocking`` defaults to False. Async H2D copies only help with pinned
    host memory; with ``pin_memory=False`` they can surface stale CUDA errors.
    """
    if non_blocking is None:
        non_blocking = False
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=non_blocking)
    batch["img"] = batch["img"].float() / 255
    return batch


# --------------------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------------------
def scaled_weight_decay(args) -> float:
    """Reproduce ``BaseTrainer`` weight-decay scaling by effective batch."""
    accumulate = max(round(args.nbs / args.batch), 1)
    return args.weight_decay * args.batch * accumulate / args.nbs


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """Build the three-group SGD used by ``BaseTrainer.build_optimizer``.

    Groups are ``weight`` (decayed), ``bn`` (no decay) and ``bias`` (no decay). The
    bias test precedes the norm test, so normalization biases land in the bias group.
    Only ``SGD`` is supported here: the protocol forbids ``optimizer=auto``.
    """
    from ultralytics.utils.torch_utils import unwrap_model

    if args.optimizer != "SGD":
        raise ValueError(
            f"Controlled protocol requires optimizer='SGD', got {args.optimizer!r}"
        )
    decay = scaled_weight_decay(args)
    norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    groups: list[dict[str, nn.Parameter]] = [{}, {}, {}]
    for module_name, module in unwrap_model(model).named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                groups[2][fullname] = param
            elif isinstance(module, norm_types) or "logit_scale" in fullname:
                groups[1][fullname] = param
            else:
                groups[0][fullname] = param

    common = dict(lr=args.lr0, momentum=args.momentum, nesterov=True)
    param_groups = [
        {"params": list(groups[0].values()), **common, "weight_decay": decay, "param_group": "weight"},
        {"params": list(groups[1].values()), **common, "weight_decay": 0.0, "param_group": "bn"},
        {"params": list(groups[2].values()), **common, "weight_decay": 0.0, "param_group": "bias"},
    ]
    return optim.SGD(param_groups)


# --------------------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------------------
def lr_lambda(args) -> Callable[[int], float]:
    """Per-epoch LR multiplier: linear ``1 -> lrf`` unless ``cos_lr`` is set."""
    if args.cos_lr:
        from ultralytics.utils.torch_utils import one_cycle

        return one_cycle(1, args.lrf, args.epochs)
    return lambda x: max(1 - x / args.epochs, 0) * (1.0 - args.lrf) + args.lrf


def warmup_iterations(args, steps_per_epoch: int) -> int:
    """``nw`` from ``BaseTrainer._do_train``; ``-1`` disables warmup."""
    if args.warmup_epochs <= 0:
        return -1
    return max(round(args.warmup_epochs * steps_per_epoch), 100)


def _grad_scaler(enabled: bool):
    """Match ``BaseTrainer`` GradScaler construction across torch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------------------
# Engine facade
# --------------------------------------------------------------------------------------
class UltralyticsParityEngine:
    """Drives warmup, LR schedule, AMP, gradient clipping and EMA for a host framework.

    Host contract per optimizer update::

        engine.before_train_iter(global_update_index)
        with engine.autocast():
            loss = ...
        engine.backward(loss)
        engine.optimizer_step()

    Call :meth:`on_epoch_end` once after each finished epoch (pass the train
    dataset so mosaic can be closed for the final ``close_mosaic`` epochs).
    """

    def __init__(self, model: nn.Module, args, steps_per_epoch: int, enable_ema: bool = True):
        from ultralytics.utils.torch_utils import ModelEMA

        self.model = model
        self.args = args
        self.steps_per_epoch = int(steps_per_epoch)
        self.amp = bool(args.amp)
        self.lf = lr_lambda(args)
        self.optimizer = build_optimizer(model, args)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.nw = warmup_iterations(args, self.steps_per_epoch)
        self.accumulate = max(round(args.nbs / args.batch), 1)
        self.scaler = _grad_scaler(self.amp)
        self.ema = ModelEMA(model, decay=EMA_DECAY, tau=EMA_TAU) if enable_ema else None
        self.epoch = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "ema": None if self.ema is None else self.ema.ema.state_dict(),
            "ema_updates": None if self.ema is None else self.ema.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state.get("epoch", 0))
        if self.ema is not None and state.get("ema") is not None:
            self.ema.ema.load_state_dict(state["ema"])
            self.ema.updates = int(state["ema_updates"])

    def autocast(self):
        """Forward-pass autocast context, matching ``BaseTrainer``."""
        from ultralytics.utils.torch_utils import autocast

        return autocast(self.amp)

    def before_train_iter(self, global_step: int) -> None:
        """Apply the per-update LR/momentum warmup ramp.

        ``global_step`` is the 0-based count of optimizer updates since training began.
        """
        if self.nw <= 0 or global_step > self.nw:
            return
        xi = [0, self.nw]
        target_scale = self.lf(self.epoch)
        for group in self.optimizer.param_groups:
            start = self.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
            group["lr"] = np.interp(
                global_step, xi, [start, group["initial_lr"] * target_scale]
            )
            if "momentum" in group:
                group["momentum"] = np.interp(
                    global_step, xi, [self.args.warmup_momentum, self.args.momentum]
                )

    def backward(self, loss: torch.Tensor) -> None:
        """Scale (when AMP) and backpropagate ``loss``."""
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self) -> None:
        """Unscale (AMP), clip, step, update scaler, zero grads, advance EMA."""
        if self.amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=GRAD_CLIP_MAX_NORM, norm_type=GRAD_CLIP_NORM_TYPE
        )
        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        if self.ema is not None:
            self.ema.update(self.model)

    def close_mosaic(self, dataset) -> None:
        """Disable mosaic/mixup/cutmix/copy_paste on ``dataset`` for the final epochs."""
        if not hasattr(dataset, "close_mosaic"):
            return
        dataset.close_mosaic(hyp=copy(self.args))

    def maybe_close_mosaic(self, dataset) -> None:
        """Close mosaic when ``epoch`` reaches ``epochs - close_mosaic``."""
        close = int(getattr(self.args, "close_mosaic", 0) or 0)
        if close > 0 and self.epoch == int(self.args.epochs) - close:
            self.close_mosaic(dataset)

    def on_epoch_end(self, dataset=None) -> None:
        """Step the epoch-wise LR schedule; optionally close mosaic for the tail."""
        self.scheduler.step()
        self.epoch += 1
        if dataset is not None:
            self.maybe_close_mosaic(dataset)

    def eval_model(self) -> nn.Module:
        """Weights the reference run validates and checkpoints: EMA when enabled."""
        if self.ema is None:
            return self.model
        self.ema.update_attr(
            self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"]
        )
        return self.ema.ema
