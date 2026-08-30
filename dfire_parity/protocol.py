"""Canonical D-Fire controlled training protocol for framework-fair comparison.

Source of truth for Ultralytics / MMDetection (SKD) / Detectron2 (GID). Values
below match the full-augmentation YOLOF solo recipe (see
``Log/YOLOF_batch64_args.yaml``) with the controlled comparison overrides:
batch/nbs=112, SGD momentum=0.9, short warmup, amp=True.
"""

from __future__ import annotations

from typing import Any

# Every field the reference protocol pins. Anything absent falls back to the
# fork's DEFAULT_CFG_DICT so future upstream defaults stay shared.
CONTROLLED_ARGS: dict[str, Any] = {
    # task / schedule
    "task": "detect",
    "mode": "train",
    "imgsz": 640,
    "epochs": 200,
    "patience": 200,
    "batch": 112,
    "nbs": 112,
    "fixed_accumulate": True,
    "seed": 0,
    "deterministic": False,
    "workers": 8,
    "cache": False,
    "rect": False,
    "single_cls": False,
    "fraction": 1.0,
    "multi_scale": 0.0,
    "amp": True,
    "compile": False,
    "dropout": 0.0,
    "freeze": None,
    "profile": False,
    "plots": False,
    # optimizer (explicit SGD; do not use optimizer=auto / MuSGD)
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    # short warmup: max(round(0.79275 * 127), 100) == 101 updates
    "warmup_epochs": 0.7927519818799547,
    "warmup_momentum": 0.9,
    "warmup_bias_lr": 0.0,
    # loss gains
    "box": 7.5,
    "cls": 0.5,
    "cls_pw": 0.0,
    "dfl": 1.5,
    # augmentation: mosaic + HSV + translate/scale + horizontal flip
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "erasing": 0.4,
    "auto_augment": "randaugment",
    "augmentations": [],
    "overlap_mask": True,
    "mask_ratio": 4,
    # inference / validation
    "val": True,
    "split": "val",
    "conf": None,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "augment": False,
    "agnostic_nms": False,
    "nms": False,
    "save_json": False,
}

# BaseTrainer builds the validation loader at twice the train batch (trainer.py
# `_build_train_pipeline`), and DetectionTrainer.build_dataset passes rect=True.
VAL_BATCH_MULTIPLIER = 2
VAL_RECT = True

# Gradient-clip norm is hard-coded in BaseTrainer.optimizer_step, not in cfg.
GRAD_CLIP_MAX_NORM = 10.0
GRAD_CLIP_NORM_TYPE = 2.0

# ModelEMA constructor defaults used by BaseTrainer._setup_train.
EMA_DECAY = 0.9999
EMA_TAU = 2000

# Dataset contract.
NUM_CLASSES = 2
CLASS_NAMES = {0: "smoke", 1: "fire"}
STRIDE = 32

# Solo gate for this full-augmentation protocol (YOLOF_batch64 peak ~0.709 under
# the same aug family). Re-measure after the first Ultra batch-112 amp run if
# the observed peak differs; until then use the batch64 Ultra peak as reference.
REFERENCE_MAP50 = 0.709
PARITY_TOLERANCE = 0.005


def build_args(**overrides: Any):
    """Return the Ultralytics ``args`` namespace for the controlled protocol.

    Args:
        **overrides: Per-experiment values such as ``data``, ``model``, ``epochs``
            or ``batch``. Applied on top of :data:`CONTROLLED_ARGS`.

    Returns:
        (IterableSimpleNamespace): Namespace accepted by the fork's dataset,
            optimizer and loss code paths.
    """
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

    unknown = set(overrides) - set(DEFAULT_CFG_DICT)
    if unknown:
        raise KeyError(f"Unknown Ultralytics cfg keys: {sorted(unknown)}")

    merged = dict(DEFAULT_CFG_DICT)
    merged.update(CONTROLLED_ARGS)
    merged.update(overrides)
    return IterableSimpleNamespace(**merged)


def steps_per_epoch(num_images: int, batch: int = CONTROLLED_ARGS["batch"]) -> int:
    """Optimizer updates per epoch, matching the fork's non-dropping dataloader."""
    return -(-int(num_images) // int(batch))
