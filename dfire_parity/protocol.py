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

# Dataset contract. CLASS_NAMES / NUM_CLASSES remain the D-Fire defaults so
# existing D-Fire tests and docs stay valid. VOC2007 is the other allowed
# host dataset (trainval→train, test→val/test). Do not silently accept DFS.
NUM_CLASSES = 2
CLASS_NAMES = {0: "smoke", 1: "fire"}
VOC2007_NUM_CLASSES = 20
VOC2007_CLASS_NAMES = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor",
}
STRIDE = 32


def _as_name_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def names_for_num_classes(num_classes: int) -> dict[int, str]:
    """Detect-head names for a supported controlled dataset."""
    count = int(num_classes)
    if count == NUM_CLASSES:
        return dict(CLASS_NAMES)
    if count == VOC2007_NUM_CLASSES:
        return dict(VOC2007_CLASS_NAMES)
    raise ValueError(
        "Controlled hosts support nc=2 (D-Fire) or nc=20 (VOC2007), got %s" % count
    )


def identify_dataset(names: Any, nc: int | None = None) -> str:
    """Return 'dfire' or 'voc2007', or raise if the yaml is some other dataset."""
    mapping = _as_name_map(names)
    if mapping == CLASS_NAMES:
        key = "dfire"
        expected_nc = NUM_CLASSES
    elif mapping == VOC2007_CLASS_NAMES:
        key = "voc2007"
        expected_nc = VOC2007_NUM_CLASSES
    else:
        raise ValueError(
            "Unsupported dataset classes %s. Controlled Ultralytics/GID/SKD/CanKD "
            "hosts accept D-Fire %s or VOC2007 %s (not DFS)."
            % (mapping, CLASS_NAMES, VOC2007_CLASS_NAMES)
        )
    if nc is not None and int(nc) != expected_nc:
        raise ValueError(
            "Dataset %s declares nc=%s but names imply nc=%s" % (key, nc, expected_nc)
        )
    return key

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
