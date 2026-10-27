"""Training protocol shared by the framework-fair comparison.

Pins the values the Ultralytics, MMDetection (SKD) and Detectron2 (GID) ports must agree on:
batch/nbs=32, SGD momentum=0.9, short warmup, amp=True.

Warmup is fixed in optimizer updates, so the same schedule applies at any batch:
``warmup_epochs`` comes from :func:`warmup_epochs_for`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# D-Fire train split and warmup length. Warmup is fixed in optimizer updates so that a
# change of physical batch does not change the schedule.
DFIRE_TRAIN_IMAGES = 14122
PROTOCOL_WARMUP_UPDATES = 101


def warmup_epochs_for(
    batch: int, images: int = DFIRE_TRAIN_IMAGES, updates: int = PROTOCOL_WARMUP_UPDATES
) -> float:
    """Warmup length in epochs for ``batch``, covering ``updates`` optimizer steps."""
    steps = max(1, -(-int(images) // int(batch)))
    return updates / steps


# Every field the reference protocol pins. Anything absent falls back to the
# fork's DEFAULT_CFG_DICT so future upstream defaults stay shared.
CONTROLLED_ARGS: dict[str, Any] = {
    # task / schedule
    "task": "detect",
    "mode": "train",
    "imgsz": 640,
    "epochs": 200,
    "patience": 200,
    "batch": 32,
    "nbs": 32,
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
    # optimizer (explicit SGD; optimizer=auto / MuSGD is not comparable across ports)
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    # 101 optimizer updates regardless of batch: 101/442 epochs at batch 32.
    "warmup_epochs": warmup_epochs_for(32),
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

# Dataset contract. CLASS_NAMES / NUM_CLASSES are the D-Fire defaults, so existing D-Fire
# tests and docs stay valid. VOC2007 is the other allowed host dataset (trainval->train,
# test->val/test).
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
            "hosts accept D-Fire %s or VOC2007 %s."
            % (mapping, CLASS_NAMES, VOC2007_CLASS_NAMES)
        )
    if nc is not None and int(nc) != expected_nc:
        raise ValueError(
            "Dataset %s declares nc=%s but names imply nc=%s" % (key, nc, expected_nc)
        )
    return key

# Solo reference for this full-augmentation protocol (YOLOF_batch64 peak ~0.709 under the
# same augmentation family).
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


# Online distillation (matches Ultralytics YOLOFDistillationModel).
# Teacher learns only from GT; KD/GID/CanKD/SKD targets stay detached.
DEFAULT_TEACHER_TASK_WEIGHT = 1.0
_TEACHER_NAME_PREFIXES = ("teacher_model.", "teacher.")
_DISTILL_NAME_TOKENS = ("cankd", "structural_kd", "adapters")
_COCO_TEACHER_INIT_NAME = "yolo26n.pt"
_IN_DATASET_TEACHER_TOKENS = (
    "teacher_n200",
    "voc2007_teacher",
    "_teacher_best.pt",
)


def is_in_dataset_finetuned_teacher(path: Any) -> bool:
    """True for a finished in-dataset YOLO26n checkpoint, not COCO yolo26n.pt."""
    name = Path(str(path or "")).name.lower()
    if not name or name == _COCO_TEACHER_INIT_NAME:
        return False
    return any(token in name for token in _IN_DATASET_TEACHER_TOKENS)


def assert_online_teacher_init(path: Any, *, online_distill: bool = True) -> None:
    """Online KD teacher must start from COCO yolo26n.pt and train with the student."""
    if not online_distill:
        return
    if not path:
        raise ValueError(
            "Online KD requires teacher init from COCO yolo26n.pt.")
    name = Path(str(path)).name.lower()
    if name != _COCO_TEACHER_INIT_NAME or is_in_dataset_finetuned_teacher(path):
        raise ValueError(
            "Online KD trains the teacher from COCO yolo26n.pt jointly with the student, "
            "not from a finished in-dataset checkpoint. Got: %s" % path)


def resolve_teacher_freeze_epoch(freeze_epoch: Any, epochs: int) -> int:
    """1-indexed freeze epoch; ``<=0`` or ``None`` means never freeze during ``epochs``."""
    if freeze_epoch in (None, "", False):
        return int(epochs)
    value = int(freeze_epoch)
    if value <= 0:
        return int(epochs)
    return value


def teacher_joint_training(
    online_distill: bool,
    freeze_epoch: Any,
    epoch_1indexed: int,
    epochs: int,
    teacher_frozen: bool = False,
) -> bool:
    """True while the teacher receives GT task loss.

    Same rule as Ultralytics: epochs ``1..freeze_epoch`` are joint. When
    ``freeze_epoch == epochs`` the teacher stays trainable for the whole run.
    """
    if not online_distill or teacher_frozen:
        return False
    freeze_ep = resolve_teacher_freeze_epoch(freeze_epoch, epochs)
    return int(epoch_1indexed) <= freeze_ep


def split_kd_grad_param_groups(model):
    """Partition trainable params for per-branch clip (student / teacher / distill)."""
    student, teacher, distill = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(_TEACHER_NAME_PREFIXES) or ".teacher_model." in name:
            teacher.append(param)
        elif any(token in name for token in _DISTILL_NAME_TOKENS):
            distill.append(param)
        else:
            student.append(param)
    return student, teacher, distill
