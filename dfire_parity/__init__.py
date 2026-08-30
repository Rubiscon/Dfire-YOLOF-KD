"""Shared Ultralytics training-protocol kernel for the D-Fire related-work comparison.

Import this from MMDetection (Structural KD) and Detectron2 (GID) so every framework
trains against the same data pipeline, optimizer grouping, schedule and EMA that the
reference Ultralytics run used.
"""

from .engine import (
    UltralyticsParityEngine,
    build_dataset,
    build_loader,
    build_optimizer,
    lr_lambda,
    preprocess_batch,
    resolve_dataset,
    scaled_weight_decay,
    val_batch_size,
    warmup_iterations,
)
from .hashing import (
    enforce_student_init_hash,
    state_dict_sha256,
    write_student_init_manifest,
)
from .protocol import (
    CLASS_NAMES,
    CONTROLLED_ARGS,
    EMA_DECAY,
    EMA_TAU,
    GRAD_CLIP_MAX_NORM,
    NUM_CLASSES,
    PARITY_TOLERANCE,
    REFERENCE_MAP50,
    STRIDE,
    VAL_BATCH_MULTIPLIER,
    VAL_RECT,
    build_args,
    steps_per_epoch,
)

__all__ = [
    "CLASS_NAMES",
    "CONTROLLED_ARGS",
    "EMA_DECAY",
    "EMA_TAU",
    "GRAD_CLIP_MAX_NORM",
    "NUM_CLASSES",
    "PARITY_TOLERANCE",
    "REFERENCE_MAP50",
    "STRIDE",
    "VAL_BATCH_MULTIPLIER",
    "VAL_RECT",
    "UltralyticsParityEngine",
    "build_args",
    "build_dataset",
    "build_loader",
    "build_optimizer",
    "enforce_student_init_hash",
    "lr_lambda",
    "preprocess_batch",
    "resolve_dataset",
    "scaled_weight_decay",
    "state_dict_sha256",
    "steps_per_epoch",
    "val_batch_size",
    "warmup_iterations",
    "write_student_init_manifest",
]
