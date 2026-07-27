"""Unified D-Fire baselines on official train/val/test split (dfire.yaml).

Baselines (kept lean):
  yolo26n       - YOLO26n FPN upper bound
  dcn-solo      - YOLO26n-DCN / YOLOF student, no KD
  early         - CrisReport early dictionary (n10↔x6, attention weight)
  early-dldx    - same recipe + saliency_dLdx (∂J_task/∂x^e)
  early-tune1/2 - editable copies for hyperparameter sweeps

Hyperparameters below match the previous script bit-for-bit for these recipes
(pretrained / batch / teacher_weights / KD gains). Sweep entries (early-B..T3,
kd-p0, …) were removed; use early-tune* instead.

Usage:
  python scripts/train_baselines.py --baseline yolo26n
  python scripts/train_baselines.py --baseline dcn-solo
  python scripts/train_baselines.py --baseline early
  python scripts/train_baselines.py --baseline early-dldx
  python scripts/train_baselines.py --baseline early-tune1,early-tune2
  python scripts/train_baselines.py --baseline all
  python scripts/train_baselines.py --baseline early --test-only

Aliases (old names still work): kd-early → early, early-S1a → early-dldx.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_TQDM_NONINTERACTIVE", "1")

import torch

torch.backends.cudnn.benchmark = True

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer, YOLOFDistillationTrainer

# ---------------------------------------------------------------------------
# Shared training config (official split: datasets/D-Fire/data)
# ---------------------------------------------------------------------------
COMMON: dict[str, Any] = {
    "data": "dfire.yaml",
    "imgsz": 640,
    "epochs": 200,
    "patience": 200,
    "device": 0,
    "workers": 8,
    "compile": False,
    "deterministic": False,
    "pretrained": False,
    "project": "dfire-baselines",
    "val": True,
}

DEFAULT_BATCH = 112  # solo / KD when VRAM allows (matches n_kd_n_batch112)
DEFAULT_KD_BATCH = 112  # proven KD recipe; drop with --batch if saliency OOM

# Shared online KD — aligned to Log/n_kd_n_batch112 (best mAP50≈0.728).
_KD_COMMON: dict[str, Any] = {
    "online_distill": True,
    "teacher_freeze_epoch": 200,
    "teacher_freeze_use_ema": True,
    "task_loss": 1.0,
    "teacher_task_loss": 1.0,
    "feature_norm": "channel",
    "feature_loss": 0.08,
    "align": True,
    "align_start_epoch": 20,
    "align_loss": 0.12,
    "align_branch": "one2many",
    "align_cls_mode": "kl",
    "distill_temperature": 3,
    "align_box": 2.0,
    "align_cls": 4.0,
    "distill_conf_thres": 0.25,
    "distill_iou_thres": 0.5,
    "dict_student_layer": 10,
    "dict_start_epoch": 0,
    "dict_weight": "attention",
    "dict_match": "hard",
    "dict_match_temp": 0.07,
    "dict_feature_norm": "channel",
    "dict_saliency_ema": 0.9,
    "dict_attn_start_epoch": 0,
    "dict_commit_loss": 0.0,
    "pretrained": "yolo26n.pt",
    "teacher_weights": "yolo26n.pt",
}


def _kd_early(**overrides: Any) -> dict[str, Any]:
    """KD early recipe: student n10 ↔ teacher x6. Overrides must not silently drop shared keys."""
    cfg: dict[str, Any] = {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "batch": DEFAULT_KD_BATCH,
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
    }
    cfg.update(overrides)
    return cfg


BASELINES: dict[str, dict[str, Any]] = {
    "yolo26n": {
        "trainer": "detect",
        "model": "yolo26n.yaml",
        "name": "baseline-yolo26n",
        "batch": DEFAULT_BATCH,
        "pretrained": "yolo26n.pt",
        "description": "YOLO26n FPN teacher upper bound",
    },
    "dcn-solo": {
        "trainer": "detect",
        "model": "yolo26n-DCN.yaml",
        "name": "baseline-dcn-solo",
        "batch": DEFAULT_BATCH,
        "pretrained": "yolo26n.pt",
        "description": "YOLO26n-DCN (YOLOF head) without distillation",
    },
    # Former kd-early / early-3 attention recipe (hyperparams unchanged).
    "early": _kd_early(
        name="baseline-kd-early",
        description="CrisReport early dictionary: n10↔x6, hard match, attention weight, pretrained student",
    ),
    # Former early-S1a (hyperparams unchanged).
    "early-dldx": _kd_early(
        name="baseline-early-S1a-dLdx",
        description="early + saliency_dLdx (mean_c|∂J_task/∂x^e|); blur/clip off",
        dict_weight="saliency_dLdx",
        dict_saliency_blur=0.0,
        dict_saliency_clip=0.0,
    ),
    # --- Sweep slots: start from known recipes; edit align / attn / weight as needed ---
    # Former early-S1a clone (0.08 / 0.25 / dLdx).
    "early-tune1": _kd_early(
        name="baseline-early-tune1",
        description="TUNABLE: edit dict_align_loss / dict_attn_loss / dict_weight (starts as early-dldx)",
        dict_weight="saliency_dLdx",
        dict_align_loss=0.08,
        dict_attn_loss=0.25,
        dict_saliency_blur=0.0,
        dict_saliency_clip=0.0,
    ),
    # Former early-SA3 (0.10 / 0.25 / dLdx) — historical best gate λ pair.
    "early-tune2": _kd_early(
        name="baseline-early-tune2",
        description="TUNABLE: edit dict_align_loss / dict_attn_loss / dict_weight (starts as align=0.10, attn=0.25, dLdx)",
        dict_weight="saliency_dLdx",
        dict_align_loss=0.10,
        dict_attn_loss=0.25,
        dict_saliency_blur=0.0,
        dict_saliency_clip=0.0,
    ),
}

# Old CLI names → current keys (resume/docs convenience).
_ALIASES: dict[str, str] = {
    "kd-early": "early",
    "early-S1a": "early-dldx",
    "early-SA3": "early-tune2",
}


def _canonical(key: str) -> str:
    return _ALIASES.get(key, key)


def resolve_baseline_keys(spec: str) -> list[str]:
    """Parse ``--baseline``: ``all``, a single name, or comma-separated names."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty --baseline")
    if spec == "all":
        return list(BASELINES)
    keys = [_canonical(k.strip()) for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in BASELINES]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s) {unknown}. Choose from: {list(BASELINES)} "
            f"(aliases: {list(_ALIASES)}) or 'all'"
        )
    return keys


def build_overrides(baseline_key: str, args: argparse.Namespace) -> dict[str, Any]:
    """Merge COMMON + baseline-specific overrides + CLI overrides."""
    baseline_key = _canonical(baseline_key)
    if baseline_key not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline_key!r}. Choose from: {list(BASELINES)}")

    cfg = deepcopy(COMMON)
    cfg.update(BASELINES[baseline_key])
    cfg.pop("trainer", None)
    cfg.pop("description", None)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.patience is not None:
        cfg["patience"] = args.patience
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.device is not None:
        cfg["device"] = args.device
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.name_suffix:
        cfg["name"] = f"{cfg['name']}-{args.name_suffix}"
    if args.resume:
        cfg["resume"] = True
        if args.weights:
            cfg["model"] = args.weights

    return cfg


def weights_path(project: str, name: str, task: str = "detect") -> Path:
    """Match Ultralytics get_save_dir: runs/{task}/{project}/{name}/weights/best.pt"""
    return Path("runs") / task / project / name / "weights" / "best.pt"


def run_test_eval(weights: Path, args: argparse.Namespace) -> None:
    """Evaluate best checkpoint on held-out test split (run once after training)."""
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}\nTrain first or pass --weights.")

    print(f"\n{'=' * 60}\nTest split evaluation: {weights}\n{'=' * 60}")
    model = YOLO(str(weights))
    model.val(
        data=COMMON["data"],
        split="test",
        imgsz=COMMON["imgsz"],
        batch=args.batch or DEFAULT_BATCH,
        device=args.device if args.device is not None else COMMON["device"],
        workers=args.workers if args.workers is not None else COMMON["workers"],
    )


def train_baseline(baseline_key: str, args: argparse.Namespace) -> Path:
    """Train one baseline; return path to best.pt."""
    baseline_key = _canonical(baseline_key)
    spec = BASELINES[baseline_key]
    overrides = build_overrides(baseline_key, args)
    trainer_cls = YOLOFDistillationTrainer if spec["trainer"] == "kd" else DetectionTrainer

    print(f"\n{'=' * 60}")
    print(f"Baseline: {baseline_key} — {spec['description']}")
    print(f"Trainer:  {trainer_cls.__name__}")
    print(f"Run name: {overrides['name']}")
    print(f"{'=' * 60}\n")

    trainer = trainer_cls(overrides=overrides)
    trainer.train()

    best = trainer.save_dir / "weights" / "best.pt"
    print(f"\nTraining done. best.pt -> {best}")
    print(f"Final test eval:\n  python scripts/train_baselines.py --baseline {baseline_key} --test-only --weights {best}")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D-Fire unified baseline training (train/val/test split)")
    parser.add_argument(
        "--baseline",
        default="yolo26n",
        help=(
            "baseline name, comma-separated list, or 'all'. "
            f"Available: {', '.join(BASELINES)}"
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help=f"override epochs (default: {COMMON['epochs']})")
    parser.add_argument("--patience", type=int, default=None, help=f"override patience (default: {COMMON['patience']})")
    parser.add_argument("--batch", type=int, default=None, help="override batch size")
    parser.add_argument("--device", default=None, help="cuda device id or 'cpu'")
    parser.add_argument("--workers", type=int, default=None, help="dataloader workers")
    parser.add_argument("--name-suffix", default="", help="append to run name, e.g. 'ep150'")
    parser.add_argument("--resume", action="store_true", help="resume from last.pt in the run directory")
    parser.add_argument(
        "--weights",
        default="",
        help="checkpoint for --resume or --test-only",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="skip training; evaluate --weights (or default best.pt) on test split",
    )
    parser.add_argument(
        "--test-after-train",
        action="store_true",
        help="run test-split val() immediately after each training run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        keys = resolve_baseline_keys(args.baseline)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.test_only:
        if len(keys) != 1:
            raise SystemExit("--test-only requires a single --baseline, not a list or 'all'")
        if args.weights:
            best = Path(args.weights)
        else:
            overrides = build_overrides(keys[0], args)
            best = weights_path(str(overrides["project"]), str(overrides["name"]))
        run_test_eval(best, args)
        return

    print(f"Queue ({len(keys)} run(s)): {' -> '.join(keys)}")
    for i, key in enumerate(keys, 1):
        print(f"\n>>> [{i}/{len(keys)}] starting {key}")
        best = train_baseline(key, args)
        if args.test_after_train:
            run_test_eval(best, args)
        print(f">>> [{i}/{len(keys)}] finished {key} -> {best}")


if __name__ == "__main__":
    main()
