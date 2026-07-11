"""Unified D-Fire baselines on official train/val/test split (dfire.yaml).

Baselines:
  1. yolo26n      - YOLO26n FPN upper bound (DetectionTrainer)
  2. dcn-solo     - YOLO26n-DCN / YOLOF student without KD (DetectionTrainer)
  3. kd-p0        - Online KD, p0-5 config (YOLOFDistillationTrainer)

Usage:
  python scripts/train_baselines.py --baseline yolo26n
  python scripts/train_baselines.py --baseline dcn-solo
  python scripts/train_baselines.py --baseline kd-p0
  python scripts/train_baselines.py --baseline all          # run 1 -> 2 -> 3 sequentially
  python scripts/train_baselines.py --baseline yolo26n --test-only  # eval best.pt on test split

After each train run, val mAP drives best.pt; use --test-only (or the printed command) for final test mAP.
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
# Shared training config (new official split: datasets/D-Fire/data)
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

DEFAULT_BATCH = 112  # shared across baselines for experiment consistency; override with --batch if OOM

BASELINES: dict[str, dict[str, Any]] = {
    "yolo26n": {
        "trainer": "detect",
        "model": "yolo26n.yaml",
        "name": "baseline-yolo26n",
        "batch": DEFAULT_BATCH,
        "description": "YOLO26n FPN teacher upper bound",
    },
    "dcn-solo": {
        "trainer": "detect",
        "model": "yolo26n-DCN.yaml",
        "name": "baseline-dcn-solo",
        "batch": DEFAULT_BATCH,
        "description": "YOLO26n-DCN (YOLOF head) without distillation",
    },
    "kd-p0": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-kd-p0",
        "batch": DEFAULT_BATCH,
        "description": "Online KD with dictionary, neck, and response distillation",
        # distillation
        "online_distill": True,
        "teacher_freeze_epoch": 110,  # 1-indexed; ep1-110 joint, ep111+ frozen teacher for distill only
        "teacher_freeze_use_ema": True,  # copy ema.ema.teacher into live teacher before freeze
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
        # backbone distillation: dictionary modules (student n10 ↔ teacher backbone taps)
        "dict_align_loss": 0.08,  # weighted (saliency) L2 between projected n10 and reorganized x4/x6
        "dict_attn_loss": 0.25,  # AT-style spatial attention restriction loss
        "dict_teacher_layers": [4, 6],
        "dict_student_layer": 10,  # student backbone output n10 (C2PSA)
        "dict_start_epoch": 0,
        "dict_weight": "saliency",  # saliency (|dL_task/dF|, falls back to attention when teacher frozen) | attention | none
    },
}


def build_overrides(baseline_key: str, args: argparse.Namespace) -> dict[str, Any]:
    """Merge COMMON + baseline-specific overrides + CLI overrides."""
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
        choices=[*BASELINES.keys(), "all"],
        default="yolo26n",
        help="which baseline to train (default: yolo26n)",
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
        help="checkpoint for --resume or --test-only (e.g. runs/detect/dfire-baselines/baseline-yolo26n/weights/best.pt)",
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

    if args.test_only:
        if args.baseline == "all":
            raise SystemExit("--test-only requires a single --baseline, not 'all'")
        if args.weights:
            best = Path(args.weights)
        else:
            overrides = build_overrides(args.baseline, args)
            best = weights_path(str(overrides["project"]), str(overrides["name"]))
        run_test_eval(best, args)
        return

    keys = list(BASELINES) if args.baseline == "all" else [args.baseline]
    for key in keys:
        best = train_baseline(key, args)
        if args.test_after_train:
            run_test_eval(best, args)


if __name__ == "__main__":
    main()
