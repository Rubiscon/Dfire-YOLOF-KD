"""Unified D-Fire baselines on official train/val/test split (dfire.yaml).

Baselines:
  yolo26n / dcn-solo / kd-early / kd-p0
  early-B..G  - attn / response-align / Grad-CAM / analytic-dLdA sweeps
  early-H1/H2 - saliency_dLdA × dict_align_loss (0.10 / 0.12)
  early-S1a/S1b - A-gated dLdA (+ optional blur/clip); main saliency track
  early-I1/I2 - attention ablations only (not the main recipe)

Usage:
  python scripts/train_baselines.py --baseline kd-early
  python scripts/train_baselines.py --baseline early-S1a,early-S1b  # gated dLdA track
  python scripts/train_baselines.py --baseline early-H1,early-H2 # dLdA × dict_align sweep
  python scripts/train_baselines.py --baseline early-I1,early-I2 # attention ablation
  python scripts/train_baselines.py --baseline all               # all registered baselines, in order
  python scripts/train_baselines.py --baseline kd-early --test-only

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

DEFAULT_BATCH = 112  # solo / KD when VRAM allows (matches n_kd_n_batch112)
DEFAULT_KD_BATCH = 112  # proven KD recipe; drop to 32 with --batch if saliency OOM

# Shared online KD settings — aligned to Log/n_kd_n_batch112 (best mAP50≈0.728).
_KD_COMMON: dict[str, Any] = {
    "online_distill": True,
    # 200 = never freeze during a 200-ep run (same as the winning log).
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
    # Winning run used broken |dL/dA| → always fell back to attention A as weight.
    # Keep attention weighting for recipe match; Grad-CAM via dict_weight=saliency.
    "dict_weight": "attention",
    "dict_match": "hard",  # match winning hard-argmax dictionary
    "dict_match_temp": 0.07,
    "dict_feature_norm": "channel",  # required for dict_loss ~O(1); none collapses KD
    "dict_saliency_ema": 0.9,
    "dict_attn_start_epoch": 0,
    "dict_commit_loss": 0.0,
    # Student init from YOLO26n backbone weights — largest gap vs early_7-19* scratch runs.
    "pretrained": "yolo26n.pt",
    "teacher_weights": "yolo26n.pt",
}

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
    "kd-early": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-kd-early",
        "batch": DEFAULT_KD_BATCH,
        # CrisReport early stage: student n10 learns local structure from teacher early tap x6 only.
        "description": "CrisReport early dictionary: n10↔x6, hard match, pretrained student",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
    },
    "kd-p0": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-kd-p0",
        "batch": DEFAULT_KD_BATCH,
        "description": "kd-early + Grad-CAM saliency (dict_weight=saliency ablation)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_weight": "saliency",  # Grad-CAM; compare against kd-early attention weights
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
    },
    # --- early hyperparameter sweep (vs kd-early / early-3) ---
    "early-B": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-B-attn035",
        "batch": DEFAULT_KD_BATCH,
        "description": "early sweep B: dict_attn_loss=0.35 (restore AT budget vs x6-only)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.35,
    },
    "early-C": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-C-attn040",
        "batch": DEFAULT_KD_BATCH,
        "description": "early sweep C: dict_attn_loss=0.40",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.40,
    },
    # Round-2 sweeps after B/C: higher AT hurt mAP50-95; pivot to lower AT / align / saliency.
    "early-D": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-D-attn020",
        "batch": DEFAULT_KD_BATCH,
        "description": "early sweep D: dict_attn_loss=0.20 (below early-3)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.20,
    },
    "early-E": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-E-align015",
        "batch": DEFAULT_KD_BATCH,
        "description": "early sweep E: align_loss=0.15 (attn kept at 0.25)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "align_loss": 0.15,
    },
    "early-F": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-F-saliency",
        "batch": DEFAULT_KD_BATCH,
        "description": "early sweep F: dict_weight=saliency (Grad-CAM), attn=0.25",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "dict_weight": "saliency",
    },
    "early-G": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-G-dLdA",
        "batch": DEFAULT_KD_BATCH,
        "description": "early sweep G: dict_weight=saliency_dLdA (analytic |∂L/∂A|), attn=0.25",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "dict_weight": "saliency_dLdA",
    },
    # --- Phase H: dLdA × dict_align (Log/S_0.10, S_0.12) ---
    # Result: H1 underperforms G; H2 recovers mAP50 but still trails attention@0.10 on mAP50-95.
    "early-H1": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-H1-dLdA-align010",
        "batch": DEFAULT_KD_BATCH,
        "description": "H1: saliency_dLdA + dict_align_loss=0.10 (attn=0.25)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.10,
        "dict_attn_loss": 0.25,
        "dict_weight": "saliency_dLdA",
    },
    "early-H2": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-H2-dLdA-align012",
        "batch": DEFAULT_KD_BATCH,
        "description": "H2: saliency_dLdA + dict_align_loss=0.12 (attn=0.25)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.12,
        "dict_attn_loss": 0.25,
        "dict_weight": "saliency_dLdA",
    },
    # --- Phase S1: stabilize analytic |∂L/∂A| (main track; λ locked to G) ---
    "early-S1a": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-S1a-dLdA-gate",
        "batch": DEFAULT_KD_BATCH,
        "description": "S1a: saliency_dLdA_gate (A·|∂L/∂A|), align=0.08, attn=0.25",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "dict_weight": "saliency_dLdA_gate",
        "dict_saliency_blur": 0.0,
        "dict_saliency_clip": 0.0,
    },
    "early-S1b": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-S1b-dLdA-gate-stable",
        "batch": DEFAULT_KD_BATCH,
        "description": "S1b: dLdA_gate + blurσ=1 + clip=0.9 (align=0.08, attn=0.25)",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "dict_weight": "saliency_dLdA_gate",
        "dict_saliency_blur": 1.0,
        "dict_saliency_clip": 0.9,
    },
    # --- Phase I: attention ablations only (upper-bound / report controls) ---
    "early-I1": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-I1-attn-align012",
        "batch": DEFAULT_KD_BATCH,
        "description": "ABLATION I1: attention + dict_align_loss=0.12",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.12,
        "dict_attn_loss": 0.25,
        "dict_weight": "attention",
    },
    "early-I2": {
        "trainer": "kd",
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "name": "baseline-early-I2-attn-align010-attn020",
        "batch": DEFAULT_KD_BATCH,
        "description": "ABLATION I2: attention + dict_align=0.10 + dict_attn=0.20",
        **_KD_COMMON,
        "dict_teacher_layers": [6],
        "dict_align_loss": 0.10,
        "dict_attn_loss": 0.20,
        "dict_weight": "attention",
    },
}


def resolve_baseline_keys(spec: str) -> list[str]:
    """Parse ``--baseline``: ``all``, a single name, or comma-separated names (e.g. early-B,early-C)."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty --baseline")
    if spec == "all":
        return list(BASELINES)
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in BASELINES]
    if unknown:
        raise ValueError(f"Unknown baseline(s) {unknown}. Choose from: {list(BASELINES)} or 'all'")
    return keys


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
        default="yolo26n",
        help=(
            "baseline name, comma-separated list, or 'all'. "
            f"Available: {', '.join(BASELINES)} (e.g. early-B,early-C)"
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
