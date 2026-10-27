#!/usr/bin/env python3
"""Evaluate D-Fire val/test checkpoints written by the distillation trainer.

The distillation trainer saves a host checkpoint whose student weights live under a
``student.*`` / ``ema.*`` prefix.  ``YOLO.val`` cannot consume that layout directly, so this
script extracts the student state dict, rewrites it as a plain Ultralytics ``.pt`` and then
runs the standard validator on the val and test splits.

Example
-------
python scripts/eval_dfire.py --data configs/datasets/dfire.yaml --out eval/paper \
    --ckpt paper runs/detect/.../weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def _strip_prefix(key: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def student_state_from_host(ckpt: dict) -> dict[str, torch.Tensor]:
    """Prefer EMA student weights used at training-time val."""
    src = None
    prefixes = (
        "module.student.model.",
        "student.model.",
        "module.student_model.",
        "student_model.",
        "module.ema.",
        "ema.",
        "module.",
    )
    if "ema_state_dict" in ckpt and isinstance(ckpt["ema_state_dict"], dict):
        src = ckpt["ema_state_dict"]
    elif isinstance(ckpt.get("parity_engine"), dict) and isinstance(
        ckpt["parity_engine"].get("ema"), dict
    ):
        src = ckpt["parity_engine"]["ema"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        src = ckpt["state_dict"]
        prefixes = (
            "module.student.model.",
            "student.model.",
            "module.student_model.",
            "student_model.",
            "module.model.student_model.",
            "model.student_model.",
            "module.",
        )
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        src = ckpt["model"]
        prefixes = (
            "module.student.model.",
            "student.model.",
            "module.student_model.",
            "student_model.",
            "module.",
        )
    else:
        raise KeyError("no student tensor dict in checkpoint keys=%s" % list(ckpt)[:20])

    out = {}
    skipped = 0
    for key, value in src.items():
        if not torch.is_tensor(value):
            skipped += 1
            continue
        name = _strip_prefix(str(key), prefixes)
        if name.startswith("teacher") or name.startswith("teacher_model") or ".teacher" in name:
            continue
        if name.startswith(("loss_gid", "gid.", "cankd.", "kd_", "adapters.")):
            continue
        out[name] = value.float().cpu()
    if not out:
        raise RuntimeError("extracted 0 student tensors; sample=%s" % list(src)[:12])
    return out


def write_ultra_pt(state: dict[str, torch.Tensor], yaml_path: Path, dest: Path) -> None:
    model = DetectionModel(str(yaml_path), nc=2, verbose=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "anchor" not in k]
    if missing:
        raise RuntimeError("missing keys: %s" % missing[:12])
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.float(), "ema": None, "epoch": -1}, dest)


def run_val(weights: Path, data: Path, split: str, device: str, batch: int, imgsz: int):
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        plots=False,
        verbose=False,
    )
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--student-yaml", default=str(ROOT / "yolo26n-DCN.yaml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--ckpt", action="append", nargs=2, metavar=("NAME", "PATH"))
    parser.add_argument("--ultra", action="append", nargs=2, metavar=("NAME", "PATH"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    jobs = []
    for name, path in args.ultra or []:
        jobs.append(("ultra", name, Path(path)))
    for name, path in args.ckpt or []:
        jobs.append(("host", name, Path(path)))

    yaml_path = Path(args.student_yaml)
    for kind, name, path in jobs:
        print("====", kind, name, path, flush=True)
        if kind == "host":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = student_state_from_host(ckpt)
            weights = out_dir / f"{name}_student.pt"
            write_ultra_pt(state, yaml_path, weights)
        else:
            weights = path
        row = {"source": str(path)}
        for split in ("val", "test"):
            row[split] = run_val(weights, Path(args.data), split, args.device, args.batch, args.imgsz)
            print(name, split, row[split], flush=True)
        summary[name] = row
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
