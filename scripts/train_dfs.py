"""Protocol-matched Ultralytics training for DFS, D-Fire, and VOC2007.

Default dataset is DFS (nc=3, 100 epochs). Pass ``--dataset dfire`` or
``--dataset voc2007`` for the same CONTROLLED_ARGS recipe.

VOC2007 is trainval (5011) / test (4952), nc=20. Init from COCO
``yolo26n.pt``. Never load a D-Fire nc=2 teacher into DFS or VOC2007.

    python scripts/convert_voc2007.py
    python scripts/train_dfs.py --dataset voc2007 --baseline yolo26n --name-suffix seed0
    python scripts/train_dfs.py --dataset voc2007 --baseline dcn-solo --name-suffix seed0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_TQDM_NONINTERACTIVE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

torch.backends.cudnn.benchmark = True

from dfire_parity.protocol import CONTROLLED_ARGS
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer, YOLOFDistillationTrainer

DATASETS: dict[str, dict[str, Any]] = {
    "dfs": {
        "yaml": ROOT / "ultralytics" / "cfg" / "datasets" / "dfs.yaml",
        "project": "dfs-baselines",
        "prefix": "dfs",
        "nc": 3,
        "epochs": 100,
        "description": "DFS fire/other/smoke",
    },
    "dfire": {
        "yaml": ROOT / "ultralytics" / "cfg" / "datasets" / "dfire.yaml",
        "project": "dfire-protocol-baselines",
        "prefix": "dfire",
        "nc": 2,
        "description": "D-Fire smoke/fire (protocol diagnostic)",
    },
    "voc2007": {
        "yaml": ROOT / "ultralytics" / "cfg" / "datasets" / "voc2007.yaml",
        "project": "voc2007-baselines",
        "prefix": "voc2007",
        "nc": 20,
        "description": "VOC2007 trainval/test (nc=20)",
    },
}

BASELINES: dict[str, dict[str, Any]] = {
    "yolo26n": {
        "trainer": "detect",
        "description": "YOLO26n FPN",
        "model": "yolo26n.yaml",
        "name": "yolo26n",
    },
    "dcn-solo": {
        "trainer": "detect",
        "description": "YOLOF-DCN student solo",
        "model": "yolo26n-DCN.yaml",
        "name": "dcn-solo",
    },
    "dcn-kd": {
        "trainer": "kd",
        "description": "YOLOF-DCN + dLdx dictionary KD (D_dldx_new_2 gains)",
        "model": "yolo26n-DCN.yaml",
        "name": "dcn-kd",
    },
}

# Same dictionary / response KD as Log/D_dldx_new_2, on CONTROLLED_ARGS (not optimizer=auto).
_KD_RECIPE: dict[str, Any] = {
    "teacher": "yolo26n.yaml",
    "online_distill": True,
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
    "dict_teacher_layers": [6],
    "dict_start_epoch": 0,
    "dict_weight": "saliency_dLdx",
    "dict_match": "hard",
    "dict_match_temp": 0.07,
    "dict_feature_norm": "channel",
    "dict_saliency_ema": 0.9,
    "dict_attn_start_epoch": 0,
    "dict_commit_loss": 0.0,
    "dict_align_loss": 0.12,
    "dict_attn_loss": 0.25,
    "dict_weight_norm": "mean",
    "dict_saliency_blur": 0.0,
    "dict_saliency_clip": 0.0,
}

_KD_EXTRA_KEYS = frozenset(_KD_RECIPE) | frozenset(
    ("teacher_weights", "teacher_freeze_epoch")
)


_REQUIRED_CFG = (
    "fixed_accumulate",
    "nbs",
    "warmup_epochs",
    "warmup_bias_lr",
    "optimizer",
    "mosaic",
    "amp",
)


def _protocol_overrides() -> dict[str, Any]:
    return dict(CONTROLLED_ARGS)


def resolve_voc2007_yaml() -> Path:
    """Prefer the yaml written next to the converted images (path: .)."""
    candidates = [
        Path(r"E:\CRIS\VOC数据集\yolo\voc2007.yaml"),
        Path("/root/datasets/VOC2007_yolo/voc2007.yaml"),
        ROOT / "datasets" / "VOC2007_yolo" / "voc2007.yaml",
        ROOT / "ultralytics" / "cfg" / "datasets" / "voc2007.yaml",
    ]
    with_images = [
        path
        for path in candidates
        if path.is_file() and (path.parent / "trainval" / "images").is_dir()
    ]
    if with_images:
        return with_images[0]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def _dataset_spec(args: argparse.Namespace) -> dict[str, Any]:
    dataset = getattr(args, "dataset", "dfs")
    if dataset not in DATASETS:
        raise ValueError(
            "Unknown dataset %s; choose %s" % (dataset, ", ".join(DATASETS))
        )
    spec = dict(DATASETS[dataset])
    if dataset == "voc2007":
        spec["yaml"] = resolve_voc2007_yaml()
    return spec


def build_overrides(baseline: str, args: argparse.Namespace) -> dict[str, Any]:
    if baseline not in BASELINES:
        raise ValueError(
            "Unknown baseline %s; choose %s" % (baseline, ", ".join(BASELINES))
        )
    spec = BASELINES[baseline]
    ds = _dataset_spec(args)
    cfg = _protocol_overrides()
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    elif ds.get("epochs") is not None:
        cfg["epochs"] = int(ds["epochs"])
    run_name = "%s-%s-%de" % (ds["prefix"], spec["name"], int(cfg["epochs"]))
    cfg.update(
        {
            "task": "detect",
            "mode": "train",
            "model": spec["model"],
            "data": str(ds["yaml"]),
            "project": args.project or ds["project"],
            "name": run_name,
            "exist_ok": False,
            "pretrained": args.pretrained or "yolo26n.pt",
            "verbose": True,
            "plots": False,
            "seed": 0,
            "device": 0 if args.device is None else args.device,
        }
    )
    if spec.get("trainer") == "kd":
        cfg.update(_KD_RECIPE)
        cfg["teacher_freeze_epoch"] = int(cfg["epochs"])
        teacher_weights = getattr(args, "teacher_weights", "") or cfg.get("pretrained")
        cfg["teacher_weights"] = teacher_weights
    if args.batch is not None:
        cfg["batch"] = int(args.batch)
        cfg["nbs"] = int(args.batch)
    if args.workers is not None:
        cfg["workers"] = int(args.workers)
    if args.name_suffix:
        cfg["name"] = "%s-%s" % (cfg["name"], args.name_suffix)
    if args.resume:
        cfg["resume"] = True
        if args.weights:
            cfg["model"] = args.weights
    return cfg


def assert_cfg_supported() -> None:
    """Fail fast if this Ultra tree is older than CONTROLLED_ARGS."""
    import ultralytics
    from ultralytics.utils import DEFAULT_CFG_DICT

    print("ultralytics:", ultralytics.__file__)
    missing = [key for key in _REQUIRED_CFG if key not in DEFAULT_CFG_DICT]
    if not missing:
        return
    raise SystemExit(
        "This Ultra tree is missing CONTROLLED_ARGS keys: %s\n"
        "You copied scripts/train_dfs.py onto an older ultralytics. Sync the current fork, at least:\n"
        "  ultralytics/cfg/default.yaml\n"
        "  ultralytics/engine/trainer.py\n"
        "  dfire_parity/\n"
        "  scripts/train_dfs.py\n"
        "Do not strip these keys: DFS was trained with this fork."
        % missing
    )


def _yolo_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep DEFAULT_CFG plus the fork's KD extras; drop leftover classify fields."""
    from ultralytics.utils import DEFAULT_CFG_DICT

    return {
        key: value
        for key, value in cfg.items()
        if key in DEFAULT_CFG_DICT or key in _KD_EXTRA_KEYS
    }


def weights_path(project: str, name: str, task: str = "detect") -> Path:
    """Match Ultralytics get_save_dir for a relative project name."""
    project_path = Path(project)
    if project_path.is_absolute():
        return project_path / name / "weights" / "best.pt"
    return Path("runs") / task / project / name / "weights" / "best.pt"


def train_one(baseline: str, args: argparse.Namespace) -> Path:
    ds = _dataset_spec(args)
    data_yaml = ds["yaml"]
    if not data_yaml.is_file():
        raise SystemExit("Missing %s" % data_yaml)
    spec = BASELINES[baseline]
    overrides = build_overrides(baseline, args)
    print("dataset:", args.dataset, ds["description"], "nc=%d" % ds["nc"])
    print("baseline:", baseline, spec["description"])
    print("model:", overrides["model"])
    print("data:", overrides["data"])
    print("pretrained:", overrides["pretrained"])
    print("epochs:", overrides["epochs"], "close_mosaic:", overrides["close_mosaic"])
    print(
        "protocol: CONTROLLED_ARGS (SGD, batch/nbs=112, mosaic=1.0, amp=True, seed=0)"
    )
    if spec.get("trainer") == "kd":
        print("teacher:", overrides.get("teacher"), overrides.get("teacher_weights"))
        print(
            "kd: dLdx hard match, dict_align=0.12, dict_attn=0.25, freeze_epoch=%s"
            % overrides.get("teacher_freeze_epoch")
        )
    if args.dataset == "dfire":
        print(
            "diagnostic: same recipe as DFS. Healthy D-Fire val mAP50 is ~0.70+; "
            "if this also stalls near 0.40, the stack is the problem."
        )
    assert_cfg_supported()
    trainer_cls = (
        YOLOFDistillationTrainer if spec.get("trainer") == "kd" else DetectionTrainer
    )
    trainer = trainer_cls(overrides=_yolo_overrides(overrides))
    trainer.train()
    best = trainer.save_dir / "weights" / "best.pt"
    print("best.pt ->", best)
    return best


def test_one(weights: Path, args: argparse.Namespace) -> None:
    overrides = build_overrides(args.baseline, args)
    model = YOLO(str(weights))
    model.val(
        data=overrides["data"],
        split="test",
        imgsz=overrides["imgsz"],
        batch=args.batch or overrides["batch"],
        device=overrides["device"],
        workers=args.workers if args.workers is not None else overrides["workers"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CONTROLLED_ARGS YOLO26n / YOLOF-DCN training (DFS or D-Fire)"
    )
    parser.add_argument("--baseline", default="dcn-solo", choices=list(BASELINES))
    parser.add_argument(
        "--dataset",
        default="dfs",
        choices=list(DATASETS),
        help="dfs, dfire, or voc2007 (VOC2007 trainval/test, nc=20)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        default="yolo26n.pt",
        help="COCO init weights (default: yolo26n.pt). Never pass a D-Fire nc=2 teacher to DFS/VOC2007.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Ultralytics project folder (default: dfs-baselines or dfire-protocol-baselines)",
    )
    parser.add_argument("--name-suffix", default="")
    parser.add_argument(
        "--teacher-weights",
        default="",
        help="DFS YOLO26n checkpoint for dcn-kd (do not pass a D-Fire nc=2 teacher).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weights", default="")
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_only:
        if args.weights:
            weights = Path(args.weights)
        else:
            overrides = build_overrides(args.baseline, args)
            weights = weights_path(str(overrides["project"]), str(overrides["name"]))
        if not weights.is_file():
            raise SystemExit("Missing weights: %s" % weights)
        test_one(weights, args)
        return
    train_one(args.baseline, args)


if __name__ == "__main__":
    main()
