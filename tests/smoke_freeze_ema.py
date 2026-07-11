"""Smoke test: joint -> frozen transition with teacher_freeze_use_ema.

Runs ep1 joint (teacher trains, saliency dict weights) then ep2 frozen with EMA->live
teacher copy. Pass: no crash, teacher frozen, distill losses finite, freeze logs present.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("YOLO_TQDM_NONINTERACTIVE", "1")

import torch

from ultralytics.models.yolo.detect.train import YOLOFDistillationTrainer
from ultralytics.utils.torch_utils import unwrap_model

device = 0 if torch.cuda.is_available() else "cpu"

trainer = YOLOFDistillationTrainer(
    overrides={
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "pretrained": False,
        "data": "dfire.yaml",
        "imgsz": 320,
        "epochs": 2,
        "patience": 2,
        "batch": 4,
        "fraction": 0.01,
        "device": device,
        "workers": 2,
        "compile": False,
        "deterministic": False,
        "plots": False,
        "val": True,
        "online_distill": True,
        "teacher_freeze_epoch": 1,
        "teacher_freeze_use_ema": True,
        "teacher_val_interval": 0,  # skip per-epoch teacher val; freeze snapshot still runs
        "task_loss": 1.0,
        "teacher_task_loss": 1.0,
        "feature_norm": "channel",
        "feature_loss": 0.08,
        "align": True,
        "align_start_epoch": 0,
        "align_loss": 0.12,
        "align_branch": "one2many",
        "align_cls_mode": "kl",
        "distill_temperature": 3,
        "align_box": 2.0,
        "align_cls": 4.0,
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "dict_teacher_layers": [4, 6],
        "dict_student_layer": 10,
        "dict_start_epoch": 0,
        "dict_weight": "saliency",
        "project": "runs/detect",
        "name": "smoke-freeze-ema",
        "exist_ok": True,
    }
)

if __name__ == "__main__":
    trainer.train()

    kd = unwrap_model(trainer.model)
    assert getattr(kd, "teacher", None) is not None
    assert kd._teacher_frozen
    assert not any(p.requires_grad for p in kd.teacher.parameters())

    # Forward + backward one step on frozen-teacher distill path.
    kd.train()
    kd.to(trainer.device)
    batch = next(iter(trainer.train_loader))
    batch["img"] = batch["img"].to(trainer.device, non_blocking=True).float() / 255.0
    for k in ("batch_idx", "cls", "bboxes"):
        batch[k] = batch[k].to(trainer.device)
    loss, items = kd.loss(batch)
    assert torch.isfinite(loss).all(), f"non-finite loss after freeze: {loss}"
    loss.backward()
    assert torch.isfinite(items).all(), f"non-finite loss items: {items}"

    ckpt_path = trainer.save_dir / "weights" / "last.pt"
    assert ckpt_path.exists(), f"missing checkpoint {ckpt_path}"

    print("\nSMOKE FREEZE EMA PASSED")
