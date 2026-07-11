"""End-to-end trainer smoke for kd-p0 + backbone dictionary distillation (tiny fraction, 1 epoch)."""

import os

os.environ.setdefault("YOLO_TQDM_NONINTERACTIVE", "1")

from ultralytics.models.yolo.detect.train import YOLOFDistillationTrainer

trainer = YOLOFDistillationTrainer(
    overrides={
        "model": "yolo26n-DCN.yaml",
        "teacher": "yolo26n.yaml",
        "pretrained": False,
        "data": "dfire.yaml",
        "imgsz": 640,
        "epochs": 2,
        "patience": 2,
        "batch": 8,
        "fraction": 0.01,
        "device": 0,
        "workers": 2,
        "compile": False,
        "deterministic": False,
        "plots": False,
        "online_distill": True,
        "teacher_freeze_epoch": 1,  # exercise joint (ep1) AND frozen (ep2) phases
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
        "distill_conf_thres": 0.25,
        "distill_iou_thres": 0.5,
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.25,
        "dict_teacher_layers": [4, 6],
        "dict_student_layer": 10,
        "dict_start_epoch": 0,
        "dict_weight": "saliency",
        "project": "runs",
        "name": "kd-dict-smoke",
        "exist_ok": True,
    }
)

if __name__ == "__main__":
    trainer.train()
    print("\nTRAINER SMOKE PASSED")
