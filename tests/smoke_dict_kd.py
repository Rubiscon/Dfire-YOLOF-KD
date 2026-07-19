"""Smoke test for backbone dictionary distillation (proposal fig. 1/2).

Builds YOLOFDistillationModel (student yolo26n-DCN + teacher yolo26n) with the new
dictionary modules and runs: joint-phase loss (Grad-CAM saliency) + backward,
frozen-phase loss (saliency EMA fallback) + backward, and an EMA-style deepcopy.
"""
from copy import deepcopy
from types import SimpleNamespace

import torch

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel
from ultralytics.nn.tasks import DetectionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgsz = 320  # small for speed

args = SimpleNamespace(
    imgsz=imgsz,
    online_distill=True,
    teacher_freeze_epoch=2,
    task_loss=1.0,
    teacher_task_loss=1.0,
    feature_norm="channel",
    feature_loss=0.08,
    align=True,
    align_start_epoch=0,
    align_loss=0.12,
    align_branch="one2many",
    align_cls_mode="kl",
    distill_temperature=3,
    align_box=2.0,
    align_cls=4.0,
    distill_conf_thres=0.25,
    distill_iou_thres=0.5,
    dict_align_loss=0.08,
    dict_attn_loss=0.25,
    dict_commit_loss=0.0,
    dict_attn_start_epoch=0,
    dict_teacher_layers=[4, 6],
    dict_student_layer=10,
    dict_start_epoch=0,
    dict_weight="attention",
    dict_match="hard",
    dict_match_temp=0.07,
    dict_feature_norm="channel",
    dict_saliency_ema=0.9,
    max_det=300,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    epochs=3,
)

nc = 2
model = YOLOFDistillationModel("yolo26n-DCN.yaml", nc=nc, ch=3, verbose=False)
model.args = args
model.nc = nc
model.names = {0: "smoke", 1: "fire"}

teacher = DetectionModel("yolo26n.yaml", nc=nc, ch=3, verbose=False)
teacher.nc = nc
teacher.names = model.names
teacher.args = args
teacher.train()
teacher.requires_grad_(True)
model.teacher = teacher

model.build_distillation_modules(imgsz=imgsz)
assert len(model.dictionary_modules) == 2, "expected dictionary modules for teacher layers 4 and 6"
assert model.dictionary_modules[0].match == "hard"
print("dictionary modules built:", [type(m).__name__ for m in model.dictionary_modules])

model.to(device)
model.train()

bs = 4
batch = {
    "img": torch.rand(bs, 3, imgsz, imgsz, device=device),
    "cls": torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=device),
    "bboxes": torch.tensor(
        [[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.25, 0.2], [0.4, 0.5, 0.3, 0.3]],
        device=device,
    ),
    "batch_idx": torch.tensor([0, 1, 2, 3], device=device),
}

# --- joint phase (epoch 0 < teacher_freeze_epoch): Grad-CAM saliency available ---
model.current_epoch = 0
total, items = model.loss(dict(batch))
assert items.shape[0] == 10, f"expected 10 loss items, got {items.shape[0]}"
assert torch.isfinite(total).all() and torch.isfinite(items).all(), "non-finite loss (joint)"
print("joint-phase items:", [round(float(x), 4) for x in items])
assert items[5] > 0.1, f"dict_loss should stay O(1) with channel norm, got {float(items[5])}"
assert items[6] > 0.05, f"dattn_loss should stay O(0.1–1) with AT sum reduction, got {float(items[6])}"
print("joint dict/dattn scale OK:", round(float(items[5]), 4), round(float(items[6]), 4))
total.backward()
proj_grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.dictionary_modules[0].proj.parameters()]
assert all(proj_grads), "dictionary projector received no gradient"
print("joint-phase backward OK; projector grads flow")
model.zero_grad(set_to_none=True)
teacher.zero_grad(set_to_none=True)

# --- frozen phase (epoch >= teacher_freeze_epoch): saliency EMA fallback ---
model.current_epoch = 2
model._apply_teacher_freeze_if_needed()
assert model._teacher_frozen, "teacher should be frozen at epoch 3 (1-indexed)"
total2, items2 = model.loss(dict(batch))
assert torch.isfinite(total2).all() and torch.isfinite(items2).all(), "non-finite loss (frozen)"
assert items2[5] > 0 and items2[6] > 0, "dict losses should still be active in frozen phase"
assert float(items2[7]) == 0.0, "teacher task loss must be 0 when frozen"
print("frozen-phase items:", [round(float(x), 4) for x in items2])
total2.backward()
print("frozen-phase backward OK")
model.zero_grad(set_to_none=True)

# --- val path: loss items padded to 10 ---
model.eval()
with torch.no_grad():
    vloss, vitems = model.loss(dict(batch))
assert vitems.shape[0] == 10, f"val items should be padded to 10, got {vitems.shape[0]}"
print("val-path loss items OK:", vitems.shape[0])
model.train()

# --- EMA/checkpoint deepcopy after a forward (non-leaf caches must be cleared) ---
_ = model._predict_once(batch["img"])  # populate last_feature caches and _student_tap
copy_model = deepcopy(model)
assert copy_model._student_tap is None
assert len(copy_model.dictionary_modules) == 2
print("deepcopy (EMA path) OK")

# Soft-match + Grad-CAM path (optional stack)
args.dict_match = "soft"
args.dict_commit_loss = 0.05
args.dict_weight = "saliency"
model2 = YOLOFDistillationModel("yolo26n-DCN.yaml", nc=nc, ch=3, verbose=False)
model2.args = args
model2.nc = nc
model2.names = model.names
teacher2 = DetectionModel("yolo26n.yaml", nc=nc, ch=3, verbose=False)
teacher2.nc = nc
teacher2.names = model.names
teacher2.args = args
teacher2.train()
teacher2.requires_grad_(True)
model2.teacher = teacher2
model2.build_distillation_modules(imgsz=imgsz)
model2.to(device)
model2.train()
model2.current_epoch = 0
total3, items3 = model2.loss(dict(batch))
assert torch.isfinite(total3).all()
assert len(model2._cached_saliency) >= 1
print("soft+saliency path OK; saliency layers:", list(model2._cached_saliency.keys()))

print("\nALL SMOKE TESTS PASSED")
