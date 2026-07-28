"""CPU smoke test for positive spatial-entropy weighted dictionary align."""

from types import SimpleNamespace

import torch

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel
from ultralytics.nn.tasks import DetectionModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgsz = 320
args = SimpleNamespace(
    imgsz=imgsz,
    online_distill=True,
    teacher_freeze_epoch=200,
    teacher_freeze_use_ema=True,
    task_loss=1.0,
    teacher_task_loss=1.0,
    feature_norm="channel",
    feature_loss=0.08,
    align=True,
    align_start_epoch=20,
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
    dict_infomax_loss=0.0,
    dict_attn_start_epoch=0,
    dict_teacher_layers=[6],
    dict_student_layer=10,
    dict_start_epoch=0,
    dict_weight="entropy",
    dict_match="hard",
    dict_match_temp=0.07,
    dict_match_norm="l2",
    dict_match_init="identity",
    dict_match_grid_divisor=16,
    dict_match_log_interval=0,
    dict_feature_norm="channel",
    dict_saliency_ema=0.9,
    dict_saliency_blur=0.0,
    dict_saliency_clip=0.0,
    dict_entropy_temp=0.1,
    dict_entropy_grid_divisor=4,
    dict_entropy_floor=0.1,
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
teacher.train().requires_grad_(True)
model.teacher = teacher
model.build_distillation_modules(imgsz=imgsz)

module = model.dictionary_modules[0]
assert module.match == "hard"

model.to(device).train()
batch = {
    "img": torch.rand(2, 3, imgsz, imgsz, device=device),
    "cls": torch.tensor([[0.0], [1.0]], device=device),
    "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]], device=device),
    "batch_idx": torch.tensor([0, 1], device=device),
}
total, items = model.loss(batch)
assert torch.isfinite(total).all() and torch.isfinite(items).all()
assert float(items[5]) > 0.0
total.backward()

proj_grads = [parameter.grad for parameter in module.proj.parameters() if parameter.requires_grad]
assert proj_grads and all(gradient is not None and torch.isfinite(gradient).all() for gradient in proj_grads)
assert any(float(gradient.abs().sum()) > 0.0 for gradient in proj_grads)

print("Entropy align integration smoke passed:", float(items[5]))
