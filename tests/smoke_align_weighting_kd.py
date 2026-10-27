"""Model-level smoke for the `pindwsz` arm: (2) size-weighted align + (3) conf-weighted box.

This is the integration check that the unit tests cannot give. Its most important assertion is
that turning the two flags ON actually CHANGES the loss: this codebase has a documented failure
mode where a cfg key not listed in `_KD_EXTRA_KEYS` / the CLI allowlist is dropped silently, with
no error and no effect. A finite-loss assertion would pass in that case; a "the loss must change"
assertion cannot.

Also exercises the real `pindw` dictionary architecture (teacher encoder removed, depthwise
student encoder) so the two changes are validated together, exactly as the arm will run.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgsz = 320

# Overridable so the same file runs locally (Windows paths) and on the training hosts.
TEACHER_CKPT = os.environ.get("SMOKE_TEACHER", "Log/yolo26n_teacher_N200_best.pt")
IMAGE_DIR = os.environ.get("SMOKE_IMAGES", "E:/CRIS/D-Fire/test/images")


def build(batch_size=4):
    args = SimpleNamespace(
        imgsz=imgsz,
        # Offline distillation: the teacher is a frozen target, so the smoke does not depend on
        # the teacher receiving a task loss. A REAL teacher checkpoint is required: a randomly
        # initialised teacher produces no boxes above the NMS threshold, `_alignment_loss` then
        # returns exactly 0.0 and a flag comparison would be vacuous.
        online_distill=False,
        teacher_freeze_epoch=0,
        task_loss=1.0,
        teacher_task_loss=0.0,
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
        # A low threshold guarantees the teacher contributes boxes on this synthetic input.
        distill_conf_thres=0.001,
        distill_iou_thres=0.5,
        # --- pindw dictionary architecture (the advisor's configuration) ---
        dict_align_loss=0.12,
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
        dict_teacher_encoder=False,
        dict_student_encoder="depthwise",
        dict_init_seed=1234,
        # --- (2)+(3) under test ---
        align_size_weight=True,
        align_size_weight_exp=1.0,
        align_size_weight_min=0.25,
        align_size_weight_max=4.0,
        align_box_conf_weight=True,
        max_det=300,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        epochs=3,
    )
    model = YOLOFDistillationModel("yolo26n-DCN.yaml", nc=2, ch=3, verbose=False)
    model.args = args
    model.nc = 2
    model.names = {0: "smoke", 1: "fire"}

    # Real trained teacher (the same checkpoint the arm will distil from).
    ckpt = Path(TEACHER_CKPT)
    assert ckpt.is_file(), "teacher checkpoint missing at %s; the align path cannot be exercised" % ckpt
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    teacher = ck.get("model", ck)
    assert isinstance(teacher, torch.nn.Module), "unexpected teacher checkpoint layout"
    teacher = teacher.float().eval().requires_grad_(False)
    teacher.nc = 2
    teacher.names = model.names
    teacher.args = args
    model.teacher = teacher
    model.build_distillation_modules(imgsz=imgsz)
    model.to(device)
    model.train()
    model.current_epoch = 0
    model._teacher_frozen = True

    torch.manual_seed(0)
    # REAL images: a trained teacher emits no boxes on uniform noise, which would leave the align
    # term at exactly 0.0 and make the flag comparison vacuous.
    from PIL import Image

    img_dir = Path(IMAGE_DIR)
    assert img_dir.is_dir(), "test images not found at %s" % img_dir
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    picks = files[:: max(len(files) // batch_size, 1)][:batch_size]
    imgs = []
    for p in picks:
        with Image.open(p) as im:
            imgs.append(torch.from_numpy(
                np.asarray(im.convert("RGB").resize((imgsz, imgsz))).copy()
            ).permute(2, 0, 1).float() / 255.0)
    batch = {
        "img": torch.stack(imgs).to(device),
        "cls": torch.tensor([[0.0], [1.0], [0.0], [1.0]][:batch_size], device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.25, 0.2],
             [0.4, 0.5, 0.3, 0.3]][:batch_size],
            device=device,
        ),
        "batch_idx": torch.arange(batch_size, device=device),
    }
    return model, args, batch


def main():
    model, args, batch = build()

    # the pindw architecture must actually be built as requested
    mods = [type(m).__name__ for m in model.dictionary_modules]
    assert mods, "no dictionary modules were built"
    assert model.dictionary_modules[0].has_teacher_encoder is False, \
        "pindw requires the teacher-side encoder to be absent"
    print("dictionary modules:", mods, "has_teacher_encoder =",
          model.dictionary_modules[0].has_teacher_encoder)

    # ---- flags ON -----------------------------------------------------------------
    total_on, items_on = model.loss(dict(batch))
    assert torch.isfinite(total_on).all() and torch.isfinite(items_on).all(), "non-finite loss (ON)"
    print("flags ON  : total=%.6f  align=%.6f  feature=%.6f"
          % (float(total_on), float(items_on[3]), float(items_on[4])))
    total_on.backward()
    tunable = [p for n, p in model.named_parameters() if p.requires_grad and "dictionary" not in n]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tunable), \
        "no student gradient reached the backbone/head"
    print("flags ON  : backward OK, student grads flow")
    model.zero_grad(set_to_none=True)

    # ---- flags OFF: same batch, same inputs, only the two switches differ ----------
    args.align_size_weight = False
    args.align_box_conf_weight = False
    total_off, items_off = model.loss(dict(batch))
    assert torch.isfinite(total_off).all(), "non-finite loss (OFF)"
    print("flags OFF : total=%.6f  align=%.6f  feature=%.6f"
          % (float(total_off), float(items_off[3]), float(items_off[4])))

    align_on, align_off = float(items_on[3]), float(items_off[3])
    assert align_on != 0.0, (
        "the align term is zero, so the flag comparison is vacuous: the teacher produced no "
        "boxes above distill_conf_thres, or the align branch is inactive"
    )
    rel = abs(align_on - align_off) / max(abs(align_off), 1e-9)
    print("align term: ON=%.6f OFF=%.6f  relative difference=%.4f" % (align_on, align_off, rel))
    assert rel > 1e-6, (
        "the align term did not change when the flags were toggled -- the cfg keys are being "
        "dropped (check _KD_EXTRA_KEYS and the CLI allowlist), or the weighting is inert"
    )
    assert float(total_on) != float(total_off), "flags had no effect on the total loss"

    # the feature term must be untouched: (2)+(3) act on the align terms only
    assert float(items_on[4]) == float(items_off[4]), \
        "feature_loss changed; the weighting leaked outside the align terms"

    # ---- unit-level behaviour confirmed through the live code path -----------------
    t_box = torch.tensor([[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 128.0, 128.0]])
    args.align_size_weight = True
    w = model._align_size_weight(t_box)
    print("size weights for sides 8/32/128 px:", [round(float(x), 3) for x in w])
    assert float(w[0]) > float(w[1]) > float(w[2]), "weights are not decreasing in object size"
    assert torch.isfinite(w).all()

    # confidence weighting must make a confident target dominate the box term
    ciou = torch.tensor([0.0, 10.0])
    conf = torch.tensor([0.05, 0.95])
    assert model._weighted_align_mean(conf, ciou) > ciou.mean()
    print("conf-weighted box mean %.4f > unweighted %.4f"
          % (float(model._weighted_align_mean(conf, ciou)), float(ciou.mean())))

    print("\nALL ALIGN-WEIGHTING SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
