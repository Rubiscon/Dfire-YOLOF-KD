"""Smoke: does `pindwsz_s6` (S[6] <-> T[6]) actually work end to end?

Pins three things the config alone cannot guarantee:
  1. the variant resolves with the shape-matched pairing and keeps (2)/(3) intact;
  2. the student tap at layer 6 is really captured during the loss (not silently None / 10);
  3. the residual is computed with NO upsampling, because S[6] and T[6] are both 40x40;
  4. it differs from `pindwsz` in exactly the pairing keys -- otherwise the arm is not a
     single-variable delta and nothing can be concluded from it.
"""
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel  # noqa: E402

TEACHER_CKPT = os.environ.get("SMOKE_TEACHER", "Log/yolo26n_teacher_N200_best.pt")
IMAGE_DIR = os.environ.get("SMOKE_IMAGES", "E:/CRIS/D-Fire/test/images")
IMGSZ = 320


def _args(**over):
    base = dict(
        imgsz=IMGSZ, online_distill=False, teacher_freeze_epoch=0, teacher_task_loss=0.0,
        task_loss=1.0, feature_norm="channel", feature_loss=0.08, align=True,
        align_start_epoch=0, align_loss=0.12, align_branch="one2many", align_cls_mode="kl",
        distill_temperature=3, align_box=2.0, align_cls=4.0, distill_conf_thres=0.001,
        distill_iou_thres=0.5,
        dict_align_loss=0.12, dict_attn_loss=0.25, dict_commit_loss=0.0, dict_infomax_loss=0.0,
        dict_attn_start_epoch=0, dict_teacher_layers=[6], dict_student_layer=6,
        dict_start_epoch=0, dict_weight="saliency_dLdx", dict_match="hard", dict_match_temp=0.07,
        dict_feature_norm="channel", dict_saliency_ema=0.9, dict_teacher_encoder=False,
        dict_student_encoder="depthwise", dict_init_seed=1234, dict_match_log_interval=100,
        align_size_weight=True, align_size_weight_exp=1.0, align_size_weight_min=0.25,
        align_size_weight_max=4.0, align_box_conf_weight=True,
        max_det=300, box=7.5, cls=0.5, dfl=1.5, epochs=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _images():
    import numpy as np
    from PIL import Image
    files = sorted(p for p in pathlib.Path(IMAGE_DIR).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = []
    for p in files[:: max(len(files) // 4, 1)][:4]:
        with Image.open(p) as im:
            out.append(torch.from_numpy(
                np.asarray(im.convert("RGB").resize((IMGSZ, IMGSZ))).copy()
            ).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


@pytest.fixture(scope="module")
def model():
    assert pathlib.Path(TEACHER_CKPT).is_file()
    args = _args()
    m = YOLOFDistillationModel("yolo26n-DCN.yaml", nc=2, ch=3, verbose=False)
    m.args, m.nc, m.names = args, 2, {0: "smoke", 1: "fire"}
    ck = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)
    t = ck.get("model", ck).float().eval().requires_grad_(False)
    t.nc, t.names, t.args = 2, m.names, args
    m.teacher = t
    m.build_distillation_modules(imgsz=IMGSZ)
    m.train()
    m.current_epoch = 0
    m._teacher_frozen = True
    return m


def test_student_tap_is_layer_6_and_matches_the_teacher_shape(model):
    assert model._dict_student_layer == 6
    assert model._dict_teacher_layers == [6]

    img = _images()
    with torch.no_grad():
        _, staps = model._forward_with_taps(model, img, {6})
        _, ttaps = model._forward_with_taps(model.teacher, img, {6})
    s, t = staps[6], ttaps[6]
    assert s.shape == t.shape, "S[6] %s vs T[6] %s" % (tuple(s.shape), tuple(t.shape))
    # 128ch, stride 16
    assert s.shape[1] == t.shape[1] == 128
    assert s.shape[-1] == t.shape[-1] == IMGSZ // 16
    print("S[6] == T[6] == %s  (no shape mismatch)" % (tuple(s.shape),))


def test_projection_does_not_resample(model):
    """Equal spatial size => P^D must not upsample, so no transposed conv may appear."""
    mod = model.dictionary_modules[0]
    assert mod.query_enc is not None
    kinds = [type(x).__name__ for x in mod.query_enc.modules()]
    assert "ConvTranspose2d" not in kinds, "P^D must not resample for S[6] <-> T[6]: %s" % kinds
    convs = [m for m in mod.query_enc.modules() if isinstance(m, torch.nn.Conv2d)]
    for c in convs:
        assert c.groups == c.in_channels == c.out_channels, "P^D must stay depthwise"
    img = _images()
    with torch.no_grad():
        _, staps = model._forward_with_taps(model, img, {6})
        sp, tr, _, _ = mod(staps[6].detach(), staps[6], collect_diagnostics=True)
    assert sp.shape == tr.shape, "%s vs %s" % (tuple(sp.shape), tuple(tr.shape))


def test_loss_runs_and_keeps_the_size_and_conf_weighting(model):
    img = _images()
    batch = {
        "img": img,
        "cls": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2],
                                [0.7, 0.6, 0.25, 0.2], [0.4, 0.5, 0.3, 0.3]]),
        "batch_idx": torch.arange(4),
    }
    model.zero_grad(set_to_none=True)
    total, items = model.loss(dict(batch))
    assert torch.isfinite(total).all()
    total.backward()
    assert float(items[3]) > 0.0, "align term (J_RKD) inactive"
    assert float(items[5]) > 0.0, "dict align (J_SSM) is exactly 0"
    g = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("teacher.")
            and not n.startswith(("feature_projectors.", "dictionary_modules.")))
    assert g > 0.0
    print("S[6] arm: total=%.4f align=%.4f feature=%.4f dict_align=%.4f student_grad=%.4g"
          % (float(total), float(items[3]), float(items[4]), float(items[5]), g))


# ------------------------------------------------------------------ config plumbing
def _resolve(v):
    from scripts import train_dfs as T
    cfg = dict(T.CONTROLLED_ARGS)
    cfg.update(T._KD_RECIPE)
    T._apply_kd_variant(cfg, "dfire", v)
    return T._yolo_overrides(cfg)


def test_variant_differs_from_pindwsz_in_exactly_the_pairing_keys():
    """The delta must be the pairing only. `pindwsz` inherits the recipe default student layer
    (10), so it does not carry the key explicitly; only `dict_student_layer` therefore differs,
    while `dict_teacher_layers` is the same [6] on both sides."""
    from scripts import train_dfs as T
    a, b = _resolve("pindwsz"), _resolve("pindwsz_s6")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"dict_student_layer"}, sorted(diff)
    # the effective pairing before the variant is the recipe default, i.e. layer 10
    assert int(T._KD_RECIPE["dict_student_layer"]) == 10
    assert b.get("dict_student_layer") == 6
    assert b.get("dict_teacher_layers") == [6] == a.get("dict_teacher_layers")


def test_variant_keeps_the_align_weighting_identical_to_pindwsz():
    a, b = _resolve("pindwsz"), _resolve("pindwsz_s6")
    for k in ("align_size_weight", "align_size_weight_exp", "align_size_weight_min",
              "align_size_weight_max", "align_box_conf_weight",
              "dict_student_encoder", "dict_teacher_encoder", "dict_align_loss", "dict_attn_loss"):
        assert a.get(k) == b.get(k), "%s changed: %r vs %r" % (k, a.get(k), b.get(k))
