"""Smoke test for `ssm_sz_s6`: the same recipe with the tap moved to the matching stage.

`ssm_sz` pairs a 20x20 student tap with a 40x40 teacher tap, so its encoder upsamples and its
matcher interpolates. With the tap at n6 both sides are 40x40 and those two steps become
no-ops; if they still resample, the arm invents resolution and the pairing change measures
nothing.
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
        # dictionary branch as in `ssm_sz`
        dict_align_loss=0.1, dict_attn_loss=0.0, dict_commit_loss=0.0, dict_infomax_loss=0.1,
        dict_infomax_marginal_weight=0.0, dict_attn_start_epoch=0,
        dict_teacher_layers=[6], dict_student_layer=6,
        dict_start_epoch=0, dict_weight="activation_mean", dict_match="straight_through",
        dict_match_temp=0.07, dict_feature_norm="channel", dict_saliency_ema=0.9,
        dict_teacher_encoder=False, dict_student_encoder="depthwise_upsample",
        dict_match_pool="none", dict_proj_form="channel_local", dict_init_seed=1234,
        dict_match_log_interval=100,
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
    m = YOLOFDistillationModel("models/yolo26n-DCN.yaml", nc=2, ch=3, verbose=False)
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


def test_pairing_is_layer_6_with_identical_shapes(model):
    assert model._dict_student_layer == 6
    img = _images()
    with torch.no_grad():
        _, staps = model._forward_with_taps(model, img, {6})
        _, ttaps = model._forward_with_taps(model.teacher, img, {6})
    s, t = staps[6], ttaps[6]
    assert s.shape == t.shape, "S[6] %s vs T[6] %s" % (tuple(s.shape), tuple(t.shape))
    assert s.shape[1] == t.shape[1] == 128
    assert s.shape[-1] == t.shape[-1] == IMGSZ // 16
    print("ssm_sz_s6: S[6] == T[6] == %s" % (tuple(s.shape),))


def test_encoder_does_not_upsample_at_equal_size(model):
    """`depthwise_upsample` must degrade to a plain depthwise block, with NO transposed conv."""
    enc = model.dictionary_modules[0].query_enc
    kinds = [type(x).__name__ for x in enc.modules()]
    assert "ConvTranspose2d" not in kinds, (
        "P^D is upsampling even though S[6] and T[6] are the same size: %s" % kinds)
    convs = [x for x in enc.modules() if isinstance(x, torch.nn.Conv2d)]
    assert convs, "encoder has no conv"
    for c in convs:
        assert c.groups == c.in_channels == c.out_channels, "P^D must stay depthwise"
        assert c.stride == (1, 1), "no spatial change should be attempted at equal size"
    # the resampler still needs BN + activation even when the scale is 1
    assert any(isinstance(x, torch.nn.BatchNorm2d) for x in enc.modules()), "P^D lost its BN"
    assert any(isinstance(x, (torch.nn.ReLU, torch.nn.SiLU, torch.nn.LeakyReLU))
               for x in enc.modules()), "P^D lost its activation"


def test_full_resolution_matching_needs_no_interpolation(model):
    """match_pool='none' must find the two sides already at the same size."""
    mod = model.dictionary_modules[0]
    assert mod.match_pool == "none"
    img = _images()
    with torch.no_grad():
        _, staps = model._forward_with_taps(model, img, {6})
        sp, tr, commit, infomax = mod(staps[6].detach(), staps[6], collect_diagnostics=True)
    assert sp.shape == tr.shape, "%s vs %s" % (tuple(sp.shape), tuple(tr.shape))
    assert float(commit) > 0.0, "straight_through must expose the soft pi_hat"
    assert float(infomax) > 0.0, "the InfoMax term must be live with marginal_weight=0"
    stats = mod.last_match_stats or {}
    # 40x40 => 1600 full-resolution tokens, i.e. no pooling to a 2x2 grid
    assert int(stats.get("channel_effective_teacher_channels", 0)) > 0
    print("  full-res matching ok: s_proj %s  commit %.4f  infomax %.4f"
          % (tuple(sp.shape), float(commit), float(infomax)))


def test_loss_is_finite_and_every_dictionary_term_contributes(model):
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
    # the align term must not be zero, which is what the signed weight map used to produce
    assert float(items[5]) > 0.0, "J_SSM (dict align) is exactly 0"
    assert float(items[3]) > 0.0, "J_RKD (align) inactive"
    # dict_attn is still COMPUTED and reported (items[6] is the raw value) but its gain is 0, so it
    # must not enter the total. Check the gain, not the reported value.
    gains = model._dict_gains()
    beta_d, beta_a, beta_c, beta_i = gains
    assert beta_a == 0.0, "dict_attn gain must be 0 in this recipe, got %r" % beta_a
    assert beta_d > 0.0 and beta_i > 0.0, "align / infomax gains must be active: %s" % (gains,)
    assert beta_c == 0.0, "commit gain must be 0: this recipe has no cosine term"
    g = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("teacher.")
            and not n.startswith(("feature_projectors.", "dictionary_modules.")))
    assert g > 0.0, "no student gradient"
    print("  total=%.4f align=%.4f feature=%.4f dict_align=%.4f student_grad=%.4g"
          % (float(total), float(items[3]), float(items[4]), float(items[5]), g))


# ------------------------------------------------------------------ config plumbing
def _resolve(v):
    from scripts import train_dfs as T
    cfg = dict(T.CONTROLLED_ARGS)
    cfg.update(T._KD_RECIPE)
    T._apply_kd_variant(cfg, "dfire", v)
    return T._yolo_overrides(cfg)


def test_delta_against_ssm_sz_is_the_student_layer_only():
    a, b = _resolve("ssm_sz"), _resolve("ssm_sz_s6")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"dict_student_layer"}, sorted(diff)
    assert a.get("dict_student_layer") == 10
    assert b.get("dict_student_layer") == 6


def test_the_dictionary_branch_is_untouched_by_the_pairing_change():
    a, b = _resolve("ssm_sz"), _resolve("ssm_sz_s6")
    for k in ("dict_match", "dict_match_pool", "dict_weight", "dict_proj_form",
              "dict_teacher_encoder", "dict_student_encoder", "dict_align_loss",
              "dict_infomax_loss", "dict_infomax_marginal_weight", "dict_attn_loss",
              "dict_commit_loss", "dict_teacher_layers",
              "align_size_weight", "align_box_conf_weight", "feature_loss", "align_loss"):
        assert a.get(k) == b.get(k), "%s differs: %r vs %r" % (k, a.get(k), b.get(k))


def test_both_arms_share_the_same_loss_gains():
    b = _resolve("ssm_sz_s6")
    assert b["dict_align_loss"] == 0.1        # SSM
    assert b["dict_infomax_loss"] == 0.1      # InfoMax
    assert b["feature_loss"] == 0.08          # RFM
    assert b["align_loss"] == 0.12            # RKD
    assert b["dict_attn_loss"] == 0.0         # not used in this recipe
    assert b["dict_commit_loss"] == 0.0
