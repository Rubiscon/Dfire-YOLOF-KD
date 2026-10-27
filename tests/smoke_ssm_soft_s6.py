"""Smoke test for `ssm_soft_s6`, which differs from `ssm_sz_s6` only in the assignment mode.

Checks that:
  1. the two variants differ in exactly one key, so any metric difference comes from soft
     assignment;
  2. `soft` produces the pi-weighted mixture of teacher slices and `straight_through` produces a
     selection, which is the observable difference between the two modes;
  3. the double sum ``sum_i pi_ij ||a-b_i||^2`` equals ``||a - sum_i pi_ij b_i||^2`` plus a term
     that does not depend on the student, which is why the soft mode needs no new module code.
"""
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel  # noqa: E402
from ultralytics.nn.modules.yolof import DictionaryModule  # noqa: E402

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
        dict_align_loss=0.1, dict_attn_loss=0.0, dict_commit_loss=0.0, dict_infomax_loss=0.1,
        dict_infomax_marginal_weight=0.0, dict_attn_start_epoch=0,
        dict_teacher_layers=[6], dict_student_layer=6,
        dict_start_epoch=0, dict_weight="activation_mean", dict_match="soft",
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


# ---------------------------------------------- why the soft mode needs no new module code
def test_eq7_double_sum_equals_residual_against_the_pi_weighted_mean():
    torch.manual_seed(0)
    B, Cs, Ct, H, W = 1, 6, 5, 4, 4
    a = torch.randn(B, Cs, H, W, dtype=torch.float64)
    b = torch.randn(B, Ct, H, W, dtype=torch.float64)
    pi = F.softmax(torch.randn(B, Cs, Ct, dtype=torch.float64), dim=2)
    M = torch.rand(B, 1, H, W, dtype=torch.float64)

    lhs = torch.zeros((), dtype=torch.float64)
    for bi in range(B):
        for j in range(Cs):
            for i in range(Ct):
                lhs = lhs + pi[bi, j, i] * (((a[bi, j] - b[bi, i]).pow(2)) * M[bi, 0]).sum()

    mix = torch.einsum("bjc,bchw->bjhw", pi, b)
    rhs = (((a - mix).pow(2)) * M).sum()

    var = torch.zeros((), dtype=torch.float64)
    for bi in range(B):
        for j in range(Cs):
            w = pi[bi, j].view(Ct, 1, 1)
            varmap = (w * b[bi].pow(2)).sum(0) - (w * b[bi]).sum(0).pow(2)
            var = var + (varmap * M[bi, 0]).sum()

    assert torch.allclose(lhs, rhs + var, atol=1e-9), (lhs.item(), rhs.item(), var.item())
    assert var > 0, "the variance term should be positive for a non-degenerate pi"

    # gradients must agree as well, which is the property that matters for training
    a1 = a.clone().requires_grad_(True)
    a2 = a.clone().requires_grad_(True)
    l1 = torch.zeros((), dtype=torch.float64)
    for bi in range(B):
        for j in range(Cs):
            for i in range(Ct):
                l1 = l1 + pi[bi, j, i] * (((a1[bi, j] - b[bi, i]).pow(2)) * M[bi, 0]).sum()
    l2 = (((a2 - mix).pow(2)) * M).sum()
    g1 = torch.autograd.grad(l1, a1)[0]
    g2 = torch.autograd.grad(l2, a2)[0]
    assert torch.allclose(g1, g2, atol=1e-9), (g1 - g2).abs().max().item()
    print("Eq7 identity ok: |lhs-(rhs+var)|=%.2e  grad diff=%.2e"
          % (abs(lhs.item() - (rhs + var).item()), (g1 - g2).abs().max().item()))


# ------------------------------------------------------- soft = mixture, hard = selection
def _mod(match):
    # infomax_marginal_weight=0 -> the term is the pure conditional entropy H(T|S) > 0, which is
    # what a uniform student marginal implies. With the default 1.0 the value is
    # H(T|S) - H(T), which can legitimately be negative.
    return DictionaryModule(64, 32, 8, 8, grid=4, match=match, teacher_encoder=False,
                            student_encoder="depthwise", init_seed=1234, match_pool="none",
                            infomax_marginal_weight=0.0)


def test_soft_mode_returns_a_mixture_not_a_selection():
    torch.manual_seed(0)
    m = _mod("soft").eval()
    t = torch.randn(1, 64, 8, 8)
    s = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        _, t_reorg, commit, infomax = m(t, s)
    # a mixture must differ from every individual teacher channel
    same = [float((t_reorg[0, 0] - t[0, c]).abs().max()) for c in range(t.shape[1])]
    assert min(same) > 1e-4, "soft output matched a single teacher channel: it is a selection"
    assert float(commit) > 0 and float(infomax) > 0, "soft must expose the InfoMax term too"
    print("soft: min distance to any teacher channel = %.4f (mixture, as intended)"
          % min(same))


def test_straight_through_mode_returns_a_selection():
    torch.manual_seed(0)
    m = _mod("straight_through").eval()
    t = torch.randn(1, 64, 8, 8)
    s = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        _, t_reorg, _, _ = m(t, s)
    for cs in range(t_reorg.shape[1]):
        best = min(float((t_reorg[0, cs] - t[0, ct]).abs().max()) for ct in range(t.shape[1]))
        assert best < 1e-4, "channel %d is not a gathered teacher channel" % cs
    print("straight_through: every channel is exactly one teacher channel (selection)")


# ------------------------------------------------------------------ end to end
def test_pairing_and_shapes(model):
    assert model._dict_student_layer == 6
    assert model.dictionary_modules[0].match == "soft"
    img = _images()
    with torch.no_grad():
        _, staps = model._forward_with_taps(model, img, {6})
        _, ttaps = model._forward_with_taps(model.teacher, img, {6})
    assert staps[6].shape == ttaps[6].shape
    assert staps[6].shape[1] == 128
    enc = model.dictionary_modules[0].query_enc
    kinds = [type(x).__name__ for x in enc.modules()]
    assert "ConvTranspose2d" not in kinds, "P^D must not resample at equal size: %s" % kinds
    print("pairing ok: S[6] == T[6] == %s, no upsampling" % (tuple(staps[6].shape),))


def test_loss_is_finite_and_all_terms_contribute(model):
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
    assert float(items[5]) > 0.0, "J_SSM is zero"
    assert float(items[3]) > 0.0, "J_RKD inactive"
    beta_d, beta_a, beta_c, beta_i = model._dict_gains()
    assert beta_d > 0 and beta_i > 0 and beta_a == 0 and beta_c == 0, (beta_d, beta_a, beta_c, beta_i)
    g = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("teacher.")
            and not n.startswith(("feature_projectors.", "dictionary_modules.")))
    assert g > 0
    print("loss ok: total=%.4f J_SSM=%.4f J_RKD=%.4f student_grad=%.4g"
          % (float(total), float(items[5]), float(items[3]), g))


# ------------------------------------------------------------------ config plumbing
def _resolve(v):
    from scripts import train_dfs as T
    cfg = dict(T.CONTROLLED_ARGS)
    cfg.update(T._KD_RECIPE)
    T._apply_kd_variant(cfg, "dfire", v)
    return T._yolo_overrides(cfg)


def test_delta_against_ssm_sz_s6_is_only_the_match_mode():
    a, b = _resolve("ssm_sz_s6"), _resolve("ssm_soft_s6")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"dict_match"}, sorted(diff)
    assert a["dict_match"] == "straight_through"
    assert b["dict_match"] == "soft"


def test_everything_else_matches_the_rest_of_the_recipe():
    b = _resolve("ssm_soft_s6")
    a = _resolve("ssm_sz_s6")
    for k in ("dict_student_layer", "dict_teacher_layers", "dict_student_encoder",
              "dict_match_pool", "dict_weight", "dict_proj_form", "dict_teacher_encoder",
              "dict_align_loss", "dict_infomax_loss", "dict_infomax_marginal_weight",
              "dict_attn_loss", "dict_commit_loss", "align_size_weight",
              "align_box_conf_weight", "feature_loss", "align_loss"):
        assert a.get(k) == b.get(k), "%s differs: %r vs %r" % (k, a.get(k), b.get(k))
    assert b["dict_student_layer"] == 6
    assert b["dict_match_pool"] == "none"
