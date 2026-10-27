"""Smoke: gradient routing of the distillation terms, and inference independence.

  A. GAIN ISOLATION. The distillation terms are separate additive contributions to the total
     loss, so:
       A1  with the task gain set to 0, the distillation terms alone still deliver non-zero
           gradients to the student;
       A2  with every distillation gain set to 0, the task loss alone still trains the student.
     A failure of A1 means the task loss gates the auxiliary supervision, e.g. through an
     early return or a detached tensor.

  B. INFERENCE INDEPENDENCE. The teacher and the distillation modules are training-only, so
     neither the distillation gains nor the teacher's weights change a test-time forward.

Needs a real teacher checkpoint and real images: a randomly initialised teacher emits no boxes,
which leaves the align branch at exactly 0 and makes the gradient assertions vacuous.
"""
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

TEACHER_CKPT = os.environ.get("SMOKE_TEACHER", "Log/yolo26n_teacher_N200_best.pt")
IMAGE_DIR = os.environ.get("SMOKE_IMAGES", "E:/CRIS/D-Fire/test/images")
IMGSZ = 320
BATCH = 4


def _args(**over):
    base = dict(
        imgsz=IMGSZ,
        online_distill=False,          # teacher is a fixed target here: isolates the distill terms
        teacher_freeze_epoch=0,
        teacher_task_loss=0.0,
        task_loss=1.0,
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
        distill_conf_thres=0.001,      # guarantee the teacher contributes boxes on this input
        distill_iou_thres=0.5,
        dict_align_loss=0.1,
        dict_attn_loss=0.0,
        dict_commit_loss=0.0,
        dict_infomax_loss=0.1,
        dict_infomax_marginal_weight=0.0,
        dict_attn_start_epoch=0,
        dict_teacher_layers=[6],
        dict_student_layer=10,
        dict_start_epoch=0,
        dict_weight="activation_mean",
        dict_match="straight_through",
        dict_match_temp=0.07,
        dict_feature_norm="channel",
        dict_saliency_ema=0.9,
        dict_teacher_encoder=False,
        dict_student_encoder="depthwise_upsample",
        dict_match_pool="none",
        dict_proj_form="channel_local",
        dict_init_seed=1234,
        align_size_weight=True,
        align_size_weight_exp=1.0,
        align_size_weight_min=0.25,
        align_size_weight_max=4.0,
        align_box_conf_weight=True,
        max_det=300,
        box=7.5, cls=0.5, dfl=1.5, epochs=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _images():
    import numpy as np
    from PIL import Image

    files = sorted(p for p in pathlib.Path(IMAGE_DIR).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    picks = files[:: max(len(files) // BATCH, 1)][:BATCH]
    out = []
    for p in picks:
        with Image.open(p) as im:
            out.append(torch.from_numpy(
                np.asarray(im.convert("RGB").resize((IMGSZ, IMGSZ))).copy()
            ).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


def _batch(device):
    return {
        "img": _images().to(device),
        "cls": torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.25, 0.2],
             [0.4, 0.5, 0.3, 0.3]], device=device),
        "batch_idx": torch.arange(BATCH, device=device),
    }


@pytest.fixture(scope="module")
def harness():
    assert pathlib.Path(TEACHER_CKPT).is_file(), "teacher checkpoint missing at %s" % TEACHER_CKPT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build(**over):
        args = _args(**over)
        model = YOLOFDistillationModel("models/yolo26n-DCN.yaml", nc=2, ch=3, verbose=False)
        model.args = args
        model.nc = 2
        model.names = {0: "smoke", 1: "fire"}
        ck = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)
        teacher = ck.get("model", ck)
        assert isinstance(teacher, torch.nn.Module)
        teacher = teacher.float().eval().requires_grad_(False)
        teacher.nc = 2
        teacher.names = model.names
        teacher.args = args
        model.teacher = teacher
        model.build_distillation_modules(imgsz=IMGSZ)
        model.to(device)
        model.current_epoch = 0
        model._teacher_frozen = True
        return model

    return {"device": device, "build": build, "batch": _batch(device)}


def _capture_dict_losses(model):
    """Wrap `_dictionary_losses` to record its 4-tuple, which the reported loss items omit.

    loss_items only has slots for (box, cls, dfl, align, feature, dict_align, dict_attn, t_box,
    t_cls, t_dfl) -- commit and InfoMax are NOT reported, so they must be captured here or a
    silent InfoMax failure would be invisible.
    """
    rec = {}
    orig = model._dictionary_losses

    def wrapped(*a, **k):
        out = orig(*a, **k)
        rec["d_align"], rec["d_attn"], rec["d_commit"], rec["d_infomax"] = (
            float(out[0]), float(out[1]), float(out[2]), float(out[3]))
        return out

    model._dictionary_losses = wrapped
    return rec


def test_dictionary_modules_and_j_pro_plumbing_are_live(harness):
    model = harness["build"]()
    assert model.dictionary_modules, "no dictionary modules were built"
    assert float(model.dictionary_modules[0].infomax_marginal_weight) == 0.0, \
        "infomax_marginal_weight arrived as %r, not 0.0 (a `x or 1.0` fallback would do this)" % (
            model.dictionary_modules[0].infomax_marginal_weight,)
    rec = _capture_dict_losses(model)
    _run_loss(model, harness["batch"])
    assert rec["d_infomax"] > 0.0, \
        "the InfoMax term produced nothing: it must be H(T|S) > 0, not 0"


def test_the_align_term_is_not_silently_zero(harness):
    """Regression guard for the ill-posed signed channel-mean weight.

    With the signed map the mass-normalised mean returned exactly 0.0 (sum(W) ~ -1e12 from
    cancellation), so J_SSM was computed, reported as 0 and contributed nothing -- with no error.
    """
    model = harness["build"]()
    rec = _capture_dict_losses(model)
    _, items = _run_loss(model, harness["batch"])
    assert float(items[5]) > 0.0, "dict align is exactly 0: the weight map is ill-posed"
    assert rec["d_align"] > 0.0
    assert rec["d_attn"] > 0.0 and rec["d_commit"] > 0.0 and rec["d_infomax"] > 0.0, (
        "every dictionary term must be live: align=%.4f attn=%.4f commit=%.4f infomax=%.4f"
        % (rec["d_align"], rec["d_attn"], rec["d_commit"], rec["d_infomax"]))
    print("dict terms live: align=%.4f attn=%.4f commit=%.4f infomax=%.4f"
          % (rec["d_align"], rec["d_attn"], rec["d_commit"], rec["d_infomax"]))


def test_j_pro_reaches_the_total_loss(harness):
    """lambda_pro must actually move the total, otherwise the term is computed and discarded."""
    m_on = harness["build"](dict_infomax_loss=0.1)
    t_on, _ = _run_loss(m_on, harness["batch"])
    m_off = harness["build"](dict_infomax_loss=0.0)
    t_off, _ = _run_loss(m_off, harness["batch"])
    assert t_on != t_off, "dict_infomax_loss had no effect on the total loss"


def _student_only_params(model):
    """Everything trainable except the teacher and the auxiliary distillation modules."""
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("teacher."):
            continue
        if name.startswith("feature_projectors.") or name.startswith("dictionary_modules."):
            continue
        out.append((name, p))
    assert out, "no student parameters found"
    return out


def _grad_norm(params):
    return float(sum(float(p.grad.abs().sum()) for _, p in params if p.grad is not None))


def _run_loss(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        total, items = model.loss(dict(batch))
    except IndexError as exc:
        if "empty" in str(exc).lower():
            pytest.skip("no teacher boxes above threshold on this input: %s" % exc)
        raise
    assert torch.isfinite(total).all(), "non-finite total loss"
    total.backward()
    return float(total), items


# --------------------------------------------------------------- A1 : task gain = 0
def test_task_gain_zero_still_trains_the_student_through_distillation(harness):
    model = harness["build"](task_loss=0.0)
    rec = _capture_dict_losses(model)
    total, items = _run_loss(model, harness["batch"])
    align, feature, d_align = float(items[3]), float(items[4]), float(items[5])
    assert align > 0.0 or feature > 0.0 or d_align > 0.0, "no distillation term was active"
    assert rec.get("d_infomax", 0.0) > 0.0, "the InfoMax term produced nothing"
    g = _grad_norm(_student_only_params(model))
    assert g > 0.0, (
        "with task_loss=0 the student received NO gradient: the distillation terms are being "
        "gated by the task gain (early return / detached tensor), so the auxiliary supervision "
        "cannot act on its own"
    )
    print("task_gain=0 -> student grad mass %.6g  (align %.4f feature %.4f dict_align %.4f infomax %.4f)"
          % (g, align, feature, d_align, rec.get("d_infomax", float("nan"))))


# --------------------------------------------------------------- A2 : all distill gains = 0
def test_all_distill_gains_zero_still_trains_the_student_from_the_task_loss(harness):
    """The mirror of A1: the task loss must not depend on the distillation branch either."""
    model = harness["build"](feature_loss=0.0, align_loss=0.0, dict_align_loss=0.0,
                             dict_infomax_loss=0.0, dict_attn_loss=0.0, dict_commit_loss=0.0)
    total, items = _run_loss(model, harness["batch"])
    g = _grad_norm(_student_only_params(model))
    assert g > 0.0, "student received no gradient with every distillation gain at 0"
    assert float(total) == float(total), "non-finite total"
    print("all distill gains 0 -> student grad mass %.6g" % g)


# --------------------------------------------------------------- B : inference independence
def test_test_time_forward_is_independent_of_the_teacher_and_the_gains(harness):
    device = harness["device"]
    batch = harness["batch"]
    model = harness["build"]()

    model.eval()
    with torch.no_grad():
        base = model(batch["img"])
    base_t = base[0] if isinstance(base, (tuple, list)) else base
    base_t = base_t.clone()

    # (a) scramble the teacher: a training-only target must not reach the inference path
    with torch.no_grad():
        for p in model.teacher.parameters():
            p.add_(torch.randn_like(p) * 5.0)
    with torch.no_grad():
        after_teacher = model(batch["img"])
    at = after_teacher[0] if isinstance(after_teacher, (tuple, list)) else after_teacher
    assert torch.allclose(base_t, at, atol=1e-6), \
        "the teacher leaked into the test-time forward"

    # (b) flip every distillation gain: these only scale losses, so predictions must not move
    model.args.feature_loss = 0.0
    model.args.align_loss = 0.0
    model.args.dict_align_loss = 0.0
    model.args.dict_infomax_loss = 0.0
    model.args.align_size_weight = False
    model.args.align_box_conf_weight = False
    with torch.no_grad():
        after_gains = model(batch["img"])
    ag = after_gains[0] if isinstance(after_gains, (tuple, list)) else after_gains
    assert torch.allclose(base_t, ag, atol=1e-6), \
        "distillation gains changed the test-time forward"
    print("inference invariant to teacher weights and to every distillation gain")


# --------------------------------------------------------------- unit : zero weights
def test_zero_anchor_weights_do_not_kill_the_remaining_gradient():
    """A zero weight must zero only its own term, not the whole weighted mean."""
    W = YOLOFDistillationModel._weighted_align_mean
    v = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    w = torch.tensor([0.0, 1.0, 0.0, 2.0])
    out = W(w, v)
    out.backward()
    assert float(v.grad[0]) == 0.0 and float(v.grad[2]) == 0.0
    assert float(v.grad[1]) > 0.0 and float(v.grad[3]) > 0.0
    assert torch.isclose(out, torch.tensor(10.0 / 3.0), atol=1e-6)


def test_weighted_mean_stays_finite_when_every_weight_is_zero():
    W = YOLOFDistillationModel._weighted_align_mean
    v = torch.tensor([1.0, 2.0], requires_grad=True)
    out = W(torch.zeros(2), v)
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(v.grad).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
