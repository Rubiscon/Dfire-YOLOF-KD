"""Tests for the `ssm` / `ssm_sz` recipes.

Each test fixes one property of the shipped configuration or of the weighting code. The main
failure mode is an unregistered cfg key being dropped without an error, which makes a run use
different settings than the launch line names.
"""
import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel  # noqa: E402
from ultralytics.nn.modules.yolof import DictionaryModule, _build_encoder  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# --------------------------------------------------- channel-mean weight: S(f^T) = mean_c(f^T)
def test_activation_mean_weight_is_the_first_order_channel_mean():
    """The weight must be a rectified channel mean, not mean_c(F^2) and not a gradient mode."""
    f = torch.randn(2, 8, 4, 4)
    shipped = YOLOFDistillationModel._spatial_attention(f)      # mean_c(F^2), second order
    expected = torch.relu(f.mean(dim=1))                        # first order, non-negative
    assert shipped.shape == expected.shape == (2, 4, 4)
    assert not torch.allclose(shipped, expected), "attention mode must stay second order"
    assert float(expected.min()) >= 0.0, "a spatial attention weight must be non-negative"
    assert torch.allclose(expected, torch.relu(f.sum(dim=1) / f.shape[1]))


def test_signed_channel_mean_is_kept_as_a_separate_ablation():
    """The signed map makes the mass-normalised mean ill posed; it must not be the default."""
    assert YOLOFDistillationModel._dict_weight_mode(_Args(dict_weight="activation_mean")) == \
        "activation_mean"
    assert YOLOFDistillationModel._dict_weight_mode(_Args(dict_weight="activation_mean_signed")) == \
        "activation_mean_signed"
    x = torch.randn(1, 8, 6, 6)
    signed = x.mean(dim=1)
    rect = torch.relu(signed)
    assert float((signed < 0).float().mean()) > 0.0
    assert float(rect.min()) >= 0.0


def test_activation_mean_mode_is_recognised_by_the_weight_parser():
    for raw in ("activation_mean", "mean_activation", "feature_energy", "eq8", "ssm_eq8"):
        assert YOLOFDistillationModel._dict_weight_mode(_Args(dict_weight=raw)) == "activation_mean"
    # must not be aliased onto the historical modes
    assert YOLOFDistillationModel._dict_weight_mode(_Args(dict_weight="attention")) == "attention"
    assert YOLOFDistillationModel._dict_weight_mode(_Args(dict_weight="saliency_dLdx")) == "saliency_dldx"
    # the rectified mean needs no task-loss graph; requiring one would add a backward pass
    mode = YOLOFDistillationModel._dict_weight_mode(_Args(dict_weight="activation_mean"))
    assert mode not in YOLOFDistillationModel._SALIENCY_GRAD_MODES


# ------------------------------- InfoMax term: H(T|S) with a uniform student marginal
def test_infomax_with_zero_marginal_weight_is_pure_conditional_entropy():
    """With a uniform student marginal H(joint) = H(S) + H(T|S), so only H(T|S) needs a term."""
    torch.manual_seed(0)
    a = torch.softmax(torch.randn(3, 5, 7), dim=2)
    val, cond, marg = DictionaryModule.infomax_loss(a, 0.0)
    assert torch.allclose(val, cond), "marginal_weight=0 must drop the H(T) term"
    assert not torch.allclose(val, cond - marg)
    # and it must be the textbook sum of row entropies, averaged over rows
    eps = torch.finfo(a.dtype).eps
    ref = -(a * a.clamp_min(eps).log()).sum(dim=2).mean()
    assert torch.allclose(cond, ref, atol=1e-6)


def test_infomax_objective_is_minimised_by_a_peaked_assignment():
    """The term is an entropy minimisation: a confident matching must score lower."""
    n = 6
    peaked = torch.zeros(1, 4, n)
    peaked[0, :, 0] = 1.0
    flat = torch.full((1, 4, n), 1.0 / n)
    lo, _, _ = DictionaryModule.infomax_loss(peaked, 0.0)
    hi, _, _ = DictionaryModule.infomax_loss(flat, 0.0)
    assert float(lo) < float(hi)
    assert math.isclose(float(hi), math.log(n), rel_tol=1e-4), "uniform rows must give log(N)"


# --------------------------------------------- argmax forward, soft pi kept for the loss
def _module(match, pool, c_t=32, c_s=16, ts=8, ss=4, marginal=0.0):
    return DictionaryModule(c_t, c_s, ts, ss, grid=4, match=match, teacher_encoder=False,
                            student_encoder="depthwise_upsample" if pool == "none" else "depthwise",
                            init_seed=1234, match_pool=pool,
                            infomax_marginal_weight=marginal)


def test_hard_matching_does_not_produce_the_soft_terms():
    """`hard` never computes commit/InfoMax, which is why the ssm recipe uses straight_through."""
    torch.manual_seed(0)
    m = _module("hard", "avg").eval()
    t = torch.randn(1, 32, 8, 8)
    s = torch.randn(1, 16, 4, 4)
    _, _, commit, infomax = m(t, s)
    assert float(commit) == 0.0 and float(infomax) == 0.0


def test_straight_through_produces_the_soft_terms_and_is_hard_in_the_forward_pass():
    torch.manual_seed(0)
    # marginal=0 -> the module returns H(T|S) alone.
    m = _module("straight_through", "avg", marginal=0.0).eval()
    t = torch.randn(1, 32, 8, 8)
    s = torch.randn(1, 16, 4, 4)
    _, t_reorg, commit, infomax = m(t, s)
    assert float(commit) > 0.0, "commit must be live in straight_through"
    assert float(infomax) > 0.0, "the InfoMax term must be live and equal to H(T|S)"
    # forward must still be a SELECTION (argmax) of teacher channels, not a mixture: every output
    # channel must be numerically equal to one teacher channel (one-hot through the einsum)
    for cs in range(t_reorg.shape[1]):
        best = min(float((t_reorg[0, cs] - t[0, ct]).abs().max()) for ct in range(t.shape[1]))
        assert best < 1e-4, "student channel %d is a mixture, not a gathered teacher channel" % cs


def test_marginal_weight_zero_is_not_coerced_by_a_truthiness_fallback():
    """Guards the failure mode: `x or 1.0` turns 0.0 into 1.0 and the term into H(T|S) - H(T)."""
    src = pathlib.Path("ultralytics/models/yolo/detect/train.py").read_text(encoding="utf-8")
    assert 'dict_infomax_marginal_weight", 1.0) or 1.0' not in src, \
        "the `or 1.0` coercion is back: infomax_marginal_weight=0.0 would silently become 1.0"


# ------------------------------------------------- full-resolution matching, depthwise resampler
def test_match_pool_none_bypasses_the_pooling_layer():
    """Matching runs on the full teacher lattice, so the grid pooling must not be used."""
    torch.manual_seed(0)
    m = _module("straight_through", "none").eval()

    def _boom(*_a, **_k):
        raise AssertionError("pool() was called although match_pool='none'")

    m.pool.register_forward_hook(_boom)
    m(torch.randn(1, 32, 8, 8), torch.randn(1, 16, 4, 4))   # must not raise


def test_match_pool_avg_pools_both_sides():
    torch.manual_seed(0)
    m = _module("straight_through", "avg").eval()
    calls = {"n": 0}
    m.pool.register_forward_hook(lambda *_a, **_k: calls.__setitem__("n", calls["n"] + 1))
    m(torch.randn(1, 32, 8, 8), torch.randn(1, 16, 4, 4))
    assert calls["n"] == 2, "the pooled mode tokenises both the teacher and the student side"


def test_depthwise_upsample_resamples_without_channel_mixing():
    enc = _build_encoder(16, "depthwise_upsample", in_size=4, out_size=8)
    convs = [x for x in enc if isinstance(x, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    assert convs, "encoder must contain convolutions"
    for c in convs:
        assert c.groups == c.in_channels == c.out_channels, \
            "the student resampler must stay depthwise: channel mixing would absorb the match"
    x = torch.zeros(1, 16, 4, 4)
    out = enc(x)
    assert out.shape[-2:] == (8, 8), "the encoder must align the student resolution to the teacher's"
    # depthwise => each output channel depends on exactly one input channel
    with torch.no_grad():
        a = enc(x)
        x2 = x.clone()
        x2[0, 5] += 1.0
        d = (enc(x2) - a).abs().sum(dim=(2, 3))[0]
    assert float(d[5]) > 0
    assert float(d[[i for i in range(16) if i != 5]].sum()) < 1e-5


# ------------------------------------------------------------------ variant plumbing / config
SHIPPED_SSM = {
    "dict_teacher_encoder": False,
    "dict_student_encoder": "depthwise_upsample",
    "dict_match_pool": "none",
    "dict_match": "straight_through",
    "dict_weight": "activation_mean",
    "dict_proj_form": "channel_local",
    "dict_align_loss": 0.1,
    "dict_infomax_loss": 0.1,
    "dict_infomax_marginal_weight": 0.0,
    "dict_attn_loss": 0.0,
    "dict_commit_loss": 0.0,
    "feature_loss": 0.08,
    "align_loss": 0.12,
}

# The size and confidence gates applied on top of the dictionary recipe.
ADDITIONS = {
    "align_size_weight": True,
    "align_size_weight_exp": 1.0,
    "align_size_weight_min": 0.25,
    "align_size_weight_max": 4.0,
    "align_box_conf_weight": True,
}


def _resolve(variant):
    from scripts import train_dfs as T
    cfg = dict(T.CONTROLLED_ARGS)
    cfg.update(T._KD_RECIPE)
    T._apply_kd_variant(cfg, "dfire", variant)
    return T._yolo_overrides(cfg)


def test_ssm_has_the_shipped_weights_and_architecture():
    got = _resolve("ssm")
    for k, v in SHIPPED_SSM.items():
        assert got.get(k) == v, "%s resolved to %r, expected %r" % (k, got.get(k), v)


def test_ssm_does_not_carry_the_additions():
    got = _resolve("ssm")
    for k in ADDITIONS:
        assert k not in got, "%s must be absent from the plain ssm recipe" % k


def test_ssm_sz_differs_from_ssm_in_exactly_the_five_addition_keys():
    a, b = _resolve("ssm"), _resolve("ssm_sz")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == set(ADDITIONS), sorted(diff)
    for k, v in ADDITIONS.items():
        assert b.get(k) == v


def test_ssm_keys_survive_the_fork_cfg_filter():
    from scripts import train_dfs as T
    for k in ("dict_infomax_loss", "dict_infomax_marginal_weight"):
        assert k in T._KD_EXTRA_KEYS, "%s would be dropped silently" % k
        assert k in T._yolo_overrides({k: 0.1})
    for k in ("dict_match_pool", "dict_proj_form", "dict_match"):
        assert k in T._KD_EXTRA_KEYS


def test_cli_cfg_check_accepts_every_new_key():
    from ultralytics.cfg import check_dict_alignment
    check_dict_alignment({"yolo26n.yaml": 1}, dict(SHIPPED_SSM, **ADDITIONS))


def test_pindwsz_and_ssm_sz_differ_in_exactly_the_dictionary_side():
    """pindwsz -> ssm_sz must change the dictionary side only, leaving the gates intact."""
    a, b = _resolve("pindwsz"), _resolve("ssm_sz")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    ref = _resolve("pindw")
    expect = {k for k, v in SHIPPED_SSM.items() if ref.get(k) != v}
    assert diff == expect, "unexpected: %s / missing: %s" % (sorted(diff - expect),
                                                             sorted(expect - diff))
    for k in ADDITIONS:
        assert a.get(k) == b.get(k) == ADDITIONS[k], "%s changed between the two" % k


def test_pindwsz_and_ssm_sz_share_the_same_align_weighting_implementation():
    """The gate keys must resolve identically, so any difference comes from the dictionary side."""
    a, b = _resolve("pindwsz"), _resolve("ssm_sz")
    for k, v in ADDITIONS.items():
        assert a.get(k) == v and b.get(k) == v
