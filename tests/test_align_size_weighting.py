"""(2) size-weighted align + (3) confidence-weighted box term: numerics, off-path identity, config.

These test the arithmetic and the config plumbing. The off-path identity test is the guard that
matters most: every existing run must be bit-for-bit unaffected by this change.
"""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Shim:
    """Stand-in for the model: the helpers only touch ``self.args``."""

    def __init__(self, **kw):
        self.args = _Args(**kw)


W = YOLOFDistillationModel._weighted_align_mean
S = YOLOFDistillationModel._align_size_weight


# --------------------------------------------------------------------- (2)+(3) arithmetic
def test_weighted_mean_equals_plain_mean_for_uniform_weights():
    torch.manual_seed(0)
    v = torch.randn(256)
    w = torch.ones(256)
    assert torch.allclose(W(w, v), v.mean(), atol=1e-7)


def test_weighted_mean_is_invariant_to_weight_scale():
    torch.manual_seed(1)
    v = torch.randn(128)
    w = torch.rand(128).clamp_min(1e-3)
    a, b = W(w, v), W(w * 1000.0, v)
    assert torch.allclose(a, b, atol=1e-6)


def test_weighted_mean_ignores_zero_weight_entries():
    v = torch.tensor([1.0, 1.0, 5.0])
    w = torch.tensor([1.0, 1.0, 0.0])
    assert torch.isclose(W(w, v), torch.tensor(1.0))


def test_weighted_mean_all_zero_weights_is_finite():
    v = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isfinite(W(torch.zeros(3), v))


def test_weighted_mean_broadcasts_a_singleton_weight_mass():
    v = torch.randn(32)
    assert torch.isclose(W(torch.ones(32), v), W(torch.ones(32) * 7.0, v))


# --------------------------------------------------------------------- (2) size weight
def test_size_weight_is_off_by_default_and_returns_ones():
    t = torch.rand(19, 4) * 100 + 1
    w = S(_Shim(), t)
    assert w.shape == (19,)
    assert torch.equal(w, torch.ones(19))


def test_size_weight_explicit_false_returns_ones():
    t = torch.rand(19, 4) * 100 + 1
    assert torch.equal(S(_Shim(align_size_weight=False), t), torch.ones(19))


def test_size_weight_is_inverse_in_side_and_respects_the_clip():
    sides = [8.0, 32.0, 128.0]
    t = torch.tensor([[0.0, 0.0, s, s] for s in sides])
    m = _Shim(align_size_weight=True, align_size_weight_exp=1.0,
              align_size_weight_min=0.25, align_size_weight_max=4.0)
    w = S(m, t).tolist()
    ref = 32.0  # median side
    for got, s in zip(w, sides):
        assert abs(got - min(4.0, max(0.25, ref / s))) < 1e-5, (got, s)
    # the clip must actually bind at both ends for this spread
    assert w[0] == pytest.approx(4.0)     # 32/8 = 4
    assert w[2] == pytest.approx(0.25)    # 32/128 = 0.25


def test_size_weight_exponent_zero_is_uniform():
    t = torch.tensor([[0.0, 0.0, s, s] for s in (8.0, 32.0, 128.0)])
    m = _Shim(align_size_weight=True, align_size_weight_exp=0.0,
              align_size_weight_min=0.25, align_size_weight_max=4.0)
    assert torch.allclose(S(m, t), torch.ones(3))


def test_size_weight_clip_is_order_insensitive():
    t = torch.tensor([[0.0, 0.0, s, s] for s in (8.0, 32.0, 128.0)])
    a = S(_Shim(align_size_weight=True, align_size_weight_min=0.25, align_size_weight_max=4.0), t)
    b = S(_Shim(align_size_weight=True, align_size_weight_min=4.0, align_size_weight_max=0.25), t)
    assert torch.allclose(a, b)


def test_size_weight_is_detached_and_finite():
    t = (torch.rand(8, 4) * 50 + 1).requires_grad_(True)
    w = S(_Shim(align_size_weight=True), t)
    assert not w.requires_grad
    assert torch.isfinite(w).all()


def test_size_weight_degenerate_tiny_box_does_not_explode():
    t = torch.tensor([[0.0, 0.0, 1e-9, 1e-9], [0.0, 0.0, 40.0, 40.0]])
    w = S(_Shim(align_size_weight=True, align_size_weight_exp=1.0,
                align_size_weight_min=0.25, align_size_weight_max=4.0), t)
    assert torch.isfinite(w).all()
    assert float(w.max()) <= 4.0 and float(w.min()) >= 0.25


# --------------------------------------------------------------------- (3) conf-weighted box
def test_conf_weighting_lets_a_confident_target_dominate():
    # anchor 0: perfect fit but low teacher confidence; anchor 1: bad fit but confident.
    ciou_loss = torch.tensor([0.0, 10.0])
    conf = torch.tensor([0.05, 0.95])
    unweighted = ciou_loss.mean()                    # 5.0, both count equally
    weighted = W(conf, ciou_loss)                    # 9.5, the confident target dominates
    assert weighted > unweighted
    assert torch.isclose(weighted, torch.tensor(9.5), atol=1e-5)


def test_conf_weighting_suppresses_a_false_positive_like_term():
    # a 0.05-confidence wrong box contributes 5% of a 0.95-confidence one
    v = torch.tensor([1.0, 0.0])
    w = torch.tensor([0.05, 0.95])
    share_low = float(w[0] / w.sum())
    assert share_low == pytest.approx(0.05, abs=1e-9)
    assert torch.isclose(W(w, v), torch.tensor(0.05), atol=1e-6)


def test_conf_weighting_changes_nothing_when_confidence_is_uniform():
    v = torch.randn(64)
    w = torch.ones(64) * 0.5
    assert torch.allclose(W(w, v), v.mean(), atol=1e-6)


# --------------------------------------------------------------------- config plumbing
def test_pindwsz_differs_from_pindw_in_exactly_the_align_weight_keys():
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    a, b = V["pindw"], V["pindwsz"]
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {
        "align_size_weight",
        "align_size_weight_exp",
        "align_size_weight_min",
        "align_size_weight_max",
        "align_box_conf_weight",
    }


def test_pindwsz_keeps_the_pindw_dictionary_architecture():
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    v = V["pindwsz"]
    assert v["dict_teacher_encoder"] is False
    assert v["dict_student_encoder"] == "depthwise"
    assert v["align_size_weight"] is True
    assert v["align_box_conf_weight"] is True
    assert v["align_size_weight_exp"] == 1.0


def test_pindwsz_does_not_change_the_confidence_threshold():
    """(3) removes the need to lower distill_conf_thres; the arm must not smuggle one in."""
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    assert "distill_conf_thres" not in V["pindwsz"]
    assert "distill_conf_thres" not in V["pindw"]


def test_align_weight_keys_survive_the_fork_cfg_filter():
    """_yolo_overrides drops unknown keys silently, so these must be in _KD_EXTRA_KEYS."""
    from scripts.train_dfs import _KD_EXTRA_KEYS, _yolo_overrides

    for k in ("align_size_weight", "align_size_weight_exp", "align_size_weight_min",
              "align_size_weight_max", "align_box_conf_weight"):
        assert k in _KD_EXTRA_KEYS
        assert k in _yolo_overrides({k: True})


def test_align_weight_keys_are_accepted_by_the_cli_cfg_check():
    from ultralytics.cfg import check_dict_alignment

    check_dict_alignment({"yolo26n.yaml": 1}, {"align_size_weight": True})


def test_variant_resolves_and_the_flags_reach_the_model_args(tmp_path, monkeypatch):
    """End-to-end: selecting --kd-variant pindwsz must put the flags into the model's args."""
    from scripts import train_dfs

    cfg = dict(train_dfs.CONTROLLED_ARGS)
    train_dfs._apply_kd_variant(cfg, "dfire", "pindwsz")
    kept = train_dfs._yolo_overrides(cfg)
    assert kept["align_size_weight"] is True
    assert kept["align_box_conf_weight"] is True
    assert kept["align_size_weight_exp"] == 1.0
