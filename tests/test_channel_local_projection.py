"""Tests for the A-path: a channel-local LEARNABLE projection plus a trainable correspondence.

The two properties that make these arms interpretable at all, and the mistakes they guard against:

  1. The projection must be channel-local *structurally* (groups=C), because channel-locality is
     what keeps the correspondence visible in the loss. A mixing projection absorbs the pairing:
     measured, swapping the assignment costs 0.03% of the align loss through the shipped
     DeconvNet and 13.4% through a channel-local one.
  2. It must have LIVE parameters. A per-channel affine (1x1) looks channel-local and learnable
     but the align loss standardises each channel over space, which cancels an affine exactly:
     `standardize(a*S + b) = sign(a)*standardize(S)`. Its gradient is 1.4e-05 against 2.0e-02 for
     the 3x3 form, and that residual is pure eps leakage, so such a projection would sit at its
     initialisation for the whole run and measure nothing.

Also pinned: the arms must keep the encoders trainable (a frozen encoder cannot learn a
correspondence), and the alpha pair / commit sweep must each move exactly one key.
"""
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dfs import _DFIRE_KD_VARIANTS, _KD_EXTRA_KEYS, build_overrides  # noqa: E402
from ultralytics.nn.modules.yolof import (  # noqa: E402
    ChannelIdentityProjection,
    ChannelLocalProjection,
    DeconvNet,
    DictionaryModule,
)

C_T, C_S, T_SIZE, S_SIZE, GRID = 128, 256, 40, 20, 2
A_ARMS = ["alocst12", "alocst06", "alocsoft12", "alocsoft06",
          "alocstc00", "alocstc10", "alocstc50"]


def _args(variant):
    return Namespace(epochs=200, batch=None, device="0", workers=8, pretrained="yolo26n.pt",
                     project="runs/x", dataset="dfire", teacher_weights="", resume=False,
                     weights="", name_suffix="s", kd_variant=variant)


def _build(proj):
    torch.manual_seed(0)
    m = DictionaryModule(C_T, C_S, T_SIZE, S_SIZE, GRID, match="hard", temperature=0.07,
                         match_norm="l2", proj_form=proj if isinstance(proj, str) else "deconv")
    m.eval()
    return m


def test_channel_local_projection_is_channel_local_and_identity_at_init():
    proj = ChannelLocalProjection(C_S, T_SIZE, kernel=3)
    assert sum(p.numel() for p in proj.parameters()) == C_S * 10, "3x3 depthwise + bias per channel"

    x = torch.randn(2, C_S, S_SIZE, S_SIZE)
    y = proj(x)
    assert y.shape == (2, C_S, T_SIZE, T_SIZE)
    # identity at init up to the spatial resize
    assert torch.allclose(y, F.interpolate(x, size=(T_SIZE, T_SIZE), mode="bilinear",
                                           align_corners=False), atol=1e-5)
    # structural channel-locality: perturbing channel c must move ONLY channel c
    for c in (0, 7, C_S - 1):
        z = x.clone()
        z[:, c] = 0.0
        y2 = proj(z)
        moved = [i for i in range(C_S) if not torch.allclose(y2[:, i], y[:, i])]
        assert moved == [c], (c, moved[:5])


def test_channel_local_kernel_must_be_odd_and_positive():
    for bad in (0, -1, 2, 4):
        with pytest.raises(ValueError):
            ChannelLocalProjection(C_S, T_SIZE, kernel=bad)


def test_unknown_proj_form_is_rejected():
    with pytest.raises(ValueError, match="proj_form"):
        DictionaryModule(C_T, C_S, T_SIZE, S_SIZE, GRID, match="hard", proj_form="magic")


def test_per_channel_affine_would_be_a_dead_no_op_but_the_3x3_is_not():
    """The trap this design avoids, asserted so it cannot be reintroduced silently.

    The align loss standardises per channel over space; that cancels an affine exactly, leaving
    only eps leakage in its gradient, whereas a 3x3 depthwise filter survives and gets a real
    gradient.
    """
    def standardize(v, eps=1e-5):
        m = v.mean(dim=(2, 3), keepdim=True)
        s = v.std(dim=(2, 3), keepdim=True)
        return (v - m) / (s + eps)

    t = torch.randn(4, C_T, 8, 8)
    s = torch.randn(4, C_S, 4, 4)
    idx = torch.as_tensor(torch.arange(C_S) % C_T)
    target = standardize(t[:, idx]).detach()

    grads = {}
    for k in (1, 3):
        proj = ChannelLocalProjection(C_S, 8, kernel=k)
        pred = proj(s)
        loss = F.mse_loss(standardize(pred), target)
        (g,) = torch.autograd.grad(loss, [proj.conv.weight], allow_unused=True)
        grads[k] = float(g.norm())
    assert grads[1] < 1e-3, "the 1x1 affine must be effectively dead, got %g" % grads[1]
    assert grads[3] > grads[1] * 50, ("the 3x3 form must be clearly live: %g vs %g"
                                      % (grads[3], grads[1]))


@pytest.mark.parametrize("variant", A_ARMS)
def test_a_path_arms_are_configured_to_learn_the_correspondence(variant):
    assert variant in _DFIRE_KD_VARIANTS, variant
    for key in _DFIRE_KD_VARIANTS[variant]:
        assert key in _KD_EXTRA_KEYS, (variant, key)
    cfg = build_overrides("dcn-kd", _args(variant))
    assert cfg["dict_proj_form"] == "channel_local", cfg["dict_proj_form"]
    assert cfg["dict_proj_kernel"] == 3, cfg["dict_proj_kernel"]
    assert cfg["dict_match"] in {"straight_through", "soft"}, cfg["dict_match"]
    assert cfg["dict_match_init"] == "identity", "must start from a meaningful correspondence"
    assert cfg["dict_freeze_encoders"] is False, "a frozen encoder cannot learn the pairing"
    assert cfg["dict_attn_loss"] == 0.25


def test_alpha_pair_differs_only_by_align_weight():
    for a, b in (("alocst12", "alocst06"), ("alocsoft12", "alocsoft06")):
        ca, cb = build_overrides("dcn-kd", _args(a)), build_overrides("dcn-kd", _args(b))
        moved = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
        assert moved == ["dict_align_loss"], (a, b, moved)
        assert {ca["dict_align_loss"], cb["dict_align_loss"]} == {0.12, 0.06}


def test_commit_sweep_differs_only_by_commit_weight():
    base = build_overrides("dcn-kd", _args("alocstc00"))
    assert base["dict_commit_loss"] == 0.0
    for arm, weight in (("alocstc10", 0.10), ("alocstc50", 0.50)):
        other = build_overrides("dcn-kd", _args(arm))
        moved = sorted(k for k in set(base) | set(other) if base.get(k) != other.get(k))
        assert moved == ["dict_commit_loss"], (arm, moved)
        assert other["dict_commit_loss"] == weight


def test_st_and_soft_differ_only_by_the_matching_mode():
    a, b = build_overrides("dcn-kd", _args("alocst06")), build_overrides("dcn-kd", _args("alocsoft06"))
    moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert moved == ["dict_match"], moved
    assert {a["dict_match"], b["dict_match"]} == {"straight_through", "soft"}


def test_a_path_arms_really_build_a_channel_local_projection():
    """End-to-end: the config must reach the module and produce the expected parameter count."""
    torch.manual_seed(0)
    m = DictionaryModule(C_T, C_S, T_SIZE, S_SIZE, GRID, match="straight_through",
                         temperature=0.07, match_norm="l2", proj_form="channel_local",
                         proj_kernel=3, match_init="identity")
    assert isinstance(m.proj, ChannelLocalProjection)
    nproj = sum(p.numel() for p in m.proj.parameters())
    assert nproj == C_S * 10, nproj
    # a channel-mixing and a parameter-free form must have different counts, or the arms would be
    # indistinguishable from the ones already run
    assert nproj != sum(p.numel() for p in DeconvNet(C_S, C_S, S_SIZE, T_SIZE).parameters())
    assert nproj != sum(p.numel() for p in ChannelIdentityProjection(T_SIZE).parameters())


def test_differentiable_assignment_keeps_encoders_trainable():
    """straight_through/soft must not freeze the encoders, which `hard` does."""
    torch.manual_seed(0)
    st = DictionaryModule(C_T, C_S, T_SIZE, S_SIZE, GRID, match="straight_through",
                          proj_form="channel_local", proj_kernel=3, match_init="identity")
    assert st.differentiable_assignment is True
    st.freeze_encoders()          # only called by the trainer for hard/index/fixed
    frozen = all(not p.requires_grad for e in st.encoders() for p in e.parameters())
    assert frozen, "freeze_encoders() must work when explicitly invoked"

    torch.manual_seed(0)
    hard = DictionaryModule(C_T, C_S, T_SIZE, S_SIZE, GRID, match="hard",
                            proj_form="channel_local", proj_kernel=3, match_init="identity")
    assert hard.differentiable_assignment is False
