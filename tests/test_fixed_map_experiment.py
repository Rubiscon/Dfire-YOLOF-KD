"""Tests for the fixed-assignment experiment.

The experiment only means anything if the correspondence assignment can actually influence
the objective. The shipped module cannot, because its projection mixes all student channels
into every output channel and each output channel has its own weights: reassigning targets
is then a relabelling of independent sub-problems. These tests pin down both halves of that
statement, so a future change cannot silently restore the absorption.
"""
import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel
from ultralytics.nn.modules.yolof import ChannelIdentityProjection, DeconvNet, DictionaryModule


def test_identity_projection_is_parameter_free_and_channel_local():
    """Output channel c' must depend on student channel c' alone."""
    proj = ChannelIdentityProjection(8)
    assert sum(p.numel() for p in proj.parameters()) == 0

    x = torch.randn(2, 4, 3, 3)
    y = proj(x)
    assert y.shape == (2, 4, 8, 8), y.shape
    for c in range(4):
        z = x.clone()
        z[:, c] = 0.0
        y2 = proj(z)
        changed = [i for i in range(4) if not torch.allclose(y2[:, i], y[:, i])]
        assert changed == [c], (c, changed)


def test_deconv_projection_mixes_channels():
    """The shipped projection is NOT channel local -- this is what absorbs the pairing."""
    torch.manual_seed(0)
    proj = DeconvNet(4, 4, 3, 6)
    x = torch.randn(1, 4, 3, 3)
    y = proj(x)
    z = x.clone()
    z[:, 0] = 0.0
    y2 = proj(z)
    changed = [i for i in range(4) if not torch.allclose(y2[:, i], y[:, i])]
    assert len(changed) > 1, changed


def test_fixed_match_uses_the_given_map_exactly_and_builds_no_encoders():
    torch.manual_seed(0)
    c_t, c_s, t_size, s_size = 8, 12, 8, 4
    fixed = torch.tensor([3, 3, 0, 7, 1, 2, 5, 0, 6, 4, 4, 1], dtype=torch.long)
    mod = DictionaryModule(
        c_t, c_s, t_size, s_size, 2,
        match="fixed", fixed_map=fixed, proj_form="identity", match_init="identity",
    )
    assert mod.key_enc is None and mod.query_enc is None
    assert mod.encoders() == []
    assert sum(p.numel() for p in mod.proj.parameters()) == 0

    t = torch.randn(2, c_t, t_size, t_size)
    s = torch.randn(2, c_s, s_size, s_size)
    with torch.no_grad():
        s_proj, t_reorg, commit, infomax = mod(t, s)
    # the target for student channel c is exactly the teacher channel fixed[c]
    assert torch.equal(t_reorg, t[:, fixed])
    # and the student side is the student tap itself, only spatially resized
    assert torch.equal(s_proj, torch.nn.functional.interpolate(
        s, size=(t_size, t_size), mode="bilinear", align_corners=False))
    assert float(commit) == 0.0 and float(infomax) == 0.0


def test_fixed_match_requires_and_validates_the_map():
    with pytest.raises(ValueError, match="requires fixed_map"):
        DictionaryModule(8, 12, 8, 4, 2, match="fixed")
    with pytest.raises(ValueError, match="3 entries"):
        DictionaryModule(8, 12, 8, 4, 2, match="fixed", fixed_map=[0, 1, 2])
    with pytest.raises(ValueError, match=r"\[0, 8\)"):
        DictionaryModule(8, 12, 8, 4, 2, match="fixed", fixed_map=[0] * 11 + [99])


def test_proj_form_unknown_is_rejected_and_default_is_unchanged():
    with pytest.raises(ValueError, match="proj_form"):
        DictionaryModule(8, 12, 8, 4, 2, match="hard", proj_form="magic")
    # the default must stay the shipped DeconvNet so historical runs are untouched
    mod = DictionaryModule(8, 12, 8, 4, 2, match="hard")
    assert isinstance(mod.proj, DeconvNet)


def test_load_fixed_match_map_verifies_sha_and_reports_stats(tmp_path):
    m = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    f = tmp_path / "map.npy"
    np.save(f, m)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()

    got = YOLOFDistillationModel._load_fixed_match_map(str(f), sha, 8, 4)
    assert torch.equal(got, torch.from_numpy(m))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        YOLOFDistillationModel._load_fixed_match_map(str(f), "0" * 64, 8, 4)
    with pytest.raises(FileNotFoundError):
        YOLOFDistillationModel._load_fixed_match_map(str(tmp_path / "nope.npy"), None, 8, 4)
    with pytest.raises(ValueError, match="9 channels"):
        YOLOFDistillationModel._load_fixed_match_map(str(f), sha, 9, 4)
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        YOLOFDistillationModel._load_fixed_match_map(str(f), sha, 8, 3)


def test_fixed_map_arms_are_identical_except_for_the_map_file():
    from scripts.train_dfs import _DFIRE_KD_VARIANTS

    base_kw = dict(
        epochs=200, batch=None, device="0", workers=8, pretrained="yolo26n.pt",
        project="runs/x", dataset="dfire", teacher_weights="", resume=False, weights="",
        name_suffix="s", dict_fixed_map="", dict_fixed_map_sha256="",
    )
    from scripts.train_dfs import build_overrides

    def cfg(arm, path="", sha=""):
        kw = dict(base_kw, kd_variant=arm, dict_fixed_map=path, dict_fixed_map_sha256=sha)
        return build_overrides("dcn-kd", Namespace(**kw))

    a = cfg("oraclemap", "/tmp/maps/oracle.npy", "aa" * 32)
    b = cfg("shufmap", "/tmp/maps/shuf.npy", "bb" * 32)
    keys = set(a) | set(b)
    moved = sorted(k for k in keys if a.get(k) != b.get(k))
    assert moved == ["dict_fixed_map", "dict_fixed_map_sha256"], moved
    assert a["dict_match"] == "fixed" and a["dict_proj_form"] == "identity"
    assert a["dict_align_loss"] == 0.12 and a["dict_attn_loss"] == 0.25
    for arm in ("oraclemap", "shufmap", "randmap"):
        assert arm in _DFIRE_KD_VARIANTS
