"""Tests for straight-through InfoMax dictionary matching."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import YOLOFDistillationTrainer
from ultralytics.nn.modules.yolof import DictionaryModule


def test_infomax_prefers_balanced_deterministic_assignments():
    """A balanced one-hot mapping has maximal mutual information."""
    balanced = torch.eye(4).view(1, 4, 4)
    collapsed = torch.zeros(1, 4, 4)
    collapsed[:, :, 0] = 1.0
    uniform = torch.full((1, 4, 4), 0.25)

    loss_balanced, cond_balanced, marginal_balanced = DictionaryModule.infomax_loss(balanced)
    loss_collapsed, _, marginal_collapsed = DictionaryModule.infomax_loss(collapsed)
    loss_uniform, cond_uniform, marginal_uniform = DictionaryModule.infomax_loss(uniform)

    assert torch.allclose(cond_balanced, torch.zeros_like(cond_balanced))
    assert torch.allclose(marginal_balanced, torch.tensor(math.log(4.0)))
    assert torch.allclose(marginal_collapsed, torch.zeros_like(marginal_collapsed))
    assert torch.allclose(cond_uniform, marginal_uniform)
    assert loss_balanced < loss_collapsed
    assert loss_balanced < loss_uniform


def test_straight_through_forward_is_hard_and_backward_trains_assignment():
    """ST keeps proposal argmax values while align gradients reach Q/K."""
    torch.manual_seed(7)
    t = torch.randn(2, 8, 12, 12)
    s = torch.randn(2, 6, 6, 6, requires_grad=True)
    module = DictionaryModule(
        8,
        6,
        12,
        6,
        grid=3,
        match="straight_through",
        temperature=0.1,
        match_norm="l2",
        match_init="identity",
    )

    s_proj, t_reorg, commit, infomax = module(t, s)
    with torch.no_grad():
        k = F.normalize(module.pool(module.key_enc(t)).flatten(2), dim=2)
        q = F.normalize(module.pool(module.query_enc(s)).flatten(2), dim=2)
        index = (q @ k.transpose(1, 2)).argmax(dim=2)
        expected = torch.gather(t, 1, index[:, :, None, None].expand(-1, -1, 12, 12))
    assert torch.allclose(t_reorg, expected, rtol=1e-6, atol=1e-7)

    loss = F.mse_loss(s_proj, t_reorg) + 0.01 * infomax + 0.01 * commit
    loss.backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    for encoder in (module.key_enc, module.query_enc):
        grads = [p.grad for p in encoder.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads)


def test_match_diagnostics_cover_collapse_and_stability_signals():
    torch.manual_seed(11)
    t = torch.randn(2, 8, 12, 12)
    s = torch.randn(2, 6, 6, 6)
    module = DictionaryModule(8, 6, 12, 6, grid=3, match="straight_through")

    module(t, s, collect_diagnostics=True)
    first = module.last_match_stats
    module(t, s, collect_diagnostics=True)
    second = module.last_match_stats

    expected = {
        "used_teacher_ratio",
        "max_teacher_share",
        "match_margin",
        "assignment_churn",
        "conditional_entropy",
        "marginal_entropy",
        "effective_teacher_channels",
        "infomax_loss",
    }
    assert set(first) == expected
    assert all(torch.isfinite(value) for value in second.values())
    assert 0.0 < float(second["used_teacher_ratio"]) <= 1.0
    assert 0.0 < float(second["max_teacher_share"]) <= 1.0
    assert float(second["match_margin"]) >= 0.0
    assert float(second["assignment_churn"]) == 0.0


def test_match_diagnostics_csv_writer():
    writer = object.__new__(YOLOFDistillationTrainer)
    writer._last_dict_match_step_written = -1
    stats = {
        "step": 100.0,
        "epoch": 2.0,
        "used_teacher_ratio": 0.75,
        "max_teacher_share": 0.1,
        "match_margin": 0.04,
        "assignment_churn": 0.2,
        "conditional_entropy": 1.5,
        "marginal_entropy": 4.0,
        "effective_teacher_channels": 54.6,
        "infomax_loss": -2.5,
    }
    with TemporaryDirectory() as tmp:
        writer.save_dir = Path(tmp)
        trainer = SimpleNamespace(model=SimpleNamespace(last_dict_match_stats=stats))
        writer._append_dict_match_diagnostics_callback(trainer)
        writer._append_dict_match_diagnostics_callback(trainer)
        lines = (Path(tmp) / "dict_match_stats.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("effective_teacher_channels,infomax_loss")
    assert lines[1].startswith("100,2,")


if __name__ == "__main__":
    test_infomax_prefers_balanced_deterministic_assignments()
    test_straight_through_forward_is_hard_and_backward_trains_assignment()
    test_match_diagnostics_cover_collapse_and_stability_signals()
    test_match_diagnostics_csv_writer()
    print("InfoMax matching tests passed")

