"""Tests for straight-through InfoMax dictionary matching."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import ultralytics.models.yolo.detect.train as detect_train
from ultralytics.models.yolo.detect.train import (
    YOLOF_DISTILLATION_LOSS_NAMES,
    YOLOFDistillationModel,
    YOLOFDistillationTrainer,
)
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


def test_straight_through_forward_is_hard_and_assignment_losses_train_qk():
    """ST stays hard forward; detached align cannot move Q/K, but commit/InfoMax can."""
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

    align = F.mse_loss(s_proj, t_reorg.detach())
    align.backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    for encoder in (module.key_enc, module.query_enc):
        grads = [p.grad for p in encoder.parameters() if p.requires_grad]
        assert grads and all(g is None or float(g.abs().sum()) == 0.0 for g in grads)

    module.zero_grad(set_to_none=True)
    s.grad = None
    _, _, commit, infomax = module(t, s)
    (infomax + commit).backward()
    for encoder in (module.key_enc, module.query_enc):
        grads = [p.grad for p in encoder.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads)


def test_straight_through_is_finite_at_extreme_logits():
    """Near one-hot soft assignments remain finite without changing the hard forward."""
    torch.manual_seed(9)
    t = torch.randn(1, 4, 8, 8)
    s = torch.randn(1, 3, 4, 4, requires_grad=True)
    module = DictionaryModule(4, 3, 8, 4, grid=2, match="straight_through", temperature=1e-9)

    s_proj, t_reorg, commit, infomax = module(t * 1e4, s * 1e4)
    loss = s_proj.square().mean() + t_reorg.square().mean() + commit + infomax
    assert all(torch.isfinite(value).all() for value in (s_proj, t_reorg, commit, infomax, loss))
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_dict_gains_return_four_values_and_schedule_infomax():
    model = object.__new__(YOLOFDistillationModel)
    model.current_epoch = 0
    assert model._dict_gains() == (0.0, 0.0, 0.0, 0.0)

    model.args = SimpleNamespace(
        dict_align_loss=0.08,
        dict_attn_loss=0.25,
        dict_commit_loss=0.0,
        dict_infomax_loss=0.01,
        dict_attn_start_epoch=0,
        dict_infomax_start_epoch=21,
        dict_infomax_warmup_epochs=10,
    )
    model.current_epoch = 19  # epoch 20, still disabled
    assert model._dict_gains() == (0.08, 0.25, 0.0, 0.0)
    model.current_epoch = 20  # epoch 21, first warmup step
    assert math.isclose(model._dict_gains()[3], 0.001)
    model.current_epoch = 29  # epoch 30, full gain
    assert math.isclose(model._dict_gains()[3], 0.01)

    del model.args.dict_infomax_start_epoch
    del model.args.dict_infomax_warmup_epochs
    model.current_epoch = 0
    assert math.isclose(model._dict_gains()[3], 0.01), "missing/new defaults must preserve legacy timing"


def test_trainer_dictionary_losses_detach_qk_from_align_and_at():
    """The trainer-level target detach keeps align/AT gradients out of Q/K."""
    torch.manual_seed(10)
    model = object.__new__(YOLOFDistillationModel)
    torch.nn.Module.__init__(model)
    module = DictionaryModule(8, 6, 12, 6, grid=3, match="straight_through")
    model.dictionary_modules = torch.nn.ModuleList([module])
    model.args = SimpleNamespace(
        dict_weight="none",
        dict_feature_norm="none",
        dict_match_log_interval=0,
    )
    model._student_tap = torch.randn(2, 6, 6, 6, requires_grad=True)
    model._dict_teacher_layers = [6]
    model._dict_match_steps = 0
    model._dict_warned = False
    model.current_epoch = 0
    model.last_dict_match_stats = None

    align, attention, _, _ = model._dictionary_losses({6: torch.randn(2, 8, 12, 12)})
    (align + attention).backward()
    assert model._student_tap.grad is not None and float(model._student_tap.grad.abs().sum()) > 0
    for encoder in (module.key_enc, module.query_enc):
        grads = [p.grad for p in encoder.parameters() if p.requires_grad]
        assert grads and all(g is None or float(g.abs().sum()) == 0.0 for g in grads)


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
        "hard_occupancy",
        "max_teacher_share",
        "match_margin",
        "mean_top1_probability",
        "cross_batch_assignment_churn",
        "conditional_entropy",
        "normalized_conditional_entropy",
        "marginal_entropy",
        "normalized_marginal_entropy",
        "effective_teacher_channels",
        "infomax_loss",
    }
    assert set(first) == expected
    assert all(torch.isfinite(value) for value in second.values())
    assert 0.0 < float(second["used_teacher_ratio"]) <= 1.0
    assert float(second["hard_occupancy"]) == float(second["used_teacher_ratio"])
    assert 0.0 < float(second["max_teacher_share"]) <= 1.0
    assert float(second["match_margin"]) >= 0.0
    assert 0.0 < float(second["mean_top1_probability"]) <= 1.0
    assert float(second["cross_batch_assignment_churn"]) == 0.0
    assert 0.0 <= float(second["normalized_conditional_entropy"]) <= 1.0
    assert 0.0 <= float(second["normalized_marginal_entropy"]) <= 1.0


def test_match_diagnostics_csv_writer():
    writer = object.__new__(YOLOFDistillationTrainer)
    writer._last_dict_match_step_written = -1
    stats = {
        "step": 100.0,
        "epoch": 2.0,
        "used_teacher_ratio": 0.75,
        "hard_occupancy": 0.75,
        "max_teacher_share": 0.1,
        "match_margin": 0.04,
        "mean_top1_probability": 0.8,
        "cross_batch_assignment_churn": 0.2,
        "conditional_entropy": 1.5,
        "normalized_conditional_entropy": 0.3,
        "marginal_entropy": 4.0,
        "normalized_marginal_entropy": 0.8,
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
    assert "cross_batch_assignment_churn" in lines[0]
    assert lines[1].startswith("100,2,")


def test_match_diagnostics_schema_migration_preserves_resume_step_and_ddp_rank():
    legacy_header = "step,epoch,used_teacher_ratio,assignment_churn"
    with TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        (save_dir / "dict_match_stats.csv").write_text(
            f"{legacy_header}\n1200,3,0.75,0.2\n", encoding="utf-8"
        )
        writer = object.__new__(YOLOFDistillationTrainer)
        writer.save_dir = save_dir
        writer._last_dict_match_step_written = -1
        model = SimpleNamespace(_dict_match_steps=0)
        writer._seed_dict_match_diagnostics_callback(SimpleNamespace(model=model))

        assert writer._dict_match_diagnostics_path().name == "dict_match_stats_v2.csv"
        assert writer._last_dict_match_step_written == 1200
        assert model._dict_match_steps == 1200

        previous_rank = detect_train.RANK
        try:
            detect_train.RANK = 1
            model.last_dict_match_stats = {"step": 1300.0}
            writer._append_dict_match_diagnostics_callback(SimpleNamespace(model=model))
            assert not (save_dir / "dict_match_stats_v2.csv").exists()
        finally:
            detect_train.RANK = previous_rank


def test_loss_names_include_raw_and_weighted_assignment_terms():
    assert len(YOLOF_DISTILLATION_LOSS_NAMES) == 14
    assert YOLOF_DISTILLATION_LOSS_NAMES[7:11] == (
        "commit_loss",
        "infomax_loss",
        "commit_weighted",
        "infomax_weighted",
    )


if __name__ == "__main__":
    test_infomax_prefers_balanced_deterministic_assignments()
    test_straight_through_forward_is_hard_and_assignment_losses_train_qk()
    test_straight_through_is_finite_at_extreme_logits()
    test_dict_gains_return_four_values_and_schedule_infomax()
    test_trainer_dictionary_losses_detach_qk_from_align_and_at()
    test_match_diagnostics_cover_collapse_and_stability_signals()
    test_match_diagnostics_csv_writer()
    test_match_diagnostics_schema_migration_preserves_resume_step_and_ddp_rank()
    test_loss_names_include_raw_and_weighted_assignment_terms()
    print("InfoMax matching tests passed")

