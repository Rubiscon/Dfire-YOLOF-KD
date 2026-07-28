"""Tests for dictionary spatial-weight normalization."""

from types import SimpleNamespace

import torch
import torch.nn as nn

import ultralytics.models.yolo.detect.train as train_module
from ultralytics.models.yolo.detect.train import YOLOFDistillationModel, YOLOFDistillationTrainer


def test_dict_weight_minmax_is_per_image_and_spatial():
    weight = torch.tensor(
        [
            [[[2.0, 4.0], [6.0, 8.0]]],
            [[[10.0, 20.0], [30.0, 40.0]]],
        ]
    )

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight)
    expected = torch.tensor(
        [
            [[[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]],
            [[[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]],
        ]
    )

    assert torch.allclose(actual, expected)


def test_dict_weight_minmax_constant_map_falls_back_to_uniform():
    weight = torch.full((2, 1, 3, 4), 7.0)

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight)

    assert torch.equal(actual, torch.ones_like(weight))
    assert torch.isfinite(actual).all()


def test_dict_weight_minmax_ignores_masked_extrema_and_zeros_padding():
    weight = torch.tensor([[[[2.0, 4.0], [-100.0, 100.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight, mask)

    assert torch.equal(actual, torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]))


def test_dict_weight_minmax_constant_content_is_one_only_inside_mask():
    weight = torch.full((1, 1, 2, 3), 7.0)
    mask = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]])

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight, mask)

    assert torch.equal(actual, mask)
    assert torch.isfinite(actual).all()


def test_dict_weight_minmax_zero_content_and_all_padding_are_finite():
    weight = torch.zeros(2, 1, 2, 2)
    mask = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ]
    )

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight, mask)

    assert torch.equal(actual[0], torch.ones_like(actual[0]))
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    assert torch.isfinite(actual).all()


def test_dict_weight_minmax_masked_normalization_is_batch_independent():
    weight = torch.tensor(
        [
            [[[1.0, 3.0], [99.0, 99.0]]],
            [[[10.0, -99.0], [20.0, 30.0]]],
        ]
    )
    mask = torch.tensor(
        [
            [[[1.0, 1.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 1.0]]],
        ]
    )

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight, mask)
    expected = torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.5, 1.0]]],
        ]
    )

    assert torch.equal(actual, expected)


def test_letterbox_ratio_pad_mask_feeds_content_aware_minmax():
    image = torch.zeros(1, 3, 4, 6)
    mask = YOLOFDistillationModel._letterbox_content_mask(image, [((1.0, 1.0), (1.0, 1.0))])
    weight = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)

    actual = YOLOFDistillationModel._minmax_normalize_weight(weight, mask)

    assert torch.equal(mask[:, :, 0], torch.zeros_like(mask[:, :, 0]))
    assert torch.equal(mask[:, :, -1], torch.zeros_like(mask[:, :, -1]))
    assert torch.equal(mask[:, :, :, 0], torch.zeros_like(mask[:, :, :, 0]))
    assert torch.equal(mask[:, :, :, -1], torch.zeros_like(mask[:, :, :, -1]))
    assert torch.equal(actual[mask == 0], torch.zeros_like(actual[mask == 0]))
    assert actual[0, 0, 1, 1] == 0.0
    assert actual[0, 0, 2, 4] == 1.0


class _IdentityDictionary(nn.Module):
    def forward(self, teacher, student):
        return student, teacher, student.new_tensor(0.0)


def _pipeline_model(weight, *, use_ema=False):
    model = YOLOFDistillationModel.__new__(YOLOFDistillationModel)
    nn.Module.__init__(model)
    model.register_parameter("_test_parameter", nn.Parameter(torch.tensor(0.0)))
    model.args = SimpleNamespace(dict_weight="saliency_dLdx", dict_feature_norm="none")
    model._student_tap = torch.ones_like(weight)
    model._dict_warned = False
    model._dict_teacher_layers = [6]
    model.dictionary_modules = nn.ModuleList([_IdentityDictionary()])
    model._saliency_ema = {6: weight.clone()} if use_ema else {}
    model._cached_saliency = {}
    model._content_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    model._collect_dict_stats = True
    model.last_dict_weight_stats = {}
    model._teacher_weight_source = "ema"
    return model


def test_dictionary_loss_pipeline_uses_masked_extrema():
    weight = torch.tensor([[[[2.0, 4.0], [100.0, 100.0]]]])
    model = _pipeline_model(weight)
    teacher = torch.zeros_like(weight)

    align, _, _ = model._dictionary_losses({6: teacher}, {6: weight})

    assert torch.isclose(align, torch.tensor(0.25))
    assert model.last_dict_weight_stats["weight_source"] == "live"
    assert model.last_dict_weight_stats["content_mask_ratio"] == 0.5


def test_dictionary_loss_pipeline_uses_ema_after_live_map_missing():
    weight = torch.tensor([[[[2.0, 4.0], [100.0, 100.0]]]])
    model = _pipeline_model(weight, use_ema=True)
    teacher = torch.zeros_like(weight)

    align, _, _ = model._dictionary_losses({6: teacher}, {})

    assert torch.isclose(align, torch.tensor(0.25))
    assert model.last_dict_weight_stats["weight_source"] == "ema"
    assert model.last_dict_weight_stats["teacher_source"] == "ema"


def test_dictionary_diagnostics_disabled_do_not_collect_stats():
    weight = torch.tensor([[[[2.0, 4.0], [100.0, 100.0]]]])
    model = _pipeline_model(weight)
    model._collect_dict_stats = False
    model._dict_weight_summary = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected stats"))

    align, _, _ = model._dictionary_losses({6: torch.zeros_like(weight)}, {6: weight})

    assert torch.isclose(align, torch.tensor(0.25))
    assert model.last_dict_weight_stats == {}


def test_freeze_stage_records_ema_teacher_source():
    model = YOLOFDistillationModel.__new__(YOLOFDistillationModel)
    nn.Module.__init__(model)
    model.teacher = nn.Linear(1, 1)
    model.args = SimpleNamespace(online_distill=True, teacher_freeze_epoch=1, teacher_freeze_use_ema=True)
    model.current_epoch = 1
    model._teacher_frozen = False
    model._teacher_weight_source = "live"
    model._sync_live_teacher_from_ema = lambda trainer: True

    model._apply_teacher_freeze_if_needed()

    assert model._teacher_frozen
    assert model._teacher_weight_source == "ema"
    assert not any(parameter.requires_grad for parameter in model.teacher.parameters())


def test_low_frequency_log_has_sources_and_effective_losses(tmp_path):
    model = nn.Linear(1, 1)
    model.last_dict_weight_stats = {
        "weight_source": "ema",
        "teacher_source": "ema",
        "dict_weight_mode": "saliency_dldx",
        "dict_weight_norm": "minmax",
        "raw_align_loss": 0.2,
        "weighted_align_loss": 0.016,
    }
    trainer = SimpleNamespace(model=model, save_dir=tmp_path, epoch=4)

    YOLOFDistillationTrainer._log_dict_weight_stats(None, trainer)

    rows = (tmp_path / "dict_weight_stats.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert "weight_source,teacher_source" in rows[0]
    assert "raw_align_loss,weighted_align_loss" in rows[0]
    assert rows[1].startswith("5,ema,ema,saliency_dldx,minmax,")
    assert model.last_dict_weight_stats == {}


def test_dict_weight_log_fresh_run_removes_stale_file(tmp_path):
    path = tmp_path / "dict_weight_stats.csv"
    path.write_text("stale\n", encoding="utf-8")
    trainer = SimpleNamespace(save_dir=tmp_path, args=SimpleNamespace(resume=False))
    callback = YOLOFDistillationTrainer.__new__(YOLOFDistillationTrainer)

    callback._prepare_dict_weight_log(trainer)

    assert not path.exists()


def test_dict_weight_log_resume_keeps_matching_schema(tmp_path):
    path = tmp_path / "dict_weight_stats.csv"
    header = ",".join(YOLOFDistillationTrainer._DICT_WEIGHT_LOG_FIELDS)
    original = f"{header}\n1,live\n"
    path.write_text(original, encoding="utf-8")
    trainer = SimpleNamespace(save_dir=tmp_path, args=SimpleNamespace(resume=True))
    callback = YOLOFDistillationTrainer.__new__(YOLOFDistillationTrainer)

    callback._prepare_dict_weight_log(trainer)

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.schema-mismatch*.csv"))


def test_dict_weight_log_resume_rotates_mismatched_schema(tmp_path):
    path = tmp_path / "dict_weight_stats.csv"
    path.write_text("epoch,old_field\n1,2\n", encoding="utf-8")
    trainer = SimpleNamespace(save_dir=tmp_path, args=SimpleNamespace(resume=True))
    callback = YOLOFDistillationTrainer.__new__(YOLOFDistillationTrainer)

    callback._prepare_dict_weight_log(trainer)

    backups = list(tmp_path.glob("dict_weight_stats.schema-mismatch*.csv"))
    assert not path.exists()
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "epoch,old_field\n1,2\n"


def test_dict_weight_log_is_disabled_by_default():
    model = nn.Linear(1, 1)
    model._collect_dict_stats = True
    model.last_dict_weight_stats = {"stale": 1.0}
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(), epoch=0)

    YOLOFDistillationTrainer._update_current_epoch(None, trainer)

    assert model._collect_dict_stats is False


def test_dict_weight_log_nonzero_rank_does_not_write(tmp_path):
    model = nn.Linear(1, 1)
    model.last_dict_weight_stats = {"raw_align_loss": 1.0}
    trainer = SimpleNamespace(model=model, save_dir=tmp_path, epoch=0)
    old_rank = train_module.RANK
    train_module.RANK = 1
    try:
        YOLOFDistillationTrainer._log_dict_weight_stats(None, trainer)
    finally:
        train_module.RANK = old_rank

    assert not (tmp_path / "dict_weight_stats.csv").exists()
