"""Focused tests for spatial-entropy align weighting and diagnostics."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import ultralytics.models.yolo.detect.train as detect_train
from ultralytics.cfg import check_dict_alignment
from ultralytics.models.yolo.detect.train import YOLOFDistillationModel, YOLOFDistillationTrainer
from ultralytics.nn.modules.yolof import DictionaryModule


def test_spatial_entropy_weight_is_positive_and_row_normalized():
    value = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    query = value.clone()

    weight = DictionaryModule.spatial_entropy_weight(
        query,
        value,
        grid_size=(1, 2),
        temperature=0.01,
        floor=0.1,
    )

    assert weight.shape == (1, 1, 1, 2)
    assert torch.isfinite(weight).all()
    assert torch.all(weight >= 0.1)
    assert torch.all(weight <= 1.0)
    assert float(weight.max()) < 0.11  # near one-hot rows have near-zero entropy


def test_spatial_entropy_weight_maps_uniform_attention_to_one():
    query = torch.zeros(2, 3, 2, 2)
    value = torch.randn(2, 3, 2, 2)

    weight = DictionaryModule.spatial_entropy_weight(
        query,
        value,
        grid_size=(2, 2),
        temperature=0.1,
        floor=0.1,
    )

    assert torch.allclose(weight, torch.ones_like(weight), atol=1e-6)


def test_default_entropy_weight_is_numerically_legacy_identical():
    torch.manual_seed(17)
    query = torch.randn(2, 5, 3, 4)
    value = torch.randn(2, 5, 3, 4)
    temperature, floor = 0.1, 0.1

    actual = DictionaryModule.spatial_entropy_weight(
        query, value, grid_size=(3, 4), temperature=temperature, floor=floor
    )
    q = F.normalize(F.adaptive_avg_pool2d(query.float(), (3, 4)).flatten(2).transpose(1, 2), dim=2)
    v = F.normalize(F.adaptive_avg_pool2d(value.float(), (3, 4)).flatten(2).transpose(1, 2), dim=2)
    attention = F.softmax((q @ v.transpose(1, 2)) / temperature, dim=2)
    entropy = -(attention * attention.clamp_min(1e-10).log()).sum(dim=2) / math.log(attention.shape[2])
    expected = (floor + (1.0 - floor) * entropy.clamp(0.0, 1.0)).reshape(2, 1, 3, 4)

    assert torch.equal(actual, expected)


def test_entropy_inverse_and_detached_maps():
    value = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    query = value.clone().requires_grad_(True)
    weight, entropy = DictionaryModule.spatial_entropy_weight(
        query,
        value,
        grid_size=(1, 2),
        temperature=0.001,
        floor=0.1,
        inverse=True,
        return_entropy=True,
    )

    assert not weight.requires_grad and not entropy.requires_grad
    assert torch.allclose(weight, torch.ones_like(weight), atol=1e-6)
    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1e-6)
    assert float((weight <= 0.1 + 1e-6).float().mean()) == 0.0


def test_weight_norm_mask_and_interpolate_are_content_aware():
    weight = torch.tensor([[[[0.2, 0.8], [0.4, 1.0]]]])
    mask = torch.tensor([[[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]]])
    resized = F.interpolate(weight, size=(2, 4), mode="bilinear", align_corners=False)
    masked = YOLOFDistillationModel._apply_content_mask(resized, mask)

    mean_norm = YOLOFDistillationModel._normalize_dict_weight(masked, "mean", mask)
    expected = masked / masked.mean(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    assert torch.allclose(mean_norm, expected)  # exact legacy baseline formula

    minmax = YOLOFDistillationModel._normalize_dict_weight(masked, "minmax", mask)
    assert torch.all(minmax[:, :, :, 2:] == 0)
    assert float(minmax[:, :, :, :2].min()) == 0.0
    assert float(minmax[:, :, :, :2].max()) == 1.0

    constant = YOLOFDistillationModel._normalize_dict_weight(mask * 0.4, "minmax", mask)
    assert torch.all(constant[:, :, :, :2] == 1)
    assert torch.all(constant[:, :, :, 2:] == 0)


def test_entropy_modes_and_task_gradient_routing_are_distinct():
    model = object.__new__(YOLOFDistillationModel)
    for mode in ("entropy", "entropy_inverse"):
        model.args = SimpleNamespace(dict_weight=mode)
        assert not model._dict_weight_needs_task_grad()  # no throwaway teacher saliency pass
    model.args = SimpleNamespace(dict_weight="dldx_entropy_gate")
    assert model._dict_weight_needs_task_grad()
    assert model._dict_weight_mode(model.args) == "dldx_entropy_gate"


def test_dict_weight_norm_is_whitelisted():
    check_dict_alignment({}, {"dict_weight_norm": "minmax"})


def test_dict_gains_without_args_still_returns_four_values():
    model = object.__new__(YOLOFDistillationModel)
    assert model._dict_gains() == (0.0, 0.0, 0.0, 0.0)


def test_spatial_weight_csv_is_independent_from_channel_entropy():
    writer = object.__new__(YOLOFDistillationTrainer)
    writer._last_dict_match_step_written = -1
    writer._last_dict_weight_step_written = -1
    spatial = {
        "step": 100.0,
        "epoch": 2.0,
        "weight_mode": "entropy",
        "weight_norm": "mean",
        "spatial_entropy_mean": 0.6,
        "spatial_entropy_std": 0.2,
        "spatial_entropy_min": 0.0,
        "spatial_entropy_max": 1.0,
        "spatial_entropy_floor_hit_ratio": 0.125,
        "spatial_weight_mean": 1.0,
        "spatial_weight_std": 0.25,
        "spatial_weight_min": 0.2,
        "spatial_weight_max": 1.8,
        "spatial_weight_nonzero_ratio": 1.0,
        "grid_h": 5.0,
        "grid_w": 8.0,
        "spatial_entropy_residual_corr": 0.3,
    }
    with TemporaryDirectory() as tmp:
        writer.save_dir = Path(tmp)
        trainer = SimpleNamespace(
            model=SimpleNamespace(last_dict_match_stats=None, last_dict_weight_stats=spatial)
        )
        writer._append_dict_match_diagnostics_callback(trainer)
        writer._append_dict_match_diagnostics_callback(trainer)
        path = Path(tmp) / "dict_weight_stats.csv"
        lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert "spatial_entropy_mean" in lines[0]
    assert "channel_conditional_entropy" not in lines[0]
    assert lines[1].startswith("100,2,entropy,mean,")


def test_diagnostic_resume_seeds_both_independent_csv_steps():
    writer = object.__new__(YOLOFDistillationTrainer)
    writer._last_dict_match_step_written = -1
    writer._last_dict_weight_step_written = -1
    model = SimpleNamespace(_dict_match_steps=3)
    with TemporaryDirectory() as tmp:
        writer.save_dir = Path(tmp)
        (writer.save_dir / "dict_match_stats.csv").write_text(
            ",".join(writer._dict_match_diagnostic_fields()) + "\n120,2,0,0,0,0,0,0,0,0\n",
            encoding="utf-8",
        )
        (writer.save_dir / "dict_weight_stats.csv").write_text(
            ",".join(writer._dict_weight_diagnostic_fields())
            + "\n150,2,entropy,mean,0,0,0,0,0,1,0,1,1,1,5,8,0\n",
            encoding="utf-8",
        )
        writer._seed_dict_match_diagnostics_callback(SimpleNamespace(model=model))

    assert writer._last_dict_match_step_written == 120
    assert writer._last_dict_weight_step_written == 150
    assert model._dict_match_steps == 150


def test_diagnostic_csv_is_rank_zero_only():
    writer = object.__new__(YOLOFDistillationTrainer)
    writer._last_dict_match_step_written = -1
    writer._last_dict_weight_step_written = -1
    trainer = SimpleNamespace(
        model=SimpleNamespace(last_dict_match_stats={"step": 1.0}, last_dict_weight_stats={"step": 1.0})
    )
    original_rank = detect_train.RANK
    try:
        detect_train.RANK = 1
        with TemporaryDirectory() as tmp:
            writer.save_dir = Path(tmp)
            writer._append_dict_match_diagnostics_callback(trainer)
            assert not list(Path(tmp).iterdir())
    finally:
        detect_train.RANK = original_rank


def test_spatial_entropy_weight_rejects_incompatible_features():
    query = torch.randn(1, 4, 2, 2)
    value = torch.randn(1, 5, 2, 2)

    try:
        DictionaryModule.spatial_entropy_weight(query, value, grid_size=(2, 2))
    except ValueError as error:
        assert "batch and channel" in str(error)
    else:
        raise AssertionError("Expected incompatible entropy features to raise ValueError")


if __name__ == "__main__":
    test_spatial_entropy_weight_is_positive_and_row_normalized()
    test_spatial_entropy_weight_maps_uniform_attention_to_one()
    test_default_entropy_weight_is_numerically_legacy_identical()
    test_entropy_inverse_and_detached_maps()
    test_weight_norm_mask_and_interpolate_are_content_aware()
    test_entropy_modes_and_task_gradient_routing_are_distinct()
    test_dict_weight_norm_is_whitelisted()
    test_dict_gains_without_args_still_returns_four_values()
    test_spatial_weight_csv_is_independent_from_channel_entropy()
    test_diagnostic_resume_seeds_both_independent_csv_steps()
    test_diagnostic_csv_is_rank_zero_only()
    test_spatial_entropy_weight_rejects_incompatible_features()
    print("Entropy align weighting tests passed")
