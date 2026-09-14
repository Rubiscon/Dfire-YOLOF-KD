"""Mass-normalized dictionary align: sum(W·r̄)/sum(W), r̄ = mean_c(residual²)."""

import torch

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel


def test_mass_normalized_weighted_mse_ignores_zero_weight_locations():
    weight = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    residual = torch.ones_like(weight)

    loss = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual)

    assert torch.isclose(loss, torch.tensor(1.0))
    assert not torch.isclose(loss, (weight * residual).mean())


def test_mass_normalized_weighted_mse_does_not_scale_with_channels():
    weight = torch.ones(2, 1, 4, 4)
    residual_c1 = torch.ones(2, 1, 4, 4)
    residual_c256 = torch.ones(2, 256, 4, 4)

    loss_c1 = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual_c1)
    loss_c256 = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual_c256)
    old_style = (weight * residual_c256).mean()

    assert torch.isclose(loss_c1, torch.tensor(1.0))
    assert torch.isclose(loss_c256, torch.tensor(1.0))
    assert torch.isclose(loss_c256, old_style)


def test_mass_normalized_weighted_mse_matches_legacy_mean_for_mean_norm_weights():
    torch.manual_seed(0)
    residual = torch.randn(3, 256, 16, 16)
    raw = torch.rand(3, 1, 16, 16).clamp_min(1e-3)
    weight = YOLOFDistillationModel._normalize_dict_weight(raw, "mean", None)
    assert torch.allclose(weight.mean(dim=(2, 3)), torch.ones(3, 1, 1, 1), atol=1e-5)

    mass = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual)
    legacy = (weight * residual).mean()
    assert torch.allclose(mass, legacy, rtol=1e-5, atol=1e-5)


def test_mass_normalized_weighted_mse_fp16_large_map_is_finite():
    weight = torch.ones(112, 1, 40, 40, dtype=torch.float16)
    residual = torch.ones(112, 256, 40, 40, dtype=torch.float16)
    assert not torch.isfinite(weight.sum())

    loss = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual)
    assert torch.isfinite(loss)
    assert torch.isclose(loss.float(), torch.tensor(1.0), atol=1e-3)


def test_mass_normalized_weighted_mse_gradients_flow_to_residual():
    weight = torch.ones(2, 1, 4, 4)
    residual = torch.randn(2, 256, 4, 4, requires_grad=True)
    loss = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual)
    loss.backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert residual.grad.abs().sum() > 0


def test_mass_normalized_weighted_mse_all_zero_weight_is_zero():
    weight = torch.zeros(2, 1, 4, 4)
    residual = torch.randn(2, 256, 4, 4)
    loss = YOLOFDistillationModel._mass_normalized_weighted_mse(weight, residual)
    assert torch.isclose(loss, torch.tensor(0.0))
