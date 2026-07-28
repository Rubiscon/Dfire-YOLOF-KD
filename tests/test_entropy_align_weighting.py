"""Tests for positive spatial-entropy align weighting."""

import torch

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
    test_spatial_entropy_weight_rejects_incompatible_features()
    print("Entropy align weighting tests passed")
