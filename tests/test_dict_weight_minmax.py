"""Tests for dictionary spatial-weight normalization."""

import torch

from ultralytics.models.yolo.detect.train import YOLOFDistillationModel


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
