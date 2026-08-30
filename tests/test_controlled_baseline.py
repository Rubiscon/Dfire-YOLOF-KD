from argparse import Namespace

import pytest

from scripts.train_baselines import build_overrides


def _args(**overrides):
    values = dict(
        epochs=None,
        patience=None,
        batch=None,
        device=None,
        workers=None,
        teacher_weights="",
        student_weights="",
        name_suffix="",
        resume=False,
        weights="",
    )
    values.update(overrides)
    return Namespace(**values)


def test_offline_controlled_baseline_requires_explicit_checkpoints():
    with pytest.raises(ValueError, match="--teacher-weights"):
        build_overrides("early-dldx-offline", _args())


def test_offline_controlled_baseline_freezes_shared_teacher(tmp_path):
    teacher = tmp_path / "teacher.pt"
    student = tmp_path / "student.pt"
    teacher.touch()
    student.touch()
    config = build_overrides(
        "early-dldx-offline",
        _args(teacher_weights=str(teacher), student_weights=str(student)),
    )
    assert config["online_distill"] is False
    assert config["teacher_task_loss"] == 0.0
    assert config["teacher_weights"] == str(teacher)
    assert config["pretrained"] == str(student)
    assert config["mosaic"] == 0.0
    assert config["fliplr"] == 0.5
    assert config["augmentations"] == []
    assert config["optimizer"] == "SGD"
    assert config["momentum"] == 0.9
    assert config["batch"] == 112
    assert config["nbs"] == 112
    assert config["fixed_accumulate"] is False


def test_controlled_solo_uses_same_initialization_and_augmentation(tmp_path):
    student = tmp_path / "student.pt"
    student.touch()
    config = build_overrides(
        "dcn-solo-controlled",
        _args(student_weights=str(student)),
    )
    assert config["pretrained"] == str(student)
    assert config["mosaic"] == 0.0
    assert config["hsv_s"] == 0.0
    assert config["translate"] == 0.0
    assert config["fliplr"] == 0.5
    assert config["optimizer"] == "SGD"
    assert config["batch"] == 112
    assert config["nbs"] == 112
