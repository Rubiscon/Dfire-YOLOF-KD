"""Unit tests for shared student-init hashing (SKD/GID fairness gate)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dfire_parity.hashing import (
    enforce_student_init_hash,
    state_dict_sha256,
    write_student_init_manifest,
)


def test_state_dict_sha256_stable_and_order_insensitive():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 2, bias=False)
            self.b = nn.Linear(2, 1, bias=False)

    torch.manual_seed(0)
    model = Tiny()
    first = state_dict_sha256(model)
    second = state_dict_sha256(model.state_dict())
    assert first == second
    assert len(first) == 64

    # Same tensors under a different dict insertion order must match.
    state = model.state_dict()
    shuffled = {k: state[k] for k in reversed(list(state))}
    assert state_dict_sha256(shuffled) == first


def test_state_dict_sha256_changes_when_weights_change():
    layer = nn.Linear(4, 4, bias=False)
    before = state_dict_sha256(layer)
    with torch.no_grad():
        layer.weight.add_(1.0)
    assert state_dict_sha256(layer) != before


def test_enforce_and_manifest_roundtrip(tmp_path):
    current = "a" * 64
    reference = tmp_path / "dfire_yolof_student_init.sha256"
    enforce_student_init_hash(current, str(reference), create_if_missing=True)
    assert reference.read_text(encoding="utf-8").strip() == current
    # Second call with same hash passes.
    enforce_student_init_hash(current, str(reference), create_if_missing=False)
    with pytest.raises(RuntimeError, match="differs from controlled reference"):
        enforce_student_init_hash("b" * 64, str(reference), create_if_missing=False)

    work = tmp_path / "run"
    path = write_student_init_manifest(
        str(work),
        seed=0,
        current_hash=current,
        student_model_yaml="/root/Ultra/models/yolo26n-DCN.yaml",
        student_init_weights="/root/Ultra/weights/yolo26n.pt",
        reference_path=str(reference),
    )
    manifest = (work / "student_init_state.json").read_text(encoding="utf-8")
    assert path.endswith("student_init_state.json")
    assert '"seed": 0' in manifest
    assert current in manifest
