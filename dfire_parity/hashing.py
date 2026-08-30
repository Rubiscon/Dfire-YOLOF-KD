"""Stable student-init hashing for cross-framework solo/KD parity.

GID (Detectron2) and SKD (MMDetection) both partial-load ``yolo26n.pt`` into
YOLOF-DCN. Unmatched layers stay randomly initialized, so the post-load
``state_dict`` hash is the fair-comparison fingerprint. This module is the
shared implementation so both hosts write the same manifest schema and can
enforce the same reference file.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


def state_dict_sha256(module_or_state: Any) -> str:
    """Stable hash over tensor names, metadata and bytes in a state dict."""
    import torch

    state = (
        module_or_state.state_dict()
        if hasattr(module_or_state, "state_dict")
        else module_or_state
    )
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def enforce_student_init_hash(
    current_hash: str, reference_path: str, *, create_if_missing: bool = True
) -> str:
    """Compare ``current_hash`` to a one-line reference file.

    If the reference is missing and ``create_if_missing`` is true, write it
    (exclusive create) so the first controlled run pins the fingerprint.
    """
    reference_path = os.path.expanduser(str(reference_path))
    if not reference_path:
        raise ValueError("reference_path must be a non-empty path")
    if os.path.exists(reference_path):
        with open(reference_path, encoding="utf-8") as handle:
            expected_hash = handle.read().strip()
        if expected_hash != current_hash:
            raise RuntimeError(
                "Student initialization differs from controlled reference: "
                f"expected {expected_hash}, got {current_hash}"
            )
        return expected_hash
    if not create_if_missing:
        raise FileNotFoundError(
            f"Student init hash reference not found: {reference_path}"
        )
    reference_dir = os.path.dirname(reference_path)
    if reference_dir:
        os.makedirs(reference_dir, exist_ok=True)
    with open(reference_path, "x", encoding="utf-8") as handle:
        handle.write(current_hash + "\n")
    return current_hash


def write_student_init_manifest(
    output_dir: str,
    *,
    seed: int,
    current_hash: str,
    student_model_yaml: str,
    student_init_weights: str,
    reference_path: str = "",
) -> str:
    """Write ``student_init_state.json`` and optionally enforce a hash reference.

    Returns:
        Absolute path of the written manifest.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "seed": int(seed),
        "student_state_dict_sha256": current_hash,
        "student_model_yaml": str(student_model_yaml),
        "student_init_weights": str(student_init_weights),
    }
    path = os.path.join(output_dir, "student_init_state.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if reference_path:
        enforce_student_init_hash(current_hash, reference_path, create_if_missing=True)
    return path
