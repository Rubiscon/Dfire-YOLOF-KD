"""End-to-end equivalence between the parity kernel and the fork's own DetectionTrainer.

These are the tests that justify the claim that a run driven by ``dfire_parity`` from
MMDetection or Detectron2 is the same experiment as a native ``yolo train`` run.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from dfire_parity import build_args, resolve_dataset
from dfire_parity import build_dataset as parity_build_dataset
from dfire_parity.engine import UltralyticsParityEngine
from ultralytics.engine.trainer import BaseTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "models" / "yolo26n-DCN.yaml"
DATA_YAML = ROOT / "Log" / "_eval_val_test" / "dfire_local.yaml"


def _student():
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    return DetectionModel(cfg=str(MODEL_YAML), ch=3, nc=2, verbose=False)


requires_data = pytest.mark.skipif(
    not DATA_YAML.exists(), reason=f"D-Fire dataset descriptor missing at {DATA_YAML}"
)
requires_model = pytest.mark.skipif(
    not MODEL_YAML.exists(), reason=f"student config missing at {MODEL_YAML}"
)


@requires_data
@requires_model
@pytest.mark.parametrize("mode", ["train", "val"])
def test_parity_dataset_is_identical_to_detection_trainer_dataset(mode):
    """The samples an external framework sees are the samples ``yolo train`` sees."""
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils.torch_utils import init_seeds

    args = build_args(data=str(DATA_YAML), model=str(MODEL_YAML))
    data = resolve_dataset(str(DATA_YAML))

    # DetectionTrainer.build_dataset, with the arguments it actually passes.
    img_path = data["train"] if mode == "train" else data["val"]
    reference = build_yolo_dataset(
        args, img_path, args.batch, data, mode=mode, rect=mode == "val", stride=32
    )
    ours = parity_build_dataset(args, data, mode=mode)

    assert len(ours) == len(reference)
    assert ours.im_files[:64] == reference.im_files[:64]
    assert ours.rect == reference.rect == (mode == "val")
    assert [type(t).__name__ for t in ours.transforms.tolist()] == [
        type(t).__name__ for t in reference.transforms.tolist()
    ]

    for index in (0, 1, 7, 123):
        init_seeds(0)
        expected = reference[index]
        init_seeds(0)
        actual = ours[index]
        assert torch.equal(actual["img"], expected["img"])
        assert torch.equal(actual["bboxes"], expected["bboxes"])
        assert torch.equal(actual["cls"], expected["cls"])
        assert actual["im_file"] == expected["im_file"]


@requires_data
@requires_model
def test_train_batch_is_uint8_rgb_square_letterbox():
    from ultralytics.data.dataset import YOLODataset

    args = build_args(data=str(DATA_YAML), model=str(MODEL_YAML))
    dataset = parity_build_dataset(args, resolve_dataset(str(DATA_YAML)), mode="train")
    batch = YOLODataset.collate_fn([dataset[i] for i in range(4)])

    assert batch["img"].shape == (4, 3, 640, 640)
    assert batch["img"].dtype == torch.uint8
    assert batch["bboxes"].shape[0] == batch["cls"].shape[0] == batch["batch_idx"].shape[0]
    # Roughly half of D-Fire has no objects; those images must survive collation with no
    # entry in batch_idx rather than being filtered out of the epoch.
    assert set(batch["batch_idx"].int().tolist()) <= set(range(4))
    assert batch["bboxes"].numel() == 0 or batch["bboxes"].max() <= 1.0


@requires_model
def test_optimizer_step_matches_base_trainer_bit_for_bit():
    """Clip, step and EMA update must land on identical weights (AMP off)."""
    args = build_args(model=str(MODEL_YAML), amp=False)
    model_a, model_b = _student(), _student()
    model_b.load_state_dict(model_a.state_dict())

    engine = UltralyticsParityEngine(model_a, args, steps_per_epoch=127)

    reference = BaseTrainer.__new__(BaseTrainer)
    reference.model = model_b
    reference.optimizer = BaseTrainer.build_optimizer(
        reference, model=model_b, name="SGD", lr=args.lr0, momentum=args.momentum, decay=0.0005
    )
    try:
        reference.scaler = torch.amp.GradScaler("cpu", enabled=False)
    except TypeError:
        reference.scaler = torch.cuda.amp.GradScaler(enabled=False)
    from ultralytics.utils.torch_utils import ModelEMA

    reference.ema = ModelEMA(model_b)

    torch.manual_seed(1)
    grads = {name: torch.randn_like(p) * 40 for name, p in model_a.named_parameters()}
    for step in range(3):
        for (_, pa), (_, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert torch.equal(pa, pb)
        for name, p in model_a.named_parameters():
            p.grad = grads[name].clone() * (step + 1)
        for name, p in model_b.named_parameters():
            p.grad = grads[name].clone() * (step + 1)
        engine.optimizer_step()
        BaseTrainer.optimizer_step(reference)

    ours = engine.ema.ema.state_dict()
    theirs = reference.ema.ema.state_dict()
    assert engine.ema.updates == reference.ema.updates == 3
    for key, value in ours.items():
        assert torch.equal(value, theirs[key]), key


@requires_model
def test_teacher_parameters_never_reach_the_optimizer():
    """A detector that stores a frozen teacher must not leak it into the update."""
    args = build_args(model=str(MODEL_YAML), amp=False)
    student = _student()
    teacher = _student()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    engine = UltralyticsParityEngine(student, args, steps_per_epoch=127)
    optimized = {id(p) for group in engine.optimizer.param_groups for p in group["params"]}
    assert optimized.isdisjoint({id(p) for p in teacher.parameters()})
    assert optimized == {id(p) for p in student.parameters()}


@requires_model
def test_ema_is_what_gets_evaluated_and_lags_the_raw_weights():
    args = build_args(model=str(MODEL_YAML), amp=False)
    model = _student()
    engine = UltralyticsParityEngine(model, args, steps_per_epoch=127)
    raw_before = deepcopy(model.state_dict())

    torch.manual_seed(2)
    for _ in range(5):
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        engine.optimizer_step()

    evaluated = engine.eval_model()
    assert evaluated is engine.ema.ema
    ema_state, raw_state = evaluated.state_dict(), model.state_dict()
    key = "model.0.conv.weight"
    # EMA trails: it has moved off its initialization but not as far as the raw model.
    assert not torch.equal(ema_state[key], raw_before[key])
    assert not torch.equal(ema_state[key], raw_state[key])
    drift_ema = (ema_state[key] - raw_before[key]).norm()
    drift_raw = (raw_state[key] - raw_before[key]).norm()
    assert drift_ema < drift_raw
