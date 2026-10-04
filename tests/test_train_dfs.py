from argparse import Namespace

import pytest

from dfire_parity.protocol import CONTROLLED_ARGS
from scripts.train_dfs import _DICT_DELAY_EPOCH, _DICT_INIT_SEED, _yolo_overrides, build_overrides


def _args(**overrides):
    values = dict(
        epochs=None,
        batch=None,
        device=None,
        workers=None,
        pretrained="yolo26n.pt",
        project=None,
        dataset="dfire",
        name_suffix="",
        teacher_weights="",
        resume=False,
        weights="",
        kd_variant="",
    )
    values.update(overrides)
    return Namespace(**values)


_PROTOCOL_KEYS = (
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "batch",
    "nbs",
    "fixed_accumulate",
    "amp",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "mosaic",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "fliplr",
    "seed",
    "patience",
)


def test_train_dfire_uses_protocol_and_dfire_yaml():
    cfg = build_overrides("yolo26n", _args(name_suffix="seed0"))
    assert cfg["data"].replace("\\", "/").endswith("ultralytics/cfg/datasets/dfire.yaml")
    assert cfg["model"] == "yolo26n.yaml"
    assert cfg["pretrained"] == "yolo26n.pt"
    assert cfg["project"] == "dfire-protocol-baselines"
    assert cfg["name"] == "dfire-yolo26n-200e-seed0"
    assert cfg["epochs"] == 200
    for key in _PROTOCOL_KEYS:
        assert cfg[key] == CONTROLLED_ARGS[key], key


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_overrides("yolo26n", _args(dataset="dfs"))


def test_assert_cfg_supported_accepts_current_fork():
    from scripts.train_dfs import _REQUIRED_CFG, assert_cfg_supported
    from ultralytics.utils import DEFAULT_CFG_DICT

    assert_cfg_supported()
    for key in _REQUIRED_CFG:
        assert key in DEFAULT_CFG_DICT
    filtered = _yolo_overrides(build_overrides("yolo26n", _args()))
    assert "fixed_accumulate" in filtered
    assert "augmentations" not in filtered or "augmentations" in DEFAULT_CFG_DICT


def test_dcn_kd_uses_dldx_recipe_and_coco_teacher():
    cfg = build_overrides(
        "dcn-kd",
        _args(epochs=150, name_suffix="seed0"),
    )
    assert cfg["name"] == "dfire-dcn-kd-150e-seed0"
    assert cfg["epochs"] == 150
    assert cfg["teacher_freeze_epoch"] == 150
    assert cfg["dict_weight"] == "saliency_dLdx"
    assert cfg["dict_align_loss"] == 0.12
    assert cfg["dict_attn_loss"] == 0.25
    assert cfg["dict_weight_norm"] == "mean"
    assert cfg["online_distill"] is True
    assert cfg["teacher_weights"] == "yolo26n.pt"
    assert cfg["data"].replace("\\", "/").endswith("ultralytics/cfg/datasets/dfire.yaml")
    filtered = _yolo_overrides(cfg)
    assert filtered["dict_align_loss"] == 0.12
    assert filtered["dict_attn_loss"] == 0.25
    assert filtered["align_start_epoch"] == 20
    assert filtered["align_box"] == 2.0
    assert filtered["teacher_weights"] == "yolo26n.pt"


def test_voc2007_kd_does_not_reuse_dfire_align_gains():
    cfg = build_overrides(
        "dcn-kd",
        _args(dataset="voc2007", epochs=200, name_suffix="seed0-vocopt"),
    )
    assert cfg["name"] == "voc2007-dcn-kd-200e-seed0-vocopt"
    assert cfg["teacher_weights"] == "yolo26n.pt"
    assert cfg["align_start_epoch"] == 80
    assert cfg["align_loss"] == 0.06
    assert cfg["align_box"] == 1.0
    assert cfg["align_cls"] == 2.0
    assert cfg["distill_conf_thres"] == 0.45
    assert cfg["feature_loss"] == 0.04
    assert cfg["dict_align_loss"] == 0.08
    assert cfg["dict_attn_loss"] == 0.10
    assert cfg["dict_attn_start_epoch"] == 80
    assert cfg["dict_weight"] == "saliency_dLdx"
    filtered = _yolo_overrides(cfg)
    assert filtered["align_start_epoch"] == 80
    assert filtered["align_box"] == 1.0
    assert filtered["dict_attn_start_epoch"] == 80


@pytest.mark.parametrize(
    "variant,expect",
    [
        (
            "align50",
            {"align_start_epoch": 50, "dict_attn_start_epoch": 50, "align_box": 1.0,
             "dict_align_loss": 0.08, "dict_attn_loss": 0.10},
        ),
        (
            "boxlate",
            {"align_start_epoch": 80, "align_box": 2.0, "align_loss": 0.08,
             "dict_align_loss": 0.08, "align_cls": 2.0},
        ),
        (
            "dict12",
            {"align_start_epoch": 80, "align_box": 1.0, "dict_align_loss": 0.12,
             "dict_attn_loss": 0.25, "dict_attn_start_epoch": 80},
        ),
    ],
)
def test_voc2007_kd_variants_keep_coco_teacher(variant, expect):
    cfg = build_overrides(
        "dcn-kd",
        _args(
            dataset="voc2007",
            epochs=200,
            name_suffix="seed0-%s" % variant,
            kd_variant=variant,
        ),
    )
    assert cfg["name"] == "voc2007-dcn-kd-200e-seed0-%s" % variant
    assert cfg["teacher_weights"] == "yolo26n.pt"
    assert cfg["online_distill"] is True
    for key, value in expect.items():
        assert cfg[key] == value, key
    filtered = _yolo_overrides(cfg)
    for key, value in expect.items():
        assert filtered[key] == value, key


def test_dfire_kd_rejects_voc_variant():
    with pytest.raises(SystemExit, match="voc2007"):
        build_overrides("dcn-kd", _args(kd_variant="align50"))


def test_unknown_voc_kd_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown VOC KD variant"):
        build_overrides(
            "dcn-kd",
            _args(dataset="voc2007", kd_variant="not-a-recipe"),
        )


@pytest.mark.parametrize(
    "variant,expect",
    [
        ("delay40", {"align_start_epoch": 40, "dict_attn_start_epoch": 0, "dict_attn_loss": 0.25}),
        ("attn50", {"align_start_epoch": 50, "dict_attn_start_epoch": 50, "dict_attn_loss": 0.25}),
        (
            "softgain",
            {
                "align_start_epoch": 20,
                "feature_loss": 0.05,
                "align_loss": 0.08,
                "align_box": 1.5,
                "align_cls": 3.0,
                "dict_align_loss": 0.10,
                "dict_attn_loss": 0.12,
            },
        ),
        (
            "dictfirst",
            {
                "align_start_epoch": 80,
                "dict_attn_start_epoch": 20,
                "dict_attn_loss": 0.18,
                "dict_align_loss": 0.15,
                "feature_loss": 0.06,
            },
        ),
        (
            "latehard",
            {
                "align_start_epoch": 40,
                "dict_attn_start_epoch": 40,
                "dict_align_loss": 0.15,
                "dict_attn_loss": 0.30,
            },
        ),
        (
            "temp2",
            {
                "distill_temperature": 2,
                "align_cls": 9.0,
                "align_start_epoch": 20,
                "dict_attn_start_epoch": 0,
            },
        ),
        (
            "hardgain",
            {
                "align_loss": 0.16,
                "align_box": 3.0,
                "align_cls": 6.0,
                "feature_loss": 0.10,
                "align_start_epoch": 20,
            },
        ),
    ],
)
def test_dfire_kd_sgd_tune_variants(variant, expect):
    cfg = build_overrides(
        "dcn-kd",
        _args(kd_variant=variant, name_suffix="seed0-sgd-%s" % variant),
    )
    assert cfg["name"] == "dfire-dcn-kd-200e-seed0-sgd-%s" % variant
    assert cfg["teacher_weights"] == "yolo26n.pt"
    assert cfg["dict_weight"] == "saliency_dLdx"
    assert cfg["optimizer"] == "SGD"
    for key, value in expect.items():
        assert cfg[key] == value, key
    filtered = _yolo_overrides(cfg)
    for key, value in expect.items():
        assert filtered[key] == value, key


def test_unknown_dfire_kd_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown D-Fire KD variant"):
        build_overrides("dcn-kd", _args(kd_variant="not-a-recipe"))


def test_dcn_kd_rejects_finetuned_dfire_teacher():
    with pytest.raises(SystemExit, match="COCO init"):
        build_overrides(
            "dcn-kd",
            _args(
                teacher_weights="/root/Ultra/weights/yolo26n_teacher_N200_best.pt",
            ),
        )


# Dictionary-structure ablations: teacher-side encoder removed, student-side encoder
# switched to a MobileNetV2 depthwise (separable) form.
_DICT_ST_DIAG = {"dict_match_log_interval": 100}
_STRUCTURE_VARIANTS = {
    "identinit": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
    },
    "dwident": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_match_init": "identity",
    },
    "res": {
        **_DICT_ST_DIAG,
        "dict_match_grid_divisor": 4,
    },
    "resident": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_match_grid_divisor": 4,
    },
    "basefix": dict(_DICT_ST_DIAG),
    "dwnf": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_freeze_encoders": False,
    },
    "indexmatch": {
        **_DICT_ST_DIAG,
        "dict_match": "index",
    },
    "nodict": {
        **_DICT_ST_DIAG,
        "dict_align_loss": 0.0,
        "dict_attn_loss": 0.0,
    },
    "unifw": {
        **_DICT_ST_DIAG,
        "dict_weight": "none",
    },
    "identlowalign": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_align_loss": 0.06,
    },
    "identlowattn": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_attn_loss": 0.12,
    },
    "identattnfix": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_attn_consistent": True,
    },
    "noalign": {
        **_DICT_ST_DIAG,
        "dict_align_loss": 0.0,
    },
    "identnoalign": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_align_loss": 0.0,
    },
    "identlowaligndw": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_align_loss": 0.06,
        "dict_init_seed": _DICT_INIT_SEED,
    },
    "dwidentlow": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_align_loss": 0.06,
    },
    "oraclemapdelay": {
        **_DICT_ST_DIAG,
        "dict_match": "fixed",
        "dict_proj_form": "identity",
        "dict_match_init": "identity",
        "dict_start_epoch": _DICT_DELAY_EPOCH,
    },
    "shufmapdelay": {
        **_DICT_ST_DIAG,
        "dict_match": "fixed",
        "dict_proj_form": "identity",
        "dict_match_init": "identity",
        "dict_start_epoch": _DICT_DELAY_EPOCH,
    },
    "oraclemapdelaylow": {
        **_DICT_ST_DIAG,
        "dict_match": "fixed",
        "dict_proj_form": "identity",
        "dict_match_init": "identity",
        "dict_start_epoch": _DICT_DELAY_EPOCH,
        "dict_align_loss": 0.06,
    },
    "shufmapdelaylow": {
        **_DICT_ST_DIAG,
        "dict_match": "fixed",
        "dict_proj_form": "identity",
        "dict_match_init": "identity",
        "dict_start_epoch": _DICT_DELAY_EPOCH,
        "dict_align_loss": 0.06,
    },
    "pinbase": {
        **_DICT_ST_DIAG,
        "dict_init_seed": _DICT_INIT_SEED,
    },
    "pindw": {
        **_DICT_ST_DIAG,
        "dict_init_seed": _DICT_INIT_SEED,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
    },
    "pindwident": {
        **_DICT_ST_DIAG,
        "dict_init_seed": _DICT_INIT_SEED,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_match_init": "identity",
    },
    "resident": {
        **_DICT_ST_DIAG,
        "dict_match_init": "identity",
        "dict_match_grid_divisor": 4,
    },
    "oracle": {
        **_DICT_ST_DIAG,
        "dict_match": "batch_corr",
        "dict_freeze_encoders": True,
    },
    "dw": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
    },
    "dwpw": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise_separable",
    },
    "dwdetach": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_detach_student_tap": True,
    },
    "dwst": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_match": "straight_through",
        "dict_freeze_encoders": False,
        "dict_commit_loss": 0.05,
    },
    "dwpwst": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise_separable",
        "dict_match": "straight_through",
        "dict_freeze_encoders": False,
        "dict_commit_loss": 0.05,
    },
    "dwstfrz": {
        **_DICT_ST_DIAG,
        "dict_teacher_encoder": False,
        "dict_student_encoder": "depthwise",
        "dict_match": "straight_through",
        "dict_freeze_encoders": True,
        "dict_commit_loss": 0.05,
    },
}


@pytest.mark.parametrize("variant,expect", sorted(_STRUCTURE_VARIANTS.items()))
def test_dfire_dict_structure_variants(variant, expect):
    cfg = build_overrides(
        "dcn-kd",
        _args(kd_variant=variant, name_suffix="seed0-dict-%s" % variant),
    )
    assert cfg["name"] == "dfire-dcn-kd-200e-seed0-dict-%s" % variant
    assert cfg["teacher_weights"] == "yolo26n.pt"
    assert cfg["optimizer"] == "SGD"
    for key, value in expect.items():
        assert cfg[key] == value, key
    # Untouched baseline knobs must still hold (unless this arm deliberately moves them,
    # e.g. the `nodict` ablation zeroes both dictionary gains).
    from scripts.train_dfs import _KD_RECIPE

    for key in ("dict_align_loss", "dict_attn_loss", "align_start_epoch", "feature_loss"):
        if key in expect:
            continue
        assert cfg[key] == _KD_RECIPE[key], key


def test_align_off_arms_keep_the_dictionary_branch_alive():
    """`noalign`/`identnoalign` must turn off the align term but NOT the whole branch.

    This is what distinguishes them from `nodict`: with the AT weight still positive,
    `_dict_active()` stays true, modules are still built, and the AT term still trains `proj`.
    If they behaved like `nodict`, they would not measure the align term at all.
    """
    from types import SimpleNamespace

    from scripts.train_dfs import _DFIRE_KD_VARIANTS
    from ultralytics.models.yolo.detect.train import YOLOFDistillationModel

    for arm, want_init in (("noalign", "default"), ("identnoalign", "identity")):
        cfg = build_overrides("dcn-kd", _args(kd_variant=arm))
        assert cfg["dict_align_loss"] == 0.0, arm
        assert cfg["dict_attn_loss"] == 0.25, arm
        assert _DFIRE_KD_VARIANTS[arm].get("dict_match_init", "default") == want_init, arm

        m = object.__new__(YOLOFDistillationModel)
        m.dictionary_modules = [object()]
        m.teacher = object()
        m.args = SimpleNamespace(dict_align_loss=cfg["dict_align_loss"],
                                 dict_attn_loss=cfg["dict_attn_loss"],
                                 dict_commit_loss=0.0, dict_infomax_loss=0.0,
                                 dict_start_epoch=0, dict_attn_start_epoch=0)
        m.current_epoch = 50
        assert m._dict_active(), "%s must keep the branch active (AT is still on)" % arm

    # contrast: nodict switches BOTH gains off, so the branch is inactive
    m.args.dict_align_loss = 0.0
    m.args.dict_attn_loss = 0.0
    assert not m._dict_active(), "nodict must deactivate the branch"


def test_noalign_is_a_one_key_ablation_against_the_recipe():
    """`noalign` must differ from the untouched recipe by exactly one key."""
    from scripts.train_dfs import _DFIRE_KD_VARIANTS

    a = build_overrides("dcn-kd", _args(kd_variant="noalign"))
    b = build_overrides("dcn-kd", _args(kd_variant="basefix"))
    moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert moved == ["dict_align_loss"], moved
    assert "noalign" in _DFIRE_KD_VARIANTS and "identnoalign" in _DFIRE_KD_VARIANTS


def test_advisor_arms_match_the_depthwise_architecture_at_the_working_alpha():
    """The advisor-conformant pair must (a) meet both of the advisor's asks and (b) use the alpha
    that recovers, while differing from each other only in whether `proj` is pinned."""
    from scripts.train_dfs import _DICT_INIT_SEED, _DFIRE_KD_VARIANTS

    a = build_overrides("dcn-kd", _args(kd_variant="identlowaligndw"))
    b = build_overrides("dcn-kd", _args(kd_variant="dwidentlow"))
    for name, cfg in (("identlowaligndw", a), ("dwidentlow", b)):
        assert cfg["dict_teacher_encoder"] is False, name
        assert cfg["dict_student_encoder"] == "depthwise", name
        assert cfg["dict_match_init"] == "identity", name
        assert cfg["dict_align_loss"] == 0.06, name
        assert cfg["dict_attn_loss"] == 0.25, name
    # only the pinned seed may differ between them
    moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert moved == ["dict_init_seed"], moved
    assert a["dict_init_seed"] == _DICT_INIT_SEED

    # and the pinned one must equal identlowalign apart from the encoder geometry and the pin
    ref = build_overrides("dcn-kd", _args(kd_variant="identlowalign"))
    diff = sorted(k for k in set(a) | set(ref)
                  if a.get(k) != ref.get(k) and k not in ("name",))
    assert diff == ["dict_init_seed", "dict_student_encoder", "dict_teacher_encoder"], diff
    assert "identlowaligndw" in _DFIRE_KD_VARIANTS and "dwidentlow" in _DFIRE_KD_VARIANTS


def test_depthwise_identity_encoders_reproduce_the_full_conv_assignment():
    """Why the advisor's architecture is free under identity init: with frozen identity encoders
    (eval mode, as training uses) the reassembled teacher target is identical whether the student
    encoder is a full convolution or a depthwise one, so only the parameter count changes."""
    import torch

    from ultralytics.nn.modules.yolof import DictionaryModule

    def build(**kw):
        torch.manual_seed(0)
        m = DictionaryModule(128, 256, 40, 20, 2, match="hard", temperature=0.07,
                             match_norm="l2", **kw)
        m.freeze_encoders()   # calls encoder.eval(), matching training
        m.eval()
        return m

    t = torch.randn(2, 128, 40, 40)
    s = torch.randn(2, 256, 20, 20)
    full = build(match_init="identity")
    dw = build(match_init="identity", teacher_encoder=False, student_encoder="depthwise")
    dw_rand = build(match_init="default", teacher_encoder=False, student_encoder="depthwise")
    with torch.no_grad():
        _, t_full, _, _ = full(t, s)
        _, t_dw, _, _ = dw(t, s)
        _, t_rand, _, _ = dw_rand(t, s)

    assert torch.equal(t_full, t_dw), "identity init must make the conv form irrelevant"
    assert not torch.equal(t_full, t_rand), "a random init must NOT be treated as equivalent"

    n_full = sum(p.numel() for p in full.parameters())
    n_dw = sum(p.numel() for p in dw.parameters())
    assert n_dw < n_full, (n_dw, n_full)
    assert n_dw == 265216 and n_full == 1000448, (n_dw, n_full)


@pytest.mark.parametrize("variant", sorted(_STRUCTURE_VARIANTS))
def test_dict_structure_keys_survive_yolo_overrides(variant):
    """Regression guard: these keys live only in this fork, so a missing entry in
    _KD_EXTRA_KEYS would make _yolo_overrides drop them silently."""
    from scripts.train_dfs import _KD_EXTRA_KEYS, _DFIRE_KD_VARIANTS

    overrides = _DFIRE_KD_VARIANTS[variant]
    for key in overrides:
        assert key in _KD_EXTRA_KEYS, key
    filtered = _yolo_overrides(build_overrides("dcn-kd", _args(kd_variant=variant)))
    for key, value in _STRUCTURE_VARIANTS[variant].items():
        assert key in filtered, key
        assert filtered[key] == value, key


@pytest.mark.parametrize("variant", ["pinbase", "pindw", "pindwident"])
def test_pinned_arms_share_one_init_seed(variant):
    """The clean set only works if every arm pins proj to the SAME value."""
    cfg = build_overrides("dcn-kd", _args(kd_variant=variant))
    assert cfg["dict_init_seed"] == _DICT_INIT_SEED
    assert cfg["dict_match"] == "hard"
    assert cfg["dict_align_loss"] == 0.12 and cfg["dict_attn_loss"] == 0.25


def test_pinned_arms_differ_only_in_their_encoders():
    base = build_overrides("dcn-kd", _args(kd_variant="pinbase"))
    dw = build_overrides("dcn-kd", _args(kd_variant="pindw"))
    dwi = build_overrides("dcn-kd", _args(kd_variant="pindwident"))

    assert sorted(k for k in set(base) | set(dw) if base.get(k) != dw.get(k)) == [
        "dict_student_encoder",
        "dict_teacher_encoder",
    ]
    keys = set(dw) | set(dwi)
    assert sorted(k for k in keys if dw.get(k) != dwi.get(k)) == ["dict_match_init"]
    assert dwi["dict_match_init"] == "identity"


@pytest.mark.parametrize(
    "delayed,base,expected",
    [
        ("oraclemapdelay", "oraclemap", ["dict_start_epoch"]),
        ("oraclemapdelaylow", "oraclemap", ["dict_align_loss", "dict_start_epoch"]),
        ("shufmapdelay", "shufmap", ["dict_start_epoch"]),
        ("shufmapdelaylow", "shufmap", ["dict_align_loss", "dict_start_epoch"]),
    ],
)
def test_delay_arms_move_only_the_schedule(delayed, base, expected):
    """A delay arm must differ from its base by the schedule and nothing else, or the
    delay cannot be credited with whatever changes."""
    a = build_overrides("dcn-kd", _args(kd_variant=delayed))
    b = build_overrides("dcn-kd", _args(kd_variant=base))
    moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert moved == sorted(expected), (delayed, moved)
    assert a["dict_start_epoch"] == _DICT_DELAY_EPOCH
    assert a["dict_match"] == "fixed" and a["dict_proj_form"] == "identity"
    assert b["dict_start_epoch"] == 0, "the base arm must stay undelayed"


def test_delay_gate_switches_the_whole_dictionary_branch_off_and_on():
    """`_dict_active` gates align AND AT, so before the start epoch nothing from the
    dictionary reaches the student -- this is the property the delay arms rely on."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.detect.train import YOLOFDistillationModel

    m = object.__new__(YOLOFDistillationModel)
    m.dictionary_modules = [object()]
    m.teacher = object()
    m.args = SimpleNamespace(dict_align_loss=0.12, dict_attn_loss=0.25, dict_commit_loss=0.0,
                             dict_infomax_loss=0.0, dict_start_epoch=_DICT_DELAY_EPOCH,
                             dict_attn_start_epoch=0)

    def active(epoch):
        m.current_epoch = epoch
        return m._dict_active()

    assert not active(0), "must be off in the first epoch"
    assert not active(_DICT_DELAY_EPOCH - 1), "must still be off one epoch before the start"
    assert active(_DICT_DELAY_EPOCH), "must be on exactly at the start epoch"
    assert active(_DICT_DELAY_EPOCH + 1)

    # and the undelayed configuration must be active from the very first epoch
    m.args.dict_start_epoch = 0
    assert active(0)

    # `_dict_active` is also false when every gain is zero, independent of the schedule
    m.args.dict_start_epoch = 0
    m.args.dict_align_loss = 0.0
    m.args.dict_attn_loss = 0.0
    assert not active(50)


# --- dictionary-branch-only armon top of the advisor's architecture -----------------------------
def test_pindwonly_is_pindw_with_only_the_two_generic_terms_switched_off():
    """`pindwonly` must differ from `pindw` in exactly the two generic weights.

    Anything else would break the single-variable reading: `pindw` (all branches on) ->
    `pindwonly` (dictionary branch only) -> `nodict` (dictionary off) are meant to be three
    points on one axis, so each step must move only the term it is named for.
    """
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    assert "pindwonly" in V
    moved = sorted(k for k in set(V["pindw"]) | set(V["pindwonly"])
                   if V["pindw"].get(k) != V["pindwonly"].get(k))
    assert moved == ["align_loss", "feature_loss"], moved
    assert V["pindwonly"]["align_loss"] == 0.0
    assert V["pindwonly"]["feature_loss"] == 0.0
    # the architecture itself must be untouched, i.e. still the advisor's slicing
    assert V["pindwonly"]["dict_teacher_encoder"] is False
    assert V["pindwonly"]["dict_student_encoder"] == "depthwise"
    assert V["pindwonly"]["dict_init_seed"] == _DICT_INIT_SEED


def test_pindwonly_keeps_the_dictionary_gains_and_does_not_touch_align_gating():
    """The dictionary pair must stay on, and `align` must stay True.

    `align=False` would also stop the TAL alignment from being computed, which is a third moved
    key and would spoil the comparison. The weight is zero, so the term carries no gradient.
    """
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    cfg = _yolo_overrides(build_overrides("dcn-kd", _args(kd_variant="pindwonly")))
    assert cfg["dict_align_loss"] > 0.0, "dictionary align must stay on -- this arm is 'dict only'"
    assert cfg["dict_attn_loss"] > 0.0, "dictionary attention must stay on"
    assert cfg["align"] is True, "align gating must stay at its default (see the docstring)"


def test_pindwonly_really_suppresses_the_two_generic_terms_end_to_end():
    """The trainer multiplies these by their weights, so zero weight means zero contribution."""
    cfg = _yolo_overrides(build_overrides("dcn-kd", _args(kd_variant="pindwonly")))
    assert cfg["align_loss"] == 0.0
    assert cfg["feature_loss"] == 0.0


# --- the VOC flavour of the same architecture ---------------------------------------------------
def test_vocpindw_is_registered_in_the_voc_namespace_not_the_dfire_one():
    """The two variant tables are separate, so the VOC arm needs its own registration.

    Without this, `--dataset voc2007 --kd-variant pindw` raises "Unknown VOC KD variant".
    """
    from scripts.train_dfs import _DFIRE_KD_VARIANTS, _VOC_KD_VARIANTS

    assert "vocpindw" in _VOC_KD_VARIANTS, "the VOC namespace needs its own entry"
    assert "pindw" in _DFIRE_KD_VARIANTS, "the D-Fire entry must remain"
    cfg = _yolo_overrides(build_overrides("dcn-kd", _args(dataset="voc2007", kd_variant="vocpindw")))
    assert cfg["dict_teacher_encoder"] is False
    assert cfg["dict_student_encoder"] == "depthwise"
    assert cfg["dict_init_seed"] == _DICT_INIT_SEED


def test_vocpindw_keeps_the_voc_recipe():
    """The VOC run must keep VOC's own weights, not D-Fire's.

    VOC applies _VOC_KD_OVERRIDES for every voc2007 KD run, and the variant only adds the
    dictionary architecture on top. If the D-Fire recipe leaked in here the arm would not be
    the VOC arm at all.
    """
    from scripts.train_dfs import _VOC_KD_OVERRIDES

    cfg = _yolo_overrides(build_overrides("dcn-kd", _args(dataset="voc2007", kd_variant="vocpindw")))
    for k, v in _VOC_KD_OVERRIDES.items():
        assert cfg[k] == v, (k, cfg[k], v)
    # and explicitly not the D-Fire values
    assert cfg["align_loss"] != 0.16, "D-Fire hardgain weights must not leak into the VOC arm"
    assert cfg["feature_loss"] != 0.10


def test_voc_rejects_a_dfire_only_variant():
    """Guard the namespace split from the other direction."""
    with pytest.raises(Exception):
        build_overrides("dcn-kd", _args(dataset="voc2007", kd_variant="pindwonly"))


# --- dictionary-only arms carrying the dictionary-side changes that measured best ---------------
DICT_ONLY_NEW = {
    # arm            -> keys that must differ from `pindwonly`
    "pindwonlyu":     ["dict_weight"],
    "pindwonlyres":   ["dict_match_grid_divisor"],
    "pindwonlyures":  ["dict_match_grid_divisor", "dict_weight"],
}


@pytest.mark.parametrize("arm,moved", sorted(DICT_ONLY_NEW.items()))
def test_new_dict_only_arms_move_only_the_named_keys(arm, moved):
    """Each new arm must add exactly the idea it is named for, on top of `pindwonly`.

    `pindwonly` is the common reference for this set, so anything beyond the named keys would make
    the comparison uninterpretable.
    """
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    assert arm in V, arm
    diff = sorted(k for k in set(V["pindwonly"]) | set(V[arm]) if V["pindwonly"].get(k) != V[arm].get(k))
    assert diff == sorted(moved), (arm, diff)
    # and every one of them is still dictionary-ONLY
    assert V[arm]["align_loss"] == 0.0 and V[arm]["feature_loss"] == 0.0, arm
    assert V[arm]["dict_teacher_encoder"] is False and V[arm]["dict_student_encoder"] == "depthwise", arm


def test_dictonlyu_isolates_the_architecture_question():
    """`dictonlyu` vs `pindwonlyu` must differ ONLY in architecture (weighting held uniform).

    Three keys move: the two encoder knobs plus the projection seed. The seed only shifts the
    projection's random start (measured ~0.0007 elsewhere), so this pair is the architecture
    comparison in dictionary-only mode.
    """
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    assert "dictonlyu" in V
    diff = sorted(k for k in set(V["dictonlyu"]) | set(V["pindwonlyu"])
                  if V["dictonlyu"].get(k) != V["pindwonlyu"].get(k))
    assert diff == ["dict_init_seed", "dict_student_encoder", "dict_teacher_encoder"], diff
    # the recipe architecture must be the default: no architecture keys set
    assert "dict_teacher_encoder" not in V["dictonlyu"]
    assert "dict_student_encoder" not in V["dictonlyu"]
    # dictionary-only and uniform, like its partner
    assert V["dictonlyu"]["align_loss"] == 0.0 and V["dictonlyu"]["feature_loss"] == 0.0
    assert V["dictonlyu"]["dict_weight"] == "none"


def test_uniform_weighting_really_reaches_the_trainer_for_these_arms():
    """A key that is filtered out would silently leave saliency weighting on."""
    for arm in ("pindwonlyu", "pindwonlyures", "dictonlyu"):
        cfg = _yolo_overrides(build_overrides("dcn-kd", _args(kd_variant=arm)))
        assert cfg["dict_weight"] == "none", arm
    cfg = _yolo_overrides(build_overrides("dcn-kd", _args(kd_variant="pindwonlyres")))
    assert cfg["dict_match_grid_divisor"] == 4


# --- removing the average pool before matching (the advisor's instruction) ----------------------
def _mk_pool_module(pool, **kw):
    import torch
    from ultralytics.nn.modules.yolof import DictionaryModule

    torch.manual_seed(0)
    m = DictionaryModule(128, 256, 40, 20, grid=2, match="hard", temperature=0.07,
                         match_norm="l2", teacher_encoder=False, student_encoder="depthwise",
                         init_seed=_DICT_INIT_SEED, match_pool=pool, **kw)
    m.eval()
    return m


def test_match_pool_default_is_avg_and_leaves_the_shipped_behaviour_alone():
    """The default must be byte-identical to what every existing arm already ran."""
    import torch

    t = torch.randn(2, 128, 40, 40)
    s = torch.randn(2, 256, 20, 20)
    with torch.no_grad():
        d_flt = _mk_pool_module("avg")(t, s)[1]
        # no match_pool argument at all -> must equal the explicit "avg"
        from ultralytics.nn.modules.yolof import DictionaryModule
        torch.manual_seed(0)
        m = DictionaryModule(128, 256, 40, 20, grid=2, match="hard", temperature=0.07,
                             match_norm="l2", teacher_encoder=False, student_encoder="depthwise",
                             init_seed=_DICT_INIT_SEED)
        m.eval()
        d_dflt = m(t, s)[1]
    assert torch.equal(d_flt, d_dflt), "default must behave exactly like match_pool='avg'"


def test_match_pool_token_counts_are_grid_squared_versus_full_map():
    """avg -> d = grid*grid; none -> d = t_size*t_size (every pixel contributes)."""
    import torch

    from ultralytics.nn.modules.yolof import DictionaryModule

    t = torch.randn(2, 128, 40, 40)
    s = torch.randn(2, 256, 20, 20)
    for pool, want_d in (("avg", 4), ("none", 1600)):
        m = _mk_pool_module(pool)
        ks = m.key_enc(t) if m.key_enc is not None else t
        qs = m.query_enc(s)
        if pool == "none":
            import torch.nn.functional as F
            if qs.shape[-2:] != ks.shape[-2:]:
                qs = F.interpolate(qs, size=ks.shape[-2:], mode="bilinear", align_corners=False)
            k, q = ks.flatten(2), qs.flatten(2)
        else:
            k, q = m.pool(ks).flatten(2), m.pool(qs).flatten(2)
        assert k.shape[-1] == want_d, (pool, k.shape)
        assert q.shape[-1] == want_d, (pool, q.shape)
        assert k.shape[1] == 128 and q.shape[1] == 256, (pool, k.shape, q.shape)


def test_match_pool_none_still_produces_a_valid_reorganisation():
    """Removing the pool must not change the OUTPUT shape contract: t_reorg is (B, Cs, Ht, Wt)."""
    import torch

    t = torch.randn(2, 128, 40, 40)
    s = torch.randn(2, 256, 20, 20)
    with torch.no_grad():
        s_proj, t_reorg, commit, infomax = _mk_pool_module("none")(t, s)
    assert tuple(t_reorg.shape) == (2, 256, 40, 40), t_reorg.shape
    assert tuple(s_proj.shape) == (2, 256, 40, 40), s_proj.shape
    # a hard assignment must select exactly one teacher channel per student channel
    assert t_reorg.shape[1] == s.shape[1]


def test_match_pool_rejects_an_unknown_value():
    import pytest

    with pytest.raises(ValueError):
        _mk_pool_module("max")


@pytest.mark.parametrize("base,arm,expect_enc", [
    ("pindw", "pindwnp", "depthwise_upsample"),
    ("pindwonly", "pindwonlynp", "depthwise_upsample"),
])
def test_nopool_arms_move_only_the_pool_and_encoder_keys(base, arm, expect_enc):
    """The `np` arms change exactly two keys, and BOTH belong to the one requested modification.

    The specification is: "use only depthwise conv + BN + activation to map the student's spatial size
    onto the teacher's, with no average-pool downsampling". Implemented literally that is
      * dict_match_pool          "none"                 -- the pooling is gone
      * dict_student_encoder     "depthwise_upsample"   -- and the conv now does the resampling
    so two keys move, not one. An earlier implementation changed only the pool and did the spatial
    mapping with a bilinear interpolate, which matched the first clause but not the second.
    """
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    assert arm in V, arm
    diff = sorted(k for k in set(V[base]) | set(V[arm]) if V[base].get(k) != V[arm].get(k))
    assert diff == ["dict_match_pool", "dict_student_encoder"], (arm, diff)
    assert V[arm]["dict_match_pool"] == "none"
    assert V[arm]["dict_student_encoder"] == expect_enc
    # and both survive config filtering
    cfg = _yolo_overrides(build_overrides("dcn-kd", _args(kd_variant=arm)))
    assert cfg["dict_match_pool"] == "none", arm
    assert cfg["dict_student_encoder"] == expect_enc, arm


def test_depthwise_upsample_encoder_is_depthwise_bn_activation_and_resamples():
    """Every clause of the requested student encoder, asserted separately.

    depthwise (groups == channels, so no channel mixing) + BN + activation, and the conv stack itself
    performs the 20x20 -> 40x40 mapping rather than an interpolation layer.
    """
    import torch
    from torch import nn

    from ultralytics.nn.modules.yolof import _build_encoder, _normalize_encoder_form

    assert _normalize_encoder_form("dwup") == "depthwise_upsample"
    enc = _build_encoder(256, "depthwise_upsample", in_size=20, out_size=40)
    convs = [m for m in enc if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    assert convs, "no convolution in the stack"
    assert all(c.groups == 256 for c in convs), "not depthwise: a conv mixes channels"
    assert any(isinstance(m, nn.BatchNorm2d) for m in enc), "no BN"
    assert any(isinstance(m, (nn.ReLU, nn.SiLU, nn.GELU, nn.LeakyReLU)) for m in enc), "no activation"
    with torch.no_grad():
        y = enc(torch.randn(2, 256, 20, 20))
    assert tuple(y.shape[-2:]) == (40, 40), y.shape


def test_depthwise_upsample_requires_sizes():
    """Without the sizes the form cannot know how far to upsample, so it must refuse."""
    import pytest

    from ultralytics.nn.modules.yolof import _build_encoder

    with pytest.raises(ValueError):
        _build_encoder(256, "depthwise_upsample")


def test_full_resolution_matching_reproduces_the_requested_pipeline():
    """Teacher has no network; student reaches the teacher's size; attention is on the full maps."""
    import torch

    t = torch.randn(2, 128, 40, 40)
    s = torch.randn(2, 256, 20, 20)
    torch.manual_seed(0)
    m = _mk_pool_module("none")
    # rebuild with the upsampling encoder, which is what the `np` arms use
    from ultralytics.nn.modules.yolof import DictionaryModule
    torch.manual_seed(0)
    m = DictionaryModule(128, 256, 40, 20, grid=2, match="hard", temperature=0.07, match_norm="l2",
                         teacher_encoder=False, student_encoder="depthwise_upsample",
                         init_seed=_DICT_INIT_SEED, match_pool="none")
    m.eval()
    assert m.key_enc is None, "the teacher side must carry no network"
    with torch.no_grad():
        qs = m.query_enc(s)
        k = torch.nn.functional.normalize(t.flatten(2), dim=2)
        q = torch.nn.functional.normalize(qs.flatten(2), dim=2)
        attn = q @ k.transpose(1, 2)
        _s_proj, t_reorg, _c, _i = m(t, s)
    assert tuple(qs.shape) == (2, 256, 40, 40), qs.shape
    assert tuple(attn.shape) == (2, 256, 128), attn.shape
    assert k.shape[-1] == 40 * 40, "matching must be on the full maps (d = 1600)"
    assert tuple(t_reorg.shape) == (2, 256, 40, 40), t_reorg.shape


def test_existing_arms_carry_no_pool_key():
    """If a previous arm started setting match_pool, its measured numbers would no longer apply."""
    from scripts.train_dfs import _DFIRE_KD_VARIANTS as V

    for arm in ("pindw", "pindwonly", "pindwonlyu", "basefix", "nodict"):
        assert "dict_match_pool" not in V[arm], arm
