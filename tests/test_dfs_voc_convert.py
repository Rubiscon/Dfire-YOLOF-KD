from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from dfire_parity.protocol import CLASS_NAMES as DFIRE_CLASS_NAMES
from dfire_parity.protocol import NUM_CLASSES as DFIRE_NUM_CLASSES
from scripts.convert_dfs_voc import (
    CLASS_NAMES,
    convert,
    load_class_names,
    parse_voc_xml,
    split_stems,
    voc_box_to_yolo,
)
from scripts.train_dfs import _yolo_overrides, build_overrides
from ultralytics.utils import YAML


def _args(**overrides):
    values = dict(
        epochs=None,
        batch=None,
        device=None,
        workers=None,
        pretrained="yolo26n.pt",
        project=None,
        dataset="dfs",
        name_suffix="",
        teacher_weights="",
        resume=False,
        weights="",
    )
    values.update(overrides)
    return Namespace(**values)


def test_voc_box_to_yolo_clips_and_normalizes():
    xc, yc, w, h = voc_box_to_yolo(390, 280, 85.786, 35.549, 112.087, 115.896)
    assert 0.0 < xc < 1.0 and 0.0 < yc < 1.0
    assert 0.0 < w < 1.0 and 0.0 < h < 1.0
    assert xc == pytest.approx(((85.786 + 112.087) / 2.0) / 390)
    assert yc == pytest.approx(((35.549 + 115.896) / 2.0) / 280)
    assert w == pytest.approx((112.087 - 85.786) / 390)
    assert h == pytest.approx((115.896 - 35.549) / 280)


def test_parse_voc_xml_maps_three_classes(tmp_path: Path):
    xml = tmp_path / "demo.xml"
    xml.write_text(
        """
        <annotation>
          <filename>demo.jpg</filename>
          <size><width>100</width><height>50</height><depth>3</depth></size>
          <object><name>fire</name><bndbox>
            <xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax>
          </bndbox></object>
          <object><name>other</name><bndbox>
            <xmin>40</xmin><ymin>10</ymin><xmax>60</xmax><ymax>30</ymax>
          </bndbox></object>
          <object><name>smoke</name><bndbox>
            <xmin>70</xmin><ymin>15</ymin><xmax>90</xmax><ymax>40</ymax>
          </bndbox></object>
          <object><name>_background_</name><bndbox>
            <xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax>
          </bndbox></object>
        </annotation>
        """,
        encoding="utf-8",
    )
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    filename, rows = parse_voc_xml(xml, class_to_id)
    assert filename == "demo.jpg"
    assert [row[0] for row in rows] == [0, 1, 2]


def test_split_stems_is_seed0_stratified_and_disjoint():
    stems = [f"large_({i})" for i in range(10)] + [f"small_({i})" for i in range(10)]
    first = split_stems(stems, seed=0)
    second = split_stems(stems, seed=0)
    assert first == second
    assigned = first["train"] + first["val"] + first["test"]
    assert sorted(assigned) == sorted(stems)
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])
    for split in first.values():
        assert any(s.startswith("large_") for s in split)
        assert any(s.startswith("small_") for s in split)


def test_load_class_names_rejects_dfire_order(tmp_path: Path):
    (tmp_path / "class_names.txt").write_text("smoke\nfire\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected DFS classes"):
        load_class_names(tmp_path)


def test_convert_writes_yolo_layout(tmp_path: Path):
    source = tmp_path / "voc"
    (source / "Annotations").mkdir(parents=True)
    (source / "JPEGImages").mkdir()
    (source / "class_names.txt").write_text(
        "_background_\nfire\nother\nsmoke\n", encoding="utf-8"
    )
    Image.new("RGB", (20, 10), color=(255, 0, 0)).save(source / "JPEGImages" / "large_(1).jpg")
    (source / "Annotations" / "large_(1).xml").write_text(
        """
        <annotation>
          <filename>large_(1).jpg</filename>
          <size><width>20</width><height>10</height><depth>3</depth></size>
          <object><name>fire</name><bndbox>
            <xmin>2</xmin><ymin>1</ymin><xmax>8</xmax><ymax>7</ymax>
          </bndbox></object>
        </annotation>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "data"
    counts = convert(source, output, mode="copy")
    assert counts["total"] == 1
    assert counts["train"] == 1
    label = (output / "train" / "labels" / "large_(1).txt").read_text(encoding="utf-8").strip()
    cls_id, xc, yc, w, h = label.split()
    assert cls_id == "0"
    assert float(w) == pytest.approx(6 / 20)
    assert float(h) == pytest.approx(6 / 10)


def test_dfs_yaml_is_three_class_and_not_dfire():
    data = YAML.load("ultralytics/cfg/datasets/dfs.yaml")
    assert data["names"] == {0: "fire", 1: "other", 2: "smoke"}
    assert DFIRE_NUM_CLASSES == 2
    assert DFIRE_CLASS_NAMES == {0: "smoke", 1: "fire"}


def test_train_dfs_overrides_use_protocol_but_not_dfire_data():
    cfg = build_overrides("dcn-solo", _args())
    assert cfg["data"].replace("\\", "/").endswith("ultralytics/cfg/datasets/dfs.yaml")
    assert cfg["model"] == "yolo26n-DCN.yaml"
    assert cfg["pretrained"] == "yolo26n.pt"
    assert cfg["seed"] == 0
    assert cfg["epochs"] == 100
    assert cfg["batch"] == cfg["nbs"] == 112
    assert cfg["optimizer"] == "SGD"
    assert cfg["mosaic"] == 1.0
    assert cfg["project"] == "dfs-baselines"
    teacher = build_overrides("yolo26n", _args(name_suffix="seed0"))
    assert teacher["model"] == "yolo26n.yaml"
    assert teacher["name"] == "dfs-yolo26n-100e-seed0"
    assert "dfire.yaml" not in teacher["data"]
    long = build_overrides("yolo26n", _args(epochs=200, name_suffix="seed0"))
    assert long["epochs"] == 200
    assert long["name"] == "dfs-yolo26n-200e-seed0"


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


def test_train_dfire_uses_identical_protocol_and_dfire_yaml():
    dfs = build_overrides("yolo26n", _args(name_suffix="seed0"))
    dfire = build_overrides("yolo26n", _args(dataset="dfire", name_suffix="seed0"))
    assert dfire["data"].replace("\\", "/").endswith("ultralytics/cfg/datasets/dfire.yaml")
    assert dfire["model"] == "yolo26n.yaml"
    assert dfire["pretrained"] == "yolo26n.pt"
    assert dfire["project"] == "dfire-protocol-baselines"
    assert dfire["name"] == "dfire-yolo26n-200e-seed0"
    assert dfire["epochs"] == 200
    for key in _PROTOCOL_KEYS:
        assert dfire[key] == dfs[key], key
    assert dfs["project"] == "dfs-baselines"
    assert dfs["epochs"] == 100
    assert dfs["name"] == "dfs-yolo26n-100e-seed0"


def test_assert_cfg_supported_accepts_current_fork():
    from scripts.train_dfs import _REQUIRED_CFG, _yolo_overrides, assert_cfg_supported
    from ultralytics.utils import DEFAULT_CFG_DICT

    assert_cfg_supported()
    for key in _REQUIRED_CFG:
        assert key in DEFAULT_CFG_DICT
    filtered = _yolo_overrides(build_overrides("yolo26n", _args()))
    assert "fixed_accumulate" in filtered
    assert "augmentations" not in filtered or "augmentations" in DEFAULT_CFG_DICT


def test_dfs_dcn_kd_keeps_dldx_recipe_and_scales_freeze():
    cfg = build_overrides(
        "dcn-kd",
        _args(
            epochs=150,
            name_suffix="seed0",
            teacher_weights="/root/autodl-tmp/runs/dfs-baselines/dfs-yolo26n-100e-seed0/weights/best.pt",
        ),
    )
    assert cfg["name"] == "dfs-dcn-kd-150e-seed0"
    assert cfg["epochs"] == 150
    assert cfg["teacher_freeze_epoch"] == 150
    assert cfg["dict_weight"] == "saliency_dLdx"
    assert cfg["dict_align_loss"] == 0.12
    assert cfg["dict_attn_loss"] == 0.25
    assert cfg["dict_weight_norm"] == "mean"
    assert cfg["online_distill"] is True
    assert cfg["data"].replace("\\", "/").endswith("ultralytics/cfg/datasets/dfs.yaml")
    filtered = _yolo_overrides(cfg)
    assert filtered["dict_align_loss"] == 0.12
    assert filtered["teacher_weights"].endswith("best.pt")
