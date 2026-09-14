from argparse import Namespace
from pathlib import Path

import pytest

from dfire_parity.protocol import (
    CLASS_NAMES as DFIRE_CLASS_NAMES,
    VOC2007_CLASS_NAMES,
    identify_dataset,
    names_for_num_classes,
)
from scripts.convert_voc2007 import (
    CLASS_NAMES,
    EXPECTED_COUNTS,
    convert,
    parse_voc_xml,
    voc_box_to_yolo,
    yaml_text,
)
from ultralytics.utils import YAML


def _write_voc_sample(root: Path, image_id: str, class_name: str, difficult: int = 0) -> None:
    from PIL import Image

    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color=(8, 8, 8)).save(root / "JPEGImages" / f"{image_id}.jpg")
    (root / "Annotations" / f"{image_id}.xml").write_text(
        """
        <annotation>
          <filename>%s.jpg</filename>
          <size><width>100</width><height>50</height><depth>3</depth></size>
          <object>
            <name>%s</name>
            <difficult>%d</difficult>
            <bndbox><xmin>11</xmin><ymin>6</ymin><xmax>31</xmax><ymax>26</ymax></bndbox>
          </object>
        </annotation>
        """
        % (image_id, class_name, difficult),
        encoding="utf-8",
    )


def test_voc_box_matches_ultralytics_one_based_formula():
    xc, yc, w, h = voc_box_to_yolo(500, 375, 48, 83, 240, 291)
    assert xc == pytest.approx(((48 + 83) / 2.0 - 1.0) / 500)
    assert yc == pytest.approx(((240 + 291) / 2.0 - 1.0) / 375)
    assert w == pytest.approx((83 - 48) / 500)
    assert h == pytest.approx((291 - 240) / 375)


def test_parse_skips_difficult_keeps_normal(tmp_path: Path):
    xml = tmp_path / "000001.xml"
    xml.write_text(
        """
        <annotation>
          <filename>000001.jpg</filename>
          <size><width>100</width><height>50</height><depth>3</depth></size>
          <object>
            <name>person</name><difficult>0</difficult>
            <bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax></bndbox>
          </object>
          <object>
            <name>car</name><difficult>1</difficult>
            <bndbox><xmin>40</xmin><ymin>10</ymin><xmax>80</xmax><ymax>40</ymax></bndbox>
          </object>
        </annotation>
        """,
        encoding="utf-8",
    )
    filename, rows, stats = parse_voc_xml(xml)
    assert filename == "000001.jpg"
    assert [row[0] for row in rows] == [14]
    assert stats["difficult"] == 1
    assert stats["kept"] == 1


def test_identify_dataset_accepts_dfire_and_voc_only():
    assert identify_dataset(DFIRE_CLASS_NAMES, nc=2) == "dfire"
    assert identify_dataset(VOC2007_CLASS_NAMES, nc=20) == "voc2007"
    assert names_for_num_classes(20)[14] == "person"
    with pytest.raises(ValueError, match="Unsupported dataset"):
        identify_dataset({0: "fire", 1: "other", 2: "smoke"})
    with pytest.raises(ValueError, match="nc=2"):
        names_for_num_classes(80)


def test_repo_voc2007_yaml_is_trainval_test_and_not_0712():
    data = YAML.load(str(Path("ultralytics/cfg/datasets/voc2007.yaml")))
    assert data["nc"] == 20
    assert data["train"] == "trainval/images"
    assert data["val"] == data["test"] == "test/images"
    assert data["names"][0] == "aeroplane"
    assert "train2012" not in str(data)
    voc07_12 = YAML.load(str(Path("ultralytics/cfg/datasets/VOC.yaml")))
    assert "train2012" in str(voc07_12["train"])


def test_convert_writes_trainval_and_test(tmp_path: Path):
    source = tmp_path / "VOC2007"
    main = source / "ImageSets" / "Main"
    main.mkdir(parents=True)
    _write_voc_sample(source, "000001", "person")
    _write_voc_sample(source, "000002", "car")
    _write_voc_sample(source, "000003", "bus", difficult=1)
    (main / "trainval.txt").write_text("000001\n000002\n", encoding="utf-8")
    (main / "test.txt").write_text("000003\n", encoding="utf-8")
    output = tmp_path / "yolo"
    counts = convert(source, output, mode="copy", strict=False)
    assert counts["trainval"] == 2
    assert counts["test"] == 1
    assert counts["difficult"] == 1
    assert counts["boxes_test"] == 0
    assert (output / "trainval" / "labels" / "000001.txt").read_text(encoding="utf-8").startswith("14 ")
    assert (output / "test" / "labels" / "000003.txt").read_text(encoding="utf-8") == ""
    data = YAML.load(str(output / "voc2007.yaml"))
    assert data["path"] == "."
    assert data["train"] == "trainval/images"
    assert data["val"] == "test/images"
    assert list(data["names"].values()) == list(CLASS_NAMES)
    assert yaml_text(".").count("aeroplane") == 1


def test_convert_rejects_trainval_test_overlap(tmp_path: Path):
    source = tmp_path / "VOC2007"
    main = source / "ImageSets" / "Main"
    main.mkdir(parents=True)
    _write_voc_sample(source, "000001", "person")
    (main / "trainval.txt").write_text("000001\n", encoding="utf-8")
    (main / "test.txt").write_text("000001\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="overlap"):
        convert(source, tmp_path / "yolo", strict=False)


def test_expected_official_counts_are_voc2007():
    assert EXPECTED_COUNTS == {"trainval": 5011, "test": 4952}


def test_train_voc2007_overrides_use_protocol_and_not_dfire():
    from scripts.train_dfs import build_overrides

    args = Namespace(
        epochs=None,
        batch=None,
        device=None,
        workers=None,
        pretrained="yolo26n.pt",
        project=None,
        dataset="voc2007",
        name_suffix="seed0",
        teacher_weights="",
        resume=False,
        weights="",
    )
    cfg = build_overrides("yolo26n", args)
    assert cfg["epochs"] == 200
    assert cfg["batch"] == 112
    assert cfg["seed"] == 0
    assert "dfire.yaml" not in cfg["data"]
    assert cfg["pretrained"] == "yolo26n.pt"
    assert cfg["name"] == "voc2007-yolo26n-200e-seed0"
