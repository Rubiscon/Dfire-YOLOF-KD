"""Convert official VOC2007 into a YOLO layout for Ultralytics / GID / SKD / CanKD.

Protocol (do not invent a split):
  train  = ImageSets/Main/trainval.txt   (5011)
  val    = ImageSets/Main/test.txt       (4952)
  test   = the same test images          (4952)

This is VOC2007-only. Do not use ultralytics/cfg/datasets/VOC.yaml (that is
07+12: 2007 trainval + 2012 trainval). Do not mix D-Fire or DFS teachers.

Label conversion matches Ultralytics VOC.yaml: skip ``difficult==1``, and
treat boxes as 1-based inclusive pixels (subtract 1 before normalizing).

    python scripts/convert_voc2007.py
    python scripts/convert_voc2007.py --dry-run
    python scripts/convert_voc2007.py \\
        --source /root/datasets/VOC2007 \\
        --output /root/datasets/VOC2007_yolo
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path(r"E:\CRIS\VOC数据集\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007")
DEFAULT_OUTPUT = Path(r"E:\CRIS\VOC数据集\yolo")
EXPECTED_COUNTS = {"trainval": 5011, "test": 4952}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")
CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def voc_box_to_yolo(
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float, float, float]:
    """Ultralytics VOC.yaml convert_box: 1-based pixels, then xywh in [0, 1]."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size %s x %s" % (width, height))
    dw, dh = 1.0 / width, 1.0 / height
    x = (xmin + xmax) / 2.0 - 1.0
    y = (ymin + ymax) / 2.0 - 1.0
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        raise ValueError("Empty box")
    return x * dw, y * dh, w * dw, h * dh


def parse_voc_xml(path: Path) -> tuple[str, list[tuple[int, float, float, float, float]], dict[str, int]]:
    root = ET.parse(path).getroot()
    filename = (root.findtext("filename") or f"{path.stem}.jpg").strip()
    size = root.find("size")
    if size is None:
        raise ValueError("%s has no <size>" % path)
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    rows = []
    stats = Counter()
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            stats["unknown_class"] += 1
            continue
        if int(obj.findtext("difficult") or 0) == 1:
            stats["difficult"] += 1
            continue
        box = obj.find("bndbox")
        if box is None:
            stats["missing_box"] += 1
            continue
        try:
            yolo = voc_box_to_yolo(
                width,
                height,
                float(box.findtext("xmin")),
                float(box.findtext("xmax")),
                float(box.findtext("ymin")),
                float(box.findtext("ymax")),
            )
        except (TypeError, ValueError):
            stats["bad_box"] += 1
            continue
        rows.append((CLASS_TO_ID[name], *yolo))
        stats["kept"] += 1
    return filename, rows, dict(stats)


def read_split_ids(voc: Path, split: str) -> list[str]:
    path = voc / "ImageSets" / "Main" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing %s. If this is trainval-only, extract VOCtest_06-Nov-2007.tar "
            "into the same VOCdevkit first." % path
        )
    ids = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        image_id = line.strip().split()[0] if line.strip() else ""
        if not image_id:
            continue
        if image_id in seen:
            raise RuntimeError("Duplicate id %s in %s" % (image_id, path))
        seen.add(image_id)
        ids.append(image_id)
    if not ids:
        raise RuntimeError("Empty split file: %s" % path)
    return ids


def find_image(voc: Path, image_id: str, filename: str) -> Path:
    folder = voc / "JPEGImages"
    candidates = [folder / filename, folder / f"{image_id}.jpg"]
    stem = Path(filename).stem
    for ext in IMAGE_EXTS:
        candidates.append(folder / f"{stem}{ext}")
        candidates.append(folder / f"{image_id}{ext}")
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    raise FileNotFoundError("No JPEG for %s (%s) under %s" % (image_id, filename, folder))


def _looks_like_voc2007_yolo(output: Path) -> bool:
    return (output / "voc2007.yaml").is_file() or (output / "trainval" / "images").is_dir()


def _clear_output(output: Path, overwrite: bool) -> None:
    if not output.exists():
        return
    if not overwrite:
        raise FileExistsError(
            "Output already exists: %s (pass --overwrite to replace)" % output
        )
    if not _looks_like_voc2007_yolo(output):
        raise RuntimeError(
            "Refusing to delete %s; it does not look like a VOC2007 YOLO layout" % output
        )
    shutil.rmtree(output)


def _place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def yaml_text(dataset_root: str = ".") -> str:
    lines = [
        "# VOC2007-only. Train = official trainval (5011). Val/test = official test (4952).",
        "# Do not use ultralytics/cfg/datasets/VOC.yaml (that file is 07+12).",
        "# Do not load a D-Fire nc=2 teacher into this dataset.",
        "path: %s" % dataset_root,
        "train: trainval/images",
        "val: test/images",
        "test: test/images",
        "nc: 20",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append("  %d: %s" % (index, name))
    lines.append("")
    return "\n".join(lines)


def convert(
    source: Path,
    output: Path,
    mode: str = "link",
    overwrite: bool = False,
    dry_run: bool = False,
    strict: bool = True,
) -> dict[str, int]:
    voc = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not (voc / "Annotations").is_dir() or not (voc / "JPEGImages").is_dir():
        raise FileNotFoundError("Expected VOC2007 Annotations/ and JPEGImages/ under %s" % voc)

    trainval_ids = read_split_ids(voc, "trainval")
    test_ids = read_split_ids(voc, "test")
    overlap = sorted(set(trainval_ids) & set(test_ids))
    if overlap:
        raise RuntimeError("trainval/test overlap: %s" % overlap[:8])

    splits = {"trainval": trainval_ids, "test": test_ids}
    counts = {
        "trainval": len(trainval_ids),
        "test": len(test_ids),
        "difficult": 0,
        "unknown_class": 0,
        "empty_label": 0,
        "boxes_trainval": 0,
        "boxes_test": 0,
    }
    if strict:
        for split, expected in EXPECTED_COUNTS.items():
            if counts[split] != expected:
                raise RuntimeError(
                    "VOC2007 %s has %d ids, official is %d. Refusing to write a "
                    "non-standard split." % (split, counts[split], expected)
                )

    records: dict[str, dict[str, tuple[Path, list[tuple[int, float, float, float, float]]]]] = {
        "trainval": {},
        "test": {},
    }
    for split, image_ids in splits.items():
        for image_id in image_ids:
            xml_path = voc / "Annotations" / f"{image_id}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError("Missing annotation %s" % xml_path)
            filename, rows, stats = parse_voc_xml(xml_path)
            counts["difficult"] += int(stats.get("difficult", 0))
            counts["unknown_class"] += int(stats.get("unknown_class", 0))
            if not rows:
                counts["empty_label"] += 1
            counts["boxes_%s" % split] += len(rows)
            image = find_image(voc, image_id, filename)
            records[split][image_id] = (image, rows)

    if counts["unknown_class"]:
        raise RuntimeError(
            "Refusing to convert: %d objects have names outside the 20 VOC classes"
            % counts["unknown_class"]
        )
    if dry_run:
        return counts

    _clear_output(output, overwrite=overwrite)
    for split, image_ids in splits.items():
        list_path = output / "ImageSets" / f"{split}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for image_id in image_ids:
            image, rows = records[split][image_id]
            image_dst = output / split / "images" / image.name
            label_dst = output / split / "labels" / f"{Path(image.name).stem}.txt"
            _place_file(image, image_dst, mode)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.write_text(
                "".join("%d %.6f %.6f %.6f %.6f\n" % row for row in rows),
                encoding="utf-8",
            )
            lines.append(str(image_dst).replace("\\", "/") + "\n")
        list_path.write_text("".join(lines), encoding="utf-8")

    (output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    (output / "voc2007.yaml").write_text(yaml_text("."), encoding="utf-8")
    manifest = {
        "dataset": "VOC2007",
        "protocol": "trainval->train, test->val/test",
        "skip_difficult": True,
        "source": str(voc).replace("\\", "/"),
        "trainval": counts["trainval"],
        "test": counts["test"],
        "boxes_trainval": counts["boxes_trainval"],
        "boxes_test": counts["boxes_test"],
        "difficult_skipped": counts["difficult"],
        "empty_label": counts["empty_label"],
        "names": list(CLASS_NAMES),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VOC2007 trainval/test to YOLO layout"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("link", "copy"), default="link")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unofficial-split",
        action="store_true",
        help="Skip the 5011/4952 count gate (tests only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = convert(
        args.source,
        args.output,
        mode=args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        strict=not args.allow_unofficial_split,
    )
    dest = "DRY-RUN" if args.dry_run else str(args.output)
    print(
        "VOC2007 YOLO -> %s  trainval=%d test=%d "
        "boxes trainval/test=%d/%d difficult_skipped=%d empty_label=%d"
        % (
            dest,
            counts["trainval"],
            counts["test"],
            counts["boxes_trainval"],
            counts["boxes_test"],
            counts["difficult"],
            counts["empty_label"],
        )
    )


if __name__ == "__main__":
    main()
