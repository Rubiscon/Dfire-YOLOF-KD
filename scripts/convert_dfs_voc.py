"""Convert E:\\CRIS\\DFS VOC fireDetectVOCfinal into a YOLO layout.

DFS is a separate dataset from D-Fire:
  class_names.txt = _background_ / fire / other / smoke
  YOLO names      = 0:fire, 1:other, 2:smoke  (background dropped)

Do not reuse D-Fire teachers or dfire_parity CLASS_NAMES. There is no official
split; this script writes a seed-0 stratified 70/15/15 split by filename prefix
(large / middle / small / other).

    python scripts/convert_dfs_voc.py
    python scripts/convert_dfs_voc.py --source /root/datasets/DFS/fireDetectVOCfinal \\
        --output /root/Ultra/datasets/DFS/data
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path(r"E:\CRIS\DFS\fireDetectVOCfinal")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "datasets" / "DFS" / "data"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
CLASS_NAMES = ("fire", "other", "smoke")
SPLIT_SEED = 0
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def load_class_names(source: Path) -> tuple[str, ...]:
    path = source / "class_names.txt"
    if not path.is_file():
        return CLASS_NAMES
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("_"):
            continue
        names.append(name)
    if tuple(names) != CLASS_NAMES:
        raise ValueError(
            "Unexpected DFS classes %s; expected %s" % (names, list(CLASS_NAMES))
        )
    return CLASS_NAMES


def voc_box_to_yolo(
    width: float,
    height: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size %s x %s" % (width, height))
    xmin = min(max(xmin, 0.0), width)
    xmax = min(max(xmax, 0.0), width)
    ymin = min(max(ymin, 0.0), height)
    ymax = min(max(ymax, 0.0), height)
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        raise ValueError("Empty box")
    xc = (xmin + xmax) / 2.0 / width
    yc = (ymin + ymax) / 2.0 / height
    return xc, yc, bw / width, bh / height


def parse_voc_xml(
    path: Path, class_to_id: dict[str, int]
) -> tuple[str, list[tuple[int, float, float, float, float]]]:
    root = ET.parse(path).getroot()
    filename = (root.findtext("filename") or f"{path.stem}.jpg").strip()
    size = root.find("size")
    if size is None:
        raise ValueError("%s has no <size>" % path)
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    rows = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            yolo = voc_box_to_yolo(
                width,
                height,
                float(box.findtext("xmin")),
                float(box.findtext("ymin")),
                float(box.findtext("xmax")),
                float(box.findtext("ymax")),
            )
        except ValueError:
            continue
        rows.append((class_to_id[name], *yolo))
    return filename, rows


def find_image(source: Path, filename: str) -> Path | None:
    stem = Path(filename).stem
    folders = (source / "JPEGImages", source / "images", source)
    candidates = [folder / filename for folder in folders]
    for folder in folders:
        for ext in IMAGE_EXTS:
            candidates.append(folder / f"{stem}{ext}")
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def split_stems(stems: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        prefix = stem.split("_", 1)[0] if "_" in stem else "rest"
        buckets[prefix].append(stem)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for prefix in sorted(buckets):
        items = sorted(buckets[prefix])
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            splits["train"].extend(items)
            continue
        if n == 2:
            splits["train"].append(items[0])
            splits["val"].append(items[1])
            continue
        n_train = max(1, int(round(n * TRAIN_FRAC)))
        n_val = max(1, int(round(n * VAL_FRAC)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val : n_train + n_val + n_test])
    for name in splits:
        splits[name] = sorted(splits[name])
    return splits


def _looks_like_dfs_layout(output: Path) -> bool:
    return (output / "classes.txt").is_file() or (output / "train" / "images").is_dir()


def _clear_output(output: Path, overwrite: bool) -> None:
    if not output.exists():
        return
    if not overwrite:
        raise FileExistsError(
            "Output already exists: %s (pass --overwrite to replace)" % output
        )
    if not _looks_like_dfs_layout(output):
        raise RuntimeError(
            "Refusing to delete %s; it does not look like a DFS YOLO layout"
            % output
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


def collect_records(
    source: Path,
) -> tuple[list[tuple[str, Path, list[tuple[int, float, float, float, float]]]], dict[str, int]]:
    source = source.expanduser().resolve()
    names = load_class_names(source)
    class_to_id = {name: index for index, name in enumerate(names)}
    xml_dir = source / "Annotations"
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError("No VOC xml under %s" % xml_dir)

    records = []
    skipped_image = 0
    empty_label = 0
    class_boxes = Counter()
    for xml_path in xml_files:
        filename, rows = parse_voc_xml(xml_path, class_to_id)
        image = find_image(source, filename)
        if image is None:
            skipped_image += 1
            continue
        if not rows:
            empty_label += 1
        for row in rows:
            class_boxes[int(row[0])] += 1
        records.append((xml_path.stem, image, rows))

    if not records:
        raise RuntimeError(
            "No VOC pairs with images. Looked for JPEGImages next to %s" % xml_dir
        )
    stats = {
        "total": len(records),
        "skipped_image": skipped_image,
        "empty_label": empty_label,
        **{f"boxes_{CLASS_NAMES[i]}": int(class_boxes[i]) for i in range(len(CLASS_NAMES))},
    }
    return records, stats


def convert(
    source: Path,
    output: Path,
    mode: str = "link",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    records, stats = collect_records(source)
    splits = split_stems([stem for stem, _, _ in records])
    by_stem = {stem: (image, rows) for stem, image, rows in records}
    counts = dict(stats)
    for split, stems in splits.items():
        counts[split] = len(stems)
    if dry_run:
        return counts

    _clear_output(output, overwrite=overwrite)

    for split, stems in splits.items():
        list_path = output / "ImageSets" / f"{split}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for stem in stems:
            image, rows = by_stem[stem]
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
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DFS VOC to YOLO")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("link", "copy"),
        default="link",
        help="Hardlink/symlink images (default) or copy them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing DFS YOLO layout at --output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse VOC and print split counts without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = convert(
        args.source,
        args.output,
        args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    dest = "DRY-RUN" if args.dry_run else str(args.output)
    print(
        "DFS YOLO layout -> %s  train=%d val=%d test=%d "
        "(kept=%d skipped_missing_image=%d empty_label=%d "
        "boxes fire/other/smoke=%d/%d/%d)"
        % (
            dest,
            counts["train"],
            counts["val"],
            counts["test"],
            counts["total"],
            counts["skipped_image"],
            counts["empty_label"],
            counts.get("boxes_fire", 0),
            counts.get("boxes_other", 0),
            counts.get("boxes_smoke", 0),
        )
    )


if __name__ == "__main__":
    main()
