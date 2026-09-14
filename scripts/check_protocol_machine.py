"""Probe an AutoDL (or local) machine for the CONTROLLED_ARGS train_dfs layout.

Old AutoDL images often lack /root/autodl-tmp, mount data under /root/autodl-fs,
or keep D-Fire at a legacy path. Run this from the fork root:

    cd /root/Ultra
    python scripts/check_protocol_machine.py
    python scripts/check_protocol_machine.py --print-launch
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHT_CANDIDATES = (
    Path("/root/autodl-fs/weights/yolo26n.pt"),
    ROOT / "weights" / "yolo26n.pt",
    Path("/root/Ultra/weights/yolo26n.pt"),
    Path("yolo26n.pt"),
)

DATA_CANDIDATES = (
    ROOT / "datasets" / "D-Fire" / "data",
    Path("/root/Ultra/datasets/D-Fire/data"),
    Path("/root/datasets/D-Fire/data"),
    Path("/root/autodl-tmp/datasets/D-Fire/data"),
    Path("/root/autodl-fs/datasets/D-Fire/data"),
    Path("/root/autodl-fs/D-Fire/data"),
    ROOT / "datasets" / "D-Fire_old" / "data",
)

RUN_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp"),
    Path("/root/autodl-fs"),
    ROOT,
)


def _ok(flag: bool) -> str:
    return "OK " if flag else "MISS"


def _count(path: Path) -> int:
    if not path.is_dir():
        return -1
    n = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    n += 1
    except OSError:
        return -1
    return n


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _disk(path: Path) -> str:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
    except OSError:
        return "n/a"
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    return "%.1fG free / %.1fG" % (free_gb, total_gb)


def find_weight() -> Path | None:
    for path in WEIGHT_CANDIDATES:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


def find_data() -> tuple[Path | None, dict[str, int]]:
    for path in DATA_CANDIDATES:
        train = path / "train" / "images"
        val = path / "val" / "images"
        test_a = path / "tests" / "images"
        test_b = path / "test" / "images"
        if train.is_dir() and val.is_dir():
            counts = {
                "train": _count(train),
                "val": _count(val),
                "tests": _count(test_a),
                "test": _count(test_b),
            }
            return path, counts
    return None, {}


def find_run_root() -> Path:
    for path in RUN_ROOT_CANDIDATES:
        if path.exists() and _writable(path / "runs"):
            return path
    fallback = ROOT / "runs"
    fallback.mkdir(parents=True, exist_ok=True)
    return ROOT


def report() -> dict[str, object]:
    print("cwd:", Path.cwd())
    print("fork:", ROOT)
    print("python:", sys.executable, sys.version.split()[0])

    train_dfs = ROOT / "scripts" / "train_dfs.py"
    has_dataset = False
    if train_dfs.is_file():
        text = train_dfs.read_text(encoding="utf-8")
        has_dataset = '"--dataset"' in text and "dfire" in text
    print("[%s] scripts/train_dfs.py --dataset dfire" % _ok(has_dataset))

    dfire_yaml = ROOT / "ultralytics" / "cfg" / "datasets" / "dfire.yaml"
    print("[%s] %s" % (_ok(dfire_yaml.is_file()), dfire_yaml))

    yolo26 = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
    print("[%s] %s" % (_ok(yolo26.is_file()), yolo26))

    parity = ROOT / "dfire_parity" / "protocol.py"
    print("[%s] %s" % (_ok(parity.is_file()), parity))

    ultra_file = None
    has_fixed_accumulate = False
    try:
        import ultralytics
        from ultralytics.utils import DEFAULT_CFG_DICT

        ultra_file = ultralytics.__file__
        has_fixed_accumulate = "fixed_accumulate" in DEFAULT_CFG_DICT
    except Exception as exc:
        ultra_file = "import failed: %s" % exc
    print("[%s] ultralytics %s" % (_ok(ultra_file is not None and "failed" not in str(ultra_file)), ultra_file))
    print("[%s] DEFAULT_CFG has fixed_accumulate" % _ok(has_fixed_accumulate))

    weight = find_weight()
    print("[%s] yolo26n.pt  %s" % (_ok(weight is not None), weight or WEIGHT_CANDIDATES[0]))

    data, counts = find_data()
    print("[%s] D-Fire data %s" % (_ok(data is not None), data or DATA_CANDIDATES[0]))
    if counts:
        print("         images train/val/tests/test = %(train)s/%(val)s/%(tests)s/%(test)s" % counts)
        if counts.get("tests", -1) <= 0 and counts.get("test", 0) > 0:
            print("         NOTE: yaml expects tests/images, this dump has test/images")

    expected = ROOT / "datasets" / "D-Fire" / "data"
    if data is not None and data.resolve() != expected.resolve():
        print("         yaml path is %s" % expected)
        print("         symlink: ln -s %s %s" % (data, expected))

    print("mounts:")
    for path in (Path("/root/autodl-tmp"), Path("/root/autodl-fs"), Path("/root"), ROOT):
        exists = path.exists()
        print(
            "  [%s] %s  %s"
            % (_ok(exists), path, _disk(path) if exists else "does not exist")
        )

    run_root = find_run_root()
    print("[%s] writable run root %s/runs  %s" % (_ok(True), run_root, _disk(run_root)))

    cuda = False
    torch_ver = "not importable"
    try:
        import torch

        torch_ver = torch.__version__
        cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        torch_ver = "import failed: %s" % exc
    print("[%s] torch %s  cuda=%s" % (_ok(cuda), torch_ver, cuda))

    issues = []
    if not has_dataset:
        issues.append("upload the current scripts/train_dfs.py (needs --dataset dfire)")
    if not has_fixed_accumulate:
        issues.append(
            "this Ultra tree is older than CONTROLLED_ARGS; sync ultralytics/cfg/default.yaml, "
            "ultralytics/engine/trainer.py, and dfire_parity/"
        )
    if weight is None:
        issues.append("place COCO yolo26n.pt (not a D-Fire nc=2 teacher)")
    if data is None:
        issues.append("D-Fire YOLO layout missing (train/images + val/images)")
    elif counts.get("train", 0) < 1000:
        issues.append("train/images count looks too small for official D-Fire")
    if not dfire_yaml.is_file():
        issues.append("ultralytics/cfg/datasets/dfire.yaml missing")
    if not cuda:
        issues.append("this python cannot see CUDA; use the image's conda/venv python")

    print("issues:" if issues else "issues: none")
    for item in issues:
        print("  -", item)

    return {
        "weight": weight,
        "data": data,
        "counts": counts,
        "run_root": run_root,
        "has_dataset": has_dataset,
        "expected_data": expected,
        "issues": issues,
    }


def print_launch(info: dict[str, object]) -> None:
    run_root = Path(str(info["run_root"]))
    weight = info["weight"] or Path("/root/autodl-fs/weights/yolo26n.pt")
    proj = run_root / "runs" / "dfire-protocol-baselines"
    log = run_root / "runs" / "dfire_yolo26n_protocol.log"
    expected = Path(str(info["expected_data"]))
    data = info["data"]
    print("")
    print("# paste as one block")
    print("cd %s" % ROOT)
    if data is not None and Path(str(data)).resolve() != expected.resolve():
        print("mkdir -p %s" % expected.parent)
        print("ln -sfn %s %s" % (data, expected))
    print("mkdir -p %s" % proj)
    print(
        "nohup python scripts/train_dfs.py --dataset dfire --baseline yolo26n "
        "--device 0 --pretrained %s --project %s --name-suffix seed0 "
        "> %s 2>&1 & echo pid $!; sleep 3; tail -n 40 %s"
        % (weight, proj, log, log)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AutoDL layout for protocol training")
    parser.add_argument("--print-launch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = report()
    if args.print_launch:
        print_launch(info)


if __name__ == "__main__":
    main()
