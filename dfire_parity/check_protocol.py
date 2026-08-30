"""Quick print of the full-aug controlled protocol for server smoke checks."""

from dfire_parity import CONTROLLED_ARGS, REFERENCE_MAP50, PARITY_TOLERANCE, build_args, steps_per_epoch

KEYS = [
    "batch", "nbs", "amp", "optimizer", "momentum", "warmup_epochs", "warmup_momentum",
    "mosaic", "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr",
]


def main():
    args = build_args()
    print("protocol gate mAP50 = {:.3f} ± {:.3f}".format(REFERENCE_MAP50, PARITY_TOLERANCE))
    print("steps/epoch @14122 =", steps_per_epoch(14122))
    for key in KEYS:
        print(f"  {key}={getattr(args, key)!r}  (CONTROLLED={CONTROLLED_ARGS[key]!r})")


if __name__ == "__main__":
    main()
