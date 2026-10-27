# Evaluation

## Checkpoints

A plain detector checkpoint (teacher or undistilled student) is a standard Ultralytics `.pt`
and can go straight into `YOLO(...).val`.

A distillation checkpoint stores both models. The student weights sit under `student.*` and
the exponential moving average under `ema.*`, which `YOLO.val` cannot read. `scripts/eval_dfire.py`
extracts the EMA student state, rewrites it against `yolo26n-DCN.yaml` and evaluates the
result, so the numbers come from the same validator as every other row in a comparison table.

```bash
python scripts/eval_dfire.py \
  --data configs/datasets/dfire.yaml \
  --student-yaml yolo26n-DCN.yaml \
  --ckpt kd runs/detect/dfire-protocol-baselines/dfire-dcn-kd-200e/weights/best.pt \
  --out eval/kd
```

`--ckpt NAME PATH` may be repeated; `--ultra NAME PATH` evaluates an ordinary checkpoint in
the same call. The script writes `<name>_student.pt` and `summary.json` with precision, recall,
mAP@0.5 and mAP@0.5:0.95 for the val and test splits.

`run_eval.py` at the repository root wraps the same script and pins `sys.path` to this tree
before importing `ultralytics`, which is needed on hosts that also have an editable install of
the package elsewhere.

## Splits

D-Fire ships an official `train`/`val`/`test` split. Model selection uses `val`; `test` is
evaluated once with the selected checkpoint. Four test images carry out-of-bounds labels and
are skipped by the loader, which reports them as corrupt; the count is the same for every arm
on the same host, so it does not change a comparison.

## Reproducing a table row

1. Read `args.yaml` from the run directory and confirm the variant, batch and loss weights.
2. Re-train with the same `--kd-variant`, or copy `best.pt` if the run is archived.
3. Evaluate with `scripts/eval_dfire.py` and record `summary.json` next to the row.
4. Report val and test separately; a val number is a selection statistic, not a held-out one.
