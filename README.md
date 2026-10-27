# DFire-YOLOF-KD

Knowledge distillation from a YOLO26n feature-pyramid teacher into a YOLOF-style student on
the D-Fire smoke/fire benchmark, inside one Ultralytics fork.


## Layout

| path | contents |
| --- | --- |
| `ultralytics/` | forked package: detector, distillation losses, trainer |
| `scripts/train_baselines.py` | teacher, undistilled student, ablation arms |
| `scripts/train_dfs.py` | protocol runner with the distillation recipes |
| `scripts/eval_dfire.py` | distillation checkpoint to val/test metrics |
| `scripts/convert_voc2007.py` | optional VOC2007 conversion |
| `dfire_parity/` | training protocol shared by the comparison runs |
| `models/` | student model and the head/tap variants |
| `tests/` | unit tests and forward smoke tests |

## Install

```bash
git clone https://github.com/Rubiscon/Dfire-YOLOF-KD.git
cd Dfire-YOLOF-KD
pip install -e .
```

Python 3.10+ and a CUDA GPU. Do not install the upstream `ultralytics` wheel alongside this
tree. `DilatedDeformBlock` uses DCNv4 when importable and a plain dilated convolution
otherwise; checkpoints from the two backends are not interchangeable.

## Data

D-Fire in YOLO layout under `datasets/D-Fire/data/`, with `train/`, `val/`, and `tests/`
folders, each holding `images/` and `labels/`. The Kaggle release names the last split
`test/`.

`ultralytics/cfg/datasets/dfire.yaml` is the dataset descriptor. Ultralytics resolves a
relative `path` against its `datasets_dir` setting, so either put an absolute path there or
set `datasets_dir` to the directory that holds `D-Fire/`. Classes are `0=smoke` and `1=fire`.

## Training

640 px, 200 epochs, batch 32, Nesterov SGD (`lr0=0.01`, `lrf=0.01`, `weight_decay=5e-4`),
and seed 0. The teacher starts from COCO `yolo26n.pt`. Warmup covers 101 optimizer updates,
so it stays fixed when `--batch` changes.

```bash
python scripts/train_baselines.py --baseline yolo26n     # teacher
python scripts/train_baselines.py --baseline dcn-solo    # student, no KD
python scripts/train_baselines.py --baseline all         # ablation queue

python scripts/train_dfs.py --dataset dfire --baseline dcn-kd \
  --kd-variant ssm_sz --batch 32 --epochs 200 --name-suffix ssm_sz
```

Shipped variants are `ssm`, `ssm_sz`, `ssm_sz_s6`, and `ssm_soft_s6`; the rest are ablation
arms. The variant is not part of the default run name, so pass `--name-suffix` when several
variants share a project. Recipes and loss weights are in `_KD_RECIPE` in
`scripts/train_dfs.py`.

Runs land in `runs/detect/<project>/<name>/`.

## Evaluation

```bash
python scripts/eval_dfire.py --data ultralytics/cfg/datasets/dfire.yaml \
  --ckpt kd runs/detect/dfire-protocol-baselines/dfire-dcn-kd-200e-ssm_sz/weights/best.pt \
  --out eval/kd
```

Distillation checkpoints keep the student under `student.*` or `ema.*`, which `YOLO.val`
cannot read. The script extracts the EMA student, runs the standard validator on val and
test, and writes `summary.json`. Plain detector checkpoints go straight into
`YOLO(...).val(...)`.

## License

Forked from [Ultralytics](https://github.com/ultralytics/ultralytics) and licensed under
**AGPL-3.0**. Redistribution and commercial use are bound by that license.
