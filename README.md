# DFire-YOLOF-KD

Knowledge distillation from a YOLO26n feature-pyramid teacher into a YOLOF-style student
on the D-Fire smoke/fire benchmark. The student keeps the single-level YOLOF detection head
and replaces the dilated encoder on `C5` with a `DilatedDeformBlock`, so the whole method is
trained and evaluated inside one Ultralytics fork.

Three terms are added to the detection loss:

| term | what it matches | where |
| --- | --- | --- |
| **SSM** | sliced semantics: each student channel is assigned a teacher location by a cosine-similarity softmax, and the assigned slice is pulled toward the student feature | `ultralytics/nn/modules/yolof.py`, `YOLOFDistillationTrainer._dictionary_loss` |
| **RFM** | receptive field: dilated student blocks are projected onto the teacher pyramid levels they overlap | `feature_projectors`, `_feature_loss` |
| **RKD** | responses: teacher NMS pseudo-labels, TAL assignment, IoU on boxes and temperature-scaled KL on class scores | `_alignment_loss` |

The detector, the loss wiring and the training scripts live here; the manuscript is in
`paper/`.

## Repository layout

```
.
├── configs/datasets/dfire.yaml     # dataset descriptor
├── docs/                           # training, evaluation and comparison protocol
├── dfire_parity/                   # shared protocol kernel for the cross-framework comparison
├── yolo26n-DCN.yaml                # student (YOLOF head + DilatedDeformBlock)
├── paper/methods.tex               # manuscript draft
├── scripts/
│   ├── train_baselines.py          # single-file baseline and ablation runner
│   ├── train_dfs.py                # full protocol runner with the KD recipes
│   ├── eval_dfire.py               # checkpoint -> val/test metrics
│   └── convert_voc2007.py          # optional VOC2007 conversion
├── tests/                          # pytest suite + forward smoke tests
└── ultralytics/                    # forked package (trainer, modules, loss code)
```

## Installation

```bash
git clone https://github.com/<org>/DFire-YOLOF-KD.git
cd DFire-YOLOF-KD
pip install -e .
```

Python 3.10 or newer and a CUDA GPU are expected. Do not install the upstream `ultralytics`
wheel alongside this tree: the trainer and several modules are fork-only.

`DilatedDeformBlock` uses DCNv4 when it is importable and falls back to a plain dilated
convolution otherwise. Checkpoints trained with one backend are not interchangeable with the
other.

## Data

D-Fire in YOLO layout:

```
datasets/D-Fire/data/
├── train/images   train/labels
├── val/images     val/labels
└── test/images    test/labels
```

Point `path` in `configs/datasets/dfire.yaml` at that root, or copy the file to
`ultralytics/cfg/datasets/dfire.yaml` where Ultralytics resolves `data=dfire.yaml`.
Class order is `0=smoke`, `1=fire`.

VOC2007 is optional and only used for the second benchmark:

```bash
python scripts/convert_voc2007.py --src /path/to/VOCdevkit/VOC2007 --dst datasets/VOC2007_yolo
```

## Training

Both entry points default to batch 32 at 640 px for 200 epochs with Nesterov SGD
(`lr0=0.01`, `lrf=0.01`, `weight_decay=5e-4`) on the official D-Fire split. Warmup is fixed
at 101 optimizer updates, so it does not move when `--batch` changes.

### Smoke tests

```bash
pytest tests -x -q
python tests/smoke_dict_kd.py
```

### Baselines

`scripts/train_baselines.py` covers the teacher, the undistilled student and the ablation
arms, one preset per `--baseline`:

```bash
python scripts/train_baselines.py --baseline yolo26n     # YOLO26n teacher, no KD
python scripts/train_baselines.py --baseline dcn-solo    # student, no KD
python scripts/train_baselines.py --baseline early-dldx  # dictionary KD + dLdx saliency
python scripts/train_baselines.py --baseline infomax
python scripts/train_baselines.py --baseline entropy-align
python scripts/train_baselines.py --baseline all         # sequential queue
```

Runs land in `runs/detect/<project>/<name>/`. `--batch`, `--epochs`, `--device`,
`--workers` and `--name-suffix` override the preset; `--test-after-train` adds the test-split
evaluation to each run.

### Reported configuration

The full protocol runner carries the KD recipes as `--kd-variant`. The configuration behind
the results below is `ssm_sz`:

```bash
python scripts/train_dfs.py \
  --dataset dfire --baseline dcn-kd --kd-variant ssm_sz \
  --batch 32 --epochs 200
```

Related variants: `ssm` (same dictionary term without the two align gates), `ssm_sz_s6`
(student tap moved to the shape-matched stage), `ssm_soft_s6` (soft instead of argmax
assignment). Older launch lines that say `paperssm*` still resolve.

Teacher initialisation is always the COCO `yolo26n.pt`, and the teacher keeps training on the
dataset until the last epoch. Pass `--pretrained` to change the initialisation; do not pass a
finished in-dataset checkpoint as `--teacher-weights`, the runner rejects it.

### Related-work comparison

`docs/related_work_protocol.md` fixes the model contract, the optimiser, the augmentation and
the acceptance gates used when this method is compared with Structural KD, GID and CanKD.

## Evaluation

```bash
python scripts/eval_dfire.py \
  --data configs/datasets/dfire.yaml \
  --ckpt kd runs/detect/dfire-protocol-baselines/dfire-dcn-kd-200e/weights/best.pt \
  --out eval/kd
```

The distillation checkpoints store the student under `student.*` / `ema.*` prefixes; the
script extracts the EMA student, rewrites a plain Ultralytics `.pt` and reports val and test
metrics into `summary.json`. For plain detector checkpoints use the standard entry point:

```bash
python -c "from ultralytics import YOLO; YOLO('runs/detect/.../weights/best.pt').val(data='dfire.yaml', split='test')"
```

## Results

D-Fire, 200 epochs, batch 32, single seed. Validation is the split used for model selection;
test is evaluated once with the best checkpoint.

| model | val mAP@0.5 | test mAP@0.5 | test mAP@0.5:0.95 |
| --- | --- | --- | --- |
| YOLO26n teacher | 0.773 | – | – |
| YOLOF student, no KD | 0.704 | – | – |
| student + SSM + RFM + RKD | 0.730 | 0.719 | 0.393 |

The ablation table, the size-bucket breakdown and the cross-framework comparison are in the
manuscript.

## Configuration

Keys below are registered in `ultralytics/cfg/__init__.py`; everything else follows the
Ultralytics default namespace.

| key | default | meaning |
| --- | --- | --- |
| `online_distill` | `True` | teacher trains jointly on the dataset instead of staying frozen |
| `teacher_freeze_epoch` | `200` | first epoch with the teacher frozen (1-indexed) |
| `feature_loss` | `0.08` | RFM weight |
| `align_loss` | `0.12` | RKD weight |
| `align_box` / `align_cls` | `2.0` / `4.0` | box and class sub-weights of RKD |
| `distill_temperature` | `3` | KL temperature for the class term |
| `dict_teacher_layers` | `[6]` | teacher pyramid taps used by SSM |
| `dict_student_layer` | `10` | student layer used by SSM |
| `dict_match` | `straight_through` | argmax forward, soft assignment for the entropy term |
| `dict_weight` | `activation_mean` | per-location weight of the align residual |
| `dict_align_loss` | `0.1` | SSM weight |
| `dict_infomax_loss` | `0.1` | entropy term on the assignment |
| `align_size_weight` | `False` | scale the align terms by inverse box size |
| `align_box_conf_weight` | `False` | scale the box term by teacher confidence |

## License

Forked from [Ultralytics](https://github.com/ultralytics/ultralytics) under **AGPL-3.0**.
Redistribution and commercial use must comply with AGPL-3.0 and the upstream license.
