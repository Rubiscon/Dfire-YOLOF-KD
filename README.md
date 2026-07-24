# D-Fire YOLOF Knowledge Distillation

Ultralytics fork for knowledge distillation on the D-Fire dataset: a **YOLO26n FPN teacher** trains a **YOLO26n-DCN YOLOF student** using early-backbone dictionary matching, neck feature alignment, and supplementary prediction alignment.

---

## Repository layout

```
Dfire-YOLOF-KD/
├── README.md
├── pyproject.toml
├── yolo26n-DCN.yaml              # student model definition
├── configs/datasets/dfire.yaml   # dataset path template
├── models/yolo26n-DCN.yaml       # same as root (structured copy)
├── scripts/
│   └── train_baselines.py       # all training entry points
├── tests/
│   └── smoke_*.py
└── ultralytics/                 # forked package (KD trainer)
```

---

## Requirements

- Python ≥ 3.10
- CUDA GPU (solo `batch=112`; KD defaults to `batch=32` — raise with `--batch` if VRAM allows; saliency uses a second teacher pass)
- PyTorch + torchvision

### Install

```bash
git clone https://github.com/<your-org>/Dfire-YOLOF-KD.git
cd Dfire-YOLOF-KD

# Use this forked ultralytics — do not pip-install the upstream package on top
pip install -e .
```

Optional: install DCNv4 for deformable conv in `DilatedDeformBlock`; the block falls back to standard dilated conv if DCN is unavailable.

---

## Dataset

Place D-Fire under:

```
datasets/D-Fire/data/
├── train/images  train/labels
├── val/images    val/labels
└── test/images   test/labels
```

Edit `configs/datasets/dfire.yaml` or `ultralytics/cfg/datasets/dfire.yaml` and set `path` to your local dataset root.

Training scripts resolve `data: dfire.yaml` via the Ultralytics dataset config path.

---

## Quick start

### Smoke tests

```bash
python tests/smoke_dict_kd.py
python tests/smoke_freeze_ema.py
```

### Baselines

Solo baselines use **`batch=112`**. KD baselines (`kd-early` / `kd-p0`) default to **`batch=32`** (student + teacher + dictionary saliency). Override with `--batch` when VRAM allows or on OOM.

```bash
# 1. YOLO26n FPN upper bound
python scripts/train_baselines.py --baseline yolo26n

# 2. YOLO26n-DCN student without distillation
python scripts/train_baselines.py --baseline dcn-solo

# 3. Early KD (dictionary n10↔x6 + neck + response; AT/late off)
python scripts/train_baselines.py --baseline kd-early

# 4. Full KD stack (same early dict + neck + response)
python scripts/train_baselines.py --baseline kd-p0

# Run all three in sequence
python scripts/train_baselines.py --baseline all
```

Outputs: `runs/detect/dfire-baselines/<run-name>/`.

### Test-split evaluation

```bash
python scripts/train_baselines.py --baseline kd-p0 --test-only \
  --weights runs/detect/dfire-baselines/baseline-kd-p0/weights/best.pt
```

---

## Distillation stack (kd-p0)

| Component | Mechanism |
|-----------|-----------|
| Backbone | Hard Q/V channel matching + Eq.(5) saliency-weighted L2 + negative-entropy attention restriction |
| Neck | `DeconvNet` projectors: student dilated blocks ↔ teacher FPN features |
| Response | Teacher NMS pseudo-labels → TAL assignment → box CIoU + cls KL; starts after epoch 20 |

### Key hyperparameters (defaults in `scripts/train_baselines.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch` | `112` (solo) / `32` (KD) | KD dual-model + saliency needs a lower default; use `--batch 112` if VRAM allows |
| `online_distill` | `True` | Joint teacher training until freeze epoch |
| `teacher_freeze_epoch` | `200` | Teacher remains joint-trained for the 200-epoch run |
| `dict_teacher_layers` | `[6]` | Early local x6 only; late x10 disabled |
| `dict_match` | `hard` | Proposal argmax channel matching |
| `dict_feature_norm` | `channel` | Stabilizes teacher/student scale; saliency controls spatial importance |
| `feature_loss` / `align_loss` | `0.08` / `0.12` | Neck feature KD / supplementary response KD |
| `dict_align_loss` / `dict_attn_loss` | `0.08` / `0.25` | Mean-normalized saliency-weighted L2 / mentor-specified negative attention entropy |
| `dict_commit_loss` | `0.0` | Hard match has no commitment term |

Full override keys are registered in `ultralytics/cfg/__init__.py`.

---

## Planned changes

| Item | Current | Target |
|------|---------|--------|
| Offline KD preset | manual overrides | `kd-offline` baseline |
| Model yaml location | root + `models/` duplicate | single path under `models/` |
| Ablation configs | manual edits | `configs/ablation/*.yaml` presets |
| `teacher_val_interval` | default `1` | set `0` during long runs |

---

## License

Forked from Ultralytics under **AGPL-3.0**. Redistribution and commercial use must comply with AGPL and the upstream license.

---
---

# D-Fire YOLOF 知识蒸馏

基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 的 fork，在 D-Fire 官方 train/val/test 划分上进行知识蒸馏：**YOLO26n FPN 教师** → **YOLO26n-DCN YOLOF 学生**，包含 backbone 字典匹配、neck 特征对齐与 response 级监督。

---

## 仓库结构

```
Dfire-YOLOF-KD/
├── README.md
├── pyproject.toml
├── yolo26n-DCN.yaml              # 学生模型定义
├── configs/datasets/dfire.yaml   # 数据集路径模板
├── models/yolo26n-DCN.yaml       # 与根目录相同（结构化备份）
├── scripts/
│   └── train_baselines.py       # 统一训练入口
├── tests/
│   └── smoke_*.py
└── ultralytics/                 # fork 后的包（含 KD trainer）
```

---

## 环境要求

- Python ≥ 3.10
- CUDA GPU（solo `batch=112`；KD 默认 `batch=32` — 显存够再上调；saliency 会多一次教师前向）
- PyTorch + torchvision

### 安装

```bash
git clone https://github.com/<your-org>/Dfire-YOLOF-KD.git
cd Dfire-YOLOF-KD

# 必须使用本仓库的 ultralytics，不要用 pip 官方包覆盖
pip install -e .
```

可选：安装 DCNv4 以启用 `DilatedDeformBlock` 的可变形卷积；未安装时会回退到普通 dilated conv。

---

## 数据集

目录结构：

```
datasets/D-Fire/data/
├── train/images  train/labels
├── val/images    val/labels
└── test/images   test/labels
```

编辑 `configs/datasets/dfire.yaml` 或 `ultralytics/cfg/datasets/dfire.yaml`，将 `path` 改为本地数据集路径。

训练脚本使用 `data: dfire.yaml`，由 Ultralytics 在配置目录中解析。

---

## 快速开始

### Smoke test

```bash
python tests/smoke_dict_kd.py
python tests/smoke_freeze_ema.py
```

### 基线训练

Solo 基线默认 **`batch=112`**。KD 基线（`kd-early` / `kd-p0`）默认 **`batch=32`**（学生+教师+dictionary saliency）。显存够或 OOM 时用 `--batch` 覆盖。

```bash
# 1. YOLO26n FPN 上界
python scripts/train_baselines.py --baseline yolo26n

# 2. YOLO26n-DCN 无蒸馏
python scripts/train_baselines.py --baseline dcn-solo

# 3. Early KD（dictionary n10↔x6 + neck + response；关闭 AT/late）
python scripts/train_baselines.py --baseline kd-early

# 4. 完整 KD（与 kd-early 相同 early dict + neck + response）
python scripts/train_baselines.py --baseline kd-p0

# 依次运行全部
python scripts/train_baselines.py --baseline all
```

输出目录：`runs/detect/dfire-baselines/<run-name>/`。

### 测试集评测

```bash
python scripts/train_baselines.py --baseline kd-p0 --test-only \
  --weights runs/detect/dfire-baselines/baseline-kd-p0/weights/best.pt
```

---

## 蒸馏结构（kd-p0）

| 模块 | 机制 |
|------|------|
| Backbone | Hard Q/V 通道匹配 + Eq.(5) saliency 加权 L2 + attention 负熵约束 |
| Neck | `DeconvNet` 投影：学生 dilated 块 ↔ 教师 FPN 特征 |
| Response | 教师 NMS 伪标签 → TAL 分配 → box CIoU + cls KL；第 20 epoch 后启用 |

### 主要超参（默认值见 `scripts/train_baselines.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch` | `112`（solo）/ `32`（KD） | KD 双模型 + saliency；显存够可用 `--batch 112` |
| `online_distill` | `True` | 冻结前教师与 GT 联合训练 |
| `teacher_freeze_epoch` | `200` | 200 epoch 内教师保持联合训练 |
| `dict_teacher_layers` | `[6]` | 仅 early x6；关闭 late x10 |
| `dict_match` | `hard` | Proposal 的 argmax 通道匹配 |
| `dict_feature_norm` | `channel` | 稳定师生特征尺度；saliency 仍独占空间权重 |
| `feature_loss` / `align_loss` | `0.08` / `0.12` | Neck 特征 KD / 补充 proposal 的 Response KD |
| `dict_align_loss` / `dict_attn_loss` | `0.08` / `0.25` | mean-norm saliency 加权 L2 / 导师指定的 attention 负熵约束 |
| `dict_commit_loss` | `0.0` | Hard match 不使用 commitment |

完整配置键见 `ultralytics/cfg/__init__.py`。

---

## 计划更改

| 项目 | 当前 | 目标 |
|------|------|------|
| 离线 KD 预设 | 手动 overrides | `kd-offline` baseline |
| 模型 yaml 路径 | 根目录与 `models/` 重复 | 统一到 `models/` |
| 消融实验配置 | 手改参数 | `configs/ablation/*.yaml` 预设 |
| `teacher_val_interval` | 默认每 epoch 验证教师 | 长训时设为 `0` |

---

## 许可证

基于 Ultralytics **AGPL-3.0** fork。分发与商用须遵守 AGPL 及上游许可。
