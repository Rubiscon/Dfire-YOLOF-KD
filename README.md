# D-Fire YOLOF Knowledge Distillation

Ultralytics fork for knowledge distillation on the D-Fire dataset: a **YOLO26n FPN teacher** trains a **YOLO26n-DCN YOLOF student** using backbone dictionary matching, neck feature alignment, and response-level supervision.

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

# 3. Early-stage KD (dictionary n10↔x6 + neck + response)
python scripts/train_baselines.py --baseline early-dldx

# 4. Straight-through InfoMax matching experiment
python scripts/train_baselines.py --baseline infomax

# 5. Positive spatial-entropy weighted align
python scripts/train_baselines.py --baseline entropy-align

# Run all baselines in sequence
python scripts/train_baselines.py --baseline all
```

Outputs: `runs/detect/dfire-baselines/<run-name>/`.

### InfoMax matching branch

The `infomax` baseline keeps the stable N losses (dLdx saliency and spatial-energy
attention) and changes only dictionary assignment:

- `A_soft = softmax(M / tau)` with L2-normalized Q/K tokens.
- Forward assignment is hard argmax; backward uses `A_soft` (straight-through).
- `L_InfoMax = H(T|S) - lambda H(T)` encourages confident student-channel matches
  without collapsing all queries onto a few teacher channels.
- Weighted align and AT use a detached reorganized-teacher target. Q/K are trained
  only by commitment and InfoMax, so assignment cannot reduce align by moving its target.

`dict_match_stats.csv` is channel-assignment diagnostics only. Its entropy columns are
explicitly named `channel_conditional_entropy` and `channel_marginal_entropy` (plus
`channel_effective_teacher_channels` and `channel_infomax_loss`) so they cannot be
confused with the spatial row entropy used by align.

### Positive spatial-entropy align

The `entropy-align` baseline keeps hard dictionary matching and the proven spatial-energy
AT loss, replacing the align spatial weight with positive cross-feature entropy:

- Pool the independently projected student query and reorganized teacher value features.
- `A = softmax(Q V^T / tau, dim=2)`, with shape `(B, Nq, Nv)`.
- `H_i = -sum_j A_ij log(A_ij)` is positive row entropy. The mentor's written
  double-negative expression would instead produce negative entropy.
- `H/log(Nv)` gives a resolution-independent `[0, 1]` map; a configurable positive
  floor prevents low-entropy positions from disabling align supervision.
- The entropy map is detached before weighting, so align optimizes the student projection
  rather than learning to hide high-error positions.

`dict_weight_stats.csv` independently records the actual normalized spatial row entropy
(mean/std/min/max), floor-hit ratio, the final align weight statistics, entropy grid, and
the entropy's per-pixel correlation with the align residual. It is emitted at
`dict_match_log_interval` for entropy-based modes.

Relevant options are `dict_entropy_temp`, `dict_entropy_grid_divisor`,
`dict_entropy_floor`, and `dict_weight_norm=mean|minmax`. The baseline explicitly keeps
`mean`; `minmax` is opt-in and removes the absolute affine scale introduced by the floor.
`dict_weight=entropy_inverse` emphasizes low-entropy/high-confidence locations, while
`dict_weight=dldx_entropy_gate` keeps task-gradient saliency as the primary weight and
uses positive entropy only as a gate. Both are opt-in candidates; existing recipes and
the `entropy-align` training formula are unchanged.

### Test-split evaluation

```bash
python scripts/train_baselines.py --baseline kd-p0 --test-only \
  --weights runs/detect/dfire-baselines/baseline-kd-p0/weights/best.pt
```

---

## Distillation stack (kd-p0)

| Component | Mechanism |
|-----------|-----------|
| Backbone | Soft dictionary matching + Grad-CAM weighted align + AT attention restriction + commit |
| Neck | `DeconvNet` projectors: student dilated blocks ↔ teacher FPN features |
| Response | Teacher NMS pseudo-labels → TAL assignment → box CIoU + cls KL |

### Key hyperparameters (defaults in `scripts/train_baselines.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch` | `112` (solo) / `32` (KD) | KD dual-model + saliency needs a lower default; use `--batch 112` if VRAM allows |
| `online_distill` | `True` | Joint teacher training until freeze epoch |
| `teacher_freeze_epoch` | `110` | 1-indexed; teacher frozen afterward |
| `dict_teacher_layers` | `[6, 10]` (`kd-p0`) / `[6]` (`kd-early`) | Early local (x6) + late semantic (x10) |
| `dict_match` | `soft` | Soft cross-attention gather; `hard` = legacy argmax |
| `dict_feature_norm` | `none` | Dict path; neck still uses `feature_norm=channel` |
| `feature_loss` / `align_loss` | `0.08` / `0.12` | Neck and response weights |
| `dict_align_loss` / `dict_attn_loss` | `0.10` / `0.06` | Attn uses mean MSE (not HW-sum) |
| `dict_commit_loss` | `0.05` | Soft-match query↔key commitment |
| `dict_attn_start_epoch` | `20` | 1-indexed delay for attention restriction |

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

# 3. Early-stage KD（dictionary n10↔x6 + neck + response）
python scripts/train_baselines.py --baseline early-dldx

# 4. Straight-through InfoMax matching 实验
python scripts/train_baselines.py --baseline infomax

# 5. 正空间熵加权 align
python scripts/train_baselines.py --baseline entropy-align

# 依次运行全部
python scripts/train_baselines.py --baseline all
```

输出目录：`runs/detect/dfire-baselines/<run-name>/`。

### InfoMax matching 分支

`infomax` 保留稳定 N 的 dLdx saliency 与旧 spatial-energy AT，仅替换 dictionary assignment：

- 对 L2 归一化后的 Q/K 使用 `A_soft=softmax(M/tau)`；
- 前向保持 proposal 的 hard argmax，反向通过 `A_soft`（straight-through）；
- 最小化 `H(T|S)-lambda H(T)`，兼顾明确匹配与教师通道均衡使用；
- weighted align 与 AT 均 detach 重组后的教师 target；Q/K 只由 commitment 和 InfoMax
  训练，避免 assignment 通过移动 target 来虚假降低 align。

训练时 `dict_match_stats.csv` 仅记录通道匹配诊断；熵字段明确命名为
`channel_conditional_entropy`、`channel_marginal_entropy`，并包含
`channel_effective_teacher_channels` 与 `channel_infomax_loss`，避免与 align 使用的
空间逐行熵混淆。

### 正空间熵 align

`entropy-align` 保留 hard dictionary matching 与已验证的 spatial-energy AT，只将 align
空间权重替换为跨特征正熵：

- 对独立投影后的学生 query 与重组教师 value 特征池化；
- `A=softmax(Q V^T/tau, dim=2)`，其形状为 `(B,Nq,Nv)`；
- `H_i=-sum_j A_ij log(A_ij)` 是正的逐行熵。导师文字中的双重负号会得到负熵；
- 用 `H/log(Nv)` 归一化至 `[0,1]`，并设置正下限，避免低熵位置完全关闭 align；
- 熵图在加权前 detach，避免模型通过压低高误差位置权重来投机。

独立的 `dict_weight_stats.csv` 按 `dict_match_log_interval` 记录真实空间逐行熵的
mean/std/min/max、floor 命中比例、最终 align 权重统计、entropy grid，以及它与
逐像素 align residual 的相关系数。

相关参数为 `dict_entropy_temp`、`dict_entropy_grid_divisor`、
`dict_entropy_floor` 与 `dict_weight_norm=mean|minmax`。baseline 显式保持 `mean`；
`minmax` 为 opt-in，且会消去 floor 的 affine 绝对尺度。`entropy_inverse` 强调
低熵/高置信位置；`dldx_entropy_gate` 以任务梯度 saliency 为主权重、正熵仅作调制。
两者均为显式候选，不改变现有 `entropy-align` 的默认训练公式。

### 测试集评测

```bash
python scripts/train_baselines.py --baseline kd-p0 --test-only \
  --weights runs/detect/dfire-baselines/baseline-kd-p0/weights/best.pt
```

---

## 蒸馏结构（kd-p0）

| 模块 | 机制 |
|------|------|
| Backbone | 软字典匹配 + Grad-CAM 加权对齐 + AT attention 约束 + commit |
| Neck | `DeconvNet` 投影：学生 dilated 块 ↔ 教师 FPN 特征 |
| Response | 教师 NMS 伪标签 → TAL 分配 → box CIoU + cls KL |

### 主要超参（默认值见 `scripts/train_baselines.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch` | `112`（solo）/ `32`（KD） | KD 双模型 + saliency；显存够可用 `--batch 112` |
| `online_distill` | `True` | 冻结前教师与 GT 联合训练 |
| `teacher_freeze_epoch` | `110` | 1-indexed；之后教师冻结 |
| `dict_teacher_layers` | `[6, 10]`（kd-p0）/ `[6]`（kd-early） | early 局部 + late 语义 |
| `dict_match` | `soft` | 软交叉注意力聚合；`hard` 为旧版 argmax |
| `dict_feature_norm` | `none` | 字典路径；neck 仍用 `feature_norm=channel` |
| `feature_loss` / `align_loss` | `0.08` / `0.12` | Neck / Response 权重 |
| `dict_align_loss` / `dict_attn_loss` | `0.10` / `0.06` | attn 为 mean MSE（非 HW-sum） |
| `dict_commit_loss` | `0.05` | 软匹配 query↔key commitment |
| `dict_attn_start_epoch` | `20` | 1-indexed，延迟启动 attention 约束 |

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
