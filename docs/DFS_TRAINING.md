# DFS dataset (separate from D-Fire)

Local source: `E:\CRIS\DFS\fireDetectVOCfinal` (VOC `Annotations` + `JPEGImages`).

This is **not** D-Fire. Classes after dropping VOC `_background_` are:

| id | name |
|---:|------|
| 0 | fire |
| 1 | other |
| 2 | smoke |

D-Fire is `0=smoke, 1=fire`. Do **not** load D-Fire `yolo26n_teacher_*.pt` or reuse SKD/GID/CanKD `dfire_parity` class gates. Report DFS as its own table.

There is no official split. `scripts/convert_dfs_voc.py` writes a **seed=0 stratified 70/15/15** layout under `datasets/DFS/data/{train,val,test}`, stratified by filename prefix (`large` / `middle` / `small` / `other`). Current VOC dump: **9462** pairs → train **6623** / val **1419** / test **1420**. Boxes: fire 14948, other 5041, smoke 5727.

Schedule matches the D-Fire controlled protocol (200e, batch 112, SGD, seed 0, fullaug). Absolute warmup *steps* scale with DFS train size; `warmup_epochs` itself stays the protocol value.

## 1. Convert (local)

```powershell
cd E:\CRIS\Dfire-YOLOF-KD
python scripts/convert_dfs_voc.py
```

Expect `datasets/DFS/data/train/images` plus `ImageSets/*.txt`. `datasets/` is gitignored; the YOLO tree is produced on each machine. Re-run with `--overwrite` only if you intend to replace that tree (do not change the seed after a run has started).

Dry-run counts without writing:

```powershell
python scripts/convert_dfs_voc.py --dry-run
```

## 2. Train (after convert)

First **YOLO26n** (future DFS teacher), then **YOLOF-DCN solo**. Distillation on DFS starts only after that teacher exists. Init from COCO `yolo26n.pt`, never from a D-Fire nc=2 checkpoint.

```powershell
python scripts/train_dfs.py --baseline yolo26n --pretrained yolo26n.pt --name-suffix seed0
python scripts/train_dfs.py --baseline dcn-solo --pretrained yolo26n.pt --name-suffix seed0
```

Runs land in `runs/detect/dfs-baselines/`. After training:

```powershell
python scripts/train_dfs.py --baseline yolo26n --test-only --weights runs/detect/dfs-baselines/dfs-yolo26n-200e-seed0/weights/best.pt
python scripts/train_dfs.py --baseline dcn-solo --test-only --weights runs/detect/dfs-baselines/dfs-dcn-solo-200e-seed0/weights/best.pt
```

## 3. Sync to AutoDL

Do **not** rsync `/root/Ultra` while a D-Fire job is writing there. DFS **VOC data** can sync in parallel because it lives outside the code tree. Convert on the GPU machine so you do not duplicate 9462 images.

Suggested cloud layout:

```text
/root/datasets/DFS/fireDetectVOCfinal/   # VOC original (rsync this)
/root/Ultra/datasets/DFS/data/           # YOLO layout from the converter
/root/autodl-tmp/runs/dfs-baselines/     # optional run dir on the data disk
```

Host alias is `autodl-dfire` (`C:\Users\crism\.ssh\config`). If the instance was rebuilt, update `HostName` / `Port` from the AutoDL console first:

```powershell
ssh autodl-dfire "hostname; nvidia-smi -L; ls /root/Ultra /root/datasets"
```

### 3.1 VOC images (bulk)

Git Bash or WSL (preferred):

```bash
ssh autodl-dfire "mkdir -p /root/datasets/DFS"
rsync -avz --progress \
  /e/CRIS/DFS/fireDetectVOCfinal/ \
  autodl-dfire:/root/datasets/DFS/fireDetectVOCfinal/
```

PowerShell fallback if `rsync` is missing:

```powershell
ssh autodl-dfire "mkdir -p /root/datasets/DFS"
scp -r E:\CRIS\DFS\fireDetectVOCfinal autodl-dfire:/root/datasets/DFS/
```

### 3.2 Code (only if no Ultra job is writing)

```bash
rsync -avz --progress \
  --exclude .git --exclude runs --exclude Log --exclude datasets --exclude __pycache__ \
  /e/CRIS/Dfire-YOLOF-KD/ \
  autodl-dfire:/root/Ultra/
```

Minimum files if you want a surgical copy while another process owns `/root/Ultra`:

```text
scripts/convert_dfs_voc.py
scripts/train_dfs.py
ultralytics/cfg/datasets/dfs.yaml
docs/DFS_TRAINING.md
```

Copy those with `scp` into the matching paths under `/root/Ultra` instead of a full tree rsync.

### 3.3 Convert + train on the GPU

```bash
cd /root/Ultra
python scripts/convert_dfs_voc.py \
  --source /root/datasets/DFS/fireDetectVOCfinal \
  --output /root/Ultra/datasets/DFS/data

PRE=/root/autodl-fs/weights/yolo26n.pt
# If that path is empty, use the same yolo26n.pt already used for D-Fire students.
PROJ=/root/autodl-tmp/runs/dfs-baselines

nohup python scripts/train_dfs.py --baseline yolo26n --device 0 \
  --pretrained $PRE --project $PROJ --name-suffix seed0 \
  > /root/autodl-tmp/runs/dfs_yolo26n.log 2>&1 &

# After yolo26n finishes:
nohup python scripts/train_dfs.py --baseline dcn-solo --device 0 \
  --pretrained $PRE --project $PROJ --name-suffix seed0 \
  > /root/autodl-tmp/runs/dfs_dcn_solo.log 2>&1 &
```

AutoDL has no tmux on some images; `nohup` is the usual pattern. Check with `nvidia-smi` and `tail -f /root/autodl-tmp/runs/dfs_yolo26n.log`.

Pull results back:

```powershell
mkdir Log\dfs-baselines -Force
scp -r autodl-dfire:/root/autodl-tmp/runs/dfs-baselines Log/dfs-baselines/
# if you left the default relative project:
# scp -r autodl-dfire:/root/Ultra/runs/detect/dfs-baselines Log/dfs-baselines/
```

## 4. What not to do

- Do not point CanKD / SKD / GID YAML at `dfs.yaml` until those wrappers accept `nc=3` and a **DFS-trained** teacher.
- Do not compare DFS mAP to D-Fire ~0.70 numbers; class set, images, and domain all differ.
- Do not convert with a different seed if you already started a run; the split is part of the protocol.
- Do not rsync `/root/Ultra` over a live training process.

## 5. Same protocol on D-Fire (diagnostic)

`scripts/train_dfs.py --dataset dfire --baseline yolo26n` reuses **identical** `CONTROLLED_ARGS` on official D-Fire. It is not `train_baselines.py --baseline yolo26n` (that path is `optimizer=auto` / different augs). Use this to test whether DFS val mAP50 ~0.40 is the recipe or the data.

Copy **the fork**, not only `train_dfs.py`. `CONTROLLED_ARGS` needs `fixed_accumulate` in `ultralytics/cfg/default.yaml`. An older `/root/Ultra` raises `SyntaxError: 'fixed_accumulate' is not a valid YOLO argument` and dumps the stock `yolo` CLI help.

Old AutoDL images often have **no** `/root/autodl-tmp` (or no `runs/` under it). Redirecting the log there makes bash exit before Python starts:

`bash: /root/autodl-tmp/runs/dfire_yolo26n_protocol.log: No such file or directory`

Probe first, then start. The checker prints a one-block launch that uses a writable disk (`/root/autodl-tmp`, else `/root/autodl-fs`, else `/root/Ultra`):

```bash
cd /root/Ultra
python scripts/check_protocol_machine.py --print-launch
```

Or create the directory yourself and launch in one block (no `\` line breaks):

```bash
cd /root/Ultra
mkdir -p /root/autodl-tmp/runs /root/autodl-fs/runs
PRE=/root/autodl-fs/weights/yolo26n.pt
if [ ! -f "$PRE" ]; then PRE=/root/Ultra/weights/yolo26n.pt; fi
if mkdir -p /root/autodl-tmp/runs 2>/dev/null; then RUN=/root/autodl-tmp/runs; else RUN=/root/autodl-fs/runs; mkdir -p "$RUN"; fi
PROJ=$RUN/dfire-protocol-baselines
LOG=$RUN/dfire_yolo26n_protocol.log
nohup python scripts/train_dfs.py --dataset dfire --baseline yolo26n --device 0 --pretrained $PRE --project $PROJ --name-suffix seed0 > $LOG 2>&1 & echo pid $!; sleep 3; tail -n 40 $LOG
```

Healthy D-Fire val mAP50 on this stack should be **~0.70+** by mid training, not a 0.40 plateau. After it finishes, copy the **whole** run dir (including `weights/`):

```powershell
scp -r autodl-dfire:/root/autodl-tmp/runs/dfire-protocol-baselines Log1/dfire-protocol-baselines/
```
