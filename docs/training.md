# Training

Two entry points share one protocol. `scripts/train_baselines.py` is the short runner used
for the teacher, the undistilled student and the ablations; `scripts/train_dfs.py` is the full
runner that carries the KD recipes as named variants and logs the protocol banner.

## Protocol

| item | value |
| --- | --- |
| input | 640 x 640, mosaic on, `close_mosaic=10` |
| epochs | 200, `patience=200` |
| batch | 32 physical, no gradient accumulation |
| optimiser | SGD, `lr0=0.01`, `lrf=0.01`, momentum `0.9`, `weight_decay=5e-4` |
| warmup | 101 optimizer updates, no bias warmup |
| amp | on for the protocol runs, off for the controlled baseline presets |
| seed | 0 |
| teacher init | COCO `yolo26n.pt` |
| teacher schedule | trains on the dataset until `teacher_freeze_epoch`, 200 by default |

Warmup is defined in optimizer updates rather than epochs, so it survives a change of batch
size: `dfire_parity.protocol.warmup_epochs_for(batch)` returns
`101 / ceil(14122 / batch)`. At batch 32 that is `101/442` epochs; at batch 112 it is about
`101/127`.

`--nbs` is separate from `--batch`. Ultralytics scales the effective weight decay by
`batch * accumulate / nbs`, so `--nbs` changes regularisation without changing the gradient
batch. Leaving it unset keeps `nbs == batch`.

## Presets

`scripts/train_baselines.py --baseline <name>`:

| name | what it trains |
| --- | --- |
| `yolo26n` | YOLO26n teacher only |
| `dcn-solo` | YOLOF student, task loss only |
| `early` | dictionary KD with attention weighting |
| `early-dldx` | dictionary KD with the dLdx saliency weight |
| `early-dldx-offline` | the same recipe against a frozen, shared teacher |
| `infomax` | straight-through matching with the entropy term |
| `entropy-align` | align weighted by positive row entropy |
| `entropy-align-minmax`, `entropy-inverse`, `dldx-entropy-gate` | entropy candidates |
| `early-tune1`, `early-tune2` | copies for hyperparameter sweeps |

`scripts/train_dfs.py --baseline dcn-kd --kd-variant <name>` selects a dictionary recipe.
The variants used in the paper are:

| variant | dictionary term |
| --- | --- |
| `ssm` | full-resolution matching, channel-mean weight, entropy term |
| `ssm_sz` | `ssm` plus size-weighted align and confidence-weighted boxes |
| `ssm_sz_s6` | `ssm_sz` with the student tap moved to the matching stage |
| `ssm_soft_s6` | `ssm_sz_s6` with soft assignment instead of argmax |

The names beginning with `paperssm` are the older spelling of the same four arms and still
resolve.

## Loss weights

The reported configuration uses

```
task_loss         1.0     detection loss
dict_align_loss   0.1     SSM
dict_infomax_loss 0.1     entropy term on the assignment
feature_loss      0.08    RFM
align_loss        0.12    RKD
align_box         2.0     box sub-weight of RKD
align_cls         4.0     class sub-weight of RKD
```

`dict_attn_loss` and `dict_commit_loss` are zero: the attention-restriction and cosine
commitment terms belong to the older saliency recipes and are kept only for those arms.

## Outputs

Runs are written to `runs/detect/<project>/<name>/`:

```
args.yaml            resolved configuration
results.csv          per-epoch training and validation metrics
weights/best.pt      best validation checkpoint
weights/last.pt      last checkpoint
dict_match_stats.csv channel-level matching diagnostics (--kd-variant runs)
```

`args.yaml` is the record of what actually ran; keep it with any reported number.
