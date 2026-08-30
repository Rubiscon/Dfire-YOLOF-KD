# Related-work distillation protocol

This document is the acceptance contract for comparing the D-Fire
YOLO26n-to-YOLOF method with Structural KD, GID, and CanKD.

## Fixed model contract

- Teacher: `ultralytics/cfg/models/26/yolo26.yaml`, scale `n`, `nc=2`.
- Student: `models/yolo26n-DCN.yaml`.
- Both retain `end2end=True`, `reg_max=1`, the fork's one-to-many/one-to-one
  Detect branches, TAL assignment, decode, and task loss.
- The student retains the actual `DilatedDeformBlock` backend used by its
  checkpoint. A standard-convolution fallback is not an accepted substitute.
- Class order is `0=smoke`, `1=fire`.

## Result tracks

### Paper-method track

- Structural KD and GID use a pretrained, frozen YOLO26n teacher.
- Structural KD uses only ground-truth task loss plus the paper's structural
  feature objective.
- GID uses GISM plus feature, relation, and response knowledge. Any response
  mapping required by the unequal teacher/student grids is reported explicitly.
- CanKD uses only ground-truth task loss plus the official NLKD-IN feature
  objective (`MSE(IN(z_s), IN(z_t)) / 2` per paired map). It is hosted in
  `CanKD-main`, not `KD-main`. Detection configs set `loss_weight=10` per
  level (effective lambda 5 after the internal `/2`). YOLO26n contributes
  three Detect inputs, not RetinaNet's four FPN maps.
- The existing proposed-method result keeps its published online-teacher
  protocol. This track demonstrates paper-method reproduction but is not used
  to attribute differences solely to the KD loss.

### Controlled track

- Every method uses the same frozen D-Fire YOLO26n checkpoint.
- Every method uses the same student initialization, input transform, optimizer,
  schedule, random seeds, split, and evaluator.
- The common optimizer is Nesterov SGD (`lr=0.01`, momentum `0.9`), physical
  batch 112 on one GPU, and weight decay `5e-4` only on non-normalization
  weights. LR warmup spans 100 optimizer updates; momentum and bias do not use
  separate warmup schedules.
- The proposed method must be rerun in offline-teacher mode for this track.
- Only the distillation objective may differ.

## Required gates

1. The external-framework solo student must match the reference Ultralytics
   student on raw outputs and task loss for a fixed tensor.
2. Its D-Fire validation result must be within `0.005` mAP50 of the matching
   controlled Ultralytics run before a KD run is accepted.
3. The same teacher checkpoint must produce matching raw features and
   predictions in every wrapper.
4. Teacher parameters must remain frozen and absent from optimizer groups.
5. Distillation-only adapters must not remain in the inference model.
6. Report AP, AP50, AP75, APS, APM, APL, parameters, FLOPs, and latency using
   the same evaluator and hardware.
7. Use at least three fixed seeds on D-Fire and report mean and standard
   deviation.

## Required checkpoint inputs

The following paths are intentionally unresolved and must not be replaced by a
COCO-only teacher or random student:

- Frozen D-Fire YOLO26n teacher `best.pt`.
- Canonical YOLO26n initialization checkpoint used to initialize the YOLOF
  student's intersecting backbone weights.

Formal training and numerical parity stop at this gate until both are supplied.

## Result schema

Each accepted run records:

```text
method, track, framework, teacher_sha256, student_init_sha256, seed,
epochs, imgsz, batch, optimizer, augmentation_id,
AP, AP50, AP75, APS, APM, APL, params, flops, latency_ms
```

Do not merge paper-method and controlled-track rows into a single ranking.
