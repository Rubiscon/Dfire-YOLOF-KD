# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import math
import random
import time
from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.modules.yolof import DeconvNet, DictionaryModule, DilatedResBlock
from ultralytics.nn.modules.dilated_dcn import DilatedDeformBlock
from ultralytics.nn.modules.dcnyolof import DilatedDCNBlock
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

# Student blocks whose residual output is cached as `last_feature` for feature distillation.
DISTILL_BLOCKS = (DilatedResBlock, DilatedDeformBlock, DilatedDCNBlock)


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        weights = (1.0 / class_counts) ** self.args.cls_pw
        weights = weights / weights.mean()
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        n = len(train_dataset)
        del train_dataset
        return super().auto_batch(max_num_obj, dataset_size=n)


class YOLOFDistillationModel(DetectionModel):
    """DetectionModel with offline FPN teacher -> YOLOF student distillation."""

    def __init__(
        self,
        cfg: str | dict | None = None,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.teacher = None
        self.feature_projectors = nn.ModuleList()
        # Early-stage backbone distillation (fig. 1/2): student n10 (layer 10) learns local
        # structure from teacher early tap x6 (layer 6) via dictionary matching + weighted align.
        self.dictionary_modules = nn.ModuleList()
        self._dict_teacher_layers: List[int] = []
        self._dict_student_layer: int | None = None
        self._student_tap: torch.Tensor | None = None
        self._dict_warned = False
        self.current_epoch = 0
        self.distill_conf_thres = 0.25
        self.distill_iou_thres = 0.5
        # Per-scale feature alignment: f1<->N1, f2<->N2, f3<->N3 (matched by receptive field / depth).
        self.distill_temperature = 3.0  # KD temperature for response-level KL (overridable via args)
        self._align_assigner: TaskAlignedAssigner | None = None
        self._align_assigner_topk: int | None = None
        self._align_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.last_align_stats: Dict[str, float] = {}
        self._teacher_frozen = False
        # Detached spatial saliency maps from a prior teacher pass (avoids retain_graph peak).
        self._cached_saliency: Dict[int, torch.Tensor] = {}
        # Running EMA of Grad-CAM saliency (layer → (1,1,H,W)); used after teacher freeze.
        self._saliency_ema: Dict[int, torch.Tensor] = {}

    def _teacher_joint_training(self) -> bool:
        """True while the teacher receives GT task loss (online joint phase).

        ``teacher_freeze_epoch`` is 1-indexed to match results.csv epoch numbers.
        Epochs 1..N train the teacher; from epoch N+1 the teacher is frozen for distillation only.
        """
        if self.teacher is None:
            return False
        if not bool(getattr(self.args, "online_distill", True)):
            return False
        freeze_ep = int(getattr(self.args, "teacher_freeze_epoch", 0) or 0)
        if freeze_ep <= 0:
            return True
        return (self.current_epoch + 1) <= freeze_ep

    @staticmethod
    def _ema_teacher_module(trainer) -> DetectionModel | None:
        """Return the EMA-smoothed teacher submodule from the KD trainer, if available."""
        if trainer is None or getattr(trainer, "ema", None) is None:
            return None
        ema_kd = unwrap_model(trainer.ema.ema)
        return getattr(ema_kd, "teacher", None)

    def _sync_live_teacher_from_ema(self, trainer) -> bool:
        """Copy ``ema.ema.teacher`` weights into the live ``self.teacher`` submodule."""
        teacher_ema = self._ema_teacher_module(trainer)
        if teacher_ema is None:
            return False
        self.teacher.load_state_dict(unwrap_model(teacher_ema).state_dict())
        return True

    def _val_teacher_submodule(self, trainer, teacher_model) -> Tuple[float, float] | None:
        """One-off teacher val via the in-training validator path (no fuse, AMP FP16 when enabled)."""
        if trainer is None or teacher_model is None or RANK not in {-1, 0}:
            return None
        plots = trainer.validator.args.plots
        trainer.validator.args.plots = False
        teacher_copy = None
        try:
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()
            teacher_copy = deepcopy(unwrap_model(teacher_model)).to(trainer.device)
            stats = trainer.validator(trainer=trainer, model=teacher_copy)
            if not stats:
                return None
            return float(stats.get("metrics/mAP50(B)", 0.0)), float(stats.get("metrics/mAP50-95(B)", 0.0))
        except Exception as exc:
            LOGGER.warning(f"{colorstr('KD:')} Teacher freeze val skipped: {exc}")
            return None
        finally:
            del teacher_copy
            trainer.validator.args.plots = plots
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()

    def _apply_teacher_freeze_if_needed(self, trainer=None) -> None:
        """Freeze teacher weights once the joint-training phase ends (scheme C).

        When ``teacher_freeze_use_ema`` is True (default), snapshot the EMA-smoothed teacher into
        the live teacher submodule before freezing so distillation targets match the stronger EMA
        weights instead of the last live optimizer step.
        """
        if self.teacher is None or self._teacher_frozen or self._teacher_joint_training():
            return

        freeze_ep = getattr(self.args, "teacher_freeze_epoch", 0)
        use_ema = bool(getattr(self.args, "teacher_freeze_use_ema", True))
        epoch = self.current_epoch + 1

        live_map = self._val_teacher_submodule(trainer, self.teacher) if trainer is not None else None
        ema_teacher = self._ema_teacher_module(trainer)
        ema_map = self._val_teacher_submodule(trainer, ema_teacher) if ema_teacher is not None else None

        copied_ema = False
        if use_ema:
            if self._sync_live_teacher_from_ema(trainer):
                copied_ema = True
            elif RANK in {-1, 0}:
                LOGGER.warning(
                    f"{colorstr('KD:')} teacher_freeze_use_ema=True but EMA teacher unavailable at epoch "
                    f"{epoch}; freezing live teacher weights"
                )

        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self._teacher_frozen = True

        if RANK in {-1, 0}:
            if live_map and ema_map:
                LOGGER.info(
                    f"{colorstr('KD:')} Teacher freeze snapshot epoch {epoch}/{freeze_ep} "
                    f"live mAP50={live_map[0]:.4f} mAP50-95={live_map[1]:.4f} | "
                    f"EMA mAP50={ema_map[0]:.4f} mAP50-95={ema_map[1]:.4f} | "
                    f"teacher_freeze_use_ema={use_ema} copied_ema={copied_ema}"
                )
            elif live_map:
                LOGGER.info(
                    f"{colorstr('KD:')} Teacher freeze snapshot epoch {epoch}/{freeze_ep} "
                    f"live mAP50={live_map[0]:.4f} mAP50-95={live_map[1]:.4f} | "
                    f"teacher_freeze_use_ema={use_ema} copied_ema={copied_ema}"
                )
            LOGGER.info(
                f"{colorstr('KD:')} Teacher frozen at epoch {epoch} "
                f"(teacher_freeze_epoch={freeze_ep}); feature/align distillation continues, teacher_task_loss=0"
            )

    @staticmethod
    def _forward_with_taps(model: DetectionModel, x: torch.Tensor, tap_layers: set):
        """Routed forward through ``model`` that also returns intermediate outputs.

        Mirrors ``BaseModel._predict_once`` exactly but captures the output of every layer
        index in ``tap_layers`` (backbone distillation: n10, x6, …).
        """
        taps: Dict[int, torch.Tensor] = {}
        y = []
        for m in model.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in model.save else None)
            if m.i in tap_layers:
                taps[m.i] = x
        return x, taps

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Student forward that additionally caches the backbone tap feature (n10) for dictionary distillation."""
        # getattr: the parent __init__ runs a stride-trace forward before our attributes exist.
        tap_layer = getattr(self, "_dict_student_layer", None)
        if profile or visualize or (embed is not None) or tap_layer is None:
            return super()._predict_once(x, profile, visualize, embed)
        out, taps = self._forward_with_taps(self, x, {tap_layer})
        self._student_tap = taps.get(tap_layer)
        return out

    def _teacher_forward_raw(self, img: torch.Tensor, tap_layers: set | None = None):
        """Teacher forward for distillation targets (frozen or offline: no grad)."""
        with torch.no_grad():
            self.teacher.eval()
            teacher_out, taps = self._forward_with_taps(self.teacher, img, set(tap_layers or ()))
            return self._parse_raw_preds(teacher_out), taps

    def _training_preds(self, img: torch.Tensor, preds: Any = None) -> Any:
        """Run routed forward (not nn.Sequential) to support YOLOF Detect heads."""
        if preds is None:
            return self._predict_once(img)
        try:
            self._parse_raw_preds(preds)
            return preds
        except TypeError:
            return self._predict_once(img)

    def _parse_raw_preds(self, preds: Any) -> Dict[str, torch.Tensor]:
        if isinstance(preds, (tuple, list)) and len(preds) == 2:
            preds = preds[1]
        if not isinstance(preds, dict):
            raise TypeError(f"Expected dict-like prediction raw outputs, got {type(preds)}")
        return preds

    def _get_pred_feats(self, preds: Any) -> List[torch.Tensor]:
        preds = self._parse_raw_preds(preds)
        if "one2many" in preds:
            return preds["one2many"]["feats"]
        return preds["feats"]

    def _get_pred_scores(self, preds: Any) -> torch.Tensor:
        preds = self._parse_raw_preds(preds)
        if "one2one" in preds:
            return preds["one2one"]["scores"]
        return preds["scores"]

    def __deepcopy__(self, memo):
        """Make the model safe to deepcopy (ModelEMA init and checkpoint saving both deepcopy it).

        Distillation blocks cache their residual output in ``last_feature`` for feature KD; after a
        forward (e.g. the AMP check), that cache is a non-leaf graph tensor which torch refuses to
        deepcopy. Null all such caches before copying, then restore them so training is unaffected.
        """
        saved = [
            (m, m.last_feature)
            for m in self.modules()
            if getattr(m, "last_feature", None) is not None
        ]
        for m, _ in saved:
            m.last_feature = None
        saved_tap = self.__dict__.get("_student_tap")
        self._student_tap = None
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        try:
            for k, v in self.__dict__.items():
                new.__dict__[k] = deepcopy(v, memo)
        finally:
            for m, v in saved:
                m.last_feature = v
            self._student_tap = saved_tap
        return new

    def _student_features(self) -> List[torch.Tensor]:
        features = [
            m.last_feature
            for m in self.model.modules()
            if isinstance(m, DISTILL_BLOCKS) and getattr(m, "last_feature", None) is not None
        ]
        if not features:
            raise RuntimeError(
                "Student model does not contain YOLOF distillation blocks with cached features. "
                "Use a YOLOF/DCN yaml (e.g. yolo26n-yolof.yaml or yolo26n-DCN.yaml)."
            )
        return features

    def _build_feature_projectors(
        self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]
    ):
        """One projector per (student block, FPN level) pair: f1<->N1, f2<->N2, f3<->N3.

        Student blocks and teacher FPN feats are both ordered by ascending depth, so the
        shallow/small-dilation block (local features) aligns to the high-resolution shallow
        FPN level, and the deep/large-dilation block aligns to the low-resolution deep level.
        """
        n = min(len(student_features), len(teacher_features))
        if len(student_features) != len(teacher_features):
            LOGGER.warning(
                f"Student distill blocks ({len(student_features)}) != teacher FPN levels "
                f"({len(teacher_features)}); pairing first {n} by ascending depth."
            )
        self.feature_projectors = nn.ModuleList(
            DeconvNet(
                student_features[i].shape[1],
                teacher_features[i].shape[1],
                student_features[i].shape[-1],
                teacher_features[i].shape[-1],
            )
            for i in range(n)
        )
        pairs = [
            f"f{i + 1}{tuple(student_features[i].shape[1:])}->N{i + 1}{tuple(teacher_features[i].shape[1:])}"
            for i in range(n)
        ]
        LOGGER.info(f"Built {n} per-scale feature projectors: {pairs}")

    def _dict_gains(self) -> Tuple[float, float, float]:
        """(align, attention-restriction, commit) gains for backbone dictionary distillation.

        ``dict_attn_start_epoch`` is 1-indexed (same as ``teacher_freeze_epoch``): attention
        restriction is delayed so early epochs focus on saliency-weighted feature alignment.
        """
        args = getattr(self, "args", None)
        if args is None:
            return 0.0, 0.0, 0.0
        beta_d = float(getattr(args, "dict_align_loss", 0.0) or 0.0)
        beta_a = float(getattr(args, "dict_attn_loss", 0.0) or 0.0)
        beta_c = float(getattr(args, "dict_commit_loss", 0.0) or 0.0)
        attn_start = int(getattr(args, "dict_attn_start_epoch", 0) or 0)
        if attn_start > 0 and (self.current_epoch + 1) < attn_start:
            beta_a = 0.0
        return beta_d, beta_a, beta_c

    def build_distillation_modules(self, imgsz: int | None = None):
        """Trace feature shapes and create channel/spatial projectors + dictionary modules."""
        if self.teacher is None:
            LOGGER.warning("Teacher model not set; skipping distillation module initialization")
            return

        imgsz = imgsz or getattr(self.args, "imgsz", 640)
        dict_on = any(
            float(getattr(self.args, k, 0.0) or 0.0) > 0
            for k in ("dict_align_loss", "dict_attn_loss", "dict_commit_loss")
        )
        if dict_on:
            t_layers = getattr(self.args, "dict_teacher_layers", None) or (6,)
            self._dict_teacher_layers = [int(x) for x in t_layers]
            self._dict_student_layer = int(getattr(self.args, "dict_student_layer", 10) or 10)

        was_training = self.training
        device = next(self.parameters()).device
        img = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)

        with torch.no_grad():
            self.train()
            _ = self._predict_once(img)
            student_features = self._student_features()

            self.teacher.eval()
            teacher_out, teacher_taps = self._forward_with_taps(self.teacher, img, set(self._dict_teacher_layers))
            teacher_raw = self._parse_raw_preds(teacher_out)
            teacher_feats = self._get_pred_feats(teacher_raw)
            self._build_feature_projectors(student_features, teacher_feats)

            if dict_on:
                s_tap = self._student_tap
                if s_tap is None:
                    raise RuntimeError(
                        f"dict_student_layer={self._dict_student_layer} produced no feature; "
                        f"check the student yaml layer indices."
                    )
                match = str(getattr(self.args, "dict_match", "soft")).lower()
                match_temp = float(getattr(self.args, "dict_match_temp", 0.07) or 0.07)
                modules, msgs = [], []
                for li in self._dict_teacher_layers:
                    t_feat = teacher_taps.get(li)
                    if t_feat is None:
                        raise RuntimeError(f"Teacher layer {li} (dict_teacher_layers) not found in teacher model.")
                    # Proposal: avg-pool teacher tokens to ~1/16 of feature map side length.
                    grid = max(int(t_feat.shape[-1]) // 16, 1)
                    grid = max(grid, 2)  # min 2x2 tokens for stable channel correlation
                    mod = DictionaryModule(
                        t_feat.shape[1],
                        s_tap.shape[1],
                        int(t_feat.shape[-1]),
                        int(s_tap.shape[-1]),
                        grid,
                        match=match,
                        temperature=match_temp,
                    )
                    if match == "hard":
                        # Hard argmax blocks encoder grads; freeze them as fixed random projections.
                        mod.freeze_encoders()
                    modules.append(mod)
                    msgs.append(
                        f"x{li}{tuple(t_feat.shape[1:])} <- n{self._dict_student_layer}{tuple(s_tap.shape[1:])} "
                        f"(token grid {grid}x{grid}, match={match})"
                    )
                self.dictionary_modules = nn.ModuleList(modules).to(device)
                LOGGER.info(f"Built {len(modules)} dictionary modules (backbone distillation): {msgs}")

        self._student_tap = None
        if not was_training:
            self.eval()

    def _decode_y(self, raw_preds: Any, model: DetectionModel | None = None) -> torch.Tensor:
        """Decode head outputs to NMS-ready boxes + scores using the given model's Detect head."""
        det_model = model if model is not None else self
        raw = self._parse_raw_preds(raw_preds)
        head = det_model.model[-1]
        branch = raw["one2one"] if getattr(head, "end2end", False) and "one2one" in raw else raw
        y = head._inference(branch)
        if getattr(head, "end2end", False):
            y = head.postprocess(y.permute(0, 2, 1))
        return y

    def _decode_y_bcn(self, raw_preds: Any, model: DetectionModel | None = None) -> torch.Tensor:
        """Decode to (B, 4+nc, anchors) for NMS with anchor indices (required for align_loss)."""
        det_model = model if model is not None else self
        raw = self._parse_raw_preds(raw_preds)
        head = det_model.model[-1]
        branch = raw["one2one"] if getattr(head, "end2end", False) and "one2one" in raw else raw
        return head._inference(branch)

    @staticmethod
    def _channel_standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Per-channel zero-mean/unit-variance over spatial dims (BCHW)."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return (x - mean) / (std + eps)

    def _feature_loss(self, student_raw: Dict[str, torch.Tensor], teacher_raw: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.feature_projectors:
            raise RuntimeError("Feature projectors are not initialized. Call build_distillation_modules() first.")

        student_features = self._student_features()
        teacher_features = self._get_pred_feats(teacher_raw)
        n = min(len(student_features), len(teacher_features), len(self.feature_projectors))
        # (C) Normalize features before MSE. Teacher activations can be orders of magnitude larger
        # than the student's, so raw MSE just chases their scale instead of the spatial pattern.
        #   "l2"      -> per-pixel unit norm over channels (align activation direction)
        #   "channel" -> per-channel standardization over space (zero-mean/unit-var)
        #   "none"    -> raw MSE (legacy behavior)
        norm = str(getattr(self.args, "feature_norm", "channel")).lower()
        loss = torch.tensor(0.0, device=teacher_features[0].device)
        for i in range(n):
            target = teacher_features[i].detach()
            pred = self.feature_projectors[i](student_features[i])
            # Val rect / multi-scale can change feature map size; always match teacher spatially.
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
            if norm == "l2":
                pred = F.normalize(pred, dim=1)
                target = F.normalize(target, dim=1)
            elif norm == "channel":
                pred = self._channel_standardize(pred)
                target = self._channel_standardize(target)
            loss = loss + F.mse_loss(pred, target)
        return loss / max(n, 1)

    def _dict_active(self) -> bool:
        """Backbone dictionary distillation enabled for the current epoch."""
        if not len(self.dictionary_modules) or self.teacher is None:
            return False
        gains = self._dict_gains()
        if all(g <= 0 for g in gains):
            return False
        start = int(getattr(self.args, "dict_start_epoch", 0) or 0)
        return self.current_epoch >= start

    @staticmethod
    def _spatial_attention(feat: torch.Tensor) -> torch.Tensor:
        """Spatial attention map A = mean_c(F^2), shape (B, H, W) per proposal."""
        return feat.float().pow(2).mean(dim=1)

    def _collect_dict_teacher_feats(self, teacher_taps: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        """Teacher tap tensors that participate in dictionary saliency / align."""
        feats: List[torch.Tensor] = []
        for li in self._dict_teacher_layers:
            f = teacher_taps.get(li)
            if f is not None and f.requires_grad:
                feats.append(f)
        return feats

    def _compute_teacher_saliency(
        self, teacher_taps: Dict[int, torch.Tensor], teacher_task_loss: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Grad-CAM saliency for weighted align (proposal |∂L/∂A| surrogate).

        Proposal writes |∂L/∂A| with A=mean_c(F²), but A is not an ancestor of L_task, so
        ``autograd.grad(L, A)`` is always ``None``. Grad-CAM is the standard surrogate:

            α_c = GAP(∂L_task/∂F_c),   S = ReLU(Σ_c α_c F_c)

        This weights object-relevant spatial loci by both gradient importance and activation
        strength — closer to the proposal's "richer dark knowledge" intent than mean(|∂L/∂F|).

        ``retain_graph=False``: call on a throwaway teacher forward (no dual-graph OOM).
        """
        if teacher_task_loss is None or not teacher_task_loss.requires_grad:
            return {}
        feats = self._collect_dict_teacher_feats(teacher_taps)
        if not feats:
            return {}
        grads = torch.autograd.grad(
            teacher_task_loss.sum(), feats, retain_graph=False, allow_unused=True
        )
        out: Dict[int, torch.Tensor] = {}
        for f, g in zip(feats, grads):
            if g is None:
                continue
            alpha = g.float().mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = F.relu((alpha * f.float()).sum(dim=1, keepdim=True))  # (B, 1, H, W)
            if float(cam.detach().amax()) <= 0.0:
                cam = g.detach().abs().mean(dim=1, keepdim=True)
            out[id(f)] = cam.detach()
        return out

    def _update_saliency_ema(self, layer_maps: Dict[int, torch.Tensor]) -> None:
        """Update per-layer EMA of batch-mean Grad-CAM maps for post-freeze weighting."""
        mom = float(getattr(self.args, "dict_saliency_ema", 0.9) or 0.0)
        if mom <= 0 or not layer_maps:
            return
        for li, w in layer_maps.items():
            cur = w.detach().float().mean(dim=0, keepdim=True)  # (1, 1, H, W)
            prev = self._saliency_ema.get(li)
            if prev is None or prev.shape != cur.shape:
                self._saliency_ema[li] = cur
            else:
                self._saliency_ema[li] = mom * prev + (1.0 - mom) * cur

    def _dict_norm_mode(self) -> str:
        """Feature normalization for dictionary align (default ``none``).

        Neck feature KD benefits from ``feature_norm=channel``, but applying the same
        channel standardization to dictionary features washes out magnitude cues that
        saliency weighting is meant to emphasize. Override with ``dict_feature_norm``.
        """
        raw = getattr(self.args, "dict_feature_norm", None)
        if raw is None or str(raw).lower() in {"", "default"}:
            return "none"
        return str(raw).lower()

    def _dictionary_losses(
        self, teacher_taps: Dict[int, torch.Tensor], saliency: Dict[int, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backbone distillation losses (CrisReport): (weighted align, attention restriction, commit).

        Weighted align: saliency-weighted MSE between projected student tap and dictionary-
        reorganized teacher feature. Weight priority:
          1) live Grad-CAM from the joint-phase throwaway pass
          2) EMA of Grad-CAM (after freeze / when live map missing)
          3) uniform (NOT raw attention A — A is already supervised by attention restriction,
             so using A as the align weight double-counts and over-regularizes)

        Attention restriction: AT-style MSE on L2-normalized spatial attention maps.
        Uses mean reduction (not sum over HW) so its scale matches weighted MSE; the old
        ``sum(dim=1)`` form was ~H·W× larger and dominated the KD budget at batch 112.

        Commit: soft-matching encoder loss (queries → matched teacher keys); 0 for hard match.
        """
        device = next(self.parameters()).device
        zero = torch.tensor(0.0, device=device)
        s_feat = self._student_tap
        if s_feat is None:
            if not self._dict_warned:
                LOGGER.warning("KD: student backbone tap unavailable; skipping dictionary distillation")
                self._dict_warned = True
            return zero, zero, zero

        mode = str(getattr(self.args, "dict_weight", "saliency")).lower()
        norm = self._dict_norm_mode()
        if saliency is None:
            saliency = self._cached_saliency if mode == "saliency" else {}

        d_align = zero.clone()
        d_attn = zero.clone()
        d_commit = zero.clone()
        n = 0
        for j, li in enumerate(self._dict_teacher_layers):
            t_feat = teacher_taps.get(li)
            if t_feat is None or j >= len(self.dictionary_modules):
                continue
            # Detach teacher activations so dict KD does not update the teacher backbone.
            # Reorganized teacher is always stopgrad'd as the KD target; soft encoders learn
            # via the commitment term instead of through the align residual.
            s_proj, t_reorg, commit = self.dictionary_modules[j](t_feat.detach(), s_feat)
            target = t_reorg.detach()
            pred = s_proj
            d_commit = d_commit + commit

            if norm == "l2":
                pred = F.normalize(pred, dim=1)
                target = F.normalize(target, dim=1)
            elif norm == "channel":
                pred = self._channel_standardize(pred)
                target = self._channel_standardize(target)

            weight = None
            if mode == "saliency":
                weight = saliency.get(li) if saliency else None
                if weight is None and saliency:
                    weight = saliency.get(id(t_feat))
                if weight is None:
                    ema = self._saliency_ema.get(li)
                    if ema is not None:
                        weight = ema.expand(pred.shape[0], -1, -1, -1)
            elif mode == "attention":
                weight = self._spatial_attention(t_feat.detach()).unsqueeze(1)
            # mode == "none" / unknown → uniform MSE

            if weight is not None:
                weight = weight.float()
                if weight.shape[-2:] != pred.shape[-2:]:
                    weight = F.interpolate(weight, size=pred.shape[-2:], mode="bilinear", align_corners=False)
                weight = (weight / weight.mean(dim=(2, 3), keepdim=True).clamp_min(1e-12)).to(pred.dtype).detach()
                d_align = d_align + (weight * (pred - target) ** 2).mean()
            else:
                d_align = d_align + F.mse_loss(pred, target)

            att_s = F.normalize(self._spatial_attention(s_proj).flatten(1), dim=1)
            att_t = F.normalize(self._spatial_attention(target).flatten(1), dim=1)
            # Mean over the flattened spatial dim keeps scale comparable to feature MSE.
            d_attn = d_attn + F.mse_loss(att_s, att_t)
            n += 1

        n = max(n, 1)
        return d_align / n, d_attn / n, d_commit / n

    def _ensure_align_assigner(self):
        """Lazy-init TAL assigner for align (defaults match task-loss TAL topk=10)."""
        head = self.model[-1]
        device = next(self.parameters()).device
        topk = int(getattr(self.args, "align_tal_topk", 10))
        if self._align_assigner is None or self._align_assigner_topk != topk:
            self._align_assigner = TaskAlignedAssigner(
                topk=topk,
                num_classes=self.nc,
                alpha=0.5,
                beta=6.0,
                stride=head.stride.tolist(),
                topk2=1,
            )
            self._align_assigner_topk = topk
        self._align_bce = self._align_bce.to(device)

    def _get_align_branch(self, student_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Select student head branch for response distillation (default one2many to match early task loss)."""
        branch = str(getattr(self.args, "align_branch", "one2many")).lower()
        if branch == "one2one" and "one2one" in student_raw:
            return student_raw["one2one"]
        if "one2many" in student_raw:
            return student_raw["one2many"]
        return student_raw

    @staticmethod
    def _pack_teacher_pseudo_gt(
        teacher_nms: List[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack teacher NMS detections (xyxy, conf, cls) into TAL ground-truth tensors."""
        counts = [int(d.shape[0]) if d is not None and d.numel() else 0 for d in teacher_nms]
        n_max = max(counts) if counts else 0
        if n_max == 0:
            empty = torch.zeros(batch_size, 0, device=device, dtype=dtype)
            return (
                torch.zeros(batch_size, 0, 1, device=device),
                torch.zeros(batch_size, 0, 4, device=device, dtype=dtype),
                torch.zeros(batch_size, 0, 1, device=device, dtype=torch.bool),
                empty,
            )

        gt_labels = torch.zeros(batch_size, n_max, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device, dtype=dtype)
        mask_gt = torch.zeros(batch_size, n_max, 1, device=device, dtype=torch.bool)
        pseudo_conf = torch.zeros(batch_size, n_max, device=device, dtype=dtype)
        for b in range(batch_size):
            det = teacher_nms[b] if b < len(teacher_nms) else teacher_nms[0][:0]
            if det is None or det.numel() == 0:
                continue
            n = det.shape[0]
            gt_bboxes[b, :n] = det[:n, :4]
            gt_labels[b, :n, 0] = det[:n, 5].long()
            pseudo_conf[b, :n] = det[:n, 4]
            mask_gt[b, :n, 0] = True
        return gt_labels, gt_bboxes, mask_gt, pseudo_conf

    def _decode_align_predictions(
        self, branch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode student branch outputs to TAL-ready tensors (mirrors v8DetectionLoss)."""
        head = self.model[-1]
        pred_distri = branch["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = branch["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(branch["feats"], head.stride, 0.5)
        reg_max = head.reg_max
        if reg_max > 1:
            b, a, c = pred_distri.shape
            proj = torch.arange(reg_max, device=pred_distri.device, dtype=pred_distri.dtype)
            pred_dist = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(proj)
        else:
            pred_dist = pred_distri
        pred_bboxes = dist2bbox(pred_dist, anchor_points, xywh=False)
        return pred_distri, pred_scores, pred_bboxes, anchor_points, stride_tensor

    def _alignment_loss(self, student_raw: Dict[str, torch.Tensor], teacher_raw: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Lalign = Lyolo26(Yf, Y_FPNNMS) via teacher-only pseudo labels + TAL assignment (P0).

        Proposal intent: teacher NMS boxes supervise the student. Old code required the student to
        already detect overlapping boxes before any align gradient could flow; this version assigns
        every teacher pseudo-box to student anchors with TaskAlignedAssigner (same family as GT loss).
        """
        if self.teacher is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        self._ensure_align_assigner()
        device = next(self.parameters()).device
        head = self.model[-1]

        branch = self._get_align_branch(student_raw)
        pred_distri, pred_scores, pred_bboxes, anchor_points, stride_tensor = self._decode_align_predictions(branch)

        # Use _decode_y (not _decode_y_bcn): end2end teachers output xyxy pixels via postprocess.
        # Feeding BCN xyxy through standard NMS applies xywh2xyxy again and corrupts pseudo boxes.
        teacher_head = self.teacher.model[-1]
        teacher_end2end = getattr(self.teacher, "end2end", False) or getattr(teacher_head, "end2end", False)
        teacher_y = self._decode_y(teacher_raw, self.teacher)
        conf_thres = float(getattr(self.args, "distill_conf_thres", self.distill_conf_thres))
        iou_thres = float(getattr(self.args, "distill_iou_thres", self.distill_iou_thres))
        max_teacher_boxes = int(getattr(self.args, "distill_max_boxes", getattr(self.args, "max_det", 300)))
        teacher_nms = non_max_suppression(
            teacher_y,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_teacher_boxes,
            nc=teacher_head.nc,
            end2end=teacher_end2end,
        )

        batch_size = pred_scores.shape[0]
        # TAL uses indexed writes into overlap buffers; keep assign path in float32 under AMP.
        assign_dtype = torch.float32
        gt_labels, gt_bboxes, mask_gt, pseudo_conf = self._pack_teacher_pseudo_gt(
            teacher_nms, batch_size, device, assign_dtype
        )
        teacher_boxes_t = mask_gt.sum()
        if teacher_boxes_t == 0:
            self.last_align_stats = {"teacher_boxes": 0.0, "fg_anchors": 0.0, "assign_ratio": 0.0}
            return torch.tensor(0.0, device=device)

        anc_points_px = (anchor_points * stride_tensor).to(assign_dtype)
        with torch.no_grad():
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self._align_assigner(
                pred_scores.detach().float().sigmoid(),
                (pred_bboxes.detach().float() * stride_tensor.to(assign_dtype)),
                anc_points_px,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        fg_count_t = fg_mask.sum()
        if fg_count_t == 0:
            self.last_align_stats = {
                "teacher_boxes": float(teacher_boxes_t),
                "fg_anchors": 0.0,
                "assign_ratio": 0.0,
            }
            return torch.tensor(0.0, device=device)

        T = float(getattr(self.args, "distill_temperature", self.distill_temperature))
        box_gain = float(getattr(self.args, "align_box", 2.0))
        cls_gain = float(getattr(self.args, "align_cls", 4.0))
        cls_mode = str(getattr(self.args, "align_cls_mode", "kl")).lower()

        strides = stride_tensor.squeeze(-1)
        bb, aa = fg_mask.nonzero(as_tuple=True)
        fg_used = bb.numel()
        self.last_align_stats = {
            "teacher_boxes": float(teacher_boxes_t),
            "fg_anchors": float(fg_count_t),
            "fg_used": float(fg_used),
            "assign_ratio": float(fg_count_t / teacher_boxes_t.clamp(min=1)),
        }

        strides_f = strides.to(assign_dtype)
        s_box_px = pred_bboxes[bb, aa].float() * strides_f[aa].unsqueeze(-1)
        t_box_px = target_bboxes[bb, aa].float()
        box_loss = (1.0 - bbox_iou(s_box_px, t_box_px, xywh=False, CIoU=True)).squeeze(-1).mean()

        if cls_mode == "bce_tal":
            target_scores_sum = max(target_scores.sum(), 1)
            cls_loss = (
                self._align_bce(pred_scores.float(), target_scores.to(assign_dtype))[bb, aa].sum()
                / target_scores_sum
            )
        else:
            # KL with soft pseudo labels that retain teacher detection confidence.
            gidx = target_gt_idx[bb, aa].long()
            tcls = gt_labels[bb, gidx, 0].long()
            tconf = pseudo_conf[bb, gidx].clamp(1e-6, 1 - 1e-6)
            teacher_prob = torch.zeros(fg_used, self.nc, device=device, dtype=assign_dtype)
            teacher_prob.scatter_(1, tcls.unsqueeze(1), tconf.unsqueeze(1))
            other_mask = torch.ones_like(teacher_prob).scatter_(1, tcls.unsqueeze(1), 0.0)
            other_share = (1.0 - tconf).unsqueeze(1) / other_mask.sum(dim=1, keepdim=True).clamp(min=1)
            teacher_prob = teacher_prob + other_mask * other_share
            student_logits = pred_scores[bb, aa].float()
            kl_per = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                teacher_prob.detach(),
                reduction="batchmean",
            )
            cls_loss = (T * T) * kl_per

        return cls_gain * cls_loss + box_gain * box_loss.to(pred_scores.dtype)

    def _criterion_device(self, crit) -> torch.device | None:
        """Return the device of a detection criterion (v8DetectionLoss or E2ELoss wrapper)."""
        if crit is None:
            return None
        inner = getattr(crit, "one2many", crit)
        return getattr(inner, "device", None)

    def _ensure_teacher_criterion(self):
        """Init or refresh teacher loss criterion on the current device.

        get_model() runs before model.to(device), so an eager init_criterion() would pin
        stride tensors and assigner state on CPU while preds live on CUDA.
        """
        if self.teacher is None:
            return
        t_dev = next(self.teacher.parameters()).device
        crit = getattr(self.teacher, "criterion", None)
        if crit is None or self._criterion_device(crit) != t_dev:
            self.teacher.criterion = self.teacher.init_criterion()

    def _set_teacher_criterion_epoch(self, epoch: int):
        """Synchronize the teacher's end-to-end loss schedule with the trainer epoch.

        Ultralytics updates only the top-level model criterion each epoch. The teacher has its
        own criterion in online distillation, so without this sync it would keep the initial
        one2many/one2one loss ratio after resume (or even during uninterrupted training).
        """
        if self.teacher is None or not bool(getattr(self.args, "online_distill", True)):
            return
        if self._teacher_frozen or not self._teacher_joint_training():
            return
        self._ensure_teacher_criterion()
        crit = getattr(self.teacher, "criterion", None)
        if not hasattr(crit, "update") or not hasattr(crit, "updates"):
            return
        if epoch <= 0:
            crit.updates = 0
            return
        # Match the state that Ultralytics' top-level criterion has at the start of this epoch:
        # after epoch e-1 completes, update() has been called e times.
        crit.updates = epoch - 1
        crit.update()

    def loss(self, batch, preds=None):
        """Collaborative distillation loss.

        Online mode (teacher trainable): total = student_task + teacher_task + feature_MSE + response_KL.
        Online + teacher_freeze_epoch: joint until epoch N, then teacher frozen (teacher_task=0, distill continues).
        Offline mode (teacher frozen): total = student_task + feature_MSE + response_KL.
        Feature/align targets are detached; the teacher only learns from GT during the joint phase.
        """
        if self.teacher is None:
            return super().loss(batch, preds)

        # Validation uses rect batches with varying sizes; only student task loss is needed for val metrics.
        if not self.training:
            student_task_loss, student_loss_items = super().loss(batch, preds)
            pad = torch.zeros(7, device=student_loss_items.device, dtype=student_loss_items.dtype)
            return student_task_loss, torch.cat([student_loss_items, pad])

        preds = self._training_preds(batch["img"], preds)
        student_raw = self._parse_raw_preds(preds)
        student_task_loss, student_loss_items = super().loss(batch, preds)

        online = bool(getattr(self.args, "online_distill", True))
        teacher_joint = online and self._teacher_joint_training()
        dict_on = self._dict_active()
        tap_layers = set(self._dict_teacher_layers) if dict_on else set()
        teacher_task_loss = torch.tensor(0.0, device=batch["img"].device)
        teacher_loss_items = torch.zeros(3, device=batch["img"].device)
        teacher_taps: Dict[int, torch.Tensor] = {}
        self._cached_saliency = {}
        want_saliency = (
            dict_on
            and teacher_joint
            and str(getattr(self.args, "dict_weight", "saliency")).lower() == "saliency"
        )
        if teacher_joint:
            self._ensure_teacher_criterion()
            self.teacher.train()
            if want_saliency:
                # Pass 1: throwaway teacher forward → |∂L/∂F| saliency, free graph (no retain_graph).
                # Holding retain_graph across student+teacher at large batch was a prime OOM/thrash cause.
                # Freeze BN running-stat updates so this pass does not double-count toward EMA stats.
                bn_backup: List[Tuple[nn.Module, float | None]] = []
                for m in self.teacher.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                        bn_backup.append((m, m.momentum))
                        m.momentum = 0.0
                try:
                    with torch.enable_grad():
                        sal_preds, sal_taps = self._forward_with_taps(self.teacher, batch["img"], tap_layers)
                        sal_loss, _ = self.teacher.loss(batch, sal_preds)
                        sal_maps = self._compute_teacher_saliency(sal_taps, sal_loss)
                        self._cached_saliency = {
                            li: sal_maps[id(sal_taps[li])]
                            for li in self._dict_teacher_layers
                            if li in sal_taps and id(sal_taps[li]) in sal_maps
                        }
                        self._update_saliency_ema(self._cached_saliency)
                        del sal_preds, sal_taps, sal_loss, sal_maps
                finally:
                    for m, mom in bn_backup:
                        m.momentum = mom
            # Pass 2 (or only pass): training graph for teacher_task + distill targets.
            teacher_preds, teacher_taps = self._forward_with_taps(self.teacher, batch["img"], tap_layers)
            teacher_task_loss, teacher_loss_items = self.teacher.loss(batch, teacher_preds)
            teacher_loss_items = teacher_loss_items.detach()
            teacher_raw = self._parse_raw_preds(teacher_preds)
        else:
            teacher_raw, teacher_taps = self._teacher_forward_raw(batch["img"], tap_layers)

        feature_loss = self._feature_loss(student_raw, teacher_raw)

        align_loss = torch.tensor(0.0, device=batch["img"].device)
        align_start = getattr(self.args, "align_start_epoch", 10)
        if getattr(self.args, "align", True) and self.current_epoch >= align_start:
            align_loss = self._alignment_loss(student_raw, teacher_raw)

        # Backbone distillation (dictionary modules): weighted align + attention restriction + commit.
        dict_align_loss = torch.tensor(0.0, device=batch["img"].device)
        dict_attn_loss = torch.tensor(0.0, device=batch["img"].device)
        dict_commit_loss = torch.tensor(0.0, device=batch["img"].device)
        if dict_on:
            dict_align_loss, dict_attn_loss, dict_commit_loss = self._dictionary_losses(
                teacher_taps, self._cached_saliency
            )

        alpha = getattr(self.args, "task_loss", 1.0)
        beta = getattr(self.args, "feature_loss", 0.5)
        gamma = getattr(self.args, "align_loss", 0.5)
        delta = getattr(self.args, "teacher_task_loss", 1.0) if teacher_joint else 0.0
        beta_d, beta_a, beta_c = self._dict_gains()
        # Task losses come back as (per-component loss * batch_size); the distillation losses are plain
        # per-batch means. Scale the distill terms by batch size (and sum the task vectors) so every
        # term shares the same per-sample footing and the nominal weights truly control relative
        # influence. Without this the feature/align gradients were ~1/batch_size too weak, which matches
        # the earlier "feature/align loss stays flat and distillation barely helps" symptom.
        bs = batch["img"].shape[0]
        total_loss = (
            alpha * student_task_loss.sum()
            + delta * teacher_task_loss.sum()
            + bs
            * (
                beta * feature_loss
                + gamma * align_loss
                + beta_d * dict_align_loss
                + beta_a * dict_attn_loss
                + beta_c * dict_commit_loss
            )
        )

        all_loss_items = torch.stack(
            [
                student_loss_items[0].detach(),
                student_loss_items[1].detach(),
                student_loss_items[2].detach(),
                align_loss.detach(),
                feature_loss.detach(),
                dict_align_loss.detach(),
                dict_attn_loss.detach(),
                teacher_loss_items[0],
                teacher_loss_items[1],
                teacher_loss_items[2],
            ]
        )
        return total_loss, all_loss_items


class YOLOFDistillationTrainer(DetectionTrainer):
    """Trainer for YOLOF distillation (offline frozen teacher or online joint + optional late freeze)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "align_loss",
            "feature_loss",
            "dict_loss",
            "dattn_loss",
            "t_box_loss",
            "t_cls_loss",
            "t_dfl_loss",
        )
        self.add_callback("on_train_start", self._freeze_teacher_callback)
        self.add_callback("on_train_start", self._log_split_grad_clip_callback)
        self.add_callback("on_train_epoch_start", self._update_current_epoch)

    _GRAD_CLIP_MAX_NORM = 10.0  # match BaseTrainer.optimizer_step

    @staticmethod
    def _grad_param_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
        """Partition trainable KD parameters for per-branch gradient clipping (scheme A).

        Returns (student, teacher, distill) parameter lists. Distill = neck projectors +
        dictionary modules; everything else trainable on the wrapper is treated as student.
        """
        kd = unwrap_model(model)
        student, teacher, distill = [], [], []
        for name, p in kd.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("teacher."):
                teacher.append(p)
            elif name.startswith("feature_projectors.") or name.startswith("dictionary_modules."):
                distill.append(p)
            else:
                student.append(p)
        return student, teacher, distill

    def _log_split_grad_clip_callback(self, trainer):
        kd = unwrap_model(trainer.model)
        if getattr(kd, "teacher", None) is None:
            return
        opt = trainer.optimizer.__class__.__name__ if trainer.optimizer is not None else "Optimizer"
        LOGGER.info(
            f"{colorstr('KD:')} Split grad clipping enabled (student / teacher / distill, "
            f"max_norm={self._GRAD_CLIP_MAX_NORM}) with shared {opt} optimizer"
        )

    def optimizer_step(self):
        """Clip gradients per branch so student/distill norms do not shrink teacher updates."""
        self.scaler.unscale_(self.optimizer)
        kd = unwrap_model(self.model)
        max_norm = self._GRAD_CLIP_MAX_NORM
        if getattr(kd, "teacher", None) is None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        else:
            for group in self._grad_param_groups(self.model):
                if group:
                    torch.nn.utils.clip_grad_norm_(group, max_norm=max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def ddp_kwargs(self) -> dict:
        """Teacher is frozen offline or after ``teacher_freeze_epoch``; its params get no grad under DDP."""
        return {"find_unused_parameters": True}

    def _freeze_teacher_callback(self, trainer):
        # Online distillation keeps the teacher trainable; only freeze in offline mode.
        if bool(getattr(trainer.args, "online_distill", True)):
            return
        model = unwrap_model(trainer.model)
        if getattr(model, "teacher", None) is not None:
            LOGGER.info("Freezing teacher model for offline distillation")
            model.teacher.eval()
            model.teacher.requires_grad_(False)
            for param in model.teacher.parameters():
                param.requires_grad = False

    def _update_current_epoch(self, trainer):
        model = unwrap_model(trainer.model)
        model.current_epoch = trainer.epoch
        if hasattr(model, "_apply_teacher_freeze_if_needed"):
            model._apply_teacher_freeze_if_needed(trainer)
        if hasattr(model, "_set_teacher_criterion_epoch"):
            model._set_teacher_criterion_epoch(trainer.epoch)

    def validate(self):
        """Run student val, then evaluate the teacher alone for monitoring."""
        metrics, fitness = super().validate()
        if RANK in {-1, 0}:
            self._maybe_validate_teacher()
        return metrics, fitness

    def _maybe_validate_teacher(self) -> None:
        """Log teacher-only val mAP every ``teacher_val_interval`` epochs (default 1).

        Evaluates the EMA-smoothed ``teacher`` submodule via the same in-training validation path
        as solo runs (``trainer.ema`` weights, FP16 when AMP is on, no AutoBackend fuse), so
        ``teacher_val.csv`` is directly comparable to ``results.csv`` from standalone teacher
        training. Only a deepcopy of the EMA teacher is validated; the live teacher is untouched.
        Set ``teacher_val_interval=0`` to disable.
        """
        interval = int(getattr(self.args, "teacher_val_interval", 1) or 0)
        if interval <= 0 or (self.epoch + 1) % interval != 0:
            return

        kd_model = unwrap_model(self.model)
        if getattr(kd_model, "teacher", None) is None:
            return

        ema_kd = unwrap_model(self.ema.ema) if self.ema is not None else None
        teacher_ema = getattr(ema_kd, "teacher", None) if ema_kd is not None else None
        if teacher_ema is None:
            LOGGER.warning(
                f"{colorstr('KD:')} Teacher EMA val skipped at epoch {self.epoch + 1}: "
                "EMA teacher submodule unavailable"
            )
            return

        frozen = bool(getattr(kd_model, "_teacher_frozen", False))
        plots = self.validator.args.plots
        self.validator.args.plots = False
        stats = None
        teacher_copy = None

        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            # Deepcopy EMA teacher only; validator(trainer=self, model=...) uses the training val
            # path (same as solo results.csv) and never mutates the live teacher or KD wrapper.
            teacher_copy = deepcopy(unwrap_model(teacher_ema)).to(self.device)
            stats = self.validator(trainer=self, model=teacher_copy)
        except Exception as exc:
            LOGGER.warning(
                f"{colorstr('KD:')} Teacher EMA val skipped at epoch {self.epoch + 1} "
                f"(training continues): {exc}"
            )
            return
        finally:
            del teacher_copy
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.validator.args.plots = plots

        if not stats:
            return

        map50 = float(stats.get("metrics/mAP50(B)", 0.0))
        map5095 = float(stats.get("metrics/mAP50-95(B)", 0.0))
        LOGGER.info(
            f"{colorstr('KD:')} Teacher EMA val epoch {self.epoch + 1}/{self.epochs} "
            f"mAP50={map50:.4f} mAP50-95={map5095:.4f} frozen={frozen}"
        )
        self._append_teacher_val_csv(map50, map5095, frozen)

    def _append_teacher_val_csv(self, map50: float, map5095: float, frozen: bool) -> None:
        """Append one teacher-val row to ``teacher_val.csv`` in the run directory."""
        path = self.save_dir / "teacher_val.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("epoch,time,frozen,metrics/t_mAP50(B),metrics/t_mAP50-95(B)\n", encoding="utf-8")
        elapsed = time.time() - self.train_time_start
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"{self.epoch + 1},{elapsed:.3g},{int(frozen)},{map50:.6g},{map5095:.6g}\n"
            )

    def _build_teacher(self, teacher_path: str):
        """Build a teacher DetectionModel from a .yaml (fresh / scratch) or a .pt checkpoint."""
        if str(teacher_path).endswith((".yaml", ".yml")):
            LOGGER.info(f"Building teacher from config {teacher_path}")
            teacher_model = DetectionModel(
                teacher_path, nc=self.data["nc"], ch=self.data["channels"], verbose=False
            )
            teacher_weights = getattr(self.args, "teacher_weights", None)
            if teacher_weights:
                LOGGER.info(f"Loading teacher pretrained weights from {teacher_weights}")
                tw_model, _ = load_checkpoint(teacher_weights)
                teacher_model.load(tw_model)
            return teacher_model

        LOGGER.info(f"Loading teacher model from checkpoint {teacher_path}")
        teacher_model, _ = load_checkpoint(teacher_path)
        if isinstance(teacher_model.args, dict):
            teacher_model.args = SimpleNamespace(**teacher_model.args)
        return teacher_model

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = YOLOFDistillationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        # set_model_attributes() runs after get_model(); attach args now so
        # build_distillation_modules() can read the dictionary-distillation config.
        model.args = self.args

        teacher_path = getattr(self.args, "teacher", None)
        if teacher_path:
            online = bool(getattr(self.args, "online_distill", True))
            with torch_distributed_zero_first(RANK):
                # Independent backbones: teacher is built as a fully separate model (own backbone).
                teacher_model = self._build_teacher(teacher_path)

                # Attach training attributes so the teacher can compute its own task loss.
                teacher_model.nc = self.data["nc"]
                teacher_model.names = self.data["names"]
                teacher_model.args = self.args
                if getattr(teacher_model, "end2end", False):
                    teacher_model.set_head_attr(max_det=self.args.max_det)
                # criterion is lazy-initialized in loss() after model.to(device)

                if online:
                    teacher_model.train()
                    teacher_model.requires_grad_(True)
                else:
                    teacher_model.eval()
                    teacher_model.requires_grad_(False)

                model.teacher = teacher_model
                model.build_distillation_modules(imgsz=self.args.imgsz)

                if online:
                    teacher_model.train()  # build_distillation_modules leaves teacher in eval

        if weights:
            # Load after attaching teacher and building projectors. On resume the checkpoint contains
            # teacher.* and feature_projectors.*; loading earlier would silently drop those keys because
            # the submodules do not exist yet, resetting the teacher/projectors while the student resumes.
            LOGGER.info(f"Loading model weights from {weights}")
            model.load(weights)
        return model

    def get_validator(self):
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
