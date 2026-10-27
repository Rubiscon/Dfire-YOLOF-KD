"""Protocol-matched Ultralytics training for D-Fire and VOC2007.

Default dataset is D-Fire. Pass ``--dataset voc2007`` for VOC2007
trainval (5011) / test (4952), nc=20. Init from COCO ``yolo26n.pt``.
Online KD starts the teacher from that same file, not a finished
in-dataset checkpoint.

    python scripts/convert_voc2007.py
    python scripts/train_dfs.py --dataset voc2007 --baseline yolo26n --name-suffix seed0
    python scripts/train_dfs.py --dataset voc2007 --baseline dcn-solo --name-suffix seed0
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_TQDM_NONINTERACTIVE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

torch.backends.cudnn.benchmark = True

from dfire_parity.protocol import CONTROLLED_ARGS, warmup_epochs_for
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer, YOLOFDistillationTrainer

DATASETS: dict[str, dict[str, Any]] = {
    "dfire": {
        "yaml": ROOT / "ultralytics" / "cfg" / "datasets" / "dfire.yaml",
        "project": "dfire-protocol-baselines",
        "prefix": "dfire",
        "nc": 2,
        "description": "D-Fire smoke/fire",
    },
    "voc2007": {
        "yaml": ROOT / "ultralytics" / "cfg" / "datasets" / "voc2007.yaml",
        "project": "voc2007-baselines",
        "prefix": "voc2007",
        "nc": 20,
        "description": "VOC2007 trainval/test (nc=20)",
    },
}

BASELINES: dict[str, dict[str, Any]] = {
    "yolo26n": {
        "trainer": "detect",
        "description": "YOLO26n FPN",
        "model": "yolo26n.yaml",
        "name": "yolo26n",
    },
    "dcn-solo": {
        "trainer": "detect",
        "description": "YOLOF-DCN student solo",
        "model": "yolo26n-DCN.yaml",
        "name": "dcn-solo",
    },
    "dcn-kd": {
        "trainer": "kd",
        "description": "YOLOF-DCN + dLdx dictionary KD (D_dldx_new_2 gains)",
        "model": "yolo26n-DCN.yaml",
        "name": "dcn-kd",
    },
    # --- Narrow stride-8 high-resolution tap arms -----------------------------------
    # The dilated encoder widens the receptive field on C5 but never adds sampling points,
    # so at 640 the detection lattice is 20x20 and AP_S is bounded by the stride, not by
    # capacity. These three arms test whether adding one finer lattice lifts that bound,
    # and by how much latency it costs.
    #
    # In this codebase the Detect head derives its hidden width from the *first* level's
    # channel count, so a tap that emits fewer channels also narrows the C5 branch:
    #   dcn-solo-p3     wide head + tap   vs  dcn-solo     -> isolates "one added lattice"
    #   dcn-solo-headn  narrow head, no tap  (control)
    #   dcn-solo-p3n    narrow head + tap vs dcn-solo-headn -> isolated, cheaper variant
    "dcn-solo-p3": {
        "trainer": "detect",
        "description": "YOLOF-DCN solo + stride-8 high-resolution tap (wide Detect head)",
        "model": "yolo26n-DCN-P3.yaml",
        "name": "dcn-solo-p3",
    },
    "dcn-solo-headn": {
        "trainer": "detect",
        "description": "YOLOF-DCN solo, narrowed Detect head, no tap (control arm)",
        "model": "yolo26n-DCN-headn.yaml",
        "name": "dcn-solo-headn",
    },
    "dcn-solo-p3n": {
        "trainer": "detect",
        "description": "YOLOF-DCN solo + stride-8 tap, narrowed Detect head",
        "model": "yolo26n-DCN-P3n.yaml",
        "name": "dcn-solo-p3n",
    },
    # --- Stride-16 tap: the latency-conscious version of the same idea -------------------
    # The stride-8 tap lifts AP_S by 40% but costs ~25% latency, because the added 80x80
    # lattice makes the head do 16x the spatial work of the existing 20x20 one. A stride-16
    # lattice is 40x40, i.e. 4x instead of 16x, and measures at only +4.7% GFLOPs against the
    # baseline (versus +27.9% for stride 8). These two arms ask whether a defensible amount of
    # the AP_S gain survives at a cost that keeps the CPU-efficiency claim intact.
    #
    #   dcn-solo-p4n     stride-16 tap, legacy head  (mirror of dcn-solo-p3n)
    #   dcn-solo-p4npl   stride-16 tap, per-level head: narrow on the 40x40 lattice, full
    #                    width restored on C5. Needs `head_width: per_level` support in Detect.
    "dcn-solo-p4n": {
        "trainer": "detect",
        "description": "YOLOF-DCN solo + stride-16 tap (legacy global head width)",
        "model": "yolo26n-DCN-P4n.yaml",
        "name": "dcn-solo-p4n",
    },
    "dcn-solo-p4npl": {
        "trainer": "detect",
        "description": "YOLOF-DCN solo + stride-16 tap, per-level head width",
        "model": "yolo26n-DCN-P4n-pl.yaml",
        "name": "dcn-solo-p4npl",
    },
    # Same head as dcn-solo-p4npl plus a two-level top-down fusion: C5 is projected,
    # upsampled, concatenated with the tap and fused back. Detect widths are held identical
    # (stride [16, 32], c2 [16, 32], c3 [48, 128]).
    #
    # Measured locally: +47,808 parameters, +0.139 GFLOPs, +8.9% latency.
    "dcn-solo-p4npl-fpnlite": {
        "trainer": "detect",
        "description": "YOLOF-DCN stride-16 tap + minimal top-down FPN fusion",
        "model": "yolo26n-DCN-P4npl-fpnlite.yaml",
        "name": "dcn-solo-p4npl-fpnlite",
    },
    # --- pindw distillation on top of the stride-8 tap -------------------------------
    # Same training objective as the existing `dcn-kd --kd-variant pindw` arms, but the student
    # is the tap model instead of the single-level one. Everything the KD machinery taps is
    # unchanged: the backbone tap is still layer 10 (C2PSA) and there are still three
    # DilatedDeformBlocks, so the three per-scale projectors and the dictionary pairing are
    # identical in count and shape. HighResTap deliberately does NOT register a distillation
    # block, so the added lattice is presently trained by the task loss alone; distilling the
    # teacher's P3 into it is a separate change.
    #
    # Run it as:  --baseline dcn-kd-p3n --kd-variant pindw
    "dcn-kd-p3n": {
        "trainer": "kd",
        "description": "YOLOF-DCN stride-8 tap + dictionary/neck/response KD (student of the tap family)",
        "model": "yolo26n-DCN-P3n.yaml",
        "name": "dcn-kd-p3n",
    },
}

# Same dictionary / response KD as Log/D_dldx_new_2, on CONTROLLED_ARGS (not optimizer=auto).
_KD_RECIPE: dict[str, Any] = {
    "teacher": "yolo26n.yaml",
    "online_distill": True,
    "teacher_freeze_use_ema": True,
    "task_loss": 1.0,
    "teacher_task_loss": 1.0,
    "feature_norm": "channel",
    "feature_loss": 0.08,
    "align": True,
    "align_start_epoch": 20,
    "align_loss": 0.12,
    "align_branch": "one2many",
    "align_cls_mode": "kl",
    "distill_temperature": 3,
    "align_box": 2.0,
    "align_cls": 4.0,
    "distill_conf_thres": 0.25,
    "distill_iou_thres": 0.5,
    "dict_student_layer": 10,
    "dict_teacher_layers": [6],
    "dict_start_epoch": 0,
    "dict_weight": "saliency_dLdx",
    "dict_match": "hard",
    "dict_match_temp": 0.07,
    "dict_feature_norm": "channel",
    "dict_saliency_ema": 0.9,
    "dict_attn_start_epoch": 0,
    "dict_commit_loss": 0.0,
    "dict_align_loss": 0.12,
    "dict_attn_loss": 0.25,
    "dict_weight_norm": "mean",
    "dict_saliency_blur": 0.0,
    "dict_saliency_clip": 0.0,
}

# VOC2007 overlay on the D-Fire dLdx recipe. Response align at epoch 21 dumped
# student mAP50 by 16 points (0.551 -> 0.393) because TAL+KL on a still-weak
# 20-class teacher dominates box/cls GT. Delay align until the online teacher
# is competent, cut box/KL gains so YOLOF can keep fitting its own boxes
# (mAP50-95), and hold attention restriction until the same point.
_VOC_KD_OVERRIDES: dict[str, Any] = {
    "align_start_epoch": 80,
    "align_loss": 0.06,
    "align_box": 1.0,
    "align_cls": 2.0,
    "distill_conf_thres": 0.45,
    "feature_loss": 0.04,
    "dict_align_loss": 0.08,
    "dict_attn_loss": 0.10,
    "dict_attn_start_epoch": 80,
}

# VOC follow-ups already trained; kept so --kd-variant still matches those logs.
_VOC_KD_VARIANTS: dict[str, dict[str, Any]] = {
    "align50": {
        "align_start_epoch": 50,
        "dict_attn_start_epoch": 50,
        "align_box": 1.0,
        "dict_align_loss": 0.08,
        "dict_attn_loss": 0.10,
    },
    "boxlate": {
        "align_start_epoch": 80,
        "align_box": 2.0,
        "align_loss": 0.08,
        "dict_align_loss": 0.08,
        "align_cls": 2.0,
    },
    "dict12": {
        "align_start_epoch": 80,
        "align_box": 1.0,
        "dict_align_loss": 0.12,
        "dict_attn_loss": 0.25,
        "dict_attn_start_epoch": 80,
    },
}

# SGD retune of D_dldx_new_2. MuSGD-tuned recipe keeps dict_attn from epoch 0
# and turns on TAL+KL align at 20 with align_loss raw ~3 vs box ~2. SGD's
# teacher/student fit boxes slower (late box ~1.42 vs MuSGD ~1.34), so these
# overlays probe schedule vs gain without changing the dictionary itself.
_DFIRE_KD_VARIANTS: dict[str, dict[str, Any]] = {
    "delay40": {
        # Teacher mAP is higher by e40 (solo ~0.63); keep dict gains.
        "align_start_epoch": 40,
    },
    "attn50": {
        # Hold spatial restriction until SGD features stabilize; dict_align still on.
        "align_start_epoch": 50,
        "dict_attn_start_epoch": 50,
    },
    "softgain": {
        # Same schedule as D_dldx_new_2; lower KD so GT boxes can keep moving.
        "feature_loss": 0.05,
        "align_loss": 0.08,
        "align_box": 1.5,
        "align_cls": 3.0,
        "dict_align_loss": 0.10,
        "dict_attn_loss": 0.12,
    },
    "dictfirst": {
        # Dictionary is the method; delay the noisy one2many TAL+KL head match.
        "align_start_epoch": 80,
        "align_loss": 0.08,
        "align_box": 1.5,
        "align_cls": 3.0,
        "dict_attn_start_epoch": 20,
        "dict_attn_loss": 0.18,
        "dict_align_loss": 0.15,
        "feature_loss": 0.06,
    },
    "latehard": {
        # Wait for SGD to fit, then distill harder than the MuSGD recipe.
        "align_start_epoch": 40,
        "dict_attn_start_epoch": 40,
        "dict_align_loss": 0.15,
        "dict_attn_loss": 0.30,
    },
    "temp2": {
        # Response KL temperature only. The teacher target is built from detection
        # confidence and is never softened by T, so T does two things at once:
        # sharpens the student softmax and rescales the term by T*T. Holding
        # T*T*align_cls at the baseline 2*2*9 == 3*3*4 == 36 keeps the gradient
        # magnitude fixed, so this run isolates the temperature shape alone.
        "distill_temperature": 2,
        "align_cls": 9.0,
    },
    "hardgain": {
        # Mirror of softgain: the response branch looked strong in the sweep, so
        # push the align gains up while keeping the schedule at the baseline.
        "align_loss": 0.16,
        "align_box": 3.0,
        "align_cls": 6.0,
        "feature_loss": 0.10,
    },
}

# Dictionary-structure ablations: drop the teacher-side dictionary encoder entirely and make
# the student-side encoder a MobileNetV2 depthwise conv. Everything else stays on the
# D_dldx_new_2 recipe so the only moved variable is the dictionary geometry.
#
# Under hard matching the encoders receive no gradient (argmax is not differentiable), so
# "depthwise + hard" is a fixed random depthwise projection. The ``*st`` arms switch to
# straight_through (hard forward, soft backward) and enable the commitment loss, because
# ``target = t_reorg.detach()`` means alignment alone cannot train Q/K.
#
# ``basefix`` is the untouched baseline recipe, measured under the same code as the ablations.
_DICT_ST_DIAG = {"dict_match_log_interval": 100}

# Pinned initialisation for the trainable dictionary projection (see the `pinbase` arm).
# One value for the whole clean set, so every arm in it starts from the same projection.
_DICT_INIT_SEED = 1234

# VOC flavour of the sliced-matching dictionary: the same setup as `pindw` (teacher-side
# encoder removed, channel-local depthwise student encoder, pinned projection) with the VOC
# recipe, which is applied automatically for `--dataset voc2007` before the variant is
# layered on.
#
# The name is deliberately NOT `pindw`. The two tables are separate namespaces when RESOLVING a
# variant, but the guard in `_apply_kd_variant` is name-based: any name present in
# `_VOC_KD_VARIANTS` is rejected for non-VOC datasets with "VOC KD variant X is voc2007-only".
# Registering `pindw` here therefore broke the D-Fire `pindw` arm outright -- caught by
# tests/test_train_dfs.py. A distinct name keeps both usable.
# Defined here rather than inline in `_VOC_KD_VARIANTS` because that table sits above
# `_DICT_INIT_SEED` and the seed must not be duplicated as a literal.
_VOC_KD_VARIANTS["vocpindw"] = {
    "dict_teacher_encoder": False,
    "dict_student_encoder": "depthwise",
    "dict_init_seed": _DICT_INIT_SEED,
}
_VOC_KD_VARIANTS["vocpindwbox"] = {
    # vocpindw plus the two boxlate gains. The VOC schedule is unchanged:
    # align still starts at 80, and dict_attn stays at 0.10 from that epoch.
    "dict_teacher_encoder": False,
    "dict_student_encoder": "depthwise",
    "dict_init_seed": _DICT_INIT_SEED,
    "align_box": 2.0,
    "align_loss": 0.08,
}

# Epoch at which the delayed fixed-map arms switch the whole dictionary branch on. 0-indexed,
# because `_dict_active()` compares it against `self.current_epoch`. Chosen at 40 because the
# measured deficit of the undelayed fixed-map arm is widest over e21-40.
_DICT_DELAY_EPOCH = 40

# A-path shared configuration: channel-local learnable projection plus a differentiable
# assignment so the correspondence can actually be trained.
#   proj_form=channel_local : depthwise 3x3, identity-init. Channel-locality is structural
#                             (groups=C), so the pairing stays visible whatever the weights do.
#                             A per-channel affine would NOT work here: the align loss
#                             standardises each channel over space, which cancels an affine
#                             exactly and leaves its parameters with no gradient (measured
#                             1.4e-05 vs 2.0e-02 for the 3x3 form, and that residual is pure
#                             eps leakage).
#   proj_kernel=3           : the smallest odd kernel that survives standardisation.
#   match_init=identity     : start from a meaningful correspondence so the differentiable
#                             assignment refines a good map instead of random noise.
#   freeze_encoders=False   : required for the assignment to be learnable at all.
_DICT_A_ST = {
    **_DICT_ST_DIAG,
    "dict_proj_form": "channel_local",
    "dict_proj_kernel": 3,
    "dict_match": "straight_through",
    "dict_match_init": "identity",
    "dict_freeze_encoders": False,
    "dict_attn_loss": 0.25,
    "dict_commit_loss": 0.05,
}
_DICT_A_SOFT = {**_DICT_A_ST, "dict_match": "soft"}
_DICT_DW = {
    **_DICT_ST_DIAG,
    "dict_teacher_encoder": False,
    "dict_student_encoder": "depthwise",
}
_DICT_DWPW = {
    **_DICT_ST_DIAG,
    "dict_teacher_encoder": False,
    "dict_student_encoder": "depthwise_separable",
}
# straight_through modifier only (keeps the student encoder form of the arm it extends).
_DICT_ST_MODS = {
    "dict_match": "straight_through",
    "dict_freeze_encoders": False,
    "dict_commit_loss": 0.05,
}
_DICT_DW_ST = {**_DICT_DW, **_DICT_ST_MODS}
_DFIRE_KD_VARIANTS.update(
    {
        # Both encoders kept (proposal figure intact) but initialised as channel-wise
        # identities, so K = pool(x^e) and Q = pool(n) exactly: the two sides stay in
        # their ORIGINAL feature spaces and are therefore comparable. Two independent
        # random projections do the opposite and sit at chance.
        # Matching accuracy against an INDEPENDENT, layout-free ground truth (per-channel
        # activation-intensity series correlated across images, ground truth estimated on
        # held-out images; chance = 0.78%): random encoders 0.72-0.75%, identity encoders
        # 1.33% at d=4 and 2.08% at d=100. So identity is genuinely better than chance but
        # the absolute accuracy is ~2%. An earlier figure of 21.4% came from a
        # spatial-pattern oracle that shares the raw features with the identity design and
        # is therefore partly circular; it must not be quoted.
        "identinit": {**_DICT_ST_DIAG, "dict_match_init": "identity"},
        # Same function as identinit (a frozen, identity-initialised encoder is an
        # identity regardless of conv type) but with the teacher-side encoder removed and
        # a depthwise student encoder, so the module drops from 1,000,448 to 265,216
        # parameters. This is the variant that is both correct AND lightweight.
        "dwident": {
            **_DICT_ST_DIAG,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_match_init": "identity",
        },
        # Token-resolution pair. The proposal reduces the teacher early feature "by 16
        # times" with x^{es} in R^{Ce x He/4 x We/4}, i.e. each side /4 (area /16), so the
        # 40x40 tap gives a 10x10 map and d = 100. The recipe default divides each side by
        # 16 instead (d = 4), which is 25x fewer tokens.
        # Against the independent held-out intensity ground truth, adding tokens raises the
        # identity matcher from 1.33% (d=4) to 2.08% (d=100); chance is 0.78% and random
        # encoders stay at 0.72-0.75% at every resolution. So resolution helps the matcher,
        # it is just nowhere near a usable correspondence either way.
        #   res      : resolution alone, recipe otherwise untouched (isolates the effect)
        #   resident : resolution + identity encoders (the proposal's stated intent:
        #              comparable spaces AND enough tokens to compare fine structure)
        # Note: batch_corr ignores tokens entirely (it correlates per-image channel means),
        # so the oracle arm is already resolution-independent.
        "res": {
            **_DICT_ST_DIAG,
            "dict_match_grid_divisor": 4,
        },
        "resident": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_match_grid_divisor": 4,
        },
        "basefix": dict(_DICT_ST_DIAG),
        # Single-key delta from basefix: only dict_freeze_encoders is flipped. This arm
        # exists to separate the two things that changed between the r1-r3 baselines and
        # every arm measured since 2026-09-22, which were confounded:
        #   (a) the freeze fix itself -- encoder BatchNorm held in eval mode and the params
        #       kept out of the optimizer, versus the historical runs where BatchNorm sat in
        #       train mode (normalising by per-batch statistics, running stats updating);
        #   (b) any other code change between 09-18 and 09-24.
        # Reading three points together decomposes them, same-machine on westb:
        #   r1-r3 (old code, unfrozen) vs nofreeze (new code, unfrozen) -> (b) alone
        #   nofreeze (new code, unfrozen) vs basefix (new code, frozen) -> (a) alone
        # Why it matters: 13 of the 21 measured arms to date run with frozen=True, so almost
        # every "arm vs the reference trio" gap in the ledger currently mixes (a) and (b)
        # with the arm's own change, and cannot be attributed. The measured size of the
        # effect is not small -- switching the encoder BatchNorm between the two modes on
        # identical weights reassigns ~90% of student channels (assignment-map agreement
        # 0.0985) and shifts the tokens by ~1.0 in relative L2 -- and for the identity-init
        # family the freeze is what makes identity initialisation actually mean the identity
        # function, since train-mode BatchNorm would re-normalise it away.
        "nofreeze": {
            **_DICT_ST_DIAG,
            "dict_freeze_encoders": False,
        },
        # Same dictionary geometry with the encoders left UNFROZEN: teacher-side dictionary
        # module removed, student-side encoder changed to a MobileNetV2 depthwise conv, hard
        # matching kept.
        #
        # Under hard matching the argmax is not differentiable, so the encoders receive no
        # gradient either way. Not freezing only keeps their BatchNorm in train mode, which is
        # the condition the older r1-r3 baselines ran under (their freeze call was undone by
        # the trainer), so this arm doubles as a same-condition reference for them.
        "dwnf": {
            **_DICT_ST_DIAG,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_freeze_encoders": False,
        },
        # Component ablations, one moved variable each:
        #   indexmatch : matching replaced by a fixed per-channel alignment
        #                (student channel s <- teacher channel s mod Ct).
        #   nodict     : dict_align_loss = dict_attn_loss = 0, feature and response
        #                branches untouched.
        #   unifw      : dict_weight=none, the same mass-normalised reduction with a
        #                uniform weight instead of S = mean_c|dLt/dxe|, so the saliency
        #                backward is skipped.
        "indexmatch": {
            **_DICT_ST_DIAG,
            "dict_match": "index",
        },
        "nodict": {
            **_DICT_ST_DIAG,
            "dict_align_loss": 0.0,
            "dict_attn_loss": 0.0,
        },
        "unifw": {
            **_DICT_ST_DIAG,
            "dict_weight": "none",
        },
        # Identity-init arms. identinit fits the align term better but scores slightly lower
        # on val, so these separate the possible causes, all built on identinit:
        #   identlowalign : halve dict_align_loss (0.12 -> 0.06).
        #   identlowattn  : halve dict_attn_loss (0.25 -> 0.12).
        #   identattnfix  : compute the AT residual as raw student energy vs raw teacher
        #                   energy instead of raw vs standardised.
        "identlowalign": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_align_loss": 0.06,
        },
        # Align-off arms. dict_align_loss is the only place the correspondence enters the
        # loss: the AT term averages over channels and is pairing-invariant, and commit and
        # InfoMax are zero here. Unlike `nodict`, the AT term and the modules stay alive.
        #   noalign       align 0.0 with the default encoders, a one-key ablation against
        #                 the reference recipe.
        #   identnoalign  align 0.0 with identity encoders; together with identinit (0.12)
        #                 and identlowalign (0.06) this is an alpha sweep at fixed init.
        "noalign": {
            **_DICT_ST_DIAG,
            "dict_align_loss": 0.0,
        },
        "identnoalign": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_align_loss": 0.0,
        },
        # Teacher encoder removed and student encoder depthwise, at the alpha that works
        # (0.06, identity init). `dwident` and `dwnf` run the same geometry at the recipe's
        # alpha=0.12, where the identity family is behind the baseline, and `identlowalign`
        # recovers at 0.06 but keeps both encoders in full-conv form.
        #
        # With identity initialisation the encoder is the identity function whatever its conv
        # type, so the reassembled teacher target is bit-identical to the full-conv version and
        # only the parameter count changes (dictionary module: 1,000,448 -> 265,216). With a
        # random init the depthwise form gives a worse assignment (top-1 0.023 vs 0.070), so
        # this is not the same thing as `dw`/`dwnf`.
        #
        # The two arms differ only in whether `proj` is pinned: resizing the encoders changes
        # how much RNG they consume before `proj` is built, which re-rolls it.
        "identlowaligndw": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_align_loss": 0.06,
            "dict_init_seed": _DICT_INIT_SEED,
        },
        "dwidentlow": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_align_loss": 0.06,
        },
        # --- A PATH: channel-local LEARNABLE projection + a correspondence that can be trained --
        # Motivation, all measured rather than assumed:
        #   * The shipped projection mixes every student channel into every output channel, so
        #     reassigning a student channel's target merely relabels independent sub-problems and
        #     the pairing is absorbed: fitting the projection to a map versus to that map with the
        #     student channels permuted gives per-channel losses agreeing to 0.0006, while changing
        #     which teacher channels are used at all is worth 0.04.
        #   * With a channel-local projection the pairing is worth 13.4% of the align loss instead
        #     of 0.03%. But the parameter-free identity projection then leaves nothing learnable,
        #     so the pairing must carry the entire benefit.
        #   * Under `hard` matching the assignment is an argmax, so it is non-differentiable: the
        #     encoders are frozen at their random init and there is nothing for a supervision term
        #     to train. That is why every accuracy improvement measured so far changed nothing.
        #
        # So these arms pair `dict_proj_form=channel_local` (depthwise 3x3, identity-init, 2560
        # params) with a DIFFERENTIABLE assignment -- `straight_through` (hard forward, soft
        # backward, preserving Figure 2's forward assignment) or full `soft` -- and use identity
        # init for the encoders so the assignment starts meaningful rather than arbitrary, then
        # is refined. `dict_commit_loss` is the only term that directly supervises the pairing.
        #
        # alpha is swept at both the recipe value (0.12) and 0.06, because a channel-local
        # projection passes about 4.7x more align gradient to the student tap than the shipped
        # DeconvNet does (measured 0.00603 vs 0.00129 at identical weights), and `identlowalign`
        # already showed that alpha 0.06 recovers most of what the identity family loses at 0.12.
        "alocst12": {**_DICT_A_ST, "dict_align_loss": 0.12},
        "alocst06": {**_DICT_A_ST, "dict_align_loss": 0.06},
        "alocsoft12": {**_DICT_A_SOFT, "dict_align_loss": 0.12},
        "alocsoft06": {**_DICT_A_SOFT, "dict_align_loss": 0.06},
        # commit sweep at the lower alpha, with the zero-commit arm as the control that separates
        # "the pairing is differentiable" from "the pairing is explicitly supervised"
        "alocstc00": {**_DICT_A_ST, "dict_align_loss": 0.06, "dict_commit_loss": 0.0},
        "alocstc10": {**_DICT_A_ST, "dict_align_loss": 0.06, "dict_commit_loss": 0.10},
        "alocstc50": {**_DICT_A_ST, "dict_align_loss": 0.06, "dict_commit_loss": 0.50},
        # --- the freeze requirement, tested PER SIDE ------------------------------------------
        # The requirement is asymmetric: the teacher-side encoder must stay fixed so the
        # distillation target is stable, while the student-side encoder may either adapt or stay
        # fixed. Gradient flow measured on the real module says the teacher side never trains
        # anyway -- align/AT consume a detached target, `commit` detaches its teacher-side target
        # (`k_n.detach()`), and `dict_infomax_loss` is 0 in every arm of this project -- so the
        # freeze flag on that side changes only its BatchNorm mode (batch vs running statistics).
        # These two arms make the choice explicit and verifiable instead of incidental, and add the
        # one cell that was never run: a straight-through arm whose STUDENT encoder is also frozen.
        #   alocst06tf : teacher frozen, student trainable  (the instruction, stated explicitly)
        #                Prediction: numerically indistinguishable from `alocst06`. If so, that is
        #                the direct confirmation that the teacher side contributed no learning.
        #   alocst06bf : teacher frozen, student ALSO frozen
        #                Isolates what letting the student query encoder adapt is worth. `proj` is a
        #                separate learnable path (s_proj = proj(s_feat)), so it keeps learning the
        #                alignment from the align term; only the commit-driven query refinement stops.
        "alocst06tf": {
            **_DICT_A_ST,
            "dict_align_loss": 0.06,
            "dict_freeze_teacher_encoder": True,
            "dict_freeze_student_encoder": False,
        },
        "alocst06bf": {
            **_DICT_A_ST,
            "dict_align_loss": 0.06,
            "dict_freeze_teacher_encoder": True,
            "dict_freeze_student_encoder": True,
        },
        "identlowattn": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_attn_loss": 0.12,
        },
        "identattnfix": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_attn_consistent": True,
        },
        # --- clean set with the trainable projection's initialisation held fixed --------
        # `proj` is built after the encoders and is the only randomly initialised parameter
        # training can still move under a frozen-encoder assignment. Any change to the
        # encoder shapes shifts the RNG stream, so `proj` starts somewhere else: measured
        # under an identical seed, removing the teacher encoder and switching the student
        # encoder to depthwise moves proj's weights by up to 0.062. That is larger than the
        # effects being measured, and with one run per arm it is not controlled for -- which
        # is why `dwident` cannot currently be compared against the baseline.
        # These three arms pin proj's initialisation to a single value so that the baseline
        # and the depthwise designs differ only by their encoders. They are comparable to
        # EACH OTHER; the historical runs used the natural (unpinned) initialisation and are
        # therefore a separate reference.
        #   pinbase     : untouched recipe (random frozen encoders), proj pinned
        #   pindw       : teacher encoder removed + depthwise student encoder, proj pinned
        #   pindwident  : as pindw but with identity encoders, proj pinned. Because the
        #                 encoders are then the identity function, pinbase vs pindwident
        #                 isolates the encoder *form* while pindw vs pindwident isolates the
        #                 encoder *initialisation*, each with proj held fixed.
        "pinbase": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
        },
        "pindw": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
        },
        # --- (2)+(3) on top of pindw: size-weighted align + confidence-weighted box term ------
        # Every number below is measured on the D-Fire test split, not assumed.
        #
        # (2) align_size_weight. ``align_loss`` is a plain mean over ASSIGNED ANCHORS, so an
        #     object's gradient share is proportional to its anchor count. On the stride-32 grid
        #     that is 2.9% for small objects vs 75.5% for large, while the teacher's headroom is
        #     0.046 small vs 0.018 large -- the gradient is spent where the teacher has least to
        #     give. w_i = clip((median_side/side_i)**1, 0.25, 4), mass-normalised by
        #     sum(w*v)/sum(w) so the loss scale (and therefore `align_box`) needs no retuning.
        #     Measured effect on the small-object gradient share: 2.9% -> 14.4% (stride 32) and
        #     7.4% -> 24.5% (stride 16). A weaker exponent barely moves it (p=0.5 -> 7.3%).
        #
        # (3) align_box_conf_weight. The class term is already soft in `tconf` (a 0.05-confidence
        #     box yields a near-uniform target), but the BOX term is an unweighted mean, so such a
        #     box pulls the student's box at full strength. Measured: the band admitted by
        #     lowering the threshold carries 9.7 false positives per new true positive. Weighting
        #     the box term by tconf makes a target's contribution proportional to the teacher's
        #     own confidence in it. The class term is deliberately NOT weighted again.
        #
        # distill_conf_thres stays at the recipe value (0.25) on purpose: (3) removes the need to
        # lower it, so this arm changes exactly two things and both are attributable.
        #
        # Caveat carried from the analysis: 57.2% of small objects receive ZERO assigned anchor
        # at stride 32 (20.3% at stride 16). No weighting can reach those, which is why this arm
        # is run on a tap student and why it is expected to be a <=1 pp effect, not a rescue.
        "pindwsz": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_size_weight": True,
            "align_size_weight_exp": 1.0,
            "align_size_weight_min": 0.25,
            "align_size_weight_max": 4.0,
            "align_box_conf_weight": True,
        },
        # --- pindwsz with the dictionary pairing moved to the SHAPE-MATCHED layer pair ------------
        # Every earlier D-Fire arm pairs teacher x6 (128ch, stride 16, 40x40) with student n10
        # (256ch, stride 32, 20x20): two axes mismatched at once, which forces the projector to
        # INVENT a 2x spatial resolution and to bridge a 2:1 channel change. Measured shapes:
        #     T[6] = 128ch 40x40 stride 16        S[6] = 128ch 40x40 stride 16
        # so S[6] <-> T[6] shares channel count, spatial size AND stride. The projector degrades to
        # a plain depthwise 3x3 with no resampling, and the matching compares two features that live
        # in the same space.
        #
        # Why this is the right experiment (motivated by CanKD, WACV 2026): CanKD's cross-attention
        # correspondence sits on the SPATIAL axis and it hard-asserts y_s.shape == y_t.shape, i.e. it
        # never has to bridge a shape mismatch; it also COARSENS the teacher with maxpool rather than
        # expanding the student. Our channel-axis correspondence has no observable anchor (measured:
        # label-aware reproducibility 0.271, below a constant central box at 0.307), so the question
        # this arm isolates is whether the failure is the CHANNEL axis itself or the SHAPE MISMATCH
        # that every previous arm confounded with it.
        #
        # `dict_student_encoder="depthwise"` (not `depthwise_upsample`): no resampling is needed at
        # equal spatial size, and keeping the plain depthwise form avoids the learnable transposed
        # conv entirely, so the residual is computed with no upsampling branch at all.
        # `dict_match_pool="avg"` is kept (the shipped tokenisation): CanKD's own softmax-free
        # attention is not adopted here, so this arm changes the PAIRING and nothing else about the
        # matching machinery -- that keeps it a single-variable delta against `pindwsz`.
        "pindwsz_s6": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_student_layer": 6,
            "dict_teacher_layers": [6],
            "align_size_weight": True,
            "align_size_weight_exp": 1.0,
            "align_size_weight_min": 0.25,
            "align_size_weight_max": 4.0,
            "align_box_conf_weight": True,
        },
        "pindwdelay40": {
            # pindw, with response align delayed from epoch 20 to 40.
            # dict_attn stays on from epoch 0, the same single change as delay40.
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_start_epoch": 40,
        },
        "pindwhardgain": {
            # pindw dictionary plus the hardgain response weights.
            # Align still starts at epoch 20; dict_attn stays on from epoch 0.
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_loss": 0.16,
            "align_box": 3.0,
            "align_cls": 6.0,
            "feature_loss": 0.10,
        },
        "pindwident": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_match_init": "identity",
        },
        # May the student FEATURE be frozen inside the dictionary loss? `dict_detach_student_tap`
        # is that switch: it decides whether the dictionary losses train the student backbone
        # that produced the tap or only the dictionary parameters. The teacher side is frozen
        # either way. `dw` vs `dwdetach` is the clean pair; `pindw` vs `pindwdetach` repeats it
        # with the projection initialisation pinned.
        "pindwdetach": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_detach_student_tap": True,
        },
        # Dictionary branch only, the counterpart of `nodict`: the two generic terms are off, so
        # the only live distillation is the matching-based pair (`dict_align_loss` +
        # `dict_attn_loss`). Two keys differ from `pindw`, which keeps the contrast
        # single-variable in both directions. The TAL alignment is still computed from
        # `align_start_epoch` while `align=True`, but at weight zero it carries no gradient.
        "pindwonly": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_loss": 0.0,
            "feature_loss": 0.0,
        },
        # --- dictionary-ONLY, carrying the dictionary-side changes that measured best ---------
        # `pindwonly` alone reached +0.0027 val95 over solo (99% of epochs positive) -- a real but
        # small effect. The question these arms answer is whether the dictionary branch does better
        # when it also carries the two implementation changes that measured as consistent net wins:
        #   uniform weighting            (saliency dLdx weighting measured +0.0060/+0.0042/+0.0046/
        #                                 +0.0025 when replaced by uniform, 91% of epochs)
        #   finer matching grid (10x10)  (measured +0.0054/+0.0039/+0.0022/+0.0020, 83% of epochs)
        # The design keeps `pindwonly` as the common reference, so each arm adds exactly one idea
        # and `pindwonlyures` is the best-case combination:
        #     pindwonly      saliency + coarse grid      (already measured)
        #     pindwonlyu     uniform  + coarse grid
        #     pindwonlyres   saliency + fine grid
        #     pindwonlyures  uniform  + fine grid
        #     dictonlyu      recipe encoders + uniform   -> isolates the ARCHITECTURE question with
        #                    the weighting held uniform, since `dictonlyu` vs `pindwonlyu` then
        #                    differs only in teacher-encoder removal + depthwise student encoder.
        "pindwonlyu": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_loss": 0.0,
            "feature_loss": 0.0,
            "dict_weight": "none",
        },
        "pindwonlyres": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_loss": 0.0,
            "feature_loss": 0.0,
            "dict_match_grid_divisor": 4,
        },
        "pindwonlyures": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "align_loss": 0.0,
            "feature_loss": 0.0,
            "dict_weight": "none",
            "dict_match_grid_divisor": 4,
        },
        # Recipe encoders (teacher-side present, full-conv student) with the dictionary branch as the
        # ONLY distillation, weighting held uniform. Deliberately sets no architecture key.
        "dictonlyu": {
            **_DICT_ST_DIAG,
            "align_loss": 0.0,
            "feature_loss": 0.0,
            "dict_weight": "none",
        },
        # Dictionary module with depthwise conv + BN + activation to map the student's spatial
        # size onto the teacher's, no average-pool downsampling, and no teacher-side network:
        #   dict_teacher_encoder=False        the teacher side carries no network
        #   dict_student_encoder="depthwise_upsample"
        #                                     depthwise conv + BN + ReLU, stacked, so the
        #                                     student's 20x20 is resampled by the conv
        #   dict_match_pool="none"            no average pooling, so the attention matrix is
        #                                     computed on the full maps (d = 40*40 = 1600)
        #
        # Two bases:
        #   pindwnp      all distillation branches on
        #   pindwonlynp  dictionary-only
        "pindwnp": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise_upsample",
            "dict_match_pool": "none",
        },
        "pindwonlynp": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise_upsample",
            "align_loss": 0.0,
            "feature_loss": 0.0,
            "dict_match_pool": "none",
        },
        # Correspondence-accuracy arms. They run with `proj_form=identity`, which is
        # parameter-free and channel-local, so the assignment decides what each student
        # channel is aligned to. The three differ only in the map file; `oraclemap` and
        # `shufmap` share a target multiset, so any gap comes from the correspondence.
        #   oraclemap : a prior-built map
        #   shufmap   : the same map with the student channels permuted
        #   randmap   : iid uniform targets (different multiset; loose control)
        # Map path and sha are filled by the queue so the bytes used are recorded in the log.
        "oraclemap": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
        },
        "shufmap": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
        },
        "randmap": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
        },
        # --- delayed / delayed+reduced versions of the fixed-map arms -------------------
        # oraclemap-r1 ran with the identity projection and a high-accuracy fixed map, and is
        # persistently behind the reference trio at every epoch sampled: the gap is -0.0043
        # over e1-20, widest at -0.0083 over e21-40, then narrows to -0.0033 by e101-120.
        # A measured property of that configuration explains a large early constraint: the
        # identity projection passes the align gradient to the student tap about 4.7x more
        # strongly than the shipped DeconvNet does at the SAME alpha (2.16 vs 0.46, stable
        # across four projection seeds). So the arm is not merely "a different correspondence",
        # it is also roughly five times harder on the backbone, and alpha=0.12 was tuned under
        # the DeconvNet form.
        #
        # These arms separate "too strong, too early" from "the correspondence does not help":
        #   *delay     hold the whole dictionary branch (align + AT + commit + InfoMax) off
        #              until dict_start_epoch, so the backbone develops before the added
        #              constraint appears. `_dict_active()` gates the entire branch, so nothing
        #              from the dictionary reaches the student until then. 40 is chosen because
        #              the widest deficit sits in e21-40.
        #   *delaylow  the same delay AND dict_align_loss halved to 0.06, which is the setting
        #              that recovered identinit in the identlowalign arm.
        # Each is defined for the oracle map AND its shuffled counterpart, because the delay
        # and the correspondence are otherwise confounded: only the (delay, delay-shuffled)
        # pair isolates whether an accurate correspondence helps once the schedule is sane.
        "oraclemapdelay": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
            "dict_start_epoch": _DICT_DELAY_EPOCH,
        },
        "shufmapdelay": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
            "dict_start_epoch": _DICT_DELAY_EPOCH,
        },
        "oraclemapdelaylow": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
            "dict_start_epoch": _DICT_DELAY_EPOCH,
            "dict_align_loss": 0.06,
        },
        "shufmapdelaylow": {
            **_DICT_ST_DIAG,
            "dict_match": "fixed",
            "dict_proj_form": "identity",
            "dict_match_init": "identity",
            "dict_start_epoch": _DICT_DELAY_EPOCH,
            "dict_align_loss": 0.06,
        },
        # Diagnostic arm: keep the baseline recipe (both encoders present, full conv,
        # default init) but replace the projected-token argmax with a data-adaptive
        # matcher that correlates per-channel activation series across the batch. The
        # only moved variable is therefore matching quality: if a near-perfect matcher
        # still does not beat basefix, the matching axis is not the lever.
        "oracle": {
            **_DICT_ST_DIAG,
            "dict_match": "batch_corr",
            "dict_freeze_encoders": True,
        },
        "dw": dict(_DICT_DW),
        "dwpw": dict(_DICT_DWPW),
        "dwdetach": {**_DICT_DW, "dict_detach_student_tap": True},
        "dwst": dict(_DICT_DW_ST),
        "dwpwst": {**_DICT_DWPW, **_DICT_ST_MODS},
        "dwstfrz": {**_DICT_DW_ST, "dict_freeze_encoders": True},
    }
)

# ----------------------------------------------------------------------------------
# Match-sweep arms: the dose-response test of "does correspondence ACCURACY help?".
#
# The shipped projection (DeconvNet) mixes all student channels into every output channel,
# so permuting the correspondence is absorbed by the projection and the pairing cannot
# influence the objective. `channel_local` is a depthwise, identity-initialised projection
# whose output channel c is structurally tied to student channel c, so the correspondence
# survives into the loss.
#
# The map comes from --dict-fixed-map; every arm uses a map with the SAME target histogram
# so the only variable is the fraction of correctly paired student channels
# (q = 0.00 / 0.50 / 1.00 built by tools_exp/build_match_maps_v2.py).
#
#   msq000 / msq050 / msq100 : q = 0 / 0.5 / 1; alpha at the channel-local recovery value
#   msctrl                   : q = 1 with alpha = 1.0 -- a positive control proving the
#                              branch can move the metric at all (the sweep is only
#                              informative if this arm is visibly harmed).
# ----------------------------------------------------------------------------------
_MSWEEP_BASE: dict[str, Any] = {
    **_DICT_ST_DIAG,
    "dict_match": "fixed",
    "dict_proj_form": "channel_local",
    "dict_proj_kernel": 3,
    "dict_match_init": "identity",
    "dict_freeze_encoders": False,
    "dict_commit_loss": 0.0,
    "dict_align_loss": 0.06,
}
_DFIRE_KD_VARIANTS.update(
    {
        "msq000": dict(_MSWEEP_BASE),
        "msq050": dict(_MSWEEP_BASE),
        "msq100": dict(_MSWEEP_BASE),
        "msctrl": {**_MSWEEP_BASE, "dict_align_loss": 1.0},
    }
)


def _apply_kd_variant(cfg: dict[str, Any], dataset: str, variant: str) -> None:
    variant = _VARIANT_ALIASES.get(str(variant or "").strip(), str(variant or "").strip())
    if not variant:
        return
    if variant in _VOC_KD_VARIANTS and dataset != "voc2007":
        raise SystemExit("VOC KD variant %s is voc2007-only" % variant)
    if dataset == "voc2007":
        if variant not in _VOC_KD_VARIANTS:
            raise ValueError("Unknown VOC KD variant: %s" % variant)
        cfg.update(_VOC_KD_VARIANTS[variant])
        return
    if dataset == "dfire":
        if variant not in _DFIRE_KD_VARIANTS:
            raise ValueError("Unknown D-Fire KD variant: %s" % variant)
        cfg.update(_DFIRE_KD_VARIANTS[variant])
        return
    raise ValueError("KD variants are only defined for dfire and voc2007")

# Dictionary-structure ablations. These keys live only in this fork's cfg, so they
# must be listed here or ``_yolo_overrides`` silently drops them (no error, no effect).
# ``dict_match_init`` / ``dict_match_grid_divisor`` are included because the identity
# and token-resolution arms set them per variant.
_DICT_STRUCTURE_KEYS = frozenset(
    (
        "dict_teacher_encoder",
        "dict_student_encoder",
        "dict_freeze_encoders",
        "dict_freeze_teacher_encoder",
        "dict_freeze_student_encoder",
        "dict_detach_student_tap",
        "dict_match_log_interval",
        "dict_match_init",
        "dict_match_grid_divisor",
        "dict_match_pool",
        "dict_attn_consistent",
        "dict_init_seed",
            "dict_proj_form",
            "dict_proj_kernel",
            "dict_fixed_map",
            "dict_fixed_map_sha256",
    )
)

# --- Shipped SSM recipe ("ssm") ------------------------------------------------------------
# Slice assignment at full student resolution: the student side is resampled with depthwise
# convolutions (groups == channels) and the teacher side stays raw, so pi is a cosine-similarity
# softmax over every teacher location instead of a pooled grid.
#
# "straight_through" keeps the argmax assignment in the forward pass while leaving the soft pi
# visible to the InfoMax term; the plain "hard" mode computes that term outside its branch, so
# the projector would get no gradient there.
#
# The spatial weight is the first-order channel mean of the teacher feature
# ("activation_mean"), not the second-order mean of squares used by "attention". The attention
# (AT) and cosine-commitment terms are off: both were tuned on the older saliency recipes and
# neither has a role here. The residual projector is channel-local (depthwise 3x3, identity
# init) because a channel-mixing DeconvNet absorbs the slice correspondence, which is the one
# thing the matching term is supposed to constrain.
_SSM: dict[str, Any] = {
    "dict_teacher_encoder": False,
    "dict_student_encoder": "depthwise_upsample",
    "dict_match_pool": "none",
    "dict_match": "straight_through",
    "dict_weight": "activation_mean",
    "dict_proj_form": "channel_local",
    "dict_align_loss": 0.1,
    "dict_infomax_loss": 0.1,
    "dict_infomax_marginal_weight": 0.0,
    "dict_attn_loss": 0.0,
    "dict_commit_loss": 0.0,
    "dict_init_seed": _DICT_INIT_SEED,
    # Diagnostics only, no effect on the loss.
    "dict_match_log_interval": 100,
}

# ---------------------------------------------------------------------------------------------
# Student/teacher tap pair. `dict_student_layer` selects the student feature; the teacher tap is
# set by the arm. The default pair (student n10, 256ch/20x20 at stride 32 -> teacher x6,
# 128ch/40x40 at stride 16) differs in both shape and stride, so the projector has to invent the
# correspondence. Pairing the same stage on both sides (n6 -> x6) removes that freedom; the `_s6`
# arms below exist to measure the difference.
# ---------------------------------------------------------------------------------------------
_SSM_SIZED: dict[str, Any] = {
    **_SSM,
    "align_size_weight": True,
    "align_size_weight_exp": 1.0,
    "align_size_weight_min": 0.25,
    "align_size_weight_max": 4.0,
    "align_box_conf_weight": True,
}

_DFIRE_KD_VARIANTS.update(
    {
        "ssm": dict(_SSM),
        "ssm_sz": dict(_SSM_SIZED),
        "ssm_sz_s6": {**_SSM_SIZED, "dict_student_layer": 6},
        # Soft assignment: the align target is the pi-weighted mean of teacher slices rather than
        # the single argmax slice, so every teacher location contributes.
        "ssm_soft_s6": {**_SSM_SIZED, "dict_match": "soft", "dict_student_layer": 6},
    }
)

# Names used by the archived queues, kept so old launch lines still resolve.
_VARIANT_ALIASES: dict[str, str] = {
    "paperssm": "ssm",
    "paperssm_sz": "ssm_sz",
    "paperssm_sz_s6": "ssm_sz_s6",
    "paperssm_soft_s6": "ssm_soft_s6",
}


# Decision-level align weighting (the `*sz` variants). These keys live only in
# this fork's cfg, so they must be listed here or ``_yolo_overrides`` silently drops them (no
# error, no effect). Kept separate from the dictionary keys because they act on
# `_alignment_loss`, not on the dictionary branch.
_ALIGN_WEIGHT_KEYS = frozenset(
    (
        "align_size_weight",
        "align_size_weight_exp",
        "align_size_weight_min",
        "align_size_weight_max",
        "align_box_conf_weight",
    )
)

# `dict_infomax_loss` and `dict_infomax_marginal_weight` are not in `_KD_RECIPE`, so without this
# they would be dropped by `_yolo_overrides` and the InfoMax term would silently stay at zero.
_SSM_EXTRA_KEYS = frozenset(
    (
        "dict_infomax_loss",
        "dict_infomax_marginal_weight",
    )
)

_KD_EXTRA_KEYS = (
    frozenset(_KD_RECIPE)
    | frozenset(_VOC_KD_OVERRIDES)
    | _DICT_STRUCTURE_KEYS
    | _ALIGN_WEIGHT_KEYS
    | _SSM_EXTRA_KEYS
    | frozenset(("teacher_weights", "teacher_freeze_epoch"))
)


_REQUIRED_CFG = (
    "fixed_accumulate",
    "nbs",
    "warmup_epochs",
    "warmup_bias_lr",
    "optimizer",
    "mosaic",
    "amp",
)


def _protocol_overrides() -> dict[str, Any]:
    return dict(CONTROLLED_ARGS)


def resolve_voc2007_yaml() -> Path:
    """Prefer the yaml written next to the converted images (path: .)."""
    candidates = [
        Path(r"E:\CRIS\VOC数据集\yolo\voc2007.yaml"),
        Path("/root/datasets/VOC2007_yolo/voc2007.yaml"),
        ROOT / "datasets" / "VOC2007_yolo" / "voc2007.yaml",
        ROOT / "ultralytics" / "cfg" / "datasets" / "voc2007.yaml",
    ]
    with_images = [
        path
        for path in candidates
        if path.is_file() and (path.parent / "trainval" / "images").is_dir()
    ]
    if with_images:
        return _absolute_dataset_yaml(with_images[0])
    for path in candidates:
        if path.is_file():
            return _absolute_dataset_yaml(path)
    return candidates[-1]


def _absolute_dataset_yaml(yaml_path: Path) -> Path:
    """Rewrite ``path: .`` to the yaml directory.

    Ultralytics ``check_det_dataset`` treats ``path: .`` as the process cwd,
    not as the folder that contains the yaml. Training is launched from
    ``/root/Ultra``, so a dump yaml with ``path: .`` looks for
    ``/root/Ultra/trainval/images`` and fails.
    """
    from ultralytics.utils import YAML

    yaml_path = Path(yaml_path)
    data = YAML.load(str(yaml_path))
    parent = yaml_path.parent.resolve()
    raw = data.get("path", ".")
    if raw in (None, "", "."):
        abs_root = parent
    else:
        candidate = Path(str(raw))
        abs_root = candidate if candidate.is_absolute() else (parent / candidate).resolve()
        if not abs_root.exists():
            abs_root = parent
    if Path(str(data.get("path", ""))).resolve() == abs_root:
        return yaml_path
    data["path"] = str(abs_root)
    dest = Path(tempfile.gettempdir()) / "voc2007.abs.yaml"
    YAML.save(str(dest), data)
    return dest


def _dataset_spec(args: argparse.Namespace) -> dict[str, Any]:
    dataset = getattr(args, "dataset", "dfire")
    if dataset not in DATASETS:
        raise ValueError(
            "Unknown dataset %s; choose %s" % (dataset, ", ".join(DATASETS))
        )
    spec = dict(DATASETS[dataset])
    if dataset == "voc2007":
        spec["yaml"] = resolve_voc2007_yaml()
    return spec


def build_overrides(baseline: str, args: argparse.Namespace) -> dict[str, Any]:
    if baseline not in BASELINES:
        raise ValueError(
            "Unknown baseline %s; choose %s" % (baseline, ", ".join(BASELINES))
        )
    spec = BASELINES[baseline]
    ds = _dataset_spec(args)
    cfg = _protocol_overrides()
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    elif ds.get("epochs") is not None:
        cfg["epochs"] = int(ds["epochs"])
    run_name = "%s-%s-%de" % (ds["prefix"], spec["name"], int(cfg["epochs"]))
    cfg.update(
        {
            "task": "detect",
            "mode": "train",
            "model": spec["model"],
            "data": str(ds["yaml"]),
            "project": args.project or ds["project"],
            "name": run_name,
            "exist_ok": False,
            "pretrained": args.pretrained or "yolo26n.pt",
            "verbose": True,
            "plots": False,
            "seed": 0,
            "device": 0 if args.device is None else args.device,
        }
    )
    if spec.get("trainer") == "kd":
        cfg.update(_KD_RECIPE)
        if args.dataset == "voc2007":
            cfg.update(_VOC_KD_OVERRIDES)
        _apply_kd_variant(cfg, args.dataset, str(getattr(args, "kd_variant", "") or ""))
        # The fixed-map arms differ ONLY in the map file, so the path and its recorded
        # sha256 are injected after the variant. Both are logged for provenance, and the
        # trainer refuses to run if the file's hash does not match.
        fixed_map = str(getattr(args, "dict_fixed_map", "") or "")
        if fixed_map:
            cfg["dict_fixed_map"] = fixed_map
            cfg["dict_fixed_map_sha256"] = str(
                getattr(args, "dict_fixed_map_sha256", "") or ""
            ).strip().lower()
        cfg["teacher_freeze_epoch"] = int(cfg["epochs"])
        pretrained = str(cfg.get("pretrained") or "yolo26n.pt")
        requested = str(getattr(args, "teacher_weights", "") or "")
        teacher_init = requested or pretrained
        from dfire_parity.protocol import assert_online_teacher_init

        try:
            assert_online_teacher_init(teacher_init)
        except ValueError as exc:
            raise SystemExit(
                "Online KD trains the teacher from COCO init on the current "
                "dataset (same start as YOLO26n solo), not from a finetuned "
                "checkpoint. Omit --teacher-weights or pass the same COCO "
                "file as --pretrained. Got: %s" % teacher_init
            ) from exc
        cfg["teacher_weights"] = teacher_init
    if args.batch is not None:
        cfg["batch"] = int(args.batch)
        cfg["nbs"] = int(args.batch)
        # Warmup is a fixed number of updates, so its epoch length follows the batch.
        cfg["warmup_epochs"] = warmup_epochs_for(int(args.batch))
    if getattr(args, "nbs", None) is not None:
        # Decouple the nominal batch size from the gradient batch. Ultralytics scales the
        # effective weight decay as wd * batch * accumulate / nbs, so this is the supported way
        # to change regularisation strength alone. Applied AFTER --batch so that omitting it
        # reproduces the old behaviour exactly (nbs == batch, the controlled protocol).
        cfg["nbs"] = int(args.nbs)
    if args.workers is not None:
        cfg["workers"] = int(args.workers)
    if args.name_suffix:
        cfg["name"] = "%s-%s" % (cfg["name"], args.name_suffix)
    if args.resume:
        cfg["resume"] = True
        if args.weights:
            cfg["model"] = args.weights
    return cfg


def assert_cfg_supported() -> None:
    """Fail fast if this Ultra tree is older than CONTROLLED_ARGS."""
    import ultralytics
    from ultralytics.utils import DEFAULT_CFG_DICT

    print("ultralytics:", ultralytics.__file__)
    missing = [key for key in _REQUIRED_CFG if key not in DEFAULT_CFG_DICT]
    if not missing:
        return
    raise SystemExit(
        "This Ultra tree is missing CONTROLLED_ARGS keys: %s\n"
        "You copied scripts/train_dfs.py onto an older ultralytics. Sync the current fork, at least:\n"
        "  ultralytics/cfg/default.yaml\n"
        "  ultralytics/engine/trainer.py\n"
        "  dfire_parity/\n"
        "  scripts/train_dfs.py\n"
        "Do not strip these keys: the protocol fork needs them."
        % missing
    )


def _yolo_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep DEFAULT_CFG plus the fork's KD extras; drop leftover classify fields."""
    from ultralytics.utils import DEFAULT_CFG_DICT

    return {
        key: value
        for key, value in cfg.items()
        if key in DEFAULT_CFG_DICT or key in _KD_EXTRA_KEYS
    }


def weights_path(project: str, name: str, task: str = "detect") -> Path:
    """Match Ultralytics get_save_dir for a relative project name."""
    project_path = Path(project)
    if project_path.is_absolute():
        return project_path / name / "weights" / "best.pt"
    return Path("runs") / task / project / name / "weights" / "best.pt"


def train_one(baseline: str, args: argparse.Namespace) -> Path:
    ds = _dataset_spec(args)
    data_yaml = ds["yaml"]
    if not data_yaml.is_file():
        raise SystemExit("Missing %s" % data_yaml)
    spec = BASELINES[baseline]
    overrides = build_overrides(baseline, args)
    print("dataset:", args.dataset, ds["description"], "nc=%d" % ds["nc"])
    print("baseline:", baseline, spec["description"])
    print("model:", overrides["model"])
    print("data:", overrides["data"])
    print("pretrained:", overrides["pretrained"])
    print("epochs:", overrides["epochs"], "close_mosaic:", overrides["close_mosaic"])
    print(
        "protocol: CONTROLLED_ARGS (SGD, batch/nbs=%d, mosaic=1.0, amp=True, seed=0)"
        % overrides["batch"]
    )
    if spec.get("trainer") == "kd":
        print("teacher:", overrides.get("teacher"), overrides.get("teacher_weights"))
        print(
            "kd: variant=%s dict_align=%s, dict_attn=%s@%s, "
            "align_start=%s, align_loss=%s, align_box=%s, feature=%s, freeze_epoch=%s"
            % (
                getattr(args, "kd_variant", "") or "default",
                overrides.get("dict_align_loss"),
                overrides.get("dict_attn_loss"),
                overrides.get("dict_attn_start_epoch"),
                overrides.get("align_start_epoch"),
                overrides.get("align_loss"),
                overrides.get("align_box"),
                overrides.get("feature_loss"),
                overrides.get("teacher_freeze_epoch"),
            )
        )
        print(
            "online teacher: trains on this dataset from the init above "
            "(not a frozen in-dataset expert)"
        )
    if args.dataset == "dfire":
        print(
            "D-Fire protocol run. Healthy val mAP50 is ~0.70+; "
            "a stall near 0.40 usually means the stack, not the data."
        )
    assert_cfg_supported()
    trainer_cls = (
        YOLOFDistillationTrainer if spec.get("trainer") == "kd" else DetectionTrainer
    )
    trainer = trainer_cls(overrides=_yolo_overrides(overrides))
    trainer.train()
    best = trainer.save_dir / "weights" / "best.pt"
    print("best.pt ->", best)
    return best


def test_one(weights: Path, args: argparse.Namespace) -> None:
    overrides = build_overrides(args.baseline, args)
    model = YOLO(str(weights))
    model.val(
        data=overrides["data"],
        split="test",
        imgsz=overrides["imgsz"],
        batch=args.batch or overrides["batch"],
        device=overrides["device"],
        workers=args.workers if args.workers is not None else overrides["workers"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CONTROLLED_ARGS YOLO26n / YOLOF-DCN training (D-Fire or VOC2007)"
    )
    parser.add_argument("--baseline", default="dcn-solo", choices=list(BASELINES))
    parser.add_argument(
        "--dataset",
        default="dfire",
        choices=list(DATASETS),
        help="dfire or voc2007 (VOC2007 trainval/test, nc=20)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--nbs",
        type=int,
        default=None,
        help=(
            "Nominal batch size, decoupled from --batch. Ultralytics scales the effective "
            "weight decay as wd * batch * accumulate / nbs, so --nbs changes the regularisation "
            "strength without touching the gradient batch. Omit to keep the controlled protocol "
            "value (nbs == batch)."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        default="yolo26n.pt",
        help="COCO init weights (default: yolo26n.pt). Do not pass an in-dataset finetuned teacher.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Ultralytics project folder (default: dfire-protocol-baselines or voc2007-baselines)",
    )
    parser.add_argument("--name-suffix", default="")
    parser.add_argument(
        "--teacher-weights",
        default="",
        help=(
            "Teacher init for dcn-kd. Omit this so the teacher starts from the "
            "same COCO yolo26n.pt as --pretrained and trains on the current "
            "dataset. Do not pass a finished in-dataset teacher checkpoint."
        ),
    )
    parser.add_argument(
        "--kd-variant",
        dest="kd_variant",
        default="",
        help="Optional dcn-kd overlay: VOC align50/boxlate/dict12 or D-Fire delay50/attn50/softgain/"
             "dictfirst/latehard/identinit/.../oraclemap/shufmap/randmap.",
    )
    parser.add_argument(
        "--dict-fixed-map",
        dest="dict_fixed_map",
        default="",
        help="Path to a .npy/.txt teacher-index map (one entry per student channel) used by "
             "the fixed-assignment arms. The exact bytes are identified by --dict-fixed-map-sha256.",
    )
    parser.add_argument(
        "--dict-fixed-map-sha256",
        dest="dict_fixed_map_sha256",
        default="",
        help="sha256 of --dict-fixed-map as recorded by the map builder; the trainer aborts "
             "if the file does not match.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weights", default="")
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_only:
        if args.weights:
            weights = Path(args.weights)
        else:
            overrides = build_overrides(args.baseline, args)
            weights = weights_path(str(overrides["project"]), str(overrides["name"]))
        if not weights.is_file():
            raise SystemExit("Missing weights: %s" % weights)
        test_one(weights, args)
        return
    train_one(args.baseline, args)


if __name__ == "__main__":
    main()
