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

from dfire_parity.protocol import CONTROLLED_ARGS
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

# Dictionary-structure ablations (advisor instruction): drop the teacher-side
# dictionary encoder entirely, and make the student-side encoder a MobileNetV2
# depthwise conv. Everything else stays on the D_dldx_new_2 recipe so the only
# moved variable is the dictionary geometry.
#
# Note: under hard matching the encoders receive no gradient (argmax is not
# differentiable), so "depthwise + hard" is a fixed random depthwise projection.
# The ``*st`` arms switch to straight_through (hard forward, soft backward, i.e.
# Figure 2's assignment is preserved) AND enable the commitment loss, because
# ``target = t_reorg.detach()`` means alignment alone cannot train Q/K.
#
# ``basefix`` is the untouched baseline recipe; it exists so the batch has a
# reference measured under the same (freeze-corrected) code as the ablations.
_DICT_ST_DIAG = {"dict_match_log_interval": 100}

# Pinned initialisation for the trainable dictionary projection (see the `pinbase` arm).
# One value for the whole clean set, so every arm in it starts from the same projection.
_DICT_INIT_SEED = 1234

# VOC flavour of the advisor's architecture: the same sliced-matching dictionary as `pindw`
# (teacher-side encoder removed, channel-local depthwise student encoder, pinned projection)
# with the VOC recipe, which is applied automatically for `--dataset voc2007` before the
# variant is layered on.
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
        # The advisor's instruction executed literally, but with the encoders left
        # UNFROZEN: teacher-side dictionary module removed, student-side encoder changed
        # to a MobileNetV2 depthwise conv, hard matching kept (proposal eq. 3 is an
        # argmax), and no freezing.
        #
        # Note on what "unfrozen" means here: under hard matching the argmax is not
        # differentiable, so the encoders receive no gradient and their weights cannot
        # change either way. The only real effect of not freezing is that their
        # BatchNorm stays in train mode, so its running statistics track the data. That
        # is exactly the condition the historical r1-r3 baselines ran under (their
        # freeze call was silently undone by the trainer), which makes this arm both the
        # faithful reading of the instruction and a same-condition reference for them.
        "dwnf": {
            **_DICT_ST_DIAG,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_freeze_encoders": False,
        },
        # --- component ablations for the narrative -------------------------------
        # The paper claims four contributions (sliced matching, spatial-energy AT, FPN
        # projection, delayed TAL) but only the dictionary has been probed. These three
        # isolate the load-bearing claims, one moved variable each:
        #   indexmatch : matching replaced by a fixed per-channel alignment
        #                (student channel s <- teacher channel s mod Ct), everything else
        #                identical to the recipe. Answers "does the argmax matching beat
        #                naive channel alignment?" -- i.e. whether "matching" earns its
        #                place in the title.
        #   nodict     : alpha = beta = 0, so BOTH the sliced weighted align and the
        #                spatial-energy AT term are off (both live in _dictionary_losses);
        #                feature/response branches untouched. Answers "is the dictionary
        #                branch necessary at all?"
        #   unifw      : dict_weight=none, i.e. the same mass-normalised reduction with a
        #                uniform weight instead of S = mean_c|dLt/dxe|. The saliency pass
        #                is skipped entirely (no extra teacher backward). Answers "does
        #                the |dLt/dxe| weighting matter?" -- the phrase in contribution 2.
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
        # --- is the identity-init harm a tuning artefact, or the implementation? --------
        # identinit made the align term fit better (dict_loss 1.703 -> 1.59/1.65) while the
        # AT residual roughly doubled (0.174 -> 0.31/0.33), and a prior probe showed the
        # matching accuracy itself improves a lot (random two independent projections give
        # chance-level top-1 0.76% vs 0.78%; identity reaches 21.4%). Yet identinit scores
        # ~0.003 lower on matched-epoch val curves. These three arms separate the possible
        # causes, one moved variable each, all built on identinit:
        #   identlowalign : halve dict_align_loss (0.12 -> 0.06). Tests over-regularisation
        #                   by the align term, whose effective force rose with the better fit.
        #   identlowattn  : halve dict_attn_loss (0.25 -> 0.12). Same idea for the AT term.
        #   identattnfix  : keep both weights, but compute the AT residual consistently
        #                   (raw student energy map vs raw teacher energy map) instead of the
        #                   shipped raw-vs-standardised mismatch. Isolates the asymmetry.
        # A positive result in any of these means the effect is tunable/fixable rather than
        # a property of the matching criterion.
        "identlowalign": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_align_loss": 0.06,
        },
        # --- "turn the align term off" arms: the cleanest test of whether the matching-based
        # term is net harmful, and whether an identity-init deficit is a weight problem.
        # The alignment term is the ONLY place the correspondence enters the loss: the AT term
        # averages over channels and is therefore invariant to the pairing, and commit/InfoMax
        # are zero in this recipe. So switching `dict_align_loss` to 0 removes the matching
        # contribution entirely while leaving the rest of the dictionary branch (AT, the
        # saliency pass, the module itself) in place -- which is what separates this pair from
        # `nodict`, that turns off align AND AT and therefore builds no modules at all.
        #
        #   noalign       align 0.0 with the recipe's default encoders: a ONE-KEY ablation
        #                 against the reference trio (which sits at 0.12), so the align term's
        #                 net effect is measured without moving anything else.
        #   identnoalign  align 0.0 with identity encoders. Together with identinit (0.12) and
        #                 identlowalign (0.06) this completes an alpha sweep at FIXED identity
        #                 init. All three share the trio's encoder geometry, hence the same
        #                 `proj` initialisation, so the sweep is a clean single-variable one:
        #                 monotone improvement with lower alpha means the deficit was
        #                 over-regularisation, a flat or worse curve at zero means the identity
        #                 configuration itself is the problem.
        "noalign": {
            **_DICT_ST_DIAG,
            "dict_align_loss": 0.0,
        },
        "identnoalign": {
            **_DICT_ST_DIAG,
            "dict_match_init": "identity",
            "dict_align_loss": 0.0,
        },
        # --- the advisor's architecture with the alpha that actually works ----------------
        # The advisor asked for the teacher-side dictionary module removed and the student-side
        # encoder changed to a depthwise convolution. Those arms exist (`dwident`, `dwnf`) but run
        # at the recipe's alpha=0.12, and at that weight the identity family is behind the
        # baseline. The configuration that recovers is `identlowalign` (identity init, alpha 0.06)
        # but it keeps BOTH encoders in full-conv form, so it does not answer the advisor.
        #
        # These two arms combine the two: identity init, alpha 0.06, teacher encoder removed,
        # depthwise student encoder. Measured, this is free -- with identity initialisation the
        # encoder IS the identity function whatever its conv type, so the reassembled teacher
        # target is bit-identical to the both-full-conv version and only the parameter count
        # changes (dictionary module: 1,000,448 -> 265,216, a 3.8x reduction). With a RANDOM init
        # the depthwise form instead gives a materially worse assignment (top-1 0.023 vs 0.070),
        # which is why this must not be conflated with `dw`/`dwnf`.
        #
        # The pair differs in exactly one thing, so that the report to the advisor cannot be
        # undermined by a hidden confound: removing the teacher encoder and resizing the student
        # encoder changes how much RNG the encoders consume before `proj` is built, which re-rolls
        # `proj` -- the only freely-learnable dictionary parameter. One arm pins that
        # initialisation (comparable to `identlowalign` and the trio), the other leaves it natural
        # (what a normal run would do).
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
        "pindwident": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_match_init": "identity",
        },
        # --- the advisor's other question: may the STUDENT FEATURE be frozen? --------------
        # The advisor's instruction has two separable parts. Removing the teacher-side dictionary
        # branch (`pindw`) settles the first. The second asks whether, inside the dictionary LOSS,
        # the student's FEATURE may be detached -- i.e. whether the dictionary losses should be
        # allowed to train the student backbone that produced the tap, or only the dictionary
        # parameters. `dict_detach_student_tap` is exactly that switch, and the teacher side is
        # frozen either way (the teacher network is requires_grad_(False), and every teacher tensor
        # entering the dictionary loss is detached).
        # The clean single-key pair is `dw` vs `dwdetach` (both unpinned), and `pindw` vs
        # `pindwdetach` repeats it with the projection initialisation pinned, so the answer can be
        # checked against the confound documented at the top of the `pinbase`/`pindw` block.
        "pindwdetach": {
            **_DICT_ST_DIAG,
            "dict_init_seed": _DICT_INIT_SEED,
            "dict_teacher_encoder": False,
            "dict_student_encoder": "depthwise",
            "dict_detach_student_tap": True,
        },
        # --- dictionary branch ONLY, on the advisor's architecture --------------------------
        # `nodict` (above) switches the dictionary branch off and answers "is it needed at all".
        # This is its exact counterpart: the two generic terms are switched off instead, so the
        # ONLY live distillation is the matching-based dictionary pair (`dict_align_loss` +
        # `dict_attn_loss`). Exactly two keys off `pindw`, which makes the contrast single-variable
        # in both directions -- against `pindw` (everything on) and against `nodict` (dictionary
        # off). Note the TAL alignment is still COMPUTED from `align_start_epoch` while
        # `align=True`, but its weight is zero so it carries no gradient; setting `align=False`
        # would remove that cost at the price of moving a third key and spoiling the comparison.
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
        # --- the advisor's specification, implemented literally --------------------------------
        # Stated in full: "in the dictionary module we use ONLY depthwise conv + BN + activation to
        # map the student's spatial size onto the teacher's, with NO average-pool downsampling; the
        # teacher side has no network at all; then, as before, compute the attention matrix between
        # the depthwise-convolved student features and the teacher features."
        #
        # Mapped onto this codebase, that is three things:
        #   dict_teacher_encoder=False        the teacher side carries no network
        #   dict_student_encoder="depthwise_upsample"
        #                                     depthwise conv (groups == channels) + BN + ReLU, stacked
        #                                     so the student's 20x20 is resampled onto the teacher's
        #                                     40x40 BY THE CONV, not by interpolation
        #   dict_match_pool="none"            no average pooling anywhere; the attention matrix is
        #                                     computed on the full maps, so d = 40*40 = 1600
        #
        # Note this supersedes an earlier, looser reading in which the student kept its stride-1
        # depthwise conv and the spatial change was done with a bilinear `F.interpolate`. That matched
        # "no average pool" but not "map the spatial size with depthwise conv + BN + activation".
        #
        # Two bases, because the question is asked of two different things:
        #   pindwnp      the advisor architecture with all branches on -- the paper candidate
        #   pindwonlynp  dictionary-only, i.e. the line currently under investigation
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
        # --- does an ACCURATE correspondence help? (the fixed-map experiment) -----------
        # The shipped projection is `ConvTranspose2d(Cs, Cs, k=2, s=2)`: every output channel
        # carries its own weights over all student channels, and output pixel (2i+a, 2j+b)
        # depends only on input pixel (i, j). Reassigning the target of student channel s is
        # therefore a relabelling of independent sub-problems. Measured by fitting the
        # projection to different target maps, the per-channel loss multiset for a map and
        # for that map with the student channels permuted agreed to 0.0006, while changing
        # which teacher channels are used at all was worth 0.04 -- so the pairing cannot
        # influence the objective while that projection is in place, however accurate it is.
        # These arms remove that escape route: `proj_form=identity` is parameter-free and
        # channel-local, so the assignment decides what each student channel is aligned to.
        # With that in place the pairing is worth 13.4% of the align loss (measured), which
        # is what makes an accuracy comparison meaningful. The three arms differ ONLY in the
        # map file; `oraclemap` and `shufmap` have byte-identical target multisets, so any
        # gap is attributable to the correspondence itself.
        #   oraclemap : a prior-built map with high measured out-of-sample reproducibility
        #   shufmap   : the same map with the student channels permuted (multiset identical)
        #   randmap   : iid uniform targets (different multiset; loose control)
        # Map path/sha are filled by the queue from the map-builder's manifest so the exact
        # bytes used are recorded in the run log.
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
    variant = str(variant or "").strip()
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

_KD_EXTRA_KEYS = (
    frozenset(_KD_RECIPE)
    | frozenset(_VOC_KD_OVERRIDES)
    | _DICT_STRUCTURE_KEYS
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
        "protocol: CONTROLLED_ARGS (SGD, batch/nbs=112, mosaic=1.0, amp=True, seed=0)"
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
