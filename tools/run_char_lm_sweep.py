#!/usr/bin/env python3
"""Run reproducible Rust char-LM model-zoo sweeps.

The script intentionally stays dependency-free: it shells out to the existing
Rust examples, captures per-run logs, writes a machine-readable sweep manifest,
and renders Markdown/JSON comparison artifacts with tools/compare_char_lm_runs.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from backend_sweep_meta import classify_failure, md_cell, returncode_label, write_failure


REPO_ROOT = Path(__file__).resolve().parents[1]

_TRAINER_TRACE_SUMMARIZER = None

EXAMPLES = {
    "finetune": "modelzoo_llm_char_finetune",
    "lstm": "modelzoo_llm_char_finetune",
    "scan": "modelzoo_llm_char_coherence_scan",
    "wave": "modelzoo_llm_char_coherence_wave",
}

DEFAULT_ARCHITECTURES = "finetune,scan,wave"
DEFAULT_FEATURES = "token-bigram"
DEFAULT_HEAD_PRIORS = "learned-unigram"
DEFAULT_SEEDS = "42"
DEFAULT_BACKEND = "cpu"
COMPARE_SUMMARY_COMMAND_SCRIPT = "compare_summary_command.sh"

PRESETS = {
    "smoke": {
        "epochs": 1,
        "batches": 2,
        "batch": 4,
        "eval_samples": 16,
        "gen": 16,
        "early_stop_patience": 0,
    },
    "small": {
        "epochs": 3,
        "batches": 8,
        "batch": 4,
        "eval_samples": 64,
        "gen": 64,
        "early_stop_patience": 2,
    },
    "base": {
        "epochs": 6,
        "batches": 24,
        "batch": 8,
        "eval_samples": 256,
        "gen": 200,
        "early_stop_patience": 3,
    },
}

RANK_MIN_PROMOTION_RECIPE_FAIL_DECISIONS = (
    "no_rank_min_evidence,needs_tuning,partial_promote_needs_tuning"
)
ROUTE_DEBT_PROMOTION_RECIPE_FAIL_DECISIONS = "no_route_debt_recommendation"

RECIPES = {
    "none": {
        "description": "do not apply a recipe",
        "defaults": {},
    },
    "guarded-lstm": {
        "description": (
            "reproduce the guarded bigram sweet-spot comparison for SpiralRNN "
            "versus LSTM"
        ),
        "defaults": {
            "architectures": "finetune,lstm",
            "features": "token-bigram",
            "head_priors": "bigram",
            "seeds": "7,11",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 8,
            "head_residual_scale_values": "2",
            "bigram_topk_guard_values": "0,0.1",
            "bigram_topk_guard_k": 5,
            "epoch_values": "1,2",
            "batches_values": "8,16",
            "eval_samples": 64,
            "gen": 8,
        },
    },
    "val-start-hardness": {
        "description": (
            "probe whether a near-full validation slice remains hard across "
            "corpus positions"
        ),
        "defaults": {
            "architectures": "finetune,lstm",
            "features": "token-bigram",
            "head_priors": "learned-unigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-bigram-prior": {
        "description": (
            "replay the hardest validation-start slice across unigram and "
            "bigram head priors"
        ),
        "defaults": {
            "architectures": "finetune,lstm",
            "features": "token-bigram",
            "head_priors": "learned-unigram,bigram,learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-bigram-guard": {
        "description": (
            "probe whether a light bigram top-k guard and rank guard preserve "
            "ordering on the hardest validation-start slice"
        ),
        "defaults": {
            "architectures": "finetune,lstm",
            "features": "token-bigram",
            "head_priors": "bigram,learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_values": "0,0.1",
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard_values": "0,0.05",
            "bigram_rank_guard_margin": 0,
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "no-prior-context-pressure": {
        "description": (
            "compare LSTM, SpiralRNN, scan, and wave under a no-prior "
            "high-residual learning-pressure window"
        ),
        "defaults": {
            "architectures": "finetune,lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7,13",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "2,5",
            "epoch_values": "3",
            "batches_values": "8",
            "batch": 4,
            "eval_samples": 64,
            "lr_values": "0.05",
            "compare_summary_limit": 24,
            "gen": 64,
        },
    },
    "no-prior-coherence-budget": {
        "description": (
            "focus scan and wave on the no-prior pressure window with a "
            "longer cheap-training budget"
        ),
        "defaults": {
            "architectures": "scan,wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7,13",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "epoch_values": "8,16",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 4,
            "lr_values": "0.05",
            "compare_summary_limit": 16,
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 64,
        },
    },
    "no-prior-coherence-shape": {
        "description": (
            "quickly probe scan context scaling and wave kernel/dilation "
            "shape knobs inside the no-prior coherence pressure window"
        ),
        "defaults": {
            "architectures": "scan,wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "context_scale_values": "0.5,1,2",
            "query_residual_scale_values": "0.5,1,2",
            "wave_kernel_values": "3,5",
            "wave_dilation_values": "1,2,4;1,2,4,8",
            "epoch_values": "3",
            "batches_values": "8",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 2,
            "lr_values": "0.05",
            "compare_summary_limit": 20,
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 0,
        },
    },
    "no-prior-coherence-shape-confirm": {
        "description": (
            "confirm promising scan/wave coherence shape knobs with the "
            "longer no-prior coherence budget"
        ),
        "defaults": {
            "architectures": "scan,wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7,13",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "context_scale_values": "0.5,1,2",
            "query_residual_scale_values": "0.5,1,2",
            "wave_kernel_values": "3,5",
            "wave_dilation_values": "1,2,4;1,2,4,8",
            "epoch_values": "8",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 4,
            "lr_values": "0.05",
            "compare_summary_limit": 20,
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 64,
        },
    },
    "no-prior-coherence-shape-winners": {
        "description": (
            "confirm the quick-probe scan winner and promoted lite wave shape "
            "with the longer no-prior coherence budget"
        ),
        "defaults": {
            "architectures": "scan,wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7,13",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "context_scale": 2,
            "query_residual_scale": 2,
            "wave_kernel": 3,
            "wave_dilations": "1",
            "epoch_values": "8,16",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 4,
            "lr_values": "0.05",
            "compare_summary_limit": 12,
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 64,
        },
    },
    "no-prior-coherence-wave-promoted": {
        "description": (
            "rerun the promoted lite wave shape after route-debt evidence "
            "selects the single-dilation branch"
        ),
        "defaults": {
            "architectures": "wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7,13,23",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "query_residual_scale": 2,
            "wave_kernel": 3,
            "wave_dilations": "1",
            "epoch_values": "10",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 4,
            "lr_values": "0.05",
            "compare_summary_limit": 12,
            "compare_summary_sort_metric": "coherence_route_debt",
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 96,
        },
    },
    "no-prior-coherence-wave-lite": {
        "description": (
            "quickly probe whether fewer wave dilation branches preserve "
            "no-prior coherence quality while lowering route debt"
        ),
        "defaults": {
            "architectures": "wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "query_residual_scale": 2,
            "wave_kernel": 3,
            "wave_dilation_values": "1;1,2;1,2,4",
            "epoch_values": "3",
            "batches_values": "8",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 2,
            "lr_values": "0.05",
            "compare_summary_limit": 12,
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 0,
        },
    },
    "no-prior-coherence-wave-lite-confirm": {
        "description": (
            "confirm the wave-lite branch-count tradeoff with the longer "
            "no-prior coherence budget"
        ),
        "defaults": {
            "architectures": "wave",
            "features": "token-bigram",
            "head_priors": "none",
            "seeds": "7,13",
            "steps": 32,
            "embed_dim": 32,
            "hidden": 64,
            "memory": 16,
            "head_residual_scale_values": "5",
            "query_residual_scale": 2,
            "wave_kernel": 3,
            "wave_dilation_values": "1;1,2;1,2,4",
            "epoch_values": "8",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 64,
            "early_stop_patience": 4,
            "lr_values": "0.05",
            "compare_summary_limit": 12,
            "compare_summary_sort_metric": "coherence_route_debt",
            "compare_summary_fail_on_route_debt_decision": (
                ROUTE_DEBT_PROMOTION_RECIPE_FAIL_DECISIONS
            ),
            "extra_arg": ["--mix-rms", "1.0"],
            "gen": 64,
        },
    },
    "hard-rank-guard-local": {
        "description": (
            "run a focused local rank-debt guard search on the hardest "
            "learned-bigram LSTM slice"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard_values": "0,0.05,0.1,0.5",
            "bigram_rank_guard_margin_values": "0,0.05",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-guard-confirm": {
        "description": (
            "confirm the local rank-debt guard sweet spot across seeds on the "
            "hardest learned-bigram LSTM slice"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard_values": "0,0.05,0.1",
            "bigram_rank_guard_margin": 0.05,
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-guard-instability-map": {
        "description": (
            "map whether rank-debt guard stability changes across validation "
            "start positions"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard_values": "0,0.05,0.1",
            "bigram_rank_guard_margin": 0.05,
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-soft-guard-local": {
        "description": (
            "probe a softer full-bigram distribution guard on the hard "
            "learned-bigram LSTM slice without pairwise rank hinge pressure"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0,
            "bigram_rank_guard_margin": 0.05,
            "bigram_soft_guard_values": "0,0.01,0.05,0.1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-soft-guard-micro-local": {
        "description": (
            "probe very small full-bigram distribution guard weights after "
            "the first soft guard sweep regressed rank debt"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0,
            "bigram_rank_guard_margin": 0.05,
            "bigram_soft_guard_values": "0,0.001,0.003,0.005,0.01",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-band-local": {
        "description": (
            "probe local bigram-rank competitor bands on the hard "
            "learned-bigram LSTM slice"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band_values": "0,0.001,0.003,0.005,0.01",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-local": {
        "description": (
            "probe adaptive minimum competitor fill for narrow bigram-rank "
            "bands on the hard learned-bigram LSTM slice"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1,2,3",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-confirm": {
        "description": (
            "confirm adaptive bigram-rank minimum competitor fill across "
            "seeds and validation-start slices on the hard learned-bigram "
            "LSTM setup"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1,2,3",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe": {
        "description": (
            "compare the stable min=1 adaptive bigram-rank fill candidate "
            "against the fixed-band baseline on the hard learned-bigram LSTM "
            "validation-start sweep"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe-arches": {
        "description": (
            "widen the stable min=1 adaptive bigram-rank fill candidate from "
            "LSTM to scan and wave char-LM architectures"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe-corpus": {
        "description": (
            "run the stable min=1 adaptive bigram-rank fill promotion gate on "
            "a broader text/corpus input across LSTM, scan, and wave"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe-corpus-frontier": {
        "description": (
            "confirm the corpus val-start=0 frontier where min=1 removed "
            "zero-guard windows but produced seed-mixed top-5 overlap"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.1,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe-corpus-frontier-topk": {
        "description": (
            "probe whether stronger top-k preservation guard recovers the "
            "corpus val-start=0 frontier while keeping min=1 adaptive fill"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,21",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_values": "0.1,0.2,0.5",
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe-corpus-frontier-topk-confirm": {
        "description": (
            "confirm top-k guard 0.5 as the corpus val-start=0 frontier "
            "candidate across the full seed set"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.5,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_fraction": 0,
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-safe-corpus-topk-confirm": {
        "description": (
            "confirm top-k guard 0.2 as the corpus all-validation-start "
            "balance candidate across the full seed set"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard": 0.2,
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-wave-topk-middle": {
        "description": (
            "probe middle top-k guard strengths for the coherence-wave "
            "residual rank-debt cases across all validation starts"
        ),
        "defaults": {
            "architectures": "wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_values": "0.3,0.4",
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-wave-topk-schedule-confirm": {
        "description": (
            "confirm a validation-start-aware top-k guard schedule for "
            "coherence-wave residual rank-debt cases"
        ),
        "defaults": {
            "architectures": "wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_schedule": "0:0.5,0.5:0.2,1:0.3",
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-confirm": {
        "description": (
            "confirm the architecture-aware top-k guard schedule: all-start "
            "0.2 for LSTM/scan with wave validation-start overrides"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 12,
            "embed_dim": 8,
            "hidden": 16,
            "memory": 12,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,wave@0:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "10",
            "batches_values": "24",
            "batch": 4,
            "eval_samples": 128,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-scale": {
        "description": (
            "scale the architecture-aware top-k guard schedule to a larger "
            "char-LM shape and longer training sanity gate"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,wave@0:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-scale-confirm": {
        "description": (
            "confirm the medium-scale architecture-aware top-k guard schedule "
            "on complementary seeds"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "21,34",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,wave@0:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-scale-full": {
        "description": (
            "run the full four-seed medium-scale architecture-aware top-k "
            "guard schedule promotion gate"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,wave@0:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "compare_summary_fail_on_rank_min_promotion_decision": (
                RANK_MIN_PROMOTION_RECIPE_FAIL_DECISIONS
            ),
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen": {
        "description": (
            "apply the medium-scale architecture-aware top-k guard schedule "
            "to a widened positional corpus input"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,lstm@0.5:0.5,"
                "scan@0:0.5,scan@0.5:0.5,wave@0:0.5,wave@0.5:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-confirm": {
        "description": (
            "confirm the widened-corpus medium-scale architecture-aware top-k "
            "guard schedule on complementary seeds"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "21,34",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,lstm@0.5:0.5,"
                "scan@0:0.5,scan@0.5:0.5,wave@0:0.5,wave@0.5:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-full": {
        "description": (
            "run the full four-seed widened-corpus medium-scale "
            "architecture-aware top-k guard promotion gate"
        ),
        "defaults": {
            "architectures": "lstm,scan,wave",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_arch_schedule": (
                "*@0:0.2,*@0.5:0.2,*@1:0.2,lstm@0.5:0.5,"
                "scan@0:0.5,scan@0.5:0.5,wave@0:0.5,wave@0.5:0.5,wave@1:0.3"
            ),
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_values": "0,0.5,1",
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "compare_summary_fail_on_rank_min_promotion_decision": (
                RANK_MIN_PROMOTION_RECIPE_FAIL_DECISIONS
            ),
            "gen": 0,
        },
    },
    "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-hotspot": {
        "description": (
            "probe the widened-corpus LSTM val-start=0.5 hotspot after raw "
            "rank precision exposed mixed min=1 alignment"
        ),
        "defaults": {
            "architectures": "lstm",
            "features": "token-bigram",
            "head_priors": "learned-bigram",
            "seeds": "7,13,21,34",
            "steps": 16,
            "embed_dim": 12,
            "hidden": 24,
            "memory": 16,
            "head_residual_scale_values": "0.5",
            "bigram_topk_guard_values": "0.2,0.3,0.4,0.5",
            "bigram_topk_guard_k": 5,
            "bigram_rank_guard": 0.1,
            "bigram_rank_guard_margin": 0.05,
            "bigram_rank_guard_band": 0.003,
            "bigram_rank_guard_min_candidates_values": "0,1",
            "epoch_values": "12",
            "batches_values": "32",
            "batch": 4,
            "eval_samples": 192,
            "val_start_fraction": 0.5,
            "lr_values": "0.0025",
            "compare_summary_limit": 12,
            "gen": 0,
        },
    },
}


@dataclass(frozen=True)
class SweepSettings:
    epochs: int
    batches: int
    batch: int
    eval_samples: int
    gen: int
    early_stop_patience: int
    steps: int | None
    embed_dim: int | None
    hidden: int | None
    memory: int | None
    lr: float | None
    curvature: float | None
    temperature: float | None
    head_residual_scale: float | None
    context_scale: float | None
    self_score_scale: float | None
    query_residual_scale: float | None
    wave_kernel: int | None
    wave_dilations: str | None
    backend: str
    bigram_topk_guard: float | None = None
    bigram_topk_guard_k: int | None = None
    bigram_rank_guard: float | None = None
    bigram_rank_guard_margin: float | None = None
    bigram_rank_guard_band: float | None = None
    bigram_rank_guard_min_candidates: int | None = None
    bigram_soft_guard: float | None = None
    val_start_fraction: float | None = None


@dataclass(frozen=True)
class CompareSummaryOptions:
    limit: int
    route_clean_only: bool
    prefer_clean_route: bool
    fail_on_route_statuses: tuple[str, ...] = ()
    fail_on_paired_quality_statuses: tuple[str, ...] = ()
    fail_on_efficiency_verdicts: tuple[str, ...] = ()
    fail_on_rank_min_promotion_decisions: tuple[str, ...] = ()
    fail_on_route_debt_decisions: tuple[str, ...] = ()
    extra_compare_paths: tuple[Path, ...] = ()
    merge_evidence_sources: bool = False
    sort_metric: str = "final_nll"


SUMMARY_ROUTE_STATUSES = {
    "clean_route",
    "scan_route_mixed",
    "scan_fallback",
    "scan_route_mismatch",
    "no_scan_route",
}

SUMMARY_SORT_METRICS = {
    "best_nll",
    "cpu_debt",
    "delta_nll",
    "final_bigram_logprob_lift",
    "final_bigram_rank_lift",
    "final_bigram_rank_debt",
    "final_top5_bigram_overlap",
    "final_nll",
    "final_vs_bigram",
    "final_vs_unigram",
    "coherence_route_debt",
    "lstm_cpu_debt",
}

SUMMARY_PAIR_QUALITY_STATUSES = {"improved", "missing", "neutral", "regressed"}

SUMMARY_EFFICIENCY_VERDICTS = {
    "candidate_better_quality_and_cost",
    "candidate_cost_regressed",
    "candidate_neutral",
    "candidate_quality_better_cost_neutral",
    "candidate_quality_neutral_cost_better",
    "candidate_quality_regressed",
    "inconclusive",
}

SUMMARY_RANK_MIN_PROMOTION_DECISIONS = {
    "no_rank_min_evidence",
    "needs_tuning",
    "partial_promote_needs_tuning",
    "promote",
    "promote_with_bounded_watch",
}

SUMMARY_ROUTE_DEBT_DECISIONS = {
    "no_route_debt_recommendation",
    "promote_lite_wave",
}


def parse_csv(raw: str, *, label: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{label} must contain at least one value")
    return values


def parse_csv_int(raw: str, *, label: str) -> list[int]:
    values = []
    for part in parse_csv(raw, label=label):
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid --{label} entry: {part}") from exc
    return values


def parse_csv_float(raw: str, *, label: str) -> list[float]:
    values = []
    for part in parse_csv(raw, label=label):
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"invalid --{label} entry: {part}") from exc
    return values


def grid_values(raw: str | None, fallback: int | None, *, label: str) -> list[int | None]:
    if raw is None:
        return [fallback]
    values = parse_csv_int(raw, label=label)
    if any(value <= 0 for value in values):
        raise ValueError(f"--{label} entries must be positive")
    return values


def count_grid_values(raw: str | None, fallback: int | None, *, label: str) -> list[int | None]:
    if raw is None:
        if fallback is not None and fallback < 0:
            raise ValueError(f"--{label} fallback must be non-negative")
        return [fallback]
    values = parse_csv_int(raw, label=label)
    if any(value < 0 for value in values):
        raise ValueError(f"--{label} entries must be non-negative")
    return values


def float_grid_values(
    raw: str | None, fallback: float | None, *, label: str
) -> list[float | None]:
    if raw is None:
        if fallback is not None and (fallback < 0.0 or not math.isfinite(fallback)):
            raise ValueError(f"--{label} fallback must be a non-negative finite float")
        return [fallback]
    values = parse_csv_float(raw, label=label)
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"--{label} entries must be non-negative finite floats")
    return values


def positive_float_grid_values(
    raw: str | None, fallback: float | None, *, label: str
) -> list[float | None]:
    values = float_grid_values(raw, fallback, label=label)
    if any(value is not None and value <= 0.0 for value in values):
        raise ValueError(f"--{label} entries must be positive finite floats")
    return values


def fraction_grid_values(
    raw: str | None, fallback: float | None, *, label: str
) -> list[float | None]:
    values = float_grid_values(raw, fallback, label=label)
    for value in values:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"--{label} entries must be between 0 and 1")
    return values


def parse_bigram_topk_guard_schedule(raw: str, *, label: str) -> dict[float, float]:
    schedule: dict[float, float] = {}
    for part in parse_csv(raw, label=label):
        if ":" in part:
            start_raw, guard_raw = part.split(":", 1)
        elif "=" in part:
            start_raw, guard_raw = part.split("=", 1)
        else:
            raise ValueError(f"--{label} entries must use START:GUARD")
        try:
            start = float(start_raw)
            guard = float(guard_raw)
        except ValueError as exc:
            raise ValueError(f"invalid --{label} entry: {part}") from exc
        if not math.isfinite(start) or not (0.0 <= start <= 1.0):
            raise ValueError(f"--{label} validation starts must be between 0 and 1")
        if not math.isfinite(guard) or guard < 0.0:
            raise ValueError(f"--{label} guards must be non-negative finite floats")
        if any(math.isclose(start, existing, rel_tol=0.0, abs_tol=1e-9) for existing in schedule):
            raise ValueError(f"duplicate --{label} validation start: {start_raw}")
        schedule[start] = guard
    return schedule


def scheduled_bigram_topk_guard(
    schedule: dict[float, float], val_start: float | None
) -> float:
    if val_start is None:
        raise ValueError("--bigram-topk-guard-schedule requires validation starts")
    for start, guard in schedule.items():
        if math.isclose(start, val_start, rel_tol=0.0, abs_tol=1e-9):
            return guard
    raise ValueError(
        "--bigram-topk-guard-schedule missing guard for "
        f"validation start {val_start:g}"
    )


def schedule_manifest_rows(schedule: dict[float, float]) -> list[dict[str, float]]:
    return [
        {"validation_start_fraction": start, "bigram_topk_guard": guard}
        for start, guard in sorted(schedule.items())
    ]


def parse_bigram_topk_guard_arch_schedule(
    raw: str, *, label: str
) -> dict[tuple[str, float], float]:
    schedule: dict[tuple[str, float], float] = {}
    for part in parse_csv(raw, label=label):
        if ":" not in part or "@" not in part:
            raise ValueError(f"--{label} entries must use ARCH@START:GUARD")
        arch_start_raw, guard_raw = part.split(":", 1)
        architecture_raw, start_raw = arch_start_raw.split("@", 1)
        architecture = architecture_raw.strip()
        if not architecture:
            raise ValueError(f"--{label} architecture names must not be empty")
        try:
            start = float(start_raw)
            guard = float(guard_raw)
        except ValueError as exc:
            raise ValueError(f"invalid --{label} entry: {part}") from exc
        if not math.isfinite(start) or not (0.0 <= start <= 1.0):
            raise ValueError(f"--{label} validation starts must be between 0 and 1")
        if not math.isfinite(guard) or guard < 0.0:
            raise ValueError(f"--{label} guards must be non-negative finite floats")
        duplicate = any(
            existing_arch == architecture
            and math.isclose(start, existing_start, rel_tol=0.0, abs_tol=1e-9)
            for existing_arch, existing_start in schedule
        )
        if duplicate:
            raise ValueError(
                f"duplicate --{label} entry for {architecture}@{start_raw}"
            )
        schedule[(architecture, start)] = guard
    return schedule


def scheduled_bigram_topk_guard_for_arch(
    schedule: dict[tuple[str, float], float],
    architecture: str,
    val_start: float | None,
) -> float:
    if val_start is None:
        raise ValueError("--bigram-topk-guard-arch-schedule requires validation starts")
    for (scheduled_architecture, start), guard in schedule.items():
        if scheduled_architecture != architecture:
            continue
        if math.isclose(start, val_start, rel_tol=0.0, abs_tol=1e-9):
            return guard
    for (scheduled_architecture, start), guard in schedule.items():
        if scheduled_architecture != "*":
            continue
        if math.isclose(start, val_start, rel_tol=0.0, abs_tol=1e-9):
            return guard
    raise ValueError(
        "--bigram-topk-guard-arch-schedule missing guard for "
        f"{architecture}@{val_start:g}"
    )


def arch_schedule_manifest_rows(
    schedule: dict[tuple[str, float], float]
) -> list[dict[str, object]]:
    return [
        {
            "architecture": architecture,
            "validation_start_fraction": start,
            "bigram_topk_guard": guard,
        }
        for (architecture, start), guard in sorted(schedule.items())
    ]


def parse_wave_dilation_values(raw: str | None, fallback: str | None) -> list[str | None]:
    if raw is None:
        return [fallback]
    values = [part.strip() for part in raw.split(";") if part.strip()]
    if not values:
        raise ValueError("--wave-dilation-values must contain at least one dilation spec")
    for spec in values:
        entries = [part.strip() for part in spec.split(",") if part.strip()]
        if not entries:
            raise ValueError("--wave-dilation-values contains an empty dilation spec")
        for entry in entries:
            try:
                value = int(entry)
            except ValueError as exc:
                raise ValueError(f"invalid --wave-dilation-values entry: {entry}") from exc
            if value <= 0:
                raise ValueError("--wave-dilation-values entries must be positive")
    return values


def slug(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("-")
    compact = "".join(safe).strip("-")
    while "--" in compact:
        compact = compact.replace("--", "-")
    return compact or "value"


def default_run_root() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return REPO_ROOT / "models" / "runs" / f"char_lm_sweep_{stamp}"


def grid_slug(label: str, value: int | None) -> str:
    return f"{label}-{slug('default' if value is None else str(value))}"


def float_grid_slug(label: str, value: float | None) -> str:
    return f"{label}-{slug('default' if value is None else f'{value:g}')}"


def apply_recipe_defaults(args: argparse.Namespace) -> argparse.Namespace:
    recipe_name = getattr(args, "recipe", "none")
    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        raise ValueError(f"unknown recipe: {recipe_name}")

    parser_defaults = {
        "architectures": DEFAULT_ARCHITECTURES,
        "features": DEFAULT_FEATURES,
        "head_priors": DEFAULT_HEAD_PRIORS,
        "seeds": DEFAULT_SEEDS,
        "backend": DEFAULT_BACKEND,
        "backends": None,
        "steps": None,
        "embed_dim": None,
        "hidden": None,
        "memory": None,
        "head_residual_scale": None,
        "head_residual_scale_values": None,
        "context_scale": None,
        "context_scale_values": None,
        "self_score_scale": None,
        "self_score_scale_values": None,
        "query_residual_scale": None,
        "query_residual_scale_values": None,
        "wave_kernel": None,
        "wave_kernel_values": None,
        "wave_dilations": None,
        "wave_dilation_values": None,
        "bigram_topk_guard": None,
        "bigram_topk_guard_values": None,
        "bigram_topk_guard_schedule": None,
        "bigram_topk_guard_arch_schedule": None,
        "bigram_topk_guard_k": None,
        "bigram_rank_guard": None,
        "bigram_rank_guard_values": None,
        "bigram_rank_guard_margin": None,
        "bigram_rank_guard_margin_values": None,
        "bigram_rank_guard_band": None,
        "bigram_rank_guard_band_values": None,
        "bigram_rank_guard_min_candidates": None,
        "bigram_rank_guard_min_candidates_values": None,
        "bigram_soft_guard": None,
        "bigram_soft_guard_values": None,
        "val_start_fraction": None,
        "val_start_values": None,
        "epoch_values": None,
        "batches_values": None,
        "eval_samples": None,
        "gen": None,
        "compare_summary_limit": 8,
        "compare_summary_sort_metric": "final_nll",
        "compare_summary_fail_on_rank_min_promotion_decision": None,
        "compare_summary_fail_on_route_debt_decision": None,
        "compare_summary_extra_compare_json": [],
        "compare_summary_merge_evidence_sources": False,
        "extra_arg": [],
    }
    for field, value in recipe["defaults"].items():
        default = parser_defaults.get(field)
        if getattr(args, field) == default:
            setattr(args, field, value)
    return args


def recipe_manifest_metadata(recipe_name: str) -> dict[str, object]:
    recipe = RECIPES.get(recipe_name, RECIPES["none"])
    defaults = recipe.get("defaults", {})
    return {
        "recipe": recipe_name,
        "recipe_description": recipe.get("description", ""),
        "recipe_defaults": dict(defaults) if isinstance(defaults, dict) else {},
    }


def settings_from_args(args: argparse.Namespace) -> SweepSettings:
    preset = PRESETS[args.preset]
    settings = SweepSettings(
        epochs=preset["epochs"],
        batches=preset["batches"],
        batch=preset["batch"],
        eval_samples=preset["eval_samples"],
        val_start_fraction=args.val_start_fraction,
        gen=preset["gen"],
        early_stop_patience=preset["early_stop_patience"],
        steps=args.steps,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        memory=args.memory,
        lr=args.lr,
        curvature=args.curvature,
        temperature=args.temperature,
        head_residual_scale=args.head_residual_scale,
        context_scale=args.context_scale,
        self_score_scale=args.self_score_scale,
        query_residual_scale=args.query_residual_scale,
        wave_kernel=args.wave_kernel,
        wave_dilations=args.wave_dilations,
        bigram_topk_guard=args.bigram_topk_guard,
        bigram_topk_guard_k=args.bigram_topk_guard_k,
        bigram_rank_guard=args.bigram_rank_guard,
        bigram_rank_guard_margin=args.bigram_rank_guard_margin,
        bigram_rank_guard_band=args.bigram_rank_guard_band,
        bigram_rank_guard_min_candidates=args.bigram_rank_guard_min_candidates,
        bigram_soft_guard=args.bigram_soft_guard,
        backend=args.backend,
    )
    for field in ("epochs", "batches", "batch", "eval_samples", "gen", "early_stop_patience"):
        value = getattr(args, field)
        if value is not None:
            settings = replace(settings, **{field: value})
    return settings


def add_optional_int(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def add_optional_float(command: list[str], flag: str, value: float | None) -> None:
    if value is not None:
        command.extend([flag, f"{value:g}"])


def effective_cargo_features(args: argparse.Namespace, backend: str) -> str | None:
    if args.cargo_features is not None:
        return args.cargo_features
    if backend == "wgpu":
        return "wgpu"
    return None


def compare_summary_options_from_args(args: argparse.Namespace) -> CompareSummaryOptions:
    if args.compare_summary_limit < 0:
        raise ValueError("--compare-summary-limit must be non-negative")
    extra_compare_paths = tuple(
        Path(part)
        for raw in (args.compare_summary_extra_compare_json or [])
        for part in parse_csv(
            raw,
            label="compare-summary-extra-compare-json",
        )
    )
    fail_on_route_statuses = tuple(
        parse_csv(
            args.compare_summary_fail_on_route_status,
            label="compare-summary-fail-on-route-status",
        )
        if args.compare_summary_fail_on_route_status
        else []
    )
    invalid_statuses = sorted(set(fail_on_route_statuses) - SUMMARY_ROUTE_STATUSES)
    if invalid_statuses:
        expected = ", ".join(sorted(SUMMARY_ROUTE_STATUSES))
        raise ValueError(
            "invalid --compare-summary-fail-on-route-status: "
            + ", ".join(invalid_statuses)
            + f" (expected {expected})"
        )
    fail_on_paired_quality_statuses = tuple(
        parse_csv(
            args.compare_summary_fail_on_paired_quality_status,
            label="compare-summary-fail-on-paired-quality-status",
        )
        if args.compare_summary_fail_on_paired_quality_status
        else []
    )
    invalid_quality_statuses = sorted(
        set(fail_on_paired_quality_statuses) - SUMMARY_PAIR_QUALITY_STATUSES
    )
    if invalid_quality_statuses:
        expected = ", ".join(sorted(SUMMARY_PAIR_QUALITY_STATUSES))
        raise ValueError(
            "invalid --compare-summary-fail-on-paired-quality-status: "
            + ", ".join(invalid_quality_statuses)
            + f" (expected {expected})"
        )
    fail_on_efficiency_verdicts = tuple(
        parse_csv(
            args.compare_summary_fail_on_efficiency_verdict,
            label="compare-summary-fail-on-efficiency-verdict",
        )
        if args.compare_summary_fail_on_efficiency_verdict
        else []
    )
    invalid_efficiency_verdicts = sorted(
        set(fail_on_efficiency_verdicts) - SUMMARY_EFFICIENCY_VERDICTS
        )
    if invalid_efficiency_verdicts:
        expected = ", ".join(sorted(SUMMARY_EFFICIENCY_VERDICTS))
        raise ValueError(
            "invalid --compare-summary-fail-on-efficiency-verdict: "
            + ", ".join(invalid_efficiency_verdicts)
            + f" (expected {expected})"
        )
    fail_on_rank_min_promotion_decisions = tuple(
        parse_csv(
            args.compare_summary_fail_on_rank_min_promotion_decision,
            label="compare-summary-fail-on-rank-min-promotion-decision",
        )
        if args.compare_summary_fail_on_rank_min_promotion_decision
        else []
    )
    invalid_rank_min_promotion_decisions = sorted(
        set(fail_on_rank_min_promotion_decisions)
        - SUMMARY_RANK_MIN_PROMOTION_DECISIONS
    )
    if invalid_rank_min_promotion_decisions:
        expected = ", ".join(sorted(SUMMARY_RANK_MIN_PROMOTION_DECISIONS))
        raise ValueError(
            "invalid --compare-summary-fail-on-rank-min-promotion-decision: "
            + ", ".join(invalid_rank_min_promotion_decisions)
            + f" (expected {expected})"
        )
    fail_on_route_debt_decisions = tuple(
        parse_csv(
            args.compare_summary_fail_on_route_debt_decision,
            label="compare-summary-fail-on-route-debt-decision",
        )
        if args.compare_summary_fail_on_route_debt_decision
        else []
    )
    invalid_route_debt_decisions = sorted(
        set(fail_on_route_debt_decisions) - SUMMARY_ROUTE_DEBT_DECISIONS
    )
    if invalid_route_debt_decisions:
        expected = ", ".join(sorted(SUMMARY_ROUTE_DEBT_DECISIONS))
        raise ValueError(
            "invalid --compare-summary-fail-on-route-debt-decision: "
            + ", ".join(invalid_route_debt_decisions)
            + f" (expected {expected})"
        )
    if args.compare_summary_sort_metric not in SUMMARY_SORT_METRICS:
        expected = ", ".join(sorted(SUMMARY_SORT_METRICS))
        raise ValueError(
            "invalid --compare-summary-sort-metric: "
            + args.compare_summary_sort_metric
            + f" (expected {expected})"
        )
    return CompareSummaryOptions(
        limit=args.compare_summary_limit,
        route_clean_only=args.compare_summary_route_clean_only,
        prefer_clean_route=args.compare_summary_prefer_clean_route,
        fail_on_route_statuses=fail_on_route_statuses,
        fail_on_paired_quality_statuses=fail_on_paired_quality_statuses,
        fail_on_efficiency_verdicts=fail_on_efficiency_verdicts,
        fail_on_rank_min_promotion_decisions=fail_on_rank_min_promotion_decisions,
        fail_on_route_debt_decisions=fail_on_route_debt_decisions,
        extra_compare_paths=extra_compare_paths,
        merge_evidence_sources=args.compare_summary_merge_evidence_sources,
        sort_metric=args.compare_summary_sort_metric,
    )


def compare_summary_has_fail_gates(options: CompareSummaryOptions) -> bool:
    return bool(
        options.fail_on_route_statuses
        or options.fail_on_paired_quality_statuses
        or options.fail_on_efficiency_verdicts
        or options.fail_on_rank_min_promotion_decisions
        or options.fail_on_route_debt_decisions
    )


def compare_summary_failure_is_fatal(options: CompareSummaryOptions) -> bool:
    return bool(
        compare_summary_has_fail_gates(options)
        or options.extra_compare_paths
        or options.merge_evidence_sources
    )


def build_command(
    *,
    cargo_bin: str,
    cargo_features: str | None,
    no_default_features: bool,
    architecture: str,
    data_paths: list[Path],
    run_dir: Path,
    char_feature: str,
    head_prior: str,
    seed: int,
    settings: SweepSettings,
    extra_args: list[str],
) -> list[str]:
    command = [
        cargo_bin,
        "run",
        "-p",
        "st-nn",
    ]
    if no_default_features:
        command.append("--no-default-features")
    if cargo_features:
        command.extend(["--features", cargo_features])
    command.extend(
        [
            "--example",
            EXAMPLES[architecture],
            "--",
        ]
    )
    command.extend(str(path) for path in data_paths)
    command.extend(
        [
            "--backend",
            settings.backend,
            "--run-dir",
            str(run_dir),
            "--events",
            str(run_dir / "trainer_trace.jsonl"),
            "--head-prior",
            head_prior,
            "--char-feature",
            char_feature,
            "--epochs",
            str(settings.epochs),
            "--batches",
            str(settings.batches),
            "--batch",
            str(settings.batch),
            "--eval-samples",
            str(settings.eval_samples),
            "--early-stop-patience",
            str(settings.early_stop_patience),
            "--gen",
            str(settings.gen),
            "--seed",
            str(seed),
        ]
    )
    add_optional_float(command, "--val-start-fraction", settings.val_start_fraction)
    add_optional_int(command, "--steps", settings.steps)
    add_optional_int(command, "--embed-dim", settings.embed_dim)
    add_optional_int(command, "--hidden", settings.hidden)
    if architecture in {"scan", "wave"}:
        add_optional_int(command, "--memory", settings.memory)
        add_optional_float(command, "--self-score-scale", settings.self_score_scale)
        add_optional_float(command, "--query-residual-scale", settings.query_residual_scale)
    if architecture == "scan":
        add_optional_float(command, "--context-scale", settings.context_scale)
    if architecture == "wave":
        add_optional_int(command, "--kernel", settings.wave_kernel)
        if settings.wave_dilations is not None:
            command.extend(["--dilations", settings.wave_dilations])
    if architecture == "lstm":
        command.extend(["--recurrent", "lstm"])
    add_optional_float(command, "--lr", settings.lr)
    add_optional_float(command, "--curvature", settings.curvature)
    add_optional_float(command, "--temperature", settings.temperature)
    add_optional_float(command, "--head-residual-scale", settings.head_residual_scale)
    add_optional_float(command, "--bigram-topk-guard", settings.bigram_topk_guard)
    add_optional_int(command, "--bigram-topk-guard-k", settings.bigram_topk_guard_k)
    add_optional_float(command, "--bigram-rank-guard", settings.bigram_rank_guard)
    add_optional_float(
        command,
        "--bigram-rank-guard-margin",
        settings.bigram_rank_guard_margin,
    )
    add_optional_float(
        command,
        "--bigram-rank-guard-band",
        settings.bigram_rank_guard_band,
    )
    add_optional_int(
        command,
        "--bigram-rank-guard-min-candidates",
        settings.bigram_rank_guard_min_candidates,
    )
    add_optional_float(command, "--bigram-soft-guard", settings.bigram_soft_guard)
    command.extend(extra_args)
    return command


def write_preflight_skipped_log(
    log_path: Path, preflight_failure: dict[str, object], failure_kind: str, failure_detail: str
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "skipped after WGPU sweep preflight failure",
                f"preflight_log_path={preflight_failure.get('log_path')}",
                f"failure_kind={failure_kind}",
                f"failure_detail={failure_detail}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, float]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + shlex.join(command) + "\n")
        log_file.flush()
        if dry_run:
            log_file.write("[dry-run] command not executed\n")
            return 0, 0.0
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode, time.time() - started


def read_json(path: Path) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    if isinstance(value, dict):
        return value
    return None


def roundtable_wgpu_statuses(run_payload: dict[str, object] | None) -> list[str]:
    if not isinstance(run_payload, dict):
        return []
    audit = run_payload.get("roundtable_backend_audit")
    if not isinstance(audit, dict):
        return []
    bands = audit.get("bands")
    if not isinstance(bands, list):
        return []
    statuses: list[str] = []
    for band in bands:
        if not isinstance(band, dict):
            continue
        band_name = str(band.get("band", "?"))
        status = band.get("wgpu_exact_status")
        if isinstance(status, str) and status:
            statuses.append(f"{band_name}:{status}")
    return statuses


def roundtable_wgpu_summary(run_payload: dict[str, object] | None) -> dict[str, Any]:
    if not isinstance(run_payload, dict):
        return {}
    audit = run_payload.get("roundtable_backend_audit")
    if not isinstance(audit, dict):
        return {}
    return {
        "requested_backend": audit.get("requested_backend"),
        "wgpu_runtime_compiled": audit.get("wgpu_runtime_compiled"),
        "wgpu_runtime_context_installed": audit.get("wgpu_runtime_context_installed"),
        "any_wgpu_exact_runtime_ready": audit.get("any_wgpu_exact_runtime_ready"),
        "statuses": roundtable_wgpu_statuses(run_payload),
    }


def trainer_trace_summarizer():
    global _TRAINER_TRACE_SUMMARIZER
    if _TRAINER_TRACE_SUMMARIZER is not None:
        return _TRAINER_TRACE_SUMMARIZER
    helper_path = REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "trainer_trace.py"
    if not helper_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_spiraltorch_trainer_trace", helper_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summarizer = getattr(module, "summarize_trainer_trace_events", None)
    if callable(summarizer):
        _TRAINER_TRACE_SUMMARIZER = summarizer
        return summarizer
    return None


def write_trainer_trace_summary(run_dir: Path) -> tuple[Path | None, str | None]:
    summarizer = trainer_trace_summarizer()
    if summarizer is None:
        return None, None
    for name in (
        "trainer_trace.jsonl",
        "spiraltorch_trainer_trace.jsonl",
        "trainer_steps.jsonl",
    ):
        trace_path = run_dir / name
        if not trace_path.exists():
            continue
        try:
            summary = summarizer(trace_path)
        except Exception as exc:
            return None, f"failed to summarize {trace_path.name}: {exc}"
        if not isinstance(summary, dict):
            return None, f"trainer trace summarizer returned {type(summary).__name__}"
        summary_path = run_dir / "trainer_trace_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary_path, None
    return None, None


def render_compare(run_dirs: Iterable[Path], run_root: Path, *, curves: bool) -> str | None:
    run_dirs = list(run_dirs)
    if not run_dirs:
        return None
    compare_script = REPO_ROOT / "tools" / "compare_char_lm_runs.py"
    compare_json_path = run_root / "compare.json"
    command = [
        sys.executable,
        "-S",
        "-s",
        str(compare_script),
        "--aggregate",
        "--json-out",
        str(compare_json_path),
    ]
    if curves:
        command.append("--curves")
    command.extend(str(path) for path in run_dirs)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    compare_path = run_root / "compare.md"
    compare_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        error_path = run_root / "compare.error.log"
        error_path.write_text(result.stderr, encoding="utf-8")
        return None
    return result.stdout


def default_compare_summary_options() -> CompareSummaryOptions:
    return CompareSummaryOptions(
        limit=8,
        route_clean_only=False,
        prefer_clean_route=True,
        fail_on_route_statuses=(),
        fail_on_paired_quality_statuses=(),
        fail_on_efficiency_verdicts=(),
        fail_on_rank_min_promotion_decisions=(),
        fail_on_route_debt_decisions=(),
        extra_compare_paths=(),
        merge_evidence_sources=False,
        sort_metric="final_nll",
    )


def build_compare_summary_command(
    run_root: Path,
    options: CompareSummaryOptions,
) -> list[str]:
    compare_json_path = run_root / "compare.json"
    summary_script = REPO_ROOT / "tools" / "summarize_char_lm_compare.py"
    summary_json_path = run_root / "compare_summary.json"
    command = [
        sys.executable,
        "-S",
        "-s",
        str(summary_script),
        str(compare_json_path),
        *(str(path) for path in options.extra_compare_paths),
        "--limit",
        str(options.limit),
        "--sort-metric",
        options.sort_metric,
        "--json-out",
        str(summary_json_path),
    ]
    if options.merge_evidence_sources:
        command.append("--merge-evidence-sources")
    if options.route_clean_only:
        command.append("--route-clean-only")
    if options.prefer_clean_route:
        command.append("--prefer-clean-route")
    for status in options.fail_on_route_statuses:
        command.extend(["--fail-on-route-status", status])
    for status in options.fail_on_paired_quality_statuses:
        command.extend(["--fail-on-paired-quality-status", status])
    for verdict in options.fail_on_efficiency_verdicts:
        command.extend(["--fail-on-efficiency-verdict", verdict])
    for decision in options.fail_on_rank_min_promotion_decisions:
        command.extend(["--fail-on-rank-min-promotion-decision", decision])
    for decision in options.fail_on_route_debt_decisions:
        command.extend(["--fail-on-route-debt-decision", decision])
    return command


def shell_array_arg(token: str) -> str:
    return shlex.quote(token)


def compare_summary_command_script_path(run_root: Path) -> Path:
    return run_root / COMPARE_SUMMARY_COMMAND_SCRIPT


def write_compare_summary_command_script(run_root: Path, command: list[str]) -> Path:
    path = compare_summary_command_script_path(run_root)
    markdown_path = run_root / "compare_summary.md"
    error_path = run_root / "compare_summary.error.log"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Generated by tools/run_char_lm_sweep.py.",
        "# Replays compare-summary generation for this sweep.",
        f"cd {shell_array_arg(str(REPO_ROOT))}",
        f"markdown_path={shell_array_arg(str(markdown_path))}",
        f"error_path={shell_array_arg(str(error_path))}",
        "",
        "cmd=(",
        *[f"  {shell_array_arg(str(token))}" for token in command],
        ")",
        "",
        "printf 'running:'",
        "printf ' %q' \"${cmd[@]}\"",
        "printf '\\n'",
        "if \"${cmd[@]}\" > \"${markdown_path}\" 2> \"${error_path}\"; then",
        "  if [[ ! -s \"${error_path}\" ]]; then",
        "    rm -f \"${error_path}\"",
        "  fi",
        "  printf 'wrote %s\\n' \"${markdown_path}\"",
        "else",
        "  status=$?",
        "  printf 'compare summary failed; see %s\\n' \"${error_path}\" >&2",
        "  exit \"${status}\"",
        "fi",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def render_compare_summary(
    run_root: Path,
    *,
    options: CompareSummaryOptions | None = None,
) -> str | None:
    compare_json_path = run_root / "compare.json"
    if not compare_json_path.exists():
        return None
    if options is None:
        options = default_compare_summary_options()
    command = build_compare_summary_command(run_root, options)
    write_compare_summary_command_script(run_root, command)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    summary_path = run_root / "compare_summary.md"
    summary_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        error_path = run_root / "compare_summary.error.log"
        error_path.write_text(result.stderr, encoding="utf-8")
        return None
    return result.stdout


def failed_runs_markdown(runs: list[dict[str, object]]) -> str:
    failed_runs = [run for run in runs if run.get("failed")]
    if not failed_runs:
        return ""
    headers = [
        "run",
        "arch",
        "backend",
        "seed",
        "run_status",
        "returncode",
        "failure_kind",
        "failure_detail",
        "log_path",
    ]
    lines = [
        "## Failed Runs",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for run in failed_runs:
        row = [
            str(run.get("name") or "-"),
            str(run.get("architecture") or "-"),
            str(run.get("backend") or "-"),
            str(run.get("seed") or "-"),
            str(run.get("run_status") or "failed"),
            returncode_label(run.get("returncode")),
            str(run.get("failure_kind") or "-"),
            str(run.get("failure_detail") or "-"),
            str(run.get("log_path") or "-"),
        ]
        lines.append("| " + " | ".join(md_cell(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def append_failed_runs_compare(run_root: Path, runs: list[dict[str, object]]) -> str | None:
    failure_markdown = failed_runs_markdown(runs)
    if not failure_markdown:
        return None
    compare_path = run_root / "compare.md"
    existing = compare_path.read_text(encoding="utf-8") if compare_path.exists() else ""
    if existing.strip():
        output = existing.rstrip() + "\n\n" + failure_markdown
    else:
        output = failure_markdown
    compare_path.write_text(output, encoding="utf-8")
    return output


def write_sweep_manifest(run_root: Path, payload: dict[str, object]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_wgpu_preflight(
    args: argparse.Namespace,
    run_root: Path,
    settings: SweepSettings,
    *,
    backend: str,
    architecture: str,
    char_feature: str,
    head_prior: str,
) -> dict[str, object] | None:
    if backend != "wgpu" or args.no_wgpu_preflight:
        return None
    run_dir = run_root / "_preflight" / f"backend-{backend}"
    log_path = run_dir / "process.log"
    preflight_settings = replace(
        settings,
        backend=backend,
        epochs=1,
        batches=1,
        batch=1,
        eval_samples=1,
        gen=0,
        early_stop_patience=0,
        steps=1,
        embed_dim=2,
        hidden=2,
        memory=2,
    )
    command = build_command(
        cargo_bin=args.cargo_bin,
        cargo_features=effective_cargo_features(args, backend),
        no_default_features=args.no_default_features,
        architecture=architecture,
        data_paths=args.data_paths,
        run_dir=run_dir,
        char_feature=char_feature,
        head_prior=head_prior,
        seed=0,
        settings=preflight_settings,
        extra_args=args.extra_arg,
    )
    returncode, elapsed = run_command(command, log_path, dry_run=args.dry_run)
    if returncode == 0:
        return None
    failure_kind, failure_detail = classify_failure(returncode, log_path)
    failure: dict[str, object] = {
        "schema": "st.char_lm_sweep_preflight_failure.v1",
        "backend": backend,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "failure_kind": failure_kind,
        "failure_detail": failure_detail,
        "command": command,
    }
    write_failure(run_dir, failure)
    return failure


def validate_choices(values: list[str], allowed: set[str], *, label: str) -> None:
    invalid = [value for value in values if value not in allowed]
    if invalid:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"invalid --{label}: {', '.join(invalid)} (expected {expected})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Rust char-LM model-zoo sweeps and compare summary artifacts."
    )
    parser.add_argument("data_paths", nargs="+", type=Path, help="text files or corpus directories")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="sweep output directory (default: models/runs/char_lm_sweep_<timestamp>)",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default="none",
        help=(
            "optional sweep recipe for validated char-LM comparison grids"
        ),
    )
    parser.add_argument(
        "--architectures",
        default=DEFAULT_ARCHITECTURES,
        help="comma-separated: finetune,lstm,scan,wave",
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help="comma-separated: token,token-bigram",
    )
    parser.add_argument(
        "--head-priors",
        default=DEFAULT_HEAD_PRIORS,
        help="comma-separated: none,unigram,learned-unigram,bigram,learned-bigram",
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="comma-separated integer seeds")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="auto|wgpu|cuda|hip|cpu")
    parser.add_argument(
        "--backends",
        default=None,
        help="comma-separated backends; overrides --backend when set",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--epoch-values",
        default=None,
        help="comma-separated epoch counts for training-budget grids; overrides --epochs",
    )
    parser.add_argument(
        "--batches-values",
        default=None,
        help="comma-separated batches-per-epoch counts for training-budget grids; overrides --batches",
    )
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument(
        "--eval-samples-values",
        default=None,
        help="comma-separated eval-sample counts for evaluation grids; overrides --eval-samples",
    )
    parser.add_argument(
        "--val-start-fraction",
        type=float,
        default=None,
        help="explicit validation slice start fraction; omitted keeps the historical tail split",
    )
    parser.add_argument(
        "--val-start-values",
        default=None,
        help=(
            "comma-separated validation start fractions for split grids; "
            "overrides --val-start-fraction"
        ),
    )
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument(
        "--step-values",
        default=None,
        help="comma-separated step/window sizes for shape grids; overrides --steps",
    )
    parser.add_argument(
        "--embed-dim-values",
        default=None,
        help="comma-separated embedding dimensions for shape grids; overrides --embed-dim",
    )
    parser.add_argument(
        "--hidden-values",
        default=None,
        help="comma-separated hidden sizes for shape grids; overrides --hidden",
    )
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--lr-values",
        default=None,
        help="comma-separated learning rates for optimizer grids; overrides --lr",
    )
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--head-residual-scale",
        type=float,
        default=None,
        help="scale residual context logits before the head prior",
    )
    parser.add_argument(
        "--head-residual-scale-values",
        default=None,
        help="comma-separated residual-logit scales; overrides --head-residual-scale",
    )
    parser.add_argument(
        "--context-scale",
        type=float,
        default=None,
        help="scan-only context gain passed to --context-scale",
    )
    parser.add_argument(
        "--context-scale-values",
        default=None,
        help="comma-separated scan context gains; overrides --context-scale",
    )
    parser.add_argument(
        "--self-score-scale",
        type=float,
        default=None,
        help="scan/wave self-token coherence score scale",
    )
    parser.add_argument(
        "--self-score-scale-values",
        default=None,
        help="comma-separated self-score scales; overrides --self-score-scale",
    )
    parser.add_argument(
        "--query-residual-scale",
        type=float,
        default=None,
        help="scan/wave residual query contribution scale",
    )
    parser.add_argument(
        "--query-residual-scale-values",
        default=None,
        help="comma-separated query-residual scales; overrides --query-residual-scale",
    )
    parser.add_argument(
        "--wave-kernel",
        type=int,
        default=None,
        help="wave-only convolution kernel size passed to --kernel",
    )
    parser.add_argument(
        "--wave-kernel-values",
        default=None,
        help="comma-separated wave kernel sizes; overrides --wave-kernel",
    )
    parser.add_argument(
        "--wave-dilations",
        default=None,
        help="wave-only dilation spec passed to --dilations, for example 1,2,4",
    )
    parser.add_argument(
        "--wave-dilation-values",
        default=None,
        help=(
            "semicolon-separated wave dilation specs, for example "
            "1,2,4;1,2,4,8; overrides --wave-dilations"
        ),
    )
    parser.add_argument(
        "--bigram-topk-guard",
        type=float,
        default=None,
        help="weight for the bigram top-k preservation loss",
    )
    parser.add_argument(
        "--bigram-topk-guard-values",
        default=None,
        help="comma-separated bigram top-k guard weights; overrides --bigram-topk-guard",
    )
    parser.add_argument(
        "--bigram-topk-guard-schedule",
        default=None,
        help=(
            "comma-separated validation-start:guard pairs, for example "
            "0:0.5,0.5:0.2,1:0.3; overrides the top-k guard grid"
        ),
    )
    parser.add_argument(
        "--bigram-topk-guard-arch-schedule",
        default=None,
        help=(
            "comma-separated ARCH@validation-start:guard pairs with optional "
            "wildcard '*', for example '*@0:0.2,wave@0:0.5'; overrides the "
            "top-k guard grid"
        ),
    )
    parser.add_argument(
        "--bigram-topk-guard-k",
        type=int,
        default=None,
        help="number of previous-token bigram candidates preserved by the guard",
    )
    parser.add_argument(
        "--bigram-rank-guard",
        type=float,
        default=None,
        help="weight for pairwise rank-debt guard against weaker bigram candidates",
    )
    parser.add_argument(
        "--bigram-rank-guard-values",
        default=None,
        help="comma-separated rank-debt guard weights; overrides --bigram-rank-guard",
    )
    parser.add_argument(
        "--bigram-rank-guard-margin",
        type=float,
        default=None,
        help="log-probability margin for the pairwise rank-debt guard",
    )
    parser.add_argument(
        "--bigram-rank-guard-margin-values",
        default=None,
        help=(
            "comma-separated rank-debt guard margins; "
            "overrides --bigram-rank-guard-margin"
        ),
    )
    parser.add_argument(
        "--bigram-rank-guard-band",
        type=float,
        default=None,
        help=(
            "maximum bigram-probability gap for pairwise rank guard competitors; "
            "0 keeps the historical unbounded candidate set"
        ),
    )
    parser.add_argument(
        "--bigram-rank-guard-band-values",
        default=None,
        help=(
            "comma-separated rank-debt guard competitor bands; "
            "overrides --bigram-rank-guard-band"
        ),
    )
    parser.add_argument(
        "--bigram-rank-guard-min-candidates",
        type=int,
        default=None,
        help=(
            "minimum narrow-band rank guard competitors to keep by adaptively "
            "filling from the unbounded bigram set"
        ),
    )
    parser.add_argument(
        "--bigram-rank-guard-min-candidates-values",
        default=None,
        help=(
            "comma-separated adaptive rank-debt competitor minimums; "
            "overrides --bigram-rank-guard-min-candidates"
        ),
    )
    parser.add_argument(
        "--bigram-soft-guard",
        type=float,
        default=None,
        help="weight for a soft full-bigram distribution guard",
    )
    parser.add_argument(
        "--bigram-soft-guard-values",
        default=None,
        help="comma-separated soft bigram guard weights; overrides --bigram-soft-guard",
    )
    parser.add_argument("--cargo-bin", default="cargo")
    parser.add_argument(
        "--cargo-features",
        default=None,
        help="feature list passed to cargo run, for example 'wgpu' or 'wgpu,kdsl'",
    )
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra arg passed to every Rust example",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=None,
        help=(
            "stop after launching this many non-skipped runs; combine with "
            "--skip-existing to resume long sweeps in chunks"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-wgpu-preflight", action="store_true")
    parser.add_argument(
        "--no-print-compare",
        action="store_true",
        help="write compare artifacts but do not print the full compare Markdown table",
    )
    parser.add_argument(
        "--quiet-runs",
        action="store_true",
        help="suppress per-run command logs; final artifact paths are still printed",
    )
    parser.add_argument(
        "--compare-summary-limit",
        type=int,
        default=8,
        help="maximum rows in generated compare_summary artifacts",
    )
    parser.add_argument(
        "--compare-summary-route-clean-only",
        action="store_true",
        help="filter generated compare_summary rows to clean LSTM scan routes",
    )
    parser.add_argument(
        "--compare-summary-extra-compare-json",
        action="append",
        default=[],
        help=(
            "additional compare.json files or sweep directories to include in "
            "generated compare_summary artifacts; may be repeated or comma-separated"
        ),
    )
    parser.add_argument(
        "--compare-summary-merge-evidence-sources",
        action="store_true",
        help=(
            "merge compatible rank-min promotion evidence across compare_summary "
            "inputs"
        ),
    )
    parser.add_argument(
        "--compare-summary-no-prefer-clean-route",
        dest="compare_summary_prefer_clean_route",
        action="store_false",
        help="rank generated compare_summary rows by metrics without route preference",
    )
    parser.add_argument(
        "--compare-summary-fail-on-route-status",
        default=None,
        help="comma-separated route statuses that should fail summary generation",
    )
    parser.add_argument(
        "--compare-summary-fail-on-paired-quality-status",
        default=None,
        help=(
            "comma-separated paired recurrent quality statuses that should fail "
            "summary generation"
        ),
    )
    parser.add_argument(
        "--compare-summary-fail-on-efficiency-verdict",
        default=None,
        help=(
            "comma-separated paired recurrent efficiency verdicts that should fail "
            "summary generation"
        ),
    )
    parser.add_argument(
        "--compare-summary-fail-on-rank-min-promotion-decision",
        default=None,
        help=(
            "comma-separated rank-min promotion gate decisions that should fail "
            "summary generation"
        ),
    )
    parser.add_argument(
        "--compare-summary-fail-on-route-debt-decision",
        default=None,
        help=(
            "comma-separated route-debt recommendation decisions that should fail "
            "summary generation"
        ),
    )
    parser.add_argument(
        "--compare-summary-sort-metric",
        default="final_nll",
        help=(
            "primary metric for generated compare_summary ranking: final_nll, "
            "best_nll, delta_nll, final_vs_unigram, final_vs_bigram, "
            "final_bigram_logprob_lift, final_bigram_rank_lift, "
            "final_bigram_rank_debt, final_top5_bigram_overlap, "
            "coherence_route_debt, cpu_debt, or lstm_cpu_debt"
        ),
    )
    parser.set_defaults(compare_summary_prefer_clean_route=True)
    parser.add_argument("--curves", action="store_true", help="include epoch curves in compare.md")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        args = apply_recipe_defaults(args)
        architectures = parse_csv(args.architectures, label="architectures")
        features = parse_csv(args.features, label="features")
        head_priors = parse_csv(args.head_priors, label="head-priors")
        backends = parse_csv(
            args.backends if args.backends is not None else args.backend,
            label="backends" if args.backends is not None else "backend",
        )
        seeds = parse_csv_int(args.seeds, label="seeds")
        settings_preview = settings_from_args(args)
        step_values = grid_values(args.step_values, settings_preview.steps, label="step-values")
        embed_dim_values = grid_values(
            args.embed_dim_values,
            settings_preview.embed_dim,
            label="embed-dim-values",
        )
        hidden_values = grid_values(
            args.hidden_values,
            settings_preview.hidden,
            label="hidden-values",
        )
        epoch_values = grid_values(
            args.epoch_values,
            settings_preview.epochs,
            label="epoch-values",
        )
        batches_values = grid_values(
            args.batches_values,
            settings_preview.batches,
            label="batches-values",
        )
        eval_samples_values = grid_values(
            args.eval_samples_values,
            settings_preview.eval_samples,
            label="eval-samples-values",
        )
        val_start_values = fraction_grid_values(
            args.val_start_values,
            settings_preview.val_start_fraction,
            label="val-start-values",
        )
        lr_values = float_grid_values(
            args.lr_values,
            settings_preview.lr,
            label="lr-values",
        )
        head_residual_scale_values = float_grid_values(
            args.head_residual_scale_values,
            settings_preview.head_residual_scale,
            label="head-residual-scale-values",
        )
        context_scale_values = positive_float_grid_values(
            args.context_scale_values,
            settings_preview.context_scale,
            label="context-scale-values",
        )
        self_score_scale_values = float_grid_values(
            args.self_score_scale_values,
            settings_preview.self_score_scale,
            label="self-score-scale-values",
        )
        query_residual_scale_values = float_grid_values(
            args.query_residual_scale_values,
            settings_preview.query_residual_scale,
            label="query-residual-scale-values",
        )
        wave_kernel_values = grid_values(
            args.wave_kernel_values,
            settings_preview.wave_kernel,
            label="wave-kernel-values",
        )
        wave_dilation_values = parse_wave_dilation_values(
            args.wave_dilation_values,
            settings_preview.wave_dilations,
        )
        bigram_topk_guard_arch_schedule = (
            parse_bigram_topk_guard_arch_schedule(
                args.bigram_topk_guard_arch_schedule,
                label="bigram-topk-guard-arch-schedule",
            )
            if args.bigram_topk_guard_arch_schedule is not None
            else None
        )
        bigram_topk_guard_schedule = None
        if bigram_topk_guard_arch_schedule is not None:
            if (
                args.bigram_topk_guard is not None
                or args.bigram_topk_guard_values is not None
                or args.bigram_topk_guard_schedule is not None
            ):
                raise ValueError(
                    "--bigram-topk-guard-arch-schedule cannot be combined with "
                    "--bigram-topk-guard, --bigram-topk-guard-values, or "
                    "--bigram-topk-guard-schedule"
                )
            scheduled_architectures = {
                architecture
                for architecture, _start in bigram_topk_guard_arch_schedule
                if architecture != "*"
            }
            unknown_architectures = sorted(scheduled_architectures - set(EXAMPLES))
            if unknown_architectures:
                raise ValueError(
                    "--bigram-topk-guard-arch-schedule uses unknown "
                    f"architectures: {','.join(unknown_architectures)}"
                )
            for architecture in architectures:
                for val_start in val_start_values:
                    scheduled_bigram_topk_guard_for_arch(
                        bigram_topk_guard_arch_schedule,
                        architecture,
                        val_start,
                    )
            bigram_topk_guard_values = [None]
        elif args.bigram_topk_guard_schedule is not None:
            bigram_topk_guard_schedule = parse_bigram_topk_guard_schedule(
                args.bigram_topk_guard_schedule,
                label="bigram-topk-guard-schedule",
            )
            if args.bigram_topk_guard is not None or args.bigram_topk_guard_values is not None:
                raise ValueError(
                    "--bigram-topk-guard-schedule cannot be combined with "
                    "--bigram-topk-guard or --bigram-topk-guard-values"
                )
            for val_start in val_start_values:
                scheduled_bigram_topk_guard(bigram_topk_guard_schedule, val_start)
            bigram_topk_guard_values = [None]
        else:
            bigram_topk_guard_values = float_grid_values(
                args.bigram_topk_guard_values,
                settings_preview.bigram_topk_guard,
                label="bigram-topk-guard-values",
            )
        bigram_rank_guard_values = float_grid_values(
            args.bigram_rank_guard_values,
            settings_preview.bigram_rank_guard,
            label="bigram-rank-guard-values",
        )
        bigram_rank_guard_margin_values = float_grid_values(
            args.bigram_rank_guard_margin_values,
            settings_preview.bigram_rank_guard_margin,
            label="bigram-rank-guard-margin-values",
        )
        bigram_rank_guard_band_values = float_grid_values(
            args.bigram_rank_guard_band_values,
            settings_preview.bigram_rank_guard_band,
            label="bigram-rank-guard-band-values",
        )
        bigram_rank_guard_min_candidates_values = count_grid_values(
            args.bigram_rank_guard_min_candidates_values,
            settings_preview.bigram_rank_guard_min_candidates,
            label="bigram-rank-guard-min-candidates-values",
        )
        bigram_soft_guard_values = float_grid_values(
            args.bigram_soft_guard_values,
            settings_preview.bigram_soft_guard,
            label="bigram-soft-guard-values",
        )
        if args.bigram_topk_guard_k is not None and args.bigram_topk_guard_k <= 0:
            raise ValueError("--bigram-topk-guard-k must be positive")
        if args.bigram_topk_guard_k is not None:
            too_large = [
                value
                for value in bigram_rank_guard_min_candidates_values
                if value is not None and value > args.bigram_topk_guard_k
            ]
            if too_large:
                raise ValueError(
                    "--bigram-rank-guard-min-candidates entries must be <= "
                    "--bigram-topk-guard-k"
                )
        compare_summary_options = compare_summary_options_from_args(args)
        if args.max_new_runs is not None and args.max_new_runs < 0:
            raise ValueError("--max-new-runs must be non-negative")
        validate_choices(architectures, set(EXAMPLES), label="architectures")
        validate_choices(features, {"token", "token-bigram"}, label="features")
        validate_choices(
            head_priors,
            {"none", "unigram", "learned-unigram", "bigram", "learned-bigram"},
            label="head-priors",
        )
        validate_choices(backends, {"auto", "wgpu", "cuda", "hip", "cpu"}, label="backends")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    settings = settings_preview
    shape_grid_is_explicit = any(
        raw is not None
        for raw in (args.step_values, args.embed_dim_values, args.hidden_values)
    )
    head_residual_scale_is_explicit = (
        args.head_residual_scale is not None or args.head_residual_scale_values is not None
    )
    context_scale_is_explicit = (
        args.context_scale is not None or args.context_scale_values is not None
    )
    self_score_scale_is_explicit = (
        args.self_score_scale is not None or args.self_score_scale_values is not None
    )
    query_residual_scale_is_explicit = (
        args.query_residual_scale is not None
        or args.query_residual_scale_values is not None
    )
    wave_kernel_is_explicit = (
        args.wave_kernel is not None or args.wave_kernel_values is not None
    )
    wave_dilations_is_explicit = (
        args.wave_dilations is not None or args.wave_dilation_values is not None
    )
    bigram_guard_is_explicit = (
        args.bigram_topk_guard is not None
        or args.bigram_topk_guard_values is not None
        or args.bigram_topk_guard_schedule is not None
        or args.bigram_topk_guard_arch_schedule is not None
    )
    bigram_rank_guard_is_explicit = (
        args.bigram_rank_guard is not None or args.bigram_rank_guard_values is not None
    )
    bigram_rank_margin_is_explicit = (
        args.bigram_rank_guard_margin is not None
        or args.bigram_rank_guard_margin_values is not None
    )
    bigram_rank_band_is_explicit = (
        args.bigram_rank_guard_band is not None
        or args.bigram_rank_guard_band_values is not None
    )
    bigram_rank_min_is_explicit = (
        args.bigram_rank_guard_min_candidates is not None
        or args.bigram_rank_guard_min_candidates_values is not None
    )
    bigram_soft_guard_is_explicit = (
        args.bigram_soft_guard is not None or args.bigram_soft_guard_values is not None
    )
    training_grid_is_explicit = (
        args.epoch_values is not None or args.batches_values is not None
    )
    eval_samples_grid_is_explicit = args.eval_samples_values is not None
    val_start_grid_is_explicit = (
        args.val_start_fraction is not None or args.val_start_values is not None
    )
    lr_grid_is_explicit = args.lr is not None or args.lr_values is not None
    run_root = args.run_root.resolve() if args.run_root else default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    runs: list[dict[str, object]] = []
    successful_run_dirs: list[Path] = []
    failed = False
    preflight_failures: dict[str, dict[str, object]] = {}

    manifest: dict[str, object] = {
        "schema": "st.char_lm_sweep.v1",
        "started_at_unix": started,
        "run_root": str(run_root),
        "preset": args.preset,
        **recipe_manifest_metadata(args.recipe),
        "settings": settings.__dict__,
        "data_paths": [str(path) for path in args.data_paths],
        "architectures": architectures,
        "features": features,
        "head_priors": head_priors,
        "backends": backends,
        "seeds": seeds,
        "shape_grid": {
            "steps": step_values,
            "embed_dims": embed_dim_values,
            "hidden": hidden_values,
        },
        "training_grid": {
            "epochs": epoch_values,
            "batches": batches_values,
            "eval_samples": eval_samples_values,
            "validation_start_fractions": val_start_values,
        },
        "learning_rates": lr_values,
        "head_residual_scales": head_residual_scale_values,
        "coherence_grid": {
            "context_scales": context_scale_values,
            "self_score_scales": self_score_scale_values,
            "query_residual_scales": query_residual_scale_values,
            "wave_kernels": wave_kernel_values,
            "wave_dilations": wave_dilation_values,
        },
        "bigram_topk_guards": (
            sorted(
                {
                    scheduled_bigram_topk_guard_for_arch(
                        bigram_topk_guard_arch_schedule,
                        architecture,
                        val_start,
                    )
                    for architecture in architectures
                    for val_start in val_start_values
                }
            )
            if bigram_topk_guard_arch_schedule is not None
            else (
                sorted(set(bigram_topk_guard_schedule.values()))
                if bigram_topk_guard_schedule is not None
                else bigram_topk_guard_values
            )
        ),
        "bigram_topk_guard_schedule": (
            schedule_manifest_rows(bigram_topk_guard_schedule)
            if bigram_topk_guard_schedule is not None
            else None
        ),
        "bigram_topk_guard_arch_schedule": (
            arch_schedule_manifest_rows(bigram_topk_guard_arch_schedule)
            if bigram_topk_guard_arch_schedule is not None
            else None
        ),
        "bigram_topk_guard_k": args.bigram_topk_guard_k,
        "bigram_rank_guards": bigram_rank_guard_values,
        "bigram_rank_guard_margins": bigram_rank_guard_margin_values,
        "bigram_rank_guard_bands": bigram_rank_guard_band_values,
        "bigram_rank_guard_min_candidates": bigram_rank_guard_min_candidates_values,
        "bigram_soft_guards": bigram_soft_guard_values,
        "cargo_features": args.cargo_features,
        "no_default_features": args.no_default_features,
        "wgpu_preflight": not args.no_wgpu_preflight,
        "preflight_failures": preflight_failures,
        "max_new_runs": args.max_new_runs,
        "new_runs_started": 0,
        "run_limit_reached": False,
        "compare_summary": {
            "limit": compare_summary_options.limit,
            "route_clean_only": compare_summary_options.route_clean_only,
            "prefer_clean_route": compare_summary_options.prefer_clean_route,
            "command": build_compare_summary_command(run_root, compare_summary_options),
            "command_cwd": str(REPO_ROOT),
            "command_script_path": str(compare_summary_command_script_path(run_root)),
            "fail_on_route_statuses": list(compare_summary_options.fail_on_route_statuses),
            "fail_on_paired_quality_statuses": list(
                compare_summary_options.fail_on_paired_quality_statuses
            ),
            "fail_on_efficiency_verdicts": list(
                compare_summary_options.fail_on_efficiency_verdicts
            ),
            "fail_on_rank_min_promotion_decisions": list(
                compare_summary_options.fail_on_rank_min_promotion_decisions
            ),
            "fail_on_route_debt_decisions": list(
                compare_summary_options.fail_on_route_debt_decisions
            ),
            "extra_compare_paths": [
                str(path) for path in compare_summary_options.extra_compare_paths
            ],
            "merge_evidence_sources": compare_summary_options.merge_evidence_sources,
            "sort_metric": compare_summary_options.sort_metric,
        },
        "dry_run": args.dry_run,
        "runs": runs,
    }
    write_sweep_manifest(run_root, manifest)

    for backend in dict.fromkeys(backends):
        failure = run_wgpu_preflight(
            args,
            run_root,
            settings,
            backend=backend,
            architecture=architectures[0],
            char_feature=features[0],
            head_prior=head_priors[0],
        )
        if failure is None:
            continue
        preflight_failures[backend] = failure
        failed = True
        write_sweep_manifest(run_root, manifest)
        if not args.continue_on_error:
            print(
                f"preflight failed for backend {backend}; see {failure.get('log_path')}",
                file=sys.stderr,
            )
            manifest["finished_at_unix"] = time.time()
            manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
            manifest["failed"] = True
            write_sweep_manifest(run_root, manifest)
            returncode = failure.get("returncode")
            if isinstance(returncode, int) and returncode > 0:
                return returncode
            return 1

    def coherence_grid_for_architecture(
        architecture: str,
    ) -> tuple[
        list[float | None],
        list[float | None],
        list[float | None],
        list[int | None],
        list[str | None],
    ]:
        if architecture == "scan":
            return (
                context_scale_values,
                self_score_scale_values,
                query_residual_scale_values,
                [None],
                [None],
            )
        if architecture == "wave":
            return (
                [None],
                self_score_scale_values,
                query_residual_scale_values,
                wave_kernel_values,
                wave_dilation_values,
            )
        return ([None], [None], [None], [None], [None])

    base_grid_count = (
        len(features)
        * len(head_priors)
        * len(backends)
        * len(step_values)
        * len(embed_dim_values)
        * len(hidden_values)
        * len(epoch_values)
        * len(batches_values)
        * len(eval_samples_values)
        * len(val_start_values)
        * len(lr_values)
        * len(head_residual_scale_values)
        * len(bigram_topk_guard_values)
        * len(bigram_rank_guard_values)
        * len(bigram_rank_guard_margin_values)
        * len(bigram_rank_guard_band_values)
        * len(bigram_rank_guard_min_candidates_values)
        * len(bigram_soft_guard_values)
        * len(seeds)
    )
    total = 0
    for architecture in architectures:
        arch_context_values, arch_self_values, arch_query_values, arch_kernel_values, arch_dilation_values = (
            coherence_grid_for_architecture(architecture)
        )
        total += (
            base_grid_count
            * len(arch_context_values)
            * len(arch_self_values)
            * len(arch_query_values)
            * len(arch_kernel_values)
            * len(arch_dilation_values)
        )
    manifest["planned_runs"] = total
    index = 0
    new_runs_started = 0
    run_limit_reached = False
    next_run_after_limit: dict[str, object] | None = None
    run_spaces = (
        product(
            [architecture],
            features,
            head_priors,
            backends,
            step_values,
            embed_dim_values,
            hidden_values,
            epoch_values,
            batches_values,
            eval_samples_values,
            val_start_values,
            lr_values,
            head_residual_scale_values,
            *coherence_grid_for_architecture(architecture),
            bigram_topk_guard_values,
            bigram_rank_guard_values,
            bigram_rank_guard_margin_values,
            bigram_rank_guard_band_values,
            bigram_rank_guard_min_candidates_values,
            bigram_soft_guard_values,
            seeds,
        )
        for architecture in architectures
    )
    run_space = (item for architecture_space in run_spaces for item in architecture_space)
    for (
        architecture,
        feature,
        head_prior,
        backend,
        step_value,
        embed_dim_value,
        hidden_value,
        epoch_value,
        batches_value,
        eval_samples_value,
        val_start_value,
        lr_value,
        head_residual_scale_value,
        context_scale_value,
        self_score_scale_value,
        query_residual_scale_value,
        wave_kernel_value,
        wave_dilation_value,
        bigram_topk_guard_value,
        bigram_rank_guard_value,
        bigram_rank_guard_margin_value,
        bigram_rank_guard_band_value,
        bigram_rank_guard_min_candidates_value,
        bigram_soft_guard_value,
        seed,
    ) in run_space:
        index += 1
        effective_bigram_topk_guard_value = (
            scheduled_bigram_topk_guard_for_arch(
                bigram_topk_guard_arch_schedule,
                architecture,
                val_start_value,
            )
            if bigram_topk_guard_arch_schedule is not None
            else (
                scheduled_bigram_topk_guard(bigram_topk_guard_schedule, val_start_value)
                if bigram_topk_guard_schedule is not None
                else bigram_topk_guard_value
            )
        )
        run_settings = replace(
            settings,
            backend=backend,
            steps=step_value,
            embed_dim=embed_dim_value,
            hidden=hidden_value,
            epochs=epoch_value if epoch_value is not None else settings.epochs,
            batches=batches_value if batches_value is not None else settings.batches,
            eval_samples=(
                eval_samples_value
                if eval_samples_value is not None
                else settings.eval_samples
            ),
            val_start_fraction=val_start_value,
            lr=lr_value,
            head_residual_scale=head_residual_scale_value,
            context_scale=context_scale_value,
            self_score_scale=self_score_scale_value,
            query_residual_scale=query_residual_scale_value,
            wave_kernel=wave_kernel_value,
            wave_dilations=wave_dilation_value,
            bigram_topk_guard=effective_bigram_topk_guard_value,
            bigram_topk_guard_k=args.bigram_topk_guard_k,
            bigram_rank_guard=bigram_rank_guard_value,
            bigram_rank_guard_margin=bigram_rank_guard_margin_value,
            bigram_rank_guard_band=bigram_rank_guard_band_value,
            bigram_rank_guard_min_candidates=bigram_rank_guard_min_candidates_value,
            bigram_soft_guard=bigram_soft_guard_value,
        )
        run_name_parts = [
            slug(architecture),
            f"feature-{slug(feature)}",
            f"head-{slug(head_prior)}",
            f"backend-{slug(backend)}",
        ]
        if shape_grid_is_explicit:
            run_name_parts.extend(
                [
                    grid_slug("steps", step_value),
                    grid_slug("embed", embed_dim_value),
                    grid_slug("hidden", hidden_value),
                ]
            )
        if head_residual_scale_is_explicit:
            run_name_parts.append(float_grid_slug("headresid", head_residual_scale_value))
        if architecture == "scan" and context_scale_is_explicit:
            run_name_parts.append(float_grid_slug("ctx", context_scale_value))
        if architecture in {"scan", "wave"} and self_score_scale_is_explicit:
            run_name_parts.append(float_grid_slug("selfscore", self_score_scale_value))
        if architecture in {"scan", "wave"} and query_residual_scale_is_explicit:
            run_name_parts.append(float_grid_slug("queryresid", query_residual_scale_value))
        if architecture == "wave" and wave_kernel_is_explicit:
            run_name_parts.append(grid_slug("kernel", wave_kernel_value))
        if architecture == "wave" and wave_dilations_is_explicit:
            run_name_parts.append(f"dil-{slug(wave_dilation_value or 'default')}")
        if bigram_guard_is_explicit:
            run_name_parts.append(
                float_grid_slug("biguard", effective_bigram_topk_guard_value)
            )
            if args.bigram_topk_guard_k is not None:
                run_name_parts.append(grid_slug("biguardk", args.bigram_topk_guard_k))
        if bigram_rank_guard_is_explicit:
            run_name_parts.append(float_grid_slug("bigrank", bigram_rank_guard_value))
            if bigram_rank_margin_is_explicit:
                run_name_parts.append(
                    float_grid_slug("bigrankm", bigram_rank_guard_margin_value)
                )
            if bigram_rank_band_is_explicit:
                run_name_parts.append(
                    float_grid_slug("bigrankband", bigram_rank_guard_band_value)
                )
            if bigram_rank_min_is_explicit:
                run_name_parts.append(
                    grid_slug("bigrankmin", bigram_rank_guard_min_candidates_value)
                )
        if bigram_soft_guard_is_explicit:
            run_name_parts.append(float_grid_slug("bigsoft", bigram_soft_guard_value))
        if training_grid_is_explicit:
            run_name_parts.extend(
                [
                    grid_slug("epochs", epoch_value),
                    grid_slug("batches", batches_value),
                ]
            )
        if eval_samples_grid_is_explicit:
            run_name_parts.append(grid_slug("eval", eval_samples_value))
        if val_start_grid_is_explicit:
            run_name_parts.append(float_grid_slug("valstart", val_start_value))
        if lr_grid_is_explicit:
            run_name_parts.append(float_grid_slug("lr", lr_value))
        run_name_parts.append(f"seed-{seed}")
        run_name = "__".join(run_name_parts)
        run_dir = run_root / run_name
        log_path = run_dir / "process.log"
        command = build_command(
            cargo_bin=args.cargo_bin,
            cargo_features=effective_cargo_features(args, backend),
            no_default_features=args.no_default_features,
            architecture=architecture,
            data_paths=args.data_paths,
            run_dir=run_dir,
            char_feature=feature,
            head_prior=head_prior,
            seed=seed,
            settings=run_settings,
            extra_args=args.extra_arg,
        )
        preflight_failure = preflight_failures.get(backend)
        failure_kind = None
        failure_detail = None
        skipped = False
        if preflight_failure is not None:
            returncode = int(preflight_failure.get("returncode") or 1)
            elapsed = 0.0
            skipped = True
            failure_kind = f"preflight_{preflight_failure.get('failure_kind', 'failed')}"
            failure_detail = str(preflight_failure.get("failure_detail") or "preflight failed")
            write_preflight_skipped_log(
                log_path,
                preflight_failure,
                failure_kind,
                failure_detail,
            )
            write_failure(
                run_dir,
                {
                    "schema": "st.char_lm_sweep_failure.v1",
                    "name": run_name,
                    "architecture": architecture,
                    "example": EXAMPLES[architecture],
                    "char_feature": feature,
                    "head_prior": head_prior,
                    "backend": backend,
                    "seed": seed,
                    "steps": step_value,
                    "embed_dim": embed_dim_value,
                    "hidden": hidden_value,
                    "epochs": run_settings.epochs,
                    "batches": run_settings.batches,
                    "batch": run_settings.batch,
                    "eval_samples": run_settings.eval_samples,
                    "validation_start_fraction": run_settings.val_start_fraction,
                    "lr": lr_value,
                    "head_residual_scale": head_residual_scale_value,
                    "context_scale": context_scale_value,
                    "self_score_scale": self_score_scale_value,
                    "query_residual_scale": query_residual_scale_value,
                    "wave_kernel": wave_kernel_value,
                    "wave_dilations": wave_dilation_value,
                    "bigram_topk_guard": effective_bigram_topk_guard_value,
                    "bigram_topk_guard_k": args.bigram_topk_guard_k,
                    "bigram_rank_guard": bigram_rank_guard_value,
                    "bigram_rank_guard_margin": bigram_rank_guard_margin_value,
                    "bigram_rank_guard_band": bigram_rank_guard_band_value,
                    "bigram_rank_guard_min_candidates": (
                        bigram_rank_guard_min_candidates_value
                    ),
                    "bigram_soft_guard": bigram_soft_guard_value,
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "returncode": returncode,
                    "failure_kind": failure_kind,
                    "failure_detail": failure_detail,
                    "command": command,
                    "preflight_failure": preflight_failure,
                },
            )
        elif args.skip_existing and (run_dir / "summary.json").exists():
            returncode = 0
            elapsed = 0.0
            skipped = True
        else:
            if args.max_new_runs is not None and new_runs_started >= args.max_new_runs:
                run_limit_reached = True
                next_run_after_limit = {
                    "index": index,
                    "name": run_name,
                    "architecture": architecture,
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
                break
            new_runs_started += 1
            manifest["new_runs_started"] = new_runs_started
            if not args.quiet_runs:
                print(f"[{index}/{total}] {run_name}")
                print("  " + shlex.join(command))
            returncode, elapsed = run_command(
                command,
                log_path,
                dry_run=args.dry_run,
            )
            if returncode != 0:
                failure_kind, failure_detail = classify_failure(
                    returncode,
                    log_path,
                )
        trace_summary_path = None
        trace_summary_error = None
        if returncode == 0 and not args.dry_run:
            trace_summary_path, trace_summary_error = write_trainer_trace_summary(run_dir)
        summary = read_json(run_dir / "summary.json")
        run_payload = read_json(run_dir / "run.json")
        run_record: dict[str, object] = {
            "name": run_name,
            "architecture": architecture,
            "example": EXAMPLES[architecture],
            "char_feature": feature,
            "head_prior": head_prior,
            "backend": backend,
            "seed": seed,
            "steps": step_value,
            "embed_dim": embed_dim_value,
            "hidden": hidden_value,
            "epochs": run_settings.epochs,
            "batches": run_settings.batches,
            "batch": run_settings.batch,
            "eval_samples": run_settings.eval_samples,
            "validation_start_fraction": run_settings.val_start_fraction,
            "lr": lr_value,
            "head_residual_scale": head_residual_scale_value,
            "context_scale": context_scale_value,
            "self_score_scale": self_score_scale_value,
            "query_residual_scale": query_residual_scale_value,
            "wave_kernel": wave_kernel_value,
            "wave_dilations": wave_dilation_value,
            "bigram_topk_guard": effective_bigram_topk_guard_value,
            "bigram_topk_guard_k": args.bigram_topk_guard_k,
            "bigram_rank_guard": bigram_rank_guard_value,
            "bigram_rank_guard_margin": bigram_rank_guard_margin_value,
            "bigram_rank_guard_band": bigram_rank_guard_band_value,
            "bigram_rank_guard_min_candidates": bigram_rank_guard_min_candidates_value,
            "bigram_soft_guard": bigram_soft_guard_value,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "command": command,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "skipped": skipped,
            "failed": False,
            "run_status": "skipped" if skipped else "ok",
            "failure_kind": failure_kind,
            "failure_detail": failure_detail,
            "summary_path": str(run_dir / "summary.json"),
            "has_summary": summary is not None,
            "run_json_path": str(run_dir / "run.json"),
            "has_run_json": run_payload is not None,
            "trainer_trace_summary_path": str(trace_summary_path)
            if trace_summary_path is not None
            else None,
            "has_trainer_trace_summary": trace_summary_path is not None,
        }
        if isinstance(run_payload, dict):
            run_record["backend"] = run_payload.get("backend")
            backend_runtime = run_payload.get("backend_runtime")
            if isinstance(backend_runtime, dict):
                run_record["backend_runtime"] = backend_runtime
            tensor_policy = run_payload.get("tensor_policy")
            if isinstance(tensor_policy, dict):
                run_record["tensor_policy"] = tensor_policy
            roundtable_summary = roundtable_wgpu_summary(run_payload)
            if roundtable_summary:
                run_record["roundtable_wgpu"] = roundtable_summary
        if trace_summary_error is not None:
            run_record["trainer_trace_summary_error"] = trace_summary_error
        if isinstance(summary, dict):
            run_record["best_validation_mean_nll"] = summary.get("best_validation_mean_nll")
            final = summary.get("final_validation")
            if isinstance(final, dict):
                run_record["final_validation_mean_nll"] = final.get("mean_nll")
        missing_summary = returncode == 0 and summary is None and not args.dry_run
        if missing_summary:
            failure_kind = "missing_summary"
            failure_detail = "summary.json missing after successful command"
        if returncode != 0 or missing_summary:
            run_record["failed"] = True
            run_record["run_status"] = "failed"
            run_record["failure_kind"] = failure_kind
            run_record["failure_detail"] = failure_detail
            if preflight_failure is None:
                write_failure(
                    run_dir,
                    {
                        "schema": "st.char_lm_sweep_failure.v1",
                        "name": run_name,
                        "architecture": architecture,
                        "example": EXAMPLES[architecture],
                        "char_feature": feature,
                        "head_prior": head_prior,
                        "backend": backend,
                        "seed": seed,
                        "steps": step_value,
                        "embed_dim": embed_dim_value,
                        "hidden": hidden_value,
                        "epochs": run_settings.epochs,
                        "batches": run_settings.batches,
                        "batch": run_settings.batch,
                        "head_residual_scale": head_residual_scale_value,
                        "context_scale": context_scale_value,
                        "self_score_scale": self_score_scale_value,
                        "query_residual_scale": query_residual_scale_value,
                        "wave_kernel": wave_kernel_value,
                        "wave_dilations": wave_dilation_value,
                        "bigram_topk_guard": effective_bigram_topk_guard_value,
                        "bigram_topk_guard_k": args.bigram_topk_guard_k,
                        "bigram_rank_guard": bigram_rank_guard_value,
                        "bigram_rank_guard_margin": bigram_rank_guard_margin_value,
                        "bigram_rank_guard_band": bigram_rank_guard_band_value,
                        "bigram_rank_guard_min_candidates": (
                            bigram_rank_guard_min_candidates_value
                        ),
                        "bigram_soft_guard": bigram_soft_guard_value,
                        "run_dir": str(run_dir),
                        "log_path": str(log_path),
                        "returncode": returncode,
                        "failure_kind": failure_kind,
                        "failure_detail": failure_detail,
                        "command": command,
                    },
                )
        runs.append(run_record)
        if returncode == 0 and summary is not None:
            successful_run_dirs.append(run_dir)
        elif returncode != 0 or missing_summary:
            failed = True
            if not args.continue_on_error:
                write_sweep_manifest(run_root, manifest)
                if missing_summary:
                    print(
                        f"run produced no summary.json: {run_name}; see {log_path}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"run failed: {run_name}; see {log_path}",
                        file=sys.stderr,
                    )
                return returncode or 1
        write_sweep_manifest(run_root, manifest)

    manifest["new_runs_started"] = new_runs_started
    manifest["run_limit_reached"] = run_limit_reached
    manifest["next_run_after_limit"] = next_run_after_limit
    compare_output = None
    compare_summary_output = None
    compare_summary_failed = False
    if not args.dry_run:
        compare_output = render_compare(successful_run_dirs, run_root, curves=args.curves)
        if compare_output is not None:
            compare_summary_output = render_compare_summary(
                run_root,
                options=compare_summary_options,
            )
            compare_summary_failed = (
                compare_summary_output is None
                and (run_root / "compare_summary.error.log").exists()
            )
            if compare_summary_failed and compare_summary_failure_is_fatal(
                compare_summary_options
            ):
                failed = True
        failure_compare_output = append_failed_runs_compare(run_root, runs)
        if failure_compare_output is not None:
            compare_output = failure_compare_output
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_at_unix"] - started
    manifest["compare_path"] = str(run_root / "compare.md") if compare_output is not None else None
    manifest["compare_json_path"] = (
        str(run_root / "compare.json") if compare_output is not None else None
    )
    compare_summary_path = run_root / "compare_summary.md"
    compare_summary_json_path = run_root / "compare_summary.json"
    compare_summary_error_path = run_root / "compare_summary.error.log"
    compare_summary_command_path = compare_summary_command_script_path(run_root)
    manifest["compare_summary_path"] = (
        str(compare_summary_path) if compare_summary_path.exists() else None
    )
    manifest["compare_summary_json_path"] = (
        str(compare_summary_json_path) if compare_summary_json_path.exists() else None
    )
    manifest["compare_summary_error_path"] = (
        str(compare_summary_error_path) if compare_summary_error_path.exists() else None
    )
    manifest["compare_summary_command_path"] = (
        str(compare_summary_command_path)
        if compare_summary_command_path.exists()
        else None
    )
    manifest["compare_summary_command_cwd"] = (
        str(REPO_ROOT) if compare_summary_command_path.exists() else None
    )
    manifest["compare_summary_failed"] = compare_summary_failed
    compare_summary_payload = read_json(compare_summary_json_path)
    if isinstance(compare_summary_payload, dict):
        manifest["compare_summary_source"] = compare_summary_payload.get("source")
        manifest["compare_summary_sources"] = compare_summary_payload.get("sources")
        manifest["compare_summary_merge_evidence_sources"] = compare_summary_payload.get(
            "merge_evidence_sources"
        )
        manifest["compare_summary_route_status_counts"] = compare_summary_payload.get(
            "route_status_counts"
        )
        manifest["compare_summary_selected_route_status_counts"] = compare_summary_payload.get(
            "selected_route_status_counts"
        )
        manifest["compare_summary_route_status_gate"] = compare_summary_payload.get(
            "route_status_gate"
        )
        manifest["compare_summary_paired_recurrent_gate"] = compare_summary_payload.get(
            "paired_recurrent_gate"
        )
        manifest["compare_summary_paired_recurrent_deltas"] = compare_summary_payload.get(
            "paired_recurrent_deltas"
        )
        manifest["compare_summary_paired_recurrent_recommendations"] = (
            compare_summary_payload.get("paired_recurrent_recommendations")
        )
        manifest["compare_summary_bigram_guard_deltas"] = compare_summary_payload.get(
            "bigram_guard_deltas"
        )
        manifest["compare_summary_bigram_guard_recommendations"] = (
            compare_summary_payload.get("bigram_guard_recommendations")
        )
        manifest["compare_summary_bigram_rank_guard_deltas"] = compare_summary_payload.get(
            "bigram_rank_guard_deltas"
        )
        manifest["compare_summary_bigram_rank_guard_recommendations"] = (
            compare_summary_payload.get("bigram_rank_guard_recommendations")
        )
        manifest["compare_summary_bigram_rank_guard_seed_deltas"] = (
            compare_summary_payload.get("bigram_rank_guard_seed_deltas")
        )
        manifest["compare_summary_bigram_rank_guard_stability"] = (
            compare_summary_payload.get("bigram_rank_guard_stability")
        )
        manifest["compare_summary_bigram_rank_band_deltas"] = (
            compare_summary_payload.get("bigram_rank_band_deltas")
        )
        manifest["compare_summary_bigram_rank_band_recommendations"] = (
            compare_summary_payload.get("bigram_rank_band_recommendations")
        )
        manifest["compare_summary_bigram_rank_band_seed_deltas"] = (
            compare_summary_payload.get("bigram_rank_band_seed_deltas")
        )
        manifest["compare_summary_bigram_rank_band_stability"] = (
            compare_summary_payload.get("bigram_rank_band_stability")
        )
        manifest["compare_summary_bigram_rank_min_deltas"] = (
            compare_summary_payload.get("bigram_rank_min_deltas")
        )
        manifest["compare_summary_bigram_rank_min_recommendations"] = (
            compare_summary_payload.get("bigram_rank_min_recommendations")
        )
        manifest["compare_summary_bigram_rank_min_seed_deltas"] = (
            compare_summary_payload.get("bigram_rank_min_seed_deltas")
        )
        manifest["compare_summary_bigram_rank_min_stability"] = (
            compare_summary_payload.get("bigram_rank_min_stability")
        )
        manifest["compare_summary_bigram_rank_min_stable_recommendations"] = (
            compare_summary_payload.get("bigram_rank_min_stable_recommendations")
        )
        manifest["compare_summary_bigram_rank_min_promotion_gate"] = (
            compare_summary_payload.get("bigram_rank_min_promotion_gate")
        )
        manifest["compare_summary_bigram_soft_guard_deltas"] = (
            compare_summary_payload.get("bigram_soft_guard_deltas")
        )
        manifest["compare_summary_bigram_soft_guard_recommendations"] = (
            compare_summary_payload.get("bigram_soft_guard_recommendations")
        )
        manifest["compare_summary_bigram_soft_guard_seed_deltas"] = (
            compare_summary_payload.get("bigram_soft_guard_seed_deltas")
        )
        manifest["compare_summary_bigram_soft_guard_stability"] = (
            compare_summary_payload.get("bigram_soft_guard_stability")
        )
        manifest["compare_summary_learning_scoreboard_rows"] = (
            compare_summary_payload.get("learning_scoreboard_rows")
        )
        manifest["compare_summary_route_debt_recommendation_summary"] = (
            compare_summary_payload.get("route_debt_recommendation_summary")
        )
        manifest["compare_summary_route_debt_recommendations"] = (
            compare_summary_payload.get("route_debt_recommendations")
        )
        manifest["compare_summary_baseline_difficulty_rows"] = (
            compare_summary_payload.get("baseline_difficulty_rows")
        )
    manifest["failed"] = failed
    write_sweep_manifest(run_root, manifest)

    print(f"sweep: {run_root}")
    print(f"manifest: {run_root / 'sweep.json'}")
    if compare_output is not None:
        print(f"compare: {run_root / 'compare.md'}")
        print(f"compare_json: {run_root / 'compare.json'}")
        if compare_summary_output is not None:
            print(f"compare_summary: {run_root / 'compare_summary.md'}")
            print(f"compare_summary_json: {run_root / 'compare_summary.json'}")
        elif compare_summary_failed:
            print(f"compare_summary: {run_root / 'compare_summary.md'}")
            print(f"compare_summary_error: {run_root / 'compare_summary.error.log'}")
        if not args.no_print_compare:
            print(compare_output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
