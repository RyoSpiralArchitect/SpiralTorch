from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "models" / "python" / "llm_char_vae_context.py"
_MISSING = object()


def _load_module():
    module_name = "_spiraltorch_llm_char_vae_context_under_test"
    cached = sys.modules.get(module_name)
    if cached is not None and hasattr(cached, "_follow_up_guidance_record"):
        return cached
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_spiraltorch = sys.modules.get("spiraltorch", _MISSING)
    sys.modules["spiraltorch"] = types.ModuleType("spiraltorch")
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_spiraltorch is _MISSING:
            sys.modules.pop("spiraltorch", None)
        else:
            sys.modules["spiraltorch"] = previous_spiraltorch
    return module


def _ancestor(*, feature: str, nll: float, raw_delta: float | None = None) -> dict:
    best_config = {
        "best_feature": feature,
        "feature_normalize": "blocks",
        "hybrid_latent_scale": 1.0,
        "mean_best_nll": nll,
    }
    if raw_delta is not None:
        best_config["mean_best_nll_delta_vs_raw"] = raw_delta
    return {
        "schema": "st.llm_char_vae_context.follow_up_ancestor.v1",
        "summary_path": "/tmp/source/summary.json",
        "run_dir": "/tmp/source",
        "generation": 0,
        "status": "improved",
        "best_feature": feature,
        "best_config": best_config,
        "mean_best_nll": nll,
        "verdict": None,
        "guidance_action": None,
        "guided_enabled": None,
        "gate_failed": None,
    }


def _summary(
    *,
    best_feature: str,
    nll: float,
    verdict: str,
    config_verdict: str,
    source_feature_verdict: str,
    source_retained: bool,
    gate_failed: bool,
    source_feature_raw_verdict: str = "unknown",
    source_feature_delta_vs_raw: float | None = None,
) -> dict:
    return {
        "status": "improved",
        "best_feature": best_feature,
        "best_config": {
            "best_feature": best_feature,
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 1.0,
            "mean_best_nll": nll,
        },
        "follow_up_result": {
            "verdict": verdict,
            "config_verdict": config_verdict,
            "source_feature_verdict": source_feature_verdict,
            "source_feature_raw_verdict": source_feature_raw_verdict,
            "source_feature_mean_best_nll_delta_vs_raw": source_feature_delta_vs_raw,
            "source_best_feature_retained": source_retained,
        },
        "follow_up_chain": {
            "generation": 1,
            "latest_verdict": verdict,
            "verdict_history": [verdict],
            "improved_streak": 1 if verdict == "improved" else 0,
            "regressed_streak": 1 if verdict == "regressed" else 0,
        },
        "follow_up_gate": {
            "failed": gate_failed,
            "verdict": verdict,
            "fail_on_verdicts": ["regressed", "unknown"],
        },
    }


def _next_follow_up() -> dict:
    return {
        "script_path": "/tmp/current/next_follow_up_command.sh",
        "script_usage": "NEW_SEEDS=17 bash /tmp/current/next_follow_up_command.sh",
        "default_follow_up_from": "/tmp/current/summary.json",
        "default_new_seeds": "17",
        "default_run_dir": "/tmp/current/follow_up_best_config",
        "default_follow_up_fail_on_verdict": "regressed,unknown",
        "shell_command": "PYTHONNOUSERSITE=1 python3 ...",
        "script_command": ["python3", "models/python/llm_char_vae_context.py"],
    }


def _seed_summary(
    seed: int,
    *,
    raw: float,
    latent: float,
    raw_latent: float,
    reconstruction_latent: float,
) -> dict:
    scores = {
        "raw": raw,
        "latent": latent,
        "raw_latent": raw_latent,
        "reconstruction_latent": reconstruction_latent,
    }
    steps = {
        "raw": 4,
        "latent": 3,
        "raw_latent": 2,
        "reconstruction_latent": 1,
    }
    ranking = [
        {
            "feature": feature,
            "best_mean_nll": nll,
            "best_accuracy": 0.25,
            "best_step": steps[feature],
            "validation_nll_mean": nll + 0.01,
            "validation_nll_initial_minus_best": 0.10,
            "validation_nll_final_minus_best": steps[feature] * 0.001,
        }
        for feature, nll in sorted(scores.items(), key=lambda item: (item[1], item[0]))
    ]
    return {
        "run": {"seed": seed},
        "best_feature": ranking[0]["feature"],
        "ranking": ranking,
        "features": [
            {
                "feature": feature,
                "best_validation": {"mean_nll": nll, "accuracy": 0.25},
                "best_step": steps[feature],
                "validation_nll_mean": nll + 0.01,
                "validation_nll_initial_minus_best": 0.10,
                "validation_nll_final_minus_best": steps[feature] * 0.001,
            }
            for feature, nll in scores.items()
        ],
        "deltas": {
            **{
                f"{feature}_best_nll_vs_raw": nll - raw
                for feature, nll in scores.items()
            },
            **{
                f"{feature}_validation_nll_mean_vs_raw": nll - raw
                for feature, nll in scores.items()
            },
        },
        "feature_diagnostics": {"features": {}},
    }


class CharVaeContextGuidanceTests(unittest.TestCase):
    def test_xavier_rescale_values_fit_requested_abs_limit(self) -> None:
        mod = _load_module()

        values = mod._rescale_values_to_abs_limit([0.1, -0.2, 0.4], 0.05)

        self.assertAlmostEqual(max(abs(value) for value in values), 0.05)
        self.assertAlmostEqual(values[0], 0.0125)
        self.assertEqual(
            mod._rescale_values_to_abs_limit([0.0, 0.0], 0.05),
            [0.0, 0.0],
        )

    def test_follow_up_defaults_inherit_source_head_init(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "summary.json"
            source.write_text(
                json.dumps(
                    {
                        "status": "improved",
                        "best_feature": "raw_latent",
                        "best_config": {
                            "best_feature": "raw_latent",
                            "feature_normalize": "blocks",
                            "hybrid_latent_scale": 4.0,
                            "mean_best_nll": 4.1,
                        },
                        "config_summaries": [
                            {
                                "feature_normalize": "blocks",
                                "hybrid_latent_scale": 4.0,
                                "feature_summary": [
                                    {
                                        "feature": "raw_latent",
                                        "best_nll": {
                                            "count": 3,
                                            "stddev": 0.03,
                                            "stderr": 0.017320508075688773,
                                        },
                                    }
                                ],
                            }
                        ],
                        "run": {
                            "features": ["raw", "raw_latent"],
                            "seeds": [101, 103, 107],
                            "head_init": "xavier",
                            "window_chars": 32,
                            "latent_dim": 8,
                            "hidden": 16,
                            "epochs": 8,
                            "batches": 16,
                            "batch_size": 4,
                            "eval_samples": 128,
                            "vae_epochs": 8,
                            "vae_batches": 16,
                            "vae_batch_size": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            argv = ["models/samples/spiral_corpus_en", "--follow-up-from", str(source)]
            args = parser.parse_args(argv)

            record = mod._apply_follow_up_defaults(args, argv)

        self.assertEqual(args.head_init, "xavier")
        self.assertEqual(record["applied_defaults"]["head_init"], "xavier")
        self.assertAlmostEqual(
            record["source_best_config"]["mean_best_nll_stderr"],
            0.017320508075688773,
        )
        self.assertIs(record["user_overrides"]["head_init"], False)

    def test_next_follow_up_avoids_source_and_current_seed_history(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["models/samples/spiral_corpus_en"])
        args.head_init = "xavier"
        best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "mean_best_nll": 4.2,
        }
        follow_up = {
            "source_seeds": [101, 103, 107],
            "source_chain": {"ancestors": []},
            "resolved": {"seeds": [109, 113, 127]},
        }

        record = mod._next_follow_up_command_record(
            args,
            ["raw", "latent", "raw_latent"],
            best_config,
            Path("/tmp/current"),
            [109, 113, 127],
            follow_up,
        )

        self.assertEqual(record["default_new_seeds"], "131,137,139")
        self.assertEqual(
            record["script_command"][
                record["script_command"].index("--head-init") + 1
            ],
            "xavier",
        )
        self.assertEqual(
            record["used_seed_history"],
            [101, 103, 107, 109, 113, 127],
        )

    def test_next_follow_up_boosts_seed_count_for_uncertainty_tie(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["models/samples/spiral_corpus_en"])
        best_config = {
            "best_feature": "reconstruction_latent",
            "runner_up_feature": "raw_latent",
            "margin_to_runner_up": 0.0004,
            "combined_runner_up_margin_stderr": 0.0005,
            "runner_up_within_uncertainty": True,
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 1.0,
            "mean_best_nll": 4.2,
        }
        follow_up = {
            "source_seeds": [101, 103, 107],
            "source_chain": {"ancestors": []},
            "resolved": {"seeds": [109, 113, 127]},
        }

        record = mod._next_follow_up_command_record(
            args,
            ["raw", "latent", "raw_latent", "reconstruction_latent"],
            best_config,
            Path("/tmp/current"),
            [109, 113, 127],
            follow_up,
        )

        self.assertEqual(record["default_new_seed_count"], 5)
        self.assertEqual(record["default_new_seeds"], "131,137,139,149,151")
        self.assertIs(
            record["seed_confirmation_policy"]["uncertainty_tie_seed_boost"],
            True,
        )
        self.assertEqual(
            record["seed_confirmation_policy"]["reason"],
            "best runner-up within combined seed uncertainty",
        )
        self.assertIn("--seeds 131,137,139,149,151", record["shell_command"])

    def test_default_follow_up_seeds_refreshes_stale_seed_history(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_summary = root / "parent_summary.json"
            first_follow_up_summary = root / "first_follow_up_summary.json"
            parent_summary.write_text(
                json.dumps({"run": {"seeds": [7, 13, 17]}}),
                encoding="utf-8",
            )
            first_follow_up_summary.write_text(
                json.dumps({"run": {"seeds": [101, 103, 107]}}),
                encoding="utf-8",
            )
            summary = {
                "run": {"seeds": [109, 113, 127]},
                "follow_up_chain": {
                    "generation": 2,
                    "ancestors": [str(parent_summary), str(first_follow_up_summary)],
                },
                "next_follow_up_command": {"default_new_seeds": "101,103,107"},
            }

            self.assertEqual(mod._default_follow_up_seeds(summary), "131,137,139")

    def test_feature_family_stability_groups_hybrid_latent_members(self) -> None:
        mod = _load_module()
        summaries = [
            _seed_summary(
                1,
                raw=4.03,
                latent=4.04,
                raw_latent=4.0004,
                reconstruction_latent=4.0,
            ),
            _seed_summary(
                2,
                raw=4.10,
                latent=4.11,
                raw_latent=4.08,
                reconstruction_latent=4.0805,
            ),
        ]

        aggregate = mod._aggregate_summaries(
            summaries,
            min_nll_delta=0.0,
            win_tolerance=0.001,
        )
        thin_aggregate = mod._aggregate_summaries(
            [
                {
                    "run": summary["run"],
                    "best_feature": summary["best_feature"],
                    "ranking": summary["ranking"],
                    "deltas": summary["deltas"],
                }
                for summary in summaries
            ],
            min_nll_delta=0.0,
            win_tolerance=0.001,
        )
        families = {
            item["family"]: item
            for item in aggregate["feature_family_stability"]
        }
        thin_families = {
            item["family"]: item
            for item in thin_aggregate["feature_family_stability"]
        }
        report = mod._aggregate_report(
            {
                "run": {"seed_count": 2},
                "seed_summaries": summaries,
                **aggregate,
            }
        )

        hybrid = families["hybrid_latent"]
        self.assertEqual(hybrid["win_count"], 2)
        self.assertEqual(hybrid["near_win_count"], 2)
        self.assertAlmostEqual(hybrid["mean_best_nll_delta_vs_raw"], -0.025)
        self.assertEqual(
            hybrid["member_best_counts"],
            {"reconstruction_latent": 1, "raw_latent": 1},
        )
        self.assertEqual(thin_families["hybrid_latent"]["win_count"], 2)
        self.assertAlmostEqual(
            thin_families["hybrid_latent"]["mean_best_nll"],
            4.04,
        )
        self.assertAlmostEqual(
            thin_aggregate["ranking"][0]["mean_best_nll"],
            4.0402,
        )
        by_feature = {item["feature"]: item for item in aggregate["ranking"]}
        self.assertAlmostEqual(by_feature["raw_latent"]["mean_best_step"], 2.0)
        self.assertAlmostEqual(
            by_feature["raw_latent"]["mean_validation_nll_mean_delta_vs_raw"],
            -0.0248,
        )
        self.assertAlmostEqual(
            by_feature["raw_latent"]["mean_validation_nll_final_minus_best"],
            0.002,
        )
        self.assertEqual(families["raw"]["win_count"], 0)
        self.assertIn("## Feature Family Stability", report)
        self.assertIn("curve_nll", report)
        self.assertIn("| hybrid_latent | 4.040000 | 25.00% | -0.025000 | 2/2", report)

    def test_best_config_summary_surfaces_runner_up_margin(self) -> None:
        mod = _load_module()
        config = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "best_feature": "raw_latent",
            "status": "improved",
            "run_dir": "/tmp/chain/parent",
            "feature_summary": [
                {
                    "feature": "raw_latent",
                    "best_nll": {"count": 3, "stderr": 0.0003},
                },
                {
                    "feature": "reconstruction_latent",
                    "best_nll": {"count": 3, "stderr": 0.0004},
                },
            ],
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.1277,
                    "mean_best_accuracy": 0.16,
                    "mean_best_nll_delta_vs_raw": -0.063,
                    "runs": 3,
                },
                {
                    "feature": "reconstruction_latent",
                    "mean_best_nll": 4.1281,
                    "mean_best_accuracy": 0.16,
                    "mean_best_nll_delta_vs_raw": -0.0626,
                    "runs": 3,
                },
            ],
        }

        best_config = mod._best_config_summary([config])
        compact = mod._compact_config_summary(config)
        report = mod._aggregate_report(
            {
                "run": {"seed_count": 3},
                "seed_summaries": [],
                "config_summaries": [config],
                "scale_summaries": [config],
                "best_config": best_config,
                "ranking": config["ranking"],
                "feature_stability": [],
                "feature_family_stability": [],
                "status": "improved",
                "best_feature": "raw_latent",
            }
        )

        self.assertEqual(best_config["runner_up_feature"], "reconstruction_latent")
        self.assertAlmostEqual(best_config["margin_to_runner_up"], 0.0004)
        self.assertAlmostEqual(best_config["mean_best_nll_stderr"], 0.0003)
        self.assertAlmostEqual(
            best_config["combined_runner_up_margin_stderr"],
            0.0005,
        )
        self.assertIs(best_config["runner_up_within_uncertainty"], True)
        self.assertEqual(compact["runner_up_feature"], "reconstruction_latent")
        self.assertAlmostEqual(compact["margin_to_runner_up"], 0.0004)
        self.assertAlmostEqual(
            compact["runner_up_mean_best_nll_stderr"],
            0.0004,
        )
        self.assertIn("- best_config_runner_up: reconstruction_latent", report)
        self.assertIn("margin=0.000400", report)
        self.assertIn("combined_stderr=0.000500", report)
        self.assertIn("within_uncertainty=True", report)

    def test_capacity_grid_best_config_and_report_surface_capacity(self) -> None:
        mod = _load_module()
        small = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 1.0,
            "latent_dim": 4,
            "hidden": 8,
            "best_feature": "raw",
            "status": "ties_raw",
            "run_dir": "/tmp/capacity/small",
            "seed_count": 1,
            "ranking": [
                {
                    "feature": "raw",
                    "mean_best_nll": 4.20,
                    "mean_best_accuracy": 0.10,
                    "mean_best_nll_delta_vs_raw": 0.0,
                }
            ],
            "feature_stability": [{"feature": "raw", "win_count": 1}],
        }
        large = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 1.0,
            "latent_dim": 8,
            "hidden": 16,
            "best_feature": "raw_latent",
            "status": "improved",
            "run_dir": "/tmp/capacity/large",
            "seed_count": 1,
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.00,
                    "mean_best_accuracy": 0.12,
                    "mean_best_nll_delta_vs_raw": -0.20,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.10,
                    "mean_best_accuracy": 0.11,
                    "mean_best_nll_delta_vs_raw": 0.0,
                },
            ],
            "feature_stability": [{"feature": "raw_latent", "win_count": 1}],
        }

        best_config = mod._best_config_summary([small, large])
        report = mod._aggregate_report(
            {
                "run": {
                    "seed_count": 1,
                    "run_count": 2,
                    "config_count": 2,
                    "normalize_count": 1,
                    "scale_count": 1,
                    "latent_dim_count": 2,
                    "hidden_size_count": 2,
                    "capacity_count": 4,
                    "seeds": [7],
                    "features": ["raw", "raw_latent"],
                    "feature_normalize": "blocks",
                    "feature_normalize_modes": ["blocks"],
                    "hybrid_latent_scale": 1.0,
                    "hybrid_latent_scales": [1.0],
                    "latent_dim": None,
                    "latent_dims": [4, 8],
                    "hidden": None,
                    "hidden_sizes": [8, 16],
                },
                "status": "improved",
                "best_feature": "raw_latent",
                "best_config": best_config,
                "ranking": large["ranking"],
                "config_summaries": [small, large],
                "seed_summaries": [
                    {
                        "seed": 7,
                        "feature_normalize": "blocks",
                        "hybrid_latent_scale": 1.0,
                        "latent_dim": 8,
                        "hidden": 16,
                        "best_feature": "raw_latent",
                        "ranking": large["ranking"],
                        "run_dir": "/tmp/capacity/large/seed_000007",
                    }
                ],
                "seed_winners": [
                    {
                        "seed": 7,
                        "feature_normalize": "blocks",
                        "hybrid_latent_scale": 1.0,
                        "latent_dim": 8,
                        "hidden": 16,
                        "near_winners": ["raw_latent"],
                        "margin_to_runner_up": 0.10,
                    }
                ],
                "feature_stability": [],
                "feature_family_stability": [],
            }
        )

        self.assertEqual(best_config["latent_dim"], 8)
        self.assertEqual(best_config["hidden"], 16)
        self.assertIn("latent_dim=8 hidden=16", report)
        self.assertIn("- latent_dims: 4, 8", report)
        self.assertIn("- hidden_sizes: 8, 16", report)
        self.assertIn("| latent_dim | hidden | status |", report)
        self.assertIn("| latent_dim | hidden | seed |", report)

    def test_follow_up_defaults_inherit_best_capacity(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "summary.json"
            source.write_text(
                json.dumps(
                    {
                        "status": "improved",
                        "best_feature": "raw_latent",
                        "best_config": {
                            "best_feature": "raw_latent",
                            "feature_normalize": "blocks",
                            "hybrid_latent_scale": 2.0,
                            "latent_dim": 12,
                            "hidden": 24,
                            "mean_best_nll": 4.0,
                        },
                        "run": {
                            "features": ["raw", "raw_latent"],
                            "seeds": [7, 13, 17],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parser.parse_args(
                ["models/samples/spiral_corpus_en", "--follow-up-from", str(source)]
            )

            record = mod._apply_follow_up_defaults(args, [])

        self.assertEqual(args.latent_dim, 12)
        self.assertEqual(args.hidden, 24)
        self.assertIsNone(args.latent_dims)
        self.assertIsNone(args.hidden_sizes)
        self.assertEqual(record["applied_defaults"]["latent_dim"], 12)
        self.assertEqual(record["applied_defaults"]["hidden"], 24)

    def test_aggregate_recovers_curve_metrics_from_legacy_history(self) -> None:
        mod = _load_module()
        summary = {
            "run": {"seed": 1},
            "features": [
                {
                    "feature": "raw",
                    "best_epoch": 1,
                    "initial_validation": {"mean_nll": 4.30},
                    "best_validation": {"mean_nll": 4.10, "accuracy": 0.20},
                    "final_validation": {"mean_nll": 4.10},
                    "history": [
                        {"validation": {"mean_nll": 4.20}},
                        {"validation": {"mean_nll": 4.10}},
                    ],
                },
                {
                    "feature": "raw_latent",
                    "best_epoch": 1,
                    "initial_validation": {"mean_nll": 4.40},
                    "best_validation": {"mean_nll": 4.00, "accuracy": 0.25},
                    "final_validation": {"mean_nll": 4.00},
                    "history": [
                        {"validation": {"mean_nll": 4.15}},
                        {"validation": {"mean_nll": 4.00}},
                    ],
                },
            ],
            "ranking": [
                {"feature": "raw_latent", "best_mean_nll": 4.00, "best_accuracy": 0.25},
                {"feature": "raw", "best_mean_nll": 4.10, "best_accuracy": 0.20},
            ],
            "deltas": {
                "raw_best_nll_vs_raw": 0.0,
                "raw_latent_best_nll_vs_raw": -0.10,
            },
        }

        aggregate = mod._aggregate_summaries(
            [summary],
            min_nll_delta=0.0,
            win_tolerance=0.0001,
        )

        by_feature = {item["feature"]: item for item in aggregate["ranking"]}
        self.assertAlmostEqual(by_feature["raw_latent"]["mean_best_step"], 2.0)
        self.assertAlmostEqual(
            by_feature["raw_latent"]["mean_validation_nll_mean"],
            (4.15 + 4.00) / 2.0,
        )
        self.assertAlmostEqual(
            by_feature["raw_latent"]["mean_validation_nll_mean_delta_vs_raw"],
            ((4.15 + 4.00) / 2.0) - ((4.20 + 4.10) / 2.0),
        )
        self.assertAlmostEqual(
            by_feature["raw_latent"]["mean_validation_nll_initial_minus_best"],
            0.40,
        )
        self.assertAlmostEqual(
            by_feature["raw_latent"]["mean_validation_nll_final_minus_best"],
            0.0,
        )

    def test_single_learning_evidence_surfaces_checkpoint_and_raw_delta(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "text_vae_weights.bin"
            checkpoint.write_bytes(b"weights")
            raw_head = Path(tmp) / "head_raw_best.json"
            raw_head.write_text("{}", encoding="utf-8")
            hybrid_head = Path(tmp) / "head_raw_latent_best.json"
            hybrid_head.write_text("{}", encoding="utf-8")
            summary = {
                "run": {
                    "text_chars": 200,
                    "train_chars": 160,
                    "validation_chars": 40,
                    "window_chars": 16,
                    "eval_samples": 32,
                    "hidden": 8,
                    "head_init": "xavier",
                    "epochs": 2,
                    "batches": 3,
                    "features": ["raw", "raw_latent"],
                    "vae": {
                        "save_path": str(checkpoint),
                        "epochs": 2,
                        "batches": 4,
                        "history": [
                            {
                                "avg_recon_loss": 0.90,
                                "avg_weighted_loss": 0.92,
                                "avg_gradient_l2": 0.50,
                                "avg_update_l2": 0.10,
                            },
                            {
                                "avg_recon_loss": 0.70,
                                "avg_weighted_loss": 0.71,
                                "avg_gradient_l2": 0.40,
                                "avg_update_l2": 0.08,
                            },
                        ],
                    },
                },
                "ranking": [
                    {
                        "feature": "raw_latent",
                        "best_mean_nll": 3.80,
                        "best_accuracy": 0.25,
                        "validation_nll_mean_delta_vs_raw": -0.08,
                    },
                    {
                        "feature": "raw",
                        "best_mean_nll": 4.00,
                        "best_accuracy": 0.20,
                    },
                ],
                "deltas": {
                    "raw_latent_best_nll_vs_raw": -0.20,
                    "raw_best_nll_vs_raw": 0.0,
                },
                "best_feature": "raw_latent",
                "features": [
                    {
                        "feature": "raw",
                        "best_checkpoint": {
                            "path": str(raw_head),
                            "exists": True,
                            "epoch": 1,
                            "step": 2,
                            "source": "epoch_1",
                            "mean_nll": 4.00,
                        },
                    },
                    {
                        "feature": "raw_latent",
                        "best_checkpoint": {
                            "path": str(hybrid_head),
                            "exists": True,
                            "epoch": 1,
                            "step": 2,
                            "source": "epoch_1",
                            "mean_nll": 3.80,
                        },
                    },
                ],
                "feature_diagnostics": {"features": {}},
            }

            evidence = mod._single_learning_evidence(summary)
            summary["learning_evidence"] = evidence
            report = mod._single_report(summary)

        self.assertEqual(evidence["status"], "beats_raw")
        self.assertTrue(evidence["checkpoint"]["exists"])
        self.assertTrue(evidence["head"]["best_checkpoints"]["all_exist"])
        self.assertEqual(evidence["head"]["best_checkpoints"]["count"], 2)
        self.assertEqual(evidence["vae"]["steps"], 8)
        self.assertEqual(evidence["head"]["total_steps"], 12)
        self.assertAlmostEqual(
            evidence["vae"]["recon_loss"]["initial_minus_final"],
            0.20,
        )
        self.assertAlmostEqual(evidence["best"]["best_nll_delta_vs_raw"], -0.20)
        self.assertIn("## Learning Evidence", report)
        self.assertIn("- status: beats_raw", report)
        self.assertIn("- best_head_checkpoints: 2 all_exist=True missing=-", report)

    def test_head_load_path_defaults_to_best_or_final_checkpoint(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--head-load-dir",
                "/tmp/heads",
            ]
        )
        self.assertEqual(
            mod._head_load_path(args, "raw_latent"),
            Path("/tmp/heads/head_raw_latent_best.json"),
        )

        args = parser.parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--head-load-dir",
                "/tmp/heads",
                "--head-load-kind",
                "final",
            ]
        )
        self.assertEqual(
            mod._head_load_path(args, "raw_latent"),
            Path("/tmp/heads/head_raw_latent.json"),
        )

    def test_eval_only_requires_vae_and_head_sources(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["models/samples/spiral_corpus_en", "--eval-only"])
        args.epochs = 0
        args.vae_epochs = 0
        with self.assertRaisesRegex(ValueError, "--eval-only requires --vae-load"):
            mod._validate_args(args)

        args = parser.parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--eval-only",
                "--vae-load",
                "/tmp/text_vae_weights.bin",
            ]
        )
        args.epochs = 0
        args.vae_epochs = 0
        with self.assertRaisesRegex(ValueError, "--eval-only requires --head-load"):
            mod._validate_args(args)

    def test_eval_only_learning_evidence_uses_loaded_head_checkpoints(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vae_checkpoint = root / "text_vae_weights.bin"
            vae_checkpoint.write_bytes(b"weights")
            raw_head = root / "head_raw_best.json"
            raw_head.write_text("{}", encoding="utf-8")
            hybrid_head = root / "head_raw_latent_best.json"
            hybrid_head.write_text("{}", encoding="utf-8")
            summary = {
                "run": {
                    "text_chars": 200,
                    "train_chars": 160,
                    "validation_chars": 40,
                    "window_chars": 16,
                    "eval_samples": 32,
                    "hidden": 8,
                    "head_init": "xavier",
                    "eval_only": True,
                    "epochs": 0,
                    "batches": 3,
                    "features": ["raw", "raw_latent"],
                    "vae": {
                        "save_path": str(vae_checkpoint),
                        "load_path": str(vae_checkpoint),
                        "epochs": 0,
                        "batches": 4,
                        "history": [],
                    },
                },
                "ranking": [
                    {
                        "feature": "raw_latent",
                        "best_mean_nll": 3.80,
                        "best_accuracy": 0.25,
                        "validation_nll_mean_delta_vs_raw": -0.08,
                    },
                    {
                        "feature": "raw",
                        "best_mean_nll": 4.00,
                        "best_accuracy": 0.20,
                    },
                ],
                "deltas": {
                    "raw_latent_best_nll_vs_raw": -0.20,
                    "raw_best_nll_vs_raw": 0.0,
                },
                "best_feature": "raw_latent",
                "features": [
                    {
                        "feature": "raw",
                        "head_load": {
                            "loaded": True,
                            "path": str(raw_head),
                            "kind": "best",
                        },
                        "best_checkpoint": {
                            "path": str(raw_head),
                            "exists": True,
                            "source": "loaded",
                            "mean_nll": 4.00,
                        },
                    },
                    {
                        "feature": "raw_latent",
                        "head_load": {
                            "loaded": True,
                            "path": str(hybrid_head),
                            "kind": "best",
                        },
                        "best_checkpoint": {
                            "path": str(hybrid_head),
                            "exists": True,
                            "source": "loaded",
                            "mean_nll": 3.80,
                        },
                    },
                ],
                "feature_diagnostics": {"features": {}},
            }

            evidence = mod._single_learning_evidence(summary)
            summary["learning_evidence"] = evidence
            report = mod._single_report(summary)

        self.assertEqual(evidence["status"], "beats_raw")
        self.assertNotIn("vae_not_trained", evidence["reasons"])
        self.assertNotIn("heads_not_trained", evidence["reasons"])
        self.assertEqual(evidence["head"]["loaded_checkpoints"]["count"], 2)
        self.assertIn("- eval_only: True", report)
        self.assertIn("- loaded_head_checkpoints: 2", report)

    def test_text_vae_binding_preflight_explains_native_requirement(self) -> None:
        mod = _load_module()

        with self.assertRaisesRegex(RuntimeError, "maturin develop"):
            mod._require_zspace_text_vae_binding()

    def test_text_vae_binding_preflight_rejects_stale_text_vae_surface(self) -> None:
        mod = _load_module()
        original_st = mod.st
        fake_text_vae = type(
            "ZSpaceTextVae",
            (),
            {
                "load": staticmethod(lambda path: None),
                "save": lambda self, path: None,
                "encode_text": lambda self, text: [],
                "encode_text_with_mellin": lambda self, text, basis: [],
            },
        )
        fake_nn = types.SimpleNamespace(
            ZSpaceTextVae=fake_text_vae,
            MellinBasis=object,
            Sequential=object,
            Linear=object,
            Relu=object,
            ZSpaceSoftmax=object,
            ModuleTrainer=object,
            CategoricalCrossEntropy=object,
            RoundtableConfig=object,
            save=lambda path, target: None,
        )
        mod.st = types.SimpleNamespace(nn=fake_nn)
        try:
            with self.assertRaisesRegex(RuntimeError, "ZSpaceTextVae.forward_mean_text"):
                mod._require_zspace_text_vae_binding()
        finally:
            mod.st = original_st

    def test_text_vae_binding_preflight_accepts_complete_training_surface(self) -> None:
        mod = _load_module()
        original_st = mod.st
        fake_text_vae = type(
            "ZSpaceTextVae",
            (),
            {
                method: (lambda *args, **kwargs: None)
                for method in mod.REQUIRED_TEXT_VAE_METHODS
            },
        )
        fake_nn = types.SimpleNamespace(
            **{symbol: object for symbol in mod.REQUIRED_NN_BINDING_SYMBOLS}
        )
        fake_nn.ZSpaceTextVae = fake_text_vae
        mod.st = types.SimpleNamespace(nn=fake_nn)
        try:
            mod._require_zspace_text_vae_binding()
        finally:
            mod.st = original_st

    def test_aggregate_learning_evidence_marks_promising_sweep(self) -> None:
        mod = _load_module()
        summary = {
            "run": {
                "seed_count": 3,
                "run_count": 6,
                "config_count": 2,
                "normalize_count": 1,
                "scale_count": 2,
                "seeds": [1, 2, 3],
                "features": ["raw", "raw_latent"],
                "feature_normalize": "blocks",
                "feature_normalize_modes": ["blocks"],
                "hybrid_latent_scale": None,
                "hybrid_latent_scales": [1.0, 2.0],
                "head_init": "xavier",
                "min_nll_delta": 0.0,
                "follow_up_confirm_tolerance": 0.0,
                "win_tolerance": 0.001,
            },
            "status": "improved",
            "best_feature": "raw_latent",
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 3.80,
                    "mean_best_accuracy": 0.25,
                    "mean_best_nll_delta_vs_raw": -0.20,
                    "runs": 6,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.00,
                    "mean_best_accuracy": 0.20,
                    "mean_best_nll_delta_vs_raw": 0.0,
                    "runs": 6,
                },
            ],
            "feature_stability": [
                {
                    "feature": "raw_latent",
                    "win_count": 4,
                    "win_rate": 4 / 6,
                    "near_win_count": 5,
                    "near_win_rate": 5 / 6,
                }
            ],
            "feature_family_stability": [],
            "feature_diagnostics_summary": [],
            "seed_summaries": [
                {
                    "seed": 1,
                    "feature_normalize": "blocks",
                    "hybrid_latent_scale": 1.0,
                    "best_feature": "raw_latent",
                    "ranking": [],
                    "run_dir": "/tmp/seed_1",
                }
            ],
            "config_summaries": [
                {
                    "feature_normalize": "blocks",
                    "hybrid_latent_scale": 2.0,
                    "status": "improved",
                    "best_feature": "raw_latent",
                    "ranking": [
                        {
                            "feature": "raw_latent",
                            "mean_best_nll": 3.80,
                            "mean_best_nll_delta_vs_raw": -0.20,
                        }
                    ],
                    "feature_stability": [
                        {"feature": "raw_latent", "win_count": 2}
                    ],
                    "seed_count": 3,
                    "run_dir": "/tmp/config",
                }
            ],
            "best_config": {
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 2.0,
                "best_feature": "raw_latent",
                "mean_best_nll": 3.80,
                "mean_best_nll_delta_vs_raw": -0.20,
                "runner_up_feature": "raw",
                "margin_to_runner_up": 0.20,
                "runner_up_within_uncertainty": False,
            },
            "seed_winners": [],
        }

        evidence = mod._aggregate_learning_evidence(summary)
        summary["learning_evidence"] = evidence
        report = mod._aggregate_report(summary)

        self.assertEqual(evidence["status"], "promising")
        self.assertEqual(evidence["coverage"]["run_count"], 6)
        self.assertTrue(evidence["raw_baseline"]["present"])
        self.assertTrue(evidence["follow_up_ready"])
        self.assertAlmostEqual(
            evidence["best"]["mean_best_nll_delta_vs_raw"],
            -0.20,
        )
        self.assertIn("- learning_status: promising", report)
        self.assertIn("## Learning Evidence", report)

    def test_aggregate_learning_evidence_promotes_stable_hybrid_family(self) -> None:
        mod = _load_module()
        summary = {
            "run": {
                "seed_count": 3,
                "run_count": 12,
                "config_count": 4,
                "normalize_count": 1,
                "scale_count": 1,
                "capacity_count": 1,
                "features": [
                    "raw",
                    "latent",
                    "raw_latent",
                    "reconstruction_latent",
                ],
            },
            "status": "improved",
            "best_feature": "raw_latent",
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.1023,
                    "mean_best_accuracy": 0.165,
                    "mean_best_nll_delta_vs_raw": -0.0867,
                },
                {
                    "feature": "reconstruction_latent",
                    "mean_best_nll": 4.1025,
                    "mean_best_accuracy": 0.164,
                    "mean_best_nll_delta_vs_raw": -0.0865,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.1890,
                    "mean_best_accuracy": 0.142,
                    "mean_best_nll_delta_vs_raw": 0.0,
                },
            ],
            "feature_stability": [
                {
                    "feature": "raw_latent",
                    "win_count": 1,
                    "win_rate": 1 / 3,
                    "near_win_count": 1,
                    "near_win_rate": 1 / 3,
                }
            ],
            "feature_family_stability": [
                {
                    "family": "hybrid_latent",
                    "win_count": 3,
                    "win_rate": 1.0,
                    "near_win_count": 3,
                    "near_win_rate": 1.0,
                    "mean_best_nll": 4.1020,
                    "mean_best_accuracy": 0.165,
                    "mean_best_nll_delta_vs_raw": -0.0870,
                    "mean_rank": 1.0,
                    "mean_gap_to_winner": 0.0,
                    "member_best_counts": {
                        "reconstruction_latent": 2,
                        "raw_latent": 1,
                    },
                },
                {
                    "family": "raw",
                    "win_count": 0,
                    "win_rate": 0.0,
                    "near_win_count": 0,
                    "near_win_rate": 0.0,
                    "mean_best_nll_delta_vs_raw": 0.0,
                    "member_best_counts": {},
                },
            ],
            "feature_diagnostics_summary": [],
            "seed_summaries": [{"seed": 211}, {"seed": 223}, {"seed": 227}],
            "config_summaries": [{"status": "improved"} for _ in range(4)],
            "best_config": {
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "latent_dim": 12,
                "hidden": 64,
                "best_feature": "raw_latent",
                "mean_best_nll": 4.1023,
                "mean_best_nll_delta_vs_raw": -0.0867,
                "runner_up_feature": "reconstruction_latent",
                "margin_to_runner_up": 0.0002,
                "runner_up_within_uncertainty": True,
            },
            "seed_winners": [],
        }

        evidence = mod._aggregate_learning_evidence(summary)
        summary["learning_evidence"] = evidence
        report = mod._aggregate_report(summary)

        self.assertEqual(evidence["status"], "promising_family_stable")
        self.assertEqual(evidence["best_family"]["family"], "hybrid_latent")
        self.assertAlmostEqual(
            evidence["best_family"]["mean_best_nll_delta_vs_raw"],
            -0.0870,
        )
        self.assertEqual(
            evidence["best_family"]["member_best_counts"],
            {"reconstruction_latent": 2, "raw_latent": 1},
        )
        self.assertIn("- learning_status: promising_family_stable", report)
        self.assertIn(
            "- best_family_vs_raw: hybrid_latent mean_delta=-0.087000 "
            "win_rate=100.00% near_win_rate=100.00% "
            "best_members=reconstruction_latent=2, raw_latent=1",
            report,
        )

    def test_follow_up_result_reports_run_budget_shift(self) -> None:
        mod = _load_module()
        source_best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "mean_best_nll": 4.164,
        }
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "seed_001"
            seed_dir.mkdir()
            (seed_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run": {
                            "window_chars": 32,
                            "latent_dim": 8,
                            "hidden": 16,
                            "epochs": 8,
                            "batches": 16,
                            "batch_size": 4,
                            "eval_samples": 128,
                            "vae": {
                                "epochs": 8,
                                "batches": 16,
                                "batch_size": 4,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            source_summary = {
                "run": {
                    "seeds": [1],
                    "window_chars": 32,
                    "latent_dim": 8,
                    "hidden": 16,
                    "epochs": 8,
                    "batches": 16,
                    "batch_size": 4,
                    "eval_samples": 128,
                },
                "seed_summaries": [{"run_dir": str(seed_dir)}],
                "best_config": source_best_config,
            }
            source_budget = mod._summary_run_budget(source_summary)
            self.assertEqual(source_budget["vae_epochs"], 8)

        current_budget = {
            "window_chars": 32,
            "latent_dim": 8,
            "hidden": 16,
            "epochs": 16,
            "batches": 32,
            "batch_size": 4,
            "eval_samples": 256,
            "vae_epochs": 16,
            "vae_batches": 32,
            "vae_batch_size": 4,
        }
        config_summary = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "best_feature": "raw_latent",
            "status": "improved",
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.184,
                    "mean_best_accuracy": 0.2,
                    "mean_best_nll_delta_vs_raw": -0.003,
                    "runs": 9,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.187,
                    "mean_best_accuracy": 0.2,
                    "mean_best_nll_delta_vs_raw": 0.0,
                    "runs": 9,
                },
            ],
        }
        result = mod._follow_up_result(
            {
                "source_summary_path": "/tmp/source/summary.json",
                "source_best_config": source_best_config,
                "source_run_budget": source_budget,
            },
            [config_summary],
            {
                "best_feature": "raw_latent",
                "mean_best_nll_delta_vs_raw": -0.003,
            },
            min_nll_delta=0.0,
            current_run_budget=current_budget,
        )

        self.assertEqual(result["verdict"], "regressed")
        self.assertEqual(result["source_feature_raw_verdict"], "improved")
        self.assertIs(result["run_budget_shifted"], True)
        self.assertIn("eval_samples", result["run_budget_shift_keys"])
        self.assertIn("epochs", result["run_budget_shift_keys"])
        self.assertIn("vae_epochs", result["run_budget_shift_keys"])
        self.assertIn(
            "eval_samples:128->256",
            mod._run_budget_shift_label(result["run_budget_shift"]),
        )
        gate = mod._follow_up_gate_record(result, ["regressed", "unknown"])
        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate["verdict"], "regressed")
        self.assertEqual(gate["effective_verdict"], "improved")
        self.assertEqual(
            gate["verdict_basis"],
            "source_feature_raw_verdict_after_run_budget_shift",
        )
        self.assertIs(gate["failed"], False)
        self.assertEqual(gate["exit_code"], 0)

        args = mod._build_parser().parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--epochs",
                "16",
                "--batches",
                "32",
                "--eval-samples",
                "256",
                "--window-chars",
                "32",
                "--latent-dim",
                "8",
                "--hidden",
                "16",
                "--batch-size",
                "4",
                "--vae-epochs",
                "16",
                "--vae-batches",
                "32",
                "--vae-batch-size",
                "4",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            record = mod._best_generation_follow_up_command_record(
                args,
                ["raw", "latent", "raw_latent", "reconstruction_latent"],
                Path(tmp),
                [1061, 1063],
                {
                    "trajectory_action": "reconfirm_best_raw_positive_generation",
                    "best_config": source_best_config,
                    "best_summary_path": "/tmp/source/summary.json",
                    "best_generation": 6,
                },
                {"used_seed_history": [1, 1061]},
                result,
            )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertIs(record["source_budget_matched"], True)
        self.assertEqual(record["command_run_budget"]["epochs"], 8)
        self.assertEqual(record["command_run_budget"]["batches"], 16)
        self.assertEqual(record["command_run_budget"]["eval_samples"], 128)
        self.assertEqual(record["command_run_budget"]["vae_epochs"], 8)
        self.assertEqual(record["command_run_budget"]["vae_batches"], 16)
        script_command = record["script_command"]
        self.assertEqual(script_command[script_command.index("--epochs") + 1], "8")
        self.assertEqual(script_command[script_command.index("--batches") + 1], "16")
        self.assertEqual(
            script_command[script_command.index("--eval-samples") + 1],
            "128",
        )
        self.assertEqual(
            script_command[script_command.index("--vae-epochs") + 1],
            "8",
        )
        self.assertEqual(
            script_command[script_command.index("--vae-batches") + 1],
            "16",
        )

    def test_follow_up_source_budget_prefers_best_config_capacity(self) -> None:
        mod = _load_module()
        source_best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "latent_dim": 12,
            "hidden": 32,
            "mean_best_nll": 4.164,
        }
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "scale_4" / "latent_000006" / "hidden_000008"
            seed_dir.mkdir(parents=True)
            (seed_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "run": {
                            "window_chars": 32,
                            "latent_dim": 6,
                            "hidden": 8,
                            "epochs": 6,
                            "batches": 12,
                            "batch_size": 4,
                            "eval_samples": 128,
                            "vae": {
                                "epochs": 6,
                                "batches": 12,
                                "batch_size": 4,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            source_summary = {
                "run": {
                    "window_chars": 32,
                    "latent_dim": None,
                    "hidden": None,
                    "epochs": 6,
                    "batches": 12,
                    "batch_size": 4,
                    "eval_samples": 128,
                    "vae_epochs": 6,
                    "vae_batches": 12,
                    "vae_batch_size": 4,
                    "latent_dims": [6, 8, 12],
                    "hidden_sizes": [8, 16, 32],
                },
                "seed_summaries": [{"run_dir": str(seed_dir)}],
                "best_config": source_best_config,
            }
            fallback_budget = mod._summary_run_budget(source_summary)
            source_budget = mod._summary_run_budget_for_best_config(
                source_summary,
                source_best_config,
            )

        self.assertEqual(fallback_budget["latent_dim"], 6)
        self.assertEqual(fallback_budget["hidden"], 8)
        self.assertEqual(source_budget["latent_dim"], 12)
        self.assertEqual(source_budget["hidden"], 32)

        config_summary = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "latent_dim": 12,
            "hidden": 32,
            "best_feature": "raw_latent",
            "status": "improved",
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.172,
                    "mean_best_accuracy": 0.2,
                    "mean_best_nll_delta_vs_raw": -0.03,
                    "runs": 5,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.202,
                    "mean_best_accuracy": 0.2,
                    "mean_best_nll_delta_vs_raw": 0.0,
                    "runs": 5,
                },
            ],
        }
        result = mod._follow_up_result(
            {
                "source_summary_path": "/tmp/source/summary.json",
                "source_best_config": source_best_config,
                "source_run_budget": source_budget,
            },
            [config_summary],
            {
                "best_feature": "raw_latent",
                "mean_best_nll_delta_vs_raw": -0.03,
            },
            min_nll_delta=0.0,
            current_run_budget=dict(source_budget),
        )

        self.assertIs(result["run_budget_shifted"], False)
        self.assertEqual(result["evaluated_config"]["latent_dim"], 12)
        self.assertEqual(result["evaluated_config"]["hidden"], 32)
        self.assertEqual(result["source_feature_evaluated"]["latent_dim"], 12)
        self.assertEqual(result["source_feature_evaluated"]["hidden"], 32)

    def test_follow_up_result_confirms_source_delta_inside_seed_noise(self) -> None:
        mod = _load_module()
        source_best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "mean_best_nll": 4.164274733570908,
        }
        config_summary = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "best_feature": "raw_latent",
            "status": "improved",
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.165183979512166,
                    "mean_best_accuracy": 0.16,
                    "mean_best_nll_delta_vs_raw": -0.03029274605893484,
                    "runs": 9,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.195476725571101,
                    "mean_best_accuracy": 0.16,
                    "mean_best_nll_delta_vs_raw": 0.0,
                    "runs": 9,
                },
            ],
            "seed_summaries": [
                {
                    "ranking": [
                        {"feature": "raw_latent", "best_mean_nll": 4.162},
                        {"feature": "raw", "best_mean_nll": 4.195},
                    ],
                },
                {
                    "ranking": [
                        {"feature": "raw_latent", "best_mean_nll": 4.165183979512166},
                        {"feature": "raw", "best_mean_nll": 4.196},
                    ],
                },
                {
                    "ranking": [
                        {"feature": "raw_latent", "best_mean_nll": 4.1684},
                        {"feature": "raw", "best_mean_nll": 4.197},
                    ],
                },
            ],
        }

        result = mod._follow_up_result(
            {
                "source_summary_path": "/tmp/source/summary.json",
                "source_best_config": source_best_config,
                "source_run_budget": {},
            },
            [config_summary],
            {
                "best_feature": "raw_latent",
                "mean_best_nll_delta_vs_raw": -0.03029274605893484,
            },
            min_nll_delta=0.0,
        )

        self.assertEqual(result["source_feature_verdict"], "confirmed")
        self.assertEqual(result["config_verdict"], "confirmed")
        self.assertEqual(result["verdict"], "confirmed")
        self.assertEqual(result["source_feature_raw_verdict"], "improved")
        self.assertGreater(
            result["effective_source_feature_min_nll_delta"],
            result["source_feature_mean_best_nll_delta_vs_source"],
        )
        self.assertAlmostEqual(
            result["source_feature_mean_best_nll_delta_vs_source"],
            0.0009092459412585185,
        )
        gate = mod._follow_up_gate_record(result, ["regressed", "unknown"])
        self.assertIs(gate["failed"], False)

    def test_follow_up_guidance_accepts_same_family_feature_swap(self) -> None:
        mod = _load_module()
        source_best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "latent_dim": 12,
            "hidden": 64,
            "mean_best_nll": 4.1023,
            "mean_best_nll_delta_vs_raw": -0.0867,
        }
        config_summary = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "latent_dim": 12,
            "hidden": 64,
            "best_feature": "reconstruction_latent",
            "status": "improved",
            "ranking": [
                {
                    "feature": "reconstruction_latent",
                    "mean_best_nll": 4.0891,
                    "mean_best_nll_delta_vs_raw": -0.1002,
                },
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.0905,
                    "mean_best_nll_delta_vs_raw": -0.0989,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.1893,
                    "mean_best_nll_delta_vs_raw": 0.0,
                },
            ],
        }

        result = mod._follow_up_result(
            {
                "source_summary_path": "/tmp/source/summary.json",
                "source_best_config": source_best_config,
                "source_run_budget": {},
            },
            [config_summary],
            {
                "best_feature": "reconstruction_latent",
                "mean_best_nll_delta_vs_raw": -0.1002,
            },
            min_nll_delta=0.0,
        )
        gate = mod._follow_up_gate_record(result, ["regressed", "unknown"])
        chain = {
            "generation": 1,
            "latest_verdict": result["verdict"],
            "verdict_history": [result["verdict"]],
            "improved_streak": 1 if result["verdict"] == "improved" else 0,
            "regressed_streak": 0,
        }
        guidance = mod._follow_up_guidance_record(
            result,
            chain,
            gate,
            _next_follow_up(),
        )

        self.assertIs(result["source_best_feature_retained"], False)
        self.assertIs(result["source_best_family_retained"], True)
        self.assertEqual(result["source_feature_verdict"], "improved")
        self.assertEqual(guidance["action"], "continue_fresh_seed_confirmation")
        self.assertTrue(guidance["promote_current_best"])
        self.assertTrue(guidance["use_next_follow_up_command"])
        self.assertIn(
            "source best feature stayed within the same family",
            guidance["reasons"],
        )
        self.assertNotIn(
            "source best feature did not retain its role",
            guidance["reasons"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {
                "status": "improved",
                "best_feature": "reconstruction_latent",
                "best_config": result["current_best_config"],
                "follow_up_result": result,
                "follow_up_chain": chain,
                "follow_up_gate": gate,
                "follow_up_guidance": guidance,
            }
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {
                    "ancestors": [
                        _ancestor(
                            feature="raw_latent",
                            nll=4.1023,
                            raw_delta=-0.0867,
                        )
                    ]
                },
                min_nll_delta=0.0,
            )

        self.assertIs(trajectory["source_feature_tradeoff"], False)
        self.assertIs(trajectory["unsafe_promotion"], False)
        self.assertIs(trajectory["current_raw_positive"], True)

    def test_confirmed_same_family_streak_promotes_candidate(self) -> None:
        mod = _load_module()
        result = {
            "verdict": "confirmed",
            "config_verdict": "confirmed",
            "source_feature_verdict": "confirmed",
            "source_feature_raw_verdict": "improved",
            "current_best_raw_verdict": "improved",
            "source_best_feature_retained": False,
            "source_best_family_retained": True,
            "current_best_config": {
                "best_feature": "raw_latent",
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "mean_best_nll": 4.0879,
                "mean_best_nll_delta_vs_raw": -0.1033,
                "runner_up_feature": "reconstruction_latent",
                "margin_to_runner_up": 0.00015,
                "combined_runner_up_margin_stderr": 0.0258,
                "runner_up_within_uncertainty": True,
            },
        }
        chain = {
            "generation": 2,
            "latest_verdict": "confirmed",
            "verdict_history": ["confirmed", "confirmed"],
            "improved_streak": 0,
            "confirmed_streak": 2,
            "regressed_streak": 0,
        }
        guidance = mod._follow_up_guidance_record(
            result,
            chain,
            {"failed": False},
            _next_follow_up(),
            {
                "trajectory_action": "continue_or_audit_mixed_trajectory",
                "trajectory_verdict": "improved",
                "trajectory_reasons": [
                    "trajectory improved but latest verdict is not improved"
                ],
                "unsafe_promotion": False,
            },
        )

        self.assertEqual(
            guidance["action"],
            "promote_stable_family_confirmation",
        )
        self.assertEqual(guidance["confirmed_streak"], 2)
        self.assertTrue(guidance["promote_current_best"])
        self.assertFalse(guidance["use_next_follow_up_command"])
        self.assertIsNone(guidance["command_usage"])
        self.assertIn("family confirmation streak=2", guidance["reasons"])
        self.assertIn(
            "confirmed family candidate is raw-positive",
            guidance["reasons"],
        )
        self.assertIn(
            "source best feature stayed within the same family",
            guidance["reasons"],
        )

    def test_follow_up_result_combines_source_and_current_seed_noise(self) -> None:
        mod = _load_module()
        source_best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "mean_best_nll": 4.100,
            "mean_best_nll_stderr": 0.009,
        }
        config_summary = {
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "best_feature": "raw_latent",
            "status": "improved",
            "ranking": [
                {
                    "feature": "raw_latent",
                    "mean_best_nll": 4.107,
                    "mean_best_accuracy": 0.16,
                    "mean_best_nll_delta_vs_raw": -0.020,
                    "runs": 3,
                },
                {
                    "feature": "raw",
                    "mean_best_nll": 4.127,
                    "mean_best_accuracy": 0.16,
                    "mean_best_nll_delta_vs_raw": 0.0,
                    "runs": 3,
                },
            ],
            "feature_summary": [
                {
                    "feature": "raw_latent",
                    "best_nll": {
                        "count": 3,
                        "stddev": 0.006,
                        "stderr": 0.003,
                    },
                }
            ],
        }

        result = mod._follow_up_result(
            {
                "source_summary_path": "/tmp/source/summary.json",
                "source_best_config": source_best_config,
                "source_run_budget": {},
            },
            [config_summary],
            {
                "best_feature": "raw_latent",
                "mean_best_nll_delta_vs_raw": -0.020,
            },
            min_nll_delta=0.0,
        )

        self.assertAlmostEqual(result["source_mean_best_nll_stderr"], 0.009)
        self.assertAlmostEqual(result["source_feature_mean_best_nll_stderr"], 0.003)
        self.assertAlmostEqual(
            result["combined_source_feature_mean_best_nll_stderr"],
            (0.009**2 + 0.003**2) ** 0.5,
        )
        self.assertEqual(result["source_feature_verdict"], "confirmed")
        self.assertEqual(result["verdict"], "confirmed")
        self.assertEqual(result["source_feature_raw_verdict"], "improved")

    def test_safe_trajectory_promotes_guided_confirmation(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="reconstruction_latent",
            nll=4.2,
            verdict="improved",
            config_verdict="improved",
            source_feature_verdict="improved",
            source_retained=True,
            gate_failed=False,
        )
        summary["follow_up_result"]["current_best_config"] = {
            "best_feature": "reconstruction_latent",
            "runner_up_feature": "raw_latent",
            "margin_to_runner_up": 0.0004,
            "combined_runner_up_margin_stderr": 0.0005,
            "runner_up_within_uncertainty": True,
        }
        next_follow_up = _next_follow_up()
        preliminary = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            next_follow_up,
        )
        summary["follow_up_guidance"] = preliminary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {"ancestors": [_ancestor(feature="reconstruction_latent", nll=4.3)]},
                min_nll_delta=0.0,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                next_follow_up,
            )
            mod._write_guided_next_follow_up_script(guided)
            guided_script = Path(guided["script_path"]).read_text(encoding="utf-8")
            report_summary = dict(summary)
            report_summary["follow_up_trajectory"] = trajectory
            report_summary["follow_up_guidance"] = guidance
            report_summary["guided_next_follow_up_command"] = guided
            report = mod._aggregate_report(report_summary)

        self.assertEqual(
            trajectory["trajectory_action"],
            "confirm_trajectory_with_fresh_seeds",
        )
        self.assertIs(trajectory["unsafe_promotion"], False)
        self.assertEqual(guidance["local_action"], "continue_fresh_seed_confirmation")
        self.assertEqual(guidance["action"], "confirm_trajectory_with_fresh_seeds")
        self.assertIs(guidance["promote_current_best"], True)
        self.assertIs(guidance["use_next_follow_up_command"], True)
        self.assertIs(guidance["tie_aware_confirmation"], True)
        self.assertEqual(
            guidance["best_config_uncertainty_tie"]["runner_up_feature"],
            "raw_latent",
        )
        self.assertTrue(
            any(
                "best runner-up within combined seed uncertainty" in reason
                for reason in guidance["reasons"]
            ),
        )
        self.assertIs(guided["enabled"], True)
        self.assertIs(guided["tie_aware_confirmation"], True)
        self.assertEqual(
            guided["trajectory_action"],
            "confirm_trajectory_with_fresh_seeds",
        )
        self.assertIs(guided["unsafe_promotion"], False)
        self.assertIn("# Tie-aware confirmation: True", guided_script)
        self.assertIn("- tie_aware_confirmation: True", report)
        self.assertIn(
            "reconstruction_latent vs raw_latent margin=0.000400",
            report,
        )

    def test_unsafe_trajectory_blocks_guided_promotion(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="latent",
            nll=4.1,
            verdict="regressed",
            config_verdict="improved",
            source_feature_verdict="regressed",
            source_retained=False,
            gate_failed=True,
        )
        next_follow_up = _next_follow_up()
        preliminary = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            next_follow_up,
        )
        summary["follow_up_guidance"] = preliminary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {"ancestors": [_ancestor(feature="reconstruction_latent", nll=4.3)]},
                min_nll_delta=0.0,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                next_follow_up,
            )

        self.assertEqual(trajectory["trajectory_verdict"], "improved")
        self.assertEqual(
            trajectory["trajectory_action"],
            "audit_feature_swap_before_promotion",
        )
        self.assertIs(trajectory["source_feature_tradeoff"], True)
        self.assertIs(trajectory["unsafe_promotion"], True)
        self.assertEqual(guidance["local_action"], "stop_on_follow_up_gate")
        self.assertEqual(guidance["action"], "audit_feature_swap_before_promotion")
        self.assertIs(guidance["promote_current_best"], False)
        self.assertIs(guidance["use_next_follow_up_command"], False)
        self.assertIn("trajectory marked promotion unsafe", guidance["reasons"])
        self.assertIs(guided["enabled"], False)
        self.assertIsNone(guided["script_path"])
        self.assertEqual(
            guided["trajectory_action"],
            "audit_feature_swap_before_promotion",
        )
        self.assertIs(guided["unsafe_promotion"], True)

    def test_feature_swap_review_command_matches_source_budget(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--epochs",
                "2",
                "--batches",
                "2",
                "--eval-samples",
                "22",
                "--vae-epochs",
                "2",
                "--vae-batches",
                "2",
            ]
        )
        summary = _summary(
            best_feature="latent",
            nll=3.802,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_feature_raw_verdict="improved",
            source_retained=True,
            gate_failed=False,
        )
        summary["follow_up_result"].update(
            {
                "current_best_config": {
                    "best_feature": "latent",
                    "feature_normalize": "blocks",
                    "hybrid_latent_scale": 0.5,
                    "mean_best_nll": 3.802,
                },
                "source_best_config": {
                    "best_feature": "latent",
                    "feature_normalize": "blocks",
                    "hybrid_latent_scale": 0.5,
                    "mean_best_nll": 3.792,
                },
                "run_budget_shifted": True,
                "source_run_budget": {
                    "epochs": 1,
                    "batches": 1,
                    "eval_samples": 6,
                    "vae_epochs": 1,
                    "vae_batches": 1,
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = mod._feature_swap_review_command_record(
                args,
                ["raw", "latent", "raw_latent"],
                root,
                [131, 137, 139],
                summary["follow_up_result"],
                _next_follow_up(),
            )
            mod._write_next_follow_up_script(record)
            script_text = Path(record["script_path"]).read_text(encoding="utf-8")

        self.assertEqual(record["action"], "review_feature_swap_before_promotion")
        self.assertIs(record["source_budget_matched"], True)
        self.assertEqual(record["command_run_budget"]["epochs"], 1)
        self.assertEqual(record["command_run_budget"]["eval_samples"], 6)
        self.assertEqual(record["command_run_budget"]["vae_epochs"], 1)
        self.assertEqual(record["default_new_seeds"], "101,103,107,109,113")
        self.assertEqual(record["default_run_dir"], str(root / "feature_swap_review"))
        self.assertEqual(record["default_follow_up_from"], str(root / "summary.json"))
        self.assertIn("--epochs", record["script_command"])
        self.assertIn("1", record["script_command"])
        self.assertIn("--eval-samples", record["script_command"])
        self.assertIn("6", record["script_command"])
        self.assertIn("feature_swap_review", record["script_usage"])
        self.assertIn("NEXT_RUN_DIR", script_text)

    def test_retained_source_feature_regression_has_precise_reason(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="latent",
            nll=4.23,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_retained=True,
            gate_failed=True,
        )

        guidance = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            _next_follow_up(),
        )

        self.assertEqual(guidance["action"], "stop_on_follow_up_gate")
        self.assertIn(
            "source best feature regressed on fresh seeds",
            guidance["reasons"],
        )
        self.assertNotIn(
            "source best feature did not retain its role",
            guidance["reasons"],
        )

    def test_raw_positive_gate_stop_widens_seed_confirmation(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="raw_latent",
            nll=4.223,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_feature_raw_verdict="improved",
            source_feature_delta_vs_raw=-0.008,
            source_retained=True,
            gate_failed=True,
        )
        next_follow_up = _next_follow_up()
        preliminary = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            next_follow_up,
        )
        summary["follow_up_guidance"] = preliminary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {"ancestors": [_ancestor(feature="raw_latent", nll=4.216)]},
                min_nll_delta=0.0,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                next_follow_up,
            )

        self.assertEqual(
            preliminary["action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )
        self.assertEqual(
            trajectory["trajectory_action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )
        self.assertIs(trajectory["current_raw_positive"], True)
        self.assertIs(trajectory["unsafe_promotion"], False)
        self.assertEqual(trajectory["raw_evidence_count"], 1)
        self.assertEqual(trajectory["raw_positive_count"], 1)
        self.assertEqual(trajectory["raw_positive_rate"], 1.0)
        self.assertEqual(trajectory["raw_positive_streak"], 1)
        self.assertEqual(trajectory["raw_negative_streak"], 0)
        self.assertAlmostEqual(trajectory["current_raw_delta_vs_raw"], -0.008)
        self.assertAlmostEqual(trajectory["mean_raw_delta_vs_raw"], -0.008)
        self.assertIs(trajectory["points"][-1]["raw_positive"], True)
        self.assertEqual(
            guidance["action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )
        self.assertIs(guidance["promote_current_best"], False)
        self.assertIs(guidance["use_next_follow_up_command"], True)
        self.assertIn("source best feature remained raw-positive", guidance["reasons"])
        self.assertIs(guided["enabled"], True)
        self.assertEqual(
            guided["guidance_action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )

        summary["follow_up_trajectory"] = trajectory
        report = mod._aggregate_report(summary)
        self.assertIn("- raw_positive_points: 1/1 (100.0%)", report)
        self.assertIn("raw_delta_vs_raw", report)

    def test_raw_positive_streak_reconfirms_best_generation(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["models/samples/spiral_corpus_en"])
        summary = _summary(
            best_feature="raw_latent",
            nll=4.214,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_feature_raw_verdict="improved",
            source_feature_delta_vs_raw=-0.017,
            source_retained=True,
            gate_failed=True,
        )
        summary["follow_up_chain"]["verdict_history"] = [
            "regressed",
            "improved",
            "regressed",
            "regressed",
        ]
        summary["follow_up_chain"]["latest_verdict"] = "regressed"
        next_follow_up = _next_follow_up()
        next_follow_up["used_seed_history"] = [
            7,
            13,
            17,
            19,
            23,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            1001,
            1003,
            1005,
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {
                    "ancestors": [
                        _ancestor(feature="raw_latent", nll=4.216, raw_delta=-0.014),
                        _ancestor(feature="raw_latent", nll=4.207, raw_delta=-0.024),
                        _ancestor(feature="raw_latent", nll=4.211, raw_delta=-0.019),
                    ]
                },
                min_nll_delta=0.0,
            )
            best_generation = mod._best_generation_follow_up_command_record(
                args,
                ["raw", "latent", "raw_latent"],
                root,
                [1001, 1003, 1005],
                trajectory,
                next_follow_up,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
                best_generation,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                best_generation,
            )

        self.assertEqual(
            trajectory["trajectory_action"],
            "reconfirm_best_raw_positive_generation",
        )
        self.assertEqual(trajectory["raw_positive_streak"], 4)
        self.assertEqual(trajectory["best_mean_best_nll"], 4.207)
        self.assertIsNotNone(best_generation)
        self.assertEqual(
            best_generation["action"],
            "reconfirm_best_raw_positive_generation",
        )
        self.assertEqual(best_generation["default_new_seeds"], "1007,1009,1011")
        self.assertEqual(
            best_generation["best_summary_path"],
            "/tmp/source/summary.json",
        )
        self.assertIn("--follow-up-used-seeds", best_generation["script_command"])
        self.assertIn(
            (
                "7,13,17,19,23,101,103,107,109,113,127,131,137,139,"
                "149,151,157,1001,1003,1005"
            ),
            best_generation["script_command"],
        )
        self.assertEqual(guidance["action"], "reconfirm_best_raw_positive_generation")
        self.assertIs(guidance["use_next_follow_up_command"], False)
        self.assertIs(guidance["use_best_generation_follow_up_command"], True)
        self.assertEqual(guidance["command_usage"], best_generation["script_usage"])
        self.assertIs(guided["enabled"], True)
        self.assertEqual(guided["default_follow_up_from"], "/tmp/source/summary.json")
        self.assertEqual(guided["default_new_seeds"], "1007,1009,1011")

    def test_improved_streak_generates_broadened_follow_up(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--epochs",
                "2",
                "--batches",
                "4",
                "--eval-samples",
                "32",
                "--vae-epochs",
                "2",
                "--vae-batches",
                "4",
            ]
        )
        summary = _summary(
            best_feature="raw_latent",
            nll=4.200,
            verdict="improved",
            config_verdict="improved",
            source_feature_verdict="improved",
            source_feature_raw_verdict="improved",
            source_feature_delta_vs_raw=-0.030,
            source_retained=True,
            gate_failed=False,
        )
        summary["follow_up_chain"]["verdict_history"] = [
            "regressed",
            "improved",
            "improved",
            "improved",
        ]
        summary["follow_up_chain"]["latest_verdict"] = "improved"
        summary["follow_up_chain"]["improved_streak"] = 3
        next_follow_up = _next_follow_up()
        next_follow_up["used_seed_history"] = [
            7,
            13,
            17,
            1013,
            1015,
            1017,
        ]
        feature_family_stability = [
            {
                "family": "hybrid_latent",
                "win_count": 3,
                "near_win_count": 3,
                "win_rate": 1.0,
                "near_win_rate": 1.0,
                "mean_best_nll": 4.198,
                "mean_best_accuracy": 0.12,
                "mean_best_nll_delta_vs_raw": -0.03,
                "mean_rank": 1.0,
                "mean_gap_to_winner": 0.0,
                "member_best_counts": {
                    "raw_latent": 2,
                    "reconstruction_latent": 1,
                },
            },
            {
                "family": "raw",
                "win_count": 0,
                "near_win_count": 0,
                "mean_best_nll_delta_vs_raw": 0.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {
                    "ancestors": [
                        _ancestor(feature="raw_latent", nll=4.216, raw_delta=-0.014),
                        _ancestor(feature="raw_latent", nll=4.204, raw_delta=-0.026),
                    ]
                },
                min_nll_delta=0.0,
            )
            broadened = mod._broadened_follow_up_command_record(
                args,
                ["raw", "latent", "raw_latent"],
                summary["best_config"],
                root,
                [1013, 1015, 1017],
                summary["follow_up_chain"],
                trajectory,
                next_follow_up,
                feature_family_stability,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
                None,
                broadened,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                broadened,
            )
            report_summary = dict(summary)
            report_summary["follow_up_trajectory"] = trajectory
            report_summary["follow_up_guidance"] = guidance
            report_summary["broadened_follow_up_command"] = broadened
            report_summary["guided_next_follow_up_command"] = guided
            report = mod._aggregate_report(report_summary)

        self.assertEqual(
            trajectory["trajectory_action"],
            "confirm_trajectory_with_fresh_seeds",
        )
        self.assertIsNotNone(broadened)
        self.assertEqual(broadened["default_new_seeds"], "101,103,107,109,113")
        self.assertEqual(broadened["broadened_epochs"], 4)
        self.assertEqual(broadened["broadened_batches"], 8)
        self.assertEqual(broadened["broadened_eval_samples"], 64)
        self.assertEqual(broadened["broadened_vae_epochs"], 4)
        self.assertEqual(broadened["broadened_vae_batches"], 8)
        self.assertEqual(
            broadened["focused_features"],
            ["raw", "latent", "raw_latent", "reconstruction_latent"],
        )
        self.assertEqual(
            broadened["feature_family_focus"]["family"],
            "hybrid_latent",
        )
        self.assertEqual(
            broadened["feature_family_focus"]["added_features"],
            ["reconstruction_latent"],
        )
        self.assertIn(
            "--features raw,latent,raw_latent,reconstruction_latent",
            broadened["shell_command"],
        )
        self.assertEqual(guidance["action"], "promote_and_broaden_after_streak")
        self.assertIs(guidance["use_next_follow_up_command"], False)
        self.assertIs(guidance["use_broadened_follow_up_command"], True)
        self.assertEqual(guidance["command_usage"], broadened["script_usage"])
        self.assertIn(
            "family focus: hybrid_latent wins=3 near_wins=3",
            guidance["reasons"],
        )
        self.assertIs(guided["enabled"], True)
        self.assertEqual(guided["default_new_seeds"], "101,103,107,109,113")
        self.assertIn("## Broadened Follow-Up Command", report)
        self.assertIn("- use_broadened_follow_up_command: True", report)
        self.assertIn("- feature_family_focus: hybrid_latent", report)
        self.assertIn(
            "- features_added_for_family: reconstruction_latent",
            report,
        )
        self.assertIn("- guidance_action: promote_and_broaden_after_streak", report)


if __name__ == "__main__":
    unittest.main()
