#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "summarize_char_vae_live_runs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_char_vae_live_runs",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SummarizeCharVaeLiveRunsTests(unittest.TestCase):
    def test_summarizes_partial_run_log_and_seed_rankings(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            _write_json(
                run / "run.json",
                {
                    "seeds": [1031, 1033],
                    "features": ["raw", "latent"],
                    "epochs": 20,
                },
            )
            (run / "run.log").write_text(
                "\n".join(
                    [
                        "sweep_normalize=blocks sweep_scale=4 sweep_seed=1031",
                        "raw[0] train_loss=4.5 val_nll=4.19 acc=17.00%",
                        (
                            "best_feature=reconstruction_latent "
                            "best_nll=3.816763 best_acc=17.58% "
                            "summary_json=/tmp/seed_001031/summary.json"
                        ),
                        "sweep_normalize=blocks sweep_scale=4 sweep_seed=1033",
                        "raw[0] train_loss=4.5 val_nll=4.19 acc=17.00%",
                        "raw[19] train_loss=4.5 val_nll=4.19 acc=17.00%",
                        "latent[10] train_loss=4.51 val_nll=4.17 acc=14.45%",
                    ]
                ),
                encoding="utf-8",
            )
            _write_json(
                run / "seed_001031" / "summary.json",
                {
                    "ranking": [
                        {
                            "feature": "reconstruction_latent",
                            "best_mean_nll": 3.816763,
                            "best_accuracy": 0.1758,
                        },
                        {
                            "feature": "raw_latent",
                            "best_mean_nll": 3.818360,
                            "best_accuracy": 0.1758,
                        },
                    ],
                    "deltas": {
                        "reconstruction_latent_best_nll_vs_raw": -0.372221,
                        "raw_latent_best_nll_vs_raw": -0.370624,
                    },
                },
            )

            summary = mod.summarize_run(run)

        self.assertFalse(summary["summary_exists"])
        self.assertEqual(summary["log"]["current_seed"], 1033)
        self.assertEqual(summary["log"]["current_feature"], "latent")
        self.assertEqual(summary["log"]["current_epoch"], 10)
        self.assertEqual(summary["log"]["completed_best_features"], 1)
        self.assertEqual(summary["completed_seed_count"], 1)
        self.assertEqual(summary["winner_counts"], {"reconstruction_latent": 1})
        evidence = summary["completed_seed_evidence"]
        self.assertEqual(evidence["completed_seed_count"], 1)
        self.assertEqual(evidence["winner_counts"], {"reconstruction_latent": 1})
        self.assertEqual(evidence["winner_rates"], {"reconstruction_latent": 1.0})
        self.assertEqual(evidence["top_winner_feature"], "reconstruction_latent")
        self.assertEqual(evidence["top_winner_count"], 1)
        self.assertAlmostEqual(evidence["top_winner_rate"], 1.0)
        self.assertAlmostEqual(evidence["mean_best_nll"], 3.816763)
        self.assertAlmostEqual(evidence["mean_best_accuracy"], 0.1758)
        self.assertAlmostEqual(evidence["mean_delta_vs_raw"], -0.372221)
        self.assertAlmostEqual(evidence["mean_margin_to_runner_up"], 0.001597)
        self.assertEqual(evidence["latest_completed_seed"], 1031)
        self.assertEqual(
            evidence["latest_completed_best_feature"],
            "reconstruction_latent",
        )
        feature_evidence = summary["completed_feature_evidence"]
        self.assertEqual(
            [item["feature"] for item in feature_evidence],
            ["reconstruction_latent", "raw_latent"],
        )
        self.assertEqual(feature_evidence[0]["seed_count"], 1)
        self.assertEqual(feature_evidence[0]["win_count"], 1)
        self.assertAlmostEqual(feature_evidence[0]["win_rate"], 1.0)
        self.assertAlmostEqual(feature_evidence[0]["mean_rank"], 1.0)
        self.assertAlmostEqual(feature_evidence[0]["mean_delta_vs_raw"], -0.372221)
        self.assertAlmostEqual(feature_evidence[0]["mean_gap_to_winner"], 0.0)
        self.assertEqual(feature_evidence[1]["seed_count"], 1)
        self.assertEqual(feature_evidence[1]["win_count"], 0)
        self.assertAlmostEqual(feature_evidence[1]["win_rate"], 0.0)
        self.assertAlmostEqual(feature_evidence[1]["mean_rank"], 2.0)
        self.assertAlmostEqual(feature_evidence[1]["mean_delta_vs_raw"], -0.370624)
        self.assertAlmostEqual(feature_evidence[1]["mean_gap_to_winner"], 0.001597)
        self.assertEqual(summary["progress"]["planned_seed_count"], 2)
        self.assertEqual(summary["progress"]["completed_seed_count"], 1)
        self.assertEqual(summary["progress"]["remaining_seeds"], [1033])
        self.assertEqual(summary["progress"]["remaining_seed_count"], 1)
        self.assertAlmostEqual(summary["progress"]["completed_seed_fraction"], 0.5)
        self.assertAlmostEqual(summary["progress"]["overall_progress_fraction"], 0.8875)
        self.assertAlmostEqual(
            summary["progress"]["remaining_overall_fraction"],
            0.1125,
        )
        self.assertEqual(summary["progress"]["active_seed_index"], 2)
        self.assertEqual(summary["progress"]["latest_completed_seed"], 1031)
        self.assertEqual(
            summary["progress"]["latest_completed_best_feature"],
            "reconstruction_latent",
        )
        self.assertEqual(
            summary["log"]["latest_progress"],
            "latent[10] train_loss=4.51 val_nll=4.17 acc=14.45%",
        )
        self.assertEqual(summary["log"]["best_so_far_feature"], "latent")
        self.assertAlmostEqual(summary["log"]["best_so_far_val_nll"], 4.17)
        self.assertAlmostEqual(summary["log"]["best_so_far_delta_vs_raw"], -0.02)
        self.assertEqual(summary["log"]["best_so_far_runner_up_feature"], "raw")
        self.assertAlmostEqual(summary["log"]["best_so_far_margin_to_runner_up"], 0.02)
        self.assertEqual(summary["log"]["planned_features"], ["raw", "latent"])
        self.assertEqual(summary["log"]["planned_feature_count"], 2)
        self.assertEqual(summary["log"]["expected_epoch_count"], 20)
        self.assertEqual(summary["log"]["expected_final_epoch"], 19)
        self.assertEqual(summary["log"]["active_feature_index"], 2)
        self.assertEqual(
            summary["log"]["active_seed_completed_features"],
            ["raw"],
        )
        self.assertEqual(summary["log"]["active_seed_completed_feature_count"], 1)
        self.assertEqual(summary["log"]["active_seed_remaining_features"], ["latent"])
        self.assertEqual(summary["log"]["active_seed_remaining_feature_count"], 1)
        self.assertEqual(summary["log"]["active_seed_next_feature"], "latent")
        self.assertAlmostEqual(summary["log"]["active_seed_completed_fraction"], 0.5)
        self.assertAlmostEqual(
            summary["log"]["active_seed_progress_fraction"],
            0.775,
        )
        active_evidence = summary["log"]["active_feature_evidence"]
        self.assertEqual(
            [item["feature"] for item in active_evidence],
            ["raw", "latent"],
        )
        self.assertEqual(active_evidence[0]["status"], "completed")
        self.assertFalse(active_evidence[0]["is_current"])
        self.assertEqual(active_evidence[0]["rank_so_far"], 2)
        self.assertAlmostEqual(active_evidence[0]["progress_fraction"], 1.0)
        self.assertAlmostEqual(active_evidence[0]["gap_to_active_best"], 0.02)
        self.assertEqual(active_evidence[1]["status"], "active")
        self.assertTrue(active_evidence[1]["is_current"])
        self.assertEqual(active_evidence[1]["rank_so_far"], 1)
        self.assertAlmostEqual(active_evidence[1]["progress_fraction"], 0.55)
        self.assertAlmostEqual(active_evidence[1]["gap_to_active_best"], 0.0)
        progress = summary["log"]["feature_progress"]
        self.assertEqual([item["feature"] for item in progress], ["raw", "latent"])
        self.assertEqual(progress[0]["latest_step"], 19)
        self.assertAlmostEqual(progress[0]["best_val_nll"], 4.19)
        self.assertAlmostEqual(progress[0]["best_delta_vs_raw"], 0.0)
        self.assertEqual(progress[1]["latest_step"], 10)
        self.assertAlmostEqual(progress[1]["best_val_nll"], 4.17)
        self.assertAlmostEqual(progress[1]["best_delta_vs_raw"], -0.02)
        self.assertEqual(summary["seed_results"][0]["seed"], 1031)
        self.assertEqual(
            summary["seed_results"][0]["best_feature"],
            "reconstruction_latent",
        )
        self.assertAlmostEqual(
            summary["seed_results"][0]["margin_to_runner_up"],
            0.001597,
        )

    def test_overall_progress_does_not_double_count_completed_current_seed(self) -> None:
        mod = _load_module()

        progress = mod._run_progress_record(
            run_metadata={"seeds": [1031, 1033, 1035]},
            seed_results=[{"seed": 1031}, {"seed": 1033}],
            log_status={
                "current_seed": 1033,
                "active_seed_progress_fraction": 1.0,
            },
        )

        self.assertAlmostEqual(progress["completed_seed_fraction"], 2 / 3)
        self.assertAlmostEqual(progress["overall_progress_fraction"], 2 / 3)
        self.assertAlmostEqual(progress["remaining_overall_fraction"], 1 / 3)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            _write_json(
                run / "run.json",
                {
                    "seeds": [1043],
                    "features": ["raw"],
                    "epochs": 1,
                },
            )
            (run / "run.log").write_text(
                "sweep_normalize=blocks sweep_scale=4 sweep_seed=1043\n"
                "raw[0] train_loss=4.5 val_nll=4.19 acc=17.00%\n",
                encoding="utf-8",
            )
            _write_json(
                run / "summary.json",
                {
                    "status": "improved",
                    "best_feature": "reconstruction_latent",
                    "best_config": {
                        "mean_best_nll": 3.8,
                        "mean_best_nll_delta_vs_raw": -0.38,
                    },
                    "follow_up_result": {"verdict": "improved"},
                    "follow_up_guidance": {"action": "confirm_trajectory_with_fresh_seeds"},
                },
            )
            json_out = root / "summary.json.out"
            md_out = root / "summary.md"

            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(run),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(md_out),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            markdown = md_out.read_text(encoding="utf-8")

        self.assertIn("# Char VAE Live Runs", result.stdout)
        self.assertEqual(
            payload["schema"],
            "st.llm_char_vae_context.live_run_summaries.v1",
        )
        self.assertEqual(payload["totals"]["run_count"], 1)
        self.assertEqual(payload["totals"]["completed_run_count"], 1)
        self.assertEqual(payload["runs"][0]["status"], "improved")
        self.assertEqual(payload["runs"][0]["progress"]["remaining_seeds"], [1043])
        self.assertEqual(payload["runs"][0]["progress"]["remaining_seed_count"], 1)
        self.assertAlmostEqual(
            payload["runs"][0]["progress"]["overall_progress_fraction"],
            1.0,
        )
        self.assertAlmostEqual(
            payload["runs"][0]["progress"]["remaining_overall_fraction"],
            0.0,
        )
        self.assertEqual(payload["runs"][0]["log"]["current_feature"], "raw")
        self.assertEqual(payload["runs"][0]["log"]["best_so_far_feature"], "raw")
        self.assertEqual(payload["runs"][0]["log"]["active_seed_remaining_features"], [])
        self.assertEqual(payload["runs"][0]["log"]["active_seed_remaining_feature_count"], 0)
        self.assertIsNone(payload["runs"][0]["log"]["active_seed_next_feature"])
        self.assertAlmostEqual(
            payload["runs"][0]["log"]["active_seed_completed_fraction"],
            1.0,
        )
        self.assertIn("## Overview", markdown)
        self.assertIn("- completed_run_count: 1", markdown)
        self.assertIn("- seed_progress: 0/1", markdown)
        self.assertIn("- overall_progress_fraction: 1.000000", markdown)
        self.assertIn("- remaining_overall_fraction: 0.000000", markdown)
        self.assertIn("- remaining_seeds: 1043", markdown)
        self.assertIn("- remaining_seed_count: 1", markdown)
        self.assertIn("- completed_seed_leader: - (0/0, rate=-)", markdown)
        self.assertIn("- completed_seed_mean_delta_vs_raw: -", markdown)
        self.assertIn("- completed_seed_mean_margin_to_runner_up: -", markdown)
        self.assertIn(
            "| completed_feature | seeds | wins | win_rate | mean_rank | mean_nll | mean_delta_vs_raw | mean_gap_to_winner |",
            markdown,
        )
        self.assertIn(
            "| active_feature | status | current | rank_so_far | progress | best_nll | delta_vs_raw | gap_to_active_best |",
            markdown,
        )
        self.assertIn(
            "| raw | completed | yes | 1 | 1.000000 | 4.190000 | 0.000000 | 0.000000 |",
            markdown,
        )
        self.assertIn("- active_feature_index: 1/1", markdown)
        self.assertIn("- active_seed_progress_fraction: 1.000000", markdown)
        self.assertIn("- active_seed_remaining_features: -", markdown)
        self.assertIn("- active_seed_remaining_feature_count: 0", markdown)
        self.assertIn("- active_seed_next_feature: -", markdown)
        self.assertIn("- active_seed_completed_fraction: 1.000000", markdown)
        self.assertIn("- best_so_far: raw@4.190000", markdown)
        self.assertIn("- best_so_far_margin_to_runner_up: -", markdown)
        self.assertIn("| raw | 0 | 4.190000 | 0 | 4.190000 | 17.00 | 0.000000 |", markdown)
        self.assertIn("follow_up_verdict: improved", markdown)


if __name__ == "__main__":
    unittest.main()
