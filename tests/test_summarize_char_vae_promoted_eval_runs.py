import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "summarize_char_vae_promoted_eval_runs.py"


def _write_eval_summary(
    path: Path,
    *,
    seed: int,
    reconstruction_nll: float,
    raw_nll: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "ranking": [
                    {
                        "feature": "reconstruction_latent",
                        "best_mean_nll": reconstruction_nll,
                        "best_accuracy": 0.2,
                    },
                    {
                        "feature": "raw",
                        "best_mean_nll": raw_nll,
                        "best_accuracy": 0.1,
                    },
                ],
                "deltas": {
                    "reconstruction_latent_best_nll_vs_raw": (
                        reconstruction_nll - raw_nll
                    ),
                    "raw_best_nll_vs_raw": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    *,
    cwd: Path,
    seeds: list[int],
    available_count: int | None = None,
    planned_seeds: list[int] | None = None,
    mainline_next_command: str | None = None,
    mainline_next_command_key: str = "mainline_scale_up_command",
) -> None:
    if planned_seeds is not None:
        mainline_run_dir = cwd / "mainline_scale_up"
        (cwd / "summary.json").write_text(
            json.dumps(
                {
                    "mainline_scale_up_command": {
                        "default_new_seeds": ",".join(
                            str(seed) for seed in planned_seeds
                        ),
                        "default_run_dir": str(mainline_run_dir),
                    }
                }
            ),
            encoding="utf-8",
        )
        if mainline_next_command is not None:
            mainline_run_dir.mkdir(parents=True, exist_ok=True)
            (mainline_run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        mainline_next_command_key: {
                            "shell_command": mainline_next_command,
                            "default_new_seeds": "211,223",
                            "default_run_dir": str(cwd / "mainline_next"),
                        }
                    }
                ),
                encoding="utf-8",
            )
    path.write_text(
        json.dumps(
            {
                "schema": "st.llm_char_vae_context.promoted_recipe_eval_run.v1",
                "summary_path": str(cwd / "summary.json"),
                "feature": "reconstruction_latent",
                "feature_family": "hybrid_latent",
                "execute": True,
                "ready_only": True,
                "complete_only": True,
                "selected_count": len(seeds),
                "available_count": available_count
                if available_count is not None
                else len(seeds),
                "cwd": str(cwd),
                "returncode": 0,
                "results": [
                    {
                        "seed": seed,
                        "run_dir": f"seed_{seed:06d}/eval_best",
                        "returncode": 0,
                        "executed": True,
                    }
                    for seed in seeds
                ],
            }
        ),
        encoding="utf-8",
    )


class PromotedEvalSummaryTests(unittest.TestCase):
    def test_json_summary_reports_reload_winners_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "promoted_recipe_eval_run.json"
            _write_report(
                report,
                cwd=root,
                seeds=[101, 103],
                planned_seeds=[101, 103],
                mainline_next_command=(
                    "PYTHONNOUSERSITE=1 python3 -S -s "
                    "models/python/llm_char_vae_context.py corpus --seeds 211,223"
                ),
            )
            _write_eval_summary(
                root / "seed_000101" / "eval_best" / "summary.json",
                seed=101,
                reconstruction_nll=3.8,
                raw_nll=4.0,
            )
            _write_eval_summary(
                root / "seed_000103" / "eval_best" / "summary.json",
                seed=103,
                reconstruction_nll=3.7,
                raw_nll=4.1,
            )

            result = subprocess.run(
                ["python3", str(SCRIPT), str(report), "--json"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            summary = payload["reports"][0]
            self.assertEqual(summary["successful_eval_count"], 2)
            self.assertTrue(summary["complete_only"])
            self.assertTrue(summary["planned_eval_complete"])
            self.assertEqual(summary["remaining_eval_count"], 0)
            self.assertEqual(summary["planned_eval_seeds"], [101, 103])
            self.assertEqual(summary["evaluated_eval_seeds"], [101, 103])
            self.assertEqual(summary["remaining_eval_seeds"], [])
            self.assertTrue(summary["promoted_mainline_summary_exists"])
            self.assertTrue(
                summary["promoted_mainline_summary_path"].endswith(
                    "mainline_scale_up/summary.json"
                )
            )
            self.assertEqual(summary["winner_counts"], {"reconstruction_latent": 2})
            self.assertEqual(summary["target_feature_win_rate"], 1.0)
            self.assertAlmostEqual(summary["mean_target_delta_vs_raw"], -0.3)
            self.assertEqual(summary["recommendation"], "promote_reload_evidence")
            self.assertIsNone(summary["recommended_next_eval_command"])
            self.assertEqual(
                summary["recommended_next_mainline_command"],
                (
                    "PYTHONNOUSERSITE=1 python3 -S -s "
                    "models/python/llm_char_vae_context.py corpus --seeds 211,223"
                ),
            )
            self.assertEqual(
                summary["recommended_next_mainline_command_source"],
                "mainline_scale_up_command",
            )
            self.assertIn(
                "tools/summarize_char_vae_promoted_eval_runs.py",
                summary["recommended_summary_command"],
            )

    def test_json_summary_uses_completed_mainline_next_follow_up_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "promoted_recipe_eval_run.json"
            _write_report(
                report,
                cwd=root,
                seeds=[101],
                planned_seeds=[101],
                mainline_next_command=(
                    "PYTHONNOUSERSITE=1 python3 -S -s "
                    "models/python/llm_char_vae_context.py corpus --seeds 301,303"
                ),
                mainline_next_command_key="next_follow_up_command",
            )
            _write_eval_summary(
                root / "seed_000101" / "eval_best" / "summary.json",
                seed=101,
                reconstruction_nll=3.8,
                raw_nll=4.0,
            )

            result = subprocess.run(
                ["python3", str(SCRIPT), str(report), "--json"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            summary = payload["reports"][0]
            self.assertEqual(summary["recommendation"], "promote_reload_evidence")
            self.assertEqual(
                summary["recommended_next_mainline_command"],
                (
                    "PYTHONNOUSERSITE=1 python3 -S -s "
                    "models/python/llm_char_vae_context.py corpus --seeds 301,303"
                ),
            )
            self.assertEqual(
                summary["recommended_next_mainline_command_source"],
                "next_follow_up_command",
            )

    def test_json_summary_continues_when_planned_eval_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "promoted_recipe_eval_run.json"
            _write_report(
                report,
                cwd=root,
                seeds=[101, 103],
                available_count=3,
                planned_seeds=[101, 103, 107],
            )
            for seed, reconstruction_nll in ((101, 3.8), (103, 3.7)):
                _write_eval_summary(
                    root / f"seed_{seed:06d}" / "eval_best" / "summary.json",
                    seed=seed,
                    reconstruction_nll=reconstruction_nll,
                    raw_nll=4.0,
                )

            result = subprocess.run(
                ["python3", str(SCRIPT), str(report), "--json"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            summary = payload["reports"][0]
            self.assertFalse(summary["planned_eval_complete"])
            self.assertEqual(summary["remaining_eval_count"], 1)
            self.assertEqual(summary["planned_eval_seeds"], [101, 103, 107])
            self.assertEqual(summary["evaluated_eval_seeds"], [101, 103])
            self.assertEqual(summary["remaining_eval_seeds"], [107])
            self.assertEqual(summary["winner_counts"], {"reconstruction_latent": 2})
            self.assertEqual(summary["recommendation"], "continue_planned_eval")
            self.assertIsNone(summary["recommended_next_mainline_command"])
            self.assertIsNone(summary["recommended_next_mainline_command_source"])
            command = summary["recommended_next_eval_command"]
            self.assertIsInstance(command, str)
            self.assertIn("tools/run_char_vae_promoted_recipe.py", command)
            self.assertIn("--seed 107", command)
            self.assertIn("--ready-only", command)
            self.assertIn("--complete-only", command)
            self.assertIn("--execute", command)
            self.assertIn("--write-report", command)

    def test_markdown_surfaces_missing_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "promoted_recipe_eval_run.json"
            _write_report(report, cwd=root, seeds=[101])

            result = subprocess.run(
                ["python3", str(SCRIPT), str(root)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("- missing_summary_count: 1", result.stdout)
            self.assertIn("- recommendation: complete_eval_summaries", result.stdout)


if __name__ == "__main__":
    unittest.main()
