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
) -> None:
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
            _write_report(report, cwd=root, seeds=[101, 103])
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
            self.assertEqual(summary["winner_counts"], {"reconstruction_latent": 2})
            self.assertEqual(summary["target_feature_win_rate"], 1.0)
            self.assertAlmostEqual(summary["mean_target_delta_vs_raw"], -0.3)
            self.assertEqual(summary["recommendation"], "promote_reload_evidence")

    def test_json_summary_continues_when_planned_eval_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "promoted_recipe_eval_run.json"
            _write_report(report, cwd=root, seeds=[101, 103], available_count=3)
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
            self.assertEqual(summary["winner_counts"], {"reconstruction_latent": 2})
            self.assertEqual(summary["recommendation"], "continue_planned_eval")

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
