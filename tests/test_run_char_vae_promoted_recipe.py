import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_char_vae_promoted_recipe.py"


def _write_summary(path: Path) -> None:
    commands = []
    for seed in (101, 103):
        seed_dir = path.parent / f"seed_{seed:06d}"
        seed_dir.mkdir()
        (seed_dir / "text_vae_weights.bin").write_bytes(b"weights")
        (seed_dir / "head_reconstruction_latent_best.json").write_text(
            "{}",
            encoding="utf-8",
        )
        commands.append(
            {
                "schema": "st.llm_char_vae_context.promoted_learning_eval_command.v1",
                "seed": seed,
                "run_dir": str(seed_dir / "eval_best"),
                "source_run_dir": str(seed_dir),
                "vae_load": str(seed_dir / "text_vae_weights.bin"),
                "head_load_dir": str(seed_dir),
                "head_load_kind": "best",
                "script_command": [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path('marker_{seed}.txt').write_text('ok')"
                    ),
                ],
            }
        )
    path.write_text(
        json.dumps(
            {
                "mainline_scale_up_command": {
                    "promoted_learning_recipe": {
                        "schema": "st.llm_char_vae_context.promoted_learning_recipe.v1",
                        "feature": "reconstruction_latent",
                        "feature_family": "hybrid_latent",
                        "eval_reload_commands": {
                            "schema": (
                                "st.llm_char_vae_context."
                                "promoted_learning_eval_commands.v1"
                            ),
                            "count": len(commands),
                            "items": commands,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


class PromotedRecipeRunnerTests(unittest.TestCase):
    def test_dry_run_reports_selected_eval_command_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            _write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(summary),
                    "--seed",
                    "103",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertFalse(payload["execute"])
            self.assertEqual(payload["selected_count"], 1)
            self.assertEqual(payload["results"][0]["seed"], 103)
            self.assertFalse((ROOT / "marker_103.txt").exists())

    def test_execute_runs_selected_eval_command_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            _write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(summary),
                    "--seed",
                    "101",
                    "--execute",
                    "--cwd",
                    str(root),
                    "--write-report",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["execute"])
            self.assertEqual(payload["selected_count"], 1)
            self.assertEqual(payload["results"][0]["returncode"], 0)
            self.assertTrue((root / "marker_101.txt").exists())
            report_path = root / "promoted_recipe_eval_run.json"
            self.assertTrue(report_path.exists())
            self.assertTrue((root / "promoted_recipe_eval_run.md").exists())
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["report_path"], str(report_path))
            self.assertEqual(report_payload["markdown_path"], payload["markdown_path"])


if __name__ == "__main__":
    unittest.main()
