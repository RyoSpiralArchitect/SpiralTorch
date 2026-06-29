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
        if seed == 101:
            (seed_dir / "summary.json").write_text(
                json.dumps({"seed": seed}),
                encoding="utf-8",
            )
            for feature in ("raw", "reconstruction_latent"):
                (seed_dir / f"head_{feature}_best.json").write_text(
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
                        f"Path('marker_{seed}.txt').write_text('ok'); "
                        f"print('child-out-{seed}')"
                    ),
                    "--features",
                    "raw,reconstruction_latent",
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


def _write_legacy_mainline_summary(path: Path) -> None:
    run_dir = path.parent / "mainline_scale_up"
    path.write_text(
        json.dumps(
            {
                "mainline_scale_up_command": {
                    "schema": "st.llm_char_vae_context.mainline_scale_up_command.v1",
                    "best_config": {
                        "best_feature": "reconstruction_latent",
                        "feature_normalize": "blocks",
                        "hybrid_latent_scale": 4.0,
                        "latent_dim": 12,
                        "hidden": 64,
                    },
                    "feature_family_focus": {
                        "family": "hybrid_latent",
                        "best_feature": "reconstruction_latent",
                    },
                    "focused_features": [
                        "raw",
                        "raw_latent",
                        "reconstruction_latent",
                    ],
                    "default_new_seeds": "101,103",
                    "default_run_dir": str(run_dir),
                    "train_window_chars": 64,
                    "train_epochs": 128,
                    "train_batches": 256,
                    "train_eval_samples": 512,
                    "train_vae_epochs": 24,
                    "train_vae_batches": 48,
                    "train_gen": 120,
                    "script_command": [
                        "python3",
                        "-S",
                        "-s",
                        "models/python/llm_char_vae_context.py",
                        "models/samples/spiral_corpus_en",
                        "--features",
                        "raw,raw_latent,reconstruction_latent",
                        "--seeds",
                        "${NEW_SEEDS}",
                        "--run-dir",
                        "${NEXT_RUN_DIR}",
                        "--follow-up-from",
                        "${FOLLOW_UP_FROM}",
                        "--epochs",
                        "128",
                        "--batches",
                        "256",
                        "--vae-epochs",
                        "24",
                        "--vae-batches",
                        "48",
                    ],
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
            self.assertFalse(payload["results"][0]["required_heads_all_exist"])
            self.assertEqual(len(payload["results"][0]["missing_head_paths"]), 2)
            self.assertFalse(payload["results"][0]["source_summary_exists"])
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
            self.assertTrue(payload["results"][0]["required_heads_all_exist"])
            self.assertTrue(payload["results"][0]["source_summary_exists"])
            self.assertIn("child-out-101", payload["results"][0]["stdout"])
            self.assertTrue((root / "marker_101.txt").exists())
            report_path = root / "promoted_recipe_eval_run.json"
            self.assertTrue(report_path.exists())
            self.assertTrue((root / "promoted_recipe_eval_run.md").exists())
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["report_path"], str(report_path))
            self.assertEqual(report_payload["markdown_path"], payload["markdown_path"])

    def test_ready_only_filters_to_commands_with_all_required_heads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            _write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(summary),
                    "--ready-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ready_only"])
            self.assertEqual(payload["selected_count"], 1)
            self.assertEqual(payload["available_count"], 2)
            self.assertEqual(payload["results"][0]["seed"], 101)
            self.assertTrue(payload["results"][0]["required_heads_all_exist"])

    def test_complete_only_filters_to_commands_with_source_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            _write_summary(summary)

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(summary),
                    "--ready-only",
                    "--complete-only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["ready_only"])
            self.assertTrue(payload["complete_only"])
            self.assertEqual(payload["selected_count"], 1)
            self.assertEqual(payload["available_count"], 2)
            self.assertEqual(payload["results"][0]["seed"], 101)
            self.assertTrue(payload["results"][0]["required_heads_all_exist"])
            self.assertTrue(payload["results"][0]["source_summary_exists"])

    def test_dry_run_synthesizes_eval_commands_from_legacy_mainline_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            _write_legacy_mainline_summary(summary)

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
            self.assertEqual(payload["feature"], "reconstruction_latent")
            self.assertEqual(payload["feature_family"], "hybrid_latent")
            self.assertEqual(payload["available_count"], 2)
            self.assertEqual(payload["selected_count"], 1)
            item = payload["results"][0]
            self.assertEqual(item["seed"], 103)
            command = item["command"]
            self.assertIn("--eval-only", command)
            self.assertNotIn("--seeds", command)
            self.assertEqual(command[command.index("--seed") + 1], "103")
            self.assertEqual(command[command.index("--epochs") + 1], "0")
            self.assertEqual(command[command.index("--vae-epochs") + 1], "0")
            self.assertTrue(
                command[command.index("--run-dir") + 1].endswith(
                    "mainline_scale_up/seed_000103/eval_best"
                )
            )
            self.assertTrue(
                command[command.index("--vae-load") + 1].endswith(
                    "mainline_scale_up/seed_000103/text_vae_weights.bin"
                )
            )


if __name__ == "__main__":
    unittest.main()
