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
        self.assertEqual(summary["log"]["completed_best_features"], 1)
        self.assertEqual(
            summary["log"]["latest_progress"],
            "latent[10] train_loss=4.51 val_nll=4.17 acc=14.45%",
        )
        self.assertEqual(summary["seed_results"][0]["seed"], 1031)
        self.assertEqual(
            summary["seed_results"][0]["best_feature"],
            "reconstruction_latent",
        )
        self.assertAlmostEqual(
            summary["seed_results"][0]["margin_to_runner_up"],
            0.001597,
        )

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
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
        self.assertEqual(payload["runs"][0]["status"], "improved")
        self.assertIn("follow_up_verdict: improved", markdown)


if __name__ == "__main__":
    unittest.main()
