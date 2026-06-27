#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_char_vae_context_chain.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_char_vae_context_chain", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharVaeContextChainTests(unittest.TestCase):
    def test_step_record_and_report_surface_follow_up_deltas(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "follow_up_01"
            run_dir.mkdir()
            summary = {
                "status": "improved",
                "best_feature": "latent",
                "best_config": {
                    "best_feature": "latent",
                    "feature_normalize": "blocks",
                    "hybrid_latent_scale": 0.5,
                    "mean_best_nll": 4.2258,
                    "mean_best_accuracy": 0.104,
                    "mean_best_nll_delta_vs_raw": -0.0023,
                },
                "follow_up_result": {
                    "verdict": "regressed",
                    "mean_best_nll_delta_vs_source": 0.001,
                    "source_feature_mean_best_nll_delta_vs_source": 0.001,
                    "source_best_feature_retained": True,
                },
                "follow_up_gate": {"failed": True},
                "follow_up_trajectory": {
                    "trajectory_action": "stop_on_follow_up_gate",
                    "trajectory_verdict": "regressed",
                },
                "follow_up_guidance": {
                    "action": "stop_on_follow_up_gate",
                    "unsafe_promotion": True,
                },
                "guided_next_follow_up_command": {"enabled": False},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            step = mod._step_record(
                index=1,
                role="follow_up",
                run_dir=run_dir,
                command=["bash", "next_follow_up_command.sh"],
                exit_code=1,
                dry_run=False,
            )
            report = mod._render_report(
                {
                    "schema": mod.SCHEMA,
                    "preset": "base",
                    "run_root": str(run_dir.parent),
                    "steps": [step],
                    "allowed_gate_stop": True,
                }
            )

        self.assertEqual(step["mean_best_nll_delta_vs_raw"], -0.0023)
        self.assertEqual(step["mean_best_nll_delta_vs_source"], 0.001)
        self.assertEqual(step["source_feature_mean_best_nll_delta_vs_source"], 0.001)
        self.assertIs(step["source_best_feature_retained"], True)
        self.assertIs(step["follow_up_gate_failed"], True)
        self.assertEqual(step["best_config_label"], "latent@normalize=blocks,scale=0.5")
        self.assertIn("delta_vs_raw", report)
        self.assertIn("source_feature_delta_vs_source", report)
        self.assertIn("latent@normalize=blocks,scale=0.5", report)
        self.assertIn("0.001000", report)


if __name__ == "__main__":
    unittest.main()
