#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "summarize_char_vae_context_chains.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_char_vae_context_chains",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_chain(path: Path, payload: dict[str, object]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    chain_path = path / "chain.json"
    chain_path.write_text(json.dumps(payload), encoding="utf-8")
    return chain_path


class SummarizeCharVaeContextChainsTests(unittest.TestCase):
    def test_summarize_chains_aggregates_seed_resolution(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_chain(
                root / "chain_a",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "smoke",
                    "run_root": str(root / "chain_a"),
                    "planned_follow_ups": 3,
                    "attempted_follow_ups": 1,
                    "stopped_reason": "follow-up 1 exited 1",
                    "accepted_step": {
                        "index": 0,
                        "best_config_label": "raw@normalize=blocks,scale=1.0",
                        "mean_best_nll": 4.4,
                    },
                    "best_step": {
                        "index": 0,
                        "best_config_label": "raw@normalize=blocks,scale=1.0",
                        "mean_best_nll": 4.4,
                        "mean_best_nll_delta_vs_raw": 0.0,
                        "runner_up_feature": "latent",
                        "margin_to_runner_up": 0.01,
                        "runner_up_within_uncertainty": False,
                    },
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 1,
                        "seed_source_counts": {"explicit_seed_group": 1},
                        "command_source_counts": {"next_follow_up_command": 1},
                        "configured_seed_group_status_counts": {"attempted_slot": 1},
                        "gate_failed_count": 1,
                        "nonzero_exit_count": 1,
                    },
                    "extra_explicit_seed_groups": ["29"],
                    "unused_explicit_seed_groups": ["19", "23"],
                },
            )
            second = _write_chain(
                root / "chain_b",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "smoke",
                    "run_root": str(root / "chain_b"),
                    "planned_follow_ups": 1,
                    "attempted_follow_ups": 1,
                    "allowed_gate_stop": True,
                    "accepted_step": {
                        "index": 1,
                        "best_config_label": "latent@normalize=blocks,scale=0.5",
                        "mean_best_nll": 4.1,
                    },
                    "best_step": {
                        "index": 1,
                        "best_config_label": "latent@normalize=blocks,scale=0.5",
                        "mean_best_nll": 4.1,
                        "mean_best_nll_delta_vs_raw": -0.2,
                        "runner_up_feature": "raw",
                        "margin_to_runner_up": 0.02,
                        "runner_up_within_uncertainty": True,
                    },
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 1,
                        "seed_source_counts": {"command_default": 1},
                        "command_source_counts": {"guided_next_follow_up_command": 1},
                        "configured_seed_group_status_counts": {"attempted_slot": 1},
                        "gate_failed_count": 0,
                        "nonzero_exit_count": 0,
                    },
                    "extra_explicit_seed_groups": [],
                    "unused_explicit_seed_groups": [],
                },
            )

            summary = mod.summarize_chains([first, second], sort_by="best")
            markdown = mod._render_markdown(summary)

        self.assertEqual(summary["schema"], mod.SCHEMA)
        self.assertEqual(summary["aggregate"]["chain_count"], 2)
        self.assertEqual(summary["aggregate"]["attempted_follow_ups"], 2)
        self.assertEqual(summary["aggregate"]["gate_failed_count"], 1)
        self.assertEqual(summary["aggregate"]["nonzero_exit_count"], 1)
        self.assertEqual(
            summary["aggregate"]["seed_source_counts"],
            {"command_default": 1, "explicit_seed_group": 1},
        )
        self.assertEqual(
            summary["aggregate"]["command_source_counts"],
            {"guided_next_follow_up_command": 1, "next_follow_up_command": 1},
        )
        self.assertEqual(summary["chains"][0]["best_mean_best_nll"], 4.1)
        self.assertIn("# Char VAE Context Chain Comparison", markdown)
        self.assertIn("- seed_source_counts: command_default:1, explicit_seed_group:1", markdown)
        self.assertIn("latent@normalize=blocks,scale=0.5", markdown)
        self.assertIn("29 | 19,23", markdown)

    def test_cli_writes_json_and_markdown_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = _write_chain(
                root / "chain",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "smoke",
                    "run_root": str(root / "chain"),
                    "planned_follow_ups": 1,
                    "attempted_follow_ups": 0,
                    "dry_run": True,
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 0,
                        "seed_source_counts": {},
                        "command_source_counts": {},
                        "configured_seed_group_status_counts": {},
                        "gate_failed_count": 0,
                        "nonzero_exit_count": 0,
                    },
                    "extra_explicit_seed_groups": ["19"],
                    "unused_explicit_seed_groups": [],
                },
            )
            json_out = root / "comparison.json"
            markdown_out = root / "comparison.md"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(chain.parent),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            markdown = markdown_out.read_text(encoding="utf-8")

        self.assertEqual(payload["aggregate"]["dry_run_count"], 1)
        self.assertIn("Char VAE Context Chain Comparison", result.stdout)
        self.assertIn("dry_run_count: 1", markdown)

    def test_recursive_discovery_finds_nested_chains_once(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_chain(
                root / "runs" / "chain_a",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "smoke",
                    "run_root": str(root / "runs" / "chain_a"),
                    "planned_follow_ups": 1,
                    "attempted_follow_ups": 0,
                    "dry_run": True,
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 0,
                        "seed_source_counts": {},
                        "command_source_counts": {},
                        "configured_seed_group_status_counts": {},
                        "gate_failed_count": 0,
                        "nonzero_exit_count": 0,
                    },
                },
            )
            _write_chain(
                root / "runs" / "chain_b",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "smoke",
                    "run_root": str(root / "runs" / "chain_b"),
                    "planned_follow_ups": 1,
                    "attempted_follow_ups": 1,
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 1,
                        "seed_source_counts": {"command_default": 1},
                        "command_source_counts": {"next_follow_up_command": 1},
                        "configured_seed_group_status_counts": {"attempted_slot": 1},
                        "gate_failed_count": 0,
                        "nonzero_exit_count": 0,
                    },
                },
            )

            summary = mod.summarize_chains(
                [root / "runs", first],
                recursive=True,
                sort_by="attempted",
            )

        self.assertEqual(summary["recursive"], True)
        self.assertEqual(summary["input_count"], 2)
        self.assertEqual(summary["discovered_chain_count"], 2)
        self.assertEqual(summary["aggregate"]["chain_count"], 2)
        self.assertEqual(summary["aggregate"]["attempted_follow_ups"], 1)
        self.assertEqual(
            summary["aggregate"]["seed_source_counts"],
            {"command_default": 1},
        )


if __name__ == "__main__":
    unittest.main()
