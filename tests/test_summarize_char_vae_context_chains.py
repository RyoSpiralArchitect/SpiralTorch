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
        self.assertEqual(
            summary["selection"]["accepted_champion"]["config"],
            "latent@normalize=blocks,scale=0.5",
        )
        self.assertEqual(
            summary["selection"]["best_champion"]["config"],
            "latent@normalize=blocks,scale=0.5",
        )
        self.assertIs(summary["selection"]["accepted_matches_best"], True)
        self.assertEqual(
            summary["recommendation"]["action"],
            "continue_from_accepted",
        )
        self.assertEqual(
            summary["recommendation"]["champion_source"],
            "accepted_champion",
        )
        self.assertEqual(summary["chains"][0]["best_mean_best_nll"], 4.1)
        self.assertIn("# Char VAE Context Chain Comparison", markdown)
        self.assertIn("## Selection", markdown)
        self.assertIn("accepted_champion", markdown)
        self.assertIn("recommendation: continue_from_accepted", markdown)
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

    def test_selection_separates_safe_accepted_from_absolute_best(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_parent_summary = root / "reviewed" / "parent" / "summary.json"
            reviewed_follow_summary = (
                root / "reviewed" / "follow_up_01" / "summary.json"
            )
            safe_follow_summary = root / "safe" / "follow_up_01" / "summary.json"
            reviewed = _write_chain(
                root / "reviewed",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "hybrid4",
                    "run_root": str(root / "reviewed"),
                    "planned_follow_ups": 2,
                    "attempted_follow_ups": 1,
                    "stopped_reason": "follow-up 1 exited 1",
                    "accepted_summary_path": str(reviewed_parent_summary),
                    "best_summary_path": str(reviewed_follow_summary),
                    "accepted_step": {
                        "index": 0,
                        "role": "parent",
                        "run_dir": str(root / "reviewed" / "parent"),
                        "summary_path": str(reviewed_parent_summary),
                        "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
                        "mean_best_nll": 4.22,
                        "mean_best_nll_delta_vs_raw": -0.02,
                    },
                    "best_step": {
                        "index": 1,
                        "role": "follow_up",
                        "run_dir": str(root / "reviewed" / "follow_up_01"),
                        "summary_path": str(reviewed_follow_summary),
                        "best_config_label": "latent@normalize=blocks,scale=4.0",
                        "mean_best_nll": 4.10,
                        "mean_best_nll_delta_vs_raw": -0.04,
                        "runner_up_feature": "raw_latent",
                        "margin_to_runner_up": 0.002,
                        "runner_up_within_uncertainty": True,
                    },
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 1,
                        "seed_source_counts": {"command_default": 1},
                        "command_source_counts": {"guided_next_follow_up_command": 1},
                        "configured_seed_group_status_counts": {"attempted_slot": 1},
                        "gate_failed_count": 1,
                        "nonzero_exit_count": 1,
                    },
                },
            )
            safe = _write_chain(
                root / "safe",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "hybrid4",
                    "run_root": str(root / "safe"),
                    "planned_follow_ups": 1,
                    "attempted_follow_ups": 1,
                    "accepted_summary_path": str(safe_follow_summary),
                    "best_summary_path": str(safe_follow_summary),
                    "accepted_step": {
                        "index": 1,
                        "role": "follow_up",
                        "run_dir": str(root / "safe" / "follow_up_01"),
                        "summary_path": str(safe_follow_summary),
                        "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
                        "mean_best_nll": 4.15,
                        "mean_best_nll_delta_vs_raw": -0.03,
                    },
                    "best_step": {
                        "index": 1,
                        "role": "follow_up",
                        "run_dir": str(root / "safe" / "follow_up_01"),
                        "summary_path": str(safe_follow_summary),
                        "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
                        "mean_best_nll": 4.15,
                        "mean_best_nll_delta_vs_raw": -0.03,
                    },
                    "follow_up_seed_resolution_summary": {
                        "attempted_follow_ups": 1,
                        "seed_source_counts": {"command_default": 1},
                        "command_source_counts": {"guided_next_follow_up_command": 1},
                        "configured_seed_group_status_counts": {"attempted_slot": 1},
                        "gate_failed_count": 0,
                        "nonzero_exit_count": 0,
                    },
                },
            )

            summary = mod.summarize_chains([reviewed, safe], sort_by="best")
            markdown = mod._render_markdown(summary)

        selection = summary["selection"]
        self.assertEqual(selection["accepted_candidate_count"], 2)
        self.assertEqual(selection["best_candidate_count"], 2)
        self.assertEqual(
            selection["accepted_champion"]["source"],
            str(safe),
        )
        self.assertEqual(
            selection["accepted_champion"]["summary_path"],
            str(safe_follow_summary),
        )
        self.assertEqual(selection["best_champion"]["source"], str(reviewed))
        self.assertEqual(
            selection["best_champion"]["summary_path"],
            str(reviewed_follow_summary),
        )
        self.assertIs(selection["accepted_matches_best"], False)
        self.assertIs(selection["best_requires_review"], True)
        self.assertAlmostEqual(selection["accepted_vs_best_nll_gap"], 0.05)
        self.assertEqual(summary["recommendation"]["action"], "review_absolute_best")
        self.assertEqual(
            summary["recommendation"]["follow_up_from_summary_path"],
            str(safe_follow_summary),
        )
        self.assertEqual(
            summary["recommendation"]["review_summary_path"],
            str(reviewed_follow_summary),
        )
        self.assertEqual(
            summary["recommendation"]["fallback"]["summary_path"],
            str(safe_follow_summary),
        )
        self.assertIn("best_requires_review: yes", markdown)
        self.assertIn("accepted_vs_best_nll_gap: 0.050000", markdown)
        self.assertIn("recommendation: review_absolute_best", markdown)
        self.assertIn(f"follow_up_from={safe_follow_summary}", markdown)
        self.assertIn(f"review={reviewed_follow_summary}", markdown)

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
