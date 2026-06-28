#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import shlex
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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

    def test_cli_writes_recommended_command_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative_script = root / "accepted scripts" / "next command.sh"
            default_run_dir = root / "chain" / "accepted" / "next run"
            relative_script.parent.mkdir(parents=True, exist_ok=True)
            relative_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        (
                            'printf "cwd=%s follow=%s seeds=%s next=%s fail=%s\\n" '
                            '"$(pwd)" "$FOLLOW_UP_FROM" "$NEW_SEEDS" '
                            '"$NEXT_RUN_DIR" "$FOLLOW_UP_FAIL_ON_VERDICT" '
                            "> wrapper.out"
                        ),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            summary_path = root / "chain" / "accepted" / "summary.json"
            _write_json(
                summary_path,
                {
                    "next_follow_up_command": {
                        "script_path": str(relative_script),
                        "script_usage": (
                            "FOLLOW_UP_FROM=accepted NEW_SEEDS=31 "
                            "NEXT_RUN_DIR=chain/accepted/next run "
                            "FOLLOW_UP_FAIL_ON_VERDICT=regressed,unknown "
                            "bash accepted scripts/next command.sh"
                        ),
                        "shell_command": "PYTHONNOUSERSITE=1 python accepted.py",
                        "default_new_seeds": "31",
                        "default_run_dir": str(default_run_dir),
                        "default_follow_up_from": "accepted",
                        "default_follow_up_fail_on_verdict": "regressed,unknown",
                    },
                },
            )
            chain = _write_chain(
                root / "chain",
                {
                    "schema": "st.llm_char_vae_context.chain.v1",
                    "preset": "smoke",
                    "run_root": str(root / "chain"),
                    "planned_follow_ups": 1,
                    "attempted_follow_ups": 1,
                    "accepted_summary_path": str(summary_path),
                    "best_summary_path": str(summary_path),
                    "accepted_step": {
                        "index": 1,
                        "summary_path": str(summary_path),
                        "best_config_label": "latent@normalize=blocks,scale=0.5",
                        "mean_best_nll": 4.1,
                    },
                    "best_step": {
                        "index": 1,
                        "summary_path": str(summary_path),
                        "best_config_label": "latent@normalize=blocks,scale=0.5",
                        "mean_best_nll": 4.1,
                    },
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
            json_out = root / "comparison.json"
            markdown_out = root / "comparison.md"
            command_dir = root / "commands"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(chain),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                    "--command-out-dir",
                    str(command_dir),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            markdown = markdown_out.read_text(encoding="utf-8")
            follow_up_script = command_dir / "recommended_follow_up.sh"
            review_script = command_dir / "recommended_review.sh"
            manifest_path = command_dir / "recommendation.json"
            readme_path = command_dir / "README.md"
            execution_cwd = str(root.resolve())
            self.assertTrue(follow_up_script.exists())
            self.assertFalse(review_script.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(readme_path.exists())
            script_text = follow_up_script.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            readme = readme_path.read_text(encoding="utf-8")
            self.assertEqual(
                payload["command_scripts"]["follow_up_path"],
                str(follow_up_script),
            )
            self.assertIsNone(payload["command_scripts"]["review_path"])
            self.assertEqual(payload["command_scripts"]["written_count"], 1)
            self.assertEqual(
                payload["command_scripts"]["execution_cwd"],
                execution_cwd,
            )
            self.assertEqual(
                payload["command_scripts"]["manifest_path"],
                str(manifest_path),
            )
            self.assertEqual(
                payload["command_scripts"]["readme_path"],
                str(readme_path),
            )
            self.assertEqual(
                manifest["recommendation"]["action"],
                "continue_from_accepted",
            )
            self.assertEqual(
                manifest["command_scripts"]["follow_up_path"],
                str(follow_up_script),
            )
            self.assertEqual(
                manifest["command_scripts"]["execution_cwd"],
                execution_cwd,
            )
            self.assertIn(f"cd {shlex.quote(execution_cwd)}", script_text)
            self.assertIn("FOLLOW_UP_FROM=accepted NEW_SEEDS=31", script_text)
            self.assertIn(
                f"NEXT_RUN_DIR={shlex.quote(str(default_run_dir))}",
                script_text,
            )
            self.assertIn(
                f"bash {shlex.quote(str(relative_script))}",
                script_text,
            )
            run_result = subprocess.run(
                ["bash", str(follow_up_script)],
                cwd=command_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)
            self.assertEqual(
                (root / "wrapper.out").read_text(encoding="utf-8").strip(),
                (
                    f"cwd={execution_cwd} follow=accepted seeds=31 "
                    f"next={default_run_dir} fail=regressed,unknown"
                ),
            )
            self.assertIn("Char VAE Chain Recommended Commands", readme)
            self.assertIn("continue_from_accepted", readme)
            self.assertIn("execution_cwd", readme)
            self.assertIn("recommended_follow_up.sh", readme)
            self.assertIn("recommended_follow_up.sh", markdown)
            self.assertIn("command_execution_cwd", markdown)
            self.assertIn("recommendation.json", markdown)
            self.assertIn("README.md", markdown)

    def test_selection_separates_safe_accepted_from_absolute_best(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_parent_summary = root / "reviewed" / "parent" / "summary.json"
            reviewed_follow_summary = (
                root / "reviewed" / "follow_up_01" / "summary.json"
            )
            safe_follow_summary = root / "safe" / "follow_up_01" / "summary.json"
            _write_json(
                reviewed_follow_summary,
                {
                    "guided_next_follow_up_command": {
                        "enabled": True,
                        "script_path": str(
                            root / "reviewed" / "follow_up_01" / "guided.sh"
                        ),
                        "script_usage": (
                            "FOLLOW_UP_FROM=reviewed NEW_SEEDS=41 "
                            "bash reviewed/guided.sh"
                        ),
                        "shell_command": "PYTHONNOUSERSITE=1 python reviewed.py",
                        "default_new_seeds": "41",
                        "default_run_dir": str(root / "reviewed" / "next"),
                        "default_follow_up_from": str(reviewed_follow_summary),
                    },
                },
            )
            _write_json(
                safe_follow_summary,
                {
                    "next_follow_up_command": {
                        "script_path": str(root / "safe" / "next.sh"),
                        "script_usage": (
                            "FOLLOW_UP_FROM=safe NEW_SEEDS=31 bash safe/next.sh"
                        ),
                        "shell_command": "PYTHONNOUSERSITE=1 python safe.py",
                        "default_new_seeds": "31",
                        "default_run_dir": str(root / "safe" / "next"),
                        "default_follow_up_from": str(safe_follow_summary),
                    },
                },
            )
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
            command_scripts = mod._write_recommended_command_scripts(
                summary,
                root / "commands",
            )
            review_script = root / "commands" / "recommended_review.sh"
            follow_up_script = root / "commands" / "recommended_follow_up.sh"
            review_script_path = str(review_script)
            follow_up_script_path = str(follow_up_script)
            review_script_exists = review_script.exists()
            follow_up_script_exists = follow_up_script.exists()
            command_readme = (root / "commands" / "README.md").read_text(
                encoding="utf-8"
            )

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
        self.assertEqual(
            summary["recommendation"]["follow_up_command"]["command_source"],
            "next_follow_up_command",
        )
        self.assertEqual(
            summary["recommendation"]["follow_up_command"]["script_usage"],
            "FOLLOW_UP_FROM=safe NEW_SEEDS=31 bash safe/next.sh",
        )
        self.assertEqual(
            summary["recommendation"]["review_command"]["command_source"],
            "guided_next_follow_up_command",
        )
        self.assertEqual(
            summary["recommendation"]["review_command"]["script_usage"],
            "FOLLOW_UP_FROM=reviewed NEW_SEEDS=41 bash reviewed/guided.sh",
        )
        self.assertIn("best_requires_review: yes", markdown)
        self.assertIn("accepted_vs_best_nll_gap: 0.050000", markdown)
        self.assertIn("recommendation: review_absolute_best", markdown)
        self.assertIn(f"follow_up_from={safe_follow_summary}", markdown)
        self.assertIn(f"review={reviewed_follow_summary}", markdown)
        self.assertIn("follow_up_command: next_follow_up_command", markdown)
        self.assertIn("review_command: guided_next_follow_up_command", markdown)
        self.assertEqual(command_scripts["written_count"], 2)
        self.assertEqual(command_scripts["follow_up_path"], follow_up_script_path)
        self.assertEqual(command_scripts["review_path"], review_script_path)
        self.assertTrue(follow_up_script_exists)
        self.assertTrue(review_script_exists)
        self.assertIn("review_absolute_best", command_readme)
        self.assertIn("recommended_follow_up.sh", command_readme)
        self.assertIn("recommended_review.sh", command_readme)

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
