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
INSPECT_SCRIPT = ROOT / "tools" / "inspect_char_vae_command_bundle.py"
RUNNER_SCRIPT = ROOT / "tools" / "run_char_vae_command_bundle.py"


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


def _inspection_command(command_dir: Path) -> str:
    return (
        "PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(INSPECT_SCRIPT.resolve()))} "
        f"{shlex.quote(str(command_dir.resolve()))} --strict --write-report"
    )


def _runner_command(command_dir: Path) -> str:
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(RUNNER_SCRIPT.resolve()))} "
        f"{shlex.quote(str(command_dir.resolve()))} "
        "--write-inspection-report --write-run-report --append-run-history "
        "--write-run-history-report"
    )


def _execution_next_command(command_dir: Path) -> str:
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(RUNNER_SCRIPT.resolve()))} "
        f"{shlex.quote(str(command_dir.resolve()))} --target execution-next "
        "--write-inspection-report --write-run-report --append-run-history "
        "--write-run-history-report"
    )


def _history_next_action_command(command_dir: Path) -> str:
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(RUNNER_SCRIPT.resolve()))} "
        f"{shlex.quote(str(command_dir.resolve()))} --use-history-next-action "
        "--write-inspection-report --write-run-report --append-run-history "
        "--write-run-history-report"
    )


def _history_report_command(command_dir: Path) -> str:
    return (
        "env PYTHONNOUSERSITE=1 python3 -P "
        f"{shlex.quote(str(RUNNER_SCRIPT.resolve()))} "
        f"{shlex.quote(str(command_dir.resolve()))} --history-report-only"
    )


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

    def test_cli_command_dir_bundles_comparison_outputs_by_default(self) -> None:
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
                },
            )
            command_dir = root / "commands"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(chain),
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
            comparison_json = command_dir / "comparison.json"
            comparison_markdown = command_dir / "comparison.md"
            comparison_json_exists = comparison_json.exists()
            comparison_markdown_exists = comparison_markdown.exists()
            comparison_json_resolved = str(comparison_json.resolve())
            comparison_markdown_resolved = str(comparison_markdown.resolve())
            manifest = json.loads(
                (command_dir / "recommendation.json").read_text(encoding="utf-8")
            )
            comparison = json.loads(comparison_json.read_text(encoding="utf-8"))
            readme = (command_dir / "README.md").read_text(encoding="utf-8")
            markdown = comparison_markdown.read_text(encoding="utf-8")
            inspection_command = _inspection_command(command_dir)

        self.assertTrue(comparison_json_exists)
        self.assertTrue(comparison_markdown_exists)
        self.assertEqual(
            comparison["command_scripts"]["comparison_json_path"],
            comparison_json_resolved,
        )
        self.assertEqual(
            comparison["command_scripts"]["comparison_markdown_path"],
            comparison_markdown_resolved,
        )
        self.assertEqual(
            manifest["command_scripts"]["comparison_json_path"],
            comparison_json_resolved,
        )
        self.assertEqual(
            manifest["command_scripts"]["comparison_markdown_path"],
            comparison_markdown_resolved,
        )
        self.assertEqual(manifest["comparison"]["chain_sources"], [str(chain)])
        self.assertEqual(
            manifest["command_scripts"]["inspection_command"],
            inspection_command,
        )
        self.assertEqual(
            manifest["command_scripts"]["inspection_json_path"],
            str((command_dir / "inspection.json").resolve()),
        )
        self.assertEqual(
            manifest["command_scripts"]["inspection_markdown_path"],
            str((command_dir / "inspection.md").resolve()),
        )
        self.assertEqual(
            manifest["command_scripts"]["runner_command"],
            _runner_command(command_dir),
        )
        self.assertEqual(
            manifest["command_scripts"]["execution_next_command"],
            _execution_next_command(command_dir),
        )
        self.assertEqual(
            manifest["command_scripts"]["history_next_action_command"],
            _history_next_action_command(command_dir),
        )
        self.assertEqual(
            manifest["command_scripts"]["history_report_command"],
            _history_report_command(command_dir),
        )
        self.assertIsNone(manifest["command_scripts"]["runner_path"])
        self.assertEqual(
            manifest["command_scripts"]["run_json_path"],
            str((command_dir / "run.json").resolve()),
        )
        self.assertEqual(
            manifest["command_scripts"]["run_markdown_path"],
            str((command_dir / "run.md").resolve()),
        )
        self.assertEqual(
            manifest["command_scripts"]["run_history_jsonl_path"],
            str((command_dir / "run_history.jsonl").resolve()),
        )
        self.assertEqual(
            manifest["command_scripts"]["run_history_markdown_path"],
            str((command_dir / "run_history.md").resolve()),
        )
        self.assertEqual(
            manifest["command_scripts"]["run_history_summary_path"],
            str((command_dir / "run_history_summary.json").resolve()),
        )
        self.assertIn("## Bundle Inspection", readme)
        self.assertIn(inspection_command, readme)
        self.assertIn(_runner_command(command_dir), readme)
        self.assertIn(_execution_next_command(command_dir), readme)
        self.assertIn(_history_next_action_command(command_dir), readme)
        self.assertIn("## History-Guided Continuation", readme)
        self.assertIn("--use-history-next-action", readme)
        self.assertIn("## Execution-Next Continuation", readme)
        self.assertIn("--target execution-next", readme)
        self.assertIn("run.json", readme)
        self.assertIn("run.md", readme)
        self.assertIn("run_history.jsonl", readme)
        self.assertIn("run_history.md", readme)
        self.assertIn("run_history_summary.json", readme)
        self.assertIn("inspection.json", readme)
        self.assertIn("inspection.md", readme)
        self.assertIn(_history_report_command(command_dir), readme)
        self.assertIn(comparison_json_resolved, readme)
        self.assertIn(comparison_markdown_resolved, readme)
        self.assertIn("command_manifest_path", markdown)
        self.assertIn("command_runner", markdown)
        self.assertIn("command_execution_next", markdown)
        self.assertIn("command_history_next_action", markdown)
        self.assertIn("command_run_json_path", markdown)
        self.assertIn("command_run_markdown_path", markdown)
        self.assertIn("command_run_history_jsonl_path", markdown)
        self.assertIn("command_run_history_markdown_path", markdown)
        self.assertIn("command_run_history_summary_path", markdown)
        self.assertIn("command_history_report_only", markdown)
        self.assertIn("recommendation.json", result.stdout)

    def test_cli_write_command_inspection_materializes_reports(self) -> None:
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
                },
            )
            command_dir = root / "commands"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(chain),
                    "--command-out-dir",
                    str(command_dir),
                    "--write-command-inspection",
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (command_dir / "recommendation.json").read_text(encoding="utf-8")
            )
            comparison = json.loads(
                (command_dir / "comparison.json").read_text(encoding="utf-8")
            )
            inspection = json.loads(
                (command_dir / "inspection.json").read_text(encoding="utf-8")
            )
            readme = (command_dir / "README.md").read_text(encoding="utf-8")
            markdown = (command_dir / "comparison.md").read_text(encoding="utf-8")
            inspection_commands = {
                item["label"]: item for item in inspection["declared_commands"]
            }
            comparison_commands = {
                item["label"]: item
                for item in comparison["command_inspection"]["declared_commands"]
            }

        self.assertTrue(inspection["bundle_ready"])
        self.assertTrue(inspection["strict_ready"])
        self.assertEqual(
            inspection["history_report_command"],
            _history_report_command(command_dir),
        )
        self.assertTrue(inspection_commands["history_report_command"]["ok"])
        self.assertTrue(comparison_commands["history_report_command"]["ok"])
        self.assertTrue(manifest["command_scripts"]["inspection_generated"])
        self.assertTrue(manifest["command_scripts"]["inspection_bundle_ready"])
        self.assertTrue(manifest["command_scripts"]["inspection_strict_ready"])
        self.assertEqual(manifest["command_scripts"]["inspection_missing_required"], [])
        self.assertEqual(manifest["command_scripts"]["inspection_missing_optional"], [])
        self.assertEqual(
            manifest["command_scripts"]["inspection_runner_wrapper_status"],
            inspection["runner_wrapper_status"],
        )
        self.assertTrue(manifest["command_scripts"]["inspection_runner_wrapper_ok"])
        self.assertIsNone(
            manifest["command_scripts"][
                "inspection_runner_wrapper_executes_runner_command"
            ]
        )
        self.assertIsNone(
            manifest["command_scripts"]["inspection_runner_wrapper_forwards_arguments"]
        )
        self.assertTrue(comparison["command_scripts"]["inspection_generated"])
        self.assertTrue(comparison["command_inspection"]["strict_ready"])
        self.assertTrue(comparison["command_scripts"]["inspection_runner_wrapper_ok"])
        self.assertEqual(
            comparison["command_inspection"]["inspection_json_path"],
            str((command_dir / "inspection.json").resolve()),
        )
        self.assertIn("- generated_now: `yes`", readme)
        self.assertIn("- strict_ready: `yes`", readme)
        self.assertIn("command_inspection_generated: yes", markdown)
        self.assertIn("command_inspection_strict_ready: yes", markdown)
        self.assertIn("command_inspection_missing_required: -", markdown)
        self.assertIn("command_inspection_runner_wrapper_ok: yes", markdown)
        self.assertIn("runner_wrapper_ok: `yes`", readme)

    def test_cli_command_dir_records_absolute_handoff_paths(self) -> None:
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
                },
            )
            command_dir = root / "commands"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(chain),
                    "--command-out-dir",
                    "commands",
                    "--write-command-inspection",
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (command_dir / "recommendation.json").read_text(encoding="utf-8")
            )
            comparison = json.loads(
                (command_dir / "comparison.json").read_text(encoding="utf-8")
            )
            readme = (command_dir / "README.md").read_text(encoding="utf-8")
            markdown = (command_dir / "comparison.md").read_text(encoding="utf-8")

        command_scripts = manifest["command_scripts"]
        self.assertEqual(command_scripts["directory"], str(command_dir.resolve()))
        self.assertIsNone(command_scripts["next_path"])
        self.assertEqual(
            command_scripts["manifest_path"],
            str((command_dir / "recommendation.json").resolve()),
        )
        self.assertEqual(
            command_scripts["inspection_command"],
            _inspection_command(command_dir),
        )
        self.assertEqual(
            command_scripts["runner_command"],
            _runner_command(command_dir),
        )
        self.assertEqual(
            command_scripts["execution_next_command"],
            _execution_next_command(command_dir),
        )
        self.assertEqual(
            command_scripts["history_next_action_command"],
            _history_next_action_command(command_dir),
        )
        self.assertEqual(
            command_scripts["history_report_command"],
            _history_report_command(command_dir),
        )
        self.assertIsNone(command_scripts["runner_path"])
        self.assertEqual(
            command_scripts["run_json_path"],
            str((command_dir / "run.json").resolve()),
        )
        self.assertEqual(
            command_scripts["run_markdown_path"],
            str((command_dir / "run.md").resolve()),
        )
        self.assertEqual(
            command_scripts["run_history_jsonl_path"],
            str((command_dir / "run_history.jsonl").resolve()),
        )
        self.assertEqual(
            command_scripts["run_history_markdown_path"],
            str((command_dir / "run_history.md").resolve()),
        )
        self.assertEqual(
            command_scripts["run_history_summary_path"],
            str((command_dir / "run_history_summary.json").resolve()),
        )
        self.assertEqual(
            comparison["command_scripts"]["directory"],
            str(command_dir.resolve()),
        )
        self.assertIn(str((command_dir / "inspection.json").resolve()), readme)
        self.assertIn(_runner_command(command_dir), readme)
        self.assertIn(_execution_next_command(command_dir), readme)
        self.assertIn(_history_next_action_command(command_dir), readme)
        self.assertIn(_history_report_command(command_dir), readme)
        self.assertIn("## History-Guided Continuation", readme)
        self.assertIn("## Execution-Next Continuation", readme)
        self.assertIn(str((command_dir / "run.json").resolve()), readme)
        self.assertIn(str((command_dir / "run_history.jsonl").resolve()), readme)
        self.assertIn(str((command_dir / "run_history.md").resolve()), readme)
        self.assertIn(str((command_dir / "run_history_summary.json").resolve()), readme)
        self.assertIn(str((command_dir / "recommendation.json").resolve()), markdown)
        self.assertIn(_runner_command(command_dir), markdown)
        self.assertIn(_execution_next_command(command_dir), markdown)
        self.assertIn(_history_next_action_command(command_dir), markdown)
        self.assertIn(_history_report_command(command_dir), markdown)
        self.assertIn(str((command_dir / "run.md").resolve()), markdown)
        self.assertIn(str((command_dir / "run_history.jsonl").resolve()), markdown)
        self.assertIn(str((command_dir / "run_history.md").resolve()), markdown)
        self.assertIn(str((command_dir / "run_history_summary.json").resolve()), markdown)

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
                    "--write-command-inspection",
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
            next_script = command_dir / "recommended_next.sh"
            runner_script = command_dir / "run_recommended_next.sh"
            follow_up_script = command_dir / "recommended_follow_up.sh"
            review_script = command_dir / "recommended_review.sh"
            manifest_path = command_dir / "recommendation.json"
            readme_path = command_dir / "README.md"
            next_script_path = str(next_script.resolve())
            runner_script_path = str(runner_script.resolve())
            follow_up_script_path = str(follow_up_script.resolve())
            manifest_path_resolved = str(manifest_path.resolve())
            readme_path_resolved = str(readme_path.resolve())
            command_dir_resolved = str(command_dir.resolve())
            execution_cwd = str(root.resolve())
            self.assertTrue(next_script.exists())
            self.assertTrue(runner_script.exists())
            self.assertTrue(follow_up_script.exists())
            self.assertFalse(review_script.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(readme_path.exists())
            next_text = next_script.read_text(encoding="utf-8")
            runner_text = runner_script.read_text(encoding="utf-8")
            script_text = follow_up_script.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inspection = json.loads(
                (command_dir / "inspection.json").read_text(encoding="utf-8")
            )
            readme = readme_path.read_text(encoding="utf-8")
            self.assertEqual(
                payload["command_scripts"]["next_path"],
                next_script_path,
            )
            self.assertEqual(payload["command_scripts"]["next_kind"], "follow_up")
            self.assertEqual(
                payload["command_scripts"]["follow_up_path"],
                follow_up_script_path,
            )
            self.assertIsNone(payload["command_scripts"]["review_path"])
            self.assertEqual(payload["command_scripts"]["written_count"], 3)
            self.assertEqual(
                payload["command_scripts"]["execution_cwd"],
                execution_cwd,
            )
            self.assertEqual(
                payload["command_scripts"]["comparison_json_path"],
                str(json_out.resolve()),
            )
            self.assertEqual(
                payload["command_scripts"]["comparison_markdown_path"],
                str(markdown_out.resolve()),
            )
            self.assertEqual(
                payload["command_scripts"]["manifest_path"],
                manifest_path_resolved,
            )
            self.assertEqual(
                payload["command_scripts"]["readme_path"],
                readme_path_resolved,
            )
            self.assertEqual(
                payload["command_scripts"]["runner_path"],
                runner_script_path,
            )
            self.assertEqual(
                manifest["recommendation"]["action"],
                "continue_from_accepted",
            )
            self.assertEqual(manifest["comparison"]["sort_by"], "input")
            self.assertIs(manifest["comparison"]["recursive"], False)
            self.assertEqual(manifest["comparison"]["chain_sources"], [str(chain)])
            self.assertEqual(manifest["aggregate"]["chain_count"], 1)
            self.assertEqual(
                manifest["selection"]["accepted_champion"]["config"],
                "latent@normalize=blocks,scale=0.5",
            )
            self.assertIs(manifest["selection"]["accepted_matches_best"], True)
            self.assertEqual(
                manifest["command_scripts"]["follow_up_path"],
                follow_up_script_path,
            )
            self.assertEqual(manifest["command_scripts"]["next_kind"], "follow_up")
            self.assertEqual(
                manifest["command_scripts"]["runner_path"],
                runner_script_path,
            )
            self.assertTrue(
                manifest["command_scripts"]["inspection_runner_wrapper_ok"]
            )
            self.assertTrue(
                manifest["command_scripts"][
                    "inspection_runner_wrapper_executes_runner_command"
                ]
            )
            self.assertTrue(
                manifest["command_scripts"][
                    "inspection_runner_wrapper_forwards_arguments"
                ]
            )
            self.assertEqual(
                manifest["command_scripts"]["inspection_runner_wrapper_status"],
                inspection["runner_wrapper_status"],
            )
            self.assertTrue(
                payload["command_scripts"]["inspection_runner_wrapper_ok"]
            )
            self.assertTrue(
                payload["command_scripts"][
                    "inspection_runner_wrapper_executes_runner_command"
                ]
            )
            self.assertEqual(
                manifest["command_scripts"]["execution_cwd"],
                execution_cwd,
            )
            self.assertEqual(
                manifest["command_scripts"]["comparison_json_path"],
                str(json_out.resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["comparison_markdown_path"],
                str(markdown_out.resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["inspection_command"],
                _inspection_command(command_dir),
            )
            self.assertEqual(
                manifest["command_scripts"]["inspection_json_path"],
                str((command_dir / "inspection.json").resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["inspection_markdown_path"],
                str((command_dir / "inspection.md").resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["runner_command"],
                _runner_command(command_dir),
            )
            self.assertEqual(
                manifest["command_scripts"]["execution_next_command"],
                _execution_next_command(command_dir),
            )
            self.assertEqual(
                manifest["command_scripts"]["history_report_command"],
                _history_report_command(command_dir),
            )
            self.assertEqual(
                manifest["command_scripts"]["run_json_path"],
                str((command_dir / "run.json").resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["run_markdown_path"],
                str((command_dir / "run.md").resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["run_history_jsonl_path"],
                str((command_dir / "run_history.jsonl").resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["run_history_markdown_path"],
                str((command_dir / "run_history.md").resolve()),
            )
            self.assertEqual(
                manifest["command_scripts"]["run_history_summary_path"],
                str((command_dir / "run_history_summary.json").resolve()),
            )
            self.assertIn("# target_kind: follow_up", next_text)
            self.assertIn("recommended_follow_up.sh", next_text)
            self.assertIn("# target_kind: follow_up", runner_text)
            self.assertIn("exec env PYTHONNOUSERSITE=1 python3 -P", runner_text)
            self.assertIn("tools/run_char_vae_command_bundle.py", runner_text)
            self.assertIn("--write-run-report", runner_text)
            self.assertIn("--append-run-history", runner_text)
            self.assertIn("--write-run-history-report", runner_text)
            self.assertIn("runner_wrapper_ok: `yes`", readme)
            self.assertIn("runner_wrapper_executes_runner_command: `yes`", readme)
            self.assertIn("## Execution-Next Continuation", readme)
            self.assertIn(_execution_next_command(command_dir), readme)
            self.assertIn("--target execution-next", readme)
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
            (root / "wrapper.out").unlink()
            next_result = subprocess.run(
                ["bash", str(next_script)],
                cwd=command_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(next_result.returncode, 0, next_result.stderr)
            self.assertEqual(
                (root / "wrapper.out").read_text(encoding="utf-8").strip(),
                (
                    f"cwd={execution_cwd} follow=accepted seeds=31 "
                    f"next={default_run_dir} fail=regressed,unknown"
                ),
            )
            self.assertIn("Char VAE Chain Recommended Commands", readme)
            self.assertIn("continue_from_accepted", readme)
            self.assertIn("## Champion", readme)
            self.assertIn("latent@normalize=blocks,scale=0.5", readme)
            self.assertIn("mean_best_nll", readme)
            self.assertIn("## Fallback", readme)
            self.assertIn("execution_cwd", readme)
            self.assertIn("## Comparison Artifacts", readme)
            self.assertIn(str(json_out.resolve()), readme)
            self.assertIn(str(markdown_out.resolve()), readme)
            self.assertIn(str(chain), readme)
            self.assertIn("## Bundle Inspection", readme)
            self.assertIn("tools/inspect_char_vae_command_bundle.py", readme)
            self.assertIn("tools/run_char_vae_command_bundle.py", readme)
            self.assertIn("inspection.json", readme)
            self.assertIn("inspection.md", readme)
            self.assertIn("recommended_next.sh", readme)
            self.assertIn(f"bash {shlex.quote(next_script_path)}", readme)
            self.assertIn("run_recommended_next.sh", readme)
            self.assertIn(f"bash {shlex.quote(runner_script_path)}", readme)
            self.assertIn("run_history.jsonl", readme)
            self.assertIn("run_history.md", readme)
            self.assertIn("run_history_summary.json", readme)
            self.assertIn("recommended_follow_up.sh", readme)
            self.assertIn("recommended_next.sh", markdown)
            self.assertIn("run_recommended_next.sh", markdown)
            self.assertIn("recommended_follow_up.sh", markdown)
            self.assertIn("next_command_kind: follow_up", markdown)
            self.assertIn("command_execution_cwd", markdown)
            self.assertIn("recommendation.json", markdown)
            self.assertIn("README.md", markdown)
            self.assertIn("command_inspection:", markdown)
            self.assertIn("tools/inspect_char_vae_command_bundle.py", markdown)
            self.assertIn("command_inspection_json_path", markdown)
            self.assertIn("command_inspection_markdown_path", markdown)
            self.assertIn("command_runner", markdown)
            self.assertIn("command_execution_next", markdown)
            self.assertIn("command_runner_script", markdown)
            self.assertIn("tools/run_char_vae_command_bundle.py", markdown)
            self.assertIn("command_run_json_path", markdown)
            self.assertIn("command_run_markdown_path", markdown)
            self.assertIn("command_run_history_jsonl_path", markdown)
            self.assertIn("command_run_history_markdown_path", markdown)
            self.assertIn("command_run_history_summary_path", markdown)
            self.assertIn("command_history_report_only", markdown)

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
            next_script = root / "commands" / "recommended_next.sh"
            runner_script = root / "commands" / "run_recommended_next.sh"
            next_script_path = str(next_script.resolve())
            runner_script_path = str(runner_script.resolve())
            review_script_path = str(review_script.resolve())
            follow_up_script_path = str(follow_up_script.resolve())
            next_script_exists = next_script.exists()
            runner_script_exists = runner_script.exists()
            review_script_exists = review_script.exists()
            follow_up_script_exists = follow_up_script.exists()
            next_script_text = next_script.read_text(encoding="utf-8")
            runner_script_text = runner_script.read_text(encoding="utf-8")
            command_manifest = json.loads(
                (root / "commands" / "recommendation.json").read_text(
                    encoding="utf-8"
                )
            )
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
        self.assertEqual(command_scripts["written_count"], 4)
        self.assertEqual(command_scripts["next_path"], next_script_path)
        self.assertEqual(command_scripts["next_kind"], "review")
        self.assertEqual(command_scripts["runner_path"], runner_script_path)
        self.assertEqual(command_scripts["follow_up_path"], follow_up_script_path)
        self.assertEqual(command_scripts["review_path"], review_script_path)
        self.assertEqual(
            command_manifest["command_scripts"]["runner_path"],
            runner_script_path,
        )
        self.assertEqual(
            command_manifest["command_scripts"]["execution_next_command"],
            _execution_next_command(root / "commands"),
        )
        self.assertEqual(command_manifest["comparison"]["sort_by"], "best")
        self.assertEqual(command_manifest["aggregate"]["chain_count"], 2)
        self.assertIs(command_manifest["selection"]["best_requires_review"], True)
        self.assertEqual(
            command_manifest["selection"]["best_champion"]["config"],
            "latent@normalize=blocks,scale=4.0",
        )
        self.assertTrue(next_script_exists)
        self.assertTrue(runner_script_exists)
        self.assertTrue(follow_up_script_exists)
        self.assertTrue(review_script_exists)
        self.assertIn("# target_kind: review", next_script_text)
        self.assertIn("recommended_review.sh", next_script_text)
        self.assertIn("# target_kind: review", runner_script_text)
        self.assertIn(
            "exec env PYTHONNOUSERSITE=1 python3 -P",
            runner_script_text,
        )
        self.assertIn("tools/run_char_vae_command_bundle.py", runner_script_text)
        self.assertIn("--write-run-report", runner_script_text)
        self.assertIn("--append-run-history", runner_script_text)
        self.assertIn("--write-run-history-report", runner_script_text)
        self.assertIn("review_absolute_best", command_readme)
        self.assertIn("## Champion", command_readme)
        self.assertIn("latent@normalize=blocks,scale=4.0", command_readme)
        self.assertIn("## Fallback", command_readme)
        self.assertIn("raw_latent@normalize=blocks,scale=4.0", command_readme)
        self.assertIn("recommended_next.sh", command_readme)
        self.assertIn("run_recommended_next.sh", command_readme)
        self.assertIn("## Execution-Next Continuation", command_readme)
        self.assertIn(_execution_next_command(root / "commands"), command_readme)
        self.assertIn("--target execution-next", command_readme)
        self.assertIn("run_history.jsonl", command_readme)
        self.assertIn("run_history.md", command_readme)
        self.assertIn("run_history_summary.json", command_readme)
        self.assertIn("recommended_follow_up.sh", command_readme)
        self.assertIn("recommended_review.sh", command_readme)

    def test_summary_command_prefers_feature_swap_review_when_guided_disabled(
        self,
    ) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            feature_swap_script = root / "feature_swap_review_command.sh"
            next_script = root / "next_follow_up_command.sh"
            _write_json(
                summary_path,
                {
                    "guided_next_follow_up_command": {
                        "enabled": False,
                        "script_path": None,
                    },
                    "feature_swap_review_command": {
                        "script_path": str(feature_swap_script),
                        "script_usage": (
                            "FOLLOW_UP_FROM=current NEW_SEEDS=101 "
                            "bash feature_swap_review_command.sh"
                        ),
                        "shell_command": "PYTHONNOUSERSITE=1 python review.py",
                        "default_new_seeds": "101",
                        "default_run_dir": str(root / "feature_swap_review"),
                        "default_follow_up_from": str(summary_path),
                    },
                    "next_follow_up_command": {
                        "script_path": str(next_script),
                        "script_usage": "NEW_SEEDS=103 bash next_follow_up_command.sh",
                        "shell_command": "PYTHONNOUSERSITE=1 python next.py",
                        "default_new_seeds": "103",
                        "default_run_dir": str(root / "next"),
                        "default_follow_up_from": str(summary_path),
                    },
                },
            )

            command = mod._summary_follow_up_command(str(summary_path))

        self.assertTrue(command["available"])
        self.assertEqual(command["command_source"], "feature_swap_review_command")
        self.assertEqual(command["script_path"], str(feature_swap_script))
        self.assertEqual(command["default_new_seeds"], "101")

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
