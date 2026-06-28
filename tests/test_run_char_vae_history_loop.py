#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_char_vae_history_loop.py"
BUNDLE_TESTS = ROOT / "tests" / "test_run_char_vae_command_bundle.py"


def _load_bundle_helpers():
    spec = importlib.util.spec_from_file_location(
        "test_run_char_vae_command_bundle_helpers",
        BUNDLE_TESTS,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_bundle(root: Path) -> Path:
    helpers = _load_bundle_helpers()
    return helpers._write_bundle(root)


def _run_loop(command_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "-P", str(SCRIPT), str(command_dir), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class RunCharVaeHistoryLoopTests(unittest.TestCase):
    def test_cli_runs_history_guided_steps_until_next_command_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--max-steps",
                "3",
                "--write-loop-report",
                "--json",
            )
            payload = json.loads(result.stdout)
            history_events = [
                json.loads(line)
                for line in (command_dir / "run_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            history_summary = json.loads(
                (command_dir / "run_history_summary.json").read_text(encoding="utf-8")
            )
            loop_report = json.loads(
                (command_dir / "run_loop.json").read_text(encoding="utf-8")
            )
            loop_markdown = (command_dir / "run_loop.md").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            payload["schema"],
            "st.llm_char_vae_context.command_bundle_history_loop.v1",
        )
        self.assertEqual(payload["step_count"], 2)
        self.assertEqual(payload["executed_count"], 2)
        self.assertEqual(payload["success_count"], 2)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["stop_reason"], "history_next_action_stopped")
        self.assertEqual(payload["returncode"], 0)
        self.assertEqual(payload["steps"][0]["target"], "next")
        self.assertEqual(payload["steps"][0]["target_kind"], "follow_up")
        self.assertEqual(
            payload["steps"][0]["run_history_next_action"]["action"],
            "run_execution_next",
        )
        self.assertEqual(
            payload["steps"][0]["run_history_next_action"]["target"],
            "execution-next",
        )
        self.assertEqual(payload["steps"][1]["target"], "execution-next")
        self.assertEqual(payload["steps"][1]["target_kind"], "execution_next")
        self.assertEqual(
            payload["steps"][1]["run_history_next_action"]["action"],
            "collect_next_command",
        )
        self.assertIs(
            payload["steps"][1]["run_history_next_action"]["should_continue"],
            False,
        )
        self.assertEqual(payload["final_next_action"]["action"], "collect_next_command")
        self.assertEqual(len(history_events), 2)
        self.assertEqual(history_summary["total_runs"], 2)
        self.assertEqual(history_summary["next_action"]["action"], "collect_next_command")
        self.assertEqual(loop_report["stop_reason"], payload["stop_reason"])
        self.assertIn("Char VAE History Loop Runner", loop_markdown)
        self.assertIn("stop_reason: history_next_action_stopped", loop_markdown)
        self.assertIn("| 1 | next | follow_up | yes | 0 | improved |", loop_markdown)
        self.assertIn(
            "| 2 | execution-next | execution_next | yes | 0 | promote |",
            loop_markdown,
        )

    def test_cli_dry_run_resolves_one_step_without_appending_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--dry-run",
                "--max-steps",
                "5",
                "--write-loop-report",
                "--json",
            )
            payload = json.loads(result.stdout)
            history_jsonl_exists = (command_dir / "run_history.jsonl").exists()
            history_summary = json.loads(
                (command_dir / "run_history_summary.json").read_text(encoding="utf-8")
            )
            runner_out_exists = (command_dir / "runner.out").exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["executed_count"], 0)
        self.assertEqual(payload["stop_reason"], "dry_run")
        self.assertEqual(payload["steps"][0]["target"], "next")
        self.assertFalse(payload["steps"][0]["executed"])
        self.assertFalse(history_jsonl_exists)
        self.assertEqual(history_summary["total_runs"], 0)
        self.assertFalse(runner_out_exists)

    def test_cli_stops_before_execution_when_history_requires_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            history_event = {
                "schema": "st.llm_char_vae_context.command_bundle_run_history_event.v1",
                "target": "next",
                "target_kind": "follow_up",
                "dry_run": False,
                "executed": True,
                "returncode": 0,
                "execution_summary": {
                    "exists": True,
                    "valid_json": True,
                    "follow_up_gate_failed": True,
                    "follow_up_verdict": "regressed",
                    "next_command": {
                        "source": "guided_next_follow_up_command",
                        "script_path": str(command_dir / "guided_next.sh"),
                        "default_new_seeds": "109,113,127",
                    },
                },
            }
            (command_dir / "run_history.jsonl").write_text(
                json.dumps(history_event, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = _run_loop(command_dir, "--max-steps", "3", "--json")
            payload = json.loads(result.stdout)
            history_events = [
                json.loads(line)
                for line in (command_dir / "run_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["executed_count"], 0)
        self.assertEqual(payload["failure_count"], 1)
        self.assertEqual(payload["stop_reason"], "history_next_action_blocked")
        self.assertEqual(
            payload["error"],
            (
                "history next action does not allow continuation: "
                "follow-up gate requested stop"
            ),
        )
        self.assertEqual(
            payload["steps"][0]["history_next_action"]["action"],
            "review_before_continuing",
        )
        self.assertFalse(payload["steps"][0]["executed"])
        self.assertEqual(len(history_events), 1)
        self.assertFalse((command_dir / "runner.out").exists())


if __name__ == "__main__":
    unittest.main()
