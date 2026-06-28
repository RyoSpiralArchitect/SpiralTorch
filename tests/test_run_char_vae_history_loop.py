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
        self.assertEqual(payload["handoff_status"], "awaiting_next_command")
        self.assertEqual(
            payload["handoff_reason"],
            "latest execution can continue but has no next script",
        )
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
        self.assertIs(payload["final_next_action_runnable"], False)
        self.assertIsNone(payload["continuation_command"])
        self.assertIsNone(payload["resume_from_report_command"])
        self.assertEqual(payload["handoff_recommended_action"], "collect_next_command")
        self.assertIsNone(payload["handoff_recommended_command"])
        self.assertEqual(len(history_events), 2)
        self.assertEqual(history_summary["total_runs"], 2)
        self.assertEqual(history_summary["next_action"]["action"], "collect_next_command")
        self.assertEqual(loop_report["stop_reason"], payload["stop_reason"])
        self.assertEqual(
            loop_report["handoff_recommended_action"],
            payload["handoff_recommended_action"],
        )
        self.assertIn("Char VAE History Loop Runner", loop_markdown)
        self.assertIn("handoff_status: awaiting_next_command", loop_markdown)
        self.assertIn(
            "handoff_recommended_action: collect_next_command",
            loop_markdown,
        )
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
        self.assertEqual(payload["handoff_status"], "dry_run")
        self.assertEqual(
            payload["handoff_reason"],
            "dry-run resolved the next history-guided step",
        )
        self.assertEqual(payload["steps"][0]["target"], "next")
        self.assertFalse(payload["steps"][0]["executed"])
        self.assertFalse(history_jsonl_exists)
        self.assertEqual(history_summary["total_runs"], 0)
        self.assertFalse(runner_out_exists)

    def test_cli_writes_loop_report_when_setup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--max-steps",
                "0",
                "--write-loop-report",
                "--json",
            )
            payload = json.loads(result.stdout)
            loop_report = json.loads(
                (command_dir / "run_loop.json").read_text(encoding="utf-8")
            )
            loop_markdown = (command_dir / "run_loop.md").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["handoff_status"], "failed")
        self.assertEqual(payload["handoff_reason"], "--max-steps must be at least 1")
        self.assertEqual(payload["handoff_recommended_action"], "inspect_failure")
        self.assertIsNone(payload["handoff_recommended_command"])
        self.assertEqual(payload["stop_reason"], "loop_setup_failed")
        self.assertEqual(payload["step_count"], 0)
        self.assertEqual(payload["returncode"], 1)
        self.assertEqual(payload["error"], "--max-steps must be at least 1")
        self.assertEqual(loop_report["handoff_status"], "failed")
        self.assertEqual(loop_report["handoff_recommended_action"], "inspect_failure")
        self.assertEqual(loop_report["stop_reason"], "loop_setup_failed")
        self.assertEqual(loop_report["returncode"], 1)
        self.assertIn("handoff_status: failed", loop_markdown)
        self.assertIn("handoff_recommended_action: inspect_failure", loop_markdown)
        self.assertIn("stop_reason: loop_setup_failed", loop_markdown)
        self.assertIn("--max-steps must be at least 1", loop_markdown)

    def test_cli_can_fail_when_max_steps_leaves_runnable_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--max-steps",
                "1",
                "--fail-on-max-steps-continuation",
                "--write-loop-report",
                "--json",
            )
            payload = json.loads(result.stdout)
            loop_report = json.loads(
                (command_dir / "run_loop.json").read_text(encoding="utf-8")
            )
            loop_markdown = (command_dir / "run_loop.md").read_text(encoding="utf-8")
            resume_result = _run_loop(command_dir, "--resume-from-report", "--json")
            resume_payload = json.loads(resume_result.stdout)
            resume_loop_report = json.loads(
                (command_dir / "run_loop.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["executed_count"], 1)
        self.assertEqual(payload["success_count"], 1)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["stop_reason"], "max_steps_reached")
        self.assertEqual(payload["handoff_status"], "continuation_ready")
        self.assertEqual(
            payload["handoff_reason"],
            "latest execution summary exposes a next command",
        )
        self.assertEqual(payload["returncode"], 1)
        self.assertIs(payload["fail_on_max_steps_continuation"], True)
        self.assertIs(payload["max_steps_continuation_failed"], True)
        self.assertEqual(
            payload["error"],
            "max steps reached with runnable final next action: run_execution_next",
        )
        self.assertEqual(payload["final_next_action"]["action"], "run_execution_next")
        self.assertEqual(payload["final_next_action"]["target"], "execution-next")
        self.assertEqual(
            payload["final_next_action"]["command_source"],
            "guided_next_follow_up_command",
        )
        self.assertIn(
            "guided_next_follow_up_command.sh",
            payload["final_next_action"]["script_path"],
        )
        self.assertEqual(
            payload["final_next_action"]["default_new_seeds"],
            "109,113,127",
        )
        self.assertIs(payload["final_next_action"]["should_continue"], True)
        self.assertIs(payload["final_next_action_runnable"], True)
        self.assertIn(
            "tools/run_char_vae_history_loop.py",
            payload["continuation_command"],
        )
        self.assertIn(str(command_dir.resolve()), payload["continuation_command"])
        self.assertIn("--max-steps 1", payload["continuation_command"])
        self.assertIn(
            "--fail-on-max-steps-continuation",
            payload["continuation_command"],
        )
        self.assertIn("--write-loop-report", payload["continuation_command"])
        self.assertIn(
            "tools/run_char_vae_history_loop.py",
            payload["resume_from_report_command"],
        )
        self.assertIn(str(command_dir.resolve()), payload["resume_from_report_command"])
        self.assertIn("--resume-from-report", payload["resume_from_report_command"])
        self.assertEqual(
            payload["handoff_recommended_action"],
            "run_resume_from_report_command",
        )
        self.assertEqual(
            payload["handoff_recommended_command"],
            payload["resume_from_report_command"],
        )
        self.assertEqual(loop_report["returncode"], 1)
        self.assertTrue(loop_report["max_steps_continuation_failed"])
        self.assertEqual(
            loop_report["continuation_command"],
            payload["continuation_command"],
        )
        self.assertEqual(
            loop_report["resume_from_report_command"],
            payload["resume_from_report_command"],
        )
        self.assertEqual(
            loop_report["handoff_recommended_command"],
            payload["resume_from_report_command"],
        )
        self.assertIn("stop_reason: max_steps_reached", loop_markdown)
        self.assertIn("handoff_status: continuation_ready", loop_markdown)
        self.assertIn(
            "handoff_recommended_action: run_resume_from_report_command",
            loop_markdown,
        )
        self.assertIn("handoff_recommended_command:", loop_markdown)
        self.assertIn("max_steps_continuation_failed: yes", loop_markdown)
        self.assertIn(
            "final_next_action_command_source: guided_next_follow_up_command",
            loop_markdown,
        )
        self.assertIn("final_next_action_script_path:", loop_markdown)
        self.assertIn("guided_next_follow_up_command.sh", loop_markdown)
        self.assertIn(
            "final_next_action_default_new_seeds: 109,113,127",
            loop_markdown,
        )
        self.assertIn("continuation_command:", loop_markdown)
        self.assertIn("resume_from_report_command:", loop_markdown)
        self.assertEqual(resume_result.returncode, 0, resume_result.stderr)
        self.assertEqual(resume_payload["step_count"], 1)
        self.assertEqual(resume_payload["executed_count"], 1)
        self.assertEqual(resume_payload["success_count"], 1)
        self.assertEqual(resume_payload["failure_count"], 0)
        self.assertEqual(resume_payload["stop_reason"], "history_next_action_stopped")
        self.assertEqual(resume_payload["handoff_status"], "awaiting_next_command")
        self.assertEqual(resume_payload["steps"][0]["target"], "execution-next")
        self.assertEqual(resume_payload["steps"][0]["target_kind"], "execution_next")
        self.assertEqual(
            resume_payload["final_next_action"]["action"],
            "collect_next_command",
        )
        self.assertIs(resume_payload["final_next_action_runnable"], False)
        self.assertIsNone(resume_payload["continuation_command"])
        self.assertIsNone(resume_payload["resume_from_report_command"])
        self.assertEqual(
            resume_payload["handoff_recommended_action"],
            "collect_next_command",
        )
        self.assertIsNone(resume_payload["handoff_recommended_command"])
        self.assertEqual(resume_loop_report["handoff_status"], "awaiting_next_command")

    def test_cli_resume_from_custom_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command_dir = _write_bundle(root)
            custom_report = root / "reports" / "custom_run_loop.json"
            custom_markdown = root / "reports" / "custom_run_loop.md"
            result = _run_loop(
                command_dir,
                "--max-steps",
                "1",
                "--json-out",
                str(custom_report),
                "--markdown-out",
                str(custom_markdown),
                "--json",
            )
            payload = json.loads(result.stdout)
            loop_report = json.loads(custom_report.read_text(encoding="utf-8"))
            resume_result = _run_loop(
                command_dir,
                "--resume-from-report",
                str(custom_report),
                "--json",
            )
            resume_payload = json.loads(resume_result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((command_dir / "run_loop.json").exists())
        self.assertEqual(payload["handoff_status"], "continuation_ready")
        self.assertIs(payload["final_next_action_runnable"], True)
        self.assertIn("--json-out", payload["continuation_command"])
        self.assertIn(str(custom_report), payload["continuation_command"])
        self.assertIn("--markdown-out", payload["continuation_command"])
        self.assertIn(str(custom_markdown), payload["continuation_command"])
        self.assertIn("--resume-from-report", payload["resume_from_report_command"])
        self.assertIn(str(custom_report.resolve()), payload["resume_from_report_command"])
        self.assertEqual(
            payload["handoff_recommended_action"],
            "run_resume_from_report_command",
        )
        self.assertEqual(
            payload["handoff_recommended_command"],
            payload["resume_from_report_command"],
        )
        self.assertEqual(
            loop_report["resume_from_report_command"],
            payload["resume_from_report_command"],
        )
        self.assertEqual(resume_result.returncode, 0, resume_result.stderr)
        self.assertEqual(resume_payload["handoff_status"], "awaiting_next_command")
        self.assertEqual(resume_payload["steps"][0]["target"], "execution-next")
        self.assertIsNone(resume_payload["resume_from_report_command"])
        self.assertEqual(
            resume_payload["handoff_recommended_action"],
            "collect_next_command",
        )

    def test_cli_omits_report_resume_command_when_report_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--max-steps",
                "1",
                "--json",
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["handoff_status"], "continuation_ready")
        self.assertIs(payload["final_next_action_runnable"], True)
        self.assertIsNone(payload["resume_from_report_command"])
        self.assertEqual(
            payload["handoff_recommended_action"],
            "run_continuation_command",
        )
        self.assertEqual(
            payload["handoff_recommended_command"],
            payload["continuation_command"],
        )
        self.assertFalse((command_dir / "run_loop.json").exists())

    def test_cli_resume_from_report_rejects_non_continuation_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--max-steps",
                "3",
                "--write-loop-report",
                "--json",
            )
            resume_result = _run_loop(command_dir, "--resume-from-report", "--json")
            payload = json.loads(resume_result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resume_result.returncode, 1)
        self.assertEqual(payload["handoff_status"], "failed")
        self.assertEqual(payload["stop_reason"], "loop_resume_failed")
        self.assertIn("not continuation-ready", payload["error"])

    def test_cli_resume_from_report_honors_forwarded_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = _run_loop(
                command_dir,
                "--max-steps",
                "1",
                "--fail-on-max-steps-continuation",
                "--write-loop-report",
                "--json",
            )
            resume_result = _run_loop(
                command_dir,
                "--resume-from-report",
                "--dry-run",
                "--max-steps",
                "2",
                "--json",
            )
            payload = json.loads(resume_result.stdout)

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(resume_result.returncode, 0, resume_result.stderr)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["max_steps"], 2)
        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["executed_count"], 0)
        self.assertEqual(payload["stop_reason"], "dry_run")
        self.assertEqual(payload["handoff_status"], "dry_run")
        self.assertEqual(payload["steps"][0]["target"], "execution-next")
        self.assertEqual(payload["steps"][0]["target_kind"], "execution_next")
        self.assertFalse(payload["steps"][0]["executed"])
        self.assertIn("--dry-run", payload["continuation_command"])
        self.assertIn("--max-steps 2", payload["continuation_command"])
        self.assertIn("--resume-from-report", payload["resume_from_report_command"])
        self.assertEqual(
            payload["handoff_recommended_action"],
            "run_without_dry_run_when_ready",
        )
        self.assertIsNone(payload["handoff_recommended_command"])

    def test_cli_fails_when_final_action_requires_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            next_script = command_dir / "recommended_next.sh"
            next_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'mkdir -p "executed_follow_up"',
                        'printf "review-needed\\n" > runner.out',
                        'cat > "executed_follow_up/summary.json" <<\'JSON\'',
                        json.dumps(
                            {
                                "schema": "st.modelzoo.run.v1",
                                "status": "regressed",
                                "best_feature": "latent",
                                "best_config": {
                                    "best_feature": "latent",
                                    "feature_normalize": "blocks",
                                    "hybrid_latent_scale": 0.5,
                                    "mean_best_nll": 4.5,
                                    "mean_best_nll_delta_vs_raw": 0.2,
                                },
                                "follow_up_result": {
                                    "verdict": "regressed",
                                    "source_best_feature_retained": False,
                                },
                                "follow_up_gate": {
                                    "effective_verdict": "regressed",
                                    "failed": True,
                                },
                                "follow_up_guidance": {
                                    "action": "review_feature_swap_before_promotion",
                                    "unsafe_promotion": True,
                                },
                            }
                        ),
                        "JSON",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            next_script.chmod(next_script.stat().st_mode | 0o755)
            result = _run_loop(
                command_dir,
                "--max-steps",
                "3",
                "--write-loop-report",
                "--json",
            )
            payload = json.loads(result.stdout)
            loop_report = json.loads(
                (command_dir / "run_loop.json").read_text(encoding="utf-8")
            )
            history_summary = json.loads(
                (command_dir / "run_history_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["step_count"], 1)
        self.assertEqual(payload["executed_count"], 1)
        self.assertEqual(payload["success_count"], 1)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["stop_reason"], "history_next_action_stopped")
        self.assertEqual(payload["handoff_status"], "needs_review")
        self.assertEqual(
            payload["handoff_reason"],
            "follow-up gate requested stop",
        )
        self.assertEqual(
            payload["fail_on_final_actions"],
            ["review_before_continuing", "inspect_history"],
        )
        self.assertTrue(payload["final_action_failed"])
        self.assertEqual(payload["returncode"], 1)
        self.assertEqual(
            payload["error"],
            "final next action requested failure: review_before_continuing",
        )
        self.assertEqual(
            payload["final_next_action"]["action"],
            "review_before_continuing",
        )
        self.assertIs(payload["final_next_action"]["should_continue"], False)
        self.assertEqual(
            payload["handoff_recommended_action"],
            "review_before_continuing",
        )
        self.assertIsNone(payload["handoff_recommended_command"])
        self.assertEqual(
            payload["steps"][0]["execution_evidence_status"],
            "gate_failed",
        )
        self.assertEqual(
            history_summary["next_action"]["action"],
            "review_before_continuing",
        )
        self.assertEqual(loop_report["returncode"], 1)
        self.assertTrue(loop_report["final_action_failed"])

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
        self.assertEqual(payload["handoff_status"], "blocked")
        self.assertEqual(
            payload["handoff_reason"],
            "follow-up gate requested stop",
        )
        self.assertEqual(
            payload["error"],
            (
                "history next action does not allow continuation: "
                "follow-up gate requested stop"
            ),
        )
        self.assertEqual(payload["handoff_recommended_action"], "repair_blocker")
        self.assertIsNone(payload["handoff_recommended_command"])
        self.assertEqual(
            payload["steps"][0]["history_next_action"]["action"],
            "review_before_continuing",
        )
        self.assertFalse(payload["steps"][0]["executed"])
        self.assertEqual(len(history_events), 1)
        self.assertFalse((command_dir / "runner.out").exists())


if __name__ == "__main__":
    unittest.main()
