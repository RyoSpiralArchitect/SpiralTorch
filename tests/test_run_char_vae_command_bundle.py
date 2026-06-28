#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_char_vae_command_bundle.py"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_script(path: Path, body: str, *, executable: bool = True) -> None:
    path.write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", body, ""]),
        encoding="utf-8",
    )
    if executable:
        path.chmod(path.stat().st_mode | 0o755)


def _chain_path(command_dir: Path) -> str:
    return str(command_dir.parent / "chain" / "chain.json")


def _assert_recommendation_context(
    testcase: unittest.TestCase,
    payload: dict[str, object],
    command_dir: Path,
    *,
    target_kind: str = "follow_up",
) -> None:
    context = payload["recommendation_context"]
    testcase.assertIsInstance(context, dict)
    assert isinstance(context, dict)
    testcase.assertEqual(
        context["schema"],
        "st.llm_char_vae_context.command_bundle_run_recommendation_context.v1",
    )
    testcase.assertEqual(context["action"], "continue_from_accepted")
    testcase.assertEqual(
        context["reason"],
        "accepted champion remains the safest continuation",
    )
    testcase.assertEqual(context["target_kind"], target_kind)
    testcase.assertEqual(context["next_kind"], "follow_up")
    testcase.assertIs(context["accepted_matches_best"], True)
    testcase.assertIs(context["best_requires_review"], False)
    testcase.assertEqual(context["accepted_vs_best_nll_gap"], 0.0)
    testcase.assertEqual(context["chain_sources"], [_chain_path(command_dir)])
    testcase.assertEqual(context["chain_count"], 1)
    testcase.assertEqual(context["attempted_follow_ups"], 1)
    testcase.assertEqual(
        context["follow_up_from_summary_path"],
        _chain_path(command_dir),
    )
    testcase.assertIsNone(context["review_summary_path"])
    testcase.assertEqual(context["champion_source"], _chain_path(command_dir))
    testcase.assertEqual(context["fallback_source"], _chain_path(command_dir))
    testcase.assertEqual(context["follow_up_command_source"], "next_follow_up_command")
    testcase.assertIsNone(context["review_command_source"])
    champion = context["champion"]
    fallback = context["fallback"]
    testcase.assertIsInstance(champion, dict)
    testcase.assertIsInstance(fallback, dict)
    assert isinstance(champion, dict)
    assert isinstance(fallback, dict)
    testcase.assertEqual(champion["config"], "latent@normalize=blocks,scale=0.5")
    testcase.assertEqual(champion["mean_best_nll"], 4.1)
    testcase.assertEqual(champion["summary_path"], _chain_path(command_dir))
    testcase.assertEqual(fallback["config"], "raw@normalize=blocks,scale=1.0")
    testcase.assertEqual(fallback["summary_path"], _chain_path(command_dir))


def _write_bundle(
    root: Path,
    *,
    non_executable_follow_up: bool = False,
    include_review: bool = False,
) -> Path:
    command_dir = root / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    chain = root / "chain" / "chain.json"
    _write_json(chain, {"schema": "st.llm_char_vae_context.chain.v1"})

    comparison_json = command_dir / "comparison.json"
    comparison_markdown = command_dir / "comparison.md"
    _write_json(comparison_json, {"schema": "comparison"})
    comparison_markdown.write_text("# comparison\n", encoding="utf-8")
    (command_dir / "README.md").write_text("# commands\n", encoding="utf-8")

    next_script = command_dir / "recommended_next.sh"
    follow_up_script = command_dir / "recommended_follow_up.sh"
    review_script = command_dir / "recommended_review.sh"
    _write_script(
        next_script,
        'printf "next cwd=%s\\n" "$(pwd)" | tee runner.out',
    )
    _write_script(
        follow_up_script,
        'printf "follow cwd=%s\\n" "$(pwd)" | tee runner.out',
        executable=not non_executable_follow_up,
    )
    if include_review:
        _write_script(
            review_script,
            'printf "review cwd=%s\\n" "$(pwd)" | tee runner.out',
        )

    _write_json(
        command_dir / "recommendation.json",
        {
            "schema": "st.llm_char_vae_context.chain_command_manifest.v1",
            "comparison": {
                "schema": "st.llm_char_vae_context.chain_comparison.v1",
                "chain_sources": [str(chain)],
            },
            "aggregate": {
                "chain_count": 1,
                "attempted_follow_ups": 1,
            },
            "selection": {
                "accepted_matches_best": True,
                "best_requires_review": False,
                "accepted_vs_best_nll_gap": 0.0,
            },
            "recommendation": {
                "action": "continue_from_accepted",
                "reason": "accepted champion remains the safest continuation",
                "follow_up_from_summary_path": str(chain),
                "review_summary_path": None,
                "champion_source": str(chain),
                "champion": {
                    "config": "latent@normalize=blocks,scale=0.5",
                    "mean_best_nll": 4.1,
                    "step": 1,
                    "summary_path": str(chain),
                },
                "fallback_source": str(chain),
                "fallback": {
                    "config": "raw@normalize=blocks,scale=1.0",
                    "mean_best_nll": 4.2,
                    "step": 0,
                    "summary_path": str(chain),
                },
                "follow_up_command": {
                    "command_source": "next_follow_up_command",
                    "source_summary_path": str(chain),
                },
                "review_command": None,
            },
            "command_scripts": {
                "schema": "st.llm_char_vae_context.chain_command_scripts.v1",
                "directory": str(command_dir),
                "next_path": str(next_script),
                "next_kind": "follow_up",
                "follow_up_path": str(follow_up_script),
                "review_path": str(review_script) if include_review else None,
                "written_count": 3 if include_review else 2,
                "comparison_json_path": str(comparison_json),
                "comparison_markdown_path": str(comparison_markdown),
                "inspection_json_path": str(command_dir / "inspection.json"),
                "inspection_markdown_path": str(command_dir / "inspection.md"),
                "run_json_path": str(command_dir / "run.json"),
                "run_markdown_path": str(command_dir / "run.md"),
                "run_history_jsonl_path": str(command_dir / "run_history.jsonl"),
                "readme_path": str(command_dir / "README.md"),
            },
        },
    )
    return command_dir


class RunCharVaeCommandBundleTests(unittest.TestCase):
    def test_cli_dry_run_reports_next_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir), "--dry-run", "--json"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            payload = json.loads(result.stdout)
            runner_out_exists = (command_dir / "runner.out").exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["target"], "next")
        self.assertEqual(payload["target_kind"], "follow_up")
        self.assertEqual(payload["script_key"], "next_path")
        self.assertEqual(payload["target_script_key"], "follow_up_path")
        self.assertEqual(
            payload["target_script_path"],
            str(command_dir / "recommended_follow_up.sh"),
        )
        _assert_recommendation_context(self, payload, command_dir)
        self.assertTrue(payload["strict_ready"])
        self.assertFalse(payload["executed"])
        self.assertIsNotNone(payload["started_at"])
        self.assertIsNotNone(payload["finished_at"])
        self.assertIsNone(payload["run_history_jsonl_path"])
        self.assertGreaterEqual(payload["duration_seconds"], 0.0)
        self.assertEqual(payload["execution_cwd"], str(command_dir.resolve()))
        self.assertEqual(
            payload["command_argv"],
            ["bash", str(command_dir / "recommended_next.sh")],
        )
        self.assertFalse(runner_out_exists)

    def test_cli_runs_next_after_strict_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir), "--json"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            payload = json.loads(result.stdout)
            runner_out = (command_dir / "runner.out").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["returncode"], 0)
        self.assertEqual(payload["target"], "next")
        self.assertEqual(payload["target_kind"], "follow_up")
        self.assertEqual(payload["target_script_key"], "follow_up_path")
        self.assertEqual(
            payload["target_script_path"],
            str(command_dir / "recommended_follow_up.sh"),
        )
        _assert_recommendation_context(self, payload, command_dir)
        self.assertIsNone(payload["run_history_jsonl_path"])
        self.assertTrue(payload["executed"])
        self.assertGreaterEqual(payload["duration_seconds"], 0.0)
        self.assertIn("next cwd=", payload["stdout"])
        self.assertIn(str(command_dir), runner_out)

    def test_cli_blocks_when_strict_inspection_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp), non_executable_follow_up=True)
            strict = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir), "--json"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            strict_payload = json.loads(strict.stdout)
            relaxed = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--no-strict",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            relaxed_payload = json.loads(relaxed.stdout)

        self.assertEqual(strict.returncode, 1)
        self.assertFalse(strict_payload["strict_ready"])
        self.assertFalse(strict_payload["executed"])
        self.assertEqual(strict_payload["target_kind"], "follow_up")
        self.assertEqual(strict_payload["target_script_key"], "follow_up_path")
        _assert_recommendation_context(self, strict_payload, command_dir)
        self.assertIsNone(strict_payload["run_history_jsonl_path"])
        self.assertGreaterEqual(strict_payload["duration_seconds"], 0.0)
        self.assertIn("follow_up_script", strict_payload["missing_optional"])
        self.assertEqual(
            strict_payload["error"],
            "command bundle did not pass the requested inspection gate",
        )
        self.assertEqual(relaxed.returncode, 0, relaxed.stderr)
        self.assertEqual(relaxed_payload["returncode"], 0)

    def test_cli_reports_missing_target_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--target",
                    "review",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["target_kind"], "review")
        self.assertEqual(payload["script_key"], "review_path")
        self.assertEqual(payload["target_script_key"], "review_path")
        self.assertIsNone(payload["target_script_path"])
        _assert_recommendation_context(
            self,
            payload,
            command_dir,
            target_kind="review",
        )
        self.assertIsNone(payload["run_history_jsonl_path"])
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["error"], "manifest does not declare review_path")

    def test_cli_runs_review_target_with_resolved_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp), include_review=True)
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--target",
                    "review",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["target"], "review")
        self.assertEqual(payload["target_kind"], "review")
        self.assertEqual(payload["script_key"], "review_path")
        self.assertEqual(payload["target_script_key"], "review_path")
        self.assertEqual(
            payload["target_script_path"],
            str(command_dir / "recommended_review.sh"),
        )
        _assert_recommendation_context(
            self,
            payload,
            command_dir,
            target_kind="review",
        )
        self.assertIsNone(payload["run_history_jsonl_path"])
        self.assertEqual(
            payload["command_argv"],
            ["bash", str(command_dir / "recommended_review.sh")],
        )
        self.assertTrue(payload["executed"])
        self.assertIn("review cwd=", payload["stdout"])

    def test_cli_writes_inspection_report_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--write-inspection-report",
                    "--write-run-report",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            payload = json.loads(result.stdout)
            markdown_exists = (command_dir / "inspection.md").exists()
            run_markdown_exists = (command_dir / "run.md").exists()
            inspection = json.loads(
                (command_dir / "inspection.json").read_text(encoding="utf-8")
            )
            run_report = json.loads(
                (command_dir / "run.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(markdown_exists)
        self.assertTrue(run_markdown_exists)
        self.assertTrue(inspection["strict_ready"])
        self.assertEqual(
            payload["inspection"]["inspection_json_path"],
            str(command_dir / "inspection.json"),
        )
        self.assertEqual(payload["run_json_path"], str(command_dir / "run.json"))
        self.assertEqual(payload["target_kind"], "follow_up")
        _assert_recommendation_context(self, payload, command_dir)
        self.assertTrue(payload["executed"])
        self.assertIsNotNone(payload["started_at"])
        self.assertEqual(run_report["returncode"], 0)
        self.assertEqual(run_report["target_kind"], "follow_up")
        _assert_recommendation_context(self, run_report, command_dir)
        self.assertEqual(run_report["target_script_key"], "follow_up_path")
        self.assertEqual(
            run_report["target_script_path"],
            str(command_dir / "recommended_follow_up.sh"),
        )
        self.assertTrue(run_report["executed"])
        self.assertGreaterEqual(run_report["duration_seconds"], 0.0)
        self.assertEqual(run_report["execution_cwd"], str(command_dir.resolve()))
        self.assertEqual(run_report["run_markdown_path"], str(command_dir / "run.md"))
        self.assertIsNone(run_report["run_history_jsonl_path"])

    def test_cli_appends_compact_run_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            dry_run = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--dry-run",
                    "--append-run-history",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            executed = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--append-run-history",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            dry_payload = json.loads(dry_run.stdout)
            executed_payload = json.loads(executed.stdout)
            history_path = command_dir / "run_history.jsonl"
            events = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertEqual(executed.returncode, 0, executed.stderr)
        self.assertEqual(dry_payload["run_history_jsonl_path"], str(history_path))
        self.assertEqual(executed_payload["run_history_jsonl_path"], str(history_path))
        self.assertEqual(len(events), 2)
        self.assertEqual(
            events[0]["schema"],
            "st.llm_char_vae_context.command_bundle_run_history_event.v1",
        )
        self.assertTrue(events[0]["dry_run"])
        self.assertFalse(events[0]["executed"])
        self.assertFalse(events[1]["dry_run"])
        self.assertTrue(events[1]["executed"])
        self.assertEqual(events[1]["returncode"], 0)
        self.assertEqual(events[1]["target_kind"], "follow_up")
        self.assertEqual(events[1]["target_script_key"], "follow_up_path")
        self.assertEqual(
            events[1]["target_script_path"],
            str(command_dir / "recommended_follow_up.sh"),
        )
        self.assertEqual(
            events[1]["recommendation_context"]["action"],
            "continue_from_accepted",
        )
        self.assertEqual(
            events[1]["recommendation_context"]["champion"]["config"],
            "latent@normalize=blocks,scale=0.5",
        )
        self.assertNotIn("stdout", events[1])
        self.assertNotIn("stderr", events[1])
        self.assertNotIn("inspection", events[1])

    def test_cli_writes_explicit_run_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command_dir = _write_bundle(root)
            json_out = root / "reports" / "bundle-run.json"
            markdown_out = root / "reports" / "bundle-run.md"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--json",
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
            payload = json.loads(result.stdout)
            report = json.loads(json_out.read_text(encoding="utf-8"))
            markdown = markdown_out.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["run_json_path"], str(json_out))
        self.assertEqual(payload["target_kind"], "follow_up")
        _assert_recommendation_context(self, payload, command_dir)
        self.assertTrue(payload["executed"])
        self.assertEqual(report["returncode"], 0)
        self.assertEqual(report["target_kind"], "follow_up")
        _assert_recommendation_context(self, report, command_dir)
        self.assertEqual(
            report["command_argv"],
            ["bash", str(command_dir / "recommended_next.sh")],
        )
        self.assertGreaterEqual(report["duration_seconds"], 0.0)
        self.assertEqual(report["run_markdown_path"], str(markdown_out))
        self.assertIsNone(report["run_history_jsonl_path"])
        self.assertIn("Char VAE Command Bundle Runner", markdown)
        self.assertIn(f"run_json_path: {json_out}", markdown)
        self.assertIn("run_history_jsonl_path: -", markdown)
        self.assertIn("target_kind: follow_up", markdown)
        self.assertIn("target_script_key: follow_up_path", markdown)
        self.assertIn("Recommendation Context", markdown)
        self.assertIn("recommendation_action: continue_from_accepted", markdown)
        self.assertIn(
            "recommendation_reason: accepted champion remains the safest continuation",
            markdown,
        )
        self.assertIn("champion_config: latent@normalize=blocks,scale=0.5", markdown)
        self.assertIn("fallback_config: raw@normalize=blocks,scale=1.0", markdown)
        self.assertIn("follow_up_command_source: next_follow_up_command", markdown)
        self.assertIn("executed: yes", markdown)
        self.assertIn("duration_seconds:", markdown)


if __name__ == "__main__":
    unittest.main()
