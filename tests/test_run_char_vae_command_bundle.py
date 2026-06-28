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
            "recommendation": {
                "action": "continue_from_accepted",
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
        self.assertEqual(payload["script_key"], "next_path")
        self.assertTrue(payload["strict_ready"])
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
        self.assertEqual(payload["script_key"], "review_path")
        self.assertEqual(payload["error"], "manifest does not declare review_path")

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
            inspection = json.loads(
                (command_dir / "inspection.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(markdown_exists)
        self.assertTrue(inspection["strict_ready"])
        self.assertEqual(
            payload["inspection"]["inspection_json_path"],
            str(command_dir / "inspection.json"),
        )


if __name__ == "__main__":
    unittest.main()
