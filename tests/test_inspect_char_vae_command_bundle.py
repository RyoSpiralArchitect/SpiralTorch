#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "inspect_char_vae_command_bundle.py"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_script(path: Path, *, executable: bool = True) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o755)


def _write_bundle(
    root: Path,
    *,
    missing_comparison_json: bool = False,
    non_executable_follow_up: bool = False,
) -> Path:
    command_dir = root / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    chain = root / "chain" / "chain.json"
    _write_json(chain, {"schema": "st.llm_char_vae_context.chain.v1"})

    comparison_json = command_dir / "comparison.json"
    comparison_markdown = command_dir / "comparison.md"
    if not missing_comparison_json:
        _write_json(comparison_json, {"schema": "comparison"})
    comparison_markdown.write_text("# comparison\n", encoding="utf-8")
    (command_dir / "README.md").write_text("# commands\n", encoding="utf-8")

    next_script = command_dir / "recommended_next.sh"
    follow_up_script = command_dir / "recommended_follow_up.sh"
    _write_script(next_script)
    _write_script(follow_up_script, executable=not non_executable_follow_up)

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
                "review_path": None,
                "written_count": 2,
                "comparison_json_path": str(comparison_json),
                "comparison_markdown_path": str(comparison_markdown),
                "readme_path": str(command_dir / "README.md"),
            },
        },
    )
    return command_dir


class InspectCharVaeCommandBundleTests(unittest.TestCase):
    def test_cli_reports_ready_bundle(self) -> None:
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
            markdown_result = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["bundle_ready"])
        self.assertTrue(payload["strict_ready"])
        self.assertEqual(payload["action"], "continue_from_accepted")
        self.assertEqual(payload["next_kind"], "follow_up")
        self.assertEqual(payload["chain_source_count"], 1)
        self.assertEqual(payload["missing_required"], [])
        self.assertEqual(payload["missing_optional"], [])
        self.assertEqual(markdown_result.returncode, 0, markdown_result.stderr)
        self.assertIn("Char VAE Command Bundle Inspection", markdown_result.stdout)
        self.assertIn("bundle_ready: yes", markdown_result.stdout)

    def test_cli_writes_default_inspection_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp))
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
                    "--write-report",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            json_report = command_dir / "inspection.json"
            markdown_report = command_dir / "inspection.md"
            json_report_exists = json_report.exists()
            markdown_report_exists = markdown_report.exists()
            report_payload = json.loads(json_report.read_text(encoding="utf-8"))
            markdown = markdown_report.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json_report_exists)
        self.assertTrue(markdown_report_exists)
        self.assertTrue(report_payload["bundle_ready"])
        self.assertTrue(report_payload["strict_ready"])
        self.assertIn("Char VAE Command Bundle Inspection", markdown)

    def test_cli_writes_explicit_inspection_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command_dir = _write_bundle(root)
            json_out = root / "reports" / "bundle.json"
            markdown_out = root / "reports" / "bundle.md"
            result = subprocess.run(
                [
                    "python3",
                    "-P",
                    str(SCRIPT),
                    str(command_dir),
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
            json_out_exists = json_out.exists()
            markdown_out_exists = markdown_out.exists()
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            markdown = markdown_out.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json_out_exists)
        self.assertTrue(markdown_out_exists)
        self.assertEqual(payload["action"], "continue_from_accepted")
        self.assertIn("strict_ready: yes", markdown)

    def test_cli_fails_when_required_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp), missing_comparison_json=True)
            result = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir), "--json"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["bundle_ready"])
        self.assertIn("comparison_json", payload["missing_required"])

    def test_strict_fails_when_declared_script_is_not_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_dir = _write_bundle(Path(tmp), non_executable_follow_up=True)
            relaxed = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir), "--json"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            strict = subprocess.run(
                ["python3", "-P", str(SCRIPT), str(command_dir), "--json", "--strict"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(relaxed.returncode, 0, relaxed.stderr)
        payload = json.loads(relaxed.stdout)
        self.assertTrue(payload["bundle_ready"])
        self.assertFalse(payload["strict_ready"])
        self.assertIn("follow_up_script", payload["missing_optional"])
        self.assertEqual(strict.returncode, 1)


if __name__ == "__main__":
    unittest.main()
