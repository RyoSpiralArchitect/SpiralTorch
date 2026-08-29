#!/usr/bin/env python3
"""Unit coverage for the Cargo workspace inventory helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "list_workspace_crates.py"
SPEC = importlib.util.spec_from_file_location("list_workspace_crates", SCRIPT)
assert SPEC and SPEC.loader
list_workspace_crates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = list_workspace_crates
SPEC.loader.exec_module(list_workspace_crates)


class WorkspaceInventoryTests(unittest.TestCase):
    def test_markdown_reports_count_and_escapes_cells(self) -> None:
        crate = list_workspace_crates.WorkspaceCrate(
            name="st-example",
            path="crates/st-example",
            version="0.1.0",
            default_member=True,
            description="line one | line two\ncontinued",
            tests=0,
            examples=0,
        )

        markdown = list_workspace_crates.render_markdown([crate])

        self.assertIn("**1 Cargo workspace package.**", markdown)
        self.assertIn("line one \\| line two continued", markdown)
        self.assertIn("| `st-example` | `crates/st-example` | yes |", markdown)

    def test_refresh_workspace_doc_replaces_only_generated_region(self) -> None:
        crate = list_workspace_crates.WorkspaceCrate(
            name="st-example",
            path="crates/st-example",
            version="0.1.0",
            default_member=False,
            description="Example crate",
            tests=0,
            examples=0,
        )
        contents = (
            "before\n"
            f"{list_workspace_crates._DOC_START}\nold\n"
            f"{list_workspace_crates._DOC_END}\n"
            "after\n"
        )

        refreshed = list_workspace_crates.refresh_workspace_doc(contents, [crate])

        self.assertTrue(refreshed.startswith("before\n"))
        self.assertTrue(refreshed.endswith("\nafter\n"))
        self.assertNotIn("\nold\n", refreshed)
        self.assertIn("| `st-example` | `crates/st-example` | no |", refreshed)

    def test_refresh_workspace_doc_requires_unique_markers(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one start and end marker"):
            list_workspace_crates.refresh_workspace_doc("no markers", [])

        reversed_markers = (
            f"{list_workspace_crates._DOC_END}\n"
            f"{list_workspace_crates._DOC_START}\n"
        )
        with self.assertRaisesRegex(ValueError, "end marker appears before start marker"):
            list_workspace_crates.refresh_workspace_doc(reversed_markers, [])

    def test_checked_in_inventory_matches_workspace_manifests(self) -> None:
        crates = list_workspace_crates.collect_workspace_crates(ROOT / "Cargo.toml")
        doc_path = ROOT / "docs" / "development" / "workspace_crates.md"
        contents = doc_path.read_text(encoding="utf-8")

        self.assertEqual(
            list_workspace_crates.refresh_workspace_doc(contents, crates),
            contents,
        )


if __name__ == "__main__":
    unittest.main()
