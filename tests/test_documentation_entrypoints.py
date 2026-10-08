#!/usr/bin/env python3
"""Keep the short entry points connected to tested, reachable reference docs."""

import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DocumentationEntrypointTests(unittest.TestCase):
    def test_entrypoints_stay_short_and_link_to_docs(self):
        for relative in ("README.md", "bindings/st-py/README.md"):
            with self.subTest(path=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertLessEqual(len(text.splitlines()), 160)
                self.assertIn("docs/", text)
                self.assertIn("ops/release.md", text)
                self.assertNotIn("--token-source", text)
                self.assertNotIn("<!-- STATS:START -->", text)

    def test_migrated_links_resolve_and_package_links_are_absolute(self):
        files = [ROOT / "README.md", ROOT / "bindings/st-py/README.md", ROOT / "docs/README.md"]
        files.extend(sorted((ROOT / "docs/reference").glob("*.md")))
        files.extend(sorted((ROOT / "docs/python").glob("*.md")))
        files.append(ROOT / "docs/repository-stats.md")
        prefix = "https://github.com/RyoSpiralArchitect/SpiralTorch/blob/main/"
        for path in files:
            in_fence = False
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                links = re.findall(r"\]\(([^\s)]+)", line)
                links.extend(re.findall(r'(?:src|href)="([^"]+)"', line))
                for link in links:
                    parsed = urlsplit(link)
                    if not parsed.path:
                        continue
                    if link.startswith(prefix):
                        target = ROOT / unquote(urlsplit(link[len(prefix):]).path)
                    elif parsed.scheme or parsed.netloc:
                        continue
                    else:
                        self.assertNotEqual(path, ROOT / "bindings/st-py/README.md", link)
                        target = path.parent / unquote(parsed.path)
                    with self.subTest(path=path.relative_to(ROOT), link=link):
                        self.assertTrue(target.exists(), str(target))

    def test_reference_python_fences_remain_in_strict_ci(self):
        runner = load_tool("run_readme_python_blocks")
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        step = ci.split("- name: Run README and reference Python blocks\n", 1)[1]
        step = step.split("\n      - name:", 1)[0]
        self.assertNotIn("--allow-stub-skips", step)
        count = 0
        for path in sorted((ROOT / "docs/reference").glob("*.md")):
            blocks = runner._parse_python_blocks(path.read_text(encoding="utf-8"))
            if blocks:
                self.assertIn(f"--readme {path.relative_to(ROOT).as_posix()}", step)
            count += len(blocks)
        self.assertGreaterEqual(count, 37)
        self.assertIn("--readme README.md", step)
        self.assertIn("--readme bindings/st-py/README.md", step)

    def test_generated_stats_update_only_the_stats_page(self):
        stats = load_tool("gen_repo_stats")
        self.assertEqual(stats.STATS_DOC, ROOT / "docs/repository-stats.md")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.md"
            original = "# Stats\n<!-- STATS:START -->\nold\n<!-- STATS:END -->\ntail\n"
            path.write_text(original)
            with mock.patch.object(stats, "STATS_DOC", path):
                stats.update_stats_section("new stats")
            self.assertEqual(path.read_text(), original.replace("old", "new stats"))
        workflow = (ROOT / ".github/workflows/repo-stats.yml").read_text()
        self.assertIn("tools/gen_repo_stats.py", workflow)

    def test_release_cadence_preserves_publication_boundaries(self):
        runbook = (ROOT / "docs/ops/release.md").read_text(encoding="utf-8")
        cadence = runbook.split("## Release Cadence\n", 1)[1].split("\n## ", 1)[0]
        for contract in ("parent-first", "Cargo.lock", "--no-clipboard", "dry-run", "hashes"):
            self.assertIn(contract, cadence)
        self.assertIn("not a timed job", cadence)
        self.assertIn("never move an existing release tag", cadence.lower())


if __name__ == "__main__":
    unittest.main()
