"""Process admission checks must never turn other GPU work into a timing win."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_resident_rank_vs_torch as bench


class GpuAdmissionTest(unittest.TestCase):
    def test_excludes_only_own_pid(self):
        with patch.object(bench.os, "getpid", return_value=123), patch.object(
            bench.subprocess, "run", return_value=SimpleNamespace(stdout="123\n456\n456\n")
        ):
            self.assertEqual(bench.foreign_gpu_processes(), [456])
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                bench.require_uncontended_gpu()

    def test_empty_and_unknown_are_not_confused(self):
        with patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="")):
            bench.require_uncontended_gpu()
        with patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="N/A\n")):
            with self.assertRaises(ValueError):
                bench.require_uncontended_gpu()


if __name__ == "__main__":
    unittest.main()
