"""Process admission checks must never turn other GPU work into a timing win."""
import sys
import math
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


class RankSuiteTest(unittest.TestCase):
    def test_default_suite_preserves_original_shapes(self):
        requests = list(bench.requests_for(bench.audit.load_bench_module(), "standard"))
        self.assertEqual(len(requests), 54)
        self.assertEqual({(r["cols"], r["k"], r["tile"]) for r in requests},
                         {(cols, k, tile) for cols, k in [(256, 8), (2048, 16)]
                          for tile in [128, 256, 512]})

    def test_boundary_suite_has_controls_fragmentation_and_ties(self):
        requests = list(bench.requests_for(bench.audit.load_bench_module(), "midk-boundary"))
        self.assertEqual(len(requests), 63)
        self.assertEqual({r["kind"] for r in requests}, {"topk", "midk", "bottomk"})
        self.assertEqual({(r["cols"] + r["tile"] - 1) // r["tile"] for r in requests},
                         {5, 32, 33, 129, 257})
        for r in requests:
            self.assertEqual(len(r["input"]), r["rows"] * r["cols"])
            if r["seed"] == 43:
                self.assertLess(len(set(r["input"])), r["cols"])
                self.assertTrue(all(v != 0 or math.copysign(1, v) == 1 for v in r["input"]))

    def test_unknown_suite_fails(self):
        with self.assertRaises(ValueError):
            list(bench.requests_for(None, "unknown"))

    def test_cuda_controls_preserve_source_index_order(self):
        for kind in ["topk", "bottomk", "midk"]:
            for values, tied in [([1., 2., 3., 4.], False),
                                ([1., 1., 3., 4.], True),
                                ([1., 2., 3., 3.], True)]:
                request = dict(kind=kind, rows=2, cols=2, input=values)
                self.assertEqual(bench.cuda_rank_operation(request),
                                 "stable_sort" if tied or kind == "midk" else "topk")
        # Equal values in different rows are not ties within a rank operation.
        self.assertEqual(bench.cuda_rank_operation(dict(kind="topk", rows=2, cols=2,
                                                       input=[1., 2., 1., 2.])), "topk")

    def test_tied_cuda_selection_cannot_pass_on_values_alone(self):
        bench.require_canonical_indices([[0, 1], [0, 1]], [[0, 1], [0, 1]])
        for actual in [[[0, 2], [0, 1]], [[1, 0], [0, 1]], [[0], [0, 1]]]:
            with self.assertRaisesRegex(RuntimeError, "canonical stable"):
                bench.require_canonical_indices(actual, [[0, 1], [0, 1]])

    def test_native_boundary_and_effective_tile_are_checked(self):
        request = dict(kind="midk", rows=2, cols=256, k=8, seed=17, tile=512)
        result = dict(request, tile=256, status="passed", mode="resident_only",
                      samples_ms={"resident_dispatch_fence_per_op": [1.]})
        bench.validate_native_result(result, request, True)
        for change in ({"tile": 512}, {"mode": "comparison"},
                       {"samples_ms": {"host_api": [1.]}}, {"seed": 29}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                bench.validate_native_result(dict(result, **change), request, True)

    def test_empty_and_unknown_are_not_confused(self):
        with patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="")):
            bench.require_uncontended_gpu()
        with patch.object(bench.subprocess, "run", return_value=SimpleNamespace(stdout="N/A\n")):
            with self.assertRaises(ValueError):
                bench.require_uncontended_gpu()


if __name__ == "__main__":
    unittest.main()
