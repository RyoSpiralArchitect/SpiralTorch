"""Process admission checks must never turn other GPU work into a timing win."""
import sys
import math
import os
import random
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_resident_rank_vs_torch as bench
import torch_rank_reference as reference


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

    def test_active_lane_suite_covers_reduction_widths_and_k(self):
        requests = list(bench.requests_for(bench.audit.load_bench_module(), "active-lanes"))
        self.assertEqual(len(requests), 243)
        self.assertEqual({(r["cols"] + 31) // 32 for r in requests},
                         {1, 2, 3, 5, 17, 33, 65, 129, 257})
        self.assertEqual({r["k"] for r in requests}, {1, 7, 31, 63, 65})
        self.assertEqual(len({(r["seed"], r["kind"], r["cols"], r["k"]) for r in requests}),
                         len(requests))
        for r in requests:
            self.assertLessEqual(r["k"], r["cols"])
            self.assertEqual(len(r["input"]), r["rows"] * r["cols"])
            self.assertTrue(all(math.isfinite(v) for v in r["input"]))
            if r["seed"] == 43:
                self.assertEqual(bench.cuda_rank_operation(r), "stable_sort")
                self.assertTrue(all(v != 0 or math.copysign(1, v) == 1 for v in r["input"]))

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


class CanonicalReferenceTest(unittest.TestCase):
    def request(self, values, k, kind="topk"):
        return dict(kind=kind, rows=1, cols=len(values), k=k, input=values, seed=17)

    def test_ties_outside_retained_set_do_not_force_a_sort(self):
        request = self.request([-5., -5., 7., 9.], 2)
        admitted = reference.contract(request)
        self.assertEqual(admitted["indices"], [3, 2])
        self.assertIn("topk", admitted["operations"])
        self.assertEqual(bench.cuda_rank_operation(request), "stable_sort")

    def test_internal_ties_allow_repair_but_cutoff_ties_do_not(self):
        for kind, values in [("topk", [0., 8., 8., 9.]), ("bottomk", [9., 1., 1., 0.])]:
            admitted = reference.contract(self.request(values, 3, kind))
            self.assertEqual(admitted["indices"], [3, 1, 2])
            self.assertEqual(admitted["operations"], ["topk_index_repair", "stable_sort", "packed_topk"])
            admitted = reference.contract(self.request(values, 2, kind))
            self.assertEqual(admitted["operations"], ["stable_sort", "packed_topk"])

    def test_mixed_signed_zeros_require_total_order(self):
        values = [0., -0., 0., -0., -1., 1.]
        for kind, expected in [("topk", [5, 0, 2]), ("bottomk", [4, 1, 3]), ("midk", [1, 3, 0])]:
            admitted = reference.contract(self.request(values, 3, kind))
            self.assertEqual(admitted["indices"], expected)
            self.assertEqual(admitted["operations"], ["packed_topk"])

    def test_admission_uses_f32_not_python_double_ties(self):
        admitted = reference.contract(self.request([1., 1. + 2**-25, 0.], 1))
        self.assertEqual(admitted["input"], [1., 1., 0.])
        self.assertEqual(admitted["indices"], [0])
        self.assertNotIn("topk", admitted["operations"])

    def test_stable_sort_admission_checks_the_returned_window(self):
        for kind, values, k in [("topk", [3., -0., 0.], 1),
                                ("bottomk", [-3., 0., -0.], 1),
                                ("midk", [0., -0., -4., -3., -2., -1., 2.], 1),
                                ("topk", [0., -0., -1.], 2),
                                ("bottomk", [-0., 0., 1.], 2),
                                ("midk", [-2., -0., 0., 1., 2.], 3)]:
            with self.subTest(kind=kind, values=values, k=k):
                admitted = reference.contract(self.request(values, k, kind))
                self.assertIn("stable_sort", admitted["operations"])
        for kind, values in [("topk", [-0., 0., -1.]), ("bottomk", [0., -0., 1.])]:
            admitted = reference.contract(self.request(values, 2, kind))
            self.assertEqual(admitted["operations"], ["packed_topk"])
        admitted = reference.contract(self.request([0., -0., -2., -1., 2.], 1, "midk"))
        self.assertEqual(admitted["operations"], ["packed_topk"])

    def test_repair_admission_accepts_already_canonical_signed_zeros(self):
        for kind, values in [("topk", [0., -0., -1.]), ("bottomk", [-0., 0., 1.])]:
            admitted = reference.contract(self.request(values, 2, kind))
            self.assertIn("topk_index_repair", admitted["operations"])
            self.assertNotIn("topk", admitted["operations"])

    def test_invalid_inputs_fail_closed(self):
        valid = self.request([1., 2.], 1)
        for change in [dict(kind="max"), dict(rows=True), dict(cols=0), dict(k=0), dict(k=3),
                       dict(cols=2**32), dict(input=[1.]), dict(input=[float("nan"), 0.]),
                       dict(input=[float("inf"), 0.]), dict(input=[1e40, 0.])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                reference.contract(dict(valid, **change))

    def test_packed_integer_order_matches_independent_numeric_order(self):
        rng = random.Random(29)
        values = [-0., 0., -1., 1.]
        for _ in range(10000):
            value = struct.unpack("<f", struct.pack("<I", rng.getrandbits(32)))[0]
            if math.isfinite(value):
                values.append(value)
        # Python's numeric order plus a signed-zero discriminator is independent
        # of the integer transform used by the eager Torch control.
        expected = sorted(values, key=lambda value: (value, math.copysign(1, value)))
        actual = sorted(values, key=reference.total_key)
        self.assertEqual([struct.pack("<f", value) for value in actual],
                         [struct.pack("<f", value) for value in expected])


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_TORCH_RANK_TESTS"), "opt-in live Torch controls")
class LiveCanonicalReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.device = os.environ.get("SPIRALTORCH_TORCH_RANK_DEVICE", "cpu")
        if cls.device not in ("cpu", "cuda"):
            raise ValueError("live rank tests require explicit cpu or cuda")
        if cls.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested live CUDA test cannot fall back to CPU")
        torch.set_num_threads(1)

    def check(self, control, admitted):
        torch = self.torch
        expected = torch.tensor(admitted["values"], dtype=torch.float32).view(torch.int32)
        self.assertTrue(torch.equal(control.values.cpu().reshape(-1).view(torch.int32), expected))
        self.assertEqual(control.indices.cpu().reshape(-1).tolist(), admitted["indices"])

    def test_all_admitted_controls_across_boundaries_and_extremes(self):
        torch = self.torch
        rng = random.Random(43)
        extreme = [0., -0., 1., -1., 3.4028234663852886e38, -3.4028234663852886e38,
                   2**-149, -(2**-149)]
        with torch.inference_mode():
            for cols in [1, 2, 3, 7, 31, 32, 33, 64, 65, 127, 128, 129, 257, 1025]:
                for fixture in ("distinct", "ties", "extreme"):
                    data = ([rng.uniform(-1, 1) for _ in range(2 * cols)] if fixture == "distinct"
                            else [rng.choice(extreme if fixture == "extreme" else [0., 1., -1., 2.])
                                  for _ in range(2 * cols)])
                    for kind in ("topk", "midk", "bottomk"):
                        for k in sorted({1, min(7, cols), cols}):
                            request = dict(kind=kind, rows=2, cols=cols, k=k, seed=43, input=data)
                            admitted = reference.contract(request)
                            device = torch.tensor(data, dtype=torch.float32, device=self.device).reshape(2, cols)
                            for operation in admitted["operations"]:
                                with self.subTest(cols=cols, fixture=fixture, kind=kind, k=k, operation=operation):
                                    control = reference.RankControl(torch, device, admitted, operation)
                                    pointers = (control.values.data_ptr(), control.indices.data_ptr())
                                    for _ in range(2):
                                        control.run()
                                        self.check(control, admitted)
                                        self.assertEqual(pointers, (control.values.data_ptr(), control.indices.data_ptr()))

    def test_packed_control_rebuilds_keys_after_in_place_upload(self):
        torch = self.torch
        with torch.inference_mode():
            for kind in ("topk", "midk", "bottomk"):
                request = dict(kind=kind, rows=1, cols=5, k=3, seed=17, input=[-0., 0., 1., 1., -2.])
                admitted = reference.contract(request)
                device = torch.tensor(admitted["input"], device=self.device).reshape(1, 5)
                control = reference.RankControl(torch, device, admitted, "packed_topk")
                for values in (request["input"], [2., -1., -1., 0., -0.], [1., 1., 1., 1., 1.]):
                    device.copy_(torch.tensor(values, device=self.device).reshape(1, 5))
                    control.run()
                    self.check(control, reference.contract(dict(request, input=values)))

    def test_zero_window_controls_preserve_exact_values_and_indices(self):
        torch = self.torch
        with torch.inference_mode():
            for kind, data, k in [("topk", [3., -0., 0.], 1),
                                  ("bottomk", [-3., 0., -0.], 1),
                                  ("topk", [0., -0., -1.], 2),
                                  ("bottomk", [-0., 0., 1.], 2),
                                  ("midk", [-2., -0., 0., 1., 2.], 3)]:
                request = dict(kind=kind, rows=1, cols=len(data), k=k, input=data)
                admitted = reference.contract(request)
                self.assertIn("stable_sort", admitted["operations"])
                device = torch.tensor(data, dtype=torch.float32, device=self.device).reshape(1, len(data))
                for name in admitted["operations"]:
                    control = reference.RankControl(torch, device, admitted, name)
                    control.run()
                    self.check(control, admitted)

    def test_control_admission_and_tensor_layout_are_enforced(self):
        torch = self.torch
        admitted = reference.contract(dict(kind="topk", rows=2, cols=2, k=1, input=[1., 1., 1., 1.]))
        device = torch.ones((2, 2), device=self.device)
        with self.assertRaises(ValueError):
            reference.RankControl(torch, device, admitted, "topk")
        for bad in (device.T, device.double(), device.clone().requires_grad_(), device[:1]):
            with self.assertRaises(ValueError):
                reference.RankControl(torch, bad, admitted, "packed_topk")

    def test_timing_boundary_requires_cuda_and_retains_all_controls(self):
        torch = self.torch
        request = dict(kind="topk", rows=1, cols=5, k=3, seed=17, input=[0., 8., 8., 9., 1.])
        with torch.inference_mode():
            device = torch.tensor(request["input"], device=self.device).reshape(1, 5)
            summarize = bench.audit.load_bench_module().summarize
            if self.device == "cpu":
                with self.assertRaisesRegex(ValueError, "requires CUDA"):
                    reference.measure(request, device, torch, summarize)
                return
            report = reference.measure(request, device, torch, summarize)
            self.assertEqual(report["operations"], ["topk_index_repair", "stable_sort", "packed_topk"])
            self.assertEqual(len(report["samples"]), 12)
            self.assertEqual(report["repetitions"], 16)
            for name in report["operations"]:
                samples = report["timings"][name]["samples_ms"]
                self.assertEqual(samples, [pair["per_op_ms"][name] for pair in report["samples"]])
                self.assertTrue(all(math.isfinite(value) and value > 0 for value in samples))
            self.assertEqual(report["best_fixed"], min(report["operations"], key=lambda name:
                sum(report["timings"][name]["samples_ms"]) / 12))
            device[0, 0] = 42
            with self.assertRaisesRegex(ValueError, "fixture differs"):
                reference.measure(request, device, torch, summarize)


if __name__ == "__main__":
    unittest.main()
