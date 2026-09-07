import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_resident_nn_vs_torch as bench


class ResidentNnBenchmarkTest(unittest.TestCase):
    def test_adapter_and_admission_fail_closed(self):
        adapter = dict(name="Apple M4", backend="Metal", device_type="IntegratedGpu")
        bench.match_adapter(adapter, "mps", "Apple M4")
        for changed in (dict(adapter, name="other"), dict(adapter, backend="Vulkan"),
                        dict(adapter, device_type="Cpu")):
            with self.assertRaises(RuntimeError):
                bench.match_adapter(changed, "mps", "Apple M4")
        with patch.object(bench.platform, "system", return_value="Linux"):
            with self.assertRaises(RuntimeError):
                bench.admit_device("mps")

    def case(self):
        request = next(bench.requests())
        values, parameters = bench.fixture(request)
        result = dict(request, status="passed", source_operations=3, gpu_stages=2,
                      input=values, parameters=parameters, reference=[0.] * len(values),
                      legacy_mean_ms=2., resident_mean_ms=1.,
                      samples=[dict(order=[0, 1] if (i + 2 + request["seed"]) % 2 == 0 else [1, 0],
                                    legacy_ms=2., resident_ms=1.) for i in range(12)])
        return request, result

    def test_fixed_matrix_and_independent_parameters(self):
        requests = list(bench.requests())
        self.assertEqual(len(requests), 9)
        self.assertEqual({r["seed"] for r in requests}, {17, 29, 43})
        request, result = self.case()
        bench.validate_native(result, request)
        values, parameters = bench.fixture(request)
        self.assertEqual(len(values), 42)
        self.assertEqual([p["gelu"] for p in parameters], [True, False])
        self.assertEqual((values, parameters), bench.fixture(request))

    def test_rejects_wrong_identity_or_samples(self):
        request, original = self.case()
        for change in (dict(status="error"), dict(shape=[6, 7]), dict(gpu_stages=1),
                       dict(source_operations=4), dict(samples=[]), dict(resident_mean_ms=3.)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench.validate_native(dict(original, **change), request)
        for field in ("legacy_ms", "resident_ms"):
            result = copy.deepcopy(original)
            result["samples"][0][field] = float("nan")
            with self.assertRaises(ValueError):
                bench.validate_native(result, request)

    def test_rejects_parameter_or_order_drift(self):
        request, original = self.case()
        result = copy.deepcopy(original)
        result["parameters"][0]["weight"][0] += 1
        with self.assertRaises(ValueError):
            bench.validate_native(result, request)
        result = copy.deepcopy(original)
        result["samples"][0]["order"].reverse()
        with self.assertRaises(ValueError):
            bench.validate_native(result, request)


if __name__ == "__main__":
    unittest.main()
