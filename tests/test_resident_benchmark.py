"""Dependency-free preflight gates for the resident PyTorch comparison."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "tools" / "bench_resident_vs_torch.py"
spec = importlib.util.spec_from_file_location("resident_bench", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class DeviceIdentityTests(unittest.TestCase):
    def test_single_matching_device(self):
        bench.validate_cuda_identity(1, "NVIDIA GeForce RTX 5090", "RTX 5090")

    def test_multiple_or_missing_devices_are_not_same_device_evidence(self):
        for count in (0, 2):
            with self.subTest(count=count), self.assertRaises(RuntimeError):
                bench.validate_cuda_identity(count, "NVIDIA RTX 5090", "RTX 5090")

    def test_mismatch_and_empty_expectation_are_rejected(self):
        for expected in ("", " ", "RTX 4090"):
            with self.subTest(expected=expected), self.assertRaises(RuntimeError):
                bench.validate_cuda_identity(1, "NVIDIA RTX 5090", expected)


if __name__ == "__main__":
    unittest.main()
