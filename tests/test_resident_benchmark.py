"""Dependency-free preflight gates for the resident PyTorch comparison."""
import importlib.util
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "tools" / "bench_resident_vs_torch.py"
spec = importlib.util.spec_from_file_location("resident_bench", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class DeviceIdentityTests(unittest.TestCase):
    def test_bad_kernel_lists_fail_before_import_or_output_creation(self):
        for kernels in (["scalar", "scalar"], ["unknown"]):
            args = [str(path), "--expected-adapter", "fixture", "--output", "unused.json", "--kernels", *kernels]
            with self.subTest(kernels=kernels), patch.object(bench.sys, "argv", args), \
                    patch.object(Path, "open") as output, contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                bench.main()
            self.assertEqual(raised.exception.code, 2)
            output.assert_not_called()

    def test_tile_parser_preserves_mnk_order_and_rejects_duplicate_labels(self):
        self.assertEqual(bench.parse_tiles("8,16,32;16,8,64"), [(8,16,32),(16,8,64)])
        for invalid in ("", "8,8", "8,8,16;8,8,16", "8,8,16;"):
            with self.subTest(spec=invalid), self.assertRaises(ValueError):
                bench.parse_tiles(invalid)

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
