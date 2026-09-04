"""Standard-library tests; no installed native module or GPU required."""

import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools" / "bench_backend_vs_torch.py"
SPEC = importlib.util.spec_from_file_location("backend_bench", PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BackendBenchmarkTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(bench.parse_sizes("32,32;7,64x64,65"), [(32, 32, 32), (7, 64, 65)])
        for invalid in ("0,0", "2,3", "2,3x4,5", "", "-1,-1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bench.parse_sizes(invalid)

    def test_fixture_is_replayable_float32(self):
        first = bench.fixture(17, 43)
        self.assertEqual(first, bench.fixture(17, 43))
        self.assertNotEqual(first, bench.fixture(17, 44))
        self.assertEqual(first[0], bench.array("f", first[0]).tolist())
        self.assertEqual(len(first[1]), 64)

    def test_paired_samples_include_each_implementation(self):
        calls = []
        functions = {name: lambda n=name: calls.append(n) for name in ("st", "torch")}
        result = bench.paired_timings(functions, 2, 7, lambda: None, 17)
        for name in functions:
            self.assertEqual(calls.count(name), 9)
            self.assertEqual(result[name]["n"], 7)
        for i in range(4, len(calls), 2):
            self.assertEqual(set(calls[i:i + 2]), set(functions))

    def test_summary_retains_raw_samples(self):
        values = list(range(1, 21))
        report = bench.summarize(values)
        self.assertEqual(report["median_ms"], 10.5)
        self.assertEqual(report["p95_ms"], 19)
        self.assertEqual(report["samples_ms"], values)


if __name__ == "__main__":
    unittest.main()
