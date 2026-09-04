"""Standard-library tests; no installed native module or GPU required."""

import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools" / "bench_backend_vs_torch.py"
SPEC = importlib.util.spec_from_file_location("backend_bench", PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

RANK_PATH = Path(__file__).resolve().parents[1] / "tools" / "bench_rank_vs_torch.py"
RANK_SPEC = importlib.util.spec_from_file_location("rank_bench", RANK_PATH)
rank_bench = importlib.util.module_from_spec(RANK_SPEC)
RANK_SPEC.loader.exec_module(rank_bench)


class BackendBenchmarkTests(unittest.TestCase):
    def test_effective_backend_identifies_non_faer_indexing(self):
        for operation in ("matmul", "gather", "scatter"):
            for backend in ("cpu", "faer", "wgpu"):
                expected = "faer" if operation == "matmul" and backend in ("cpu", "faer") else (
                    "cpu" if backend == "faer" else backend
                )
                self.assertEqual(bench.effective_backend(operation, backend), expected)

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

    def test_rank_build_identity_must_match_clean_source(self):
        source = {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "tracked_dirty": False,
        }
        identity = {
            "schema": rank_bench.BUILD_IDENTITY_SCHEMA,
            "manifest": {
                "pkg": {"name": "st-core"},
                "git": {
                    "commit": source["commit"],
                    "tree": source["tree"],
                    "dirty": False,
                },
            },
        }
        binding = rank_bench.validate_source_binding(identity, source)
        self.assertTrue(binding["valid"])
        self.assertTrue(all(binding["checks"].values()))

    def test_rank_build_identity_rejects_stale_or_dirty_binary(self):
        source = {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "tracked_dirty": False,
        }
        for field, value in (
            ("commit", "c" * 40),
            ("tree", "d" * 40),
            ("dirty", True),
        ):
            with self.subTest(field=field):
                git = {
                    "commit": source["commit"],
                    "tree": source["tree"],
                    "dirty": False,
                }
                git[field] = value
                identity = {
                    "schema": rank_bench.BUILD_IDENTITY_SCHEMA,
                    "manifest": {"pkg": {"name": "st-core"}, "git": git},
                }
                self.assertFalse(
                    rank_bench.validate_source_binding(identity, source)["valid"]
                )


if __name__ == "__main__":
    unittest.main()
