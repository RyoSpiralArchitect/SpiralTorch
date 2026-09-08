"""Dependency-light benchmark admission checks; not GPU execution evidence."""

import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "nd_bench",
    Path(__file__).resolve().parents[1] / "tools/bench_nd_tensor_vs_torch.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

spec = importlib.util.spec_from_file_location(
    "nd_validation",
    Path(__file__).resolve().parents[1] / "tools/validate_nd_tensor_bench.py",
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class Admission(unittest.TestCase):
    def test_fixed_bounded_recipes(self):
        values = bench.recipes()
        self.assertEqual(len(values), 6)
        self.assertEqual({v["seed"] for v in values}, {17, 29, 43})
        self.assertEqual(
            {tuple(v["shape"]) for v in values}, {(8, 16, 64), (16, 32, 128)}
        )
        self.assertTrue(all(v["iterations"] == 20 for v in values))

    def test_fixture_binds_all_inputs_and_identity(self):
        config = bench.recipes()[0]
        shape, seed = config["shape"], config["seed"]
        value = dict(
            config,
            schema="spiraltorch.nd_bench.fixture.v1",
            identity={"frozen": True},
            gain=0.75,
            input=[
                ((i * 13 + seed) % 61) / 64.0 - 0.46875 for i in range(math.prod(shape))
            ],
            bias=[(i % 5) / 32.0 - 0.0625 for i in range(shape[2])],
        )
        bench.validate_fixture(value, config, {"frozen": True})
        for key, bad in (
            ("schema", "other"),
            ("shape", [1, 2, 3]),
            ("gain", 1.0),
            ("input", value["input"][:-1]),
            ("bias", [0.0] * shape[2]),
            ("identity", {}),
            ("seed", 29),
            ("iterations", 19),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_fixture(
                    dict(value, **{key: bad}), config, {"frozen": True}
                )

    def test_sample_rejects_bad_timings_guards_shapes_and_captures(self):
        valid = dict(
            shape=[2, 2, 4], elapsed_ms=1.0, values=[0.0] * 16, finite_checked=True
        )
        bench.validate_sample(valid, [2, 3, 4], True, True)
        for key, bad in (
            ("shape", [16]),
            ("elapsed_ms", True),
            ("elapsed_ms", 0),
            ("elapsed_ms", float("nan")),
            ("elapsed_ms", float("inf")),
            ("finite_checked", False),
            ("finite_checked", 1),
            ("values", [0.0] * 15),
            ("values", [float("nan")] * 16),
            ("values", [True] * 16),
            ("values", None),
        ):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                bench.validate_sample(dict(valid, **{key: bad}), [2, 3, 4], True, True)
        with self.assertRaises(ValueError):
            bench.validate_sample(valid, [2, 3, 4], False, True)
        uncaptured = copy.copy(valid)
        uncaptured["values"] = None
        bench.validate_sample(uncaptured, [2, 3, 4], False, True)


class Reaggregation(unittest.TestCase):
    def test_pointwise_reaggregation_checks_each_lane_and_execution(self):
        row = self.row()
        lanes = ["sequential", "batched", "fused", "torch"]
        for iteration, sample in enumerate(row["intervals"]):
            base = sample.pop("rust")
            for lane, elapsed in (
                ("sequential", 2.0),
                ("batched", 1.0),
                ("fused", 0.5),
            ):
                sample[lane] = dict(base, execution=lane, elapsed_ms=elapsed)
            offset = (iteration + 1) % 4
            sample["order"] = lanes[offset:] + lanes[:offset]
            if (iteration // 4) % 2:
                sample["order"].reverse()
        row["median_ms"] = dict(sequential=2.0, batched=1.0, fused=0.5, torch=1.0)
        row["torch_over_rust"] = dict(sequential=0.5, batched=1.0, fused=2.0)
        row["lane_errors"] = dict(sequential=0.0, batched=0.0, fused=0.0)
        result = validation.validate_case(row, bench.recipes()[0], {}, 0, True)
        self.assertEqual(result["rust_over_torch"]["fused"], 0.5)
        for mutate in (
            lambda r: r["intervals"][0]["order"].reverse(),
            lambda r: r["intervals"][0]["fused"].update(execution="batched"),
            lambda r: r["median_ms"].update(fused=0.1),
            lambda r: r["lane_errors"].update(batched=1.0),
            lambda r: r["intervals"][2]["fused"]["values"].__setitem__(0, 1.0),
        ):
            changed = copy.deepcopy(row)
            mutate(changed)
            with self.assertRaises(ValueError):
                validation.validate_case(changed, bench.recipes()[0], {}, 0, True)

    def row(self):
        config = bench.recipes()[0]
        shape, seed = config["shape"], config["seed"]
        fixture = dict(
            config,
            schema="spiraltorch.nd_bench.fixture.v1",
            identity={},
            gain=0.75,
            input=[
                ((i * 13 + seed) % 61) / 64.0 - 0.46875 for i in range(math.prod(shape))
            ],
            bias=[(i % 5) / 32.0 - 0.0625 for i in range(shape[2])],
        )
        samples = []
        out = [shape[1] - 1, shape[0], shape[2]]
        for iteration in range(10):
            samples.append(
                dict(
                    iteration=iteration,
                    warmup=iteration < 2,
                    order=(
                        ["rust", "torch"]
                        if (iteration + 1) % 2 == 0
                        else ["torch", "rust"]
                    ),
                )
            )
            for lane, elapsed in (("rust", 2.0), ("torch", 1.0)):
                samples[-1][lane] = dict(
                    shape=out,
                    elapsed_ms=elapsed,
                    finite_checked=True,
                    values=[0.0] * math.prod(out) if iteration == 2 else None,
                )
        return dict(
            fixture=fixture,
            intervals=samples,
            max_abs_error=0.0,
            median_ms=dict(rust=2.0, torch=1.0),
            torch_over_rust=0.5,
        )

    def test_recomputes_all_intervals_without_selection(self):
        row = self.row()
        result = validation.validate_case(row, bench.recipes()[0], {}, 0)
        self.assertEqual(result["rust_over_torch"], 2.0)
        self.assertEqual(result["retained_per_lane"], 8)
        for key, value in (
            ("iteration", 2),
            ("warmup", False),
            ("warmup", 1),
            ("order", ["rust", "torch"]),
        ):
            bad = copy.deepcopy(row)
            bad["intervals"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validation.validate_case(bad, bench.recipes()[0], {}, 0)
        row["intervals"].pop()
        with self.assertRaises(ValueError):
            validation.validate_case(row, bench.recipes()[0], {}, 0)

    def test_rejects_changed_medians_and_captures(self):
        for key, value in (
            ("median_ms", dict(rust=1.0, torch=1.0)),
            ("torch_over_rust", 2.0),
            ("max_abs_error", 1.0),
        ):
            row = self.row()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validation.validate_case(row, bench.recipes()[0], {}, 0)
        row = self.row()
        row["intervals"][2]["rust"]["values"][0] = 1.0
        with self.assertRaises(ValueError):
            validation.validate_case(row, bench.recipes()[0], {}, 0)


class WorkerDeadlines(unittest.TestCase):
    def test_deadline_retains_error_report_and_never_restarts(self):
        fake = mock.Mock()
        fake.request.side_effect = TimeoutError("response deadline exceeded")
        torch = types.SimpleNamespace(
            __version__="fixture", set_num_threads=lambda _: None
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "worker"
            binary.write_bytes(b"fixture")
            output = Path(directory) / "report.json"
            with mock.patch.dict(
                sys.modules, {"numpy": types.ModuleType("numpy"), "torch": torch}
            ), mock.patch.object(
                bench.audit, "source_identity", return_value={"tracked_dirty": False}
            ), mock.patch.object(
                bench.audit, "file_identity", return_value={}
            ), mock.patch.object(
                bench.audit, "read_native_build_identity", return_value={}
            ), mock.patch.object(
                bench.audit, "validate_source_binding", return_value={"valid": True}
            ), mock.patch.object(
                bench.audit, "git_bytes", return_value=b""
            ), mock.patch.object(
                bench, "Worker", return_value=fake
            ) as constructor:
                with self.assertRaises(TimeoutError):
                    bench.run(binary, output, "cpu")
            report = bench.json.loads(output.read_text())
            self.assertEqual(report["status"], "error")
            self.assertIn("deadline exceeded", report["error"])
            self.assertEqual(report["cases"], [])
            constructor.assert_called_once()
            fake.abort.assert_called_once()
            fake.close_streams.assert_called_once()

    def worker(self, program, **kwargs):
        worker = bench.Worker(
            [sys.executable, "-I", "-c", program],
            subprocess.DEVNULL,
            response_timeout=1.0,
            exit_timeout=0.2,
            **kwargs
        )
        self.addCleanup(worker.close_streams)
        self.addCleanup(worker.abort)
        return worker

    def test_complete_response_and_clean_exit(self):
        worker = self.worker(
            "import json,sys; x=json.loads(sys.stdin.readline()); print(json.dumps(x),flush=True)"
        )
        self.assertEqual(worker.request({"value": 3}), {"value": 3})
        worker.finish()

    def test_silent_and_partial_responses_have_deadlines_without_restart(self):
        for prefix in ("", "sys.stdout.write('{'); sys.stdout.flush();"):
            worker = self.worker("import sys,time; " + prefix + "time.sleep(30)")
            pid = worker.process.pid
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                worker.request({})
            worker.abort()
            self.assertLess(time.monotonic() - start, 3.0)
            self.assertEqual(worker.process.pid, pid)
            self.assertIsNotNone(worker.process.poll())

    def test_eof_oversize_and_exit_hang_are_bounded(self):
        worker = self.worker("import sys; sys.stdin.readline()")
        with self.assertRaises(RuntimeError):
            worker.request({})
        worker = self.worker(
            "import sys; sys.stdin.readline(); print('x'*100,flush=True)", limit=16
        )
        with self.assertRaises(ValueError):
            worker.request({})
        worker = self.worker(
            "import sys,time; sys.stdin.readline(); print('{}',flush=True); sys.stdin.read(); time.sleep(30)"
        )
        self.assertEqual(worker.request({}), {})
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.finish()

    @unittest.skipIf(os.name == "nt", "POSIX ignored-SIGTERM fixture")
    def test_cleanup_kills_worker_that_ignores_termination(self):
        worker = self.worker(
            "import signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); sys.stdin.readline(); print('{}',flush=True); time.sleep(30)"
        )
        self.assertEqual(worker.request({}), {})
        start = time.monotonic()
        worker.abort()
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertIsNotNone(worker.process.poll())


if __name__ == "__main__":
    unittest.main()
