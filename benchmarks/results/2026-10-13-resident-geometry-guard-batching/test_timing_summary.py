"""Synthetic counterexamples only; these are not measured timings."""
import copy
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import bench_native
import summarize_performance as s


class TimingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(Path(__file__).with_name("performance-protocol.json").read_bytes())
        self.report = b'{"synthetic":true}\n'
        self.p["report_sha256"] = {e:s.sha(self.report) for e in ("native","browser")}
        self.raw = json.dumps(self.p).encode()
        self.records = {}
        for engine,timer in (("native","fresh_process_perf_counter_ns_including_output_write"),
                             ("browser","awaited_rust_export_including_gpu_runtime_and_rust_json")):
            docs = []
            for page in ([None] if engine=="native" else [1,2,3]):
                d = dict(schema="spiraltorch.geometry_guard_batching.timing.v1",engine=engine,timer=timer,
                    completed=True,protocol_sha256=s.sha(self.raw),fixture_sha256=self.p["fixture_sha256"],samples=[])
                if page is not None:
                    d["block"] = page
                for block in ([page] if page is not None else [1,2,3]):
                    for stage,order in (("warmup",["A","B"]),("measure",["A","B","B","A"])):
                        for index,arm in enumerate(order):
                            row = dict(block=block,stage=stage,index=index,arm=arm,
                                status="completed",
                                seconds=(100. if stage=="warmup" else 1. if arm=="A" else .9),
                                report_sha256=s.sha(self.report))
                            if engine=="native":
                                row.update(exit_code=0,report_file=f"block-{block}-{stage}-{index}-{arm}.json")
                            else:
                                row["report_json"] = self.report.decode()
                            d["samples"].append(row)
                docs.append(d)
            self.records[engine] = docs

    def result(self, records=None, reader=None):
        return s.summarize(self.raw,records or self.records,reader or (lambda _:self.report))

    def test_all_samples_retained_but_warmups_do_not_change_ratios(self):
        result = self.result()
        self.assertTrue(result["combined_screen_passed"])
        self.assertFalse(result["adopted"])
        self.assertFalse(result["original_browser_model_qualification_passed"])
        for r in result["runtimes"].values():
            self.assertEqual(len(r["samples"]),18)
            self.assertAlmostEqual(r["geometric_mean_ratio"],.9)

    def test_rejects_regression_even_when_geometric_mean_passes(self):
        for row in self.records["native"][0]["samples"]:
            if row["stage"]=="measure" and row["arm"]=="B":
                row["seconds"] = 1.01 if row["block"]==1 else .8
        result = self.result()
        self.assertLess(result["runtimes"]["native"]["geometric_mean_ratio"],.95)
        self.assertFalse(result["combined_screen_passed"])

    def test_rejects_small_gain_below_declared_minimum(self):
        for d in self.records["browser"]:
            for row in d["samples"]:
                if row["arm"]=="B": row["seconds"] = .96
        self.assertFalse(self.result()["combined_screen_passed"])

    def test_counterexamples_reject(self):
        mutations = [
            lambda r:r["browser"].pop(),
            lambda r:r["browser"][1].update(block=1),
            lambda r:r["native"][0].update(completed=False),
            lambda r:r["native"][0].update(protocol_sha256="foreign"),
            lambda r:r["browser"][0].update(persistence_error="storage lost"),
            lambda r:r["native"][0]["samples"].pop(),
            lambda r:r["native"][0]["samples"][2].update(arm="B"),
            lambda r:r["native"][0]["samples"][0].update(exit_code=1),
            lambda r:r["native"][0]["samples"][0].update(report_file="../../foreign"),
            lambda r:r["browser"][0]["samples"][0].update(report_json="changed"),
            lambda r:r["browser"][0]["samples"][0].update(seconds=float("nan")),
            lambda r:r["browser"][0]["samples"][0].update(seconds=0),
            lambda r:r["browser"][0]["samples"][0].update(seconds=True),
            lambda r:r["browser"][0]["samples"][0].update(status="started_or_interrupted"),
        ]
        for mutate in mutations:
            r = copy.deepcopy(self.records)
            mutate(r)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.result(r)
        with self.assertRaises(ValueError):
            self.result(reader=lambda _:b"changed")


class NativeInvocationTests(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory(prefix="geometry-timing-unit-") as directory:
            root = Path(directory)
            a, b, fixture, protocol, output = [root/n for n in ("A","B","fixture","protocol.json","out")]
            a.write_bytes(b"synthetic A, never executed")
            b.write_bytes(b"synthetic B, never executed")
            fixture.write_bytes(b"synthetic fixture")
            good = b'{"synthetic":true}\n'
            p = json.loads(Path(__file__).with_name("performance-protocol.json").read_bytes())
            p["fixture_sha256"] = bench_native.digest(fixture)
            p["native_binary_sha256"] = {"A":bench_native.digest(a),"B":bench_native.digest(b)}
            p["report_sha256"]["native"] = s.sha(good)
            protocol.write_text(json.dumps(p))

            def child(*args, **kwargs):
                kwargs["stdout"].write(good if mode=="success" else b"partial or malformed report")
                if mode=="interrupt":
                    raise KeyboardInterrupt("synthetic interruption")
                return SimpleNamespace(returncode=0)

            clock = itertools.count(0,1_000_000)
            with mock.patch.object(sys,"argv",["bench",str(protocol),str(a),str(b),str(fixture),str(output)]), \
                 mock.patch.object(bench_native.subprocess,"run",side_effect=child) as run, \
                 mock.patch.object(bench_native.time,"perf_counter_ns",side_effect=lambda:next(clock)), \
                 contextlib.redirect_stdout(io.StringIO()):
                if mode=="success":
                    bench_native.main()
                else:
                    with self.assertRaises(KeyboardInterrupt if mode=="interrupt" else ValueError):
                        bench_native.main()
                calls = run.call_count
            record = json.loads((output/"timing.json").read_bytes())
            events = [json.loads(line) for line in (output/"samples.jsonl").read_text().splitlines()]
            if mode=="success":
                self.assertTrue(record["completed"])
                self.assertEqual(calls,18)
                self.assertEqual(len(events),36)
                self.assertTrue(all(r["status"]=="completed" for r in record["samples"]))
            else:
                self.assertFalse(record["completed"])
                self.assertIn("error",record)
                self.assertEqual(calls,1)
                self.assertEqual([e["event"] for e in events],["started","finished"])
                row = record["samples"][0]
                self.assertEqual(row["status"],"failed")
                self.assertEqual(row["seconds"],.001)
                self.assertEqual((output/row["report_file"]).read_bytes(),b"partial or malformed report")
                self.assertEqual(row["report_sha256"],s.sha(b"partial or malformed report"))

    def test_complete_schedule_without_launching_binaries(self):
        self.exercise("success")

    def test_keyboard_interrupt_keeps_attempt_elapsed_and_partial_output(self):
        self.exercise("interrupt")

    def test_changed_output_is_kept_and_not_retried(self):
        self.exercise("bad_output")


if __name__ == "__main__":
    unittest.main()
