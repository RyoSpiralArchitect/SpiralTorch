"""Offline negative controls and call-order tests, not GPU qualification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_long_resume as long
from test_byte_corpus_optimizer import state as fabricated_state


def fabricated_report():
    raw = long.request_bytes()
    request = long.protocol.decode_json(raw)
    controls = long.optimizer.recipe(request)
    cases = []
    for case in request["cases"]:
        final = [[long.protocol.f32(value if long.protocol.parameter_frozen(case, p["name"])
                                    else value + .01) for value in p["values"]]
                 for p in case["parameters"]]
        cases.append(dict(name=case["name"], seed=case["seed"], geometry=case["geometry"],
                          geometry_update=case["geometry_update"],
                          training=[dict(revision=i, ce=5.0 - i / 128) for i in range(1, 129)],
                          validation=[dict(revision=i, batch_losses=[4.0]) for i in (0, 64, 128)],
                          final_parameters=final,
                          optimizer_state=fabricated_state(case, controls, 128)))
    return dict(schema="spiraltorch.byte_corpus.result.v10", request_sha256=long.protocol.digest(raw),
                design_schema=request["design_schema"], optimizer=controls,
                gradient_clip=long.clipping.recipe(request), cases=cases)


class FakeCheckpoint:
    def __init__(self, cursor, changed=False):
        self.completed_updates = cursor
        self.changed = changed

    def to_json(self):
        return json.dumps({"fake_only": True, "cursor": self.completed_updates, "changed": self.changed})


class FakeStudy:
    def __init__(self, defect=None):
        self.defect, self.calls = defect, []
        self.total_updates = 128
        self.request_sha256 = long.protocol.digest(long.request_bytes())
        self.report = long.protocol.encoded(fabricated_report()).decode()

    def validate_segment(self, cursor):
        self.calls.append(("validate", cursor))

    def run_wgpu(self):
        self.calls.append(("run",))
        return self.report if self.defect != "empty_report" else "{}"

    def advance_wgpu(self, cursor=None, *, resume=None):
        self.calls.append(("advance", cursor, None if resume is None else resume.completed_updates))
        if self.defect == "execution_error" and resume is not None:
            raise RuntimeError("synthetic execution failure")
        cursor = 128 if cursor is None else cursor
        report = self.report if cursor == 128 else json.dumps({"fake_partial": cursor})
        if self.defect == "report" and resume is not None:
            report += " "
        return SimpleNamespace(report_json=report,
                               checkpoint=FakeCheckpoint(cursor, self.defect == "checkpoint" and resume is not None))

    def checkpoint_from_json(self, raw):
        cursor = json.loads(raw)["cursor"]
        self.calls.append(("load", cursor))
        return FakeCheckpoint(cursor)

    def client(self):
        def parse(raw):
            if raw != long.request_bytes():
                raise ValueError("request changed")
            return self
        return SimpleNamespace(nn=SimpleNamespace(ByteCorpusStudy=SimpleNamespace(from_json=parse)))


class LongResumeOffline(unittest.TestCase):
    def test_recipe_is_fixed_and_each_batch_is_owned(self):
        raw = long.request_bytes()
        self.assertEqual(raw, long.request_bytes())
        self.assertEqual(long.protocol.digest(raw), long.REQUEST_SHA256)
        request = long.protocol.decode_json(raw)
        self.assertEqual(long.protocol.request_version(request), 10)
        self.assertEqual(tuple(c["name"] for c in request["cases"]), long.CASE_NAMES)
        self.assertEqual(len(request["train_batches"]), 128)
        self.assertEqual(request["checkpoint_every"], 64)
        self.assertEqual(long.SPLITS, (1, 64))
        self.assertEqual(request["optimizer"]["kind"], "adamw_zero_rate_freeze.v1")
        self.assertEqual(request["gradient_clip"], {"kind": "active_global_l2.v1", "max_norm": .01})
        for index, batch in enumerate(request["train_batches"]):
            self.assertEqual(batch, [[0, 4 * (index % 2)]])
        request["train_batches"][0][0][1] = 99
        self.assertEqual(request["train_batches"][2], [[0, 0]])
        self.assertEqual(long.request_bytes(), raw)

    def test_recipe_drift_and_foreign_request_reject(self):
        original = long.learning_design

        def changed_design():
            request = original()
            request["rate"] = .002
            return request

        with patch.object(long, "learning_design", changed_design):
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                long.request_bytes()
        with self.assertRaisesRegex(ValueError, "request identity"):
            long.validate_completed_report(long.protocol.encoded(fabricated_report()),
                                           long.request_bytes() + b" ")

    def test_fabricated_complete_report_is_structurally_valid_only(self):
        report = fabricated_report()
        self.assertEqual(long.validate_completed_report(long.protocol.encoded(report), long.request_bytes()), report)

    def test_dropped_history_missing_cases_metadata_and_clocks_reject(self):
        mutations = {
            "empty": lambda r: r.clear(),
            "drop_case": lambda r: r["cases"].pop(),
            "reorder": lambda r: r["cases"].reverse(),
            "identity": lambda r: r.update(request_sha256="0" * 64),
            "metadata": lambda r: r["cases"][0].update(seed=True),
            "history": lambda r: r["cases"][0]["training"].pop(),
            "revision": lambda r: r["cases"][0]["training"][0].update(revision=True),
            "validation": lambda r: r["cases"][0]["validation"].pop(),
            "shape": lambda r: r["cases"][0]["final_parameters"][0].pop(),
            "clock": lambda r: r["cases"][0]["optimizer_state"]["slots"][0].update(steps=127),
            "nonfinite": lambda r: r["cases"][0]["training"][0].update(ce=float("nan")),
        }
        for name, mutate in mutations.items():
            report = fabricated_report()
            mutate(report)
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                long.validate_completed_report(json.dumps(report), long.request_bytes())

    def test_noop_backbone_and_geometry_cannot_pass_exact_equality(self):
        request = long.protocol.decode_json(long.request_bytes())
        for case_index, geometry_only in ((0, False), (1, True)):
            report = fabricated_report()
            for index, descriptor in enumerate(request["cases"][case_index]["parameters"]):
                if not geometry_only or descriptor["name"].startswith("geometry."):
                    report["cases"][case_index]["final_parameters"][index] = copy.deepcopy(descriptor["values"])
            with self.subTest(case=case_index), self.assertRaisesRegex(ValueError, "active parameter"):
                long.validate_completed_report(long.protocol.encoded(report), long.request_bytes())

    def test_frozen_geometry_and_moments_are_not_silently_ignored(self):
        request = long.protocol.decode_json(long.request_bytes())
        slot = next(i for i, p in enumerate(request["cases"][2]["parameters"])
                    if p["name"].startswith("geometry."))
        for mutate_moment in (False, True):
            report = fabricated_report()
            if mutate_moment:
                report["cases"][2]["optimizer_state"]["slots"][slot]["first"][0] = 1e-7
            else:
                report["cases"][2]["final_parameters"][slot][0] += 1e-7
            with self.subTest(moment=mutate_moment), self.assertRaises(ValueError):
                long.validate_completed_report(long.protocol.encoded(report), long.request_bytes())

    def test_one_changed_weight_or_moment_and_whitespace_fail_exact_comparison(self):
        expected = fabricated_report()
        for kind in ("weight", "moment", "whitespace"):
            changed = copy.deepcopy(expected)
            if kind == "weight":
                changed["cases"][0]["final_parameters"][0][0] += 1e-12
            if kind == "moment":
                changed["cases"][0]["optimizer_state"]["slots"][0]["first"][0] += 1e-12
            actual = long.protocol.encoded(changed) + (b" " if kind == "whitespace" else b"")
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                long.require_exact(long.protocol.encoded(expected), actual, kind)

    def test_call_order_outputs_and_no_release_promotion(self):
        study = FakeStudy()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "long"
            result = long.exercise(study.client(), output)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["splits"], [1, 64])
            self.assertFalse(result["release_ready"])
            self.assertEqual(json.loads((output / "result.json").read_bytes()), result)
            self.assertEqual(study.calls, [
                ("validate", 1), ("validate", 64), ("validate", 128), ("run",),
                ("advance", None, None), ("advance", 1, None), ("load", 1),
                ("advance", None, 1), ("advance", 64, None), ("load", 64), ("advance", None, 64)])
            for name, record in result["files"].items():
                raw = (output / name).read_bytes()
                self.assertEqual(long.protocol.digest(raw), record["sha256"])
                self.assertEqual(len(raw), record["bytes"])
            with self.assertRaises(FileExistsError):
                long.exercise(study.client(), output)

    def test_first_failure_retained_and_no_second_split_or_retry(self):
        for defect in ("report", "checkpoint", "execution_error", "empty_report"):
            study = FakeStudy(defect)
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "long"
                with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                    long.exercise(study.client(), output)
                result = json.loads((output / "result.json").read_bytes())
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["release_ready"])
                self.assertNotIn(("advance", 64, None), study.calls)
                self.assertEqual(study.calls.count(("run",)), 1)
                self.assertTrue((output / "ordinary-report.json").is_file())
                if defect != "empty_report":
                    self.assertTrue((output / "split-1/first-checkpoint.json").is_file())
                if defect in ("report", "checkpoint"):
                    self.assertTrue((output / "split-1/resumed-checkpoint.json").is_file())


if __name__ == "__main__":
    unittest.main()
