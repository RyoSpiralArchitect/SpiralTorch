"""Fail-closed result validation and deterministic, paired input controls."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("study", Path(__file__).with_name("byte_corpus_study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture():
    config = dict(batch=1, steps=2, width=2, hidden=2, heads=1, blocks=[False],
                  geometry_cols=2, curvature=-.75)
    plain, geometry = study.parameters(config, 11)
    cases = [dict(name="plain", seed=11, geometry=False, parameters=plain),
             dict(name="geometry", seed=11, geometry=True, parameters=geometry)]
    request = dict(schema=study.REQUEST, config=config, cases=cases,
                   train_batches=[[(0, 0)], [(0, 2)]], validation_batches=[[(0, 0)]],
                   train_documents=[[1, 2, 3, 4, 5]], validation_documents=[[8, 9, 10]],
                   checkpoint_every=2, rate=.05)
    raw = study.encoded(request)
    rows = []
    for c in cases:
        rows.append(dict(name=c["name"], seed=11, geometry=c["geometry"],
                         parameter_tensors=len(c["parameters"]),
                         parameter_scalars=sum(len(p["values"]) for p in c["parameters"]),
                         training=[dict(revision=1, ce=5.), dict(revision=2, ce=4.)],
                         validation=[dict(revision=r, batch_losses=[loss], mean_ce=loss,
                                          bits_per_byte=loss / math.log(2), target_bytes=2)
                                     for r, loss in [(0, 5.), (2, 4.)]],
                         final_parameters=[[x + .01 for x in p["values"]] for p in c["parameters"]]))
    ref = dict(schema=study.RESULT, request_sha256=study.digest(raw), cases=rows,
               engine="independent_pytorch")
    actual = copy.deepcopy(ref)
    actual["engine"] = "spiraltorch"
    actual["adapter"] = "test-only GPU adapter"
    return raw, ref, actual


class Controls(unittest.TestCase):
    def test_serialized_numeric_controls_fail_before_preparation(self):
        for seeds in [[], [-1], [1 << 53], [1, 1], list(range(9))]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                study.validate_numbers(seeds, .05)
        for rate in [0., -.1, 1e-50, 1e40, float("nan"), float("inf")]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                study.validate_numbers([11, 23, 37], rate)
        self.assertEqual(study.validate_numbers([0, (1 << 53) - 1], .05), study.f32(.05))

    def test_preparation_is_deterministic_and_backbone_identical(self):
        c = dict(width=4, hidden=6, steps=4, heads=2, blocks=[False, True], geometry_cols=4)
        a, b = study.parameters(c, 761)
        self.assertEqual((a, b), study.parameters(c, 761))
        self.assertEqual(a, [p for p in b if not p["name"].startswith("geometry.")])
        self.assertNotEqual(a, study.parameters(c, 762)[0])

    def test_window_targets_do_not_overlap_or_cross_documents(self):
        docs = [bytes(range(8)), bytes(range(20, 26))]
        choices = study.windows(docs, 3)
        self.assertEqual(choices, [(0, 0), (0, 3), (1, 0)])
        targets = [(d, i) for d, s in choices for i in range(s + 1, s + 4)]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertTrue(all(i < len(docs[d]) for d, i in targets))

    def test_positive_control(self):
        result = study.compare(*fixture())
        self.assertTrue(result["numerical_checks_passed"])
        self.assertEqual(result["paired_deltas"], [dict(seed=11, geometry_minus_ordinary_bpb=0.)])

    def test_float32_conversion_is_not_a_geometry_learning_delta(self):
        raw, ref, actual = fixture()
        request = json.loads(raw)
        for i, parameter in enumerate(request["cases"][1]["parameters"]):
            if parameter["name"].startswith("geometry."):
                parameter["values"] = [.3] * len(parameter["values"])
                for report in (ref, actual):
                    report["cases"][1]["final_parameters"][i] = [study.f32(.3)] * len(parameter["values"])
        raw = study.encoded(request)
        for report in (ref, actual):
            report["request_sha256"] = study.digest(raw)
        with self.assertRaisesRegex(ValueError, "geometry learning delta"):
            study.compare(raw, ref, actual)

    def test_noncanonical_initial_values_with_real_updates_pass(self):
        raw, ref, actual = fixture()
        request = json.loads(raw)
        for i, parameter in enumerate(request["cases"][1]["parameters"]):
            if parameter["name"].startswith("geometry."):
                parameter["values"] = [.3] * len(parameter["values"])
                for report in (ref, actual):
                    report["cases"][1]["final_parameters"][i] = [study.f32(.3 + 1e-4)] * len(parameter["values"])
        raw = study.encoded(request)
        for report in (ref, actual):
            report["request_sha256"] = study.digest(raw)
        self.assertTrue(study.compare(raw, ref, actual)["numerical_checks_passed"])

    def test_missing_or_misidentified_results_fail(self):
        edits = [
            lambda a: a["cases"].pop(),
            lambda a: a.update(request_sha256="wrong"),
            lambda a: a["cases"][0].update(seed=12),
            lambda a: a["cases"][0]["training"].pop(),
            lambda a: a["cases"][0]["validation"].pop(),
            lambda a: a["cases"][0]["final_parameters"].pop(),
            lambda a: a["cases"][0]["final_parameters"][0].pop(),
            lambda a: a["cases"][0]["validation"][0].update(target_bytes=3),
            lambda a: a["cases"][0]["validation"][0].update(mean_ce=2.),
            lambda a: a["cases"][0]["training"][0].update(ce=float("nan")),
            lambda a: a["cases"][0]["validation"][0]["batch_losses"].append(3.),
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                raw, ref, actual = fixture()
                edit(actual)
                with self.assertRaises(ValueError):
                    study.compare(raw, ref, actual)

    def test_tiny_disconnected_geometry_cannot_hide_beneath_absolute_tolerance(self):
        raw, ref, actual = fixture()
        import json
        initial = json.loads(raw)["cases"][1]["parameters"][2]["values"]
        ref["cases"][1]["final_parameters"][2] = [x + 1e-6 for x in initial]
        actual["cases"][1]["final_parameters"][2] = initial
        with self.assertRaisesRegex(ValueError, "geometry learning delta"):
            study.compare(raw, ref, actual)

    def test_reference_is_not_accepted_as_native_result(self):
        raw, ref, _ = fixture()
        with self.assertRaises(ValueError):
            study.compare(raw, ref, ref)

    def test_nonzero_incorrect_geometry_delta_is_rejected_relatively(self):
        raw, ref, actual = fixture()
        import json
        initial = json.loads(raw)["cases"][1]["parameters"][2]["values"]
        ref["cases"][1]["final_parameters"][2] = [x + 1e-6 for x in initial]
        actual["cases"][1]["final_parameters"][2] = [x + 5e-7 for x in initial]
        with self.assertRaisesRegex(ValueError, "geometry learning delta"):
            study.compare(raw, ref, actual)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU PyTorch")
class TorchReferenceControls(unittest.TestCase):
    def reference(self, request):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "request.json", root / "reference.json"
            study.write_new(source, request)
            study.reference(SimpleNamespace(request=source, output=output))
            return source.read_bytes(), json.loads(output.read_bytes())

    def test_integer_parameter_literals_preserve_float32_reference(self):
        request = json.loads(fixture()[0])
        _, expected = self.reference(request)
        for case in request["cases"]:
            for parameter in case["parameters"]:
                parameter["values"] = [int(x) if int(x) == x else x for x in parameter["values"]]
        _, actual = self.reference(request)
        self.assertEqual(actual["dtype"], "float32")
        self.assertEqual(actual["cases"], expected["cases"])

    def test_zero_gradient_reference_cannot_qualify_rounding_as_learning(self):
        request = json.loads(fixture()[0])
        request["train_batches"] = request["train_batches"][:1]
        request["checkpoint_every"] = 1
        for case in request["cases"]:
            for parameter in case["parameters"]:
                if parameter["name"].startswith("geometry."):
                    parameter["values"] = [.3] * len(parameter["values"])
                elif parameter["name"] == "head.output.weight":
                    parameter["values"] = [0.] * len(parameter["values"])
        raw, ref = self.reference(request)
        for parameter, final in zip(request["cases"][1]["parameters"], ref["cases"][1]["final_parameters"]):
            if parameter["name"].startswith("geometry."):
                self.assertEqual(final, [study.f32(x) for x in parameter["values"]])
        measured = copy.deepcopy(ref)
        measured.update(engine="spiraltorch", adapter="test-only GPU adapter")
        with self.assertRaisesRegex(ValueError, "geometry learning delta"):
            study.compare(raw, ref, measured)


if __name__ == "__main__":
    unittest.main()
