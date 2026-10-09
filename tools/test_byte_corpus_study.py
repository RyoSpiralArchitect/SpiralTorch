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


def frozen_fixture():
    raw, reference, actual = fixture()
    request = json.loads(raw)
    request["schema"] = study.REQUEST_V2
    request["cases"].append(copy.deepcopy(request["cases"][1]))
    request["cases"][-1]["name"] = "frozen"
    for i, case in enumerate(request["cases"]):
        case["geometry_update"] = "frozen" if i == 2 else "train"
    raw = study.encoded(request)
    for report in (reference, actual):
        report["schema"] = "spiraltorch.byte_corpus.result.v2"
        report["request_sha256"] = study.digest(raw)
        report["cases"].append(copy.deepcopy(report["cases"][1]))
        report["cases"][-1]["name"] = "frozen"
        for spec, case in zip(request["cases"], report["cases"]):
            case["geometry_update"] = spec["geometry_update"]
            case["trainable_parameter_scalars"] = study.trainable_scalars(spec)
            if study.frozen(spec):
                for i, parameter in enumerate(spec["parameters"]):
                    if parameter["name"].startswith("geometry."):
                        case["final_parameters"][i] = [study.f32(v) for v in parameter["values"]]
    return raw, reference, actual


def metric_fixture():
    raw, reference, actual = frozen_fixture()
    request = study.decode_json(raw)
    request["schema"] = study.REQUEST_V3
    for value in (request, reference, actual):
        value["cases"] += copy.deepcopy(value["cases"][1:3])
        value["cases"][3]["name"], value["cases"][4]["name"] = "flat", "flat_frozen"
    for index, case in enumerate(request["cases"]):
        if index:
            case["pair_metric"] = study.POINCARE if index < 3 else study.FLAT
    raw = study.encoded(request)
    for report in (reference, actual):
        report.update(schema="spiraltorch.byte_corpus.result.v3", request_sha256=study.digest(raw))
        for index, (case, spec) in enumerate(zip(report["cases"], request["cases"])):
            case["pair_metric"] = spec.get("pair_metric")
            loss = (4., 3.9, 4.1, 3.7, 4.)[index]
            case["validation"][-1].update(batch_losses=[loss], mean_ce=loss, bits_per_byte=loss / math.log(2))
    return raw, reference, actual


class Controls(unittest.TestCase):
    def test_contrasts_do_not_use_tolerance_accepted_summary_perturbations(self):
        raw, reference, actual = metric_fixture()
        expected = study.compare(raw, reference, actual)
        actual["cases"][3]["validation"][-1]["bits_per_byte"] += 1e-4
        actual["cases"][3]["validation"][-1]["mean_ce"] += 1e-4
        self.assertEqual(study.compare(raw, reference, actual), expected)
        reference["cases"][3]["validation"][-1]["bits_per_byte"] -= 1e-4
        self.assertEqual(study.compare(raw, reference, actual), expected)

    def test_metric_factorial_contrasts_and_reordered_arms(self):
        raw, reference, actual = metric_fixture()
        expected = study.compare(raw, reference, actual)
        self.assertEqual(expected["schema"], "spiraltorch.byte_corpus.comparison.v3")
        contrast = expected["paired_deltas"][0]
        self.assertAlmostEqual(contrast["flat_minus_poincare_bpb"], -.2 / math.log(2))
        self.assertAlmostEqual(contrast["flat_frozen_minus_poincare_frozen_bpb"], -.1 / math.log(2))
        self.assertAlmostEqual(contrast["metric_by_training_interaction_bpb"], -.1 / math.log(2))
        request = study.decode_json(raw)
        for value in (request, reference, actual):
            value["cases"].reverse()
        raw = study.encoded(request)
        for value in (reference, actual):
            value["request_sha256"] = study.digest(raw)
        self.assertEqual(study.compare(raw, reference, actual)["paired_deltas"], expected["paired_deltas"])

    def test_metric_requests_require_five_distinct_matched_arms(self):
        for variant in ("missing", "duplicate", "absent", "null", "ordinary_metric", "geometry_bits",
                        "backbone_bits", "old_version", "missing_policy"):
            request = study.decode_json(metric_fixture()[0])
            if variant == "missing":
                request["cases"].pop()
            elif variant == "duplicate":
                request["cases"][3]["pair_metric"] = study.POINCARE
            elif variant == "absent":
                request["cases"][3].pop("pair_metric")
            elif variant == "null":
                request["cases"][3]["pair_metric"] = None
            elif variant == "ordinary_metric":
                request["cases"][0]["pair_metric"] = None
            elif variant in ("geometry_bits", "backbone_bits"):
                request["cases"][3]["parameters"][2 if variant == "geometry_bits" else 0]["values"][0] += .01
            elif variant == "old_version":
                request["schema"] = study.REQUEST_V2
            elif variant == "missing_policy":
                request["cases"][4].pop("geometry_update")
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                study.request_version(request)

    def test_metric_case_budget_rejects_four_otherwise_valid_seed_groups(self):
        request = study.decode_json(metric_fixture()[0])
        group = request["cases"]
        request["cases"] = []
        for seed in (11, 23, 37, 41):
            copied = copy.deepcopy(group)
            for case in copied:
                case["seed"] = seed
                case["name"] = f"seed{seed}_{case['name']}"
            single = dict(request, cases=copied)
            self.assertEqual(study.request_version(single), 3)
            request["cases"] += copied
            if seed != 41:
                self.assertEqual(study.request_version(request), 3)
        self.assertEqual(len(request["cases"]), 20)
        with self.assertRaisesRegex(ValueError, "case count"):
            study.request_version(request)

    def test_metric_reports_reject_relabels_freeze_drift_and_boolean_numbers(self):
        for variant in ("metric", "missing_metric", "ordinary_metric", "policy", "frozen_drift", "boolean"):
            raw, reference, actual = metric_fixture()
            if variant == "metric":
                actual["cases"][3]["pair_metric"] = study.POINCARE
            elif variant == "missing_metric":
                actual["cases"][0].pop("pair_metric")
            elif variant == "ordinary_metric":
                actual["cases"][0]["pair_metric"] = study.FLAT
            elif variant == "policy":
                actual["cases"][4]["geometry_update"] = "train"
            elif variant == "frozen_drift":
                actual["cases"][4]["final_parameters"][2][0] += 1e-7
            else:
                actual["cases"][0]["final_parameters"][3][0] = False
                reference["cases"][0]["final_parameters"][3][0] = 0.
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                study.compare(raw, reference, actual)

    def test_json_decoder_preserves_signed_zero_and_rejects_ambiguity(self):
        value = study.decode_json('[0, -0, -0.0, -0e0, 12]')
        self.assertEqual(value[0], 0)
        self.assertIs(type(value[4]), int)
        self.assertTrue(all(math.copysign(1., x) == -1. for x in value[1:4]))
        for raw in ('{"a":1,"a":2}', '{"nested":{"policy":null,"policy":"train"}}',
                    '[NaN]', '[Infinity]', '[-Infinity]', '[1e400]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                study.decode_json(raw)

    def test_duplicate_policy_keys_cannot_qualify_even_with_rebound_hashes(self):
        raw, ref, actual = frozen_fixture()
        raw = raw.replace(b'"geometry_update":"train"', b'"geometry_update":"frozen","geometry_update":"train"', 1)
        for report in (ref, actual):
            report["request_sha256"] = study.digest(raw)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            study.compare(raw, ref, actual)

    def test_negative_zero_literal_initialization_matches_preserved_bits(self):
        raw, ref, actual = frozen_fixture()
        request = json.loads(raw)
        for case in request["cases"][1:]:
            case["parameters"][2]["values"][0] = -0.0
        for report in (ref, actual):
            report["cases"][1]["final_parameters"][2][0] = .01
            report["cases"][2]["final_parameters"][2][0] = -0.0
        raw = study.encoded(request).replace(b'-0.0,', b'-0,')
        self.assertIn(b'[-0,', raw)
        for report in (ref, actual):
            report["request_sha256"] = study.digest(raw)
        self.assertTrue(study.compare(raw, ref, actual)["numerical_checks_passed"])

    def test_frozen_control_positive_and_contrasts(self):
        result = study.compare(*frozen_fixture())
        self.assertEqual(result["schema"], "spiraltorch.byte_corpus.comparison.v2")
        self.assertEqual(result["paired_deltas"][0]["learned_geometry_minus_frozen_bpb"], 0.)
        self.assertEqual(result["cases"][0]["trainable_parameter_scalars"], result["cases"][2]["trainable_parameter_scalars"])
        self.assertTrue(all(p["frozen_bits_preserved"] for p in result["cases"][2]["geometry_deltas"]))

    def test_frozen_policy_or_small_weight_mutation_is_rejected(self):
        for field in ("policy", "count", "weight", "reference_weight", "schema"):
            raw, ref, actual = frozen_fixture()
            if field == "policy":
                actual["cases"][2]["geometry_update"] = "train"
            elif field == "count":
                actual["cases"][2]["trainable_parameter_scalars"] += 1
            elif field == "schema":
                actual["schema"] = study.RESULT
            else:
                report = ref if field == "reference_weight" else actual
                report["cases"][2]["final_parameters"][2][0] += 1e-7
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.compare(raw, ref, actual)

    def test_control_requests_reject_ambiguous_modes_or_initial_values(self):
        for variant in range(6):
            request = json.loads(frozen_fixture()[0])
            if variant == 0:
                request["cases"].pop()
            elif variant == 1:
                request["cases"][2]["geometry_update"] = None
            elif variant == 2:
                request["cases"][0]["geometry_update"] = "frozen"
            elif variant == 3:
                request["cases"][2]["parameters"][2]["values"][0] += .01
            elif variant == 4:
                request["cases"][2]["parameters"][0]["values"][0] += .01
            else:
                request["schema"] = study.REQUEST
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                study.request_version(request)

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
    def test_metric_reference_runs_all_arms_with_full_frozen_pullback(self):
        request = study.decode_json(metric_fixture()[0])
        # Width two's LayerNorm can erase this tiny fixture's metric contrast.
        # Use a sensitive four-channel control, not a changed comparison gate.
        request["config"].update(steps=4, width=4, hidden=6, heads=2, geometry_cols=4)
        plain, geometric = study.parameters(request["config"], 11)
        for p in geometric:
            if p["name"].startswith("geometry.raw_gain."):
                p["values"] = [4.] * len(p["values"])
        for case in request["cases"]:
            case["parameters"] = copy.deepcopy(geometric if case["geometry"] else plain)
        request.update(train_documents=[list(range(1, 10))], validation_documents=[list(range(11, 16))],
                       train_batches=[[(0, 0)], [(0, 4)]], validation_batches=[[(0, 0)]])
        _, reference = self.reference(request)
        self.assertEqual(len(reference["cases"]), 5)
        for index in (2, 4):
            case, report = request["cases"][index], reference["cases"][index]
            for p, final in zip(case["parameters"], report["final_parameters"]):
                if p["name"].startswith("geometry."):
                    self.assertEqual(final, [study.f32(v) for v in p["values"]])
            for slot in (0, 1):
                self.assertNotEqual(report["final_parameters"][slot], case["parameters"][slot]["values"])
            self.assertEqual(report["training"][0], reference["cases"][index - 1]["training"][0])
            self.assertEqual(report["validation"][0], reference["cases"][index - 1]["validation"][0])
        self.assertNotEqual(reference["cases"][1]["validation"][0], reference["cases"][3]["validation"][0])

    def test_raw_negative_zero_literal_is_preserved_by_frozen_reference(self):
        request = json.loads(frozen_fixture()[0])
        for case in request["cases"][1:]:
            case["parameters"][2]["values"][0] = -0.0
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "request.json", Path(directory) / "reference.json"
            source.write_bytes(study.encoded(request).replace(b'-0.0,', b'-0,'))
            study.reference(SimpleNamespace(request=source, output=output))
            result = study.decode_json(output.read_bytes())
        self.assertEqual(math.copysign(1., result["cases"][2]["final_parameters"][2][0]), -1.)

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

    def test_frozen_reference_keeps_geometry_but_trains_both_embeddings(self):
        request = json.loads(frozen_fixture()[0])
        _, reference = self.reference(request)
        case, report = request["cases"][2], reference["cases"][2]
        for p, final in zip(case["parameters"], report["final_parameters"]):
            if p["name"].startswith("geometry."):
                self.assertEqual(final, [study.f32(x) for x in p["values"]])
        for i in (0, 1):
            self.assertNotEqual(report["final_parameters"][i], case["parameters"][i]["values"])
        self.assertEqual(report["training"][0], reference["cases"][1]["training"][0])
        self.assertEqual(report["validation"][0], reference["cases"][1]["validation"][0])

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
