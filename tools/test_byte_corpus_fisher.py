"""Fabricated protocol/verifier counterexamples, never model-quality evidence."""
import copy
import math
import unittest

import byte_corpus_study as study
import byte_corpus_calibration as calibration
import verify_byte_corpus_calibrated_controls as controls
import verify_byte_corpus_preparation as preparation
from test_byte_corpus_calibration import fixture as flat_fixture
from test_verify_byte_corpus_preparation import fixture as packet_fixture
from test_verify_byte_corpus_resume import RESUME, calibrated_fixture


def fixture():
    raw, reference, actual = flat_fixture()
    request = study.decode_json(raw)
    request["schema"] = study.REQUEST_V5
    for original in request["cases"][3:5]:
        case = copy.deepcopy(original)
        case.update(name=case["name"] + "_fisher_rao_rms_matched",
                    pair_metric=study.FISHER, bias_initialization=study.MATCHED)
        case["parameters"][6]["values"] = [4.5]
        request["cases"].append(case)
    raw = study.encoded(request)
    for report in (reference, actual):
        for original, case in zip(report["cases"][5:7], request["cases"][7:]):
            row = copy.deepcopy(original)
            row.update(name=case["name"], pair_metric=study.FISHER)
            row["final_parameters"][6] = [4.5 if study.frozen(case) else study.f32(4.51)]
            report["cases"].append(row)
        report.update(schema="spiraltorch.byte_corpus.result.v5", request_sha256=study.digest(raw))
    records = reference["calibration_verification"]["cases"]
    records[0]["pair_metric"] = study.FLAT
    records.append(dict(copy.deepcopy(records[0]), pair_metric=study.FISHER))
    return raw, reference, actual


def packet():
    source, result = packet_fixture()
    request = fixture()[0].rstrip(b"\n")
    report = study.decode_json(result["report_json"])
    report.update(schema="spiraltorch.byte_corpus.bias_preparation.v2",
                  request_sha256=study.digest(request))
    report["cases"][0]["pair_metric"] = study.FLAT
    fisher = copy.deepcopy(report["cases"][0])
    fisher.update(pair_metric=study.FISHER, raw_gains=[[4.5]])
    for name in ("before_checkpoint_json", "fitted_checkpoint_json"):
        checkpoint = study.decode_json(fisher[name])
        checkpoint["model"]["geometry"]["pair_metric"] = study.FISHER
        if name == "fitted_checkpoint_json":
            checkpoint["model"]["geometry"]["raw_gains"] = [[4.5]]
        fisher[name] = study.encoded(checkpoint).decode()
    before = study.decode_json(fisher["before_checkpoint_json"])
    old_gain = before["model"]["geometry"]["raw_gains"][0][0]
    fisher["candidate_rms"] = [[
        fisher["reference_rms"][0][0] * math.log1p(math.exp(old_gain)) / math.log1p(math.exp(4.5))
    ]]
    report["cases"].append(fisher)
    result.update(request_json=request.decode(), report_json=study.encoded(report).decode())
    return source, result


def resume_fixture():
    args = calibrated_fixture()
    request = args[0]
    request["schema"] = study.REQUEST_V5
    for value in (request, *args[2:]):
        value["cases"] += copy.deepcopy(value["cases"][5:7])
        for index in (7, 8):
            value["cases"][index]["name"] += "_fisher"
    for case in request["cases"][7:]:
        case["pair_metric"] = study.FISHER
        case["parameters"][6]["values"] = [4.5]
    for report in args[2:5]:
        report["schema"] = report["schema"].removesuffix("v4") + "v5"
        for case in report["cases"][7:]:
            case["pair_metric"] = study.FISHER
            case["final_parameters"][6] = [4.5]
    args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v5"
    for case in args[5]["cases"][7:]:
        model = study.decode_json(case["model_json"])
        model["model"]["geometry"].update(pair_metric=study.FISHER, raw_gains=[[4.5]])
        case["model_json"] = study.encoded(model).decode()
    return args


class FisherStudy(unittest.TestCase):
    def test_preparation_binds_candidate_rms_to_each_metric_gain(self):
        source, prepared = packet()
        self.assertTrue(preparation.verify(source, prepared)["passed"])
        for index in range(2):
            changed = copy.deepcopy(prepared)
            report = study.decode_json(changed["report_json"])
            report["cases"][index]["candidate_rms"][0][0] *= 2
            changed["report_json"] = study.encoded(report).decode()
            with self.subTest(metric=index), self.assertRaisesRegex(ValueError, "gain transformation"):
                preparation.verify(source, changed)

    def test_metric_qualification_cannot_redefine_the_shared_reference(self):
        raw, reference, actual = fixture()
        record = reference["calibration_verification"]["cases"][1]
        record.update(reference_rms=[[20.]], fitted_rms=[[20.]], realized_relative_errors=[[0.]])
        with self.assertRaisesRegex(ValueError, "one Poincare RMS target"):
            study.compare(raw, reference, actual)
        source, prepared = packet()
        report = study.decode_json(prepared["report_json"])
        report["cases"][1].update(reference_rms=[[20.]], fitted_rms=[[20.]], realized_relative_errors=[[0.]])
        prepared["report_json"] = study.encoded(report).decode()
        with self.assertRaisesRegex(ValueError, "one Poincare RMS target"):
            preparation.verify(source, prepared)
        request = study.decode_json(fixture()[0])
        request["config"].update(heads=2, blocks=[False, False])
        qualification = fixture()[1]["calibration_verification"]
        for case in qualification["cases"]:
            case.update(reference_rms=[[2., 3.], [4., 5.]], fitted_rms=[[2., 3.], [4., 5.]],
                        realized_relative_errors=[[0., 0.], [0., 0.]])
        calibration.verify_qualification(request, qualification)
        for block in range(2):
            for head in range(2):
                changed = copy.deepcopy(qualification)
                for key in ("reference_rms", "fitted_rms"):
                    changed["cases"][1][key][block][head] *= 10
                with self.assertRaisesRegex(ValueError, "one Poincare RMS target"):
                    calibration.verify_qualification(request, changed)

    def test_first_update_controls_keep_old_seven_and_check_fisher_pair(self):
        raw, _, full = fixture()
        source, _, retained = flat_fixture()
        first = copy.deepcopy(full)
        first["schema"] = "spiraltorch.byte_corpus.partial.v5"
        for case in first["cases"]:
            case["training"] = case["training"][:1]
            case["validation"] = case["validation"][:1]
        result = controls.verify(raw, source, full, retained, first)
        self.assertEqual(len(result["inherited_full_reports_exact"]), 7)
        self.assertEqual(len(result["pairs"]), 4)
        for field in ("backbone", "frozen", "metric", "history"):
            changed = copy.deepcopy(first)
            if field == "metric":
                changed["cases"][8]["pair_metric"] = study.FLAT
            elif field == "history":
                changed["cases"][7]["training"].pop()
            else:
                changed["cases"][8]["final_parameters"][0 if field == "backbone" else 6][0] += .001
            with self.assertRaises(ValueError):
                controls.verify(raw, source, full, retained, changed)

    def test_resume_binds_fisher_metric_gain_policy_and_complete_history(self):
        self.assertTrue(RESUME.verify(*resume_fixture())["passed"])
        for mutation in ("metric", "gain", "schema", "history", "policy", "frozen"):
            args = resume_fixture()
            if mutation in ("metric", "gain", "schema"):
                model = study.decode_json(args[5]["cases"][8]["model_json"])
                if mutation == "metric":
                    model["model"]["geometry"]["pair_metric"] = study.FLAT
                elif mutation == "gain":
                    model["model"]["geometry"]["raw_gains"] = [[.25]]
                else:
                    model["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v1"
                args[5]["cases"][8]["model_json"] = study.encoded(model).decode()
            elif mutation == "history":
                args[5]["cases"][7]["training"].pop()
            else:
                for report in args[2:5]:
                    case = report["cases"][8]
                    if mutation == "policy":
                        case["geometry_update"] = "train"
                    else:
                        case["final_parameters"][6] = [4.5001]
            with self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_nine_modes_keep_old_seven_and_are_order_independent(self):
        raw, ref, actual = fixture()
        request = study.decode_json(raw)
        self.assertEqual(study.request_version(request), 5)
        self.assertEqual(request["cases"][:7], study.decode_json(flat_fixture()[0])["cases"])
        expected = study.compare(raw, ref, actual)["paired_deltas"]
        self.assertIn("fisher_minus_calibrated_flat_bpb", expected[0])
        for obj in (request, ref, actual):
            obj["cases"].reverse()
        raw = study.encoded(request)
        for obj in (ref, actual):
            obj["request_sha256"] = study.digest(raw)
        self.assertEqual(study.compare(raw, ref, actual)["paired_deltas"], expected)

    def test_missing_relabelled_unpaired_and_legacy_modes_reject(self):
        for change in (
            lambda r: r["cases"].pop(),
            lambda r: r["cases"][7].__setitem__("pair_metric", study.FLAT),
            lambda r: r["cases"][8].__setitem__("geometry_update", "train"),
            lambda r: r["cases"][7].__setitem__("bias_initialization", study.ORIGINAL),
            lambda r: r["cases"][8]["parameters"][6]["values"].__setitem__(0, -0.0),
            lambda r: r["cases"][7]["parameters"][2]["values"].__setitem__(0, .8),
            lambda r: r.__setitem__("schema", study.REQUEST_V4),
        ):
            request = study.decode_json(fixture()[0])
            change(request)
            with self.assertRaises(ValueError):
                study.request_version(request)
        old = study.decode_json(flat_fixture()[0])
        old["schema"] = study.REQUEST_V5
        with self.assertRaises(ValueError):
            study.request_version(old)

    def test_three_seeds_and_no_relaxed_old_capacity(self):
        request = study.decode_json(fixture()[0])
        group = request["cases"]
        request["cases"] = []
        for seed in (11, 23, 37, 41):
            for original in group:
                case = copy.deepcopy(original)
                case.update(seed=seed, name=f"{seed}_{case['name']}")
                request["cases"].append(case)
            if seed != 41:
                self.assertEqual(study.request_version(request), 5)
        with self.assertRaises(ValueError):
            study.request_version(request)

    def test_independent_qualification_requires_every_metric(self):
        for change in (
            lambda c: c.pop(),
            lambda c: c[1].__setitem__("pair_metric", study.FLAT),
            lambda c: c[1].pop("pair_metric"),
            lambda c: c[1]["fitted_rms"][0].__setitem__(0, 2.1),
            lambda c: c[1].__setitem__("seed", True),
        ):
            raw, ref, actual = fixture()
            change(ref["calibration_verification"]["cases"])
            with self.assertRaises(ValueError):
                study.compare(raw, ref, actual)

    def test_frozen_fisher_and_metric_report_identity_are_required(self):
        for change in (
            lambda a: a["cases"][8]["final_parameters"][6].__setitem__(0, 4.5001),
            lambda a: a["cases"][7].__setitem__("pair_metric", study.FLAT),
            lambda a: a["cases"][7].__setitem__("bias_initialization", study.ORIGINAL),
        ):
            raw, ref, actual = fixture()
            change(actual)
            with self.assertRaises(ValueError):
                study.compare(raw, ref, actual)

    def test_preparation_binds_both_metrics_and_actual_fitted_checkpoint(self):
        source, prepared = packet()
        self.assertTrue(preparation.verify(source, prepared)["passed"])
        for change in (
            lambda r: r["cases"].pop(),
            lambda r: r["cases"][1].__setitem__("pair_metric", study.FLAT),
            lambda r: r["cases"][1].__setitem__("raw_gains", [[.25]]),
            lambda r: r["cases"][1].__setitem__("fitted_checkpoint_json", r["cases"][0]["fitted_checkpoint_json"]),
        ):
            source, prepared = packet()
            report = study.decode_json(prepared["report_json"])
            change(report)
            prepared["report_json"] = study.encoded(report).decode()
            with self.assertRaises(ValueError):
                preparation.verify(source, prepared)


if __name__ == "__main__":
    unittest.main()
