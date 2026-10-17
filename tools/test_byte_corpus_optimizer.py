"""Synthetic receipts test fail-closed verification, not learning efficacy."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STUDY = load("byte_corpus_study")
OPT = load("byte_corpus_optimizer")
FIXTURE = load("test_byte_corpus_study")
RESUME_FIXTURE = load("test_verify_byte_corpus_resume")
RESUME = load("verify_byte_corpus_resume")


def recipe():
    return dict(kind=OPT.ADAMW, beta1=.9, beta2=.999, epsilon=1e-5,
                weight_decay=.01, geometry_weight_decay=0.)


def wrap(request, optimizer=None):
    result = copy.deepcopy(request)
    result.update(schema=STUDY.REQUEST_V9, design_schema=request["schema"],
                  optimizer=recipe() if optimizer is None else optimizer)
    return result


def state(case, controls, cursor):
    if controls["kind"] == OPT.SGD:
        return None
    slots = []
    for p in case["parameters"]:
        clock = 0 if STUDY.parameter_frozen(case, p["name"]) else cursor
        slots.append(dict(shape=p["shape"], steps=clock,
                          first=[STUDY.f32(.01) if clock else 0.] * len(p["values"]),
                          second=[STUDY.f32(.001) if clock else 0.] * len(p["values"])))
    return {**{key: controls[key] for key in ("beta1", "beta2", "epsilon")}, "slots": slots}


def comparison_fixture(optimizer=None):
    raw, reference, actual = FIXTURE.frozen_fixture()
    request = wrap(STUDY.decode_json(raw), optimizer)
    raw = STUDY.encoded(request)
    controls = OPT.recipe(request)
    for report in (reference, actual):
        report.update(schema="spiraltorch.byte_corpus.result.v9",
                      request_sha256=STUDY.digest(raw), design_schema=request["design_schema"],
                      optimizer=controls)
        for case, row in zip(request["cases"], report["cases"]):
            row["optimizer_state"] = state(case, controls, 2)
    return raw, reference, actual


def resume_fixture(optimizer=None):
    args = RESUME_FIXTURE.frozen_fixture()
    args[0]["rate"] = .001
    args[0] = wrap(args[0], optimizer)
    controls = OPT.recipe(args[0])
    for report in args[2:5]:
        report["schema"] = report["schema"][:-1] + "9"
        report.update(design_schema=args[0]["design_schema"], optimizer=controls)
        for case, row in zip(args[0]["cases"], report["cases"]):
            row["optimizer_state"] = state(case, controls, 1 if report is args[3] else 2)
    args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v9"
    if controls["kind"] == OPT.ADAMW:
        for case, saved in zip(args[0]["cases"], args[5]["cases"]):
            model = json.loads(saved["model_json"])
            model.update(model_schema=model["schema"], schema="spiraltorch.nn.byte_decoder_checkpoint.v6",
                         update_rule=OPT.ADAMW, optimizer=state(case, controls, 1))
            saved["model_json"] = json.dumps(model)
    return args


class OptimizerStudyTests(unittest.TestCase):
    def test_eight_geometry_designs_keep_validation_after_wrapping(self):
        sources = [FIXTURE.fixture()[0], FIXTURE.frozen_fixture()[0], FIXTURE.metric_fixture()[0]]
        sources += [load("test_byte_corpus_" + name).fixture()[0]
                    for name in ("calibration", "fisher", "headwise", "mixture")]
        sources.append(load("test_byte_corpus_query").fixture()[1])
        for index, raw in enumerate(sources, 1):
            request = STUDY.decode_json(raw)
            for optimizer in (recipe(), {"kind": OPT.SGD}):
                wrapped = wrap(request, optimizer)
                self.assertEqual(STUDY.request_version(wrapped), 9)
                self.assertEqual(STUDY.request_version(OPT.design(wrapped)), index)
                self.assertEqual(OPT.design(wrapped), request)

    def test_null_unknown_and_lossy_controls_reject(self):
        request = wrap(STUDY.decode_json(FIXTURE.fixture()[0]))
        for key in ("optimizer", "design_schema"):
            for null in (False, True):
                invalid = copy.deepcopy(request)
                if null:
                    invalid[key] = None
                else:
                    del invalid[key]
                with self.assertRaises(ValueError):
                    STUDY.request_version(invalid)
        for key in OPT.FIELDS:
            for bad in (None, True, -1., 1e100, 1e-100, 1e-40, float("nan")):
                invalid = copy.deepcopy(request)
                invalid["optimizer"][key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    STUDY.request_version(invalid)
        for controls in ({}, {"kind": OPT.SGD, "weight_decay": 0.}, {"kind": "adam"}):
            with self.assertRaises(ValueError):
                STUDY.request_version(wrap(OPT.design(request), controls))
        for key in ("optimizer", "design_schema"):
            legacy = OPT.design(request)
            legacy[key] = request[key]
            with self.assertRaises(ValueError):
                STUDY.request_version(legacy)

    def test_comparison_checks_optimizer_history_and_existing_contrasts(self):
        for controls in (recipe(), {"kind": OPT.SGD}):
            args = comparison_fixture(controls)
            result = STUDY.compare(*args)
            self.assertEqual(result["schema"], "spiraltorch.byte_corpus.comparison.v9")
            self.assertIn("learned_geometry_minus_frozen_bpb", result["paired_deltas"][0])
            for key in ("optimizer", "design_schema"):
                invalid = copy.deepcopy(args)
                del invalid[2][key]
                with self.assertRaises(ValueError):
                    STUDY.compare(*invalid)
            invalid = copy.deepcopy(args)
            del invalid[2]["cases"][0]["optimizer_state"]
            with self.assertRaises(ValueError):
                STUDY.compare(*invalid)
        for field, value in (("steps", 1), ("steps", True), ("shape", [1]),
                             ("first", [float("nan")]), ("second", [-1.])):
            args = comparison_fixture()
            args[2]["cases"][0]["optimizer_state"]["slots"][0][field] = value
            with self.assertRaises(ValueError):
                STUDY.compare(*args)
        args = comparison_fixture()
        args[2]["cases"][0]["optimizer_state"]["slots"][0]["first"][0] += .1
        with self.assertRaises(ValueError):
            STUDY.compare(*args)
        args = comparison_fixture()
        args[2]["cases"][2]["optimizer_state"]["slots"][2]["first"][0] = -0.
        with self.assertRaises(ValueError):
            STUDY.compare(*args)

    def test_resume_binds_model_history_to_paused_readback(self):
        for controls in (recipe(), {"kind": OPT.SGD}):
            args = resume_fixture(controls)
            result = RESUME.verify(*args)
            self.assertTrue(result["saved_optimizer_bits_and_clocks_match"])
            self.assertEqual(result["schema"], "spiraltorch.byte_corpus.resume_verification.v9")
        for mutation in ("moment", "clock", "beta", "topology", "dropped", "frozen"):
            args = resume_fixture()
            index = 2 if mutation == "frozen" else 0
            model = json.loads(args[5]["cases"][index]["model_json"])
            if mutation == "moment":
                model["optimizer"]["slots"][0]["first"][0] += .01
            elif mutation == "clock":
                model["optimizer"]["slots"][0]["steps"] = 2
            elif mutation == "beta":
                model["optimizer"]["beta1"] = .8
            elif mutation == "topology":
                model["model_schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v5"
            elif mutation == "frozen":
                model["optimizer"]["slots"][2]["steps"] = 1
            else:
                del model["optimizer"]
            args[5]["cases"][index]["model_json"] = json.dumps(model)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_erased_tiny_moments_cannot_hide_below_weight_tolerance(self):
        for moment in ("first", "second"):
            args = comparison_fixture()
            for report in args[1:]:
                for row in report["cases"]:
                    for slot in row["optimizer_state"]["slots"]:
                        if slot["steps"]:
                            slot[moment] = [1e-20] * len(slot[moment])
            self.assertTrue(STUDY.compare(*args)["numerical_checks_passed"])
            for row in args[2]["cases"]:
                for slot in row["optimizer_state"]["slots"]:
                    slot[moment] = [0.] * len(slot[moment])
            with self.assertRaisesRegex(ValueError, "unqualified optimizer moment"):
                STUDY.compare(*args)

    def test_completed_checkpoint_must_equal_no_op_resume_including_moments(self):
        args = resume_fixture()
        args[3] = copy.deepcopy(args[2])
        args[5]["completed_updates"] = 2
        for row, saved in zip(args[3]["cases"], args[5]["cases"]):
            saved["training"] = copy.deepcopy(row["training"])
            saved["validation"] = [{key: point[key] for key in ("revision", "batch_losses")}
                                   for point in row["validation"]]
            model = json.loads(saved["model_json"])
            model["attempted_revision"] = "2"
            model["optimizer"] = copy.deepcopy(row["optimizer_state"])
            saved["model_json"] = json.dumps(model)
        self.assertTrue(RESUME.verify(*args)["passed"])
        model = json.loads(args[5]["cases"][0]["model_json"])
        model["optimizer"]["slots"][0]["first"][0] = 123.
        args[5]["cases"][0]["model_json"] = json.dumps(model)
        args[3]["cases"][0]["optimizer_state"]["slots"][0]["first"][0] = 123.
        with self.assertRaisesRegex(ValueError, "no-op resume"):
            RESUME.verify(*args)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "independent PyTorch optional")
    def test_real_torch_optimizer_respects_none_gradient_freeze(self):
        request = wrap(STUDY.decode_json(FIXTURE.frozen_fixture()[0]))
        request["rate"] = .001
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            STUDY.write_new(path / "request.json", request)
            STUDY.reference(SimpleNamespace(request=path / "request.json", output=path / "reference.json"))
            report = STUDY.decode_json((path / "reference.json").read_bytes())
        OPT.metadata(request, report)
        for case, row in zip(request["cases"], report["cases"]):
            OPT.validate_state(case, row["optimizer_state"], OPT.recipe(request), 2, STUDY.parameter_frozen)
        self.assertNotEqual(report["cases"][0]["final_parameters"][0], request["cases"][0]["parameters"][0]["values"])


if __name__ == "__main__":
    unittest.main()
