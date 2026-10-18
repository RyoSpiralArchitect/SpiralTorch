"""Clipping recipe and independent-reference controls; not quality evidence."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

# Fixture modules also import sibling reference tools, never the Python binding.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OPT_TEST = load("test_byte_corpus_optimizer")
STUDY, OPT = OPT_TEST.STUDY, OPT_TEST.OPT
CLIP = load("byte_corpus_gradient_clip")


def bind(request, clipping=None):
    value = copy.deepcopy(request)
    value.update(schema=STUDY.REQUEST_V10, gradient_clip=(
        dict(kind=CLIP.ACTIVE, max_norm=.01) if clipping is None else clipping))
    return value


def learning_design():
    request = STUDY.decode_json(OPT_TEST.FIXTURE.frozen_fixture()[0])
    request["config"].update(width=4, hidden=8, heads=2, steps=4)
    plain, geometry = STUDY.parameters(request["config"], 11)
    for case in request["cases"]:
        case["parameters"] = copy.deepcopy(geometry if case["geometry"] else plain)
    request.update(rate=.001, train_documents=[list(b"geometry changes attention")],
                   validation_documents=[list(b"unseen document")],
                   train_batches=[[(0, 0)], [(0, 4)]])
    return request


def comparison_fixture():
    raw, reference, actual = OPT_TEST.comparison_fixture()
    request = bind(STUDY.decode_json(raw))
    raw = STUDY.encoded(request)
    for report in (reference, actual):
        report.update(schema="spiraltorch.byte_corpus.result.v10", request_sha256=STUDY.digest(raw),
                      gradient_clip=CLIP.recipe(request))
    return raw, reference, actual


def resume_fixture():
    args = OPT_TEST.resume_fixture()
    args[0] = bind(args[0])
    args[1] = STUDY.digest(STUDY.encoded(args[0]))
    for report in args[2:5]:
        report.update(schema=report["schema"].rsplit(".v", 1)[0] + ".v10",
                      request_sha256=args[1], gradient_clip=CLIP.recipe(args[0]))
    args[5].update(schema="spiraltorch.byte_corpus.checkpoint.v10", request_sha256=args[1])
    return args


class RecipeTests(unittest.TestCase):
    def test_all_eight_designs_and_both_optimizers_remain_explicit(self):
        sources = [OPT_TEST.FIXTURE.fixture()[0], OPT_TEST.FIXTURE.frozen_fixture()[0], OPT_TEST.FIXTURE.metric_fixture()[0]]
        sources += [load("test_byte_corpus_" + name).fixture()[0]
                    for name in ("calibration", "fisher", "headwise", "mixture")]
        sources.append(load("test_byte_corpus_query").fixture()[1])
        for index, raw in enumerate(sources, 1):
            design = STUDY.decode_json(raw)
            self.assertEqual(CLIP.recipe(design), {"kind": CLIP.DISABLED})
            for optimizer in (OPT_TEST.recipe(), {"kind": OPT.SGD}):
                v9 = OPT_TEST.wrap(design, optimizer)
                for policy in ({"kind": CLIP.DISABLED}, {"kind": CLIP.ACTIVE, "max_norm": .01}):
                    value = bind(v9, policy)
                    self.assertEqual(STUDY.request_version(value), 10)
                    self.assertEqual(STUDY.request_version(OPT.design(value)), index)
                    self.assertEqual(OPT.design(value), design)

    def test_invalid_policy_cannot_silently_disable_or_change_meaning(self):
        base = bind(OPT_TEST.wrap(STUDY.decode_json(OPT_TEST.FIXTURE.fixture()[0])))
        for field in ("gradient_clip", "optimizer", "design_schema"):
            for omitted in (False, True):
                changed = copy.deepcopy(base)
                if omitted:
                    del changed[field]
                else:
                    changed[field] = None
                with self.assertRaises(ValueError):
                    STUDY.request_version(changed)
        for value in (None, {}, {"kind": "unknown"}, {"kind": CLIP.ACTIVE},
                      {"kind": CLIP.DISABLED, "max_norm": 1.},
                      {"kind": CLIP.ACTIVE, "max_norm": 1., "extra": 0}):
            with self.assertRaises(ValueError):
                STUDY.request_version(bind(base, value if value is not None else {}))
        for value in (0., -0., -1., True, None, 1e100, 1e-100, 1e-40, float("nan")):
            with self.assertRaises(ValueError):
                STUDY.request_version(bind(base, {"kind": CLIP.ACTIVE, "max_norm": value}))
        for legacy in (OPT.design(base), OPT_TEST.wrap(OPT.design(base))):
            for value in (None, {"kind": CLIP.DISABLED}, {"kind": CLIP.ACTIVE, "max_norm": 1.}):
                changed = copy.deepcopy(legacy)
                changed["gradient_clip"] = value
                with self.assertRaises(ValueError):
                    STUDY.request_version(changed)

    def test_comparison_and_resume_require_matching_policy(self):
        args = comparison_fixture()
        self.assertEqual(STUDY.compare(*args)["schema"], "spiraltorch.byte_corpus.comparison.v10")
        for value in (None, {"kind": CLIP.DISABLED}, {"kind": CLIP.ACTIVE, "max_norm": .02}):
            changed = copy.deepcopy(args)
            changed[2]["gradient_clip"] = value
            with self.assertRaises(ValueError):
                STUDY.compare(*changed)
        args = resume_fixture()
        result = OPT_TEST.RESUME.verify(*args)
        self.assertEqual(result["schema"], "spiraltorch.byte_corpus.resume_verification.v10")
        self.assertTrue(result["saved_optimizer_bits_and_clocks_match"])
        for field in ("gradient_clip", "optimizer"):
            changed = copy.deepcopy(args)
            del changed[3][field]
            with self.assertRaises(ValueError):
                OPT_TEST.RESUME.verify(*changed)


@unittest.skipUnless(importlib.util.find_spec("torch"), "independent PyTorch optional")
class TorchTests(unittest.TestCase):
    def test_active_norm_does_not_mutate_or_include_frozen_gradients(self):
        import torch
        inputs = (torch.tensor([3., -4.]), torch.tensor([torch.finfo(torch.float32).max]))
        cloned = tuple(v.clone() for v in inputs)
        outputs = CLIP.torch_gradients(torch, dict(kind=CLIP.ACTIVE, max_norm=1.), inputs, [True, False])
        torch.testing.assert_close(outputs[0], torch.tensor([.6, -.8]), rtol=0, atol=0)
        self.assertIs(outputs[1], inputs[1])
        for actual, expected in zip(inputs, cloned):
            self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
        self.assertIs(CLIP.torch_gradients(torch, dict(kind=CLIP.DISABLED), inputs, [True, False]), inputs)
        with self.assertRaises(ValueError):
            CLIP.torch_gradients(torch, dict(kind=CLIP.ACTIVE, max_norm=1.),
                                 (inputs[0], torch.tensor([float("inf")])), [True, False])

    def test_floor_above_is_not_rounded_to_noop(self):
        import torch
        value = torch.tensor([0x332aaaab], dtype=torch.int32).view(torch.float32).repeat(9)
        output, = CLIP.torch_gradients(torch, dict(kind=CLIP.ACTIVE, max_norm=1e-30), (value,), [True])
        self.assertTrue((output > 0).all())
        self.assertLess(output.max().item(), 1e-30)

    def test_real_torch_reference_changes_updates_and_preserves_frozen_geometry(self):
        design = learning_design()
        for optimizer in ({"kind": OPT.SGD}, OPT_TEST.recipe()):
            reports = []
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for label, policy in (("off", {"kind": CLIP.DISABLED}), ("active", None)):
                    request = bind(OPT_TEST.wrap(design, optimizer), policy)
                    STUDY.write_new(root / (label + ".json"), request)
                    STUDY.reference(SimpleNamespace(request=root / (label + ".json"), output=root / (label + "-result.json")))
                    report = STUDY.decode_json((root / (label + "-result.json")).read_bytes())
                    CLIP.metadata(request, report)
                    reports.append(report)
                    for case, row in zip(request["cases"], report["cases"]):
                        OPT.validate_state(case, row["optimizer_state"], OPT.recipe(request), 2, STUDY.parameter_frozen)
                        for parameter, values in zip(case["parameters"], row["final_parameters"]):
                            if STUDY.parameter_frozen(case, parameter["name"]):
                                self.assertEqual(values, parameter["values"])
                        if case["geometry"] and not STUDY.frozen(case):
                            self.assertTrue(any(values != parameter["values"]
                                for parameter, values in zip(case["parameters"], row["final_parameters"])
                                if parameter["name"].startswith("geometry.")))
            self.assertNotEqual(reports[0]["cases"][0]["final_parameters"], reports[1]["cases"][0]["final_parameters"])


if __name__ == "__main__":
    unittest.main()
