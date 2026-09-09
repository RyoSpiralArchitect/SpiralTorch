"""Synthetic admission checks only, not evidence of GPU execution."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("forward_validation",
    Path(__file__).resolve().parents[1] / "tools/verify_resident_graph_forward_torch.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def fixture():
    report = dict(schema="spiraltorch.resident_graph_forward.v1", status="passed", cases=[],
                  adapter=dict(backend="BrowserWebGpu", device_type="synthetic-not-evidence"),
                  guards={name: True for name in ("pointwise_masked_overflow", "dense_masked_overflow",
                    "whole_graph_guard_capture", "repeated_inherited_guard", "gpu_input_guard",
                    "device_mismatch_atomic", "recovery", "broadcast_permute", "dense_v1_v2_specialized")})
    for seed in (17, 29):
        for shape in ([4], [3, 4], [2, 129, 4]):
            for kernel, accumulation in (("scalar", "sequential"), ("register_2x2", "compensated")):
                plan = dict(schema="spiraltorch.nn.inference_plan.v2", input_shape=shape,
                    parameters=[dict(role=role, shape=dims) for role, dims in zip(
                        ("gain", "weight", "bias", "gain", "weight", "bias"), ([4], [4,7], [7], [7], [7,3], [3]))],
                    stages=[dict(kind=kind) for kind in ("pointwise", "linear", "pointwise", "pointwise", "linear")])
                c = dict(seed=seed, shape=shape, kernel=kernel, accumulation=accumulation, plan=plan,
                         input=[0.] * math.prod(shape), dispatches_before_capture=18,
                         pre_gains=[1., .5, -.75, 1.25], post_shift=[-.125, .25, .5])
                c.update({field: [0.] * (math.prod(shape[:-1]) * 3)
                          for field in ("host", "strided", "prediction", "frozen", "post", "chained")})
                report["cases"].append(c)
    return report


class Admission(unittest.TestCase):
    def test_complete_recipe_admission(self):
        self.assertEqual(len(validator.admit(fixture())), 12)

    def test_failed_incomplete_duplicate_changed_recipe(self):
        for mutate in (lambda r: r.update(status="error"), lambda r: r["cases"].pop(),
                       lambda r: r["cases"].__setitem__(0, copy.deepcopy(r["cases"][1])),
                       lambda r: r["cases"][0].update(dispatches_before_capture=1)):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError): validator.admit(report)

    def test_failed_missing_or_nonstrict_guard(self):
        for key in fixture()["guards"]:
            for value in (False, 1, "true"):
                report = fixture()
                report["guards"][key] = value
                with self.assertRaises(ValueError): validator.admit(report)
            report = fixture()
            del report["guards"][key]
            with self.assertRaises(ValueError): validator.admit(report)

    def test_shape_and_role_drift(self):
        for mutate in (lambda c: c["plan"]["parameters"][0].update(role="bias"),
                       lambda c: c["plan"]["parameters"][0].update(shape=[1]),
                       lambda c: c["plan"].update(input_shape=[99]),
                       lambda c: c["input"].pop(), lambda c: c["chained"].pop(),
                       lambda c: c["plan"]["stages"].pop()):
            report = fixture()
            mutate(report["cases"][0])
            with self.assertRaises(ValueError): validator.admit(report)

    def test_cpu_adapter_and_browser_errors(self):
        for mutate in (lambda r: r["adapter"].update(device_type="Cpu"),
                       lambda r: r.update(adapter={}), lambda r: r.update(page_errors=["GPU trap"])):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError): validator.admit(report)


if __name__ == "__main__": unittest.main()
