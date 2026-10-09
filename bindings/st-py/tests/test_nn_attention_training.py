"""Public resident projection training against the frozen CPU-f32 PyTorch oracle."""
import gc
import json
import math
import os
from pathlib import Path
import unittest

import spiraltorch as st


FIXTURE = Path(__file__).resolve().parents[3] / "crates/st-nn/tests/fixtures/resident_attention_training_torch.json"


def fixture():
    value = json.loads(FIXTURE.read_text())
    assert value["schema"] == "spiraltorch.resident_attention_training_torch.v1"
    assert len(value["cases"]) == 30 and len(value["training"]["losses"]) == 16
    return value


def projections(case):
    assert len(case["projections"]) == 4
    result = []
    for p in case["projections"]:
        inner, cols = p["weight_shape"]
        result.append(st.nn.InferencePlan.from_json(json.dumps({
            "schema": "spiraltorch.nn.inference_plan.v1",
            "input_shape": case["input_shape"][:2] + [inner],
            "stages": [{"inner": inner, "cols": cols, "weight": p["weight"],
                        "bias": p["bias"], "gelu": False}],
        })))
    return result


def plan(case):
    return st.nn.AttentionInferencePlan.from_projection_plans(
        *projections(case), heads=case["heads"], causal_offset=0 if case["causal"] else None)


def biases(device, case):
    batch, tokens, _ = case["input_shape"]
    shapes = {"z_bias": [batch, case["heads"], tokens],
              "pair_bias": [batch, case["heads"], tokens, tokens]}
    return {name: device.upload(shape, case[name])
            for name, shape in shapes.items() if case[name] is not None}


def values(tensor):
    return tensor.snapshot().read_values()


class AttentionPlanSurface(unittest.TestCase):
    def test_public_exports_are_opaque(self):
        for name in ("AttentionInferencePlan", "ResidentAttentionTraining", "AttentionForward",
                     "AttentionGradients", "ResidentParameterUpdate"):
            self.assertIn(name, st.nn.__all__)
            with self.assertRaises(TypeError):
                getattr(st.nn, name)()

    def test_shapes_strict_integers_and_richer_plan_rejection(self):
        case = fixture()["cases"][0]
        ps = projections(case)
        p = plan(case)
        self.assertEqual(p.input_shape, tuple(case["input_shape"]))
        self.assertEqual(p.output_shape, (*case["input_shape"][:2], case["projections"][3]["weight_shape"][1]))
        for bad in (True, -1, 0.5, "2", 2**65):
            for key in ("heads", "causal_offset"):
                options = {"heads": case["heads"], key: bad}
                with self.subTest(key=key, bad=bad), self.assertRaises((TypeError, ValueError, OverflowError)):
                    st.nn.AttentionInferencePlan.from_projection_plans(*ps, **options)
        for index in range(4):
            for field in ("gelu", "input_shape"):
                payload = json.loads(ps[index].to_json())
                if field == "gelu":
                    payload["stages"][0]["gelu"] = True
                else:
                    payload["input_shape"][:2] = [1, 6]
                mutated = ps.copy()
                mutated[index] = st.nn.InferencePlan.from_json(json.dumps(payload))
                with self.assertRaises(ValueError):
                    st.nn.AttentionInferencePlan.from_projection_plans(*mutated, heads=case["heads"])

    def test_cpu_only_compile_fails_explicitly(self):
        if st.wgpu_kernel_reports_available():
            self.skipTest("GPU-enabled artifact; CPU-only lane checks the feature gate")
        with self.assertRaisesRegex(NotImplementedError, "wgpu"):
            plan(fixture()["cases"][0]).compile_training_wgpu()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class AttentionTrainingGpu(unittest.TestCase):
    def close(self, tensor, expected):
        actual = values(tensor)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(math.isfinite(a) and math.isfinite(e))
            self.assertLessEqual(abs(a - e), 3e-6 + 5e-5 * abs(e))

    def parameter_list(self, tensors, expected):
        self.assertEqual(len(tensors), 4)
        self.assertEqual(len(expected), 4)
        for tensor, reference in zip(tensors, expected):
            self.close(tensor, reference)

    def test_30_forward_vjp_cases_and_owning_handles(self):
        for case in fixture()["cases"]:
            with self.subTest(case=case["name"]):
                model = plan(case).compile_training_wgpu()
                device = model.tensor_device()
                self.assertNotEqual(device.adapter_info()["device_type"], "Cpu")
                self.assertEqual(model.input_shape, tuple(case["input_shape"]))
                x = device.upload(model.input_shape, case["input"])
                cotangent = device.upload(model.output_shape, case["upstream"])
                bs = biases(device, case)
                forward = model.forward(x, **bs)
                self.assertEqual(forward.parameter_revision, 0)
                gradients = model.backward(forward, cotangent)
                output = forward.prediction_tensor()
                dx = gradients.input_gradient_tensor()
                dp = gradients.parameter_gradient_tensors()
                dz = gradients.z_bias_gradient_tensor()
                db = gradients.pair_bias_gradient_tensor()
                del model, device, x, cotangent, bs, forward, gradients
                gc.collect()
                self.close(output, case["expected"])
                self.close(dx, case["input_gradient"])
                self.parameter_list(dp, case["parameter_gradients"])
                for tensor, key in ((dz, "z_bias_gradient"), (db, "pair_bias_gradient")):
                    if case[key] is None:
                        self.assertIsNone(tensor)
                    else:
                        self.close(tensor, case[key])

    def test_16_mse_sgd_updates_and_immutable_source(self):
        case = fixture()["training"]
        source = plan(case)
        model = source.compile_training_wgpu()
        device = model.tensor_device()
        x = device.upload(model.input_shape, case["input"])
        target = device.upload(model.output_shape, case["target"])
        bs = biases(device, case)
        initial = model.parameter_tensors()
        initial_values = [values(t) for t in initial]
        loss_fn = st.nn.MeanSquaredError()
        for index, expected in enumerate(case["losses"]):
            forward = model.forward(x, **bs)
            loss = loss_fn.evaluate_resident(forward.prediction_tensor(), target)
            self.close(loss.loss_tensor(), [expected])
            gradients = model.backward(forward, loss.prediction_gradient_tensor())
            update = model.sgd(gradients, case["rate"])
            self.assertIsInstance(update, st.nn.ResidentParameterUpdate)
            self.assertEqual(update.attempted_revision, index + 1)
            self.assertEqual(update.read(), index + 1)
        self.assertEqual(model.attempted_updates, 16)
        self.parameter_list(model.parameter_tensors(), case["final_parameters"])
        self.close(model.forward(x, **bs).prediction_tensor(), case["final_prediction"])
        fresh = source.compile_training_wgpu()
        self.assertEqual([values(t) for t in initial], initial_values)
        self.assertEqual([values(t) for t in fresh.parameter_tensors()], initial_values)

    def test_owner_stale_tokens_nonfinite_rejection_and_recovery(self):
        case = fixture()["training"]
        model, foreign = [plan(case).compile_training_wgpu() for _ in range(2)]
        device = model.tensor_device()
        x = device.upload(model.input_shape, case["input"])
        seed = device.upload(model.output_shape, case["upstream"])
        bs = biases(device, case)
        forward = model.forward(x, **bs)
        foreign.forward(x, **bs)
        with self.assertRaisesRegex(ValueError, "latest forward"):
            foreign.backward(forward, seed)
        good = model.backward(forward, seed)
        opposite_seed = device.upload(model.output_shape, [-v for v in case["upstream"]])
        opposite = model.backward(forward, opposite_seed)
        self.parameter_list(good.parameter_gradient_tensors(), case["parameter_gradients"])
        self.parameter_list(opposite.parameter_gradient_tensors(),
                            [[-v for v in a] for a in case["parameter_gradients"]])
        with self.assertRaisesRegex(ValueError, "owner's current version"):
            foreign.sgd(good, case["rate"])
        for rate in (-1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                model.sgd(good, rate)
        before = [values(t) for t in model.parameter_tensors()]
        huge = device.upload(model.output_shape, [3e38] * len(case["upstream"]))
        invalid = huge.mul(huge)
        bad = model.backward(forward, invalid)
        update = model.sgd(bad, case["rate"])
        self.assertEqual(update.attempted_revision, 1)
        with self.assertRaises(ValueError) as raised:
            update.read()
        self.assertEqual(raised.exception.code, "training_step_rejected")
        self.assertEqual([values(t) for t in model.parameter_tensors()], before)
        with self.assertRaisesRegex(ValueError, "owner's current version"):
            model.sgd(good, case["rate"])
        with self.assertRaisesRegex(ValueError, "latest forward"):
            model.backward(forward, seed)
        current = model.forward(x, **bs)
        gradients = model.backward(current, seed)
        retained_update = model.sgd(gradients, 0.)
        retained = current.prediction_tensor()
        del model, foreign, device, current, gradients
        gc.collect()
        self.assertEqual(retained_update.read(), 2)
        self.close(retained, case["expected"])


if __name__ == "__main__":
    unittest.main()
