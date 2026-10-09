"""Public residual attention clients against the immutable Torch fixture."""
import ast
import gc
import json
import math
import os
from pathlib import Path
import unittest

import spiraltorch as st

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "crates/st-nn/tests/fixtures/residual_attention_torch.json"


def fixture():
    result = json.loads(FIXTURE.read_text())
    assert result["schema"] == "spiraltorch.residual_attention_torch.v1"
    assert len(result["cases"]) == 60 and len(result["training"]) == 2
    assert all(len(c["losses"]) == 32 for c in result["training"])
    return result


def component_records(c):
    width = c["input_shape"][-1]
    def norm(record):
        return [{"role": "gain", "shape": [width], "values": record["gain"]},
                {"role": "bias", "shape": [width], "values": record["bias"]}]
    stage = {"kind": "layer_norm", "gain": 0, "bias": 1, "epsilon": 1e-5}
    pre = {"schema": "spiraltorch.nn.inference_plan.v3", "input_shape": c["input_shape"],
           "parameters": norm(c["pre"]), "stages": [stage]}
    f = c["feed_forward"]
    parameters = norm(f) + [
        {"role": "weight", "shape": [width, f["hidden"]], "values": f["up_weight"]},
        {"role": "bias", "shape": [f["hidden"]], "values": f["up_bias"]},
    ]
    stages = [stage, {"kind": "linear", "weight": 2, "bias": 3, "gelu": True}]
    if c["topos"]:
        parameters.append({"role": "gate", "shape": [f["hidden"]], "values": f["gate"]})
        stages.append({"kind": "topos_resonator", "gate": 4, "coupling": 0.2,
                       "iterations": 4, "saturation": 0.12, "porosity": 0.3,
                       "max_volume": math.prod(c["input_shape"][:2]) * f["hidden"]})
    index = len(parameters)
    parameters.extend([
        {"role": "weight", "shape": [f["hidden"], width], "values": f["down_weight"]},
        {"role": "bias", "shape": [width], "values": f["down_bias"]},
    ])
    stages.append({"kind": "linear", "weight": index, "bias": index + 1, "gelu": False})
    feed = {"schema": "spiraltorch.nn.inference_plan.v5" if c["topos"] else pre["schema"],
            "input_shape": c["input_shape"], "parameters": parameters, "stages": stages}
    return pre, feed


def parts(c):
    pre, feed = [st.nn.InferencePlan.from_json(json.dumps(p)) for p in component_records(c)]
    projections = [st.nn.InferencePlan.from_json(json.dumps({
        "schema": "spiraltorch.nn.inference_plan.v1",
        "input_shape": c["input_shape"][:2] + [p["weight_shape"][0]],
        "stages": [{"inner": p["weight_shape"][0], "cols": p["weight_shape"][1],
                    "weight": p["weight"], "bias": p["bias"], "gelu": False}],
    })) for p in c["projections"]]
    attention = st.nn.AttentionInferencePlan.from_projection_plans(
        *projections, heads=c["heads"], causal_offset=0 if c["causal"] else None)
    return pre, attention, feed


def plan(c):
    return st.nn.ResidualAttentionPlan.from_plans(*parts(c))


def biases(device, c):
    b, t, _ = c["input_shape"]
    shapes = {"z_bias": [b, c["heads"], t], "pair_bias": [b, c["heads"], t, t]}
    return {key: device.upload(shape, c[key]) for key, shape in shapes.items()
            if c[key] is not None}


def values(tensor):
    return tensor.snapshot().read_values()


class ResidualPlanSurface(unittest.TestCase):
    def test_public_exports_and_stubs(self):
        names = ("ResidualAttentionPlan", "ResidentResidualAttentionTraining",
                 "ResidualAttentionForward", "ResidualAttentionGradients")
        for name in names:
            self.assertIn(name, st.nn.__all__)
            with self.assertRaises(TypeError):
                getattr(st.nn, name)()
        tree = ast.parse((ROOT / "bindings/st-py/spiraltorch/__init__.pyi").read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        exposed = {node.target.id for node in classes["_NnModule"].body
                   if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        for name in (*names, "AttentionInferencePlan", "ResidentAttentionTraining",
                     "AttentionForward", "AttentionGradients", "ResidentParameterUpdate"):
            self.assertIn("_Nn" + name, classes)
            self.assertIn(name, exposed)

    def test_exact_layouts_and_richer_graphs(self):
        for c in fixture()["training"]:
            source = parts(c)
            result = st.nn.ResidualAttentionPlan.from_plans(*source)
            self.assertEqual(result.input_shape, tuple(c["input_shape"]))
            self.assertEqual(result.output_shape, tuple(c["input_shape"]))
            for index in (0, 2):
                record = json.loads(source[index].to_json())
                record["input_shape"][:2] = [1, 6]
                changed = list(source)
                changed[index] = st.nn.InferencePlan.from_json(json.dumps(record))
                with self.assertRaises(ValueError):
                    st.nn.ResidualAttentionPlan.from_plans(*changed)
            self.assertIn('"layer_norm"', source[0].to_json())
            self.assertEqual('"topos_resonator"' in source[2].to_json(), c["topos"])

    def test_real_module_construction(self):
        pre = st.nn.Sequential()
        pre.add(st.nn.LayerNorm("pre", 4, -1.0, 1e-5))
        feed = st.nn.Sequential()
        feed.add(st.nn.LayerNorm("post", 4, -1.0, 1e-5))
        feed.add(st.nn.Linear("up", 4, 7))
        feed.add(st.nn.Gelu())
        feed.add_topos_resonator("gate", st.Tensor(1, 7, [0.8] * 7),
                                st.ToposResonatorKernel(coupling=0.2, iterations=4,
                                                      saturation=0.12, porosity=0.3,
                                                      max_values=42))
        feed.add(st.nn.Linear("down", 7, 4))
        attention = parts(fixture()["training"][0])[1]
        result = st.nn.ResidualAttentionPlan.from_plans(
            pre.inference_plan([2, 3, 4]), attention, feed.inference_plan([2, 3, 4]))
        self.assertEqual(result.output_shape, (2, 3, 4))

    def test_cpu_only_gate(self):
        if st.wgpu_kernel_reports_available():
            self.skipTest("CPU-only artifact required")
        with self.assertRaisesRegex(NotImplementedError, "wgpu"):
            plan(fixture()["training"][0]).compile_training_wgpu()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1",
                     "real WGPU opt-in")
class ResidualTrainingGpu(unittest.TestCase):
    def close(self, tensor, expected):
        actual = values(tensor)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(math.isfinite(a) and math.isfinite(e))
            self.assertLessEqual(abs(a - e), 3e-6 + 5e-5 * abs(e))

    def parameters(self, tensors, expected):
        self.assertEqual(len(tensors), len(expected))
        self.assertIn(len(tensors), (12, 13))
        for tensor, target in zip(tensors, expected):
            self.close(tensor, target)

    def test_60_vjps_and_retained_handles(self):
        for c in fixture()["cases"]:
            with self.subTest(case=c["name"]):
                model = plan(c).compile_training_wgpu()
                device = model.tensor_device()
                self.assertNotEqual(device.adapter_info()["device_type"], "Cpu")
                x = device.upload(c["input_shape"], c["input"])
                seed = device.upload(c["input_shape"], c["upstream"])
                bs = biases(device, c)
                forward = model.forward(x, **bs)
                self.assertEqual(forward.parameter_revision, 0)
                gradients = model.backward(forward, seed)
                output = forward.prediction_tensor()
                dx = gradients.input_gradient_tensor()
                dp = gradients.parameter_gradient_tensors()
                dz, db = gradients.z_bias_gradient_tensor(), gradients.pair_bias_gradient_tensor()
                del model, device, forward, gradients, x, seed, bs
                gc.collect()
                self.close(output, c["expected"])
                self.close(dx, c["input_gradient"])
                self.parameters(dp, c["parameter_gradients"])
                for tensor, key in ((dz, "z_bias_gradient"), (db, "pair_bias_gradient")):
                    if c[key] is None:
                        self.assertIsNone(tensor)
                    else:
                        self.assertIsNotNone(tensor)
                        self.close(tensor, c[key])

    def test_both_32_update_loops_without_intermediate_readback(self):
        for c in fixture()["training"]:
            with self.subTest(topos=c["topos"]):
                source = plan(c)
                model = source.compile_training_wgpu()
                device = model.tensor_device()
                x = device.upload(c["input_shape"], c["input"])
                target = device.upload(c["input_shape"], c["target"])
                bs = biases(device, c)
                initial = model.parameter_tensors()
                losses, updates = [], []
                loss_fn = st.nn.MeanSquaredError()
                for _ in range(32):
                    forward = model.forward(x, **bs)
                    loss = loss_fn.evaluate_resident(forward.prediction_tensor(), target)
                    losses.append(loss.loss_tensor())
                    gradients = model.backward(forward, loss.prediction_gradient_tensor())
                    updates.append(model.sgd(gradients, c["rate"]))
                self.assertEqual(model.attempted_updates, 32)
                for index, (loss, update) in enumerate(zip(losses, updates)):
                    self.close(loss, [c["losses"][index]])
                    self.assertEqual(update.read(), index + 1)
                self.parameters(model.parameter_tensors(), c["final_parameters"])
                self.close(model.forward(x, **bs).prediction_tensor(), c["final_prediction"])
                self.parameters(initial, c["initial_parameters"])
                self.parameters(source.compile_training_wgpu().parameter_tensors(),
                                c["initial_parameters"])

    def test_guards_strides_owner_rejection_and_recovery(self):
        c = fixture()["training"][1]
        model, foreign = [plan(c).compile_training_wgpu() for _ in range(2)]
        device = model.tensor_device()
        x = device.upload(c["input_shape"], c["input"]).permute([0, 2, 1]).contiguous().permute([0, 2, 1])
        seed = device.upload(c["input_shape"], c["upstream"]).permute([0, 2, 1]).contiguous().permute([0, 2, 1])
        bs = biases(device, c)
        first, current = model.forward(x, **bs), model.forward(x, **bs)
        foreign.forward(x, **bs)
        for owner, token in ((model, first), (foreign, current)):
            with self.assertRaisesRegex(ValueError, "latest forward"):
                owner.backward(token, seed)
        invalid_shape = device.upload([1], [1.])
        with self.assertRaises(ValueError):
            model.forward(invalid_shape, **bs)
        with self.assertRaises(ValueError):
            model.backward(current, invalid_shape)
        good = model.backward(current, seed)
        negative = device.upload(c["input_shape"], [-v for v in c["upstream"]])
        opposite = model.backward(current, negative)
        self.parameters(good.parameter_gradient_tensors(), c["parameter_gradients"])
        self.parameters(opposite.parameter_gradient_tensors(),
                        [[-v for v in row] for row in c["parameter_gradients"]])
        with self.assertRaisesRegex(ValueError, "owner's current version"):
            foreign.sgd(good, c["rate"])
        for rate in (-1., math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                model.sgd(good, rate)
        self.assertEqual(model.attempted_updates, 0)
        before = [values(t) for t in model.parameter_tensors()]
        huge = device.upload(c["input_shape"], [3e38] * len(c["input"]))
        bad = model.backward(current, huge.mul(huge))
        rejected = model.sgd(bad, c["rate"])
        with self.assertRaises(ValueError) as raised:
            rejected.read()
        self.assertEqual(raised.exception.code, "training_step_rejected")
        self.assertEqual([values(t) for t in model.parameter_tensors()], before)
        with self.assertRaisesRegex(ValueError, "owner's current version"):
            model.sgd(good, 0.)
        with self.assertRaisesRegex(ValueError, "latest forward"):
            model.backward(current, seed)
        current = model.forward(x, **bs)
        recovered = model.backward(current, seed)
        receipt = model.sgd(recovered, 0.)
        retained = current.prediction_tensor()
        del model, foreign, device, current, recovered
        gc.collect()
        self.assertEqual(receipt.read(), 2)
        self.close(retained, c["expected"])

    def test_terminal_overflow_rejects_every_derivative_and_update(self):
        def linear(weight, bias):
            return st.nn.InferencePlan.from_json(json.dumps({
                "schema": "spiraltorch.nn.inference_plan.v1", "input_shape": [1, 1, 1],
                "stages": [{"inner": 1, "cols": 1, "weight": [weight],
                            "bias": [bias], "gelu": False}],
            }))
        for output_failure in (True, False):
            with self.subTest(output_failure=output_failure):
                zero, one = linear(0., 0.), linear(1., 0.)
                huge_bias = 2e38 if output_failure else 0.
                attention = st.nn.AttentionInferencePlan.from_projection_plans(
                    zero, zero, one, linear(1., huge_bias), heads=1)
                model = st.nn.ResidualAttentionPlan.from_plans(
                    one, attention, linear(0., huge_bias)).compile_training_wgpu()
                device = model.tensor_device()
                x = device.upload([1, 1, 1], [0.])
                seed = device.upload([1, 1, 1], [1. if output_failure else 2e38])
                z = device.upload([1, 1, 1], [0.])
                pair = device.upload([1, 1, 1, 1], [0.])
                before = [values(t) for t in model.parameter_tensors()]
                self.assertEqual(len(before), 8)
                forward = model.forward(x, z_bias=z, pair_bias=pair)
                if output_failure:
                    with self.assertRaisesRegex(ValueError, "tensor contains a non-finite value"):
                        values(forward.prediction_tensor())
                else:
                    self.close(forward.prediction_tensor(), [0.])
                gradients = model.backward(forward, seed)
                dz, db = gradients.z_bias_gradient_tensor(), gradients.pair_bias_gradient_tensor()
                self.assertIsNotNone(dz)
                self.assertIsNotNone(db)
                parameters = gradients.parameter_gradient_tensors()
                self.assertEqual(len(parameters), 8)
                for tensor in [gradients.input_gradient_tensor(),
                               *parameters, dz, db]:
                    with self.assertRaisesRegex(ValueError, "tensor contains a non-finite value"):
                        values(tensor)
                update = model.sgd(gradients, 0.01)
                with self.assertRaises(ValueError) as raised:
                    update.read()
                self.assertEqual(raised.exception.code, "training_step_rejected")
                after = model.parameter_tensors()
                self.assertEqual(len(after), len(before))
                self.assertEqual([values(t) for t in after], before)


if __name__ == "__main__":
    unittest.main()
