"""Existing Python NN modules use Rust-owned plans, not a Python NN compiler."""
import gc
import json
import math
import os
import unittest

import spiraltorch as st
from spiraltorch.nn import InferencePlan


def model():
    net = st.nn.Sequential()
    net.add(st.nn.Linear(2, 3, name="up"))
    net.add(st.nn.Gelu())
    net.add(st.nn.Linear(3, 2, name="down"))
    net.load_state_dict([
        ("up::weight", st.Tensor(2, 3, [0.5, -0.25, 0.125, -0.5, 0.25, 0.75])),
        ("up::bias", st.Tensor(1, 3, [0.125, -0.25, 0.5])),
        ("down::weight", st.Tensor(3, 2, [0.25, -0.5, 0.125, 0.75, -0.25, 0.5])),
        ("down::bias", st.Tensor(1, 2, [0.125, -0.25])),
    ])
    return net


def reference(values):
    # Independent small f64 oracle; only a test, never production routing/math.
    first = ((0.5, -0.25, 0.125), (-0.5, 0.25, 0.75))
    second = ((0.25, -0.5), (0.125, 0.75), (-0.25, 0.5))
    output = []
    for offset in range(0, len(values), 2):
        x = values[offset:offset+2]
        h = [sum(x[k] * first[k][j] for k in range(2)) + (0.125, -0.25, 0.5)[j]
             for j in range(3)]
        h = [0.5 * y * (1 + math.tanh(0.7978845834732056 * (y + 0.044715 * y**3))) for y in h]
        output.extend(sum(h[k] * second[k][j] for k in range(3)) + (0.125, -0.25)[j]
                      for j in range(2))
    return output


class ResidentPlanSurface(unittest.TestCase):
    def test_existing_model_roundtrip_and_frozen_state(self):
        net = model()
        plan = net.inference_plan([2, 3, 2])
        self.assertEqual((plan.input_shape, plan.output_shape), ((2, 3, 2), (2, 3, 2)))
        self.assertEqual((plan.stage_count, plan.source_operation_count), (2, 3))
        payload = plan.to_json()
        self.assertIs(InferencePlan, st.nn.InferencePlan)
        restored = InferencePlan.from_json(payload)
        self.assertEqual(restored.to_json(), payload)
        net.load_state_dict([(name, st.Tensor(*tensor.shape(), [0.] * math.prod(tensor.shape())))
                             for name, tensor in net.state_dict()])
        self.assertEqual(plan.to_json(), payload)
        self.assertNotEqual(net.inference_plan([2, 3, 2]).to_json(), payload)
        self.assertEqual(len(net), 3)
        linear = st.nn.Linear(2, 3)
        self.assertEqual(linear.inference_plan([2]).output_shape, (3,))
        with self.assertRaises(TypeError):
            st.nn.ResidentInference()
        with self.assertRaises(TypeError):
            st.nn.InferenceSnapshot()

    def test_invalid_plans_and_shapes_fail_closed(self):
        net = model()
        for shape in ([], [0, 2], [2, 3], [True, 2], [2.0, 2], [2, -1]):
            with self.subTest(shape=shape), self.assertRaises((ValueError, TypeError, OverflowError)):
                net.inference_plan(shape)
        unsupported = st.nn.Sequential()
        unsupported.add(st.nn.Relu())
        with self.assertRaises(ValueError):
            unsupported.inference_plan([2, 2])
        payload = net.inference_plan([2, 2]).to_json()
        with self.assertRaises(ValueError):
            st.nn.InferencePlan.from_json(payload, max_bytes=len(payload.encode()) - 1)
        self.assertEqual(st.nn.InferencePlan.from_json(payload, max_bytes=len(payload.encode())).to_json(), payload)
        corrupted = json.loads(payload)
        corrupted["stages"][0]["cols"] += 1
        with self.assertRaises(ValueError):
            st.nn.InferencePlan.from_json(json.dumps(corrupted))

        with self.assertRaises(ValueError):
            st.nn.InferencePlan.from_json(payload.replace("spiraltorch.nn.inference_plan.v1", "unknown"))
        corrupted = json.loads(payload)
        corrupted["input_shape"] = [4294967295, 2]
        with self.assertRaises(ValueError):
            st.nn.InferencePlan.from_json(json.dumps(corrupted))

    def test_dense_facade_rejects_valid_rich_graph_before_device_allocation(self):
        rich = {"schema": "spiraltorch.nn.inference_plan.v2", "input_shape": [2, 2],
                "parameters": [], "stages": [{"kind": "pointwise", "parameters": [],
                "steps": [{"op": "relu", "rhs": None}]}]}
        with self.assertRaisesRegex(ValueError, "dense-only API"):
            st.nn.InferencePlan.from_json(json.dumps(rich))

    def test_cpu_only_plan_does_not_claim_gpu_execution(self):
        if not st.wgpu_kernel_reports_available():
            with self.assertRaises(NotImplementedError):
                model().inference_plan([2, 2]).compile_wgpu()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class ResidentPlanGpu(unittest.TestCase):
    def close(self, values, expected):
        self.assertEqual(len(values), len(expected))
        for a, b in zip(values, expected):
            self.assertTrue(math.isfinite(a) and abs(a-b) <= 1e-5 + 1e-4 * abs(b))

    def test_existing_forward_and_kernel_choices(self):
        net = model()
        plan = net.inference_plan([2, 2])
        values = [0.25, -0.5, 1., -1.]
        input = st.Tensor(2, 2, values)
        expected = reference(values)
        self.close([v for row in net(input).tolist() for v in row], expected)
        for kernel in ("scalar", "register_2x2"):
            for accumulation in ("sequential", "tiled", "compensated"):
                with self.subTest(kernel=kernel, accumulation=accumulation):
                    gpu = plan.compile_wgpu(kernel=kernel, accumulation=accumulation)
                    self.assertNotEqual(gpu.adapter_info()["device_type"], "Cpu")
                    output = gpu(input)
                    self.assertEqual(output.shape(), (2, 2))
                    self.close([v for row in output.tolist() for v in row], expected)
        for kwargs in (dict(kernel="unknown"), dict(accumulation="unknown"),
                       dict(tile_mnk=(True, 8, 16)), dict(tile_mnk=(8.5, 8, 16)),
                       dict(tile_mnk=(8, 8)), dict(tile_mnk=(0, 8, 16))):
            with self.assertRaises((ValueError, TypeError)):
                plan.compile_wgpu(**kwargs)

    def test_nd_snapshot_lifetime_and_invalid_uploads(self):
        plan = model().inference_plan([2, 3, 2])
        gpu = plan.compile_wgpu()
        with self.assertRaises(ValueError): gpu.dispatch()
        with self.assertRaises(ValueError): gpu.snapshot()
        values = [(i - 6) / 8 for i in range(12)]
        gpu.upload_values(values)
        with self.assertRaises(ValueError): gpu.snapshot()
        generation = gpu.dispatch()
        first = gpu.snapshot()
        for invalid in ([0.] * 11, [float("nan")] * 12, [float("inf")] * 12):
            with self.assertRaises(ValueError): gpu.upload_values(invalid)
        self.assertEqual(gpu.generation, generation)
        with self.assertRaises(ValueError): gpu.upload(st.Tensor(2, 6, values))
        with self.assertRaises(ValueError): gpu(st.Tensor(6, 2, values))
        unchanged = gpu.snapshot()
        next_values = [v / 2 for v in values]
        gpu.upload(st.Tensor(6, 2, next_values))
        with self.assertRaises(ValueError): gpu.snapshot()
        gpu.dispatch()
        second = gpu.snapshot()
        del gpu, plan
        gc.collect()
        self.assertEqual((first.shape, first.generation), ((2, 3, 2), generation))
        with self.assertRaises(ValueError): first.read_tensor()
        self.close(first.read_values(), reference(values))
        with self.assertRaises(RuntimeError): first.read_values()
        self.close(unchanged.read_values(), reference(values))
        self.close(second.read_values(), reference(next_values))

    def test_imported_plan_retains_intermediate_failure(self):
        payload = dict(schema="spiraltorch.nn.inference_plan.v1", input_shape=[1], stages=[
            dict(inner=1, cols=1, weight=[1.], bias=[0.], gelu=True)])
        gpu = st.nn.InferencePlan.from_json(json.dumps(payload)).compile_wgpu()
        for value in (1e20, -1e20, 1e13, -1e13):
            gpu.upload_values([value])
            gpu.dispatch()
            failure = gpu.snapshot()
            gpu.upload_values([0.25])
            gpu.dispatch()
            success = gpu.snapshot()
            with self.assertRaises(ValueError): failure.read_values()
            self.assertTrue(math.isfinite(success.read_values()[0]))


if __name__ == "__main__":
    unittest.main()
