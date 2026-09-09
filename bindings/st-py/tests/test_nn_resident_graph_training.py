"""Public graph training: no Python math/optimizer in the production path."""
import gc
import json
import math
import os
import unittest

import spiraltorch as st


def graph_plan(shape=(2, 1, 1), weight=1., gain=1.):
    return st.nn.InferencePlan.from_json(json.dumps(dict(
        schema="spiraltorch.nn.inference_plan.v2", input_shape=list(shape),
        parameters=[dict(role="weight", shape=[1, 1], values=[weight]),
                    dict(role="bias", shape=[1], values=[0.]),
                    dict(role="gain", shape=[1], values=[gain])],
        stages=[dict(kind="linear", weight=0, bias=1, gelu=False),
                dict(kind="pointwise", parameters=[2], steps=[dict(op="multiply", rhs=1)])])))


class GraphSurface(unittest.TestCase):
    def test_imports_opaque_ownership_and_rich_lowering(self):
        from spiraltorch.nn import ResidentGraphTraining, GraphTrainingSnapshot
        from spiraltorch.nn import GraphTrainingParametersSnapshot, GraphTrainingState
        for cls in (ResidentGraphTraining, GraphTrainingSnapshot,
                    GraphTrainingParametersSnapshot, GraphTrainingState):
            self.assertIs(cls, getattr(st.nn, cls.__name__))
            with self.assertRaises(TypeError): cls()
        model = st.nn.Sequential()
        model.add(st.nn.Scaler.from_gain("scale", st.Tensor(1, 1, [2.])))
        model.add(st.nn.Relu())
        plan = model.inference_plan([2, 3, 1])
        self.assertFalse(plan.is_dense)
        self.assertEqual((plan.stage_count, plan.source_operation_count), (2, 2))
        payload = json.loads(plan.to_json())
        self.assertEqual(payload["schema"], "spiraltorch.nn.inference_plan.v2")
        self.assertEqual(payload["parameters"], [dict(role="gain", shape=[1], values=[2.])])
        self.assertEqual(st.nn.InferencePlan.from_json(plan.to_json()).to_json(), plan.to_json())

    def test_policy_is_required_canonical_and_cpu_execution_is_unavailable(self):
        plan = graph_plan()
        with self.assertRaises(TypeError): plan.compile_graph_training_wgpu()
        for policy in ("", "auto", "EXACT", " exact", "module-compatible", True, 1, None):
            with self.assertRaises((ValueError, TypeError)):
                plan.compile_graph_training_wgpu(gradient_policy=policy)
        if not st.wgpu_kernel_reports_available():
            with self.assertRaisesRegex(NotImplementedError, "wgpu"):
                plan.compile_graph_training_wgpu(gradient_policy="exact")


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class GraphGpu(unittest.TestCase):
    def test_raw_effective_gradients_owning_snapshots_and_resume(self):
        for policy, gain_gradient in (("exact", 5.), ("module_compatible", 2.5)):
            plan = graph_plan()
            original = plan.to_json()
            gpu = plan.compile_graph_training_wgpu(gradient_policy=policy, kernel="register_2x2")
            self.assertNotEqual(gpu.adapter_info()["device_type"], "Cpu")
            self.assertEqual((gpu.stage_count, gpu.parameter_count, gpu.gradient_policy), (2, 3, policy))
            initial = gpu.parameter_snapshot()
            for call in (lambda: gpu.step(.1), gpu.state_snapshot, gpu.loss_snapshot):
                with self.assertRaises(ValueError): call()
            x, y = [1., 2.], [0., 0.]
            gpu.upload_batch_values(x, y)
            x[:] = y[:] = [99., 99.]
            gpu.step(.1)
            snapshot, loss, parameters = gpu.state_snapshot(), gpu.loss_snapshot(), gpu.parameter_snapshot()
            gpu.upload_batch(st.Tensor(2, 1, [0., 0.]), st.Tensor(2, 1, [0., 0.]))
            with self.assertRaises(ValueError): gpu.state_snapshot()
            gpu.step(0.)
            later = gpu.state_snapshot()
            del gpu
            gc.collect()
            self.assertEqual((snapshot.input_shape, snapshot.output_shape), ((2, 1, 1), (2, 1, 1)))
            self.assertEqual((snapshot.submitted_step, snapshot.batch_generation, snapshot.gradient_policy), (1, 1, policy))
            state = snapshot.read_state()
            self.assertEqual((state.loss, loss.read()), (2.5, 2.5))
            self.assertEqual((state.stage_count, state.parameter_count, state.gradient_policy), (2, 3, policy))
            self.assertEqual((state.input_shape, state.output_shape), ((2, 1, 1), (2, 1, 1)))
            self.assertEqual((state.submitted_step, state.batch_generation), (1, 1))
            self.assertEqual(state.prediction_values(), [1., 2.])
            self.assertEqual(state.input_gradient_values(), [1., 2.])
            for i, (role, shape, raw, effective, updated) in enumerate([
                ("weight", (1, 1), 5., 5., .5), ("bias", (1,), 3., 3., -.3),
                ("gain", (1,), 5., gain_gradient, 1. - .1 * gain_gradient),
            ]):
                self.assertEqual((state.parameter_role(i), state.parameter_shape(i)), (role, shape))
                self.assertEqual(state.parameter_gradient_values(i), [raw])
                self.assertEqual(state.effective_gradient_values(i), [effective])
                self.assertAlmostEqual(state.parameter_values(i)[0], updated)
            exported = state.to_plan().to_json()
            self.assertEqual(initial.read_plan().to_json(), original)
            self.assertEqual(parameters.read_plan().to_json(), exported)
            self.assertEqual(plan.to_json(), original)
            self.assertEqual((later.read_state().submitted_step, later.batch_generation), (2, 2))
            state.parameter_values(2)[0] = 99.
            state.prediction_values()[0] = 99.
            self.assertEqual(state.to_plan().to_json(), exported)
            for call in (snapshot.read_state, loss.read, initial.read_plan, parameters.read_plan, later.read_state):
                with self.assertRaisesRegex(RuntimeError, "consumed"): call()
            for method in (state.parameter_values, state.parameter_shape, state.parameter_role,
                           state.parameter_gradient_values, state.effective_gradient_values):
                for invalid in (True, .5, -1, 3, 2**65):
                    with self.assertRaises((TypeError, IndexError, OverflowError)): method(invalid)
            resumed = st.nn.InferencePlan.from_json(exported).compile_graph_training_wgpu(gradient_policy=policy)
            self.assertEqual((resumed.submitted_steps, resumed.batch_generation), (0, 0))
            resumed.upload_batch_values([1., 2.], [0., 0.])
            resumed.step(0.)
            prediction = resumed.state_snapshot().read_state().prediction_values()
            gain = 1. - .1 * gain_gradient
            for actual, expected in zip(prediction, [.2 * gain, .7 * gain]):
                self.assertAlmostEqual(actual, expected)

    def test_invalid_batch_and_rate_leave_current_state_unchanged(self):
        plan = graph_plan()
        for options in (dict(kernel="bad"), dict(accumulation="bad"), dict(tile_mnk=(True, 8, 16)),
                        dict(tile_mnk=(0, 8, 16)), dict(tile_mnk=(8, 8))):
            with self.assertRaises((ValueError, TypeError)):
                plan.compile_graph_training_wgpu(gradient_policy="exact", **options)
        gpu = plan.compile_graph_training_wgpu(gradient_policy="exact")
        gpu.upload_batch_values([1., 2.], [0., 0.])
        gpu.step(0.)
        for x, y in (([9.], [0., 0.]), ([9., 9.], [0.]),
                     ([float("nan"), 0.], [0., 0.]), ([9., 9.], [float("inf"), 0.])):
            with self.assertRaises(ValueError): gpu.upload_batch_values(x, y)
        with self.assertRaises(ValueError):
            gpu.upload_batch(st.Tensor(2, 1, [9., 9.]), st.Tensor(1, 2, [0., 0.]))
        for rate in (True, "0.1", -1., float("nan"), float("inf"), 1e100):
            with self.assertRaises((TypeError, ValueError, OverflowError)): gpu.step(rate)
        self.assertEqual((gpu.batch_generation, gpu.submitted_steps), (1, 1))
        self.assertEqual(gpu.loss_snapshot().read(), 2.5)
        gpu.step(0.)
        self.assertEqual(gpu.state_snapshot().read_state().prediction_values(), [1., 2.])

    def test_gain_failures_roll_back_all_parameters_and_recover(self):
        maximum = float.fromhex("0x1.fffffep+127")
        for rows, weight, gain, x, y, rate in [
            (1, 1., .0625, 16., 0., maximum / 8.),
            (2, maximum / 2., 0., 1., -1.5, .1),
        ]:
            plan = graph_plan((rows, 1), weight, gain)
            gpu = plan.compile_graph_training_wgpu(gradient_policy="exact")
            gpu.upload_batch_values([x] * rows, [y] * rows)
            gpu.step(rate)
            failed_loss, failed_state = gpu.loss_snapshot(), gpu.state_snapshot()
            parameters = gpu.parameter_snapshot()
            gpu.upload_batch_values([0.] * rows, [0.] * rows)
            gpu.step(.1)
            accepted = gpu.loss_snapshot()
            del gpu
            gc.collect()
            self.assertEqual(parameters.read_plan().to_json(), plan.to_json())
            self.assertEqual(accepted.read(), 0.)
            for read in (failed_loss.read, failed_state.read_state):
                with self.assertRaises(ValueError) as caught: read()
                self.assertEqual(caught.exception.code, "training_step_rejected")
                self.assertIs(type(caught.exception.stage), int)
                self.assertGreater(caught.exception.flags, 0)
                with self.assertRaisesRegex(RuntimeError, "consumed"): read()

    def test_parameterless_relu_and_legacy_dense_plan(self):
        net = st.nn.Sequential()
        net.add(st.nn.Relu())
        plan = net.inference_plan([2])
        gpu = plan.compile_graph_training_wgpu(gradient_policy="exact")
        self.assertEqual(gpu.parameter_count, 0)
        self.assertEqual(gpu.parameter_snapshot().read_plan().to_json(), plan.to_json())
        gpu.upload_batch_values([-1., 2.], [0., 0.])
        gpu.step(.1)
        state = gpu.state_snapshot().read_state()
        self.assertEqual(state.prediction_values(), [0., 2.])
        self.assertEqual(state.input_gradient_values(), [0., 2.])
        self.assertEqual(state.to_plan().to_json(), plan.to_json())
        dense = st.nn.Linear(1, 1).inference_plan([1])
        self.assertTrue(dense.is_dense)
        gpu = dense.compile_graph_training_wgpu(gradient_policy="exact")
        gpu.upload_batch_values([0.], [1.])
        gpu.step(.01)
        self.assertTrue(math.isfinite(gpu.loss_snapshot().read()))
        self.assertFalse(gpu.parameter_snapshot().read_plan().is_dense)


if __name__ == "__main__":
    unittest.main()
