"""Public owning training clients. Real GPU tests are explicit opt-in."""
import gc
import json
import math
import os
import unittest

import spiraltorch as st


def scalar_plan(weights=(1.,), *, gelu=False, bias=0., shape=(1,)):
    return st.nn.InferencePlan.from_json(json.dumps(dict(
        schema="spiraltorch.nn.inference_plan.v1", input_shape=list(shape),
        stages=[dict(inner=1, cols=1, weight=[w], bias=[bias], gelu=gelu) for w in weights])))


class TrainingSurface(unittest.TestCase):
    def test_registered_owning_classes_have_no_public_constructor(self):
        from spiraltorch.nn import ResidentTraining, TrainingLossSnapshot, TrainingSnapshot
        from spiraltorch.nn import TrainingParametersSnapshot, TrainingState
        for cls in (ResidentTraining, TrainingLossSnapshot, TrainingSnapshot,
                    TrainingParametersSnapshot, TrainingState):
            self.assertIs(cls, getattr(st.nn, cls.__name__))
            with self.assertRaises(TypeError):
                cls()

    def test_cpu_only_is_explicit(self):
        if not st.wgpu_kernel_reports_available():
            with self.assertRaises(NotImplementedError):
                scalar_plan().compile_training_wgpu()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class TrainingGpu(unittest.TestCase):
    def test_snapshot_ownership_metadata_and_vjp(self):
        plan = scalar_plan(shape=(2, 1, 1))
        original = plan.to_json()
        gpu = plan.compile_training_wgpu()
        self.assertNotEqual(gpu.adapter_info()["device_type"], "Cpu")
        initial_parameters = gpu.parameter_snapshot()
        for call in (lambda: gpu.step(.1), gpu.loss_snapshot, gpu.state_snapshot):
            with self.assertRaises(ValueError): call()
        x, y = [1., 2.], [0., 0.]
        gpu.upload_batch_values(x, y)
        x[:] = [99., 99.]
        y[:] = [99., 99.]
        self.assertEqual(gpu.step(.1), 1)
        first, loss, parameters = gpu.state_snapshot(), gpu.loss_snapshot(), gpu.parameter_snapshot()
        self.assertEqual((first.input_shape, first.output_shape), ((2, 1, 1), (2, 1, 1)))
        self.assertEqual((first.submitted_step, first.batch_generation), (1, 1))
        gpu.upload_batch(st.Tensor(2, 1, [0., 0.]), st.Tensor(2, 1, [0., 0.]))
        with self.assertRaises(ValueError): gpu.state_snapshot()
        gpu.step(0.)
        second = gpu.state_snapshot()
        del gpu, plan
        gc.collect()
        state = first.read_state()
        self.assertEqual(state.loss, 2.5)
        self.assertEqual(state.prediction_values(), [1., 2.])
        self.assertEqual(state.input_gradient_values(), [1., 2.])
        self.assertEqual(state.weight_gradient(0).tolist(), [[5.]])
        self.assertEqual(state.bias_gradient(0).tolist(), [[3.]])
        self.assertAlmostEqual(state.weight(0).tolist()[0][0], .5)
        self.assertAlmostEqual(state.bias(0).tolist()[0][0], -.3)
        self.assertEqual((state.submitted_step, state.batch_generation, state.stage_count), (1, 1, 1))
        self.assertEqual(loss.read(), 2.5)
        self.assertEqual((loss.submitted_step, loss.batch_generation), (1, 1))
        exported = state.to_plan().to_json()
        self.assertEqual(parameters.read_plan().to_json(), exported)
        self.assertEqual(initial_parameters.read_plan().to_json(), original)
        values = state.prediction_values()
        values[0] = 42.
        self.assertEqual(state.prediction_values(), [1., 2.])
        self.assertEqual(state.to_plan().to_json(), exported)
        next_state = second.read_state()
        self.assertEqual((next_state.submitted_step, next_state.batch_generation), (2, 2))
        for call in (first.read_state, second.read_state, loss.read, parameters.read_plan,
                     initial_parameters.read_plan):
            with self.assertRaisesRegex(RuntimeError, "consumed"): call()
        for method in (state.weight, state.bias, state.weight_gradient, state.bias_gradient):
            for index in (True, .5, -1, 1, 2**65):
                with self.assertRaises((TypeError, IndexError, OverflowError)): method(index)

    def test_invalid_batch_is_atomic_and_rate_does_not_enqueue(self):
        plan = scalar_plan(shape=(2, 1))
        for options in (dict(kernel="unknown"), dict(accumulation="unknown"),
                        dict(tile_mnk=(True, 8, 16)), dict(tile_mnk=(8.5, 8, 16)),
                        dict(tile_mnk=(8, 8)), dict(tile_mnk=(0, 8, 16))):
            with self.assertRaises((ValueError, TypeError)): plan.compile_training_wgpu(**options)
        gpu = plan.compile_training_wgpu()
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

    def assert_rejected(self, read):
        with self.assertRaises(ValueError) as caught:
            read()
        error = caught.exception
        self.assertEqual(error.code, "training_step_rejected")
        self.assertIs(type(error.stage), int)
        self.assertGreater(error.flags, 0)

    def test_all_layer_rollback_and_late_error_read(self):
        plan = scalar_plan((1., .001))
        original = plan.to_json()
        gpu = plan.compile_training_wgpu()
        gpu.upload_batch_values([.5], [-.9995])
        gpu.step(float.fromhex("0x1.fffffep+127"))
        failed_loss, failed_state = gpu.loss_snapshot(), gpu.state_snapshot()
        after_rejection = gpu.parameter_snapshot()
        gpu.step(.1)
        accepted = gpu.loss_snapshot()
        del gpu
        gc.collect()
        self.assertEqual(after_rejection.read_plan().to_json(), original)
        self.assert_rejected(failed_loss.read)
        self.assert_rejected(failed_state.read_state)
        self.assertTrue(math.isfinite(accepted.read()))
        with self.assertRaisesRegex(RuntimeError, "consumed"): failed_state.read_state()
        with self.assertRaisesRegex(RuntimeError, "consumed"): failed_loss.read()

    def test_nonfinite_intermediates_reject_even_for_zero_rate(self):
        maximum = float.fromhex("0x1.fffffep+127")
        cases = [(scalar_plan((maximum,)), [0.], [-1.]),
                 (scalar_plan(), [0.], [maximum])]
        cases.extend((scalar_plan(gelu=True), [v], [0.]) for v in (1e20, -1e20, 1e13, -1e13))
        for plan, x, y in cases:
            with self.subTest(x=x, y=y):
                gpu = plan.compile_training_wgpu()
                gpu.upload_batch_values(x, y)
                gpu.step(0.)
                self.assert_rejected(gpu.state_snapshot().read_state)
                self.assertEqual(gpu.parameter_snapshot().read_plan().to_json(), plan.to_json())

    def test_zero_rate_signed_zero_and_saturated_gelu(self):
        for rate in (0., -0.):
            plan = scalar_plan((-0.,), bias=-0.)
            gpu = plan.compile_training_wgpu()
            gpu.upload_batch_values([0.], [1.])
            gpu.step(rate)
            state = gpu.state_snapshot().read_state()
            self.assertEqual(state.to_plan().to_json(), plan.to_json())
            self.assertEqual(math.copysign(1., state.weight(0).tolist()[0][0]), -1.)
        gpu = scalar_plan(gelu=True).compile_training_wgpu()
        for x in (8., -8., 12., -12.):
            gpu.upload_batch_values([x], [0.])
            gpu.step(0.)
            state = gpu.state_snapshot().read_state()
            self.assertTrue(math.isfinite(state.loss))
            self.assertAlmostEqual(state.prediction_values()[0], max(x, 0.), places=4)
            self.assertAlmostEqual(state.input_gradient_values()[0], 2 * max(x, 0.), places=4)


if __name__ == "__main__":
    unittest.main()
