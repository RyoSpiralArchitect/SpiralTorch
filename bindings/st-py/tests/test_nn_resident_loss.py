"""The public Rust Loss seeds GPU learning, never a Python loss reimplementation."""
import gc
import json
import os
import unittest

import spiraltorch as st


def read(tensor):
    return tensor.snapshot().read_values()


class Surface(unittest.TestCase):
    def test_shared_class_and_explicit_cpu_only_boundary(self):
        from spiraltorch.nn import ResidentLoss
        self.assertIs(ResidentLoss, st.nn.ResidentLoss)
        with self.assertRaises(TypeError): ResidentLoss()
        loss = st.nn.MeanSquaredError()
        host = st.Tensor(1, 1, [1.])
        self.assertEqual(loss.forward(host, host).tolist(), [[0.]])
        expected = TypeError if st.wgpu_kernel_reports_available() else NotImplementedError
        with self.assertRaises(expected): loss.evaluate_resident(host, host)


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_nd_views_retained_pair_and_nonfinite_guards(self):
        device = st.WgpuTensorDevice.create()
        loss = st.nn.MeanSquaredError()
        x = device.upload([3, 2, 1], [1., 2., 3., 4., 5., 6.]).permute([1, 0, 2])
        y = device.upload([], [1.]).broadcast_to([2, 3, 1])
        actual = read(x)
        pair = loss.evaluate_resident(x, y)
        value, gradient = pair.loss_tensor(), pair.prediction_gradient_tensor()
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(gradient.shape), (2, 3, 1))
        with self.assertRaises(TypeError): loss.evaluate_resident(x, st.Tensor(1, 6, [0.] * 6))
        with self.assertRaises(ValueError): loss.evaluate_resident(x, device.upload([6], [0.] * 6))
        huge = device.upload([2, 3, 1], [1e20] * 6)
        bad = loss.evaluate_resident(huge, y)
        maximum = device.upload([2, 3, 1], [float.fromhex("0x1.fffffep+127")] * 6)
        zero = device.upload([], [0.])
        poisoned = maximum.add(maximum).mul(zero)
        bad_target = loss.evaluate_resident(x, poisoned)
        for _ in range(16): loss.evaluate_resident(x, y)
        del pair, loss, device, x, y, huge, maximum, zero, poisoned
        gc.collect()
        self.assertAlmostEqual(read(value)[0], sum((v - 1) ** 2 for v in actual) / 6, places=5)
        for a, b in zip(read(gradient), actual): self.assertAlmostEqual(a, 2 * (b - 1) / 6, places=5)
        for item in (bad, bad_target):
            for tensor in (item.loss_tensor(), item.prediction_gradient_tensor()):
                with self.assertRaises(ValueError): read(tensor)

    def test_loss_to_vjp_to_update_and_explicit_model_handoff(self):
        for policy, scale in [("exact", 1.), ("module_compatible", .25)]:
            model = st.nn.Sequential()
            model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [2.])))
            model.add(st.nn.Relu())
            baseline = model.inference_plan([2, 2, 1])
            learner = baseline.compile_graph_learner_wgpu(gradient_policy=policy)
            device = learner.tensor_device()
            x = device.upload([2, 2, 1], [1.] * 4)
            y = device.upload([2, 2, 1], [1.] * 4)
            learner.set_input_tensor(x)
            objective = st.nn.MeanSquaredError()
            held, expected = [], 2.
            for step in range(32):
                forward = learner.forward()
                pair = objective.evaluate_resident(forward.prediction_tensor(), y)
                gradient = learner.backward(forward, pair.prediction_gradient_tensor())
                learner.sgd(gradient, .1)
                held.append((pair, learner.update_snapshot(), expected))
                expected -= .1 * scale * 2 * (expected - 1.)
            # A failed loss must block the entire update, even if its seed is finite.
            before = learner.parameter_snapshot().read_plan().to_json()
            forward = learner.forward()
            huge = device.upload([2, 2, 1], [1e20] * 4)
            bad = objective.evaluate_resident(forward.prediction_tensor(), huge)
            gradient = learner.backward(forward, bad.prediction_gradient_tensor())
            learner.sgd(gradient, 0.)
            rejected = learner.update_snapshot()
            self.assertEqual(learner.parameter_snapshot().read_plan().to_json(), before)
            with self.assertRaises(ValueError): rejected.read()
            fresh = learner.forward()
            pair = objective.evaluate_resident(fresh.prediction_tensor(), y)
            learner.sgd(learner.backward(fresh, pair.prediction_gradient_tensor()), 0.)
            self.assertEqual(learner.update_snapshot().read(), 34)
            updated = learner.parameter_snapshot().read_plan()
            self.assertEqual(baseline.apply_parameters_to(model, updated), 1)
            self.assertAlmostEqual(read(model(x))[0], expected, places=5)
            del learner, device, model, objective
            gc.collect()
            for step, (pair, receipt, value) in enumerate(held, 1):
                self.assertEqual(receipt.read(), step)
                self.assertAlmostEqual(read(pair.loss_tensor())[0], (value-1.)**2, places=5)
                for g in read(pair.prediction_gradient_tensor()): self.assertAlmostEqual(g, .5*(value-1.), places=5)


if __name__ == "__main__":
    unittest.main()
