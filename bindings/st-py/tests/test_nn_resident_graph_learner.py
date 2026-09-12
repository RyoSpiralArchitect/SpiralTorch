"""Custom-objective learning through the public Rust-owned GPU client."""
import gc
import json
import os
import unittest
import spiraltorch as st


def plan():
    model = st.nn.Sequential()
    model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [2.])))
    model.add(st.nn.Relu())
    return model.inference_plan([2, 2, 1]).fuse_pointwise()


def weights(gpu):
    return json.loads(gpu.parameter_snapshot().read_plan().to_json())["parameters"][0]["values"]


class Surface(unittest.TestCase):
    def test_explicit_policy_and_exports(self):
        from spiraltorch.nn import ResidentGraphLearner, GraphGradientBatch, GraphUpdateSnapshot
        for cls in (ResidentGraphLearner, GraphGradientBatch, GraphUpdateSnapshot):
            self.assertIs(cls, getattr(st.nn, cls.__name__))
        for cls in (ResidentGraphLearner, GraphUpdateSnapshot):
            with self.assertRaises(TypeError): cls()
        with self.assertRaises(TypeError): plan().compile_graph_learner_wgpu()
        if not st.wgpu_kernel_reports_available():
            with self.assertRaisesRegex(NotImplementedError, "wgpu"):
                plan().compile_graph_learner_wgpu(gradient_policy="exact")


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_weighted_learning_ownership_checkpoint_and_policies(self):
        for policy, normalization in [("exact", 1.), ("module_compatible", .25)]:
            gpu = plan().compile_graph_learner_wgpu(gradient_policy=policy)
            self.assertEqual((gpu.stage_count, gpu.parameter_count, gpu.gradient_policy), (1, 1, policy))
            self.assertEqual((gpu.input_shape, gpu.output_shape), ((2, 2, 1), (2, 2, 1)))
            self.assertNotEqual(gpu.adapter_info()["device_type"], "Cpu")
            device = gpu.tensor_device()
            gpu.set_input_tensor(device.upload([2, 2, 1], [1.] * 4))
            norm, negative = device.upload([], [.25]), device.upload([], [-1.])
            receipts, expected = [], 2.

            def update(learner):
                forward = learner.forward()
                error = forward.prediction_tensor().add(negative)
                first = learner.backward(forward, error.mul(norm))
                second = learner.backward(forward, error.mul(error).mul(error).mul(norm))
                batch = st.nn.GraphGradientBatch()
                batch.add(first, .75)
                batch.add(second, .25)
                self.assertEqual(len(batch), 2)
                del first, second
                learner.sgd_weighted(batch, .1)
                with self.assertRaises(ValueError): learner.sgd_weighted(batch, 0.)
                with self.assertRaises(ValueError): learner.backward(forward, error)
                return learner.update_snapshot()

            for step in range(32):
                receipts.append(update(gpu))
                error = expected - 1.
                expected -= .1 * normalization * (.75 * error + .25 * error ** 3)
            exported = gpu.parameter_snapshot().read_plan()
            self.assertAlmostEqual(weights(gpu)[0], expected, places=5)
            self.assertLess((expected - 1.) ** 2, 1.)
            resumed = st.nn.InferencePlan.from_json(exported.to_json()).compile_graph_learner_wgpu(gradient_policy=policy)
            resumed.upload_values([1.] * 4)
            a, b = update(gpu), update(resumed)
            self.assertEqual(weights(gpu), weights(resumed))
            self.assertEqual((a.read(), b.read()), (33, 1))
            self.assertEqual((gpu.input_generation, gpu.submitted_forwards, gpu.submitted_backwards, gpu.submitted_updates), (1, 33, 66, 33))
            del gpu, resumed, device
            gc.collect()
            for i, receipt in enumerate(receipts):
                self.assertEqual((receipt.input_generation, receipt.submitted_forward, receipt.submitted_update), (1, i + 1, i + 1))
                self.assertEqual(receipt.read(), i + 1)
                with self.assertRaisesRegex(RuntimeError, "consumed"): receipt.read()

    def test_batch_identity_limits_atomic_rejection_and_recovery(self):
        gpu = plan().compile_graph_learner_wgpu(gradient_policy="exact")
        other = plan().compile_graph_learner_wgpu(gradient_policy="exact")
        device = gpu.tensor_device()
        for owner in (gpu, other): owner.upload_values([1.] * 4)
        seed = device.upload([2, 2, 1], [1.] * 4)
        forward = gpu.forward()
        good = gpu.backward(forward, seed)
        alien = other.backward(other.forward(), seed)
        batch = st.nn.GraphGradientBatch()
        with self.assertRaises(ValueError): gpu.update_snapshot()
        with self.assertRaises(ValueError): gpu.sgd_weighted(batch, .1)
        batch.add(good, 1.)
        for invalid in (lambda: batch.add(alien, 1.), lambda: batch.add(good, float("nan"))):
            with self.assertRaises(ValueError): invalid()
            self.assertEqual(len(batch), 1)
        for rate in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError): gpu.sgd_weighted(batch, rate)
        for _ in range(255): batch.add(good, 0.)
        self.assertEqual(len(batch), 256)
        with self.assertRaises(ValueError): batch.add(good, 0.)
        self.assertEqual(len(batch), 256)
        self.assertEqual(gpu.submitted_updates, 0)
        self.assertEqual(weights(gpu), [2.])
        self.assertEqual(gpu.sgd_weighted(batch, 0.), 1)
        self.assertEqual(gpu.update_snapshot().read(), 1)
        with self.assertRaises(ValueError): gpu.sgd_weighted(batch, .1)
        del batch

        maximum = device.upload([], [float.fromhex("0x1.fffffep+127")])
        poison = seed.mul(maximum).mul(device.upload([], [2.]))
        fresh = gpu.forward()
        bad = gpu.backward(fresh, poison)
        good = gpu.backward(fresh, seed)
        batch = st.nn.GraphGradientBatch()
        batch.add(bad, 0.)
        batch.add(good, 1.)
        gpu.sgd_weighted(batch, 0.)
        rejected = gpu.update_snapshot()
        unchanged = gpu.parameter_snapshot()
        with self.assertRaises(ValueError): gpu.sgd(good, 0.)
        fresh = gpu.forward()
        gradient = gpu.backward(fresh, seed)
        difference = st.nn.GraphGradientBatch()
        difference.add(gradient, 2.)
        difference.add(gradient, -1.)
        gpu.sgd_weighted(difference, .1)
        recovered = gpu.update_snapshot()
        gpu.upload_values([0.] * 4)
        gpu.forward()
        self.assertEqual((recovered.input_generation, recovered.submitted_forward, recovered.submitted_update), (1, 3, 3))
        self.assertAlmostEqual(weights(gpu)[0], 1.6, places=6)
        del gpu, other, device
        gc.collect()
        with self.assertRaisesRegex(ValueError, "no parameters were committed"): rejected.read()
        with self.assertRaisesRegex(RuntimeError, "consumed"): rejected.read()
        self.assertEqual(json.loads(unchanged.read_plan().to_json())["parameters"][0]["values"], [2.])
        self.assertEqual(recovered.read(), 3)


if __name__ == "__main__":
    unittest.main()
