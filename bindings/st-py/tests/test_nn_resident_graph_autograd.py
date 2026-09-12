"""Public clients use the Rust-owned exact VJP and opaque forward token."""
import gc
import os
import unittest
import spiraltorch as st


def plan(gain=2., shape=(2, 2, 1)):
    model = st.nn.Sequential()
    model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [gain])))
    model.add(st.nn.Relu())
    return model.inference_plan(shape).fuse_pointwise()


def read(tensor):
    return tensor.snapshot().read_values()


class Surface(unittest.TestCase):
    def test_cpu_exports_and_no_forged_tokens(self):
        from spiraltorch.nn import ResidentGraphAutograd, GraphForward, GraphGradients
        for cls in (ResidentGraphAutograd, GraphForward, GraphGradients):
            self.assertIs(cls, getattr(st.nn, cls.__name__))
            with self.assertRaises(TypeError):
                cls()
        if not st.wgpu_kernel_reports_available():
            with self.assertRaisesRegex(NotImplementedError, "wgpu"):
                plan().compile_graph_autograd_wgpu()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_resident_seed_exact_vjp_repeat_and_ownership(self):
        gpu = plan().compile_graph_autograd_wgpu()
        device = gpu.tensor_device()
        self.assertNotEqual(gpu.adapter_info()["device_type"], "Cpu")
        self.assertEqual((gpu.input_shape, gpu.output_shape, gpu.stage_count, gpu.parameter_count), ((2, 2, 1), (2, 2, 1), 1, 1))
        gpu.set_input_tensor(device.upload([2, 2, 1], [-2., -0., 1., 2.]))
        forward = gpu.forward()
        prediction = forward.prediction_tensor()
        cotangent = prediction.mul(device.upload([], [.5]))
        grads = gpu.backward(forward, cotangent)
        self.assertEqual((grads.input_generation, grads.submitted_forward, grads.submitted_backward), (1, 1, 1))
        self.assertEqual((forward.input_generation, forward.submitted_forward), (1, 1))
        self.assertEqual(read(grads.input_gradient_tensor()), [0., 0., 2., 4.])
        self.assertEqual(read(grads.parameter_gradient_tensor(0)), [5.])  # no extra row average
        self.assertEqual(grads.parameter_count, 1)
        for index in (-1, 1):
            with self.assertRaises((IndexError, OverflowError)):
                grads.parameter_gradient_tensor(index)
        zero = gpu.backward(forward, device.upload([], [0.]).broadcast_to([2, 2, 1]))
        self.assertEqual(read(zero.parameter_gradient_tensor(0)), [0.])
        self.assertEqual((gpu.input_generation, gpu.submitted_forwards, gpu.submitted_backwards), (1, 1, 2))
        fresh = gpu.forward()
        with self.assertRaisesRegex(ValueError, "token"):
            gpu.backward(forward, cotangent)
        self.assertEqual(read(fresh.prediction_tensor()), read(prediction))
        gpu.upload_values([0.] * 4)
        del gpu, device, forward, fresh, zero
        gc.collect()
        self.assertEqual(read(prediction), [0., 0., 2., 4.])
        self.assertEqual(read(grads.parameter_gradient_tensor(0)), [5.])

    def test_bad_seed_recovery_does_not_mask_forward_failure(self):
        gpu = plan().compile_graph_autograd_wgpu()
        other = plan().compile_graph_autograd_wgpu()
        device = gpu.tensor_device()
        with self.assertRaises(ValueError): gpu.forward()
        gpu.upload_values([1.] * 4)
        other.upload_values([1.] * 4)
        forward, alien = gpu.forward(), other.forward()
        seed = device.upload([2, 2, 1], [1.] * 4)
        for call in (lambda: gpu.backward(alien, seed), lambda: gpu.upload_values([]),
                     lambda: gpu.upload_values([float("nan")] * 4),
                     lambda: gpu.backward(forward, seed.reshape([4])),
                     lambda: gpu.set_input_tensor(seed.reshape([4]))):
            with self.assertRaises(ValueError): call()
        self.assertEqual((gpu.input_generation, gpu.submitted_forwards, gpu.submitted_backwards), (1, 1, 0))
        poisoned = seed.mul(device.upload([], [float.fromhex("0x1.fffffep+127")])).mul(device.upload([], [2.]))
        bad = gpu.backward(forward, poisoned)
        good = gpu.backward(forward, seed)
        self.assertEqual(read(good.parameter_gradient_tensor(0)), [4.])
        self.assertEqual(read(forward.prediction_tensor()), [2.] * 4)
        for value in (bad.input_gradient_tensor(), bad.parameter_gradient_tensor(0)):
            with self.assertRaisesRegex(ValueError, "non-finite"): read(value)
        gpu.set_input_tensor(poisoned)
        with self.assertRaises(ValueError): gpu.backward(forward, seed)
        invalid = gpu.forward()
        zero = seed.mul(device.upload([], [0.]))
        invalid_grad = gpu.backward(invalid, zero)
        gpu.upload_values([1.] * 4)
        fresh = gpu.forward()
        self.assertEqual(read(gpu.backward(fresh, seed).parameter_gradient_tensor(0)), [4.])
        for value in (invalid.prediction_tensor(), invalid_grad.input_gradient_tensor(), invalid_grad.parameter_gradient_tensor(0)):
            with self.assertRaisesRegex(ValueError, "non-finite"): read(value)


if __name__ == "__main__":
    unittest.main()
