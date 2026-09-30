"""Public clients must preserve the real Rust classifier's ownership and restart rules."""
import json
import math
import os
import unittest

import spiraltorch as st


def config_json():
    config = json.loads(st.ResidentConvNeXtClassifier.default_config_json())
    config.update(input_channels=1, input_hw=[8, 8], stage_dims=[2, 4],
                  stage_depths=[1, 1], patch_size=[2, 2], epsilon=1e-3)
    return json.dumps(config)


def read(tensor):
    return tensor.snapshot().read_values()


def batch(device):
    dataset = st.TensorVisionDataset("CIFAR10")
    for n in range(2):
        image = st.ImageTensor(1, 12, 14, [((i * 31 + n * 17) % 257) / 256 for i in range(168)])
        dataset.push(image, target=st.Tensor(1, 1, [float(n)]))
    pipeline = st.TransformPipeline(seed=77)
    pipeline.add_normalize([0.5], [0.25])
    pipeline.add_resize(10, 10)
    pipeline.add_horizontal_flip(0.5)
    pipeline.add_center_crop(8, 8)
    pipeline.enable_wgpu()
    return dataset.dataloader(2, seed=19, pipeline=pipeline).next_resident_batch(device)


def step(owner, x, y):
    forward = owner.forward(x)
    loss = st.nn.CrossEntropyWithLogits().evaluate_resident(forward.prediction_tensor(), y)
    gradients = owner.backward(forward, loss.prediction_gradient_tensor())
    update = owner.sgd(gradients, 0.01)
    return forward, loss, gradients, update


class Surface(unittest.TestCase):
    def test_aliases_and_config_are_rust_owned(self):
        for name in ("ResidentConvNeXtClassifier", "ConvNeXtForward", "ConvNeXtGradients",
                     "ConvNeXtUpdate", "ConvNeXtCheckpointSnapshot"):
            self.assertIs(getattr(st, name), getattr(st.vision, name))
            self.assertIn(name, st.__all__)
        config = json.loads(config_json())
        self.assertEqual(config["curvature"], -1.0)
        with self.assertRaises(TypeError):
            st.ResidentConvNeXtClassifier()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_resident_loader_training_frozen_checkpoint_and_bitwise_resume(self):
        device = st.WgpuTensorDevice.create()
        self.assertNotEqual(device.adapter_info()["device_type"], "Cpu")
        owner = st.ResidentConvNeXtClassifier.create(device, config_json(), 2, 2, 17)
        data = batch(owner.tensor_device())
        x, y = data.images(), data.upload_targets()
        self.assertEqual(owner.input_shape, [2, 1, 8, 8])
        self.assertEqual(owner.output_shape, [2, 2])
        self.assertEqual(len(owner.parameter_names()), 24)
        first = step(owner, x, y)
        second = step(owner, x, y)
        frozen = owner.checkpoint_snapshot()
        third = step(owner, x, y)
        payload = frozen.read_json()
        self.assertEqual(json.loads(payload)["backbone"]["attempted_updates"], 2)
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            frozen.read_json()
        resumed = st.ResidentConvNeXtClassifier.from_checkpoint_json(device, payload)
        resumed_third = step(resumed, x, y)
        self.assertEqual(read(third[0].prediction_tensor()), read(resumed_third[0].prediction_tensor()))
        for a, b in zip(owner.parameter_tensors(), resumed.parameter_tensors()):
            self.assertEqual(read(a), read(b))
        fourth = step(owner, x, y)
        resumed_fourth = step(resumed, x, y)
        for i, record in enumerate((first, second, third, fourth), 1):
            self.assertEqual(record[3].attempted_revision, i)
            self.assertEqual(record[3].read(), i)
            self.assertTrue(all(math.isfinite(v) for v in read(record[1].loss_tensor())))
            self.assertEqual(len(record[2].parameter_gradient_tensors()), 24)
            self.assertEqual(record[2].input_gradient_tensor().shape, x.shape)
        self.assertEqual(resumed_third[3].read(), 3)
        self.assertEqual(resumed_fourth[3].read(), 4)
        for a, b in zip(owner.parameter_tensors(), resumed.parameter_tensors()):
            self.assertEqual(read(a), read(b))
        # Resume never revives another owner's derivative token.
        with self.assertRaises(ValueError):
            resumed.sgd(fourth[2], 0.01)
        final = owner.checkpoint_snapshot().read_json()
        host = st.ResidentConvNeXtClassifier.host_from_checkpoint_json(final)
        pixels = read(x)
        images = [st.ImageTensor(1, 8, 8, pixels[i:i + 64]) for i in (0, 64)]
        expected = host.forward(images).tolist()
        actual = read(owner.forward(x).prediction_tensor())
        for a, b in zip(actual, [v for row in expected for v in row]):
            self.assertLess(abs(a - b) / (1 + abs(b)), 2e-4)

    def test_rejection_retry_stale_tokens_and_retained_handles(self):
        device = st.WgpuTensorDevice.create()
        owner = st.ResidentConvNeXtClassifier.create(device, config_json(), 2, 2, 17)
        data = batch(device)
        x, y = data.images(), data.upload_targets()
        forward = owner.forward(x)
        other = st.ResidentConvNeXtClassifier.create(device, config_json(), 2, 2, 17)
        with self.assertRaises(ValueError):
            other.backward(forward, device.upload([2, 2], [1.0] * 4))
        owner.forward(x)
        with self.assertRaises(ValueError):
            owner.backward(forward, device.upload([2, 2], [1.0] * 4))
        before = [read(t) for t in owner.parameter_tensors()]
        bad = step(owner, x, device.upload([2, 1], [9.0, 9.0]))
        self.assertEqual(owner.attempted_updates, 1)
        with self.assertRaises(ValueError):
            bad[3].read()
        self.assertEqual([read(t) for t in owner.parameter_tensors()], before)
        good = step(owner, x, y)
        self.assertEqual(good[3].read(), 2)
        with self.assertRaises(ValueError):
            owner.sgd(good[2], 0.01)
        self.assertEqual(owner.attempted_updates, 2)
        checkpoint = owner.checkpoint_snapshot()
        held = good[0].prediction_tensor()
        del owner, other
        self.assertTrue(all(math.isfinite(v) for v in read(held)))
        self.assertEqual(json.loads(checkpoint.read_json())["backbone"]["attempted_updates"], 2)

    def test_invalid_config_checkpoint_and_shape_fail_explicitly(self):
        device = st.WgpuTensorDevice.create()
        for value in ("{}", config_json().replace('"epsilon": 0.001', '"epsilon": -1')):
            with self.assertRaises(ValueError):
                st.ResidentConvNeXtClassifier.create(device, value, 2, 2)
        with self.assertRaises(ValueError):
            st.ResidentConvNeXtClassifier.from_checkpoint_json(device, "{}")
        owner = st.ResidentConvNeXtClassifier.create(device, config_json(), 2, 2)
        with self.assertRaises(ValueError):
            owner.forward(device.upload([1, 1, 8, 8], [0.0] * 64))
        self.assertEqual(owner.attempted_updates, 0)
