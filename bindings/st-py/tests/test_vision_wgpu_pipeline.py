import unittest

import spiraltorch as st


class VisionWgpuPipelineTests(unittest.TestCase):
    def test_normalized_resident_loader_and_continuation(self):
        cpu = st.TransformPipeline(seed=271)
        gpu = st.TransformPipeline(seed=271)
        for pipeline in (cpu, gpu):
            pipeline.add_normalize([0.25, 0.5], [0.5, 2.0])
            pipeline.add_resize(8, 10)
            pipeline.add_horizontal_flip(0.5)
            pipeline.add_center_crop(6, 6)
            pipeline.add_normalize([0.125], [0.75])
        try:
            gpu.enable_wgpu()
            device = st.WgpuTensorDevice.create()
        except (RuntimeError, NotImplementedError) as exc:
            if "adapter" in str(exc).lower() or "wgpu" in str(exc).lower():
                self.skipTest(str(exc))
            raise
        dataset = st.TensorVisionDataset("CIFAR10")
        for n in range(3):
            image = st.ImageTensor(2, 9, 11, [((i * 31 + n * 17) % 257) / 256 for i in range(198)])
            dataset.push(image, target=st.Tensor(1, 1, [float(n)]), label=str(n))
        host = dataset.dataloader(2, seed=19, pipeline=cpu)
        resident = dataset.dataloader(2, seed=19, pipeline=gpu)
        self.assertIs(st.vision.ResidentVisionBatch, st.ResidentVisionBatch)
        for count in (2, 1):
            reference = host.next_batch()
            batch = resident.next_resident_batch(device)
            self.assertIsInstance(batch, st.ResidentVisionBatch)
            self.assertEqual(len(batch), count)
            self.assertEqual(batch.images().shape, (count, 2, 6, 6))
            self.assertEqual(batch.labels(), reference.labels())
            expected = [value for image in reference.images() for value in image.flatten()]
            actual = batch.images().snapshot().read_values()
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, delta=1e-5)
            self.assertEqual(batch.upload_targets().shape, (count, 1))
        self.assertIsNone(resident.next_resident_batch(device))
        followup = st.TransformPipeline(seed=1)
        followup.add_normalize([0.5], [0.25])
        followup.enable_wgpu()
        data = device.upload([1, 1, 2, 2], [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(followup.apply_from_resident(data).snapshot().read_values(), [-2.0, -1.0, 0.0, 2.0])

    def test_normalization_contract_and_inherited_guard(self):
        for bad in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0):
            with self.assertRaises(Exception):
                st.TransformPipeline().add_normalize([0.0], [bad])
        pipeline = st.TransformPipeline(seed=2)
        pipeline.add_normalize([0.0], [0.5])
        pipeline.add_center_crop(1, 1)
        try:
            pipeline.enable_wgpu()
            device = st.WgpuTensorDevice.create()
        except (RuntimeError, NotImplementedError) as exc:
            if "adapter" in str(exc).lower() or "wgpu" in str(exc).lower():
                self.skipTest(str(exc))
            raise
        data = [3.4028234663852886e38] + [1.0] * 8
        output = pipeline.apply_resident(st.ImageTensor(1, 3, 3, data), device)
        with self.assertRaisesRegex(Exception, "non-finite"):
            output.snapshot().read_values()

    def test_opt_in_geometry_matches_cpu_and_can_be_disabled(self):
        cpu = st.TransformPipeline(seed=17)
        gpu = st.TransformPipeline(seed=17)
        for pipeline in (cpu, gpu):
            pipeline.add_resize(8, 10)
            pipeline.add_horizontal_flip(0.5)
            pipeline.add_center_crop(6, 6)
            pipeline.add_horizontal_flip(0.0)
            pipeline.add_horizontal_flip(1.0)

        self.assertFalse(gpu.has_gpu_dispatcher())
        try:
            gpu.enable_wgpu()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "adapter" in message or "not available in this" in message:
                self.skipTest(str(exc))
            raise
        self.assertTrue(gpu.has_gpu_dispatcher())

        for frame in range(12):
            values = [((i * 37 + frame * 13) % 257) / 256 for i in range(3 * 12 * 14)]
            image = st.ImageTensor(3, 12, 14, values)
            expected = cpu.apply(image)
            actual = gpu.apply(image)
            self.assertEqual(actual.shape(), expected.shape())
            for left, right in zip(actual.flatten(), expected.flatten()):
                self.assertAlmostEqual(left, right, delta=1e-5)

        gpu.disable_wgpu()
        self.assertFalse(gpu.has_gpu_dispatcher())

    def test_invalid_crop_does_not_consume_flip_seed(self):
        cpu = st.TransformPipeline(seed=19)
        gpu = st.TransformPipeline(seed=19)
        fresh = st.TransformPipeline(seed=19)
        for pipeline in (cpu, gpu, fresh):
            pipeline.add_horizontal_flip(0.5)
            pipeline.add_center_crop(13, 13)

        try:
            gpu.enable_wgpu()
        except RuntimeError as exc:
            message = str(exc).lower()
            if "adapter" in message or "not available in this" in message:
                self.skipTest(str(exc))
            raise

        bad = st.ImageTensor(3, 12, 14, [0.25] * (3 * 12 * 14))
        for pipeline in (cpu, gpu):
            with self.assertRaisesRegex(Exception, "center_crop_size"):
                pipeline.apply(bad)

        values = [((i * 17 + 3) % 257) / 256 for i in range(3 * 14 * 14)]
        valid = st.ImageTensor(3, 14, 14, values)
        expected = fresh.apply(valid)
        for pipeline in (cpu, gpu):
            actual = pipeline.apply(valid)
            self.assertEqual(actual.shape(), expected.shape())
            for left, right in zip(actual.flatten(), expected.flatten()):
                self.assertAlmostEqual(left, right, delta=1e-5)

    def test_resident_geometry_flows_into_nn_without_intermediate_readback(self):
        cpu = st.TransformPipeline(seed=31)
        gpu = st.TransformPipeline(seed=31)
        for pipeline in (cpu, gpu):
            pipeline.add_horizontal_flip(0.5)
            pipeline.add_center_crop(6, 6)
        try:
            gpu.enable_wgpu()
            device = st.WgpuTensorDevice.create()
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if "adapter" in message or "not available" in message or "wgpu" in message:
                self.skipTest(str(exc))
            raise

        bad = st.ImageTensor(1, 5, 7, [0.25] * 35)
        with self.assertRaisesRegex(Exception, "center_crop_size"):
            gpu.apply_resident(bad, device)

        values = [i / 63 for i in range(64)]
        image = st.ImageTensor(1, 8, 8, values)
        expected = cpu.apply(image).flatten()
        resident = gpu.apply_resident(image, device)
        self.assertEqual(resident.shape, (1, 6, 6))
        flat = resident.reshape([1, 36])
        self.assertTrue(flat.shares_storage_with(resident))
        net = st.nn.Sequential()
        net.add(st.nn.Scaler.from_gain("vision_gain", st.Tensor(1, 36, [2.0] * 36)))
        actual = net(flat).snapshot().read_values()
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right * 2, delta=1e-5)

        with self.assertRaisesRegex(Exception, "non-finite"):
            gpu.apply_resident(st.ImageTensor(1, 8, 8, [float("nan")] * 64), device)

    def test_resident_batch_matches_sequential_cpu_and_nn(self):
        cpu = st.TransformPipeline(seed=271)
        gpu = st.TransformPipeline(seed=271)
        for pipeline in (cpu, gpu):
            pipeline.add_resize(8, 10)
            pipeline.add_horizontal_flip(0.5)
            pipeline.add_center_crop(6, 6)
            pipeline.add_horizontal_flip(0.5)
        try:
            gpu.enable_wgpu()
            device = st.WgpuTensorDevice.create()
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if "adapter" in message or "not available" in message or "wgpu" in message:
                self.skipTest(str(exc))
            raise

        images = [
            st.ImageTensor(2, 9, 11, [((i * 31 + frame * 17) % 257) / 256 for i in range(198)])
            for frame in range(5)
        ]
        with self.assertRaises(Exception):
            gpu.apply_resident_batch([images[0], st.ImageTensor.zeros(2, 8, 11)], device)
        expected = [value for image in images for value in cpu.apply(image).flatten()]
        resident = gpu.apply_resident_batch(images, device)
        self.assertEqual(resident.shape, (5, 2, 6, 6))
        flat = resident.reshape([5, 72])
        self.assertTrue(flat.shares_storage_with(resident))
        net = st.nn.Sequential()
        net.add(st.nn.Scaler.from_gain("vision_batch_gain", st.Tensor(1, 72, [2.0] * 72)))
        actual = net(flat).snapshot().read_values()
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right * 2, delta=1e-5)

        with self.assertRaisesRegex(Exception, "non-finite"):
            gpu.apply_resident_batch([st.ImageTensor(2, 9, 11, [float("nan")] * 198)], device)

    def test_resident_batch_feeds_depthwise_without_intermediate_readback(self):
        cpu = st.TransformPipeline(seed=77)
        gpu = st.TransformPipeline(seed=77)
        for pipeline in (cpu, gpu):
            pipeline.add_horizontal_flip(0.5)
            pipeline.add_center_crop(4, 4)
        try:
            gpu.enable_wgpu()
            device = st.WgpuTensorDevice.create()
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if "adapter" in message or "not available" in message or "wgpu" in message:
                self.skipTest(str(exc))
            raise

        images = [
            st.ImageTensor(2, 6, 6, [((i * 7 + frame * 11) % 53) / 52 for i in range(72)])
            for frame in range(2)
        ]
        expected = []
        for image in images:
            values = cpu.apply(image).flatten()
            expected.extend(max(value * [2.0, -1.0][i // 16] + [0.1, 0.2][i // 16], 0.0)
                            for i, value in enumerate(values))
        transformed = gpu.apply_resident_batch(images, device)
        kernel = [0.0] * 18
        kernel[4], kernel[13] = 2.0, -1.0
        weights = device.upload([2, 3, 3], kernel)
        bias = device.upload([2], [0.1, 0.2])
        output = transformed.depthwise_conv2d(weights, bias, [1, 1], [1, 1], [1, 1]).relu()
        self.assertEqual(output.shape, (2, 2, 4, 4))
        actual = output.snapshot().read_values()
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=1e-5)
        with self.assertRaisesRegex(ValueError, "shape"):
            transformed.depthwise_conv2d(weights, device.upload([1], [0.0]), [1, 1], [1, 1], [1, 1])

    def test_resident_batch_feeds_dense_conv2d_without_intermediate_readback(self):
        cpu = st.TransformPipeline(seed=91)
        gpu = st.TransformPipeline(seed=91)
        for pipeline in (cpu, gpu):
            pipeline.add_center_crop(4, 4)
        try:
            gpu.enable_wgpu()
            device = st.WgpuTensorDevice.create()
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if "adapter" in message or "not available" in message or "wgpu" in message:
                self.skipTest(str(exc))
            raise

        images = [
            st.ImageTensor(2, 6, 6, [((i * 7 + frame * 11) % 53) / 52 for i in range(72)])
            for frame in range(2)
        ]
        transformed = gpu.apply_resident_batch(images, device)
        weight_values = [1.0, 0.0, 0.0, 1.0, 0.5, -0.25]
        bias_values = [0.1, -0.1, 0.2]
        weights = device.upload([3, 2, 1, 1], weight_values)
        bias = device.upload([3], bias_values)
        output = transformed.conv2d(weights, bias, [1, 1], [0, 0], [1, 1]).relu()
        self.assertEqual(output.shape, (2, 3, 4, 4))

        expected = []
        for image in images:
            values = cpu.apply(image).flatten()
            for output_channel in range(3):
                for pixel in range(16):
                    result = bias_values[output_channel]
                    for input_channel in range(2):
                        result += values[input_channel * 16 + pixel] * weight_values[
                            output_channel * 2 + input_channel
                        ]
                    expected.append(max(result, 0.0))
        actual_values = output.snapshot().read_values()
        self.assertEqual(len(actual_values), len(expected))
        for actual, reference in zip(actual_values, expected):
            self.assertAlmostEqual(actual, reference, delta=1e-5)

        with self.assertRaisesRegex(ValueError, "weights must be"):
            transformed.conv2d(device.upload([2, 1, 1], [1.0, 1.0]), bias, [1, 1], [0, 0], [1, 1])
        with self.assertRaisesRegex(ValueError, "bias mismatch"):
            transformed.conv2d(weights, device.upload([1], [0.0]), [1, 1], [0, 0], [1, 1])
        with self.assertRaises(TypeError):
            transformed.conv2d(weights, bias, [True, 1], [0, 0], [1, 1])

        invalid = device.upload([1, 1, 2, 2], [0.0, 0.0, 0.0, 3.4028234663852886e38])
        invalid = invalid.mul(device.upload([], [2.0]))
        with self.assertRaisesRegex(Exception, "non-finite"):
            invalid.conv2d(
                device.upload([1, 1, 1, 1], [1.0]),
                device.upload([1], [0.0]), [2, 2], [0, 0], [1, 1]
            ).snapshot().read_values()
