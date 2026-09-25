import unittest

import spiraltorch as st


class VisionWgpuPipelineTests(unittest.TestCase):
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
