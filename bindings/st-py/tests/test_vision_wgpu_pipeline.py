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

        for frame in range(3):
            values = [((i * 37 + frame * 13) % 257) / 256 for i in range(3 * 12 * 14)]
            image = st.ImageTensor(3, 12, 14, values)
            expected = cpu.apply(image)
            actual = gpu.apply(image)
            self.assertEqual(actual.shape(), expected.shape())
            for left, right in zip(actual.flatten(), expected.flatten()):
                self.assertAlmostEqual(left, right, delta=1e-5)

        gpu.disable_wgpu()
        self.assertFalse(gpu.has_gpu_dispatcher())
