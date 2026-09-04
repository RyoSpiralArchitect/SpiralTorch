"""Opt-in real-device tests, without silently accepting a CPU Tensor fallback."""
import os
import unittest

import spiraltorch as st


class WgpuResidentSurface(unittest.TestCase):
    def test_surface_is_shared(self):
        self.assertIs(st.WgpuMatmul, st.wgpu.WgpuMatmul)
        if not st.wgpu_kernel_reports_available():
            with self.assertRaises(NotImplementedError):
                st.WgpuMatmul(2, 3, 4)
            with self.assertRaises(NotImplementedError):
                st.WgpuMatmul(2, 3, 4, tile_mnk=(16, 16, 32))

    @unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
    def test_reuse_and_chain(self):
        first = st.WgpuMatmul(2, 3, 2)
        self.assertEqual(first.shape, (2, 3, 2))
        self.assertEqual(first.tile_mnk, (8, 8, 16))
        for tile in ((0, 8, 16), (32, 32, 16), (8, 8, 65)):
            with self.assertRaises(ValueError):
                st.WgpuMatmul(2, 3, 2, tile_mnk=tile)
        self.assertIn("backend", first.adapter_info())
        with self.assertRaises(ValueError):
            first.readback()
        with self.assertRaises(ValueError):
            first.dispatch()
        a = st.Tensor(2, 3, [1., 2., 3., 4., 5., 6.])
        b = st.Tensor(3, 2, [1., 0., 0., 1., 1., 1.])
        first.upload(a, b)
        generation = first.dispatch()
        first.synchronize()
        self.assertEqual(first.readback().tolist(), [[4., 5.], [10., 11.]])
        with self.assertRaises(ValueError):
            first.upload(a, st.Tensor(1, 6, [0.] * 6))
        self.assertEqual(first.generation, generation)
        self.assertTrue(first.output_is_current)
        for invalid in (0, 1025):
            with self.assertRaises(ValueError):
                first.dispatch(invalid)
        next_layer = st.WgpuMatmul(2, 2, 1, tile_mnk=(16, 16, 32))
        self.assertEqual(next_layer.tile_mnk, (16, 16, 32))
        next_layer.upload_rhs(st.Tensor(2, 1, [1., 1.]))
        next_layer.set_lhs_from(first)
        next_layer.dispatch(3)
        self.assertEqual(next_layer.readback().tolist(), [[9.], [21.]])
        first.upload_rhs(st.Tensor(3, 2, [0.] * 6))
        self.assertFalse(first.output_is_current)
        with self.assertRaises(ValueError):
            next_layer.set_lhs_from(first)
        first.dispatch()
        self.assertEqual(first.readback().tolist(), [[0., 0.], [0., 0.]])


if __name__ == "__main__":
    unittest.main()
