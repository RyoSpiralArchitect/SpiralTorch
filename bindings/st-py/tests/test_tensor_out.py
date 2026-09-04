"""Exercise real native out= entrypoints; no facade stubs or GPU required."""
import os
import unittest

import spiraltorch as st


class TensorOutTests(unittest.TestCase):
    def setUp(self):
        self.lhs = st.Tensor(2, 2, [1, 2, 3, 4])
        self.rhs = st.Tensor(2, 2, [5, 6, 7, 8])
        self.residual = st.Tensor(2, 2, [1, 2, 3, 4])
        self.bias = [-20.0, -25.0]
        self.packed = st.cpu_simd_prepack_rhs(self.rhs)

    def operations(self):
        return {
            "matmul": lambda **kw: self.lhs.matmul(self.rhs, backend="naive", **kw),
            "packed": lambda **kw: self.lhs.matmul_simd_prepacked(self.packed, **kw),
            "relu": lambda **kw: self.lhs.matmul_bias_relu(
                self.rhs, self.bias, backend="naive", **kw),
            "gelu": lambda **kw: self.lhs.matmul_bias_gelu(
                self.rhs, self.bias, backend="naive", **kw),
            "add_relu": lambda **kw: self.lhs.matmul_bias_add_relu(
                self.rhs, self.bias, self.residual, backend="naive", **kw),
            "add_gelu": lambda **kw: self.lhs.matmul_bias_add_gelu(
                self.rhs, self.bias, self.residual, backend="naive", **kw),
        }

    def test_outputs_match_allocating_paths_and_remain_reusable(self):
        for name, operation in self.operations().items():
            with self.subTest(name=name):
                expected = operation().tolist()
                out = st.Tensor(2, 2, [-9.0] * 4)
                first = operation(out=out)
                self.assertEqual(out.tolist(), expected)
                self.assertEqual(first.tolist(), expected)
                self.assertEqual(operation(out=out).tolist(), expected)
                self.assertEqual(first.tolist(), expected)
        self.assertEqual(self.operations()["matmul"]().tolist(), [[19.0, 22.0], [43.0, 50.0]])
        self.assertEqual(self.operations()["packed"]().tolist(), [[19.0, 22.0], [43.0, 50.0]])
        self.assertEqual(self.operations()["relu"]().tolist(), [[0.0, 0.0], [23.0, 25.0]])
        self.assertEqual(self.operations()["add_relu"]().tolist(), [[0.0, 0.0], [26.0, 29.0]])

    def test_lhs_alias_rejected_without_mutation(self):
        before = self.lhs.tolist()
        for name, operation in self.operations().items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                operation(out=self.lhs)
            self.assertEqual(self.lhs.tolist(), before)

    def test_residual_alias_rejected_without_mutation(self):
        before = self.residual.tolist()
        for name in ("add_relu", "add_gelu"):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.operations()[name](out=self.residual)
            self.assertEqual(self.residual.tolist(), before)

    def test_invalid_destination_type_is_rejected(self):
        for name, operation in self.operations().items():
            with self.subTest(name=name), self.assertRaises(TypeError):
                operation(out=[])

    @unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
    def test_wgpu_output_reuse(self):
        probe = st.WgpuMatmul(1, 1, 1)
        self.assertNotEqual(probe.adapter_info()["device_type"], "Cpu")
        out = st.Tensor(2, 2, [0.0] * 4)
        for _ in range(2):
            result = self.lhs.matmul(self.rhs, backend="wgpu", out=out)
            self.assertEqual(result.tolist(), [[19.0, 22.0], [43.0, 50.0]])
            self.assertEqual(out.tolist(), result.tolist())

    def test_shape_error_releases_destination_borrow(self):
        out = st.Tensor(1, 1, [9])
        for name in ("matmul", "packed", "relu", "add_relu"):
            expected_error = ValueError if name == "packed" else RuntimeError
            with self.subTest(name=name), self.assertRaises(expected_error):
                self.operations()[name](out=out)
            self.assertEqual(out.tolist(), [[9.0]])
            st.Tensor(1, 1, [2]).matmul(st.Tensor(1, 1, [3]), backend="naive", out=out)
            self.assertEqual(out.tolist(), [[6.0]])
            out = st.Tensor(1, 1, [9])


if __name__ == "__main__":
    unittest.main()
