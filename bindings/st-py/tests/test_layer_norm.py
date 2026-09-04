"""Native LayerNorm numerics and the shared backward statistics API."""

import unittest

import numpy as np
import spiraltorch as st


def tensor(values):
    values = np.asarray(values, dtype=np.float32)
    return st.Tensor(*values.shape, values.ravel().tolist())


class LayerNormTests(unittest.TestCase):
    def test_offsets_and_affine_match_f64_reference(self):
        for offset in [0.0, 1e4, 1e7, -1e7]:
            with self.subTest(offset=offset):
                values = np.array([[0, 1, 2, 3], [3, 1, 4, -2]], np.float32) + offset
                wide = values.astype(np.float64)
                centered = wide - wide.mean(1, keepdims=True)
                inv_std = 1.0 / np.sqrt(
                    np.mean(centered**2, axis=1, keepdims=True) + 1e-5
                )
                normalized, actual_inv_std = tensor(values).layer_norm_stats(
                    epsilon=1e-5
                )
                np.testing.assert_allclose(
                    normalized.tolist(), centered * inv_std, rtol=2e-6, atol=2e-6
                )
                np.testing.assert_allclose(actual_inv_std.tolist(), inv_std, rtol=2e-6)
                gamma = np.array([[1, -2, 0.5, 0]], np.float32)
                beta = np.array([[0.2, -0.1, 0.5, 1]], np.float32)
                output = tensor(values).layer_norm_affine(
                    tensor(gamma), tensor(beta), epsilon=1e-5
                )
                np.testing.assert_allclose(
                    output.tolist(),
                    centered * inv_std * gamma + beta,
                    rtol=3e-5,
                    atol=3e-5,
                )

    def test_residual_rounding_uses_the_same_normalization(self):
        values = np.full((1, 4), 1e7, dtype=np.float32)
        residual = np.array([[0, 1, 2, 3]], dtype=np.float32)
        gamma = tensor([[1, 1, 1, 1]])
        beta = tensor([[0, 0, 0, 0]])
        fused = tensor(values).layer_norm_affine_add(
            tensor(residual), gamma, beta, epsilon=1e-5
        )
        expected = tensor(values + residual).layer_norm_affine(
            gamma, beta, epsilon=1e-5
        )
        np.testing.assert_allclose(
            fused.tolist(), expected.tolist(), rtol=2e-6, atol=2e-6
        )

    def test_extreme_finite_variance_and_constant_rows(self):
        maximum = np.finfo(np.float32).max
        normalized, inverse_std = tensor([[maximum, -maximum]]).layer_norm_stats(
            epsilon=0.0
        )
        self.assertEqual(normalized.tolist(), [[1.0, -1.0]])
        self.assertGreater(inverse_std.tolist()[0][0], 0)
        tiny = float(np.nextafter(np.float32(0), np.float32(1)))
        normalized, inverse_std = tensor([[7, 7]]).layer_norm_stats(epsilon=tiny)
        self.assertEqual(normalized.tolist(), [[0.0, 0.0]])
        self.assertTrue(np.isfinite(inverse_std.tolist()).all())

    def test_empty_and_invalid_stats(self):
        normalized, inverse_std = st.Tensor(0, 4, []).layer_norm_stats()
        self.assertEqual(normalized.shape(), (0, 4))
        self.assertEqual(inverse_std.shape(), (0, 1))
        for epsilon in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                tensor([[1, 2]]).layer_norm_stats(epsilon=epsilon)
        with self.assertRaises(ValueError):
            tensor([[1, 1]]).layer_norm_stats(epsilon=0)
        with self.assertRaises(ValueError):
            tensor([[float("nan"), 1]]).layer_norm_stats()
        with self.assertRaises(ValueError):
            st.Tensor(1, 0, []).layer_norm_stats()

    def test_forward_matches_optional_pytorch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch reference is not installed")
        values = np.array([[1e7, 1e7 + 1, 1e7 + 2, 1e7 + 3]], dtype=np.float32)
        expected = torch.nn.functional.layer_norm(
            torch.tensor(values, dtype=torch.float64), (4,), eps=1e-5
        )
        actual = tensor(values).layer_norm_affine(
            tensor([[1, 1, 1, 1]]), tensor([[0, 0, 0, 0]]), epsilon=1e-5
        )
        np.testing.assert_allclose(
            actual.tolist(), expected.numpy(), rtol=3e-5, atol=3e-5
        )


if __name__ == "__main__":
    unittest.main()
