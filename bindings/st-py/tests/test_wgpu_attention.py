"""Resident attention clients use the frozen independent PyTorch oracle."""
import gc
import json
import os
from pathlib import Path
import unittest

import spiraltorch as st


class AttentionSurface(unittest.TestCase):
    def test_gradient_alias_and_opaque_constructor(self):
        from spiraltorch.wgpu import WgpuAttentionGradients
        self.assertIs(st.WgpuAttentionGradients, WgpuAttentionGradients)
        self.assertIn("WgpuAttentionGradients", st.__all__)
        self.assertIn("WgpuAttentionGradients", st.wgpu.__all__)
        with self.assertRaises(TypeError):
            WgpuAttentionGradients()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class AttentionGpu(unittest.TestCase):
    def close(self, tensor, expected):
        actual = tensor.snapshot().read_values()
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertLessEqual(abs(a - e), 3e-6 + 5e-5 * abs(e))

    def test_all_bias_modes_and_logical_gradients(self):
        root = Path(__file__).resolve().parents[3]
        fixture = json.loads((root / "crates/st-backend-wgpu/tests/fixtures/resident_attention_vjp_torch.json").read_text())
        device = st.WgpuTensorDevice.create()
        self.assertNotEqual(device.adapter_info()["device_type"], "Cpu")
        self.assertEqual(len(fixture["cases"]), 18)
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                qs, ks = case["query_shape"], case["key_shape"]
                q, k, v, u = [device.upload(shape, case[name]) for name, shape in
                              (("query", qs), ("key", ks), ("value", ks), ("upstream", qs))]
                options = {"causal_offset": case["query_offset"]}
                for name, shape in (("z_bias", ks[:3]), ("pair_bias", qs[:3] + [ks[2]])):
                    if case[name] is not None:
                        options[name] = device.upload(shape, case[name])
                output = q.scaled_dot_attention(k, v, case["scale"], **options)
                gradients = q.scaled_dot_attention_vjp(k, v, u, case["scale"], **options)
                del q, k, v, u, options
                gc.collect()
                self.close(output, case["expected"])
                for name, expected in case["gradients"].items():
                    if expected is None:
                        self.assertIsNone(getattr(gradients, name))
                    else:
                        tensor = getattr(gradients, name)
                        self.close(tensor, expected)
                retained = gradients.query
                del gradients
                gc.collect()
                self.close(retained, case["gradients"]["query"])

    def test_offsets_are_strict_and_upstream_shape_is_checked(self):
        device = st.WgpuTensorDevice.create()
        q = device.upload([1, 1, 1, 2], [0.2, 0.3])
        for bad in (True, -1, 0.5, "0", 2**65):
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                q.scaled_dot_attention(q, q, 1., causal_offset=bad)
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                q.scaled_dot_attention_vjp(q, q, q, 1., causal_offset=bad)
        with self.assertRaises(ValueError):
            q.scaled_dot_attention_vjp(q, q, device.upload([2], [1., 1.]), 1.)
        with self.assertRaises(ValueError):
            q.scaled_dot_attention(q, q, float("nan"))


if __name__ == "__main__":
    unittest.main()
