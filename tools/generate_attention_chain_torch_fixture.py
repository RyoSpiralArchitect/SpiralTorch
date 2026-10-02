#!/usr/bin/env python3
"""Independent CPU reference for frozen Linear -> Z-RBF attention -> Linear.

The Z-RBF formula here is a test oracle, not a second runtime implementation.
Disable optional global Spiralton patches before launching Python.
"""

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("parity", "benchmark"), default="parity")
    args = parser.parse_args()
    for key in ("SPIRALTON_MAGIC", "SPIRALTON_TORCH", "SPIRALTON_MODEL_PATCHES", "SPIRALTON_NUMPY"):
        if os.environ.get(key) != "0":
            parser.error(f"Set {key}=0 before starting Python")

    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.set_num_threads(1)

    def data(shape, phase, amplitude=0.2):
        return torch.tensor([math.sin(i * 0.17 + phase) * amplitude for i in range(math.prod(shape))],
                            dtype=torch.float32, device="cpu").reshape(shape)

    def flat(t):
        return t.flatten().tolist()

    scenarios = []
    shapes = [
        ("small_tail", 2, 5, 6, 6, 2, 4),
        ("transformer_width", 1, 17, 32, 32, 4, 32),
    ] if args.suite == "parity" else [
        ("short_prefill", 1, 32, 64, 64, 4, 64),
        ("batched_prefill", 2, 128, 128, 128, 4, 128),
        ("wide_prefill", 1, 256, 256, 256, 8, 256),
    ]
    for name, b, t, inner, width, heads, out in shapes:
        x = data((b, t, inner), 0.2, 0.5)
        weights = [data((inner if i < 3 else width, width if i < 3 else out), i * 0.7) for i in range(4)]
        biases = [data((width if i < 3 else out,), i * 0.3, 0.05) for i in range(4)]
        qkv = (x @ torch.cat(weights[:3], dim=1) + torch.cat(biases[:3])).reshape(b, t, 3, heads, width // heads)
        q, k, v = [qkv[:, :, i].permute(0, 2, 1, 3).contiguous() for i in range(3)]
        indices = [[i % 3, (i // 2) % 2, i % 8] for i in range(t)]
        coordinates = torch.tensor(indices, dtype=torch.float32, device="cpu")
        distances = (coordinates[:, None, :] - coordinates[None, :, :]).abs()
        distances[:, :, 2] = torch.minimum(distances[:, :, 2], 8 - distances[:, :, 2])
        metric = torch.tensor([1., 0.7, 0.5], dtype=torch.float32, device="cpu")
        metric = metric / metric.sum()
        kernels = []
        for head in range(heads):
            ard_scale = torch.tensor(1. + 0.1 * head, dtype=torch.float32, device="cpu")
            lengths = torch.tensor([1., 0.8, 0.6], dtype=torch.float32, device="cpu") * ard_scale
            parts = torch.exp(-0.5 * ((distances * metric) / lengths).square())
            kernels.append(parts[:, :, 0] * parts[:, :, 1] * parts[:, :, 2])
        kernel = torch.stack(kernels)
        cases = []
        for causal in (False, True):
            for mode, strength in (("plain", None), ("zero_geometry", 0.0), ("zrbf", 1.0)):
                score_bias = torch.zeros((1, heads, t, t), dtype=torch.float32, device="cpu")
                if strength is not None:
                    score_bias += kernel[None, :] * strength
                if causal:
                    score_bias.masked_fill_(torch.arange(t)[None, :] > torch.arange(t)[:, None], -math.inf)
                with sdpa_kernel(SDPBackend.MATH):
                    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=score_bias,
                        dropout_p=0.0, is_causal=False, scale=1. / math.sqrt(width // heads))
                result = attended.permute(0, 2, 1, 3).reshape(b, t, width) @ weights[3] + biases[3]
                assert torch.isfinite(result).all()
                cases.append({"name": ("causal/" if causal else "unmasked/") + mode,
                              "causal": causal, "geometry_strength": strength, "expected": flat(result)})
        scenarios.append({"name": name, "input_shape": [b, t, inner], "heads": heads, "width": width,
                          "output_width": out, "input": flat(x), "weights": [flat(w) for w in weights],
                          "biases": [flat(bias) for bias in biases], "indices": indices,
                          "frame_shape": [3, 2, 8], "expected_kernel": flat(kernel), "cases": cases})
    result = {"schema": "spiraltorch.attention_chain_torch.v1", "torch_version": torch.__version__,
              "device": "cpu", "dtype": "float32", "scenarios": scenarios}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {sum(len(s['cases']) for s in scenarios)} complete-chain controls")


if __name__ == "__main__":
    main()
