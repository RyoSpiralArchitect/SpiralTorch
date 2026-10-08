#!/usr/bin/env python3
"""Generate numerical SDPA oracles, not performance or model-quality evidence.

Run with SPIRALTON_MAGIC/TORCH/MODEL_PATCHES/NUMPY=0 before Python starts.
No models, datasets, network calls, or automatic device selection are used.
"""

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("SPIRALTON_MAGIC", "SPIRALTON_TORCH", "SPIRALTON_MODEL_PATCHES", "SPIRALTON_NUMPY"):
        if os.environ.get(key) != "0":
            parser.error(f"Set {key}=0 before starting Python to disable implicit patches")

    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.set_num_threads(1)

    def values(shape, phase):
        n = math.prod(shape)
        return torch.tensor(
            [math.sin(i * 0.17 + phase) * 0.5 for i in range(n)],
            dtype=torch.float32, device="cpu",
        ).reshape(shape)

    cases = []
    for scenario, b, h, q_len, k_len, d, offset in [
        ("rectangular", 2, 2, 3, 5, 7, None),
        ("prefill", 2, 2, 5, 5, 7, 0),
        ("cached_decode", 2, 2, 1, 5, 7, 4),
        ("cached_chunk", 2, 2, 3, 5, 7, 2),
        ("head_tail", 1, 1, 2, 3, 65, 1),
    ]:
        q = values((b, h, q_len, d), 0.1)
        k = values((b, h, k_len, d), 1.2)
        v = values((b, h, k_len, d), -0.7)
        for mode, label in enumerate(("plain", "z_bias", "pair_bias", "both_biases")):
            z = values((b, h, k_len), 0.4) if mode & 1 else None
            pair = values((b, h, q_len, k_len), -0.5) if mode & 2 else None
            bias = torch.zeros((b, h, q_len, k_len), dtype=torch.float32, device="cpu")
            if z is not None:
                bias += z.unsqueeze(-2)
            if pair is not None:
                bias += pair
            if offset is not None:
                absolute_q = torch.arange(q_len, device="cpu") + offset
                key_pos = torch.arange(k_len, device="cpu")
                bias.masked_fill_(key_pos[None, :] > absolute_q[:, None], -math.inf)
            # Explicit offset mask is essential for rectangular cached decode.
            # A fused top-left is_causal mask would evaluate a different function.
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False, scale=0.375,
                )
            assert expected.device.type == "cpu" and torch.isfinite(expected).all()
            flat = lambda tensor: None if tensor is None else tensor.flatten().tolist()
            cases.append({
                "name": f"{scenario}/{label}",
                "query_shape": list(q.shape), "key_shape": list(k.shape),
                "scale": 0.375, "query_offset": offset,
                "query": flat(q), "key": flat(k), "value": flat(v),
                "z_bias": flat(z), "pair_bias": flat(pair), "expected": flat(expected),
            })
    payload = {
        "schema": "spiraltorch.resident_attention_torch.v1",
        "torch_version": torch.__version__, "device": "cpu", "dtype": "float32",
        "reference": "torch.nn.functional.scaled_dot_product_attention / SDPBackend.MATH",
        "scope": "numerical forward parity only; no speed or learning claim",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} CPU float32 SDPA cases to {args.output}")


if __name__ == "__main__":
    main()
