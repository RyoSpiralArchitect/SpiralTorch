#!/usr/bin/env python3
"""Freeze CPU PyTorch attention gradients and a bounded, synthetic SGD oracle.

No models, datasets, network calls or automatic accelerator selection. This is
correctness evidence, not a speed comparison or a language-quality experiment.
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
            parser.error(f"Set {key}=0 before starting Python")
    if args.output.exists():
        parser.error("Refusing to overwrite a frozen fixture; select a new output")

    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def values(shape, phase):
        return torch.tensor([math.sin(i * 0.17 + phase) * 0.5 for i in range(math.prod(shape))],
                            device="cpu", dtype=torch.float32).reshape(shape).requires_grad_(True)

    def tensors(b, h, q, k, d, mode):
        return [values((b, h, q, d), 0.1), values((b, h, k, d), 1.2),
                values((b, h, k, d), -0.7),
                values((b, h, k), 0.4) if mode & 1 else None,
                values((b, h, q, k), -0.5) if mode & 2 else None]

    def attention(inputs, scale, offset):
        q, k, v, z, pair = inputs
        scores = (q @ k.transpose(-1, -2)) * scale
        if z is not None:
            scores = scores + z.unsqueeze(-2)
        if pair is not None:
            scores = scores + pair
        if offset is not None:
            future = torch.arange(k.shape[-2])[None, :] > torch.arange(q.shape[-2])[:, None] + offset
            scores = scores.masked_fill(future, -math.inf)
        return scores.softmax(-1) @ v

    def flat(tensor):
        return None if tensor is None else tensor.detach().flatten().tolist()

    names = ("query", "key", "value", "z_bias", "pair_bias")

    def payload(inputs, scale, offset):
        return {"query_shape": list(inputs[0].shape), "key_shape": list(inputs[1].shape),
                "scale": scale, "query_offset": offset,
                **{name: flat(t) for name, t in zip(names, inputs)}}

    cases = []
    configurations = [(f"mask_{offset}_bias_{mode}", 2, 2, 3, 5, 7, mode, 0.375, offset)
                      for offset in (None, 0, 2) for mode in range(4)]
    configurations += [
        ("head_tail", 1, 1, 2, 3, 65, 3, 0.125, 1),
        ("max_head", 1, 1, 1, 2, 256, 0, 0.0625, None),
        ("key_tile_tail", 1, 1, 2, 131, 17, 3, 0.25, 129),
        ("zero_scale", 1, 2, 3, 5, 7, 3, 0.0, None),
        ("negative_scale", 1, 2, 3, 5, 7, 3, -0.25, 0),
        ("empty_query", 2, 2, 0, 3, 7, 3, 0.375, None),
    ]
    for name, b, h, q, k, d, mode, scale, offset in configurations:
        inputs = tensors(b, h, q, k, d, mode)
        upstream = values(inputs[0].shape, 2.1).detach()
        output = attention(inputs, scale, offset)
        (output * upstream).sum().backward()
        gradients = {n: flat(t.grad) if t is not None else None for n, t in zip(names, inputs)}
        assert all(t is None or (t.grad is not None and torch.isfinite(t.grad).all()) for t in inputs)
        cases.append({"name": name, **payload(inputs, scale, offset),
                      "upstream": flat(upstream), "expected": flat(output), "gradients": gradients})

    inputs = tensors(1, 2, 3, 5, 7, 3)
    target = values(inputs[0].shape, -2.3).detach()
    learning = {**payload(inputs, 0.375, 2), "target": flat(target), "learning_rate": 0.08,
                "steps": 16, "loss": "mean squared error", "trace": []}
    optimizer = torch.optim.SGD(inputs, lr=learning["learning_rate"])
    for step in range(learning["steps"]):
        optimizer.zero_grad(set_to_none=True)
        output = attention(inputs, learning["scale"], learning["query_offset"])
        loss = (output - target).square().mean()
        loss.backward()
        learning["trace"].append({"step": step, "loss": loss.item(), "output": flat(output)})
        optimizer.step()
    learning["final"] = {name: flat(t) for name, t in zip(names, inputs)}
    assert learning["trace"][-1]["loss"] < learning["trace"][0]["loss"]
    for name, tensor in zip(names, inputs):
        assert flat(tensor) != learning[name], f"{name} did not participate in updates"
    document = {"schema": "spiraltorch.resident_attention_vjp_torch.v1",
                "torch_version": torch.__version__, "device": "cpu", "dtype": "float32",
                "reference": "torch matmul + structural mask + softmax + autograd; torch.optim.SGD",
                "scope": "synthetic first-order gradient/update parity, not quality or speed",
                "cases": cases, "learning": learning}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(cases), "steps": learning["steps"],
                      "initial_loss": learning["trace"][0]["loss"],
                      "last_pre_update_loss": learning["trace"][-1]["loss"]}))


if __name__ == "__main__":
    main()
