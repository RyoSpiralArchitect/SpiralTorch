"""Freeze a complete, independently evaluated CPU-f32 byte decoder oracle."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def values(tensor):
    return tensor.detach().reshape(-1).tolist()


def block_forward(x, p, heads, z, pair, topos):
    batch, steps, width = x.shape
    y = F.layer_norm(x, (width,), p[0], p[1], eps=1e-5)
    qkv = (y @ p[2] + p[3]).reshape(batch, steps, 3, heads, width // heads)
    q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
    scale = torch.tensor(width // heads, dtype=torch.float32, device="cpu").sqrt().reciprocal()
    scores = (q @ k.transpose(-1, -2)) * scale
    if z is not None:
        scores = scores + z.unsqueeze(-2) + pair
    mask = torch.ones(steps, steps, dtype=torch.bool, device="cpu").triu(1)
    scores = scores.masked_fill(mask, -float("inf"))
    attention = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, steps, width)
    residual = x + attention @ p[4] + p[5]
    y = F.layer_norm(residual, (width,), p[6], p[7], eps=1e-5)
    y = F.gelu(y @ p[8] + p[9], approximate="tanh")
    if topos:
        drive = y * p[10]
        state = torch.zeros_like(drive)
        for _ in range(4):
            raw = drive + 0.2 * state
            magnitude = raw.abs()
            relative = 0.12 / magnitude.clamp_min(0.12)
            bleed = (1 - relative) / (1 + relative)
            softened = raw.sign() * (0.12 * (1 - 0.075 * bleed))
            state = torch.where(magnitude <= 0.12, raw, softened)
        y = state
    return residual + (y @ p[-2] + p[-1])


def forward(parameters, ids, config, biases):
    batch, steps = ids.shape
    positions = torch.arange(steps, device="cpu", dtype=torch.long).expand(batch, steps)
    x = F.embedding(ids, parameters[0]) + F.embedding(positions, parameters[1])
    embedded = x
    offset = 2
    for block, (z, pair) in zip(config["blocks"], biases):
        count = 13 if block["topos"] else 12
        x = block_forward(x, parameters[offset:offset + count], config["heads"], z, pair, block["topos"])
        offset += count
    head = parameters[offset:]
    x = F.layer_norm(x, (config["width"],), head[0], head[1], eps=1e-5)
    return x @ head[2] + head[3], embedded


def case(block_count, geometry, seed):
    width, heads, hidden, steps, batch = 4, 2, 6, 4, 2
    config = {"width": width, "heads": heads, "hidden": hidden, "steps": steps,
              "batch": batch, "position_capacity": steps + 2,
              "blocks": [{"topos": i == 1} for i in range(block_count)]}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    names, parameters = [], []

    def add(name, shape, amplitude=0.12, shift=0.):
        names.append(name)
        parameters.append((shift + (torch.rand(shape, generator=rng, device="cpu") - 0.5) * 2 * amplitude).requires_grad_())

    add("token_embedding", (256, width), 0.4)
    add("position_embedding", (steps + 2, width), 0.08)
    for i, block in enumerate(config["blocks"]):
        prefix = f"block.{i}."
        add(prefix + "pre.gain", (width,), 0.05, 1.)
        add(prefix + "pre.bias", (width,), 0.04)
        add(prefix + "qkv.weight", (width, 3 * width))
        add(prefix + "qkv.bias", (3 * width,), 0.04)
        add(prefix + "output.weight", (width, width))
        add(prefix + "output.bias", (width,), 0.04)
        add(prefix + "feed.gain", (width,), 0.05, 1.)
        add(prefix + "feed.bias", (width,), 0.04)
        add(prefix + "feed.up_weight", (width, hidden))
        add(prefix + "feed.up_bias", (hidden,), 0.04)
        if block["topos"]:
            add(prefix + "feed.topos_gate", (hidden,), 0.15, 0.8)
        add(prefix + "feed.down_weight", (hidden, width))
        add(prefix + "feed.down_bias", (width,), 0.04)
    add("head.gain", (width,), 0.05, 1.)
    add("head.bias", (width,), 0.04)
    add("head.weight", (width, 256), 0.12)
    add("head.output_bias", (256,), 0.04)
    initial = [{"name": name, "shape": list(p.shape), "values": values(p)} for name, p in zip(names, parameters)]
    # Invalid UTF-8 and zero remain ordinary bytes. Each row is a separate window.
    windows = [[97, 98, 97, 0, 255], [120, 121, 120, 128, 122]]
    inputs = torch.tensor([w[:-1] for w in windows], device="cpu", dtype=torch.long)
    target = torch.tensor([w[1:] for w in windows], device="cpu", dtype=torch.long)
    biases = []
    for i in range(block_count):
        if geometry:
            z = (torch.arange(steps, device="cpu", dtype=torch.float32) * (i + 1) * 0.04)
            z = z.expand(batch, heads, steps).clone().requires_grad_()
            pos = torch.arange(steps, device="cpu", dtype=torch.float32)
            pair = (-(pos[:, None] - pos[None, :]).square() * 0.03)
            pair = pair.expand(batch, heads, steps, steps).clone().requires_grad_()
            biases.append((z, pair))
        else:
            biases.append((None, None))
    logits, embedded = forward(parameters, inputs, config, biases)
    seed_tensor = (torch.rand(logits.shape, generator=rng, device="cpu") - 0.5) * 0.2
    extras = [b for group in biases for b in group if b is not None]
    gradients = torch.autograd.grad(logits, [*parameters, embedded, *extras], seed_tensor)
    result = {"name": f"blocks{block_count}_geometry{geometry}", "config": config,
              "parameters": initial, "windows": windows, "inputs": values(inputs), "targets": values(target),
              "biases": [{"z": None if z is None else values(z), "pair": None if pair is None else values(pair)} for z, pair in biases],
              "cotangent": values(seed_tensor), "output": values(logits),
              "parameter_gradients": [values(g) for g in gradients[:len(parameters)]],
              "embedding_output_gradient": values(gradients[len(parameters)]),
              "bias_gradients": [values(g) for g in gradients[len(parameters) + 1:]]}
    trace = []
    for _ in range(16):
        logits, _ = forward(parameters, inputs, config, biases)
        loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
        gradient = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for p, g in zip(parameters, gradient):
                p -= 0.125 * g
        trace.append({"loss": loss.item(), "parameters": [values(p) for p in parameters]})
    result["learning"] = {"steps": 16, "rate": 0.125, "trace": trace}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = {"schema": "spiraltorch.resident_byte_decoder.torch_fixture.v1",
               "torch_version": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1,
               "tolerance": {"atol": 3e-6, "rtol": 5e-5}, "cases": [case(1, False, 761), case(2, True, 863)]}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(payload, output, separators=(",", ":"), allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
