"""Freeze an independent CPU-f32 pre-norm residual attention/MLP reference."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def values(tensor):
    return tensor.detach().reshape(-1).tolist()


def packed(parameters):
    return [*map(values, parameters[:2]),
            values(torch.cat(parameters[2:8:2], dim=1)),
            values(torch.cat(parameters[3:8:2])),
            *map(values, parameters[8:])]


def forward(x, p, heads, causal, z, pair, topos):
    batch, sequence, width = x.shape
    dim = p[2].shape[1] // heads
    normalized = F.layer_norm(x, (width,), p[0], p[1], eps=1e-5)
    q, k, v = [
        (normalized @ p[i] + p[i + 1]).reshape(batch, sequence, heads, dim).transpose(1, 2)
        for i in (2, 4, 6)
    ]
    scale = torch.tensor(dim, dtype=torch.float32).sqrt().reciprocal()
    scores = (q @ k.transpose(-1, -2)) * scale
    if z is not None:
        scores = scores + z.unsqueeze(-2)
    if pair is not None:
        scores = scores + pair
    if causal:
        scores = scores.masked_fill(torch.ones(sequence, sequence, dtype=torch.bool).triu(1), -float("inf"))
    merged = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, sequence, heads * dim)
    residual = x + (merged @ p[8] + p[9])
    hidden = F.layer_norm(residual, (width,), p[10], p[11], eps=1e-5)
    hidden = F.gelu(hidden @ p[12] + p[13], approximate="tanh")
    if topos:
        drive = hidden * p[14]
        state = torch.zeros_like(drive)
        for _ in range(4):
            raw = drive + 0.2 * state
            magnitude = raw.abs()
            relative = 0.12 / magnitude.clamp_min(0.12)
            bleed = (1 - relative) / (1 + relative)
            softened = raw.sign() * (0.12 * (1 - 0.075 * bleed))
            state = torch.where(magnitude <= 0.12, raw, softened)
        hidden = state
    return residual + (hidden @ p[-2] + p[-1])


def case(shape, heads, dim, hidden, causal, mode, topos, seed):
    batch, sequence, width = shape
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def sample(dims, amplitude=0.3, shift=0.0):
        return (shift + (torch.rand(dims, generator=rng) - 0.5) * (2 * amplitude)).requires_grad_()

    x = sample(shape, 0.6)
    p = [sample((width,), 0.1, 1.), sample((width,), 0.1)]
    for k, n in [(width, heads * dim)] * 3 + [(heads * dim, width)]:
        p.extend((sample((k, n)), sample((n,))))
    p.extend((sample((width,), 0.1, 1.), sample((width,), 0.1),
              sample((width, hidden)), sample((hidden,))))
    if topos:
        p.append(sample((hidden,), 0.4, 0.8))
    p.extend((sample((hidden, width)), sample((width,))))
    upstream = sample(shape).detach()
    z = sample((batch, heads, sequence))
    pair = sample((batch, heads, sequence, sequence))
    if mode == "zero":
        z = torch.zeros_like(z, requires_grad=True)
        pair = torch.zeros_like(pair, requires_grad=True)
    else:
        z = z if mode in ("z", "both") else None
        pair = pair if mode in ("pair", "both") else None
    output = forward(x, p, heads, causal, z, pair, topos)
    gradients = torch.autograd.grad(output, [x, *p, *[b for b in (z, pair) if b is not None]], upstream)
    bias_gradients = iter(gradients[len(p) + 1:])
    result = {
        "name": f"b{batch}_t{sequence}_h{heads}_d{dim}_causal{causal}_{mode}_topos{topos}",
        "input_shape": list(shape), "heads": heads, "causal": causal, "topos": topos,
        "pre": {"gain": values(p[0]), "bias": values(p[1])},
        "projections": [
            {"weight_shape": list(p[i].shape), "weight": values(p[i]), "bias": values(p[i + 1])}
            for i in (2, 4, 6, 8)
        ],
        "feed_forward": {
            "gain": values(p[10]), "bias": values(p[11]), "hidden": hidden,
            "up_weight": values(p[12]), "up_bias": values(p[13]),
            "gate": values(p[14]) if topos else None,
            "down_weight": values(p[-2]), "down_bias": values(p[-1]),
        },
        "input": values(x), "upstream": values(upstream),
        "z_bias": None if z is None else values(z),
        "pair_bias": None if pair is None else values(pair),
        "expected": values(output), "input_gradient": values(gradients[0]),
        "parameter_gradients": packed(gradients[1:len(p) + 1]),
        "initial_parameters": packed(p),
        "z_bias_gradient": None if z is None else values(next(bias_gradients)),
        "pair_bias_gradient": None if pair is None else values(next(bias_gradients)),
    }
    return result, (x, p, z, pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a frozen fixture")
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    cases = []
    for i, (shape, heads, dim, hidden) in enumerate([
        ((2, 3, 4), 2, 2, 7), ((1, 4, 3), 3, 2, 5), ((1, 1, 4), 1, 3, 6),
    ]):
        for causal in (False, True):
            for mode in ("none", "z", "pair", "both", "zero"):
                for topos in (False, True):
                    result, _ = case(shape, heads, dim, hidden, causal, mode, topos, 1829 + i)
                    cases.append(result)
    training = []
    for topos in (False, True):
        result, (x, p, z, pair) = case((2, 3, 4), 2, 2, 7, True, "both", topos, 1837)
        target = torch.rand((2, 3, 4), generator=torch.Generator().manual_seed(902)) - 0.5
        losses = []
        rate = 0.02
        for _ in range(32):
            output = forward(x, p, 2, True, z, pair, topos)
            loss = (output - target).square().mean()
            losses.append(loss.item())
            gradients = torch.autograd.grad(loss, p)
            p = [(parameter - rate * gradient).detach().requires_grad_()
                 for parameter, gradient in zip(p, gradients)]
        result.update(target=values(target), rate=rate, losses=losses,
                      final_parameters=packed(p),
                      final_prediction=values(forward(x, p, 2, True, z, pair, topos)))
        training.append(result)
    report = {
        "schema": "spiraltorch.residual_attention_torch.v1",
        "oracle": {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1},
        "tolerance": {"atol": 3e-6, "rtol": 5e-5},
        "cases": cases, "training": training,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"cases": len(cases), "training_steps": [len(t["losses"]) for t in training]}))


if __name__ == "__main__":
    main()
