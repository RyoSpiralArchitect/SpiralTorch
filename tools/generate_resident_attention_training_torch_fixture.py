"""Independent CPU-f32 projection/attention VJP and plain-SGD oracle."""
import argparse
import json
from pathlib import Path

import torch


def values(tensor):
    return tensor.detach().reshape(-1).tolist()


def packed(parameters):
    return [
        values(torch.cat(parameters[:6:2], dim=1)),
        values(torch.cat(parameters[1:6:2])),
        values(parameters[6]),
        values(parameters[7]),
    ]


def forward(x, parameters, heads, causal, z, pair):
    batch, sequence, _ = x.shape
    width = parameters[0].shape[1]
    dim = width // heads
    q, k, v = [
        (x @ parameters[i] + parameters[i + 1]).reshape(batch, sequence, heads, dim).transpose(1, 2)
        for i in (0, 2, 4)
    ]
    scale = torch.tensor(dim, dtype=torch.float32).sqrt().reciprocal()
    scores = (q @ k.transpose(-1, -2)) * scale
    if z is not None:
        scores = scores + z.unsqueeze(-2)
    if pair is not None:
        scores = scores + pair
    if causal:
        mask = torch.ones(sequence, sequence, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    merged = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, sequence, width)
    return merged @ parameters[6] + parameters[7]


def case(shape, heads, dim, out, causal, mode, seed):
    batch, sequence, inner = shape
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def sample(dims, amplitude=0.3):
        return ((torch.rand(dims, generator=rng) - 0.5) * (2 * amplitude)).requires_grad_()

    x = sample(shape)
    parameters = []
    for k, n in [(inner, heads * dim)] * 3 + [(heads * dim, out)]:
        parameters.extend((sample((k, n)), sample((n,))))
    upstream = sample((batch, sequence, out)).detach()
    z = sample((batch, heads, sequence))
    pair = sample((batch, heads, sequence, sequence))
    if mode == "zero":
        z = torch.zeros_like(z, requires_grad=True)
        pair = torch.zeros_like(pair, requires_grad=True)
    else:
        z = z if mode in ("z", "both") else None
        pair = pair if mode in ("pair", "both") else None
    output = forward(x, parameters, heads, causal, z, pair)
    active_biases = [bias for bias in (z, pair) if bias is not None]
    gradients = torch.autograd.grad(output, [x, *parameters, *active_biases], upstream)
    bias_gradients = iter(gradients[9:])
    result = {
        "name": f"b{batch}_t{sequence}_h{heads}_d{dim}_{causal}_{mode}",
        "input_shape": list(shape), "heads": heads, "causal": causal,
        "projections": [
            {"weight_shape": list(parameters[i].shape), "weight": values(parameters[i]), "bias": values(parameters[i+1])}
            for i in (0, 2, 4, 6)
        ],
        "input": values(x), "upstream": values(upstream),
        "z_bias": None if z is None else values(z),
        "pair_bias": None if pair is None else values(pair),
        "expected": values(output),
        "input_gradient": values(gradients[0]),
        "parameter_gradients": packed(gradients[1:9]),
        "z_bias_gradient": None if z is None else values(next(bias_gradients)),
        "pair_bias_gradient": None if pair is None else values(next(bias_gradients)),
    }
    return result, (x, parameters, z, pair)


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
    for index, (shape, heads, dim, out) in enumerate([
        ((2, 3, 4), 2, 2, 5), ((1, 4, 3), 3, 3, 4), ((1, 1, 3), 1, 2, 2),
    ]):
        for causal in (False, True):
            for mode in ("none", "z", "pair", "both", "zero"):
                result, _ = case(shape, heads, dim, out, causal, mode, 1729 + index)
                cases.append(result)
    training, (x, parameters, z, pair) = case((2, 3, 4), 2, 2, 5, True, "both", 1737)
    rng = torch.Generator().manual_seed(901)
    target = torch.rand((2, 3, 5), generator=rng) - 0.5
    losses = []
    rate = 0.03
    for _ in range(16):
        output = forward(x, parameters, 2, True, z, pair)
        loss = (output - target).square().mean()
        losses.append(loss.item())
        gradients = torch.autograd.grad(loss, parameters)
        parameters = [(p - rate * g).detach().requires_grad_() for p, g in zip(parameters, gradients)]
    training.update(target=values(target), rate=rate, losses=losses,
                    final_parameters=packed(parameters),
                    final_prediction=values(forward(x, parameters, 2, True, z, pair)))
    report = {
        "schema": "spiraltorch.resident_attention_training_torch.v1",
        "oracle": {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1},
        "cases": cases, "training": training,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"cases": len(cases), "steps": len(losses), "first_loss": losses[0], "last_pre_update_loss": losses[-1]}))


if __name__ == "__main__":
    main()
