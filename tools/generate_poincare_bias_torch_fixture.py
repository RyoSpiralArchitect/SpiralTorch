"""Freeze independent causal Poincare pair-bias values and autograd VJPs."""
import argparse
import json
from pathlib import Path

import torch


def flat(value):
    return value.detach().reshape(-1).tolist()


def metric(points, raw_gain, curvature):
    c = -curvature
    margin = 1 - c * points.square().sum(-1)
    if not bool((margin > 0).all()):
        raise ValueError("outside open ball")
    square = (points[:, :, None, :] - points[:, None, :, :]).square().sum(-1)
    v = c * square / (margin[:, :, None] * margin[:, None, :])
    # The analytic extension at coincidence avoids sqrt's undefined VJP.
    safe = torch.where(v == 0, torch.ones_like(v), v)
    distance = torch.where(v == 0, 4 * v / c, 4 * safe.sqrt().asinh().square() / c)
    mask = torch.ones(points.shape[1:2] * 2, dtype=torch.bool, device="cpu").tril()
    return torch.where(mask[None, None], -torch.nn.functional.softplus(raw_gain)[None, :, None, None] * distance[:, None], 0.)


def case(index, shape, heads, curvature, mode):
    rng = torch.Generator(device="cpu").manual_seed(2709 + 53 * index)
    points = (torch.rand(shape, generator=rng, device="cpu") - .5) * .7
    points = .9 * points / (1 + points.square().sum(-1, keepdim=True)).sqrt() / (-curvature) ** .5
    if mode == "coincident":
        points[:, 1] = points[:, 0]
    elif mode == "near":
        points[:, 1] = points[:, 0] + 1e-6
    elif mode == "boundary":
        points *= 0
        points[..., 0] = torch.linspace(.91, .99, shape[1], device="cpu") / (-curvature) ** .5
    raw = torch.linspace(-6., 2., heads, device="cpu") if heads > 1 else torch.tensor([.3], device="cpu")
    points.requires_grad_()
    raw.requires_grad_()
    scores = metric(points, raw, curvature)
    seed = (torch.rand(scores.shape, generator=rng, device="cpu") - .5) * .5
    gp, gg = torch.autograd.grad(scores, (points, raw), seed)
    return dict(name=f"case{index}_{mode}", shape=shape, heads=heads, curvature=curvature,
                coordinates=flat(points), raw_gain=flat(raw), seed=flat(seed), scores=flat(scores),
                coordinates_vjp=flat(gp), raw_gain_vjp=flat(gg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    config = [([1, 1, 1], 1, -1., "single"), ([2, 3, 4], 2, -.75, "ordinary"),
              ([1, 5, 3], 3, -.01, "ordinary"), ([2, 7, 2], 2, -100., "ordinary"),
              ([2, 4, 4], 3, -1., "coincident"), ([1, 3, 2], 2, -1., "near"),
              ([1, 5, 2], 2, -1., "boundary"), ([1, 17, 6], 4, -.3, "ordinary")]
    cases = [case(i, *c) for i, c in enumerate(config)]
    payload = dict(schema="spiraltorch.poincare_bias.torch_fixture.v1", torch_version=torch.__version__,
                   device="cpu", dtype="float32", threads=1, tolerance=dict(atol=3e-6, rtol=8e-5), cases=cases)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, separators=(",", ":"), allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
