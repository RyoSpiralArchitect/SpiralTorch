"""Freeze independent CPU-f32 causal complex-filter/chart values and VJPs."""
import argparse
import json
import math
from pathlib import Path

import torch


def flat(t):
    return t.detach().reshape(-1).tolist()


def wave(drive, decay, phase, initial, curvature):
    batch, steps, cols = drive.shape
    pairs = cols // 2
    rho = 0.99 * decay.sigmoid()
    theta = math.pi * phase.tanh()
    cos, sin = theta.cos(), theta.sin()
    state = initial.reshape(batch, pairs, 2)
    radius = torch.tensor(0.95, device="cpu", dtype=torch.float32) / torch.tensor(-curvature, device="cpu", dtype=torch.float32).sqrt()
    outputs = []
    for t in range(steps):
        x, y = state[..., 0], state[..., 1]
        rotated = torch.stack((cos * x - sin * y, sin * x + cos * y), -1)
        state = rho[None, :, None] * rotated + (1 - rho)[None, :, None] * drive[:, t].reshape(batch, pairs, 2)
        s = state.reshape(batch, cols)
        outputs.append(radius * s / (1 + s.square().sum(-1, keepdim=True)).sqrt())
    return torch.stack(outputs, 1), state.reshape(batch, cols)


def make_case(index, shape, curvature, seed_mode, zero=False, extremes=False):
    batch, steps, cols = shape
    rng = torch.Generator(device="cpu").manual_seed(209 + index * 37)
    def random(shape, scale):
        return (torch.rand(shape, device="cpu", generator=rng) - 0.5) * scale
    x = torch.zeros(shape, device="cpu") if zero else random(shape, 1.2)
    initial = torch.zeros((batch, cols), device="cpu") if zero else random((batch, cols), 0.8)
    decay = random((cols // 2,), 2.)
    phase = random((cols // 2,), 1.6)
    if extremes:
        decay = torch.linspace(-15, 15, cols // 2, device="cpu")
        phase = torch.linspace(-2, 2, cols // 2, device="cpu")
    inputs = [t.requires_grad_() for t in (x, decay, phase, initial)]
    features, final = wave(*inputs, curvature)
    seed = random(shape, 0.7) if seed_mode != "terminal" else torch.zeros_like(features)
    terminal = random((batch, cols), 0.5) if seed_mode != "features" else torch.zeros_like(final)
    gradients = torch.autograd.grad((features, final), inputs, (seed, terminal))
    return {"name": f"case{index}_{seed_mode}", "shape": shape, "curvature": curvature,
            "drive": flat(x), "raw_decay": flat(decay), "raw_phase": flat(phase), "initial_state": flat(initial),
            "feature_seed": flat(seed), "terminal_seed": flat(terminal), "features": flat(features), "final_state": flat(final),
            "gradients": dict(zip(("drive", "raw_decay", "raw_phase", "initial_state"), map(flat, gradients)))}


def learning(case):
    shape = case["shape"]
    x = torch.tensor(case["drive"], device="cpu").reshape(shape)
    initial = torch.tensor(case["initial_state"], device="cpu").reshape(shape[0], shape[2])
    decay = torch.tensor(case["raw_decay"], device="cpu", requires_grad=True)
    phase = torch.tensor(case["raw_phase"], device="cpu", requires_grad=True)
    with torch.no_grad():
        target, final_target = wave(x, decay + 0.35, phase - 0.25, initial, case["curvature"])
    trace = []
    for _ in range(16):
        features, final = wave(x, decay, phase, initial, case["curvature"])
        lf, ls = (features - target).square().mean(), (final - final_target).square().mean()
        gd, gp = torch.autograd.grad(lf + ls, (decay, phase))
        with torch.no_grad():
            decay -= 0.125 * gd
            phase -= 0.125 * gp
        trace.append({"feature_loss": lf.item(), "terminal_loss": ls.item(),
                      "raw_decay_gradient": flat(gd), "raw_phase_gradient": flat(gp),
                      "raw_decay": flat(decay), "raw_phase": flat(phase)})
    return {"case_index": 2, "steps": 16, "rate": 0.125, "target": flat(target), "terminal_target": flat(final_target), "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = [([1, 1, 2], -1., "features"), ([1, 1, 2], -1., "terminal"),
              ([2, 5, 4], -0.75, "both"), ([3, 7, 6], -1.25, "both"),
              ([2, 17, 8], -0.3, "both"), ([2, 4, 6], -2., "both"),
              ([1, 9, 4], -1., "both"), ([2, 3, 4], -0.01, "both"),
              ([2, 3, 4], -100., "both"), ([1, 31, 8], -1., "both")]
    cases = [make_case(i, s, c, m, zero=i == 5, extremes=i == 6) for i, (s, c, m) in enumerate(config)]
    payload = {"schema": "spiraltorch.causal_zspace_wave.torch_fixture.v1", "torch_version": torch.__version__,
               "device": "cpu", "dtype": "float32", "threads": 1,
               "tolerance": {"atol": 3e-6, "rtol": 8e-5}, "cases": cases, "learning": learning(cases[2])}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, separators=(",", ":"), allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
