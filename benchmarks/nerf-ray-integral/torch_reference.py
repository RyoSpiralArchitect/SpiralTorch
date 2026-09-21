"""Independent eager CPU PyTorch rendering and one-step autograd control.

Consumes a native fixture report. All tensor payloads stay in the local report;
the public evidence publisher can retain their hashes and validation metrics.
"""
import json
import platform
import sys
import time

import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def setup(meta, gradients=False):
    params = {
        name: torch.tensor(p["values"], dtype=torch.float32).reshape(p["shape"])
        .requires_grad_(gradients)
        for name, p in meta["parameters"].items()
    }
    batch, samples = meta["batch"], meta["samples"]
    origins = torch.tensor(meta["origins"], dtype=torch.float32).reshape(batch, 3)
    directions = torch.tensor(meta["directions"], dtype=torch.float32).reshape(batch, 3)
    bounds = torch.tensor(meta["bounds"], dtype=torch.float32).reshape(batch, 2).double()
    targets = torch.tensor(meta["targets"], dtype=torch.float32).reshape(batch, 3)
    frequencies = {
        bands: (2.0 ** torch.arange(bands, dtype=torch.float32))[None, :, None]
        for bands in [meta["position_bands"], meta["direction_bands"]]
    }
    midpoints = torch.arange(samples, dtype=torch.float64)[None, :] + 0.5

    def encode(x, bands, residual):
        phases = x[:, None, :] * frequencies[bands]
        features = torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)
        return torch.cat((x, features), dim=1) if residual else features

    def linear(x, name):
        return x @ params[name + "::weight"] + params[name + "::bias"]

    def render(output_f32=False):
        width = (bounds[:, 1] - bounds[:, 0]) / samples
        t = bounds[:, :1] + midpoints * width[:, None]
        positions = (origins.double()[:, None, :] + directions.double()[:, None, :] * t[:, :, None]).float().reshape(-1, 3)
        dirs = directions[:, None, :].expand(batch, samples, 3).reshape(-1, 3)
        trunk = linear(encode(positions, meta["position_bands"], True), "trunk_fc0").relu()
        density = linear(trunk, "density").relu().reshape(batch, samples).double()
        features = linear(trunk, "feature")
        color_input = torch.cat((features, encode(dirs, meta["direction_bands"], False)), dim=1)
        hidden = linear(color_input, "color_fc0").relu()
        rgb = linear(hidden, "color_out").reshape(batch, samples, 3).double()
        tau = density * width[:, None]
        # Prefix optical depth is independent of the Rust product recurrence.
        exclusive = torch.cat((torch.zeros(batch, 1, dtype=torch.float64), tau[:, :-1].cumsum(dim=1)), dim=1)
        weights = (-exclusive).exp() * -(-tau).expm1()
        color = (weights[:, :, None] * rgb).sum(dim=1)
        trans = (-tau.sum(dim=1)).exp().mean()
        return color.float() if output_f32 else color, trans

    return render, params, targets


source = json.load(open(sys.argv[1]))
cases = []
for case in source["cases"]:
    render, _, _ = setup(case["metadata"])
    with torch.no_grad():
        for _ in range(5):
            render(output_f32=True)
        elapsed = []
        for _ in range(9):
            start = time.perf_counter_ns()
            for _ in range(4):
                render(output_f32=True)
            elapsed.append((time.perf_counter_ns() - start) / 4)
        colors, _ = render(output_f32=True)
    actual = colors.float().flatten().tolist()
    assert len(actual) == len(case["values"])
    max_abs = max(abs(a - b) for a, b in zip(actual, case["values"]))
    assert max_abs <= 3e-6, (case["metadata"], max_abs)
    cases.append({"batch": case["metadata"]["batch"], "samples": case["metadata"]["samples"],
                  "varying": case["metadata"]["varying"], "values": actual,
                  "max_abs_vs_rust": max_abs, "elapsed_ns": elapsed})

training = source["contract"]["training"]
render, params, targets = setup(training["before"], gradients=True)
color, trans = render()
loss = ((color - targets.double()) ** 2).mean()
loss.backward()
updates = {}
max_update_error = 0.0
with torch.no_grad():
    for name, p in params.items():
        updated = p - training["before"]["learning_rate"] * p.grad
        actual = updated.flatten().tolist()
        expected = training["after"][name]["values"]
        assert len(actual) == len(expected)
        error = max(abs(a - b) for a, b in zip(actual, expected))
        max_update_error = max(max_update_error, error)
        updates[name] = actual
assert abs(loss.item() - training["loss"]) <= 3e-6
assert abs(trans.item() - training["avg_transmittance"]) <= 3e-6
assert max_update_error <= 3e-6, max_update_error
print(json.dumps({"runtime": "pytorch-eager-cpu", "torch": torch.__version__,
    "torch_file": torch.__file__, "python": sys.version, "platform": platform.platform(),
    "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
    "warmups": 5, "intervals": 9, "repetitions": 4, "cases": cases,
    "training": {"loss": loss.item(), "transmittance": trans.item(),
                 "max_parameter_error": max_update_error, "updates": updates},
    "boundary": "CPU eager vectorized reference, not torch.compile or a fastest-PyTorch claim. Timing includes sample construction, field and compositor but not setup/validation/serialization; Rust also validates rays. Float64 ray integration with float32 field and parameter gradients."}))
