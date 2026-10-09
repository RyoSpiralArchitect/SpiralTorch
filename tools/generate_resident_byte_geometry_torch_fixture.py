"""Freeze full-model causal geometry gradients and matched CPU-f32 updates."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load("generate_resident_byte_decoder_torch_fixture")
wave = load("generate_causal_zspace_wave_torch_fixture").wave
metric = load("generate_poincare_bias_torch_fixture").metric
flat = base.values


def case(block_count, external, metric_only, seed):
    original = base.case(block_count, external, seed)
    config = original["config"]
    batch, steps, width = config["batch"], config["steps"], config["width"]
    cols, curvature = 4, -.75
    config["causal_geometry"] = {"cols": cols, "curvature": curvature, "metric_only_scores": metric_only}
    rng = torch.Generator(device="cpu").manual_seed(seed + 1047)
    projection = (torch.rand((width, cols), generator=rng, device="cpu") - .5) * 1.6
    extra = [("projection.weight", projection),
             ("projection.bias", torch.tensor([.1, -.08, .07, -.02], device="cpu")),
             ("raw_decay", torch.tensor([-.7, .4], device="cpu")),
             ("raw_phase", torch.tensor([-.2, .3], device="cpu"))]
    for i in range(block_count):
        extra.append((f"raw_gain.{i}", torch.tensor([-.6 + .3 * i, .3 - .7 * i], device="cpu")))
    extra = [{"name": "geometry." + n, "shape": list(t.shape), "values": flat(t)} for n, t in extra]
    descriptors = original["parameters"][:2] + extra + original["parameters"][2:]
    p = [torch.tensor(d["values"], device="cpu").reshape(d["shape"]).requires_grad_() for d in descriptors]
    end = 2 + len(extra)
    if metric_only:
        with torch.no_grad():
            for d, value in zip(descriptors, p):
                if d["name"].endswith("qkv.weight"):
                    value[:, :2 * width] = 0
                if d["name"].endswith("qkv.bias"):
                    value[:2 * width] = 0
    initial = [{**d, "values": flat(t)} for d, t in zip(descriptors, p)]
    ids = torch.tensor(original["inputs"], device="cpu", dtype=torch.long).reshape(batch, steps)
    target = torch.tensor(original["targets"], device="cpu", dtype=torch.long).reshape(batch, steps)
    biases = []
    for b in original["biases"]:
        z = None if b["z"] is None else torch.tensor(b["z"], device="cpu").reshape(batch, config["heads"], steps).requires_grad_()
        pair = None if b["pair"] is None else torch.tensor(b["pair"], device="cpu").reshape(batch, config["heads"], steps, steps).requires_grad_()
        biases.append((z, pair))

    def forward(enabled=True, detach=False):
        positions = torch.arange(steps, device="cpu").expand(batch, steps)
        embedded = F.embedding(ids, p[0]) + F.embedding(positions, p[1])
        drive = embedded @ p[2] + p[3]
        coordinates, _ = wave(drive, p[4], p[5], torch.zeros((batch, cols), device="cpu"), curvature)
        x, offset, combined = embedded, end, []
        for i, (block, (z, pair)) in enumerate(zip(config["blocks"], biases)):
            if enabled:
                geometric = metric(coordinates, p[6 + i], curvature)
                if detach:
                    geometric = geometric.detach()
                pair = geometric if pair is None else geometric + pair
            combined.append({"z": None if z is None else flat(z), "pair": None if pair is None else flat(pair)})
            count = 13 if block["topos"] else 12
            x = base.block_forward(x, p[offset:offset + count], config["heads"], z, pair, block["topos"])
            offset += count
        h = p[offset:]
        logits = F.layer_norm(x, (width,), h[0], h[1], eps=1e-5) @ h[2] + h[3]
        return logits, embedded, combined

    logits, embedded, combined = forward()
    seed_tensor = torch.tensor(original["cotangent"], device="cpu").reshape(batch, steps, 256)
    external_tensors = [b for group in biases for b in group if b is not None]
    gradients = torch.autograd.grad(logits, [*p, embedded, *external_tensors], seed_tensor)
    result = {"name": f"causal_geometry_blocks{block_count}_metric_only{metric_only}", "config": config,
              "parameters": initial, "windows": original["windows"], "inputs": original["inputs"], "targets": original["targets"],
              "biases": original["biases"], "cotangent": original["cotangent"], "output": flat(logits),
              "parameter_gradients": [flat(g) for g in gradients[:len(p)]],
              "embedding_output_gradient": flat(gradients[len(p)]),
              "bias_gradients": [flat(g) for g in gradients[len(p) + 1:]]}
    off, _, _ = forward(enabled=False)
    detached, detached_embedding, _ = forward(detach=True)
    detached_gradient = torch.autograd.grad(detached, detached_embedding, seed_tensor)[0]
    result["controls"] = {"off_output": flat(off), "detached_output": flat(detached),
                          "detached_embedding_gradient": flat(detached_gradient), "combined_biases": combined}
    trace = []
    for _ in range(16):
        logits, _, _ = forward()
        loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
        gradient = torch.autograd.grad(loss, p)
        with torch.no_grad():
            for value, g in zip(p, gradient):
                value -= .125 * g
        trace.append({"loss": loss.item(), "parameters": [flat(t) for t in p],
                      "geometry_gradients": [flat(g) for g in gradient[2:end]]})
    result["learning"] = {"steps": 16, "rate": .125, "trace": trace}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    payload = {"schema": "spiraltorch.resident_byte_geometry.torch_fixture.v1",
               "torch_version": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1,
               "tolerance": {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002},
               "cases": [case(1, False, True, 1761), case(2, True, False, 1863)]}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, separators=(",", ":"), allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
