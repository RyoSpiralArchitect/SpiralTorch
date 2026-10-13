"""Same-state diagnostic. Never replaces the frozen qualification reference."""
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("geometry", root / "tools/generate_resident_byte_geometry_torch_fixture.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
projective = geometry.load("projective_reference").metric
fisher = geometry.load("fisher_rao_reference").metric
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def evaluate(case, values, dtype, objective="seed", detached_router=False, detach_geometry=False, module_residual=False):
    config = case["config"]
    batch, steps, width, heads = (config[n] for n in ("batch", "steps", "width", "heads"))
    g = config["causal_geometry"]
    blocks = len(config["blocks"])
    mixture = g.get("mixture")
    contextual = bool(mixture and "query_projection" in mixture)
    p = [torch.tensor(v, device="cpu", dtype=dtype).reshape(d["shape"]).requires_grad_()
         for v, d in zip(values, case["parameters"])]
    ids = torch.tensor(case["inputs"], device="cpu", dtype=torch.long).reshape(batch, steps)
    targets = torch.tensor(case["targets"], device="cpu", dtype=torch.long).reshape(batch, steps)
    embedded = F.embedding(ids, p[0]) + F.embedding(torch.arange(steps, device="cpu").expand(batch, steps), p[1])
    coordinates, _ = geometry.wave(embedded @ p[2] + p[3], p[4], p[5],
        torch.zeros((batch, g["cols"]), device="cpu", dtype=dtype), g["curvature"])
    end = 6 + blocks + (blocks if mixture else 0) + (2 * blocks if contextual else 0)
    offset, x = end, embedded
    for i, (block, external) in enumerate(zip(config["blocks"], case["biases"])):
        if mixture:
            distance = 4 * (coordinates[:, :, None] - coordinates[:, None, :]).square().sum(-1)
            causal = torch.ones((steps, steps), device="cpu", dtype=torch.bool).tril()
            euclidean = (-F.softplus(p[6+i])[None, :, None, None] * distance[:, None]).masked_fill(~causal, 0.)
            components = [geometry.metric(coordinates, p[6+i], g["curvature"]),
                euclidean, fisher(coordinates, p[6+i])]
            components.append(projective(coordinates, p[6+i]))
            components = torch.stack(components, -1)
            logits = p[6 + blocks + i]
            if contextual:
                q = 6 + 2 * blocks + 2 * i
                router_source = coordinates.detach() if detached_router else coordinates
                projected = router_source @ p[q] + p[q+1]
                logits = projected.reshape(batch, steps, heads, 4).permute(0, 2, 1, 3) + logits[None, :, None, :]
                weights = logits.softmax(-1)
            else:
                weights = logits.softmax(-1).reshape(1, heads, 1, 4)
            scales = torch.tensor(mixture["scales"][i], device="cpu", dtype=dtype).reshape(1, heads, 1, 1, 4)
            bias = (components * scales * weights.unsqueeze(-2)).sum(-1)
        else:
            bias = projective(coordinates, p[6 + i])
        if detach_geometry:
            bias = bias.detach()
        z = None if external["z"] is None else torch.tensor(external["z"], device="cpu", dtype=dtype).reshape(batch, heads, steps)
        if external["pair"] is not None:
            bias = bias + torch.tensor(external["pair"], device="cpu", dtype=dtype).reshape(batch, heads, steps, steps)
        count = 13 if block["topos"] else 12
        x = geometry.base.block_forward(x, p[offset:offset+count], heads, z, bias, block["topos"], module_residual=module_residual)
        offset += count
    h = p[offset:]
    logits = F.layer_norm(x, (width,), h[0], h[1], eps=1e-5) @ h[2] + h[3]
    if objective == "ce":
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        gradients = torch.autograd.grad(loss, [*p, embedded], allow_unused=True)
    else:
        loss = None
        seed = torch.tensor(case["cotangent"], device="cpu", dtype=dtype).reshape(batch, steps, 256)
        gradients = torch.autograd.grad(logits, [*p, embedded], seed, allow_unused=True)
    flat = lambda t: t.detach().reshape(-1).double()
    return {"output": flat(logits), "loss": None if loss is None else loss.item(),
        "gradients": [None if t is None else flat(t) for t in gradients[:-1]],
        "embedding": flat(gradients[-1]), "geometry_end": end}


def tensor(value):
    return torch.tensor(value, device="cpu", dtype=torch.float64)


def errors(actual, expected):
    delta, norm = (actual-expected).norm().item(), expected.norm().item()
    return {"absolute_l2": delta, "relative_l2": delta/norm if norm else None,
        "reference_norm": norm, "max_abs": (actual-expected).abs().max().item()}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Private directory with reference.json, native-observed.json and browser.json")
    parser.add_argument("out", type=Path, help="Fresh private output directory; outputs are never overwritten")
    args = parser.parse_args()
    source, out = args.source, args.out
    out.mkdir(parents=True, exist_ok=True)
    if any((out / n).exists() for n in ("same-state.json", "stopped-reference.json")):
        raise FileExistsError("Use a fresh diagnostic output directory")
    fixture = json.loads((source / "reference.json").read_text())
    reports = {runtime: json.loads((source / name).read_text())
               for runtime, name in [("native", "native-observed.json"), ("browser", "browser.json")]}
    initial, phase_trace, stopped_reference = [], [], []
    for ci, case in enumerate(fixture["cases"]):
        values = [p["values"] for p in case["parameters"]]
        f32 = evaluate(case, values, torch.float32)
        f64 = evaluate(case, values, torch.float64)
        # The diagnostic must reproduce the frozen CPU-f32 graph before use.
        assert torch.equal(f32["output"], tensor(case["output"])), case["name"]
        for descriptor, actual, expected in zip(case["parameters"], f32["gradients"], case["parameter_gradients"]):
            assert torch.equal(actual, tensor(expected)), (case["name"], descriptor["name"], errors(actual, tensor(expected)))
        assert torch.equal(f32["embedding"], tensor(case["embedding_output_gradient"])), case["name"]
        row = {"case": case["name"], "f32_exact": True,
            "torch_f32_to_f64_embedding": errors(f32["embedding"], f64["embedding"]), "runtimes": {}}
        for runtime, report in reports.items():
            actual = report["checks"][ci]
            row["runtimes"][runtime] = {
                "embedding_to_f64": errors(tensor(actual["embedding_output_gradient"]), f64["embedding"]),
                "parameters": [{"name": p["name"],
                    "gpu_to_f64": errors(tensor(a), b), "torch_f32_to_f64": errors(c, b)}
                    for p, a, b, c in zip(case["parameters"], actual["parameter_gradients"], f64["gradients"], f32["gradients"])]}
        if ci >= 4:
            stopped32 = evaluate(case, values, torch.float32, detached_router=True)
            stopped64 = evaluate(case, values, torch.float64, detached_router=True)
            assert torch.equal(stopped32["output"], f32["output"])
            d32, d64 = stopped32["embedding"], stopped64["embedding"]
            stopped_reference.append({"name": case["name"],
                "embedding": d32.tolist(), "parameters": [v.tolist() for v in stopped32["gradients"]],
                "paired_f64": (f64["embedding"] - d64).tolist()})
            assert torch.equal(d32, tensor(case["query_controls"]["router_detached_embedding_gradient"]))
            ref_delta = (f32["embedding"].float() - d32.float()).double()
            exact_delta = f64["embedding"] - d64
            row["router"] = {"torch_f32_paired_to_f64": errors(ref_delta, exact_delta),
                "torch_f64_total_minus_f32_detached_to_f32_paired": errors(f64["embedding"] - d32, ref_delta),
                "f64_paired_norm": exact_delta.norm().item(), "f32_paired_norm": ref_delta.norm().item()}
        initial.append(row)
        if ci == 1:
            for runtime, report in reports.items():
                trace = report["checks"][ci]["learning"]["trace"]
                for step, actual in enumerate(trace):
                    values = [p["values"] for p in case["parameters"]] if step == 0 else trace[step-1]["parameters"]
                    a32 = evaluate(case, values, torch.float32, "ce")
                    a64 = evaluate(case, values, torch.float64, "ce")
                    observed = tensor(actual["geometry_gradients"][3])
                    frozen = tensor(case["learning"]["trace"][step]["geometry_gradients"][3])
                    phase_trace.append({"runtime": runtime, "step": step + 1,
                        "gpu_vs_frozen": errors(observed, frozen),
                        "gpu_vs_same_state_f32": errors(observed, a32["gradients"][5]),
                        "gpu_vs_same_state_f64": errors(observed, a64["gradients"][5]),
                        "f32_vs_f64": errors(a32["gradients"][5], a64["gradients"][5])})
    result = {"scope": "diagnostic; immutable original gates remain in force; f64 retains legacy constants",
        "torch": torch.__version__, "initial": initial, "phase_trace": phase_trace}
    with (out / "same-state.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    with (out / "stopped-reference.json").open("x") as f:
        json.dump({"schema": "spiraltorch.query_stop.torch_reference.v1",
                   "cases": stopped_reference}, f, allow_nan=False)
        f.write("\n")
    print(json.dumps({"cases": len(initial), "phase_steps": len(phase_trace), "f32_exact": True}))


if __name__ == "__main__":
    main()
