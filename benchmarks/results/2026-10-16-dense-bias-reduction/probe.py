"""Read-only independent gradient controls for the fixed saved language study."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct

INDICES = [0, 127, 511, 1023]
PINS = {
    "request": "b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
    "saved": "858a5576ba95c643fc223044fe6eaab4f502fa9afd0fc15b744eaa7bdb0b3dfe",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def inputs(request_path, saved_path):
    raw = {"request": request_path.read_bytes(), "saved": saved_path.read_bytes()}
    if {k: digest(v) for k, v in raw.items()} != PINS:
        raise ValueError("pinned input drift")
    return tuple(json.loads(raw[name]) for name in ("request", "saved"))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(values):
    return b"".join(struct.pack("<f", v) for v in values)


def high_precision_block(torch, x, p, heads, pair):
    f = torch.nn.functional
    b, t, w = x.shape
    y = f.layer_norm(x, (w,), p[0], p[1], eps=f32(1e-5))
    qkv = (y @ p[2] + p[3]).reshape(b, t, 3, heads, w // heads)
    q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
    # Hold the Rust/f32 scale control fixed while widening the arithmetic.
    scale = torch.tensor(w // heads, dtype=torch.float32).sqrt().reciprocal().item()
    scores = (q @ k.transpose(-1, -2)) * scale
    if pair is not None:
        scores = scores + pair
    scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), -float("inf"))
    attention = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, w)
    residual = x + (attention @ p[4] + p[5])
    y = f.layer_norm(residual, (w,), p[6], p[7], eps=f32(1e-5))
    y = f.gelu(y @ p[8] + p[9], approximate="tanh")
    return residual + (y @ p[10] + p[11])


def high_precision_wave(torch, drive, decay, phase, curvature):
    b, t, cols = drive.shape
    rho, theta = f32(0.99) * decay.sigmoid(), f32(math.pi) * phase.tanh()
    cos, sin = theta.cos(), theta.sin()
    state = torch.zeros((b, cols // 2, 2), dtype=torch.float64)
    radius = (torch.tensor(0.95, dtype=torch.float32) /
              torch.tensor(-curvature, dtype=torch.float32).sqrt()).item()
    output = []
    for index in range(t):
        x, y = state[..., 0], state[..., 1]
        rotated = torch.stack((cos * x - sin * y, sin * x + cos * y), -1)
        state = rho[None, :, None] * rotated + (1 - rho)[None, :, None] * drive[:, index].reshape(b, cols // 2, 2)
        flat = state.reshape(b, cols)
        output.append(radius * flat / (1 + flat.square().sum(-1, keepdim=True)).sqrt())
    return torch.stack(output, 1)


def reference(args):
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    request, saved = inputs(args.request, args.saved)
    tools = args.repo / "tools"
    base = load(tools / "generate_resident_byte_decoder_torch_fixture.py")
    wave = load(tools / "generate_causal_zspace_wave_torch_fixture.py").wave
    metric = load(tools / "generate_poincare_bias_torch_fixture.py").metric
    c = request["config"]
    if any(c["blocks"]) or len(c["blocks"]) != 1:
        raise ValueError("probe is scoped to the saved one-block non-Topos study")
    b, t, w = c["batch"], c["steps"], c["width"]
    if [v["name"] for v in saved["cases"]] != [v["name"] for v in request["cases"]]:
        raise ValueError("case identity drift")
    result = []
    for case, state in zip(request["cases"], saved["cases"]):
        modes = {}
        for mode in ("float32_legacy", "float32_module", "float64_module"):
            dtype = torch.float64 if mode == "float64_module" else torch.float32
            p = [torch.tensor(values, dtype=torch.float32).to(dtype).reshape(d["shape"]).requires_grad_()
                 for d, values in zip(case["parameters"], state["final_parameters"])]
            rows = []
            for index in INDICES:
                windows = [request["train_documents"][d][s:s + t + 1] for d, s in request["train_batches"][index]]
                ids = torch.tensor([v[:-1] for v in windows], dtype=torch.long)
                targets = torch.tensor([v[1:] for v in windows], dtype=torch.long)
                position = torch.arange(t).expand(b, t)
                x = torch.nn.functional.embedding(ids, p[0]) + torch.nn.functional.embedding(position, p[1])
                offset, pair = 2, None
                if case["geometry"]:
                    drive = x @ p[2] + p[3]
                    if dtype == torch.float64:
                        coordinates = high_precision_wave(torch, drive, p[4], p[5], c["curvature"])
                    else:
                        coordinates, _ = wave(drive, p[4], p[5], torch.zeros((b, c["geometry_cols"])), c["curvature"])
                    pair = metric(coordinates, p[6], c["curvature"])
                    offset = 7
                block = p[offset:offset + 12]
                if dtype == torch.float64:
                    x = high_precision_block(torch, x, block, c["heads"], pair)
                else:
                    x = base.block_forward(x, block, c["heads"], None, pair, False,
                                           module_residual=mode == "float32_module")
                head = p[offset + 12:]
                epsilon = f32(1e-5) if dtype == torch.float64 else 1e-5
                logits = torch.nn.functional.layer_norm(x, (w,), head[0], head[1], eps=epsilon) @ head[2] + head[3]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
                gradients = torch.autograd.grad(loss, p)
                if not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise ValueError("nonfinite reference gradient")
                rows.append({"batch_index": index, "loss": loss.item(),
                             "gradients": [g.detach().reshape(-1).tolist() for g in gradients]})
            modes[mode] = rows
        result.append({"name": case["name"], "modes": modes})
    inputs(args.request, args.saved)
    write(args.output, {"schema": "spiraltorch.byte_gradient_reference.v1", "pins": PINS,
                        "torch_version": torch.__version__, "cases": result,
                        "scope": "f64 arithmetic with f32 parameters and fixed f32 controls; tanh GELU; no optimizer"})


def errors(actual, expected):
    if len(actual) != len(expected) or not all(math.isfinite(v) for v in actual + expected):
        raise ValueError("invalid gradient shape or value")
    l2 = math.sqrt(math.fsum((a - e) ** 2 for a, e in zip(actual, expected)))
    norm = math.sqrt(math.fsum(e * e for e in expected))
    return {"max_abs_error": max(abs(a - e) for a, e in zip(actual, expected)),
            "error_l2": l2, "reference_l2": norm, "relative_l2_error": l2 / norm if norm else None}


def compare(args):
    request, saved = inputs(args.request, args.saved)
    raw = {name: getattr(args, name).read_bytes() for name in ("reference", "before", "after")}
    reference, before, after = (json.loads(raw[name]) for name in ("reference", "before", "after"))
    if reference["schema"] != "spiraltorch.byte_gradient_reference.v1" or any(
            report["schema"] != "spiraltorch.byte_gradient_probe.v1" for report in (before, after)):
        raise ValueError("gradient report schema drift")
    names = [c["name"] for c in request["cases"]]
    for report in (reference, before, after):
        if [c["name"] for c in report["cases"]] != names:
            raise ValueError("case coverage drift")
    result = []
    w = request["config"]["width"]
    for i, case in enumerate(request["cases"]):
        descriptors = case["parameters"]
        for report in (before, after):
            row = report["cases"][i]
            if report["request_sha256"] != PINS["request"] or not row["checkpoint_unchanged"]:
                raise ValueError("native identity or immutability failure")
            if ([bits(p) for p in row["parameter_values"]] !=
                    [bits(p) for p in saved["cases"][i]["final_parameters"]]):
                raise ValueError("native parameter bits differ from reference state")
            if [p["batch_index"] for p in row["probes"]] != INDICES:
                raise ValueError("batch coverage drift")
        modes = reference["cases"][i]["modes"]
        if set(modes) != {"float32_legacy", "float32_module", "float64_module"}:
            raise ValueError("reference mode coverage drift")
        if reference["pins"] != PINS or any([p["batch_index"] for p in rows] != INDICES for rows in modes.values()):
            raise ValueError("reference coverage drift")
        for j, batch in enumerate(INDICES):
            probes = {"native_before": before["cases"][i]["probes"][j],
                      "native_after": after["cases"][i]["probes"][j],
                      **{name: rows[j] for name, rows in modes.items()}}
            if any(len(p["gradients"]) != len(descriptors) for p in probes.values()):
                raise ValueError("gradient parameter coverage drift")
            controls = {}
            for name, probe in probes.items():
                if not math.isfinite(probe["loss"]):
                    raise ValueError("nonfinite probe loss")
                params = []
                for k, descriptor in enumerate(descriptors):
                    values = probe["gradients"][k]
                    if len(values) != math.prod(descriptor["shape"]):
                        raise ValueError("gradient shape differs from parameter")
                    measured = {"name": descriptor["name"],
                                "vs_f64": errors(values, probes["float64_module"]["gradients"][k]),
                                "vs_f32_module": errors(values, probes["float32_module"]["gradients"][k])}
                    if descriptor["name"] == "block.0.qkv.bias":
                        measured["key_max_abs"] = max(abs(v) for v in values[w:2*w])
                    params.append(measured)
                controls[name] = {"loss": probe["loss"], "parameters": params}
            result.append({"name": case["name"], "batch_index": batch, "controls": controls})
    write(args.output, {"schema": "spiraltorch.byte_gradient_comparison.v1", "pins": PINS,
                        "sources": {name: digest(value) for name, value in raw.items()},
                        "probes": result, "scope": "12 fixed-state single-backward probes; not a long-run or quality gate"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, extra in (("reference", ("repo",)), ("compare", ("reference", "before", "after"))):
        p = sub.add_parser(name)
        for flag in ("request", "saved", "output", *extra):
            p.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
