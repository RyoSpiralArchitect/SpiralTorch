"""Read-only train-window sensitivity probe; does not optimize any parameter."""
import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def validate_native_snapshot(request, native, request_sha256):
    """Reject partial/foreign reports before labeling retained weights native_128."""
    if (native.get("schema") != "spiraltorch.byte_corpus.result.v7"
            or native.get("engine") != "spiraltorch"
            or type(native.get("adapter")) is not str or not native["adapter"]
            or re.search(r"\bbackend:\s*(Metal|Vulkan|Dx12|Gl)\s*\}$", native["adapter"]) is None
            or native.get("request_sha256") != request_sha256
            or len(request["train_batches"]) != 128):
        raise ValueError("expected a complete native v7 update-128 report")
    expected = request["cases"]
    actual = native.get("cases")
    if type(actual) is not list or len(actual) != len(expected):
        raise ValueError("incomplete native case coverage")
    for case, row in zip(expected, actual):
        if type(row) is not dict or any(row.get(key) != case.get(key) for key in
                ("name", "seed", "geometry", "geometry_update", "bias_initialization", "mixture", "pair_metric")):
            raise ValueError("native case identity mismatch")
        history = row.get("training")
        if (type(history) is not list or len(history) != 128
                or any(type(s) is not dict or type(s.get("revision")) is not int or s["revision"] != i
                       or type(s.get("ce")) not in (int, float) or not math.isfinite(s["ce"])
                       for i, s in enumerate(history, 1))):
            raise ValueError("native snapshot did not complete all 128 updates")
        parameters = row.get("final_parameters")
        if type(parameters) is not list or len(parameters) != len(case["parameters"]):
            raise ValueError("native parameter family coverage")
        for descriptor, values in zip(case["parameters"], parameters):
            if (type(values) is not list or len(values) != math.prod(descriptor["shape"])
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           or abs(v) > float.fromhex("0x1.fffffep+127") for v in values)):
                raise ValueError("native parameter shape or float32 mismatch")


def load(root, name):
    spec = importlib.util.spec_from_file_location(name, root / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "native", "output", "fixture"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    request_raw, native_raw = args.request.read_bytes(), args.native.read_bytes()
    request, native = json.loads(request_raw), json.loads(native_raw)
    study = load(args.root, "byte_corpus_study")
    assert study.request_version(request) == 7
    validate_native_snapshot(request, native, hashlib.sha256(request_raw).hexdigest())
    cfg = request["config"]
    assert cfg["blocks"] == [False]
    batch, steps, width, heads = (cfg[k] for k in ("batch", "steps", "width", "heads"))
    block_forward = load(args.root, "generate_resident_byte_decoder_torch_fixture").block_forward
    wave = load(args.root, "generate_causal_zspace_wave_torch_fixture").wave
    poincare = load(args.root, "generate_poincare_bias_torch_fixture").metric
    fisher = load(args.root, "fisher_rao_reference").metric
    causal = torch.ones((steps, steps), dtype=torch.bool).tril()
    causal64 = causal.to(torch.float64)[None, None, :, :, None]
    counts = causal64.sum(-2, keepdim=True)
    windows = []
    for selected in request["train_batches"]:
        rows = [request["train_documents"][d][s:s + steps + 1] for d, s in selected]
        assert len(rows) == batch and all(len(row) == steps + 1 for row in rows)
        windows.append((torch.tensor([r[:-1] for r in rows]), torch.tensor([r[1:] for r in rows])))
    rows, fixtures = [], []
    for c, completed in zip(request["cases"], native["cases"]):
        if c.get("mixture", {}).get("router_update") != "train":
            continue
        for state in ("initial", "native_128"):
            values = [d["values"] for d in c["parameters"]] if state == "initial" else completed["final_parameters"]
            p = [torch.tensor(v).reshape(d["shape"]) for d, v in zip(c["parameters"], values)]
            p[7].requires_grad_()
            k = len(c["mixture"]["metrics"])
            scales = torch.tensor(c["mixture"]["scales"][0]).reshape(1, heads, 1, 1, k)
            gram = torch.zeros((heads, k, k), dtype=torch.float64)
            qk_square = torch.zeros(heads, dtype=torch.float64)
            mixed_square = torch.zeros(heads, dtype=torch.float64)
            qgrad_sum = torch.zeros((heads, k), dtype=torch.float64)
            qgrad_norm_sum = torch.zeros(heads, dtype=torch.float64)
            batch_router_norm_sum = torch.zeros(heads, dtype=torch.float64)
            tv_sum = torch.zeros(heads, dtype=torch.float64)
            max_grad_error = 0.0
            losses = []
            for index, (ids, target) in enumerate(windows):
                embedded = F.embedding(ids, p[0]) + F.embedding(torch.arange(steps).expand(batch, steps), p[1])
                coordinates, _ = wave(embedded @ p[2] + p[3], p[4], p[5],
                                      torch.zeros((batch, cfg["geometry_cols"])), cfg["curvature"])
                parts = []
                for metric in c["mixture"]["metrics"]:
                    if metric == study.POINCARE:
                        part = poincare(coordinates, p[6], cfg["curvature"])
                    elif metric == study.FLAT:
                        square = 4 * (coordinates[:, :, None] - coordinates[:, None, :]).square().sum(-1)
                        part = (-F.softplus(p[6])[None, :, None, None] * square[:, None]).tril()
                    else:
                        assert metric == study.FISHER
                        part = fisher(coordinates, p[6])
                    parts.append(part)
                components = (torch.stack(parts, dim=-1) * scales).detach().requires_grad_()
                weights = p[7].softmax(-1)
                pair = (components * weights[None, :, None, None]).sum(-1)
                x = block_forward(embedded, p[8:20], heads, None, pair, False)
                logits = F.layer_norm(x, (width,), p[20], p[21], eps=1e-5) @ p[22] + p[23]
                loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))
                seed, router_gradient, component_gradient = torch.autograd.grad(loss, (pair, p[7], components))
                losses.append(loss.item())
                expanded = p[7].detach()[None, :, None, :].expand(batch, heads, steps, k).clone().requires_grad_()
                independent = (components.detach() * expanded.softmax(-1).unsqueeze(-2)).sum(-1)
                query_gradient = torch.autograd.grad(independent, expanded, seed)[0]
                max_grad_error = max(max_grad_error, (query_gradient.double().sum((0, 2)) - router_gradient.double()).abs().max().item())
                with torch.no_grad():
                    query = query_gradient.double()
                    qgrad_sum += query.sum((0, 2))
                    qgrad_norm_sum += query.norm(dim=-1).sum((0, 2))
                    batch_router_norm_sum += query.sum((0, 2)).norm(dim=-1)
                    centered = (components.double() - components.double().sum(-2, keepdim=True) / counts) * causal64
                    gram += torch.einsum("bhqsk,bhqsl->hkl", centered, centered)
                    y = F.layer_norm(embedded, (width,), p[8], p[9], eps=1e-5)
                    qkv = (y @ p[10] + p[11]).reshape(batch, steps, 3, heads, width // heads)
                    q, key = [qkv[:, :, i].transpose(1, 2) for i in (0, 1)]
                    score = (q @ key.transpose(-1, -2)) * torch.tensor(width // heads).sqrt().reciprocal()
                    score_centered = (score.double() - (score.double() * causal).sum(-1, keepdim=True) / causal.sum(-1, keepdim=True)) * causal
                    qk_square += score_centered.square().sum((0, 2, 3))
                    mixed_square += (centered * weights.double()[None, :, None, None]).sum(-1).square().sum((0, 2, 3))
                    plain = score.masked_fill(~causal, -float("inf")).softmax(-1)
                    geometric = (score + pair).masked_fill(~causal, -float("inf")).softmax(-1)
                    tv_sum += (plain.double() - geometric.double()).abs().sum(-1).sum((0, 2)) / 2
                if index == 0:
                    if state == "initial":
                        assert abs(loss.item() - completed["training"][0]["ce"]) < 3e-6
                    flat = lambda v: v.detach().reshape(-1).tolist()
                    fixtures.append(dict(name=c["name"] + "_" + state,
                        shape=list(components.shape), input=flat(components), logits=flat(expanded), seed=flat(seed),
                        output=flat(independent), weights=flat(expanded.softmax(-1)),
                        input_gradient=flat(component_gradient), logits_gradient=flat(query_gradient)))
            assert max_grad_error < 3e-6
            diag = gram.diagonal(dim1=-2, dim2=-1).sqrt()
            correlation = gram / (diag[:, :, None] * diag[:, None, :])
            ratio = lambda numerator, denominator: [None if d == 0 else n / d for n, d in zip(numerator.tolist(), denominator.tolist())]
            rows.append(dict(name=c["name"], seed=c["seed"], state=state, metrics=c["mixture"]["metrics"],
                train_batches=len(windows), mean_train_ce=math.fsum(losses) / len(losses),
                component_row_centered_cosine=correlation.tolist(),
                mean_attention_total_variation=(tv_sum / (len(windows) * batch * steps)).tolist(),
                geometry_to_qk_centered_rms_ratio=ratio(mixed_square.sqrt(), qk_square.sqrt()),
                within_batch_query_retention=ratio(batch_router_norm_sum, qgrad_norm_sum),
                across_batch_retention=ratio(qgrad_sum.norm(dim=-1), batch_router_norm_sum),
                query_gradient_l2_sum=qgrad_norm_sum.tolist(), router_batch_gradient_l2_sum=batch_router_norm_sum.tolist(),
                max_query_to_shared_gradient_abs_error=max_grad_error))
    assert len(rows) == len(fixtures) == 24
    identity = dict(request_sha256=hashlib.sha256(request_raw).hexdigest(), native_sha256=hashlib.sha256(native_raw).hexdigest())
    with args.output.open("x") as out:
        json.dump(dict(schema="spiraltorch.geometry_router_sensitivity.v1", **identity,
            scope="Read-only CPU-f32 probe of initial and native-final weights on all 128 declared training batches. No held-out use or parameter updates; query decomposition is not quality evidence.",
            torch_version=torch.__version__, cases=rows), out, indent=2, allow_nan=False)
        out.write("\n")
    with args.fixture.open("x") as out:
        json.dump(dict(schema="spiraltorch.querywise_mixture.torch_fixture.v1", **identity,
            source="First declared training batch of each learned mixture at initial and native-final states", cases=fixtures), out, separators=(",", ":"), allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(cases=len(rows), train_batch_forwards=len(rows) * len(windows), updated_parameters=False)))


if __name__ == "__main__":
    main()
