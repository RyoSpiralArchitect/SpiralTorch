"""Local linear/Attention VJPs on operands from the pinned saved language models."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct

INDICES = [0, 127, 511, 1023]
PINS = ["b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
        "858a5576ba95c643fc223044fe6eaab4f502fa9afd0fc15b744eaa7bdb0b3dfe"]
IDENTITY = ("name", "batch_index", "operation", "kind", "shape")


def expected_cases():
    shapes = {"geometry.projection": [64,16,4], "block.0.qkv": [64,16,48],
              "block.0.attention": [2,2,32,8], "block.0.output": [64,16,16],
              "block.0.up": [64,16,32], "block.0.down": [64,32,16], "head": [64,16,256]}
    return [dict(zip(IDENTITY, [name, index, op, "attention" if op.endswith("attention") else "linear", shape]))
            for name in ("seed11_ordinary", "seed11_geometry", "seed11_geometry_frozen")
            for index in INDICES for op, shape in shapes.items()
            if name != "seed11_ordinary" or op != "geometry.projection"]


def validate_fixture(fixture):
    if fixture.get("schema") != "spiraltorch.language_local_vjp.fixture.v1" or fixture.get("pins") != PINS:
        raise ValueError("fixture identity drift")
    if [{key: case[key] for key in IDENTITY} for case in fixture["cases"]] != expected_cases():
        raise ValueError("fixed case coverage drift")
    for case in fixture["cases"]:
        if set(case["reference"]) != {"f32", "f64"}:
            raise ValueError("reference coverage drift")
        if case["kind"] == "linear":
            rows, inner, cols = case["shape"]
            lengths = {"forward": rows*cols, "input": rows*inner, "weight": inner*cols, "bias": cols}
        else:
            lengths = dict.fromkeys(("forward", "query", "key", "value"), math.prod(case["shape"]))
            if case["name"] != "seed11_ordinary":
                lengths["pair"] = 2*2*32*32
        for vectors in case["reference"].values():
            if set(vectors) != set(lengths) or any(len(vectors[k]) != n for k,n in lengths.items()):
                raise ValueError("reference shape drift")


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def flat(tensor):
    return tensor.detach().reshape(-1).tolist()


def attention(torch, q, k, v, pair, scale):
    scores = (q @ k.transpose(-1, -2)) * scale
    if pair is not None:
        scores = scores + pair
    mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).triu(1)
    return scores.masked_fill(mask, -float("inf")).softmax(-1) @ v


def local_reference(torch, item, dtype):
    tensor = lambda key, shape: torch.tensor(item[key], dtype=torch.float32).to(dtype).reshape(shape).requires_grad_()
    if item["kind"] == "linear":
        rows, inner, cols = item["shape"]
        x, w, b = tensor("input", [rows, inner]), tensor("weight", [inner, cols]), tensor("bias", [cols])
        out, names, inputs = x @ w + b, ["input", "weight", "bias"], [x, w, b]
    else:
        shape = item["shape"]
        q, k, v = [tensor(key, shape) for key in ("query", "key", "value")]
        pair = None if item["pair"] is None else tensor("pair", [*shape[:2], shape[2], shape[2]])
        out = attention(torch, q, k, v, pair, item["scale"])
        names, inputs = ["query", "key", "value"], [q, k, v]
        if pair is not None:
            names.append("pair")
            inputs.append(pair)
    seed = torch.tensor(item["upstream"], dtype=torch.float32).to(dtype).reshape(out.shape)
    gradients = torch.autograd.grad(out, inputs, seed)
    return {"forward": flat(out), **{name: flat(g) for name, g in zip(names, gradients)}}


def extract(args):
    import torch
    from torch.nn import functional as f
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = [args.request, args.saved]
    raw = [p.read_bytes() for p in paths]
    if [sha(v) for v in raw] != PINS:
        raise ValueError("pinned input drift")
    request, saved = map(json.loads, raw)
    c = request["config"]
    if c["blocks"] != [False] or (c["batch"], c["steps"], c["width"], c["heads"]) != (2, 32, 16, 2):
        raise ValueError("only the pinned one-block study is supported")
    wave = load(args.repo / "tools/generate_causal_zspace_wave_torch_fixture.py").wave
    metric = load(args.repo / "tools/generate_poincare_bias_torch_fixture.py").metric
    previous = load(args.repo / "tools/generate_resident_byte_decoder_torch_fixture.py")
    cases = []
    b, t, w, h = c["batch"], c["steps"], c["width"], c["heads"]
    scale = torch.tensor(w // h, dtype=torch.float32).sqrt().reciprocal().item()
    if [s["name"] for s in saved["cases"]] != [v["name"] for v in request["cases"]]:
        raise ValueError("case order drift")
    for case, state in zip(request["cases"], saved["cases"]):
        p = [torch.tensor(v, dtype=torch.float32).reshape(d["shape"]).requires_grad_()
             for v, d in zip(state["final_parameters"], case["parameters"])]
        for index in INDICES:
            windows = [request["train_documents"][d][s:s+t+1] for d, s in request["train_batches"][index]]
            ids, targets = [torch.tensor([v[a:z] for v in windows], dtype=torch.long) for a,z in [(0,-1),(1,None)]]
            x = f.embedding(ids, p[0]) + f.embedding(torch.arange(t).expand(b,t), p[1])
            nodes = []
            def linear(name, x, weight, bias):
                out = x @ weight + bias
                nodes.append(({"kind":"linear", "operation":name,
                               "shape":[b*t, weight.shape[0], weight.shape[1]],
                               "input":flat(x), "weight":flat(weight), "bias":flat(bias)}, out))
                return out
            pair, offset = None, 2
            if case["geometry"]:
                drive = linear("geometry.projection", x, p[2], p[3])
                coordinates, _ = wave(drive, p[4], p[5], torch.zeros((b,c["geometry_cols"])), c["curvature"])
                pair = metric(coordinates, p[6], c["curvature"])
                offset = 7
            block = p[offset:offset+12]
            embedded = x
            y = f.layer_norm(x, (w,), block[0], block[1], eps=1e-5)
            qkv = linear("block.0.qkv", y, block[2], block[3]).reshape(b,t,3,h,w//h)
            q,k,v = [qkv[:,:,i].transpose(1,2) for i in range(3)]
            out = attention(torch, q,k,v,pair,scale)
            nodes.append(({"kind":"attention", "operation":"block.0.attention", "shape":list(q.shape),
                           "query":flat(q), "key":flat(k), "value":flat(v), "scale":scale,
                           "pair":None if pair is None else flat(pair)}, out))
            merged = out.transpose(1,2).reshape(b,t,w)
            residual = x + linear("block.0.output", merged, block[4], block[5])
            y = f.layer_norm(residual, (w,), block[6], block[7], eps=1e-5)
            y = f.gelu(linear("block.0.up", y, block[8], block[9]), approximate="tanh")
            x = residual + linear("block.0.down", y, block[10], block[11])
            if not torch.equal(x, previous.block_forward(embedded, block, h, None, pair, False, module_residual=True)):
                raise ValueError("instrumented block changed reference output")
            head = p[offset+12:]
            y = f.layer_norm(x, (w,), head[0], head[1], eps=1e-5)
            logits = linear("head", y, head[2], head[3])
            loss = f.cross_entropy(logits.reshape(-1,256), targets.reshape(-1))
            seeds = torch.autograd.grad(loss, [output for _,output in nodes])
            for (item, _), seed in zip(nodes, seeds):
                item.update(name=case["name"], batch_index=index, upstream=flat(seed))
                item["reference"] = {name:local_reference(torch,item,dtype) for name,dtype in
                                     [("f32",torch.float32),("f64",torch.float64)]}
                cases.append(item)
    if raw != [p.read_bytes() for p in paths]:
        raise ValueError("inputs changed during extraction")
    write(args.output, {"schema":"spiraltorch.language_local_vjp.fixture.v1", "pins":PINS,
                        "torch":torch.__version__, "cases":cases})


def errors(a,b):
    if len(a)!=len(b) or not a or not all(type(v) in (int,float) and math.isfinite(v) for v in a+b):
        raise ValueError("invalid measured vector")
    return {"max_abs":max(abs(x-y) for x,y in zip(a,b)),
            "l2":math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))}


def key_sum(values, shape):
    b,h,t,d = shape
    return max(abs(math.fsum(values[((bi*h+hi)*t+ti)*d+di] for bi in range(b) for ti in range(t)))
               for hi in range(h) for di in range(d))


def compare(args):
    raw = [p.read_bytes() for p in (args.fixture, args.native)]
    fixture, native = map(json.loads, raw)
    validate_fixture(fixture)
    if native.get("schema") != "spiraltorch.language_local_vjp.native.v1" or native["fixture_sha256"] != sha(raw[0]):
        raise ValueError("fixture identity drift")
    if len(native["cases"]) != len(fixture["cases"]):
        raise ValueError("case coverage drift")
    rows = []
    for expected, actual in zip(fixture["cases"],native["cases"]):
        identity = {k:expected[k] for k in IDENTITY}
        if identity != {k:actual[k] for k in identity}:
            raise ValueError("case identity drift")
        modes = ["native"] if expected["kind"]=="attention" else ["sequential","tiled","compensated"]
        if set(actual["modes"]) != set(modes):
            raise ValueError("mode coverage drift")
        controls = {"torch_f32":expected["reference"]["f32"], **actual["modes"]}
        reference = expected["reference"]["f64"]
        result = {}
        for mode, vectors in controls.items():
            if set(vectors)!=set(reference):
                raise ValueError("component coverage drift")
            result[mode] = {name:errors(values,reference[name]) for name,values in vectors.items()}
        row = {**identity, "vs_f64":result}
        if expected["kind"] == "attention":
            rounded = [struct.unpack('<f',struct.pack('<f',v))[0] for v in reference["key"]]
            row["key_bias_residual"] = {name:key_sum(vectors["key"],expected["shape"]) for name,vectors in controls.items()}
            row["key_bias_residual"].update(f64=key_sum(reference["key"],expected["shape"]),
                                           rounded_f64=key_sum(rounded,expected["shape"]))
        rows.append(row)
    write(args.output, {"schema":"spiraltorch.language_local_vjp.comparison.v1", "sources":dict(zip(["fixture","native"],map(sha,raw))),
                        "rows":rows, "scope":"local VJPs on identical f32 operands; not a whole-model or quality gate"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    for action, names in [("extract",["repo","request","saved","output"]),("compare",["fixture","native","output"])]:
        command = sub.add_parser(action)
        for name in names:
            command.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
