"""Prepare a frozen byte-corpus pilot, run independent Torch, and compare results.

No SpiralTorch import is used by the reference. Rust/native and browser clients
consume the request only, never the expected outputs or reference gradients.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import struct
import subprocess

REQUEST = "spiraltorch.byte_corpus.request.v1"
RESULT = "spiraltorch.byte_corpus.result.v1"
ATOL, RTOL, DELTA_RTOL = 3e-6, 5e-5, .002


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as output:
        output.write(encoded(value))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def validate_numbers(seeds, rate):
    if (not seeds or len(seeds) > 8 or len(set(seeds)) != len(seeds)
            or any(not 0 <= seed <= (1 << 53) - 1 for seed in seeds)):
        raise ValueError("seeds must be unique, nonnegative browser-safe integers (at most eight)")
    try:
        rate = f32(rate)
    except (OverflowError, struct.error) as error:
        raise ValueError("rate must be representable as positive finite float32") from error
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must remain positive finite float32 after conversion")
    return rate


def parameters(config, seed):
    rng = random.Random(seed)
    w, h, t = config["width"], config["hidden"], config["steps"]
    result = []

    def add(name, shape, scale=0., offset=0.):
        result.append({"name": name, "shape": shape,
                       "values": [f32(offset + rng.uniform(-scale, scale)) for _ in range(math.prod(shape))]})

    def norm(name):
        add(name + ".gain", [w], offset=1.)
        add(name + ".bias", [w])

    def linear(name, a, b):
        add(name + ".weight", [a, b], math.sqrt(6. / (a + b)))
        add(name + ".bias", [b])

    add("token_embedding", [256, w], .4)
    add("position_embedding", [t, w], .08)
    for i, topos in enumerate(config["blocks"]):
        name = f"block.{i}"
        norm(name + ".pre")
        linear(name + ".qkv", w, 3 * w)
        linear(name + ".output", w, w)
        norm(name + ".feed")
        linear(name + ".up", w, h)
        if topos:
            add(name + ".topos_gate", [h], offset=.8)
        linear(name + ".down", h, w)
    norm("head")
    linear("head.output", w, 256)
    core = result
    result = []
    rng = random.Random(seed ^ 0x5A17)
    g = config["geometry_cols"]
    add("geometry.projection.weight", [w, g], .5)
    add("geometry.projection.bias", [g], .05)
    add("geometry.raw_decay", [g // 2], .4, -.5)
    add("geometry.raw_phase", [g // 2], .3)
    for i in range(len(config["blocks"])):
        add(f"geometry.raw_gain.{i}", [config["heads"]], .3, -.3)
    return core, core[:2] + result + core[2:]


def windows(documents, steps):
    # Adjacent windows share one context byte; their target spans do not overlap.
    return [(i, start) for i, doc in enumerate(documents)
            for start in range(0, len(doc) - steps, steps)]


def prepare(args):
    rate = validate_numbers(args.seed, args.rate)
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.revision + "^{commit}"], text=True).strip()
    config = dict(batch=args.batch, steps=args.steps, width=args.width, hidden=args.hidden,
                  heads=args.heads, blocks=[False] * args.blocks, geometry_cols=4, curvature=-.75)
    if not (1 <= args.batch <= 8 and 2 <= args.steps <= 128 and 2 <= args.width <= 128
            and 2 <= args.hidden <= 256 and 1 <= args.blocks <= 4 and args.heads > 0
            and args.width % args.heads == 0 and 1 <= args.updates <= 1024
            and 1 <= args.checkpoint_every <= 64 and 1 <= args.validation_batches <= 128
            and math.isfinite(args.rate) and args.rate > 0 and len(args.seed) <= 8):
        raise ValueError("invalid bounded study configuration")

    def documents(paths):
        values, records = [], []
        for name in paths:
            if Path(name).is_absolute() or ".." in Path(name).parts or ":" in name:
                raise ValueError("use repository-relative source paths")
            raw = subprocess.check_output(["git", "show", revision + ":" + name])
            if len(raw) <= args.steps:
                raise ValueError("source document has no complete window")
            values.append(list(raw))
            records.append(dict(path=name, bytes=len(raw), sha256=digest(raw)))
        if len({r["sha256"] for r in records}) != len(records):
            raise ValueError("duplicate source documents")
        return values, records

    train, train_info = documents(args.train_doc)
    valid, valid_info = documents(args.validation_doc)
    if {d["sha256"] for d in train_info} & {d["sha256"] for d in valid_info}:
        raise ValueError("training and validation documents overlap")
    candidates = windows(train, args.steps)
    rng = random.Random(20261009)
    chosen = []
    while len(chosen) < args.updates * args.batch:
        epoch = candidates.copy()
        rng.shuffle(epoch)
        chosen.extend(epoch)
    train_batches = [chosen[i * args.batch:(i + 1) * args.batch] for i in range(args.updates)]
    candidates_valid = windows(valid, args.steps)
    count = args.validation_batches * args.batch
    if len(candidates_valid) < count:
        raise ValueError("validation corpus is too small for distinct windows")
    # Evenly spaced selection, fixed before either model is evaluated.
    selected = [candidates_valid[i * len(candidates_valid) // count] for i in range(count)]
    valid_batches = [selected[i:i + args.batch] for i in range(0, count, args.batch)]
    cases = []
    pair_records = []
    for seed in args.seed:
        plain, geometry = parameters(config, seed)
        pair_records.append(dict(seed=seed, common_initial_sha256=digest(encoded(plain))))
        for enabled, values in [(False, plain), (True, geometry)]:
            cases.append(dict(name=f"seed{seed}_{'geometry' if enabled else 'ordinary'}", seed=seed,
                              geometry=enabled, parameters=values))
    request = dict(schema=REQUEST, config=config, train_documents=train, validation_documents=valid,
                   train_batches=train_batches, validation_batches=valid_batches,
                   checkpoint_every=args.checkpoint_every, rate=rate, cases=cases)
    manifest = dict(schema="spiraltorch.byte_corpus.preparation.v1", source_revision=revision,
                    config=config, train_sources=train_info, validation_sources=valid_info,
                    training_window_selection="seed-20261009 shuffled complete windows, repeated only after an epoch",
                    validation_window_selection="evenly spaced, distinct complete windows, no score-based selection",
                    train_target_bytes=args.updates * args.batch * args.steps,
                    validation_target_bytes=count * args.steps, pairs=pair_records,
                    request_sha256=digest(encoded(request)),
                    criteria=dict(atol=ATOL, rtol=RTOL, geometry_delta_relative_l2=DELTA_RTOL,
                                  minimum_geometry_reference_delta_l2=1e-8),
                    scope="Small document-held-out repository-prose pilot, not broad language quality or equal-parameter/equal-compute comparison")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "request.json", request)
    write_new(args.output / "preparation.json", manifest)
    print(json.dumps({"request_sha256":manifest["request_sha256"], "cases":len(cases),
                      "train_target_bytes":manifest["train_target_bytes"],
                      "validation_target_bytes":manifest["validation_target_bytes"]}))


def load_reference(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(args):
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    block_forward = load_reference("generate_resident_byte_decoder_torch_fixture").block_forward
    wave = load_reference("generate_causal_zspace_wave_torch_fixture").wave
    metric = load_reference("generate_poincare_bias_torch_fixture").metric
    raw = args.request.read_bytes()
    request = json.loads(raw)
    if request["schema"] != REQUEST:
        raise ValueError("wrong request schema")
    config = request["config"]
    b, t, w = config["batch"], config["steps"], config["width"]

    def batch(documents, selected):
        rows = [documents[d][s:s + t + 1] for d, s in selected]
        if len(rows) != b or any(len(row) != t + 1 for row in rows):
            raise ValueError("invalid document window")
        return (torch.tensor([row[:-1] for row in rows], device="cpu", dtype=torch.long),
                torch.tensor([row[1:] for row in rows], device="cpu", dtype=torch.long))

    train = [batch(request["train_documents"], s) for s in request["train_batches"]]
    valid = [batch(request["validation_documents"], s) for s in request["validation_batches"]]
    reports = []
    for case in request["cases"]:
        p = [torch.tensor(v["values"], device="cpu", dtype=torch.float32).reshape(v["shape"]).requires_grad_()
             for v in case["parameters"]]

        def forward(ids):
            pos = torch.arange(t, device="cpu").expand(b, t)
            embedded = functional.embedding(ids, p[0]) + functional.embedding(pos, p[1])
            offset = 2
            if case["geometry"]:
                drive = embedded @ p[2] + p[3]
                coordinates, _ = wave(drive, p[4], p[5],
                                      torch.zeros((b, config["geometry_cols"]), device="cpu"), config["curvature"])
                offset = 6 + len(config["blocks"])
            x = embedded
            for i, topos in enumerate(config["blocks"]):
                count = 13 if topos else 12
                pair = metric(coordinates, p[6 + i], config["curvature"]) if case["geometry"] else None
                x = block_forward(x, p[offset:offset + count], config["heads"], None, pair, topos)
                offset += count
            head = p[offset:]
            return functional.layer_norm(x, (w,), head[0], head[1], eps=1e-5) @ head[2] + head[3]

        def evaluate(revision):
            with torch.no_grad():
                losses = [functional.cross_entropy(forward(ids).reshape(-1, 256), targets.reshape(-1)).item()
                          for ids, targets in valid]
            mean = math.fsum(losses) / len(losses)
            return dict(revision=revision, batch_losses=losses, mean_ce=mean,
                        bits_per_byte=mean / math.log(2), target_bytes=b * t * len(losses))

        evaluation = [evaluate(0)]
        training = []
        for step, (ids, targets) in enumerate(train, 1):
            loss = functional.cross_entropy(forward(ids).reshape(-1, 256), targets.reshape(-1))
            gradients = torch.autograd.grad(loss, p)
            with torch.no_grad():
                for value, gradient in zip(p, gradients):
                    value -= request["rate"] * gradient
            training.append(dict(revision=step, ce=loss.item()))
            if step % request["checkpoint_every"] == 0 or step == len(train):
                evaluation.append(evaluate(step))
        reports.append(dict(name=case["name"], seed=case["seed"], geometry=case["geometry"],
                            parameter_tensors=len(p), parameter_scalars=sum(v.numel() for v in p),
                            training=training, validation=evaluation,
                            final_parameters=[v.detach().reshape(-1).tolist() for v in p]))
    write_new(args.output, dict(schema=RESULT, engine="independent_pytorch", torch_version=torch.__version__,
                               dtype="float32", device="cpu", threads=1, request_sha256=digest(raw), cases=reports))


def close(actual, expected, label):
    if len(actual) != len(expected):
        raise ValueError(label + ": length mismatch")
    maximum = 0.
    for a, e in zip(actual, expected):
        error = abs(a - e)
        if not math.isfinite(a) or not math.isfinite(e) or error > ATOL + RTOL * abs(e):
            raise ValueError(f"{label}: {a} != {e}")
        maximum = max(maximum, error)
    return maximum


def compare(request_raw, reference_report, actual):
    request = json.loads(request_raw)
    if (request["schema"] != REQUEST or reference_report.get("engine") != "independent_pytorch"
            or actual.get("engine") != "spiraltorch" or not actual.get("adapter")):
        raise ValueError("wrong request, reference, or measured engine identity")
    expected_cases = request["cases"]
    names = [case["name"] for case in expected_cases]
    if len(set(names)) != len(names):
        raise ValueError("duplicate request case")
    for report in [reference_report, actual]:
        if report["schema"] != RESULT or report["request_sha256"] != digest(request_raw):
            raise ValueError("result source mismatch")
        if [c["name"] for c in report["cases"]] != names:
            raise ValueError("case identity/order mismatch")
    n = len(request["train_batches"])
    checkpoints = [0] + [i for i in range(1, n + 1) if i % request["checkpoint_every"] == 0 or i == n]
    summaries = []
    for spec, expected, got in zip(expected_cases, reference_report["cases"], actual["cases"]):
        name = spec["name"]
        for row in [expected, got]:
            if row["seed"] != spec["seed"] or row["geometry"] != spec["geometry"]:
                raise ValueError("case mode/seed mismatch")
            if row["parameter_tensors"] != len(spec["parameters"]) or row["parameter_scalars"] != sum(len(p["values"]) for p in spec["parameters"]):
                raise ValueError("parameter count mismatch")
            if [s["revision"] for s in row["training"]] != list(range(1, n + 1)):
                raise ValueError("incomplete training revisions")
            if [s["revision"] for s in row["validation"]] != checkpoints:
                raise ValueError("incomplete held-out checkpoints")
            for point in row["validation"]:
                losses = point["batch_losses"]
                if len(losses) != len(request["validation_batches"]) or point["target_bytes"] != request["config"]["batch"] * request["config"]["steps"] * len(losses):
                    raise ValueError("validation coverage mismatch")
                mean = math.fsum(losses) / len(losses)
                close([point["mean_ce"], point["bits_per_byte"]], [mean, mean / math.log(2)], "validation aggregation")
            if len(row["final_parameters"]) != len(spec["parameters"]):
                raise ValueError("incomplete final parameter family")
        train_error = close([s["ce"] for s in got["training"]], [s["ce"] for s in expected["training"]], name + ".training")
        eval_error = max(close(a["batch_losses"], e["batch_losses"], name + ".heldout")
                         for a, e in zip(got["validation"], expected["validation"]))
        parameter_error, geometry_deltas = 0., []
        for desc, a, e in zip(spec["parameters"], got["final_parameters"], expected["final_parameters"]):
            if len(a) != len(desc["values"]) or len(e) != len(desc["values"]):
                raise ValueError("parameter shape mismatch")
            parameter_error = max(parameter_error, close(a, e, name + "." + desc["name"]))
            if desc["name"].startswith("geometry."):
                # Rust uploads float32 values; decimal conversion is not learning.
                initial = [f32(x) for x in desc["values"]]
                expected_norm = math.sqrt(math.fsum((x - y)**2 for x, y in zip(e, initial)))
                actual_norm = math.sqrt(math.fsum((x - y)**2 for x, y in zip(a, initial)))
                error = math.sqrt(math.fsum((x - y)**2 for x, y in zip(a, e)))
                if expected_norm <= 1e-8 or actual_norm == 0 or error / expected_norm > DELTA_RTOL:
                    raise ValueError("unqualified geometry learning delta: " + desc["name"])
                geometry_deltas.append(dict(name=desc["name"], actual_delta_l2=actual_norm,
                                            reference_delta_l2=expected_norm, relative_l2_error=error / expected_norm))
        summaries.append(dict(name=name, seed=spec["seed"], geometry=spec["geometry"],
                              parameter_scalars=got["parameter_scalars"], train_ce_max_error=train_error,
                              heldout_ce_max_error=eval_error, final_parameter_max_error=parameter_error,
                              initial_heldout_bpb=got["validation"][0]["bits_per_byte"],
                              final_heldout_bpb=got["validation"][-1]["bits_per_byte"], geometry_deltas=geometry_deltas))
    pairs = []
    for seed in sorted({c["seed"] for c in summaries}):
        paired = [c for c in summaries if c["seed"] == seed]
        if len(paired) != 2 or {c["geometry"] for c in paired} != {False, True}:
            raise ValueError("incomplete paired seed")
        plain = next(c for c in paired if not c["geometry"])
        geometry = next(c for c in paired if c["geometry"])
        pairs.append(dict(seed=seed, geometry_minus_ordinary_bpb=geometry["final_heldout_bpb"] - plain["final_heldout_bpb"]))
    return dict(schema="spiraltorch.byte_corpus.comparison.v1", request_sha256=digest(request_raw),
                numerical_checks_passed=True, criteria=dict(atol=ATOL, rtol=RTOL, geometry_delta_relative_l2=DELTA_RTOL),
                cases=summaries, paired_deltas=pairs,
                mean_geometry_minus_ordinary_bpb=statistics.mean(p["geometry_minus_ordinary_bpb"] for p in pairs),
                scope="Fixed small corpus/recipe/seeds; different parameter counts and compute, no general quality or speed claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("output", type=Path)
    p.add_argument("--revision", default="HEAD")
    p.add_argument("--train-doc", action="append", required=True)
    p.add_argument("--validation-doc", action="append", required=True)
    p.add_argument("--seed", type=int, action="append", required=True)
    for name, default in [("batch",2),("steps",32),("width",16),("hidden",32),("heads",2),
                          ("blocks",1),("updates",128),("checkpoint-every",64),("validation-batches",32)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--rate", type=float, default=.05)
    p = commands.add_parser("reference")
    p.add_argument("request", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("compare")
    p.add_argument("request", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("actual", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "reference":
        reference(args)
    else:
        result = compare(args.request.read_bytes(), json.loads(args.reference.read_bytes()), json.loads(args.actual.read_bytes()))
        write_new(args.output, result)
        print(json.dumps({"numerical_checks_passed":True, "paired_deltas":result["paired_deltas"]}))


if __name__ == "__main__":
    main()
