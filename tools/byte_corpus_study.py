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
REQUEST_V2 = "spiraltorch.byte_corpus.request.v2"
REQUEST_V3 = "spiraltorch.byte_corpus.request.v3"
POINCARE, FLAT = "poincare_squared.v1", "euclidean_chord_squared.v1"
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


def decode_json(raw):
    """Preserve Rust's signed zero and reject ambiguous JSON before comparison."""
    def object_value(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite JSON number")
        return number

    return json.loads(raw, object_pairs_hook=object_value, parse_float=finite_float,
                      parse_int=lambda value: -0.0 if value == "-0" else int(value),
                      parse_constant=finite_float)


def frozen(case):
    return case.get("geometry_update") == "frozen"


def trainable_scalars(case):
    return sum(len(p["values"]) for p in case["parameters"]
               if not frozen(case) or not p["name"].startswith("geometry."))


def request_version(request):
    if request["schema"] != REQUEST_V3 and any("pair_metric" in c for c in request["cases"]):
        raise ValueError("v1/v2 have no explicit pair metric")
    if request["schema"] == REQUEST:
        if any("geometry_update" in c for c in request["cases"]):
            raise ValueError("v1 has no explicit geometry update policy")
        return 1
    if request["schema"] not in (REQUEST_V2, REQUEST_V3):
        raise ValueError("wrong request schema")
    version = 3 if request["schema"] == REQUEST_V3 else 2
    modes = {(False, "train", None), (True, "train", POINCARE), (True, "frozen", POINCARE),
             (True, "train", FLAT), (True, "frozen", FLAT)} if version == 3 else {
                 (False, "train", None), (True, "train", None), (True, "frozen", None)}
    def key(parameters):
        for p in parameters:
            if (not p["shape"] or any(type(d) is not int or d <= 0 for d in p["shape"])
                    or math.prod(p["shape"]) != len(p["values"])
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in p["values"])):
                raise ValueError("invalid parameter shape or value")
        return [(p["name"], p["shape"], [struct.pack("<f", x) for x in p["values"]]) for p in parameters]
    if not 2 <= len(request["cases"]) <= 16 or any(type(c["seed"]) is not int or not 0 <= c["seed"] <= (1 << 53) - 1 for c in request["cases"]):
        raise ValueError("invalid control study case count or seed")
    for seed in {c["seed"] for c in request["cases"]}:
        group = [c for c in request["cases"] if c["seed"] == seed]
        if (len(group) != len(modes) or any(type(c["geometry"]) is not bool for c in group)
                or {(c["geometry"], c.get("geometry_update"), c.get("pair_metric")) for c in group} != modes
                or (version == 3 and any(not c["geometry"] and "pair_metric" in c for c in group))):
            raise ValueError("missing, duplicate or invalid metric/update arm per seed")
        core = [key([p for p in c["parameters"] if not p["name"].startswith("geometry.")]) for c in group]
        if any(c != core[0] for c in core[1:]):
            raise ValueError("control backbones differ")
        geometric = [key(c["parameters"]) for c in group if c["geometry"]]
        if any(g != geometric[0] for g in geometric[1:]):
            raise ValueError("metric/update arms have different initial geometry bits")
    if not request["cases"]:
        raise ValueError("empty control study")
    return version


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
    controls = getattr(args, "frozen_geometry_control", False)
    metrics = getattr(args, "metric_geometry_control", False)
    if metrics and controls:
        raise ValueError("select either the v2 frozen control or the v3 metric control")
    version = 3 if metrics else 2 if controls else 1
    if metrics and len(args.seed) > 3:
        raise ValueError("five-arm controls allow at most three seeds within the 16-case budget")
    if controls and len(args.seed) > 5:
        raise ValueError("three-arm controls allow at most five seeds within the 16-case budget")
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
            if controls or metrics:
                cases[-1]["geometry_update"] = "train"
            if metrics and enabled:
                cases[-1]["pair_metric"] = POINCARE
        if controls or metrics:
            cases.append(dict(name=f"seed{seed}_geometry_frozen", seed=seed, geometry=True,
                              geometry_update="frozen", parameters=geometry))
        if metrics:
            cases[-1]["pair_metric"] = POINCARE
            for update in ("train", "frozen"):
                cases.append(dict(name=f"seed{seed}_flat" + ("_frozen" if update == "frozen" else ""),
                                  seed=seed, geometry=True, geometry_update=update, pair_metric=FLAT, parameters=geometry))
    request = dict(schema=f"spiraltorch.byte_corpus.request.v{version}", config=config, train_documents=train, validation_documents=valid,
                   train_batches=train_batches, validation_batches=valid_batches,
                   checkpoint_every=args.checkpoint_every, rate=rate, cases=cases)
    manifest = dict(schema=f"spiraltorch.byte_corpus.preparation.v{version}", source_revision=revision,
                    config=config, train_sources=train_info, validation_sources=valid_info,
                    training_window_selection="seed-20261009 shuffled complete windows, repeated only after an epoch",
                    validation_window_selection="evenly spaced, distinct complete windows, no score-based selection",
                    train_target_bytes=args.updates * args.batch * args.steps,
                    validation_target_bytes=count * args.steps, pairs=pair_records,
                    request_sha256=digest(encoded(request)),
                    criteria=dict(atol=ATOL, rtol=RTOL, geometry_delta_relative_l2=DELTA_RTOL,
                                  minimum_geometry_reference_delta_l2=1e-8),
                    scope="Small document-held-out repository-prose pilot, not broad language quality or equal-parameter/equal-compute comparison")
    request_version(request)
    if controls:
        manifest["geometry_control"] = "identical initial geometry weights; frozen tensors keep their full embedding pullback"
    if metrics:
        manifest["geometry_control"] = "Poincare/flat distance crossed with trained/frozen weights; all four geometric arms share initial bits and full embedding pullbacks"
        manifest["primary_contrasts"] = ["flat_minus_poincare_bpb", "flat_frozen_minus_poincare_frozen_bpb",
                                         "metric_by_training_interaction_bpb"]
        manifest["scope"] = "Fixed small corpus; geometric arms match parameters and SGD budget, not compute or initial function; ordinary has fewer parameters; no broad quality/speed claim"
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
    request = decode_json(raw)
    version = request_version(request)
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
                pair = None
                if case["geometry"]:
                    if case.get("pair_metric") == FLAT:
                        square = (coordinates[:, :, None, :] - coordinates[:, None, :, :]).square().sum(-1)
                        pair = (-functional.softplus(p[6 + i])[None, :, None, None] * (4 * square[:, None])).tril()
                    else:
                        pair = metric(coordinates, p[6 + i], config["curvature"])
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
                for descriptor, value, gradient in zip(case["parameters"], p, gradients):
                    if not (frozen(case) and descriptor["name"].startswith("geometry.")):
                        value -= request["rate"] * gradient
            training.append(dict(revision=step, ce=loss.item()))
            if step % request["checkpoint_every"] == 0 or step == len(train):
                evaluation.append(evaluate(step))
        reports.append(dict(name=case["name"], seed=case["seed"], geometry=case["geometry"],
                            parameter_tensors=len(p), parameter_scalars=sum(v.numel() for v in p),
                            training=training, validation=evaluation,
                            final_parameters=[v.detach().reshape(-1).tolist() for v in p]))
        if version >= 2:
            reports[-1].update(geometry_update=case["geometry_update"], trainable_parameter_scalars=trainable_scalars(case))
        if version == 3:
            reports[-1]["pair_metric"] = case.get("pair_metric")
    write_new(args.output, dict(schema=f"spiraltorch.byte_corpus.result.v{version}", engine="independent_pytorch", torch_version=torch.__version__,
                               dtype="float32", device="cpu", threads=1, request_sha256=digest(raw), cases=reports))


def close(actual, expected, label):
    if len(actual) != len(expected):
        raise ValueError(label + ": length mismatch")
    maximum = 0.
    for a, e in zip(actual, expected):
        if type(a) not in (int, float) or type(e) not in (int, float):
            raise ValueError(label + ": nonnumeric scalar")
        error = abs(a - e)
        if not math.isfinite(a) or not math.isfinite(e) or error > ATOL + RTOL * abs(e):
            raise ValueError(f"{label}: {a} != {e}")
        maximum = max(maximum, error)
    return maximum


def heldout_bpb(point):
    losses = point["batch_losses"]
    return math.fsum(losses) / len(losses) / math.log(2)


def compare(request_raw, reference_report, actual):
    request = decode_json(request_raw)
    version = request_version(request)
    if (reference_report.get("engine") != "independent_pytorch"
            or actual.get("engine") != "spiraltorch" or not actual.get("adapter")):
        raise ValueError("wrong request, reference, or measured engine identity")
    expected_cases = request["cases"]
    names = [case["name"] for case in expected_cases]
    if len(set(names)) != len(names):
        raise ValueError("duplicate request case")
    for report in [reference_report, actual]:
        if report["schema"] != f"spiraltorch.byte_corpus.result.v{version}" or report["request_sha256"] != digest(request_raw):
            raise ValueError("result source mismatch")
        if [c["name"] for c in report["cases"]] != names:
            raise ValueError("case identity/order mismatch")
    n = len(request["train_batches"])
    checkpoints = [0] + [i for i in range(1, n + 1) if i % request["checkpoint_every"] == 0 or i == n]
    summaries = []
    for spec, expected, got in zip(expected_cases, reference_report["cases"], actual["cases"]):
        name = spec["name"]
        for row in [expected, got]:
            if type(row["seed"]) is not int or row["seed"] != spec["seed"] or row["geometry"] is not spec["geometry"]:
                raise ValueError("case mode/seed mismatch")
            if version >= 2 and (row.get("geometry_update") != spec["geometry_update"]
                                or type(row.get("trainable_parameter_scalars")) is not int
                                or row["trainable_parameter_scalars"] != trainable_scalars(spec)):
                raise ValueError("case update policy or trainable count mismatch")
            if version == 3 and ("pair_metric" not in row or row["pair_metric"] != spec.get("pair_metric")):
                raise ValueError("case metric identity mismatch")
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
                if frozen(spec):
                    original = [struct.pack("<f", x) for x in initial]
                    if any([struct.pack("<f", x) for x in values] != original for values in (a, e)):
                        raise ValueError("frozen geometry parameter bits changed: " + desc["name"])
                    geometry_deltas.append(dict(name=desc["name"], frozen_bits_preserved=True))
                    continue
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
                              initial_heldout_bpb=heldout_bpb(got["validation"][0]),
                              final_heldout_bpb=heldout_bpb(got["validation"][-1]), geometry_deltas=geometry_deltas))
        if version >= 2:
            summaries[-1].update(geometry_update=spec["geometry_update"], trainable_parameter_scalars=trainable_scalars(spec))
        if version == 3:
            summaries[-1]["pair_metric"] = spec.get("pair_metric")
            summaries[-1]["reference_final_heldout_bpb"] = heldout_bpb(expected["validation"][-1])
            summaries[-1]["final_heldout_bpb_abs_error"] = abs(summaries[-1]["final_heldout_bpb"] - summaries[-1]["reference_final_heldout_bpb"])
    pairs = []
    for seed in sorted({c["seed"] for c in summaries}):
        paired = [c for c in summaries if c["seed"] == seed]
        if len(paired) != {1: 2, 2: 3, 3: 5}[version] or {c["geometry"] for c in paired} != {False, True}:
            raise ValueError("incomplete paired seed")
        plain = next(c for c in paired if not c["geometry"])
        geometry = next(c for c in paired if c["geometry"] and not frozen(c) and c.get("pair_metric", POINCARE) == POINCARE)
        pairs.append(dict(seed=seed, geometry_minus_ordinary_bpb=geometry["final_heldout_bpb"] - plain["final_heldout_bpb"]))
        if version >= 2:
            fixed = next(c for c in paired if frozen(c) and c.get("pair_metric", POINCARE) == POINCARE)
            pairs[-1].update(frozen_geometry_minus_ordinary_bpb=fixed["final_heldout_bpb"] - plain["final_heldout_bpb"],
                             learned_geometry_minus_frozen_bpb=geometry["final_heldout_bpb"] - fixed["final_heldout_bpb"])
        if version == 3:
            flat = next(c for c in paired if c.get("pair_metric") == FLAT and not frozen(c))
            flat_frozen = next(c for c in paired if c.get("pair_metric") == FLAT and frozen(c))
            flat_training = flat["final_heldout_bpb"] - flat_frozen["final_heldout_bpb"]
            pairs[-1].update(flat_minus_poincare_bpb=flat["final_heldout_bpb"] - geometry["final_heldout_bpb"],
                             flat_frozen_minus_poincare_frozen_bpb=flat_frozen["final_heldout_bpb"] - fixed["final_heldout_bpb"],
                             flat_learned_minus_frozen_bpb=flat_training,
                             metric_by_training_interaction_bpb=flat_training - pairs[-1]["learned_geometry_minus_frozen_bpb"])
            r = "reference_final_heldout_bpb"
            reference_contrasts = dict(geometry_minus_ordinary_bpb=geometry[r] - plain[r],
                frozen_geometry_minus_ordinary_bpb=fixed[r] - plain[r], learned_geometry_minus_frozen_bpb=geometry[r] - fixed[r],
                flat_minus_poincare_bpb=flat[r] - geometry[r], flat_frozen_minus_poincare_frozen_bpb=flat_frozen[r] - fixed[r],
                flat_learned_minus_frozen_bpb=flat[r] - flat_frozen[r],
                metric_by_training_interaction_bpb=(flat[r] - flat_frozen[r]) - (geometry[r] - fixed[r]))
            pairs[-1]["reference_contrasts"] = reference_contrasts
            pairs[-1]["absolute_torch_discrepancy"] = {k: abs(pairs[-1][k] - value) for k, value in reference_contrasts.items()}
            pairs[-1]["same_sign_as_torch"] = {k: ((pairs[-1][k] > 0) - (pairs[-1][k] < 0)) == ((v > 0) - (v < 0))
                                              for k, v in reference_contrasts.items()}
    result = dict(schema=f"spiraltorch.byte_corpus.comparison.v{version}", request_sha256=digest(request_raw),
                numerical_checks_passed=True, criteria=dict(atol=ATOL, rtol=RTOL, geometry_delta_relative_l2=DELTA_RTOL),
                cases=summaries, paired_deltas=pairs,
                mean_geometry_minus_ordinary_bpb=statistics.mean(p["geometry_minus_ordinary_bpb"] for p in pairs),
                scope="Fixed small corpus/recipe/seeds; different parameter counts and compute, no general quality or speed claim")
    if version == 3:
        for key in ("flat_minus_poincare_bpb", "flat_frozen_minus_poincare_frozen_bpb", "metric_by_training_interaction_bpb"):
            result["mean_" + key] = statistics.mean(p[key] for p in pairs)
        result["scope"] = "Poincare/flat geometric arms match parameters and SGD budget, not compute or initial functions; ordinary is a lower-parameter context arm; no general quality/speed claim"
    return result


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
    control = p.add_mutually_exclusive_group()
    control.add_argument("--frozen-geometry-control", action="store_true", help="v2: add an identical-initialization geometry-frozen arm per seed")
    control.add_argument("--metric-geometry-control", action="store_true", help="v3: cross Poincare/flat metric with trained/frozen geometry, plus ordinary (five arms per seed)")
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
        result = compare(args.request.read_bytes(), decode_json(args.reference.read_bytes()), decode_json(args.actual.read_bytes()))
        write_new(args.output, result)
        print(json.dumps({"numerical_checks_passed":True, "paired_deltas":result["paired_deltas"]}))


if __name__ == "__main__":
    main()
