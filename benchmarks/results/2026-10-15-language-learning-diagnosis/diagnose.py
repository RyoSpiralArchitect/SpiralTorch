"""Frozen-weight language diagnosis, not a promotion or training-quality test."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
        out.write(raw)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packed(values):
    require(all(math.isfinite(v) for v in values), "non-finite logits")
    return struct.pack("<" + "f" * len(values), *values)


def probabilities(row):
    maximum = max(row)
    exp = [math.exp(v - maximum) for v in row]
    denominator = math.fsum(exp)
    return [v / denominator for v in exp], (maximum, math.log(denominator))


def score(values, targets):
    require(len(values) == len(targets) * 256, "logit/target shape mismatch")
    require(targets and all(type(v) is int and 0 <= v < 256 for v in targets), "invalid targets")
    require(all(math.isfinite(v) for v in values), "non-finite logits")
    nll, entropy, correct, spaces = [], [], 0, 0
    for i, target in enumerate(targets):
        row = values[i * 256:(i + 1) * 256]
        p, (maximum, shifted_logsum) = probabilities(row)
        nll.append((maximum - row[target]) + shifted_logsum)
        entropy.append(-math.fsum(v * math.log(v) for v in p if v))
        prediction = max(range(256), key=row.__getitem__)
        correct += prediction == target
        spaces += prediction == 32
    n = len(targets)
    return dict(targets=n, mean_ce=math.fsum(nll) / n,
                mean_entropy=math.fsum(entropy) / n, top1_correct=correct,
                top1_space=spaces)


def rows(documents, selection, steps):
    result = [bytes(documents[d][s:s + steps + 1]) for d, s in selection]
    require(all(len(r) == steps + 1 for r in result), "incomplete window")
    return result


def bound_inputs(plan):
    for name, expected in plan["sources"].items():
        require(sha(Path(name).read_bytes()) == expected, "source drift: " + name)
    archive = Path(plan["archive"])
    request_raw = (archive / "input/request.json").read_bytes()
    saved_raw = (archive / "browser-checkpoint-128.json").read_bytes()
    request, saved = json.loads(request_raw), json.loads(saved_raw)
    require(saved["request_sha256"] == sha(request_raw) and saved["completed_updates"] == 128, "checkpoint lineage")
    gate = json.loads(Path(plan["gate"]).read_bytes())
    cohorts = [("historical-heldout", request["validation_documents"], request["validation_batches"])]
    cohorts += [(c["name"], gate["documents"], c["batches"]) for c in gate["cohorts"]]
    require(len(cohorts) == 4, "expected historical heldout and three named cohorts")
    return request_raw, saved_raw, request, saved, cohorts


def fitted_baselines(request, cohorts, completed_updates):
    require(type(completed_updates) is int and 0 < completed_updates <= len(request["train_batches"]),
            "baseline requires a valid completed training cursor")
    t = request["config"]["steps"]
    unigram, bigram = Counter(), Counter()
    for batch in request["train_batches"][:completed_updates]:
        for window in rows(request["train_documents"], batch, t):
            unigram.update(window[1:])
            bigram.update(zip(window[:-1], window[1:]))
    predecessor = Counter()
    for (x, _), count in bigram.items():
        predecessor[x] += count
    total = sum(unigram.values())
    baselines = []
    for name, documents, batches in cohorts:
        pairs = [(x,y) for batch in batches for row in rows(documents, batch, t) for x,y in zip(row[:-1],row[1:])]
        uni = [-math.log((unigram[y]+1)/(total+256)) for _,y in pairs]
        bi = [-math.log((bigram[x,y]+1)/(predecessor[x]+256)) for x,y in pairs]
        baselines.append(dict(cohort=name, targets=len(pairs), unigram_ce=math.fsum(uni)/len(uni), bigram_ce=math.fsum(bi)/len(bi)))
    return dict(baselines=baselines, training_target_presentations=total,
                available_nonoverlapping_training_targets=sum((len(d)-1)//t*t for d in request["train_documents"]),
                training_target_entropy=-math.fsum(n/total*math.log(n/total) for n in unigram.values()))


def prepare(args):
    req = args.archive / "input/request.json"
    saved = args.archive / "browser-checkpoint-128.json"
    report = args.archive / "browser.json"
    request = json.loads(req.read_bytes())
    selection = []
    for seed in (11, 23, 37):
        for mode in ("ordinary", "train", "frozen"):
            found = [c for c in request["cases"] if c["seed"] == seed and
                     ((not c["geometry"]) if mode == "ordinary" else
                      c.get("mixture", {}).get("metrics") == ["poincare_squared.v1", "euclidean_chord_squared.v1"]
                      and c.get("mixture", {}).get("query_projection", {}).get("weight_update") == mode)]
            require(len(found) == 1, "ambiguous case selection")
            selection.append(dict(seed=seed, mode=mode, name=found[0]["name"]))
    sources = {str(p.resolve()): sha(p.read_bytes()) for p in (req, saved, report, args.gate)}
    helpers = ["byte_corpus_study", "verify_byte_corpus_resume", "byte_corpus_query",
               "byte_corpus_mixture", "generate_resident_byte_decoder_torch_fixture",
               "generate_causal_zspace_wave_torch_fixture", "generate_poincare_bias_torch_fixture"]
    sources.update({str(args.repo / "tools" / (name + ".py")):
                    sha((args.repo / "tools" / (name + ".py")).read_bytes()) for name in helpers})
    sources[str(Path(__file__).resolve())] = sha(Path(__file__).read_bytes())
    plan = dict(schema="spiraltorch.byte_corpus.frozen_diagnosis_plan.v1",
                source_revision=subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
                repo=str(args.repo), archive=str(args.archive), gate=str(args.gate),
                sources=sources, selections=selection, checkpoint_updates=128,
                context_batches_per_cohort=2, generation_bytes=32,
                generation_prompts="First two windows of first pre-existing named cohort; no prompt search",
                baseline="Add-one unigram and conditional byte bigram, fitted only to actual training-window transitions",
                context="Reverse first T-1 input bytes, preserve final input and next target; last-position diagnosis only, not a quality estimate",
                parity=dict(atol=3e-6, rtol=5e-5), new_training_updates=0,
                scope="Posthoc diagnosis of nine historical browser-trained checkpoints. Native WGPU vs CPU Torch evaluates identical weights. No unseen benchmark, broad quality, speed or geometry-superiority claim.")
    write(args.output, plan)
    print(json.dumps(dict(prepared=True, cases=len(selection), plan_sha256=sha(args.output.read_bytes()))), flush=True)


def context_pair(original, reversed_values):
    require(len(original) == len(reversed_values), "context comparison shape")
    difference = [a-b for a, b in zip(original, reversed_values)]
    tv = []
    for i in range(0, len(original), 256):
        p, _ = probabilities(original[i:i+256])
        q, _ = probabilities(reversed_values[i:i+256])
        tv.append(0.5 * math.fsum(abs(a-b) for a,b in zip(p,q)))
    return dict(logits_rms=math.sqrt(math.fsum(v*v for v in difference)/len(difference)),
                mean_total_variation=math.fsum(tv)/len(tv))


def torch_forward(repo, request, case, model_json, cursor):
    import torch
    import torch.nn.functional as f
    source = load_module(repo / "tools/verify_byte_corpus_resume.py")
    stored = source.stored_parameters(json.loads(model_json), cursor, mixture=case.get("mixture"))
    require([p["shape"] for p in stored] == [p["shape"] for p in case["parameters"]], "checkpoint topology")
    p = [torch.tensor(v["values"], dtype=torch.float32, device="cpu").reshape(v["shape"]) for v in stored]
    cfg = request["config"]
    require(cfg["blocks"] == [False], "this pinned reference supports one non-Topos block")
    block = load_module(repo / "tools/generate_resident_byte_decoder_torch_fixture.py").block_forward
    wave = load_module(repo / "tools/generate_causal_zspace_wave_torch_fixture.py").wave
    metric = load_module(repo / "tools/generate_poincare_bias_torch_fixture.py").metric
    mixture = load_module(repo / "tools/byte_corpus_mixture.py").pair_bias

    def forward(windows):
        with torch.no_grad():
            ids = torch.tensor([list(row[:-1]) for row in windows], dtype=torch.long, device="cpu")
            pos = torch.arange(cfg["steps"], device="cpu").expand_as(ids)
            x = f.embedding(ids, p[0]) + f.embedding(pos, p[1])
            offset, pair = 2, None
            if case["geometry"]:
                coordinates, _ = wave(x @ p[2] + p[3], p[4], p[5],
                                      torch.zeros((cfg["batch"], cfg["geometry_cols"]), device="cpu"), cfg["curvature"])
                pair = mixture(coordinates, p[6], p[7], case["mixture"], 0,
                               cfg["curvature"], metric, None, p[8:10])
                offset = 10
            x = block(x, p[offset:offset+12], cfg["heads"], None, pair, False, module_residual=True)
            head = p[offset+12:]
            require(len(head) == 4, "head topology")
            logits = f.layer_norm(x, (cfg["width"],), head[0], head[1], eps=1e-5) @ head[2] + head[3]
            return logits.reshape(-1).tolist()
    return forward


def run(args):
    raw_plan = args.plan.read_bytes()
    plan = json.loads(raw_plan)
    repo = Path(plan["repo"])
    request_raw, saved_raw, request, saved, cohorts = bound_inputs(plan)
    t, b = request["config"]["steps"], request["config"]["batch"]
    require(t == 32 and b == 2, "pinned shape changed")
    baselines = fitted_baselines(request, cohorts, saved["completed_updates"])
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if args.engine == "native":
        import spiraltorch as st
        study = st.nn.ByteCorpusStudy.from_json(request_raw)
        checkpoint = study.checkpoint_from_json(saved_raw)
        runtime = dict(package_version=st.__version__)
    else:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        runtime = dict(torch_version=torch.__version__, device="cpu", dtype="float32", threads=1)
    result = dict(schema="spiraltorch.byte_corpus.frozen_diagnosis.v1", engine=args.engine,
                  plan_sha256=sha(raw_plan), runtime=runtime, **baselines, cases=[])
    for ci, selection in enumerate(plan["selections"]):
        case = next(c for c in request["cases"] if c["name"] == selection["name"])
        model = next(c for c in saved["cases"] if c["name"] == selection["name"])
        record = dict(selection=selection, model_json_sha256=sha(model["model_json"].encode()), cohorts=[])
        if args.engine == "native":
            artifact = study.case_artifact(checkpoint, case["name"], policy="fixed_window_frozen.v1")
            study.validate_artifact(checkpoint, artifact)
            owner = artifact.restore_wgpu()
            adapter = owner.tensor_device().adapter_info()
            require(adapter["device_type"] != "Cpu", "not a real GPU")
            runtime["adapter"] = adapter
            record["artifact_id"] = artifact.artifact_id
            def forward(windows):
                host = st.nn.ByteLmBatch.from_windows(windows)
                return owner.forward(owner.prepare_batch(host)).prediction_tensor().snapshot().read_values()
        else:
            forward = torch_forward(repo, request, case, model["model_json"], 128)
        for ni, (name, documents, batches) in enumerate(cohorts):
            scores, targets, old_last, new_last, next_targets = [], [], [], [], []
            for i, selected in enumerate(batches):
                window = rows(documents, selected, t)
                values = forward(window)
                require(len(values) == b*t*256, "forward shape")
                scores.extend(values)
                targets.extend(v for row in window for v in row[1:])
                if i < plan["context_batches_per_cohort"]:
                    shuffled = [row[:t-1][::-1] + row[t-1:] for row in window]
                    changed = forward(shuffled)
                    for row in range(b):
                        start = ((row+1)*t-1)*256
                        old_last.extend(values[start:start+256])
                        new_last.extend(changed[start:start+256])
                        next_targets.append(window[row][-1])
            blob = packed(scores + new_last)
            filename = f"case-{ci}-cohort-{ni}.f32le"
            write(args.output / filename, blob)
            record["cohorts"].append(dict(name=name, metrics=score(scores, targets),
                original_last=score(old_last,next_targets), reversed_prefix_last=score(new_last,next_targets),
                context_sensitivity=context_pair(old_last,new_last), file=filename, sha256=sha(blob)))
        prompts = [r[:-1] for r in rows(cohorts[1][1], cohorts[1][2][0], t)]
        if args.engine == "native":
            generation = json.loads(owner.generate_greedy(prompts, max_new_bytes=plan["generation_bytes"]))
            require(generation["artifact_id"] == artifact.artifact_id and owner.attempted_updates == 128, "generation owner changed")
            generated = generation["generated_bytes"]
            del owner
        else:
            generated = [[] for _ in prompts]
            for _ in range(plan["generation_bytes"]):
                values = forward([p+b"\xff" for p in prompts])
                for row in range(b):
                    start = ((row+1)*t-1)*256
                    token = max(range(256), key=values[start:start+256].__getitem__)
                    generated[row].append(token)
                    prompts[row] = prompts[row][1:] + bytes([token])
        record["generated_bytes"] = generated
        result["cases"].append(record)
        write(args.output / f"case-{ci}.json", record)
        print(json.dumps(dict(engine=args.engine, name=case["name"], ce=record["cohorts"][0]["metrics"]["mean_ce"])), flush=True)
    write(args.output / "result.json", result)


def read_scores(record, directory, cohort, ci, ni, batch, steps, context_batches):
    name, documents, selections = cohort
    require(record["name"] == name and record["file"] == f"case-{ci}-cohort-{ni}.f32le", "score selection identity")
    raw = (directory/record["file"]).read_bytes()
    require(sha(raw) == record["sha256"], "raw score hash")
    n = len(selections)*batch*steps*256
    last_count = min(context_batches, len(selections))*batch*256
    require(len(raw) == (n+last_count)*4, "raw score shape")
    values = list(struct.unpack("<"+"f"*(n+last_count),raw))
    require(all(math.isfinite(v) for v in values), "non-finite comparison")
    scores, new_last = values[:n], values[n:]
    targets, last_targets, old_last = [], [], []
    for i, selected in enumerate(selections):
        windows = rows(documents, selected, steps)
        targets.extend(v for row in windows for v in row[1:])
        if i < context_batches:
            last_targets.extend(row[-1] for row in windows)
            for row in range(batch):
                start = ((i*batch+row+1)*steps-1)*256
                old_last.extend(scores[start:start+256])
    expected = dict(metrics=score(scores,targets), original_last=score(old_last,last_targets),
                    reversed_prefix_last=score(new_last,last_targets),
                    context_sensitivity=context_pair(old_last,new_last))
    return values, expected


def verified_scores(record, directory, cohort, ci, ni, batch, steps, context_batches):
    values, expected = read_scores(record,directory,cohort,ci,ni,batch,steps,context_batches)
    require(all(record[k] == v for k,v in expected.items()), "diagnostic summary differs from pinned raw scores")
    return values


def require_engines(native, reference):
    require(native["schema"] == reference["schema"] == "spiraltorch.byte_corpus.frozen_diagnosis.v1", "report schema")
    require(native["engine"] == "native" and reference["engine"] == "torch", "expected native versus torch engines")


def validate_generation(rows, batch, length):
    require(type(batch) is int and batch > 0 and type(length) is int and length > 0,
            "generation requires a positive planned shape")
    require(type(rows) is list and len(rows) == batch, "generation row coverage")
    for row in rows:
        require(type(row) is list and len(row) == length, "generation length coverage")
        require(all(type(value) is int and 0 <= value < 256 for value in row), "invalid generated byte")


def compare(args):
    plan = json.loads(args.plan.read_bytes())
    native = json.loads((args.native / "result.json").read_bytes())
    reference = json.loads((args.reference / "result.json").read_bytes())
    require_engines(native,reference)
    require(native["plan_sha256"] == reference["plan_sha256"] == sha(args.plan.read_bytes()), "plan mismatch")
    _, _, request, saved, cohorts = bound_inputs(plan)
    baselines = fitted_baselines(request,cohorts,saved["completed_updates"])
    for k,v in baselines.items():
        require(native[k] == reference[k] == v, "baseline mismatch")
    require(len(native["cases"]) == len(reference["cases"]) == len(plan["selections"]), "case coverage")
    rows_out = []
    for ci, (a, e, selected) in enumerate(zip(native["cases"], reference["cases"], plan["selections"])):
        model = next(c for c in saved["cases"] if c["name"] == selected["name"])
        require(a["selection"] == e["selection"] == selected and a["model_json_sha256"] == e["model_json_sha256"] == sha(model["model_json"].encode()), "checkpoint case mismatch")
        require(len(a["cohorts"]) == len(e["cohorts"]) == len(cohorts) == 4, "cohort coverage")
        for record in (a, e):
            validate_generation(record["generated_bytes"],request["config"]["batch"],plan["generation_bytes"])
        errors = []
        for ni, (x,y,cohort) in enumerate(zip(a["cohorts"],e["cohorts"],cohorts)):
            verify_args = (cohort,ci,ni,request["config"]["batch"],request["config"]["steps"],plan["context_batches_per_cohort"])
            av = verified_scores(x,args.native,*verify_args)
            ev = verified_scores(y,args.reference,*verify_args)
            differences = [abs(v-w) for v,w in zip(av,ev)]
            passed = all(d <= plan["parity"]["atol"] + plan["parity"]["rtol"]*abs(w) for d,w in zip(differences,ev))
            errors.append(dict(cohort=x["name"], passed=passed, compared_logits=len(av), max_abs=max(differences),
                               absolute_ce_error=abs(x["metrics"]["mean_ce"]-y["metrics"]["mean_ce"])))
        rows_out.append(dict(selection=selected, errors=errors, greedy_bytes_equal=a["generated_bytes"] == e["generated_bytes"]))
    result = dict(schema="spiraltorch.byte_corpus.frozen_diagnosis_comparison.v1", plan_sha256=sha(args.plan.read_bytes()),
                  native_result_sha256=sha((args.native/"result.json").read_bytes()), reference_result_sha256=sha((args.reference/"result.json").read_bytes()),
                  cases=rows_out, passed=all(c["greedy_bytes_equal"] and all(e["passed"] for e in c["errors"]) for c in rows_out))
    write(args.output,result)
    print(json.dumps(dict(passed=result["passed"], cases=len(rows_out))),flush=True)
    require(result["passed"], "numeric parity failed; preserve evidence and investigate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("repo", "archive", "gate", "output"):
        prep.add_argument("--"+name,type=Path,required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--engine",choices=("native","torch"),required=True)
    for name in ("plan","output"):
        runner.add_argument("--"+name,type=Path,required=True)
    checker = sub.add_parser("compare")
    for name in ("plan","native","reference","output"):
        checker.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    {"prepare":prepare,"run":run,"compare":compare}[args.command](args)


if __name__ == "__main__":
    main()
