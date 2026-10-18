"""Private bounded pilot orchestrator; all model semantics remain in Rust."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = value.encode() if isinstance(value, str) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
        out.write(raw)


def load(repo, name):
    spec = importlib.util.spec_from_file_location(name, repo / "tools" / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    import spiraltorch as st
    source = (args.output / "v3/request.json").read_bytes()
    prepared = st.nn.ByteCorpusBiasCalibration.from_json(source, [0,1,2,3], fisher_rao_control=True).prepare_wgpu()
    stages = [("v5", prepared, "verify_byte_corpus_preparation")]
    for version, prepared, verifier_name in stages:
        packet = dict(schema="spiraltorch.byte_corpus.prepared.v1", request_json=prepared.request_json, report_json=prepared.report_json)
        if version == "v8":
            verified_raw, verify = load(args.repo, verifier_name).verify_preparation(source, packet)
            require(verified_raw == prepared.request_json.encode(), "query preparation identity")
        else:
            verify = load(args.repo, verifier_name).verify(source, packet)
        write(args.output / (version+"-packet.json"), packet)
        write(args.output / (version+"-verification.json"), verify)
        write(args.output / (version+"-request.json"), prepared.request_json)
        source = prepared.request_json.encode()
        if version == "v5":
            stages.append(("v7", st.nn.ByteCorpusMixtureControls.from_json(source).prepare_wgpu(), "verify_byte_corpus_mixture_preparation"))
        elif version == "v7":
            stages.append(("v8", st.nn.ByteCorpusQueryMixtureControls.from_json(source).prepare(), "byte_corpus_query"))
    raw = prepared.request_json.encode()
    new = json.loads(raw)
    old_raw = args.historical.read_bytes()
    old = json.loads(old_raw)
    require(len(new["train_batches"]) == 512 and len(new["cases"]) == 25, "bounded budget/coverage")
    for key in ("config", "train_documents", "validation_documents", "validation_batches", "rate", "checkpoint_every"):
        require(new[key] == old[key], "historical recipe changed: " + key)
    require(new["train_batches"][:128] == old["train_batches"], "historical train prefix changed")
    historic = {c["name"]:c for c in old["cases"] if c["seed"] == 11}
    require(set(historic) == {c["name"] for c in new["cases"]}, "historical arm coverage")
    for case in new["cases"]:
        expected = historic[case["name"]]
        require({k:v for k,v in case.items() if k != "parameters"} ==
                {k:v for k,v in expected.items() if k != "parameters"}, "arm policy changed")
        require(len(case["parameters"]) == len(expected["parameters"]), "parameter coverage")
        for a,b in zip(case["parameters"],expected["parameters"]):
            require(a["name"] == b["name"] and a["shape"] == b["shape"], "initial shape/name changed")
            require([struct.pack("<f",v) for v in a["values"]] == [struct.pack("<f",v) for v in b["values"]], "initial float32 bits changed: "+case["name"]+":"+a["name"])
    study = prepared.study()
    study.validate_segment()
    write(args.output / "request.json", prepared.request_json)
    write(args.output / "prepared.json", dict(schema="spiraltorch.byte_corpus.budget_preparation.v1",
        plan_sha256=digest((args.output/"PLAN.md").read_bytes()), runner_sha256=digest(Path(__file__).read_bytes()),
        historical_sha256=digest(old_raw), request_sha256=digest(raw),
        all_initial_bits_equal=True, historical_train_prefix_equal=True,
        arms=25, updates=512, seed=11, checkpoint_size_bound=study.checkpoint_size_bound()))
    print(json.dumps(dict(prepared=True, request_sha256=digest(raw), checkpoint_size_bound=study.checkpoint_size_bound())), flush=True)


def run(args):
    import spiraltorch as st
    raw = (args.output / "request.json").read_bytes()
    preflight = json.loads((args.output / "prepared.json").read_bytes())
    require(digest(raw) == preflight["request_sha256"], "request drift")
    require(digest((args.output/"PLAN.md").read_bytes()) == preflight["plan_sha256"], "plan drift")
    require(digest(Path(__file__).read_bytes()) == preflight["runner_sha256"], "runner drift")
    study = st.nn.ByteCorpusStudy.from_json(raw)
    checkpoint = None
    for stop in (128,256,512):
        started = time.monotonic()
        print(json.dumps(dict(event="segment_started", stop_after=stop)), flush=True)
        segment = study.advance_wgpu(stop, resume=checkpoint)
        checkpoint = segment.checkpoint
        require(checkpoint.completed_updates == stop and checkpoint.request_sha256 == study.request_sha256, "segment lineage/cursor")
        write(args.output / f"native-{stop}.json", segment.report_json)
        write(args.output / f"native-checkpoint-{stop}.json", checkpoint.to_json())
        elapsed = time.monotonic()-started
        write(args.output / f"native-segment-{stop}.json", dict(stop_after=stop, seconds=elapsed,
             request_sha256=study.request_sha256, checkpoint_sha256=digest(checkpoint.to_json().encode()),
             report_sha256=digest(segment.report_json.encode())))
        print(json.dumps(dict(event="segment_completed", stop_after=stop, seconds=elapsed)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    for name in ("repo","output","historical"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    {"prepare":prepare,"run":run}[args.command](args)


if __name__ == "__main__":
    main()
