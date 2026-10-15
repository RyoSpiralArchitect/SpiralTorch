"""Preflight and serial fresh execution of the fixed capacity/budget matrix."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

REVISION = "02f211f72a029ce6455aed2c706ca35989490c03"
WIDTHS, SEEDS, STOPS = (16, 32), (11, 29, 47), (0, 256, 512, 1024)
RECIPE = dict(kind="adamw_zero_rate_freeze.v1", beta1=.9, beta2=.999,
              epsilon=1e-5, weight_decay=.01, geometry_weight_decay=0.)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    # Dictionary insertion order is not part of parameter identity. Float
    # spellings, including signed zero, remain exact in these Python fixtures.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as output:
        output.write(data)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def names():
    return [f"w{width}-seed{seed}" for width in WIDTHS for seed in SEEDS]


def validate_request(protocol, request, width, seed, documents):
    require(request["schema"] == protocol.REQUEST_V9 and request["design_schema"] == protocol.REQUEST_V2, "design")
    config = dict(batch=2, steps=32, width=width, hidden=width*2, heads=2,
                  blocks=[False], geometry_cols=4, curvature=-.75)
    require(request["config"] == config and request["optimizer"] == RECIPE, "architecture/optimizer")
    require(request["rate"] == protocol.f32(.001) and request["checkpoint_every"] == 64, "rate/schedule")
    train, valid = documents
    require(request["train_documents"] == train and request["validation_documents"] == valid, "ordered declared corpus")
    rng = random.Random(20261009)
    candidates = protocol.windows(train, 32)
    require(bool(candidates), "training corpus has no complete window")
    selected = []
    while len(selected) < 2048:
        epoch = candidates.copy()
        rng.shuffle(epoch)
        selected.extend(epoch)
    expected = [[list(v) for v in selected[i:i+2]] for i in range(0, 2048, 2)]
    require(request["train_batches"] == expected, "training order/budget")
    valid_candidates = protocol.windows(valid, 32)
    require(len(valid_candidates) >= 64, "validation corpus too small")
    selected = [list(valid_candidates[i*len(valid_candidates)//64]) for i in range(64)]
    require(request["validation_batches"] == [selected[i:i+2] for i in range(0,64,2)], "validation windows")
    plain, geometry = protocol.parameters(config, seed)
    expected = [dict(name=f"seed{seed}_ordinary", seed=seed, geometry=False, geometry_update="train", parameters=plain),
                dict(name=f"seed{seed}_geometry", seed=seed, geometry=True, geometry_update="train", parameters=geometry),
                dict(name=f"seed{seed}_geometry_frozen", seed=seed, geometry=True, geometry_update="frozen", parameters=geometry)]
    require(canonical(request["cases"]) == canonical(expected), "initial parameter bits/arm coverage")
    protocol.request_version(request)


def preflight(repo, root, historical):
    import spiraltorch as st
    protocol = load(repo / "tools/byte_corpus_study.py")
    plan = Path(__file__).with_name("PLAN.md").read_bytes()
    require((root / "PLAN.md").read_bytes() == plan, "plan drift")
    train_paths = ["docs/byte_corpus_learning.md", "docs/byte_corpus_calibration.md"]
    valid_paths = ["docs/resident_adamw.md"]
    documents = [[list(subprocess.check_output(["git", "show", REVISION+":"+path], cwd=repo))
                  for path in paths] for paths in (train_paths, valid_paths)]
    old_raw = (historical / "request.json").read_bytes()
    previous = json.loads((repo / "benchmarks/results/2026-10-15-language-adamw-256/selection.json").read_bytes())
    require(digest(old_raw) == previous["request_sha256"], "historical request not pinned by public result")
    old = protocol.decode_json(old_raw)
    entries = []
    for width in WIDTHS:
        for seed in SEEDS:
            name = f"w{width}-seed{seed}"
            data = (root / name / "request.json").read_bytes()
            request = protocol.decode_json(data)
            validate_request(protocol, request, width, seed, documents)
            study = st.nn.ByteCorpusStudy.from_json(data)
            for stop in STOPS:
                study.validate_segment(stop)
            bound = study.checkpoint_size_bound()
            require(bound * len(STOPS) <= st.nn.BYTE_CORPUS_STUDY_MAX_BYTES, "combined snapshot budget")
            require(request["train_batches"][:256] == old["train_batches"], "historical data prefix")
            if width == 16:
                require(canonical(request["cases"]) == canonical([c for c in old["cases"] if c["seed"] == seed]), "historical initial bits")
            entries.append(dict(name=name, width=width, seed=seed, request_sha256=digest(data),
                                checkpoint_size_bound=bound,
                                parameter_scalars=[sum(len(p["values"]) for p in c["parameters"]) for c in request["cases"]]))
    sources = [Path(__file__).resolve(), Path(__file__).resolve().with_name("worker.py"),
               repo / "tools/byte_corpus_study.py", repo / "tools/byte_corpus_optimizer.py",
               repo / "tools/run_byte_corpus_segments.py",
               repo / "tools/generate_resident_byte_decoder_torch_fixture.py",
               repo / "tools/generate_causal_zspace_wave_torch_fixture.py",
               repo / "tools/generate_poincare_bias_torch_fixture.py"]
    manifest = dict(schema="spiraltorch.byte_corpus.capacity_preparation.v1", plan_sha256=digest(plan),
                    corpus_revision=REVISION, historical_request_sha256=digest(old_raw), stops=list(STOPS),
                    all_data_prefixes_and_width16_initial_bits_equal=True, entries=entries,
                    sources=[dict(path=str(path.relative_to(repo)), sha256=digest(path.read_bytes())) for path in sources],
                    native_extension_sha256=digest(Path(st._rs.__file__).read_bytes()))
    write(root / "prepared.json", manifest)
    print(json.dumps(manifest, indent=2))


def validated_inputs(repo, root, *, allow_recorded_launcher=False):
    manifest = json.loads((root / "prepared.json").read_bytes())
    require([e["name"] for e in manifest["entries"]] == names(), "complete fixed matrix")
    require(manifest["stops"] == list(STOPS), "saved cursors")
    require(digest((root / "PLAN.md").read_bytes()) == manifest["plan_sha256"], "plan drift")
    require((root / "PLAN.md").read_bytes() == Path(__file__).with_name("PLAN.md").read_bytes(), "source plan drift")
    for row in manifest["sources"]:
        current = digest((repo/row["path"]).read_bytes())
        if (current != row["sha256"] and allow_recorded_launcher
                and (repo/row["path"]).resolve() == Path(__file__).resolve()):
            # Read-only qualification may inspect a completed run's original
            # launcher. Execution never accepts this historical fallback.
            current = digest((root/"launch-source.py").read_bytes())
        require(current == row["sha256"], "runner/source drift")
    for entry in manifest["entries"]:
        require(digest((root / entry["name"] / "request.json").read_bytes()) == entry["request_sha256"], "request drift")
    return manifest


def execute(repo, root, reference):
    require(sys.flags.isolated, "launch execution with isolated Python (-I)")
    repo, root = repo.resolve(), root.resolve()
    manifest = validated_inputs(repo, root)
    prepared_hash = digest((root/"prepared.json").read_bytes())
    claim = "reference" if reference else "native"
    for entry in manifest["entries"]:
        destination = root / entry["name"] / ("torch.json" if reference else "native")
        require(not destination.exists(), "matrix contains an already claimed run")
    write(root / (claim + "-launch.json"), dict(prepared_sha256=prepared_hash,
                                               pid=os.getpid(), before_execution=True))
    for entry in manifest["entries"]:
        require(digest((root/"prepared.json").read_bytes()) == prepared_hash, "preparation drift between conditions")
        validated_inputs(repo, root)
        directory = root / entry["name"]
        print(json.dumps(dict(event="started", engine=claim, study=entry["name"])), flush=True)
        args = [sys.executable, "-I", "-B", str(Path(__file__).resolve().with_name("worker.py")),
                "--repo", str(repo), "--root", str(root), "--name", entry["name"],
                "--engine", claim, "--prepared-sha256", prepared_hash]
        subprocess.run(args, cwd=repo, check=True)
        require(digest((root/"prepared.json").read_bytes()) == prepared_hash, "preparation drift during condition")
        validated_inputs(repo, root)
        child = json.loads((directory/(claim+"-child.json")).read_bytes())
        expected = dict(prepared_sha256=prepared_hash, request_sha256=entry["request_sha256"],
                        native_extension_sha256=None if reference else manifest["native_extension_sha256"],
                        engine=claim, isolated=True, input_bytes_bound=True)
        require(child == expected, "child execution receipt differs")
        print(json.dumps(dict(event="completed", engine=claim, study=entry["name"])), flush=True)
    write(root / (claim + "-completed.json"), dict(prepared_sha256=prepared_hash, studies=names()))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "native", "reference"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    args = parser.parse_args()
    args.repo, args.root = args.repo.resolve(), args.root.resolve()
    if args.historical is not None:
        args.historical = args.historical.resolve()
    if args.command == "preflight":
        require(args.historical is not None, "historical request is required")
        preflight(args.repo, args.root, args.historical)
    else:
        execute(args.repo, args.root, args.command == "reference")


if __name__ == "__main__":
    main()
