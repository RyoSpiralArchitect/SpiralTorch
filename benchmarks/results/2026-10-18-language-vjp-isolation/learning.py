"""Repeat the pinned language study; do not loosen or replace its strict gate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

PINS = {
    "request": "b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
    "before": "858a5576ba95c643fc223044fe6eaab4f502fa9afd0fc15b744eaa7bdb0b3dfe",
    "torch": "96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(paths):
    raw = {name: path.read_bytes() for name, path in paths.items()}
    if {name: sha(value) for name, value in raw.items()} != PINS:
        raise ValueError("pinned study input drift")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "prior", "reference", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    repo, prior, reference, output = (getattr(args, key).resolve() for key in ("repo","prior","reference","output"))
    paths = {"request": prior/"language/request.json", "before": prior/"language/native.json",
             "torch": reference/"w16-seed11/torch.json"}
    raw = read_inputs(paths)
    helper_path = repo/"benchmarks/results/2026-10-16-attention-adamw-numerics/language_probe.py"
    builder_path = repo/"benchmarks/results/2026-10-16-attention-shift-invariance/language_probe.py"
    helpers, builder = load(helper_path), load(builder_path)
    source_paths = {repo/"Cargo.toml", repo/"Cargo.lock", Path(__file__).resolve(),
                    Path(__file__).with_name("PLAN.md").resolve(), helper_path, builder_path,
                    *repo.glob("tools/byte_corpus*.py"),
                    *repo.glob("benchmarks/results/2026-10-16-language-capacity-budget/*.py")}
    for crate in ("st-nn", "st-backend-wgpu", "st-kernel-contracts"):
        source_paths.update(p for p in (repo/"crates"/crate).rglob("*")
                            if p.is_file() and p.suffix in (".rs", ".wgsl", ".toml"))
    sources = {p: sha(p.read_bytes()) for p in source_paths}
    output.mkdir(mode=0o700, exist_ok=False)
    with helpers.private_writer(output/"request.json") as f:
        f.write(raw["request"])
    with helpers.private_writer(output/"runner.py") as f:
        f.write(Path(__file__).read_bytes())
    binary, binary_hash = builder.build_native(repo, output, sources, helpers)
    archived = output/"resident_byte_learning"
    shutil.copyfile(binary, archived)
    archived.chmod(0o700)
    if sha(archived.read_bytes()) != binary_hash:
        raise ValueError("archived binary drift")
    receipt_hash = sha((output/"build-receipt.json").read_bytes())
    helpers.write(output/"started.json", {"pins": PINS, "binary_sha256": binary_hash,
                                          "build_receipt_sha256": receipt_hash})

    def verify():
        read_inputs(paths)
        builder.check_sources(sources)
        if sha(archived.read_bytes()) != binary_hash or (output/"request.json").read_bytes() != raw["request"]:
            raise ValueError("executed input/binary drift")

    verify()
    started = time.monotonic()
    with helpers.private_writer(output/"native.json") as stdout, helpers.private_writer(output/"native.log") as stderr:
        subprocess.run([str(archived), str(output/"request.json"), "--checkpoint-out", str(output/"checkpoint.json")],
                       cwd=repo, stdout=stdout, stderr=stderr, check=True, timeout=7200)
    elapsed = time.monotonic() - started
    verify()
    protocol = load(repo/"tools/byte_corpus_study.py")
    diagnostics = load(repo/"benchmarks/results/2026-10-16-language-capacity-budget/diagnose.py")
    expected = protocol.decode_json(raw["torch"])
    actual_raw = (output/"native.json").read_bytes()
    actual = protocol.decode_json(actual_raw)
    result = {"schema":"spiraltorch.language_range_followup.v1", "pins":PINS,
              "binary_sha256":binary_hash, "build_receipt_sha256":receipt_hash,
              "actual_sha256":sha(actual_raw), "checkpoint_sha256":sha((output/"checkpoint.json").read_bytes()),
              "execution_wall_seconds":elapsed,
              "before":helpers.summarize(protocol,diagnostics,raw["request"],expected,protocol.decode_json(raw["before"])),
              "after":helpers.summarize(protocol,diagnostics,raw["request"],expected,actual),
              "scope":"one seed; cumulative numerical repair; unchanged strict gate; not a quality or speed claim"}
    verify()
    helpers.write(output/"comparison.json", result)
    print(json.dumps({"completed":True,"cases":len(actual["cases"]),"strict_gate":result["after"]["strict_gate"]["passed"],"seconds":elapsed}))


if __name__ == "__main__":
    main()
