"""Pinned, private long-run clipping comparison and metrics-only publication."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PINS = {
    "request": "b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
    "torch": "96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b",
    "native": "ddc47f8bcd053391daf265cb1a3f914bdc78bceeb8c4ea4371792772ef80f280",
    "wheel": "74fdba8c31dbca9e6200992467447e986fbfc742a681c9bb85e8fdc78c0f36d9",
}
NATIVE = "f1539edd9305f54ce8ac25534b267cfca8e85a8d7520058e12d727440af5d305"
POLICIES = {"disabled": {"kind": "disabled.v1"},
            "active": {"kind": "active_global_l2.v1", "max_norm": 0.01}}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    return module(ROOT / "tools/byte_corpus_study.py").decode_json(path.read_bytes())


def write(path, value):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as stream:
        stream.write(encoded(value))


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sources():
    paths = {Path(__file__).resolve(), HERE / "PLAN.md", ROOT / "Cargo.toml", ROOT / "Cargo.lock",
             ROOT / "tools/qualify_byte_corpus_runs.py"}
    for pattern in ("tools/byte_corpus*.py", "tools/generate_*torch_fixture.py",
                    "tools/fisher_rao_reference.py",
                    "benchmarks/results/2026-10-16-language-capacity-budget/*.py",
                    "benchmarks/results/2026-10-16-attention-adamw-numerics/language_probe.py"):
        paths.update(ROOT.glob(pattern))
    for directory in ("crates/st-nn", "crates/st-backend-wgpu", "crates/st-kernel-contracts",
                      "bindings/st-py/src", "bindings/st-py/spiraltorch"):
        paths.update(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".rs", ".wgsl", ".toml", ".py", ".pyi"))
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(paths)}


def derived(original, policy):
    require(original["schema"] == "spiraltorch.byte_corpus.request.v9", "expected pinned v9 request")
    result = copy.deepcopy(original)
    result.update(schema="spiraltorch.byte_corpus.request.v10", gradient_clip=copy.deepcopy(policy))
    return result


def execute(output, label, command, timeout):
    started = time.monotonic()
    with os.fdopen(os.open(output / (label + ".log"), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                          0o600), "wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the qualification worker's descendants too, not just its CLI.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
    return dict(exit_code=code, timed_out=timed_out, timeout_seconds=timeout,
                elapsed_seconds=time.monotonic() - started,
                log_sha256=sha((output / (label + ".log")).read_bytes()))


def verify(root):
    manifest = read(root / "started.json")
    require(sources() == manifest["sources"], "source or plan drift")
    for name in POLICIES:
        require(sha((root / (name + ".json")).read_bytes()) == manifest["requests"][name], "request drift")
    return manifest


def summarize(root, name):
    protocol = module(ROOT / "tools/byte_corpus_study.py")
    diagnostics = module(ROOT / "benchmarks/results/2026-10-16-language-capacity-budget/diagnose.py")
    helper = module(ROOT / "benchmarks/results/2026-10-16-attention-adamw-numerics/language_probe.py")
    raw = (root / (name + ".json")).read_bytes()
    request = protocol.decode_json(raw)
    reference_path = root / (name + "-torch.json")
    report_path = root / name / "continuous/report-1024.json"
    reference, actual = read(reference_path), read(report_path)
    result = helper.summarize(protocol, diagnostics, raw, reference, actual)
    # Publish scalar diagnostics, not the private parameter-bearing exception.
    try:
        protocol.compare(raw, reference, actual)
    except ValueError as error:
        error_path = root / (name + "-strict-error.json")
        if error_path.exists():
            require(read(error_path) == {"error": str(error)}, "strict failure changed")
        else:
            write(error_path, {"error": str(error)})
    result.update(request_sha256=sha(raw), reference_sha256=sha(reference_path.read_bytes()),
                  report_sha256=sha(report_path.read_bytes()))
    curves, geometry = [], []
    for spec, expected, got in zip(request["cases"], reference["cases"], actual["cases"]):
        curves.append(dict(name=spec["name"], native=got["validation"], reference=expected["validation"]))
        for p, a, e in zip(spec["parameters"], got["final_parameters"], expected["final_parameters"]):
            if not p["name"].startswith("geometry."):
                continue
            initial = [protocol.f32(v) for v in p["values"]]
            norm = lambda values: math.sqrt(math.fsum(v * v for v in values))
            geometry.append(dict(case=spec["name"], parameter=p["name"],
                frozen=protocol.parameter_frozen(spec, p["name"]),
                native_delta_l2=norm(x - y for x, y in zip(a, initial)),
                reference_delta_l2=norm(x - y for x, y in zip(e, initial)),
                difference_l2=norm(x - y for x, y in zip(a, e))))
    result.update(heldout_curves=curves, geometry_deltas=geometry)
    if name == "disabled":
        result["baseline_case_payloads_equal"] = {
            key: encoded(read(root / ("prior-" + key + ".json"))["cases"]) == encoded(value["cases"])
            for key, value in (("native", actual), ("torch", reference))}
    return result


def run(args):
    import spiraltorch as st
    paths = {key: getattr(args, key) for key in PINS}
    require({key: sha(path.read_bytes()) for key, path in paths.items()} == PINS, "input pin drift")
    require(sha(Path(st._rs.__file__).read_bytes()) == NATIVE, "installed extension pin drift")
    original = read(args.request)
    require(len(original["train_batches"]) == 1024 and len(original["cases"]) == 3, "study size changed")
    root = args.output
    root.mkdir(mode=0o700, exist_ok=False)
    for key in ("request", "native", "torch"):
        with os.fdopen(os.open(root / ("prior-" + key + ".json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                              0o600), "wb") as stream:
            stream.write(paths[key].read_bytes())
    requests = {}
    for name, policy in POLICIES.items():
        value = derived(original, policy)
        write(root / (name + ".json"), value)
        raw = (root / (name + ".json")).read_bytes()
        study = st.nn.ByteCorpusStudy.from_json(raw)
        require(study.total_updates == 1024, "Rust update count differs")
        study.validate_segment(512)
        requests[name] = sha(raw)
    source_manifest = sources()
    write(root / "started.json", dict(schema="spiraltorch.byte_clipping.long_inputs.v1",
        pins=PINS, native_extension_sha256=NATIVE, requests=requests, sources=source_manifest,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        semantics="Only request schema and explicit clip policy differ from the pinned input"))
    rows = []
    for name in POLICIES:
        verify(root)
        require(sha(args.wheel.read_bytes()) == PINS["wheel"], "wheel drift")
        request = root / (name + ".json")
        print(json.dumps(dict(condition=name, phase="torch_reference")), flush=True)
        reference = execute(root, name + "-reference", [sys.executable, "-I", "-B",
            str(ROOT / "tools/byte_corpus_study.py"), "reference", str(request),
            str(root / (name + "-torch.json"))], 1800)
        print(json.dumps(dict(condition=name, phase="native_recovery", reference_exit=reference["exit_code"])), flush=True)
        recovery = execute(root, name + "-qualification", [sys.executable, "-I", "-B",
            str(ROOT / "tools/qualify_byte_corpus_runs.py"), "run", str(request), str(root / name),
            "--stop", "512", "--wheel", str(args.wheel), "--timeout-seconds", "1800"], 6000)
        verify(root)
        row = dict(name=name, reference_execution=reference, recovery_execution=recovery)
        if recovery["exit_code"] == 0:
            row["recovery"] = read(root / name / "result.json")
        if reference["exit_code"] == 0 and (root / name / "continuous/report-1024.json").is_file():
            row["comparison"] = summarize(root, name)
        write(root / (name + "-result.json"), row)
        rows.append(row)
        print(json.dumps(dict(condition=name, recovery_passed=recovery["exit_code"] == 0,
            strict_gate=row.get("comparison", {}).get("strict_gate", {}).get("passed", False))), flush=True)
    verify(root)
    write(root / "result.json", dict(schema="spiraltorch.byte_clipping.long_study.v1", rows=rows,
        scope="One seed; no speed or general quality claim; failures retained without gate relaxation"))


def screen(value):
    if isinstance(value, dict):
        forbidden = {"request_json", "model_json", "checkpoint_json", "report_json", "final_parameters",
                     "train_documents", "validation_documents", "prompts", "generated_bytes", "logits"}
        require(not forbidden.intersection(value), "private payload in export")
        for item in value.values():
            screen(item)
    elif isinstance(value, list):
        for item in value:
            screen(item)


def publish(args):
    root = args.private
    manifest = verify(root)
    for key in ("request", "native", "torch"):
        require(sha((root / ("prior-" + key + ".json")).read_bytes()) == PINS[key], "prior pin drift")
    result = read(root / "result.json")
    require([r["name"] for r in result["rows"]] == list(POLICIES), "condition coverage changed")
    for row in result["rows"]:
        name = row["name"]
        require(read(root / (name + "-result.json")) == row, "result changed")
        if "comparison" in row:
            require(summarize(root, name) == row["comparison"], "comparison changed")
        if "recovery" in row:
            require(read(root / name / "result.json") == row["recovery"], "recovery changed")
            for kind in ("report", "checkpoint"):
                a = (root / name / f"continuous/{kind}-1024.json").read_bytes()
                b = (root / name / f"resumed/{kind}-1024.json").read_bytes()
                require(a == b and sha(a) == row["recovery"][kind + "_sha256"], "recovery payload drift")
    screen(result)
    screen(manifest)
    artifacts = {str(p.relative_to(root)): dict(sha256=sha(p.read_bytes()), bytes=p.stat().st_size)
                 for p in sorted(root.rglob("*")) if p.is_file()}
    args.output.mkdir(exist_ok=False)
    write(args.output / "results.json", result)
    write(args.output / "inputs.json", manifest)
    write(args.output / "artifacts.json", artifacts)
    print("Exported both conditions, including failures; private inputs and weights retained locally.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    for key in (*PINS, "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p = sub.add_parser("publish")
    p.add_argument("private", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    (run if args.command == "run" else publish)(args)


if __name__ == "__main__":
    main()
