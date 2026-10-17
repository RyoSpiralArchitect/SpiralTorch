"""Qualify installed segment recovery with real Rust/WGPU and a killed worker.

All inputs, weights and logs stay in a new private directory. Only result.json
is designed for publication. This tests process interruption, not power loss.
Run with an isolated interpreter from the source-candidate wheel environment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import zipfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def case_metadata(case):
    # Preserve legacy omission rather than reconstructing Rust's policy default.
    return {key: case[key] for key in ("seed", "geometry", "geometry_update") if key in case}


def write(path, raw):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as handle:
        handle.write(raw)


def timeout_seconds(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("timeout must be an integer in [1, 7200]") from error
    if str(result) != str(value) or not 1 <= result <= 7200:
        raise argparse.ArgumentTypeError("timeout must be an integer in [1, 7200]")
    return result


def run_child(root, label, arguments, expected=0, *, timeout=600):
    timeout = timeout_seconds(timeout)
    with os.fdopen(os.open(root / f"{label}.log", os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                          0o600), "wb") as log:
        result = subprocess.run([sys.executable, "-I", "-B", *map(str, arguments)],
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == expected, f"{label} exited {result.returncode}; inspect its private log")
    return result.returncode


def interrupt(request, output, stop):
    import spiraltorch as st
    from spiraltorch import byte_corpus_runs as store
    total = st.nn.ByteCorpusStudy.from_json(request.read_bytes()).total_updates
    original = store._write_new

    def kill_after_seal(directory, name, raw):
        result = original(directory, name, raw)
        if name == f"segment-{stop}.json":
            # The Rust operation and durable seal finished, but the Python loop
            # has not yet appended its in-memory receipt or terminal summary.
            os.kill(os.getpid(), signal.SIGKILL)
        return result

    store._write_new = kill_after_seal
    st.run_byte_corpus_segments(request, output, [stop, total])
    raise RuntimeError("interruption injection did not run")


def qualify(args):
    import spiraltorch as st
    from spiraltorch import byte_corpus_runs as store
    require(__debug__, "qualification must not run with Python -O")
    require(os.name == "posix", "qualification requires POSIX")
    raw = args.request.read_bytes()
    study = st.nn.ByteCorpusStudy.from_json(raw)
    require(0 < args.stop < study.total_updates, "stop must be strictly inside the study")
    study.validate_segment(args.stop)
    request = json.loads(raw)  # Metadata/windows only; transport uses original bytes.
    module_raw = Path(store.__file__).read_bytes()
    repo = Path(__file__).resolve().parents[1]
    require(module_raw == (repo / "bindings/st-py/spiraltorch/byte_corpus_runs.py").read_bytes(),
            "installed run primitive differs from this checkout")
    native = Path(st._rs.__file__)
    with zipfile.ZipFile(args.wheel) as wheel:
        require(wheel.read("spiraltorch/byte_corpus_runs.py") == module_raw,
                "installed module differs from wheel")
        require(wheel.read("spiraltorch/" + native.name) == native.read_bytes(),
                "installed native extension differs from wheel")
    for name in store.__all__:
        require(name in st.__all__ and getattr(st, name) is getattr(store, name),
                "missing public package export: " + name)
    root = args.output
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    pinned = root / "request.json"
    write(pinned, raw)
    cli = Path(sys.executable).parent / "spiral-byte-corpus-run"
    require(cli.is_file(), "installed console entry point is absent")
    timeout = timeout_seconds(args.timeout_seconds)
    run_child(root, "continuous", [cli, pinned, root / "continuous", "--stop", study.total_updates],
              timeout=timeout)
    interrupted_exit = run_child(root, "interrupted", [Path(__file__).resolve(), "interrupt",
                                 pinned, root / "interrupted", "--stop", args.stop], -signal.SIGKILL,
                                 timeout=timeout)
    source = root / "interrupted"
    before = {p.name: digest(p.read_bytes()) for p in source.iterdir()}
    require("failed.json" not in before and "completed.json" not in before,
            "killed worker unexpectedly wrote a terminal summary")
    require(f"segment-{study.total_updates}.json" not in before, "killed worker reached final segment")
    saved = st.load_byte_corpus_segment(source, args.stop)
    run_child(root, "resumed", [cli, pinned, root / "resumed", "--stop", study.total_updates,
                               "--resume-from", source, "--resume-after", args.stop,
                               "--expected-resume-sha256", saved.receipt_sha256], timeout=timeout)
    require(before == {p.name: digest(p.read_bytes()) for p in source.iterdir()}, "source run was mutated")
    parent = json.loads((root / "resumed/claim.json").read_bytes())["parent"]
    require(parent == dict(claim_sha256=saved.claim_sha256, segment_sha256=saved.receipt_sha256,
                           checkpoint_sha256=saved.checkpoint_sha256, completed_updates=args.stop),
            "resume ancestry does not match selected seal")
    payloads = {}
    for kind in ("report", "checkpoint"):
        name = f"{kind}-{study.total_updates}.json"
        control = (root / "continuous" / name).read_bytes()
        resumed = (root / "resumed" / name).read_bytes()
        require(control == resumed, f"continuous/resumed {kind} bytes differ")
        payloads[kind + "_sha256"] = digest(resumed)
    finished = st.load_byte_corpus_segment(root / "resumed", study.total_updates)
    control = st.load_byte_corpus_segment(root / "continuous", study.total_updates)
    steps, batch_size = request["config"]["steps"], request["config"]["batch"]
    documents = [bytes(row) for row in request["validation_documents"]]
    windows = [tuple(row) for row in request["validation_batches"][0]]
    batch = st.nn.ByteLmBatch.from_documents(documents, windows, steps)
    prompts = [documents[doc][start:start + steps] for doc, start in windows]
    rows = []
    for case in request["cases"]:
        outputs = []
        for segment in (control, finished):
            artifact = segment.study.case_artifact(segment.checkpoint, case["name"],
                                                   policy="fixed_window_frozen.v1")
            owner = artifact.restore_wgpu()
            adapter = owner.tensor_device().adapter_info()
            require(adapter["device_type"] != "Cpu", "software adapter is not qualified")
            forward = owner.forward(owner.prepare_batch(batch))
            logits = forward.prediction_tensor().snapshot().read_values()
            require(len(logits) == batch_size * steps * 256 and all(map(math.isfinite, logits)),
                    "inference returned wrong-size or nonfinite logits")
            generated = owner.generate_greedy(prompts, max_new_bytes=4)
            value = json.loads(generated)
            require(len(value["generated_bytes"]) == batch_size
                    and all(len(row) == 4 for row in value["generated_bytes"]),
                    "generation returned wrong number of bytes")
            require(value["artifact_id"] == artifact.artifact_id
                    and value["revision"] == str(study.total_updates), "generation identity mismatch")
            for row in range(batch_size):
                last = logits[((row + 1) * steps - 1) * 256:(row + 1) * steps * 256]
                require(value["generated_bytes"][row][0] == max(range(256), key=last.__getitem__),
                        "generation does not start at the measured next-byte argmax")
            outputs.append(dict(artifact_id=artifact.artifact_id,
                                logits_sha256=digest(struct.pack(f"<{len(logits)}f", *logits)),
                                generation_sha256=digest(generated.encode())))
            del owner
        require(outputs[0] == outputs[1], "restored inference or generation differs after resume")
        rows.append(dict(**case_metadata(case), **outputs[0]))
    require(len(rows) == len(request["cases"]) > 0, "empty case evidence")
    result = dict(schema="spiraltorch.byte_corpus.segment_recovery_qualification.v1",
                  request_sha256=digest(raw), total_updates=study.total_updates,
                  interrupted_after=args.stop, interrupted_exit=interrupted_exit,
                  optimizer=json.loads(study.optimizer_recipe_json), cases=rows,
                  exact_report_equal=True, exact_checkpoint_equal=True, source_unchanged=True,
                  explicit_parent_verified=True, no_terminal_record_at_death=True,
                  entrypoint="installed_console_script", python=sys.version.split()[0],
                  worker_timeout_seconds=timeout,
                  wheel_sha256=digest(args.wheel.read_bytes()),
                  native_extension_sha256=digest(native.read_bytes()),
                  run_primitive_sha256=digest(module_raw), qualifier_sha256=digest(Path(__file__).read_bytes()),
                  adapter={key: adapter[key] for key in ("name", "backend", "device_type")},
                  process_interruption_only=True, power_loss_tested=False,
                  geometric_quality_benefit_established=False, **payloads)
    if request["schema"] == "spiraltorch.byte_corpus.request.v10":
        result["gradient_clip"] = json.loads(study.gradient_clip_recipe_json)
    write(root / "result.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(dict(status="passed", cases=len(rows), interrupted_after=args.stop,
                          total_updates=study.total_updates, **payloads)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("run", "interrupt"):
        command = sub.add_parser(name)
        command.add_argument("request", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--stop", type=int, required=True)
        if name == "run":
            command.add_argument("--wheel", type=Path, required=True)
            command.add_argument("--timeout-seconds", type=timeout_seconds, default=600,
                                 help="Per-worker limit, 1..7200 seconds (default: 600)")
    args = parser.parse_args()
    if args.mode == "interrupt":
        interrupt(args.request, args.output, args.stop)
    else:
        qualify(args)


if __name__ == "__main__":
    main()
