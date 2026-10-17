"""Recheck private receipts and export metrics/hashes, including every failure."""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import byte_corpus_study as protocol

LABELS = ("sgd-off", "sgd-active", "adamw-off", "adamw-active")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return protocol.decode_json(path.read_bytes())


def identity(path):
    raw = path.read_bytes()
    return dict(sha256=protocol.digest(raw), bytes=len(raw))


def geometry_deltas(request, reference, actual):
    rows = []
    require(len(request["cases"]) == len(reference["cases"]) == len(actual["cases"]), "case count differs")
    for spec, expected, observed in zip(request["cases"], reference["cases"], actual["cases"]):
        require(spec["name"] == expected["name"] == observed["name"], "case order differs")
        require(len(spec["parameters"]) == len(expected["final_parameters"]) == len(observed["final_parameters"]),
                "parameter count differs")
        for parameter, want, got in zip(spec["parameters"], expected["final_parameters"], observed["final_parameters"]):
            if not parameter["name"].startswith("geometry."):
                continue
            initial = [protocol.f32(v) for v in parameter["values"]]
            require(len(initial) == len(want) == len(got), "parameter shape differs")
            norm = lambda values: math.sqrt(math.fsum(x * x for x in values))
            rows.append(dict(case=spec["name"], parameter=parameter["name"],
                frozen=protocol.parameter_frozen(spec, parameter["name"]),
                reference_delta_l2=norm(a - b for a, b in zip(want, initial)),
                actual_delta_l2=norm(a - b for a, b in zip(got, initial)),
                difference_l2=norm(a - b for a, b in zip(want, got))))
    return rows


def screen(value):
    if isinstance(value, dict):
        forbidden = {"request_json", "model_json", "checkpoint_json", "report_json", "parameters", "final_parameters",
                     "train_documents", "validation_documents", "prompts", "generated_bytes", "logits"}
        require(not forbidden.intersection(value), "private payload in public record")
        for child in value.values():
            screen(child)
    elif isinstance(value, list):
        for child in value:
            screen(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    private = args.private
    browser = private / "browser"
    manifest = load(browser / "inputs.json")
    require(identity(browser / "input.json") == manifest["fixture"], "fixture changed")
    for path, expected in manifest["source"].items():
        require(identity(browser / "site" / path) == expected, "frozen browser asset changed")
    for path, expected in manifest["original_source"].items():
        require(identity(ROOT / path) == expected, "browser source changed")
    for path, expected in manifest["package"].items():
        require(identity(private / "webgpu-package" / path) == expected, "package changed")
    inputs = load(browser / "input.json")["requests"]
    training = load(browser / "training.json")["runs"]
    browser_main, resume = load(browser / "main.json"), load(browser / "resume.json")
    for field in ("legacy_disabled_exact", "clipping_changes_learning", "foreign_policy_rejected"):
        require(browser_main[field] is True, field)
    expected_names = [name + suffix for name in ("sgd", "adamw") for suffix in ("-v9", "-off", "-active")]
    require([r["name"] for r in inputs] == [r["name"] for r in training] ==
            [r["name"] for r in browser_main["rows"]] == expected_names, "browser case coverage differs")
    for item, raw, row in zip(inputs, training, browser_main["rows"]):
        require(protocol.digest(item["request_json"].encode()) == row["request_sha256"], "browser request hash differs")
        for kind in ("report", "checkpoint"):
            require(protocol.digest(raw[kind + "_json"].encode()) == row[kind + "_sha256"], "browser payload hash differs")
    require([r["name"] for r in resume["rows"]] == ["sgd-active", "adamw-active"], "resume coverage differs")
    for row in resume["rows"]:
        full, = [r for r in browser_main["rows"] if r["name"] == row["name"]]
        require(row["exact_report"] is True and row["exact_checkpoint"] is True, "resume not exact")
        require(all(row[k] == full[k] for k in ("report_sha256", "checkpoint_sha256")), "resume hashes differ")
    public = {"browser.json": browser_main, "browser-resume.json": resume}
    recoveries = []
    for mode, directory in (("native", private / "native"), ("browser", private / "browser-comparison")):
        matrix = load(directory / "matrix.json")
        require([r["name"] for r in matrix["rows"]] == list(LABELS), "comparison coverage differs")
        rows = []
        for recorded in matrix["rows"]:
            name = recorded["name"]
            request_raw = (browser / (name + ".json")).read_bytes()
            request = protocol.decode_json(request_raw)
            reference_path = directory / (name + "-torch.json")
            reference = load(reference_path)
            if mode == "native":
                run = directory / name
                actual_raw = (run / "continuous/report-2.json").read_bytes()
                recovery = load(run / "result.json")
                require(recorded["recovery"] == recovery, "recovery receipt differs")
                for field in ("exact_report_equal", "exact_checkpoint_equal", "source_unchanged", "explicit_parent_verified"):
                    require(recovery[field] is True, field)
                for kind in ("report", "checkpoint"):
                    a = (run / f"continuous/{kind}-2.json").read_bytes()
                    b = (run / f"resumed/{kind}-2.json").read_bytes()
                    require(a == b and protocol.digest(a) == recovery[kind + "_sha256"], "native resume bytes differ")
                recoveries.append(dict(name=name, **recovery))
            else:
                run, = [r for r in training if r["name"] == name]
                actual_raw = run["report_json"].encode()
            actual = protocol.decode_json(actual_raw)
            row = dict(name=name, request_sha256=protocol.digest(request_raw),
                       report_sha256=protocol.digest(actual_raw), reference=identity(reference_path))
            row["cases"] = [dict(name=c["name"], updates=len(c["training"]),
                initial_bpb=c["validation"][0]["bits_per_byte"],
                final_bpb=c["validation"][-1]["bits_per_byte"]) for c in actual["cases"]]
            try:
                comparison = protocol.compare(request_raw, reference, actual)
                require(recorded["passed"] is True, "recorded failure unexpectedly passed")
                require(comparison == load(directory / (name + "-comparison.json")), "comparison changed")
                row.update(passed=True, comparison=comparison)
            except ValueError as error:
                require(recorded["passed"] is False and str(error) == recorded["error"], "failure disposition changed")
                row.update(passed=False, error=str(error))
            row["geometry_deltas"] = geometry_deltas(request, reference, actual)
            rows.append(row)
        public[mode + "-comparison.json"] = dict(schema=matrix["schema"], mode=mode,
            passed=all(r["passed"] for r in rows), rows=rows, scope=matrix["scope"])
    public["python-recovery.json"] = dict(rows=recoveries)
    for name in ("node-nn.json", "node-webgpu.json"):
        public[name] = load(private / name)
        require(public[name]["feature_matrix_passed"] is True, "Node export matrix failed")
    sources = [ROOT / "Cargo.lock", ROOT / "tools/byte_corpus_gradient_clip.py",
        ROOT / "tools/test_byte_corpus_gradient_clip.py", ROOT / "tools/test_byte_corpus_study.py",
        ROOT / "tools/test_byte_corpus_optimizer.py", ROOT / "bindings/st-py/tests/test_nn_byte_study.py",
        ROOT / "tools/byte_corpus_optimizer.py", ROOT / "tools/byte_corpus_study.py",
        ROOT / "tools/verify_byte_corpus_resume.py", ROOT / "tools/qualify_byte_corpus_runs.py",
        ROOT / "crates/st-nn/src/resident/byte_study.rs",
        ROOT / "bindings/st-py/src/nn_resident/byte_study.rs", ROOT / "bindings/st-wasm/src/byte_lm.rs"]
    sources += list((ROOT / "crates/st-nn/src/resident/byte_study").rglob("*.rs"))
    sources += list(Path(__file__).parent.glob("*.py"))
    public["inputs.json"] = dict(schema="spiraltorch.byte_clipping.inputs.v1",
        source={str(p.relative_to(ROOT)): identity(p) for p in sorted(set(sources))},
        browser_manifest=manifest, requests={r["name"]: protocol.digest(r["request_json"].encode()) for r in inputs},
        wheels={p.name: identity(p) for p in (private / "wheels").glob("*.whl")},
        logs={p.name: identity(p) for p in private.glob("*.log")},
        scope="Hashes bind retained evidence; not independent attestation. Raw inputs and weights remain private.")
    for value in public.values():
        screen(value)
    args.output.mkdir(exist_ok=False)
    for name, value in public.items():
        protocol.write_new(args.output / name, value)
    print("Exported all four native/browser conditions, including failed numerical gates.")


if __name__ == "__main__":
    main()
