"""Offline, pinned one-chapter corpus training qualification, not language quality."""
import argparse
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import platform
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "bindings/st-py/examples/native_geometry_corpus_training.py"
HELPER = ROOT / "benchmarks/results/2026-10-18-native-gpt2-learning/probe.py"
CHAPTER = "a6616e876864219ac136ce75fcb0899cbc957ed0df4c583717fb06f3b3d5fb5c"
MODEL = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
}
ARTIFACT = "sha256:fb295546689a3c3c83578d5b959da6a9057ba427257f34ac7eaf1b7b78713b94"
BASE = "sha256:0c552e7a3d7fae610edd7a22831f91e0fa718b212d38d15a07ff875e3979fb3c"
RECIPE = {"context_length": 32, "microbatch_windows": 2, "accumulation_steps": 2,
          "epochs": 2, "learning_rate": .001, "weight_decay": .01,
          "label_smoothing": .1, "max_gradient_norm": .1}
GATES = {"loss_atol": 2e-4, "parameter_atol": 3e-5, "parameter_rtol": 1e-3}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(path):
    return json.loads(path.read_bytes())


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__).resolve(), EXAMPLE, HELPER)}


def check_model(path):
    if {name: digest(path / name) for name in MODEL} != MODEL:
        raise ValueError("model/tokenizer snapshot mismatch")


def forbid_ml():
    forbidden = {"torch", "numpy", "transformers", "accelerate", "safetensors"}
    if any(name.split(".")[0] in forbidden for name in sys.modules):
        raise ValueError("native process already imported optional ML")

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise AssertionError("foreign ML import forbidden: " + fullname)
    sys.meta_path.insert(0, Guard())


def prepare(args):
    forbid_ml()
    import tokenizers
    check_model(args.model_dir)
    if digest(args.chapter) != CHAPTER or "sha256:" + digest(args.artifact) != ARTIFACT:
        raise ValueError("only the previously admitted train chapter and pinned artifact are allowed")
    tokenizer = tokenizers.Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    ids = tokenizer.encode(args.chapter.read_text(encoding="utf-8"), add_special_tokens=False).ids
    windows = len(ids) // RECIPE["context_length"]
    updates_per_epoch = math.ceil(windows / (RECIPE["microbatch_windows"] * RECIPE["accumulation_steps"]))
    if windows < 5:
        raise ValueError("fixture is too small for this continuation check")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(args.output / "data.json", {"token_documents": [ids]})
    save(args.output / "recipe.json", RECIPE)
    with (args.output / "initial.json").open("xb") as stream:
        stream.write(args.artifact.read_bytes())
    save(args.output / "contract.json", {
        "schema": "spiraltorch.native_corpus_qualification.v1", "source_sha256": source_hashes(),
        "model_sha256": MODEL, "chapter_sha256": CHAPTER, "tokenizers": tokenizers.__version__,
        "encoding": "one existing train chapter; no added special tokens/EOS/padding/truncation",
        "base_model_identity": BASE, "initial_geometry_artifact_id": ARTIFACT,
        "input_sha256": {name: digest(args.output / name) for name in ("data.json", "recipe.json", "initial.json")},
        "recipe": RECIPE, "gates": GATES, "reference_ce_dtype": args.reference_ce_dtype or "f32",
        "documents": 1, "tokens": len(ids), "windows_per_epoch": windows,
        "dropped_tokens": len(ids) % RECIPE["context_length"],
        "target_tokens_per_epoch": windows * (RECIPE["context_length"] - 1),
        "updates_per_epoch": updates_per_epoch, "total_updates": updates_per_epoch * RECIPE["epochs"],
        "resume_from_update": 2, "scope": "execution/update/resume qualification, not speed or language quality",
    })
    print(json.dumps(load(args.output / "contract.json")), flush=True)


def cohort(args):
    contract = load(args.cohort / "contract.json")
    if contract["source_sha256"] != source_hashes() or contract["recipe"] != RECIPE or contract["gates"] != GATES:
        raise ValueError("source, recipe or gates changed after predeclaration")
    if contract["reference_ce_dtype"] not in ("f32", "f64"):
        raise ValueError("unsupported reference loss precision")
    if any(digest(args.cohort / n) != h for n, h in contract["input_sha256"].items()):
        raise ValueError("sealed cohort input changed")
    check_model(args.model_dir)
    return contract


def native(args):
    forbid_ml()
    cohort(args)
    argv = ["--model-dir", str(args.model_dir), "--data-json", str(args.cohort / "data.json"),
            "--output", str(args.output), "--save-every", "2"]
    if args.resume_run:
        path = args.resume_run / "checkpoint-000002.json"
        argv += ["--resume", str(path), "--checkpoint-id", "sha256:" + digest(path)]
    else:
        argv += ["--recipe-json", str(args.cohort / "recipe.json"), "--artifact", str(args.cohort / "initial.json"),
                 "--artifact-id", ARTIFACT]
    runpy.run_path(str(EXAMPLE))["main"](argv)
    import spiraltorch as st
    save(args.output / "runtime.json", {"python": platform.python_version(), "platform": platform.platform(),
        "native_extension_sha256": digest(Path(st._rs.__file__)), "optional_ml_imports_forbidden": True,
        "qualification_contract_sha256": digest(args.cohort / "contract.json")})


def reference(args):
    contract = cohort(args)
    import torch
    import transformers
    import spiraltorch as st
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    model.config.use_cache = False
    versions = [p._version for p in model.parameters()]
    artifact = st.GeometryAdapterArtifact.from_json((args.cohort / "initial.json").read_text(), ARTIFACT, BASE)
    stack = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=BASE)
    attach = runpy.run_path(str(HELPER))["attach_reference"]
    optimizer = torch.optim.AdamW(stack.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"],
                                 betas=(.9, .999), eps=1e-8, foreach=False)
    docs = load(args.cohort / "data.json")["token_documents"]
    # Independent spans, not the native trainer's emitted plan.
    context = RECIPE["context_length"]
    spans = [(d, s) for d, doc in enumerate(docs) for s in range(0, len(doc) - context + 1, context)]
    if len(spans) != contract["windows_per_epoch"]:
        raise ValueError("independent reference window count differs")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []
    group = RECIPE["microbatch_windows"] * RECIPE["accumulation_steps"]
    for epoch in range(RECIPE["epochs"]):
        for start in range(0, len(spans), group):
            end = min(start + group, len(spans))
            count = (end - start) * (context - 1)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for offset in range(start, end, RECIPE["microbatch_windows"]):
                selected = spans[offset:min(offset + RECIPE["microbatch_windows"], end)]
                ids = torch.tensor([docs[d][s:s+context] for d, s in selected])
                with attach(stack, model):
                    logits = model(ids[:, :-1], use_cache=False).logits
                    if contract["reference_ce_dtype"] == "f64":
                        logits = logits.double()
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                        reduction="sum", label_smoothing=RECIPE["label_smoothing"]) / count
                    loss.backward()
                loss_sum += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(stack.parameters(), RECIPE["max_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            row = {"update": len(rows) + 1, "epoch": epoch, "window_start": start, "window_end": end,
                   "target_tokens": count, "loss": loss_sum, "gradient_norm_before_clip": float(norm)}
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    if versions != [p._version for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise ValueError("reference frozen base mutated or received gradients")
    final = stack.export_artifact(base_model_identity=BASE)
    with (args.output / "final-geometry.json").open("x") as stream:
        stream.write(final.to_json())
    save(args.output / "result.json", {"rows": rows, "base_parameter_versions_unchanged": True,
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "intraop_threads": torch.get_num_threads(),
        "cross_entropy_dtype": contract["reference_ce_dtype"],
        "native_extension_sha256": digest(Path(st._rs.__file__)),
        "qualification_contract_sha256": digest(args.cohort / "contract.json")})


def parameters(record):
    return {(p["path"], t["name"]): (t["shape"], t["values"]) for p in record["placements"]
            for t in p["tensors"] if t["role"] == "parameter"}


def compare(args):
    contract = load(args.cohort / "contract.json")
    if (contract["source_sha256"] != source_hashes() or contract["recipe"] != RECIPE
            or contract["gates"] != GATES or contract["model_sha256"] != MODEL
            or contract["chapter_sha256"] != CHAPTER or contract["base_model_identity"] != BASE
            or contract["initial_geometry_artifact_id"] != ARTIFACT):
        raise ValueError("qualification contract changed")
    if any(digest(args.cohort / n) != h for n, h in contract["input_sha256"].items()):
        raise ValueError("sealed cohort input changed")
    full, resumed, ref = args.cohort / "native-full", args.cohort / "native-resumed", args.cohort / "reference"
    summary, restored, reference_result = (load(p / "result.json") for p in (full, resumed, ref))
    runtime, resume_runtime = (load(p / "runtime.json") for p in (full, resumed))
    if runtime != resume_runtime or runtime["optional_ml_imports_forbidden"] is not True:
        raise ValueError("native and resume runtime contracts differ")
    for record in (runtime, reference_result):
        if record["qualification_contract_sha256"] != digest(args.cohort / "contract.json"):
            raise ValueError("runs used different qualification contracts")
    if reference_result["base_parameter_versions_unchanged"] is not True:
        raise ValueError("reference base was not frozen")
    if (contract["reference_ce_dtype"] not in ("f32", "f64")
            or reference_result["cross_entropy_dtype"] != contract["reference_ce_dtype"]
            or reference_result["native_extension_sha256"] != runtime["native_extension_sha256"]):
        raise ValueError("reference loss precision or native geometry binary differs")
    native_rows, resumed_rows = ([json.loads(s) for s in (p / "steps.jsonl").read_text().splitlines()] for p in (full, resumed))
    expected = contract["total_updates"]
    if len(native_rows) != expected or len(reference_result["rows"]) != expected:
        raise ValueError("incomplete cohort")
    if summary["total_updates"] != expected or summary["completed_updates"] != expected or not summary["exhausted"]:
        raise ValueError("native run did not finish its sealed plan")
    if (not restored["exhausted"] or restored["completed_updates"] != expected or restored["data_id"] != summary["data_id"]
            or restored["initial"]["updates"] != contract["resume_from_update"] or summary["initial"]["updates"] != 0
            or restored["config"] != summary["config"] or restored["plan"] != summary["plan"]
            or summary["base_model_identity"] != BASE or restored["base_model_identity"] != BASE):
        raise ValueError("resume scope differs")
    checkpoint = f"checkpoint-{expected:06d}.json"
    exact = resumed_rows == native_rows[contract["resume_from_update"]:] and digest(full / checkpoint) == digest(resumed / checkpoint)
    native_final = json.loads(load(full / checkpoint)["geometry_json"])
    a, b = parameters(native_final), parameters(load(ref / "final-geometry.json"))
    if a.keys() != b.keys() or any(a[key][0] != b[key][0] or len(a[key][1]) != len(b[key][1]) for key in a):
        raise ValueError("reference parameter coverage differs")
    errors, parameter_ok = [], True
    for key in a:
        for x, y in zip(a[key][1], b[key][1]):
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("nonfinite parameters")
            errors.append(abs(x-y))
            parameter_ok &= abs(x-y) <= GATES["parameter_atol"] + GATES["parameter_rtol"] * abs(y)
    losses = []
    for row, other in zip(native_rows, reference_result["rows"]):
        if any(row[k] != other[k] for k in ("update", "epoch", "window_start", "window_end", "target_tokens")):
            raise ValueError("reference chronology or denominator differs")
        if not math.isfinite(row["loss"]) or not math.isfinite(other["loss"]):
            raise ValueError("nonfinite losses")
        losses.append(abs(row["loss"] - other["loss"]))
    files = [args.cohort / "contract.json", args.cohort / "data.json", args.cohort / "recipe.json", args.cohort / "initial.json"]
    files += sorted(p for parent in (full, resumed, ref) for p in parent.iterdir() if p.is_file())
    report = {"schema": "spiraltorch.native_corpus_comparison.v1", "contract": contract,
        "comparison_kind": "posthoc_f64_ce_control" if contract["reference_ce_dtype"] == "f64" else "f32_comparison",
        "native_runtime": runtime, "resume_runtime": resume_runtime,
        "reference_runtime": {k: v for k, v in reference_result.items() if k != "rows"},
        "native_steps": native_rows, "reference_steps": reference_result["rows"],
        "native_data_id": summary["data_id"], "native_plan": summary["plan"],
        "fresh_process_remaining_steps_and_final_checkpoint_exact": exact,
        "maximum_loss_absolute_error": max(losses), "maximum_final_parameter_absolute_error": max(errors),
        "all_final_parameters_within_gate": bool(parameter_ok),
        "passed": exact and max(losses) <= GATES["loss_atol"] and bool(parameter_ok),
        "private_file_sha256": {str(p.relative_to(args.cohort)): digest(p) for p in files}}
    save(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k in {"passed", "maximum_loss_absolute_error", "maximum_final_parameter_absolute_error", "fresh_process_remaining_steps_and_final_checkpoint_exact"}}))
    if not report["passed"]:
        raise ValueError("predeclared comparison failed; report retained, do not relax gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "native", "reference", "compare"])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--chapter", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--reference-ce-dtype", choices=["f32", "f64"],
                        help="prepare only: seal reference CE precision; f64 is a separate post-failure control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "prepare" and args.reference_ce_dtype is not None:
        parser.error("reference loss precision is sealed only at prepare, not overridden per run")
    required = {"prepare": ("model_dir", "chapter", "artifact"), "native": ("model_dir", "cohort"),
                "reference": ("model_dir", "cohort"), "compare": ("cohort",)}[args.mode]
    if any(getattr(args, name) is None for name in required):
        parser.error("missing required inputs for " + args.mode)
    {"prepare": prepare, "native": native, "reference": reference, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
