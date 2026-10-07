#!/usr/bin/env python3
"""Replay one saved mixed-geometry update through a newly pinned native package.

Reuses the frozen example client and an existing midpoint/expected state. Never
retrains or rescales the original study. This is an auxiliary migration check,
not a new quality observation or a throughput benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import transformers
import spiraltorch as st
import spiraltorch.geometry_autograd as bridge
import spiraltorch.spiraltorch as native


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_update(proof, actual, expected, report):
    cursor = report["config"]["checkpoint_after"]
    require(actual["start_cursor"] == cursor and len(actual["records"]) == 1, "replay budget differs")
    require(proof.equal(actual["records"], report["public_records"][cursor:]), "update records differ")
    require(proof.equal(actual["gradients"], expected["public"]["gradients"][cursor:]), "raw gradients differ")
    require(proof.equal(actual["final"], expected["public"]["final"]), "adapter/Adam/RNG differ")
    require(actual["base_sha256"] == report["base_sha256"], "base identity differs")


def check_transport(calls, report, topos_capture):
    shape, features = report["shape"], [report["config"]["features"]]
    # Forward WaveGate (x/gate/bias), optional Topos (x/gate), then reverse VJPs.
    expected = [shape, features, features] + ([shape] * 3 if topos_capture else []) + [shape]
    require(calls == expected, "geometry bulk transport call sequence differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "client-root", "model-dir", "corpus", "package-root", "runtime-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--topos-capture", action="store_true",
                        help="Require the new Topos captured bulk path as well as WaveGate")
    args = parser.parse_args()
    require(not args.output.exists(), "output exists")
    report = json.loads((args.previous / "report.json").read_bytes())
    require(report["schema"] == "spiraltorch.geometry_adapter_stack_example.v1"
            and report["status"] == "bitwise_exact", "unverified source run")
    config = report["config"]
    require(config["steps"] == 3 and config["checkpoint_after"] == 2, "expected one remaining update")
    for key, file in (("state", "states.pt"), ("midpoint", "midpoint.pt")):
        require(digest(args.previous / file) == report[key]["sha256"], "source artifact differs")
    for file, expected in {"hf_geometry_adapter_stack.py": report["example_sha256"],
                           **{n: h for n, h in report["source_sha256"].items()
                              if n != "geometry_adapters.py"}}.items():
        require(digest(args.client_root / file) == expected, "frozen client differs")
    sys.path.insert(0, str(args.client_root.resolve()))
    import hf_geometry_adapter_stack as client
    proof = client.proof
    require(Path(st.__file__).resolve().parent == (args.package_root / "spiraltorch").resolve(), "wrong package")
    manifest = json.loads(args.runtime_manifest.read_bytes())
    proof.verify_runtime(args.package_root, manifest)
    require(args.model_dir.name == config["model_snapshot"] and digest(args.corpus) == config["corpus_sha256"],
            "model/corpus identity differs")
    require(str(torch.__version__) == report["torch"] and str(transformers.__version__) == report["transformers"],
            "Torch/Transformers version differs")
    require(bridge._buffer_transport_available(), "bulk transport unavailable")
    if args.topos_capture:
        require(hasattr(st.ToposResonatorKernel, "capture_buffer"), "Topos capture unavailable")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    text = args.corpus.read_text()
    require(text.count(config["corpus_end_marker"]) == 1, "corpus marker differs")
    text = text[:text.index(config["corpus_end_marker"])]
    cut = text.rfind("\n\n", 0, int(len(text) * .9))
    require(cut == report["training_split_character_offset"], "split differs")
    ids = tokenizer.encode(text[:cut], add_special_tokens=False)
    size = config["block_size"]
    tokens = torch.tensor(ids[:len(ids) // size * size], dtype=torch.long).reshape(-1, size)
    require(proof.tensor_receipt(tokens) == report["training_tokens"], "training token identity differs")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    saved = torch.load(args.previous / "midpoint.pt", weights_only=True, map_location="cpu")
    expected = torch.load(args.previous / "states.pt", weights_only=True, map_location="cpu")
    require(proof.model_digest(model) == report["base_sha256"], "base weights differ before replay")
    calls, original = [], bridge._buffer_values

    def observe(value):
        calls.append(list(value.shape))
        return original(value)

    args.output.mkdir(parents=True, exist_ok=False)
    try:
        bridge._buffer_values = observe
        try:
            actual = client.run(model, tokens, config, True, saved=saved)
        finally:
            bridge._buffer_values = original
        compare_update(proof, actual, expected, report)
        check_transport(calls, report, args.topos_capture)
        proof.verify_runtime(args.package_root, manifest)
        state = proof.save_state(args.output / "state.pt", actual)
        result = {"schema": "spiraltorch.geometry_stack_native_replay.v2" if args.topos_capture
                  else "spiraltorch.geometry_stack_native_replay.v1", "status": "bitwise_exact",
                  "source_report_sha256": digest(args.previous / "report.json"),
                  "source_midpoint_sha256": digest(args.previous / "midpoint.pt"),
                  "source_states_sha256": digest(args.previous / "states.pt"),
                  "runtime_manifest_sha256": digest(args.runtime_manifest), "native_sha256": digest(Path(native.__file__)),
                  "replay_client_sha256": digest(Path(__file__)), "state": state,
                  "records": actual["records"], "parameter_count": actual["parameter_count"],
                  "base_sha256": actual["base_sha256"], "base_unchanged": True,
                  "bulk_transport_shapes": calls, "auxiliary_updates": 1,
                  "topos_capture_required": args.topos_capture,
                  "raw_gradients_adapter_adam_rng_exact": True, "heldout_scoring": False, "timing_evidence": False}
        proof.write_json(args.output / "report.json", result)
        print(json.dumps({"status": result["status"], "auxiliary_updates": 1, "parameter_count": actual["parameter_count"]}), flush=True)
    except Exception as error:
        proof.write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
