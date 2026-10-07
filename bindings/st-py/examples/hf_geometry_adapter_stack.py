#!/usr/bin/env python3
"""Offline mixed-geometry learning through the public placement API.

Runs a small explicit-insertion reference, the public hook API, and saved-state
continuation. This is a connection check, not a quality or speed comparison.
See docs/geometry_adapter_stack.md for ordinary application usage.
"""

import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path

import torch
import transformers
import spiraltorch as st
import spiraltorch.geometry_adapters as placement
import spiraltorch.spiraltorch as native

# Repository-only evidence helpers; the public API itself needs neither tools
# nor Transformers. Reuse receipt/state verification, not geometric arithmetic.
import probe_fractional_multi_adapter as proof

FAMILIES = ("WaveGateAdapter", "ToposResonatorAdapter", "EllipticAnchoredResidualAdapter",
            "FractionalAngleGainHistoryAdapter")


def make_stack(config):
    torch.manual_seed(config["seed"])
    rows = config["placements"]
    proof.require(isinstance(rows, list) and len(rows) == len(FAMILIES)
                  and [r["adapter"] for r in rows] == list(FAMILIES)
                  and len({r["path"] for r in rows}) == len(rows), "invalid mixed placement recipe")
    return st.GeometryAdapterStack({row["path"]: getattr(st, row["adapter"])(
        config["features"], **row["options"]) for row in rows})


@contextmanager
def manual_insertion(model, stack):
    """Independent reference for the previous example-level placement idiom."""
    originals = []
    try:
        for path, adapter in zip(stack.paths, stack.adapters):
            prefix, _, child = path.rpartition(".")
            parent = model.get_submodule(prefix) if prefix else model
            original = parent.get_submodule(child)
            parent.add_module(child, torch.nn.Sequential(original, adapter))
            originals.append((parent, child, original))
        yield
    finally:
        for parent, child, original in reversed(originals):
            parent.add_module(child, original)


def run(model, tokens, config, public, saved=None):
    stack = make_stack(config)
    optimizer = proof.make_optimizer(stack, config)
    schedule = proof.batch_schedule(config, len(tokens))
    cursor = 0
    if saved:
        proof.require(saved["cursor"] == config["checkpoint_after"], "checkpoint cursor differs")
        stack.load_state_dict(saved["adapter"])
        proof.common.named_adam(saved, stack, optimizer, saved["cursor"])
        torch.set_rng_state(saved["rng"])
        cursor = saved["cursor"]
    before = proof.model_digest(model)
    original_names = list(dict(model.named_parameters()))
    records, raw_gradients, midpoint = [], [], None

    def snapshot(step):
        return copy.deepcopy({"cursor": step, "adapter": stack.state_dict(),
                              "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()})

    with (stack.attach(model) if public else manual_insertion(model, stack)):
        if public:
            proof.require(list(dict(model.named_parameters())) == original_names, "base registration changed")
        for step in range(cursor, config["steps"]):
            optimizer.zero_grad(set_to_none=True)
            batch = tokens[schedule[step]]
            loss = model(batch, labels=batch, use_cache=False).loss
            proof.require(bool(torch.isfinite(loss)), "nonfinite loss")
            loss.backward()
            gradients = {name: p.grad.detach().clone() for name, p in stack.named_parameters() if p.grad is not None}
            proof.require(len(gradients) == len(list(stack.parameters())), "missing geometry gradient")
            receipts = {name: proof.tensor_receipt(value) for name, value in gradients.items()}
            if step >= config["checkpoint_after"]:
                proof.require(all(row["nonzero"] > 0 for row in receipts.values()), "inactive geometry parameter")
            optimizer.step()
            records.append({"step": step + 1, "batch_indices": schedule[step],
                            "loss": proof.tensor_receipt(loss), "loss_value": float(loss.detach()),
                            "gradients": receipts,
                            "parameters": {n: proof.tensor_receipt(p) for n, p in stack.named_parameters()}})
            raw_gradients.append(gradients)
            print(json.dumps({"public_api": public, "start_cursor": cursor,
                              "step": step + 1, "loss": records[-1]["loss_value"]}), flush=True)
            if step + 1 == config["checkpoint_after"]:
                midpoint = snapshot(step + 1)
    proof.require(proof.model_digest(model) == before and all(p.grad is None for p in model.parameters()),
                  "frozen base changed")
    final = snapshot(config["steps"])
    proof.common.named_adam(final, stack, optimizer, config["steps"])
    return {"start_cursor": cursor, "records": records, "gradients": raw_gradients,
            "final": final, "midpoint": midpoint,
            "parameter_count": sum(p.numel() for p in stack.parameters()),
            "backends": [a.execution_backend for a in stack.adapters], "base_sha256": before}


def compare(reference, public, resumed, checkpoint_after):
    proof.require(proof.equal(reference["records"], public["records"])
                  and proof.equal(reference["records"][checkpoint_after:], resumed["records"]), "update replay differs")
    proof.require(proof.equal(reference["gradients"], public["gradients"])
                  and proof.equal(reference["gradients"][checkpoint_after:], resumed["gradients"]), "raw gradients differ")
    proof.require(proof.equal(reference["final"], public["final"])
                  and proof.equal(reference["final"], resumed["final"]), "saved parameters/Adam/RNG differ")
    proof.require(reference["base_sha256"] == public["base_sha256"] == resumed["base_sha256"], "base identity differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "corpus", "config", "package-root", "runtime-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    proof.require(not args.output.exists(), "output exists")
    config = json.loads(args.config.read_bytes())
    proof.require(config["schema"] == "spiraltorch.geometry_adapter_stack_example.v1"
                  and config["steps"] == 3 and config["checkpoint_after"] == 2,
                  "this auxiliary example has a fixed three-update budget")
    proof.require(args.model_dir.name == config["model_snapshot"], "model snapshot differs")
    proof.require(proof.common.digest(args.corpus) == config["corpus_sha256"], "corpus differs")
    proof.require(Path(st.__file__).resolve().parent == (args.package_root / "spiraltorch").resolve(), "wrong package")
    manifest = json.loads(args.runtime_manifest.read_bytes())
    proof.verify_runtime(args.package_root, manifest)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.model_max_length = 10**9
    text = args.corpus.read_text()
    proof.require(text.count(config["corpus_end_marker"]) == 1, "corpus end marker differs")
    text = text[:text.index(config["corpus_end_marker"])]
    cut = text.rfind("\n\n", 0, int(len(text) * .9))
    proof.require(cut > 0, "training split missing")
    ids = tokenizer.encode(text[:cut], add_special_tokens=False)
    size = config["block_size"]
    tokens = torch.tensor(ids[:len(ids) // size * size], dtype=torch.long).reshape(-1, size)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32).cpu().eval().requires_grad_(False)
    model.config.use_cache = False
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        reference = run(model, tokens, config, False)
        public = run(model, tokens, config, True)
        midpoint = proof.save_state(args.output / "midpoint.pt", reference["midpoint"])
        saved = torch.load(args.output / "midpoint.pt", weights_only=True, map_location="cpu")
        resumed = run(model, tokens, config, True, saved=saved)
        compare(reference, public, resumed, config["checkpoint_after"])
        state = proof.save_state(args.output / "states.pt", {"reference": reference, "public": public, "resumed": resumed})
        proof.verify_runtime(args.package_root, manifest)
        report = {"schema": config["schema"], "status": "bitwise_exact", "config": config,
                  "source_sha256": {Path(m.__file__).name: proof.common.digest(Path(m.__file__))
                      for m in (placement, proof, proof.common)},
                  "example_sha256": proof.common.digest(Path(__file__)),
                  "native_sha256": proof.common.digest(Path(native.__file__)),
                  "runtime_manifest_sha256": proof.common.digest(args.runtime_manifest),
                  "torch": str(torch.__version__), "transformers": str(transformers.__version__),
                  "training_tokens": proof.tensor_receipt(tokens), "training_split_character_offset": cut,
                  "shape": [config["batch_size"], config["block_size"], config["features"]],
                  "auxiliary_updates": 7, "unique_trajectory_updates": 3,
                  "reference_records": reference["records"], "public_records": public["records"],
                  "resumed_records": resumed["records"], "backends": public["backends"],
                  "parameter_count": public["parameter_count"], "base_sha256": public["base_sha256"],
                  "base_unchanged": True, "state": state, "midpoint": midpoint,
                  "heldout_scoring": False, "timing_evidence": False,
                  "scope": "Public multi-site placement and mixed-family learning connection, not a quality result."}
        proof.write_json(args.output / "report.json", report)
    except Exception as error:
        proof.write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
