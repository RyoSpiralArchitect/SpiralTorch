"""Bounded offline GPT-2 mechanics, not a language-quality or speed study."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
PRIOR_PATH = ROOT / "benchmarks/results/2026-10-18-native-lm-generation/qualify.py"
PRIOR = runpy.run_path(str(PRIOR_PATH))
SOURCES = [
    "crates/st-tensor/src/classification/unlikelihood.rs",
    "crates/st-core/src/runtime/zspace_repetition_unlikelihood.rs",
    "crates/st-nn/src/causal_lm/repetition.rs", "crates/st-nn/src/causal_lm/objective.rs",
    "crates/st-nn/src/causal_lm/learning.rs", "crates/st-nn/src/causal_lm/learning/accumulation.rs",
    "crates/st-nn/src/causal_lm/training.rs", "bindings/st-py/src/causal_lm.rs",
    "bindings/st-py/src/causal_lm/accumulation.rs", "bindings/st-py/src/causal_lm/training.rs",
]
CONFIGS = {
    name: {"kind": "native_v1", "config": {
        "strength": .3, "context_window": 32, "max_candidates_per_position": 8,
        "candidate_source": {"kind": name, **(
            {"ngram_order": 2} if name == "prior_continuation" else {"proposal_top_k": 32})},
    }}
    for name in ("prior_continuation", "model_topk_history", "model_topk_periodic")
}
digest, save, load = PRIOR["digest"], PRIOR["save"], PRIOR["load"]


def pinned_input(args):
    PRIOR["PRIOR"]["check_model"](args.model_dir)
    if digest(args.data) != PRIOR["DATA"] or digest(args.checkpoint) != PRIOR["CHECKPOINT"]:
        raise ValueError("only existing admitted data and checkpoint are allowed")
    tokens = load(args.data)["token_documents"][0]
    # One ordinary train window and one explicitly synthetic repetition stress window.
    return [tokens[:32], tokens[128:130] * 15 + tokens[130:132]]


def protocol(st, documents):
    paths = [Path(__file__).resolve(), PRIOR_PATH, *[ROOT / p for p in SOURCES]]
    return {"schema": "spiraltorch.native_repetition_qualification.v1",
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
            "extension_sha256": digest(Path(st._rs.__file__)), "configs": CONFIGS,
            "data_sha256": PRIOR["DATA"], "checkpoint_sha256": PRIOR["CHECKPOINT"],
            "base_model_identity": PRIOR["PRIOR"]["BASE"], "model_sha256": PRIOR["PRIOR"]["MODEL"],
            "documents_sha256": hashlib.sha256(json.dumps(documents, separators=(",", ":")).encode()).hexdigest(),
            "windows": ["train_doc0[0:32]", "synthetic:train_doc0[128:130]*15+train_doc0[130:132]"],
            "training": {"context_length": 32, "microbatch_windows": 1, "accumulation_steps": 2,
                         "epochs": 2, "learning_rate": .001, "max_gradient_norm": .1},
            "readout": "f64", "gates": {"resume_reports_exact": True, "resume_checkpoint_bytes_exact": True},
            "scope": "six geometry-only updates on frozen GPT-2; includes synthetic stress, not quality or speed evidence"}


def checkpoint(path, trainer):
    payload = trainer.to_json()
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    if "sha256:" + digest(path) != trainer.checkpoint_id:
        raise ValueError("checkpoint identity mismatch")
    return {"file": path.name, "sha256": digest(path)}


def validate_results(cohort, raw, replay):
    if set(raw) != set(CONFIGS) or set(replay) != set(CONFIGS):
        raise ValueError("all declared conditions are required")
    for name in CONFIGS:
        if replay[name] is not True:
            raise ValueError("exact continuation must be true")
        row = raw[name]
        if len(row["reports"]) != 2 or [r["update"] for r in row["reports"]] != [1, 2]:
            raise ValueError("both accepted updates must be retained")
        for field, number in (("first", 1), ("final", 2)):
            path = cohort / (name + "-step" + str(number) + ".json")
            if row[field] != {"file": path.name, "sha256": digest(path)}:
                raise ValueError("retained checkpoint receipt mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "native", "resume", "public"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    PRIOR["PRIOR"]["forbid_ml"]()
    import spiraltorch as st
    documents = pinned_input(args)
    contract = protocol(st, documents)
    if args.mode == "prepare":
        args.cohort.mkdir(mode=0o700, parents=True, exist_ok=False)
        save(args.cohort / "contract.json", contract)
        return
    if load(args.cohort / "contract.json") != contract:
        raise ValueError("source, binary or conditions changed after sealing")
    if args.mode == "public":
        raw, replay = load(args.cohort / "native.json"), load(args.cohort / "resume.json")
        validate_results(args.cohort, raw, replay)
        if args.output is None:
            raise ValueError("--output is required for publication")
        save(args.output, {"contract": contract, "conditions": raw, "resume_exact": replay,
                          "raw_receipts": {name: digest(args.cohort / name) for name in
                                           ("contract.json", "native.json", "resume.json")}})
        return
    owners = PRIOR["model"](args, st)
    base = owners["base"].with_readout_accumulator("f64")
    initial = owners["learned"].with_readout_accumulator("f64")
    result = {}
    for name, recipe in CONFIGS.items():
        if args.mode == "native":
            optimizer = st.NativeGeometryAdamW(initial.geometry_artifact(), max_gradient_norm=.1)
            trainer = st.NativeCausalLmTrainer(initial, optimizer, documents, 32,
                learning_rate=.001, microbatch_windows=1, accumulation_steps=2, epochs=2, repetition=recipe)
            reports = [json.loads(trainer.step().to_json())]
            first = checkpoint(args.cohort / (name + "-step1.json"), trainer)
            reports.append(json.loads(trainer.step().to_json()))
            final = checkpoint(args.cohort / (name + "-step2.json"), trainer)
            if not trainer.exhausted or trainer.completed_updates != 2:
                raise ValueError("unexpected native cursor")
            result[name] = {"reports": reports, "first": first, "final": final}
        else:
            expected = load(args.cohort / "native.json")[name]
            first = args.cohort / (name + "-step1.json")
            final = args.cohort / (name + "-step2.json")
            if digest(first) != expected["first"]["sha256"] or digest(final) != expected["final"]["sha256"]:
                raise ValueError("retained checkpoint bytes changed")
            trainer = st.NativeCausalLmTrainer.from_json(base, documents, first.read_text(),
                                                        "sha256:" + expected["first"]["sha256"])
            report = json.loads(trainer.step().to_json())
            result[name] = report == expected["reports"][1] and trainer.to_json() == final.read_text()
            if not result[name]:
                raise ValueError("fresh-process continuation mismatch: " + name)
        print(name + " " + args.mode + " complete", flush=True)
    save(args.cohort / (args.mode + ".json"), result)


if __name__ == "__main__":
    main()
