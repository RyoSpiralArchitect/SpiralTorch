"""Score pinned native snapshots on one pinned, labeled corpus without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import spiraltorch as st
from spiraltorch.native_session_io import read_limited, read_pinned_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True, help="JSON object with token_documents and unshifted labels")
    parser.add_argument("--documents-sha256", required=True, help="separately retained SHA-256 of that exact JSON file")
    kinds = parser.add_mutually_exclusive_group(required=True)
    kinds.add_argument("--variant", action="append", nargs=4,
                       metavar=("NAME", "CHECKPOINT", "PIN", "BOUNDARY"))
    kinds.add_argument("--conditional-variant", action="append", nargs=3,
                       metavar=("NAME", "CHECKPOINT", "PIN"), help="include the frozen source as variant 'source'")
    parser.add_argument("--source-lora", type=Path, help="frozen source LoRA JSON for conditional comparison")
    parser.add_argument("--source-geometry", type=Path, help="frozen source geometry JSON for conditional comparison")
    parser.add_argument("--source-geometry-id", help="separately retained geometry ID")
    parser.add_argument("--source-readout", choices=("f32", "f64", "f32_wide_backward"))
    parser.add_argument("--reference", required=True, help="variant to subtract from each other variant")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--microbatch-windows", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    conditional = args.conditional_variant is not None
    variants = args.conditional_variant if conditional else args.variant
    if not conditional and any(v is not None for v in (
        args.source_lora, args.source_geometry, args.source_geometry_id, args.source_readout,
    )):
        parser.error("source flags require conditional variants")
    if (args.source_geometry is None) != (args.source_geometry_id is None):
        parser.error("source geometry requires both exact bytes and its separately retained ID")
    names = (["source"] if conditional else []) + [arm[0] for arm in variants]
    if (not 2 <= len(names) <= st.NativeCausalLmAnswerComparison.MAX_VARIANTS or len(set(names)) != len(names)
            or not all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", name) for name in names)):
        parser.error("select 2 to 16 variants with distinct short names")
    if args.reference not in names:
        parser.error("reference must name a selected variant")
    if os.path.lexists(args.output):
        parser.error("output must be a new file")
    if not args.output.parent.is_dir():
        parser.error("output needs an existing parent directory")
    raw = read_limited(args.documents, 256 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != args.documents_sha256:
        parser.error("evaluation corpus differs from pinned SHA-256")
    read_corpus = getattr(st._rs, "_read_labeled_answer_corpus", None)
    if read_corpus is None:
        parser.error("bounded answer corpus loading needs a matching SpiralTorch build")
    try:
        documents, labels = read_corpus(raw, len(names))
    except ValueError as exc:
        parser.error(f"invalid labeled evaluation corpus: {exc}")
    del raw

    snapshots = []
    if conditional:
        serving = getattr(st, "NativeCausalLmConditionalServingSnapshot", None)
        if serving is None or not hasattr(serving, "evaluate_variants_by_answer"):
            parser.error("conditional comparison needs a matching SpiralTorch build")
        for _, path, pin in variants:
            raw = read_limited(Path(path), st.NativeCausalLmConditionalAnswerSession.MAX_CHECKPOINT_BYTES)
            if "sha256:" + hashlib.sha256(raw).hexdigest() != pin:
                raise ValueError("conditional checkpoint differs from pinned identity")
            snapshots.append((raw, pin))

    base = st.NativeCausalLm.from_gpt2(read_limited((args.model_dir/"config.json").resolve(strict=True), 1_048_576).decode(),
                                       st.SafeTensorArchive(args.model_dir/"model.safetensors"))
    common = (documents, args.context)
    options = dict(microbatch_windows=args.microbatch_windows, labels=labels)
    if conditional:
        if args.source_readout is not None:
            base = base.with_readout_accumulator(args.source_readout)
        if args.source_lora is not None:
            base = base.with_lora_json(read_limited(args.source_lora, 64 << 20).decode())
        if args.source_geometry is not None:
            geometry = st.GeometryAdapterArtifact.from_json(
                read_limited(args.source_geometry, 64 << 20).decode(), args.source_geometry_id, base.base_model_identity)
            base = base.with_geometry(geometry)
        decode = st.NativeCausalLmConditionalAnswerSession.serving_from_checkpoint
        scored = serving.evaluate_variants_by_answer(base, [decode(base, *saved) for saved in snapshots], *common, **options)
        scores = json.loads(scored.to_json())
        boundaries = [dict(name="source", variant=0)] + [
            dict(name=names[entry["variant"]], **entry) for entry in scores["snapshots"]
        ]
        schema = "spiraltorch.pinned_conditional_answer_comparison.v1"
    else:
        models, boundaries = [], []
        for name, checkpoint, pin, metadata in variants:
            raw, boundary = read_pinned_session(Path(checkpoint), pin, Path(metadata))
            models.append(st.NativeCausalLm.from_session_checkpoint_at(base, raw, pin, **boundary))
            del raw
            boundaries.append(dict(name=name, checkpoint_id=pin, **boundary))
        if len({(b["update"], b["recorded_data_id"]) for b in boundaries}) != 1:
            parser.error("snapshots must agree on completed update and recorded training corpus")
        scored = st.NativeCausalLm.evaluate_variants_by_answer(models, *common, **options)
        scores = json.loads(scored.to_json())
        schema = "spiraltorch.pinned_answer_comparison.v1"
    ref = names.index(args.reference)
    result = dict(schema=schema, reference=args.reference,
                  documents_sha256=args.documents_sha256, variants=boundaries,
                  contrasts={name:scored.contrast(ref,i) for i,name in enumerate(names) if i != ref},
                  scores=scores, training_documents_verified=False, split_independence_verified=False)
    # Publish only the complete result; a raced output name is never overwritten.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp)/"result"
        with open(staged, "x", encoding="utf8", opener=lambda p,f: os.open(p,f,0o600)) as stream:
            json.dump(result, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.link(staged, args.output)
    print(json.dumps({"reference":args.reference, "answers":len(scored), "contrasts":result["contrasts"]}, allow_nan=False))


if __name__ == "__main__":
    main()
