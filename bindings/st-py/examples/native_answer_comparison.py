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
    parser.add_argument("--legacy-conditional-checkpoints", action="store_true",
                        help="read historical repeated/family artifacts for inference only")
    parser.add_argument("--source-lora", type=Path, help="frozen source LoRA JSON for conditional comparison")
    parser.add_argument("--source-geometry", type=Path, help="frozen source geometry JSON for conditional comparison")
    parser.add_argument("--source-geometry-id", help="separately retained geometry ID")
    parser.add_argument("--source-readout", choices=("f32", "f64", "f32_wide_backward"))
    parser.add_argument("--reference", required=True, help="variant to subtract from each other variant")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--microbatch-windows", type=int, default=1)
    parser.add_argument("--paired-greedy", type=int, nargs=2, metavar=("MAX_NEW_TOKENS", "STOP_ID"),
                        help="also compare prompt-only generation (conditional variants only)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    conditional = args.conditional_variant is not None
    variants = args.conditional_variant if conditional else args.variant
    if args.legacy_conditional_checkpoints and not conditional:
        parser.error("legacy conditional checkpoints require conditional variants")
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
    paired = args.paired_greedy
    if paired is not None and (not conditional or not 2 <= paired[0] <= 128 or paired[1] < 0):
        parser.error("paired greedy requires conditional variants, 2 to 128 new tokens and a nonnegative stop ID")
    if os.path.lexists(args.output):
        parser.error("output must be a new file")
    if not args.output.parent.is_dir():
        parser.error("output needs an existing parent directory")
    raw = read_limited(args.documents, (2 if paired is not None else 256) * 1024 * 1024)
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
    prompts = None
    if paired is not None:
        try:
            prompts = paired_prompts(documents, labels, len(names), args.context, *paired)
        except ValueError as exc:
            parser.error(str(exc))

    snapshots = []
    if conditional:
        serving = getattr(st, "NativeCausalLmConditionalServingSnapshot", None)
        if serving is None or not hasattr(serving, "evaluate_variants_by_answer"):
            parser.error("conditional comparison needs a matching SpiralTorch build")
        session = st.NativeCausalLmConditionalAnswerSession
        decode = (getattr(session, "serving_from_legacy_checkpoint", None) if args.legacy_conditional_checkpoints
                  else session.serving_from_checkpoint)
        if decode is None:
            parser.error("legacy conditional serving needs a matching SpiralTorch build")
        if prompts is None:
            snapshots = [
                read_pinned_projection(path, pin, session.MAX_CHECKPOINT_BYTES)
                for _, path, pin in variants
            ]

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
        if prompts is not None:
            try:
                recipe = admit_paired_generation(base, prompts, *paired)
            except ValueError as exc:
                parser.error(f"invalid paired greedy request: {exc}")
            owners = [
                decode(base, *read_pinned_projection(path, pin, session.MAX_CHECKPOINT_BYTES))
                for _, path, pin in variants
            ]
        else:
            owners = [decode(base, *saved) for saved in snapshots]
        del snapshots
        scored = serving.evaluate_variants_by_answer(base, owners, *common, **options)
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
    if args.legacy_conditional_checkpoints:
        result["legacy_conditional_checkpoints"] = True
    if prompts is not None:
        result["paired_greedy"] = paired_generation(base, owners, names, ref, scored, prompts, documents, recipe)
    # Publish only the complete result; a raced output name is never overwritten.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp)/"result"
        with open(staged, "x", encoding="utf8", opener=lambda p,f: os.open(p,f,0o600)) as stream:
            json.dump(result, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.link(staged, args.output)
    print(json.dumps({"reference":args.reference, "answers":len(scored), "contrasts":result["contrasts"]}, allow_nan=False))


def read_pinned_projection(path, pin, max_bytes):
    raw = read_limited(Path(path), max_bytes)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("conditional checkpoint differs from pinned identity")
    return raw, pin


def paired_prompts(documents, labels, variants, context, limit, stop):
    if not 1 <= len(documents) <= 128 or variants * len(documents) * limit > 8192:
        raise ValueError("paired greedy exceeds the case or generation step budget")
    prompts = []
    for doc, row in zip(documents, labels):
        start = next((i for i, label in enumerate(row) if label != -100), len(row))
        target = doc[start:]
        if (start == 0 or not 2 <= len(target) <= limit or row[start:] != target
                or target[-1] != stop or stop in target[:-1]):
            raise ValueError("paired greedy needs a masked prompt and an exact answer suffix ending in one stop ID")
        if start + limit > context:
            raise ValueError("paired greedy requires the full prompt and generation budget to fit one scoring window")
        prompts.append(doc[:start])
    # Full-prefix decoding recomputes the whole prefix at every step.
    if variants * sum(limit * len(p) + limit * (limit - 1) // 2 for p in prompts) > 131072:
        raise ValueError("paired greedy exceeds the full-prefix token budget")
    return prompts


def admit_paired_generation(base, prompts, limit, stop):
    recipe = {"max_new_tokens": limit, "eos_token_ids": [stop],
              "execution": {"kind": "full_prefix"}, "decoding": {"selection": "greedy"}}
    config = json.dumps(recipe)
    for length in sorted({len(prompt) for prompt in prompts}):
        st.NativeCausalLmGenerator.validate_prompt_length(base, length, config)
    return recipe


def paired_generation(base, owners, names, ref, scored, prompts, documents, recipe):
    limit, (stop,) = recipe["max_new_tokens"], recipe["eos_token_ids"]
    config = json.dumps(recipe)
    factories = [lambda prompt, policy: st.NativeCausalLmGenerator(base, prompt, policy)]
    factories.extend(owner.generator for owner in owners)
    rows = []
    joint = {name: {} for i, name in enumerate(names) if i != ref}
    if len(scored) != len(prompts):
        raise ValueError("paired greedy expected one scored answer per document")
    for j, (prompt, doc) in enumerate(zip(prompts, documents)):
        target = doc[len(prompt):]
        answer = scored.answer(j)
        if (answer["document"], answer["start"], answer["end"]) != (j, len(prompt), len(doc)):
            raise ValueError("paired greedy and supervised answer boundaries disagree")
        results = {}
        for i, (name, factory) in enumerate(zip(names, factories)):
            generator = factory(prompt, config)
            ids = generator.advance_ids(limit)
            reason = "eos" if ids and ids[-1] == stop else "length"
            if (not generator.finished or generator.finish_reason != reason or not 1 <= len(ids) <= limit
                    or stop in ids[:-1] or (reason == "length" and len(ids) != limit)
                    or generator.token_ids != prompt + ids):
                raise ValueError("invalid paired greedy generation boundary")
            results[name] = dict(generated_ids=ids, finish_reason=reason, exact=ids == target,
                                 first_token_correct=ids[0] == target[0], nll=answer["scores"][i]["nll"])
        reference = results[names[ref]]
        for name, counts in joint.items():
            candidate = results[name]
            delta = "lower" if candidate["nll"] < reference["nll"] else (
                "higher" if candidate["nll"] > reference["nll"] else "equal")
            change = int(candidate["exact"]) - int(reference["exact"])
            outcome = {-1: "lost", 0: "unchanged", 1: "gained"}[change]
            key = f"nll_{delta}_exact_{outcome}"
            counts[key] = counts.get(key, 0) + 1
        rows.append(dict(document=j, variants=results))
    return dict(schema="spiraltorch.native_paired_greedy.v1", recipe=recipe, cases=rows,
                joint_counts=joint, exact_metric="answer_token_ids_including_stop")


if __name__ == "__main__":
    main()
