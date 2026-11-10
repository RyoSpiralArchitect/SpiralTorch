"""Offline encoded-document QKV training/resume with a Rust-owned corpus schedule."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import spiraltorch as st
from spiraltorch.native_session_io import read_limited, read_pinned_session
from spiraltorch.native_qkv_runs import run_qkv_corpus


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recipe-json", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint-id")
    p.add_argument("--artifact", type=Path)
    p.add_argument("--artifact-id")
    p.add_argument("--session-checkpoint", type=Path, help="frozen LoRA/geometry source, not this QKV run")
    p.add_argument("--session-checkpoint-id")
    p.add_argument("--session-boundary", type=Path)
    p.add_argument("--readout-accumulator", choices=("f32", "f32_wide_backward", "f64"), default="f32")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--save-every", type=int, default=10)
    a = p.parse_args(argv)
    if (a.resume is None) != (a.checkpoint_id is None) or (a.resume is None) == (a.recipe_json is None):
        p.error("choose recipe-json, or resume with checkpoint-id and no replacement recipe")
    if (a.artifact is None) != (a.artifact_id is None):
        p.error("geometry source needs artifact and artifact-id")
    source = (a.session_checkpoint is not None, a.session_checkpoint_id is not None, a.session_boundary is not None)
    if any(source) and (not all(source) or a.artifact is not None):
        p.error("choose pinned geometry, or all three session-source arguments")
    if not 1 <= a.save_every <= st.NativeCausalLmQkvTrainer.MAX_CHECKPOINT_INTERVAL:
        p.error("save-every exceeds native checkpoint interval")
    if a.stop_after is not None and a.stop_after < 0:
        p.error("stop-after must be nonnegative")
    if os.path.lexists(a.output):
        p.error("output must be a new private directory")
    payload = None
    if a.resume is not None:
        payload = read_limited(a.resume, st.NativeCausalLmQkvTrainer.MAX_CHECKPOINT_BYTES)
        if "sha256:" + hashlib.sha256(payload).hexdigest() != a.checkpoint_id:
            raise ValueError("QKV session checkpoint differs from its retained pin")
    data = json.loads(read_limited(a.data_json, 256 * 1024 * 1024))
    if not isinstance(data, dict) or "token_documents" not in data or set(data) - {"token_documents", "labels"}:
        raise ValueError("data JSON needs token_documents and optional unshifted labels")
    model = st.NativeCausalLm.from_gpt2(
        read_limited((a.model_dir / "config.json").resolve(strict=True), 1_048_576).decode(),
        st.SafeTensorArchive(a.model_dir / "model.safetensors"),
    ).with_readout_accumulator(a.readout_accumulator)
    if a.artifact is not None:
        model = model.with_geometry(st.GeometryAdapterArtifact.from_json(
            read_limited(a.artifact, 64 * 1024 * 1024).decode(), a.artifact_id, model.base_model_identity))
    elif all(source):
        source_raw, boundary = read_pinned_session(a.session_checkpoint, a.session_checkpoint_id, a.session_boundary)
        model = st.NativeCausalLm.from_session_checkpoint_at(model, source_raw, a.session_checkpoint_id, **boundary)
        if model.readout_accumulator != a.readout_accumulator:
            raise ValueError("frozen source and explicit readout policy differ")
    if payload is not None:
        trainer = st.NativeCausalLmQkvTrainer.from_bytes(model, data["token_documents"], payload,
                                                       a.checkpoint_id, labels=data.get("labels"))
    else:
        recipe = json.loads(read_limited(a.recipe_json, 65_536))
        if not isinstance(recipe, dict) or set(recipe) != {"layer", "optimizer", "training"}:
            raise ValueError("recipe needs exactly layer, optimizer, training")
        if not isinstance(recipe["optimizer"], dict) or set(recipe["optimizer"]) != {"beta1", "beta2", "epsilon", "max_gradient_norm"}:
            raise ValueError("provide the complete optimizer/clipping recipe")
        if not isinstance(recipe["training"], dict) or set(recipe["training"]) != {
                "context_length", "learning_rate", "weight_decay", "label_smoothing", "epochs", "batch_windows", "max_rows"}:
            raise ValueError("provide the complete native corpus training recipe")
        opt = st.NativeCausalLmQkvAdamW(model, recipe["layer"], **recipe["optimizer"])
        trainer = st.NativeCausalLmQkvTrainer(opt, data["token_documents"], labels=data.get("labels"), **recipe["training"])
    result = run_qkv_corpus(trainer, a.output, stop_after=a.stop_after, save_every=a.save_every)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
