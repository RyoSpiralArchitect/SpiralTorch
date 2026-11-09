"""Train native adapters through the Rust-owned document/session contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

import spiraltorch as st


def read_limited(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
            raise ValueError("input must be a regular file within byte budget")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(info.st_size)
            if len(payload) != info.st_size or stream.read(1):
                raise ValueError("input size changed while reading")
    finally:
        os.close(fd)
    return payload


def text_documents(data, tokenizer):
    if not isinstance(data, dict) or set(data) != {"examples"} or not isinstance(data["examples"], list):
        raise ValueError("text input must be an object containing an examples array")
    if not 1 <= len(data["examples"]) <= st.NativeCausalLmTrainingSession.MAX_DOCUMENTS:
        raise ValueError("text example count outside training budget")
    documents, labels = [], []
    remaining = st.NativeCausalLmTrainingSession.MAX_TOKENS
    for i, item in enumerate(data["examples"]):
        try:
            if isinstance(item, dict) and set(item) == {"prompt", "completion"}:
                prompt, answer = item["prompt"], item["completion"]
                if not isinstance(prompt, str) or not isinstance(answer, str):
                    raise ValueError("prompt and completion must be strings")
                length = len(prompt) + len(answer)
                if length > st.CausalTargetAlignment.MAX_SOURCE_UNITS:
                    raise ValueError("text exceeds causal alignment budget")
                text, spans = prompt + answer, [(len(prompt), length)]
                ids, target = st.encode_supervised_document(tokenizer, text, spans)
            elif isinstance(item, dict) and set(item) == {"prompt", "completion", "terminator"}:
                example = st.encode_supervised_completion(tokenizer, item["prompt"], item["completion"], terminator=item["terminator"])
                ids, target = example.training_document()
            elif isinstance(item, dict) and set(item) == {"text", "target_spans"}:
                ids, target = st.encode_supervised_document(tokenizer, item["text"], item["target_spans"])
            else:
                raise ValueError("expected prompt/completion or text/target_spans")
            if len(ids) > remaining:
                raise ValueError("encoded token count exceeds training budget")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"text example {i}: {exc}") from exc
        remaining -= len(ids)
        documents.append(ids)
        labels.append(target)
    return documents, labels


def boundary(session, pin):
    model = session.model
    return {"checkpoint_id": pin, "recorded_data_id": session.data_id,
            "post_lora_id": model.lora_artifact_id, "post_geometry_id": model.geometry_artifact_id,
            "completed_updates": session.completed_updates, "total_updates": session.total_updates,
            "batch_windows": session.batch_windows, "readout_accumulator": model.readout_accumulator}


def checkpointed(session, output, updates, every, resume_pin):
    index = output.with_name(output.name + ".progress.jsonl")
    stop = min(session.completed_updates + updates, session.total_updates)

    def intermediate(cursor):
        return output.with_name(f"{output.name}.u{cursor:09d}.json")

    with open(index, "x", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        def publish(path, pin, reports):
            item = {**boundary(session, pin), "checkpoint_file": path.name if path else None,
                    "source": "new_checkpoint" if path else "pinned_resume_input",
                    "reports": [json.loads(r.to_json()) for r in reports]}
            stream.write(json.dumps(item, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        start = session.completed_updates
        if start == stop:
            pin = session.save_checkpoint_new(output)
            publish(output, pin, ())
        else:
            if resume_pin is None:
                anchor = intermediate(start)
                publish(anchor, session.save_checkpoint_new(anchor), ())
            else:
                publish(None, resume_pin, ())
            while session.completed_updates < stop:
                end = min(stop, (session.completed_updates // every + 1) * every)
                target = output if end == stop else intermediate(end)
                reports, pin = session.advance_to_checkpoint_new(end, target)
                publish(target, pin, reports)
    return {**boundary(session, pin), "progress_file": index.name, "reports_count": stop - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True,
                        help="token ID arrays, token_documents/labels, or text examples with a pinned local tokenizer")
    parser.add_argument("--tokenizer-json", type=Path, help="encode raw text examples with this local tokenizer.json")
    parser.add_argument("--tokenizer-sha256", help="independently retained SHA-256 of --tokenizer-json (64 hex digits)")
    parser.add_argument("--geometry", type=Path, help="Native geometry artifact JSON, required for joint/geometry")
    parser.add_argument("--mode", choices=("lora", "geometry", "joint"), default="joint")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-windows", type=int, help="independent fixed-width windows in one native SGD update")
    parser.add_argument("--window-policy", choices=("disjoint", "covering_targets"),
                        help="default: disjoint full windows; covering_targets keeps short documents and tails")
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--readout", choices=("f32", "f64", "f32_wide_backward"), default="f32")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, metavar="N",
                        help="commit at absolute N-update boundaries and at the final step; write OUTPUT.progress.jsonl")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--expected-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 0:
        parser.error("--updates must be nonnegative")
    if (args.tokenizer_json is None) != (args.tokenizer_sha256 is None):
        parser.error("raw text needs both --tokenizer-json and --tokenizer-sha256")
    if args.tokenizer_json is not None and args.window_policy == "disjoint":
        parser.error("raw-text training requires covering_targets to retain all labeled targets")
    if args.resume and args.batch_windows is not None:
        parser.error("resumed sessions own their batch size")
    if args.resume and args.window_policy is not None:
        parser.error("resumed sessions own their window policy")
    if bool(args.resume) != bool(args.expected_id):
        parser.error("resume needs both --resume and the separately retained --expected-id")
    if os.path.lexists(args.output):
        parser.error("output must be a new checkpoint file")
    if args.checkpoint_every is not None:
        if not 1 <= args.checkpoint_every <= st.NativeCausalLmTrainingSession.MAX_CHECKPOINT_INTERVAL:
            parser.error("checkpoint-every must be in the native checkpoint interval range")
        if os.path.lexists(args.output.with_name(args.output.name + ".progress.jsonl")):
            parser.error("checkpoint progress file must be new")
    payload = None
    if args.resume:
        payload = read_limited(args.resume, st.NativeCausalLmTrainingSession.MAX_CHECKPOINT_BYTES)
        if "sha256:" + hashlib.sha256(payload).hexdigest() != args.expected_id:
            raise ValueError("session checkpoint differs from pinned identity")
    documents = json.loads(read_limited(args.documents, 256 * 1024 * 1024))
    labels = None
    if args.tokenizer_json is not None:
        codec = read_limited(args.tokenizer_json, 32 * 1024 * 1024)
        if hashlib.sha256(codec).hexdigest() != args.tokenizer_sha256:
            parser.error("local tokenizer differs from pinned SHA-256")
        try:
            from tokenizers import Tokenizer  # never imported for encoded input
        except ModuleNotFoundError as exc:
            if exc.name != "tokenizers":
                raise
            parser.error("raw text requires the optional tokenizers package; encoded documents do not")
        documents, labels = text_documents(documents, Tokenizer.from_buffer(codec))
        del codec
    elif isinstance(documents, dict):
        if set(documents) != {"token_documents", "labels"} or not isinstance(documents["labels"], list):
            parser.error("labeled documents need token_documents and a labels array")
        documents, labels = documents["token_documents"], documents["labels"]
    if not isinstance(documents, list):
        parser.error("token documents must be a JSON array")
    weights = st.SafeTensorArchive(args.weights)
    base = st.NativeCausalLm.from_gpt2(read_limited(args.config, 1_048_576).decode(), weights)
    if payload is not None:
        session = st.NativeCausalLmTrainingSession.from_json(
            base, documents, payload, args.expected_id, labels=labels)
        del payload
        if args.tokenizer_json is not None and session.window_policy != "covering_targets":
            parser.error("raw-text resume requires a covering_targets checkpoint")
    else:
        model = base.with_readout_accumulator(args.readout)
        if args.mode in {"lora", "joint"}:
            model = model.with_lora(args.layer, args.rank, args.alpha)
        if args.mode in {"geometry", "joint"}:
            if not args.geometry:
                parser.error("geometry/joint mode needs --geometry")
            model = model.with_geometry(st.GeometryAdapterArtifact.create_json(read_limited(args.geometry, 64 * 1024 * 1024).decode()))
        session = st.NativeCausalLmTrainingSession(
            model, documents, args.context, learning_rate=args.learning_rate,
            mode=args.mode, epochs=args.epochs, batch_windows=args.batch_windows,
            labels=labels, window_policy=args.window_policy or ("covering_targets" if args.tokenizer_json else "disjoint"))
    if args.checkpoint_every is not None:
        print(json.dumps(checkpointed(session, args.output, args.updates, args.checkpoint_every, args.expected_id),
                         indent=2, allow_nan=False))
        return
    reports = []
    for _ in range(args.updates):
        report = session.step()
        if report is None:
            break
        reports.append(json.loads(report.to_json()))
    pin = session.save_checkpoint_new(args.output)
    print(json.dumps({**boundary(session, pin), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
