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
    parser.add_argument("--documents", type=Path, required=True, help="JSON array of encoded documents")
    parser.add_argument("--geometry", type=Path, help="Native geometry artifact JSON, required for joint/geometry")
    parser.add_argument("--mode", choices=("lora", "geometry", "joint"), default="joint")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-windows", type=int, help="independent fixed-width windows in one native SGD update")
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--readout", choices=("f32", "f64"), default="f32")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, metavar="N",
                        help="commit at absolute N-update boundaries and at the final step; write OUTPUT.progress.jsonl")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--expected-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 0:
        parser.error("--updates must be nonnegative")
    if args.resume and args.batch_windows is not None:
        parser.error("resumed sessions own their batch size")
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
    weights = st.SafeTensorArchive(args.weights)
    base = st.NativeCausalLm.from_gpt2(read_limited(args.config, 1_048_576).decode(), weights)
    documents = json.loads(read_limited(args.documents, 256 * 1024 * 1024))
    if payload is not None:
        session = st.NativeCausalLmTrainingSession.from_json(
            base, documents, payload, args.expected_id)
        del payload
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
            mode=args.mode, epochs=args.epochs, batch_windows=args.batch_windows)
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
