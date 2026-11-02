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
    if args.output.exists():
        parser.error("output must be a new checkpoint file")
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
    reports = []
    for _ in range(args.updates):
        report = session.step()
        if report is None:
            break
        reports.append(json.loads(report.to_json()))
    # Writing an output file is caller-owned; the Rust checkpoint itself is atomic
    # in memory, not a power-loss-safe filesystem transaction.
    payload, pin = session.checkpoint_with_id()
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"checkpoint_id": pin,
                      "recorded_data_id": session.data_id,
                      "post_lora_id": session.model.lora_artifact_id,
                      "post_geometry_id": session.model.geometry_artifact_id,
                      "completed_updates": session.completed_updates,
                      "total_updates": session.total_updates,
                      "batch_windows": session.batch_windows,
                      "readout_accumulator": session.model.readout_accumulator,
                      "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
