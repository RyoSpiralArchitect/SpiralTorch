"""Train native adapters through the Rust-owned document/session contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import spiraltorch as st


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
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--readout", choices=("f32", "f64"), default="f32")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--expected-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 0:
        parser.error("--updates must be nonnegative")
    weights = st.SafeTensorArchive(args.weights)
    base = st.NativeCausalLm.from_gpt2(args.config.read_text(), weights)
    documents = json.loads(args.documents.read_text())
    if args.resume:
        if not args.expected_id:
            parser.error("--resume needs the separately retained --expected-id")
        session = st.NativeCausalLmTrainingSession.from_json(
            base, documents, args.resume.read_text(), args.expected_id)
    else:
        model = base.with_readout_accumulator(args.readout)
        if args.mode in {"lora", "joint"}:
            model = model.with_lora(args.layer, args.rank, args.alpha)
        if args.mode in {"geometry", "joint"}:
            if not args.geometry:
                parser.error("geometry/joint mode needs --geometry")
            model = model.with_geometry(st.GeometryAdapterArtifact.create_json(args.geometry.read_text()))
        session = st.NativeCausalLmTrainingSession(
            model, documents, args.context, learning_rate=args.learning_rate,
            mode=args.mode, epochs=args.epochs)
    reports = []
    for _ in range(args.updates):
        report = session.step()
        if report is None:
            break
        reports.append(json.loads(report.to_json()))
    # Writing an output file is caller-owned; the Rust checkpoint itself is atomic
    # in memory, not a power-loss-safe filesystem transaction.
    payload, pin = session.checkpoint_with_id()
    args.output.write_text(payload)
    print(json.dumps({"checkpoint_id": pin,
                      "recorded_data_id": session.data_id,
                      "post_lora_id": session.model.lora_artifact_id,
                      "post_geometry_id": session.model.geometry_artifact_id,
                      "completed_updates": session.completed_updates,
                      "total_updates": session.total_updates,
                      "readout_accumulator": session.model.readout_accumulator,
                      "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
