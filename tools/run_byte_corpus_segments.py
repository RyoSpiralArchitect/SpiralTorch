"""Run one fresh, private study through declared checkpoints via the Rust owner.

Existing output directories are never reused. Failed runs retain their claim
and partial files; recovery is an explicit operation, not an automatic retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    raw = value.encode() if isinstance(value, str) else (
        json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as output:
        output.write(raw)
    return digest(raw)


def run(request, output, stops, *, study_type=None):
    if study_type is None:
        import spiraltorch as st
        study_type = st.nn.ByteCorpusStudy
    raw = request.read_bytes()
    study = study_type.from_json(raw)
    if (not stops or len(stops) > 32
            or any(type(stop) is not int for stop in stops)
            or stops != sorted(set(stops)) or stops[-1] != study.total_updates):
        raise ValueError("declare 1..32 increasing cursors ending at total_updates")
    for stop in stops:
        study.validate_segment(stop)
    # mkdir is the exclusive launch claim, before the first GPU request.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_new(output / "claim.json", dict(
        schema="spiraltorch.byte_corpus.segment_launch.v1",
        pid=os.getpid(), request_sha256=digest(raw), stops=stops,
        runner_sha256=digest(Path(__file__).read_bytes()), before_gpu_launch=True))
    write_new(output / "request.json", raw.decode("utf-8"))
    checkpoint = None
    receipts = []
    try:
        for stop in stops:
            print(json.dumps(dict(event="segment_started", stop_after=stop)), flush=True)
            started = time.monotonic()
            segment = study.advance_wgpu(stop, resume=checkpoint)
            checkpoint = segment.checkpoint
            if (checkpoint.completed_updates != stop
                    or checkpoint.request_sha256 != study.request_sha256):
                raise ValueError("Rust segment returned different cursor or request")
            report_hash = write_new(output / f"report-{stop}.json", segment.report_json)
            checkpoint_hash = write_new(output / f"checkpoint-{stop}.json", checkpoint.to_json())
            receipt = dict(stop_after=stop, seconds=time.monotonic() - started,
                           report_sha256=report_hash, checkpoint_sha256=checkpoint_hash)
            write_new(output / f"segment-{stop}.json", receipt)
            receipts.append(receipt)
            print(json.dumps(dict(event="segment_completed", **receipt)), flush=True)
        write_new(output / "completed.json", dict(request_sha256=study.request_sha256,
                                                   segments=receipts))
    except BaseException as error:
        write_new(output / "failed.json", dict(type=type(error).__name__, message=str(error),
                                                completed_segments=receipts))
        raise
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stop", type=int, action="append", required=True)
    args = parser.parse_args()
    run(args.request, args.output, args.stop)


if __name__ == "__main__":
    main()
