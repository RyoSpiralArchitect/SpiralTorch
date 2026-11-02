"""Offline only: compare a future instrumented trio, not the historical run."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from byte_corpus_study import decode_json

SELECTION = dict(
    request_sha256="b18f2e92fa1b87c5254ad865e07102b26a9df7df870ef342eb73d0b3a5bfdd8f",
    case="geometry", update=1, token=2, feature=0, input_position=[0, 1], rate_f32_bits=1028443341,
)
STAGES = ("weight_before", "embedding_output_gradient", "token_parameter_gradient", "weight_after")
FINAL = "final_report_weight"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(outcome, record):
    # Canonical JSON comparison rejects bool/float aliases for integer metadata.
    expected = json.dumps(SELECTION, sort_keys=True)
    require(outcome["schema"] == "spiraltorch.byte_embedding_update.observation.v1"
            and outcome["status"] == "completed" and outcome["capture_status"] == "views_retained"
            and outcome["release_ready"] is False, "incomplete or different observed run")
    require(record["schema"] == "spiraltorch.byte_embedding_update.bits.v1"
            and json.dumps(record["attempted_revisions"]) == "[0, 1]"
            and all(json.dumps(v["selection"], sort_keys=True) == expected for v in (outcome, record)),
            "different request, selection, or revisions")
    report = outcome["report"]
    require(report["schema"] == "spiraltorch.byte_corpus.result.v1" and report["engine"] == "spiraltorch"
            and report["request_sha256"] == SELECTION["request_sha256"]
            and [c["name"] for c in report["cases"]] == ["plain", "geometry"]
            and all(json.dumps([p["revision"] for p in c["training"]]) == "[1, 2]" for c in report["cases"]),
            "different or incomplete report")
    bits = record["f32_bits"]
    require(set(bits) == set(STAGES) and all(type(v) is int and 0 <= v <= 0xffffffff
            and v & 0x7f800000 != 0x7f800000 for v in bits.values()), "expected four finite f32 words")
    table = report["cases"][1]["final_parameters"][0]
    require(len(table) == 512 and type(table[4]) in (int, float) and math.isfinite(table[4]),
            "missing final token-2 weight")
    return report["adapter"], {**bits, FINAL: struct.unpack("<I", struct.pack("<f", table[4]))[0]}


def compare(serial, worker0, worker1):
    adapter, reference = checked(*serial)
    rows = []
    for worker, pair in enumerate((worker0, worker1)):
        target, actual = checked(*pair)
        require(isinstance(adapter, str) and adapter and adapter == target, "different reported adapter")
        differences = [{"stage": stage, "serial_bits": reference[stage], "worker_bits": actual[stage]}
                       for stage in (*STAGES, FINAL) if reference[stage] != actual[stage]]
        rows.append({"worker": worker, "first_observed_difference": differences[0]["stage"] if differences else None,
                     "differences": differences})
    return {"selection": SELECTION, "workers": rows, "scope": "one coordinate, not causality or qualification"}


def read_pair(path):
    return tuple(decode_json((path / name).read_bytes()) for name in ("run-outcome.json", "bits.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("serial", "worker0", "worker1"):
        parser.add_argument(role, type=Path)
    args = parser.parse_args()
    paths = [args.serial, args.worker0, args.worker1]
    require(len({p.resolve() for p in paths}) == 3, "expected three separate run directories")
    print(json.dumps(compare(*(read_pair(p) for p in paths)), indent=2))
