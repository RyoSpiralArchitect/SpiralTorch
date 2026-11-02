"""File-only readback of the local trio and its pinned historical comparator."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_embedding_trio import ARCHIVE, NATIVE, ROLES, SELECTION, WHEEL
from byte_release_artifacts import sha256
from compare_embedding_bits import compare, compare_reports, decode_json, read_pair, report_bytes, require
from byte_study_diagnostics import exact_report_difference_summary


def word(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def exact(w):
    return Fraction(struct.unpack("<f", struct.pack("<I", w))[0])


def digest(b):
    return hashlib.sha256(b).hexdigest()


def summary(root, archive):
    files = {"exit.json", "protocol.json", "preflight.json", "worker-started.json", "worker.log",
             "result.json", "full-reports.json"} | {
        f"{role}/{name}" for role in ROLES for name in ("entered.json", "host-call.json", "run-outcome.json", "bits.json")}
    require(not any(p.is_symlink() for p in root.rglob("*"))
            and {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == files,
            "unexpected private attempt files")
    read = lambda name: decode_json((root / name).read_bytes())
    receipt, plan, result = (read(name + ".json") for name in ("preflight", "protocol", "result"))
    require(read("exit.json") == {"returncode": 0, "retries": 0} and plan["max_seconds"] == 600
            and plan["retries"] == 0 and plan["roles"] == list(ROLES)
            and plan["wheel_sha256"] == receipt["installation"]["wheel_sha256"] == WHEEL
            and plan["native_sha256"] == receipt["installation"]["native_sha256"] == NATIVE
            and plan["archive_sha256"] == receipt["archive_sha256"] == ARCHIVE
            and plan["selection"] == receipt["selection"] == SELECTION
            and receipt["installation"]["all_original_files_match"] is True,
            "unexpected run identity or terminal status")
    envelopes = [(root / role / "run-outcome.json").read_bytes() for role in ROLES]
    reports = compare_reports(*envelopes)
    pairs = [read_pair(root / role) for role in ROLES]
    coords = compare(*pairs)
    times = [read(role + "/host-call.json") for role in ROLES]
    overlap = min(t["end_ns"] for t in times[1:]) - max(t["start_ns"] for t in times[1:])
    require(times[0]["end_ns"] <= min(t["start_ns"] for t in times[1:])
            and all(t["end_ns"] > t["start_ns"] for t in times)
            and result["worker_host_overlap_ns"] == overlap > 0
            and result["native_calls"] == 3 and result["status"] == "no_difference_in_this_instrumented_trio"
            and result["release_ready"] is False and result["coordinate"] == coords
            and result["full_reports"] == read("full-reports.json") == reports
            and all(w["raw_equal"] for w in reports["workers"]) and not any(w["differences"] for w in coords["workers"]),
            "recorded local null result does not follow from saved outputs")
    require(sha256(archive) == ARCHIVE, "different historical archive")
    local = report_bytes(envelopes[0])
    cross = []
    with zipfile.ZipFile(archive) as z:
        request = z.read("round-0/request.json")
        require(digest(request) == SELECTION["request_sha256"], "different historical request")
        table = decode_json(request)["cases"][1]["parameters"][0]
        require(table["name"] == "token_embedding" and table["shape"] == [256, 2], "unexpected table")
        initial = word(table["values"][4])
        for name in ("expected", "actual-0", "actual-1"):
            historical = z.read(f"round-0/{name}.json")
            diff = exact_report_difference_summary(historical, local)
            require(diff["complete"], "incomplete cross-environment comparison")
            value = decode_json(historical)["cases"][1]["final_parameters"][0][4]
            cross.append({"archive_member": name, "hosted_final_word": f"{word(value):08x}",
                          "full_report_byte_equal": diff["raw_equal"],
                          "differing_fields": {k: v["count"] for k, v in diff["groups"].items()}})
    bits = pairs[0][1]["f32_bits"]
    w0, g, w1 = (bits[k] for k in ("weight_before", "token_parameter_gradient", "weight_after"))
    # This particular observed weight is positive and normal. Test the interior
    # of its rounding cell with exact rationals, independent of any GPU output.
    require(0x00800000 < w1 < 0x7f7fffff, "unsupported rounding-cell probe")
    mid = exact(w1)
    low, high = (exact(w1-1) + mid)/2, (exact(w1+1) + mid)/2
    product = exact(SELECTION["rate_f32_bits"]) * exact(g)  # exact in binary64 for these f32 inputs
    return {
        "schema": "spiraltorch.pr2301.local_embedding_trio.v1", "status": result["status"],
        "request_sha256": SELECTION["request_sha256"], "wheel_sha256": WHEEL, "native_sha256": NATIVE,
        "archive_sha256": ARCHIVE, "harness_sha256": plan["harness_sha256"],
        "comparator_sha256": plan["comparator_sha256"], "adapter": reports["adapter"],
        "native_calls": 3, "retries": 0, "worker_host_overlap_ns": overlap,
        "local_complete_report_sha256": dict(zip(ROLES, reports["report_sha256"], strict=True)),
        "request_initial_word": f"{initial:08x}",
        "local_coordinate_f32_hex": {k: f"{v:08x}" for k, v in bits.items()},
        "local_final_word": f'{word(decode_json(local)["cases"][1]["final_parameters"][0][4]):08x}',
        "local_sgd_inside_observed_rounding_cell": {
            "exact_product_then_subtract": low < exact(w0) - product < high,
            "rounded_f32_product_then_subtract": low < exact(w0) - exact(word(float(product))) < high,
        },
        "hosted_vs_local_serial": cross,
        "private_file_sha256": {name: sha256(root / name) for name in sorted(files)},
        "historical_intermediate_values": "not recorded", "cause": "not established", "release_ready": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_trio", type=Path)
    parser.add_argument("historical_zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(summary(args.private_trio, args.historical_zip), indent=2))
