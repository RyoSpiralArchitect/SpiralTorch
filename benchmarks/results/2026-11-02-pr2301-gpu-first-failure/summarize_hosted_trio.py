"""Reconcile the sealed hosted run, original CI and M4 without loading a wheel."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hosted_embedding_environment as host
from compare_embedding_bits import compare, compare_reports, decode_json, read_pair, report_bytes, require
from run_embedding_trio import ARCHIVE, NATIVE, ROLES, SELECTION, WHEEL, sha256
from summarize_local_trio import digest, exact, word
from byte_study_diagnostics import exact_report_difference_summary


def inverse_negative_sgd(initial, rate, target, measured, *, split):
    """Exact preimage at a positive normal output, for g <= 0 and RN-even."""
    require(0x00800000 < initial < target < 0x7f7fffff and 0 < rate < 0x3f800000
            and 0x80000000 < measured < 0xff800000, "outside this coordinate's inverse scope")
    before, r, mid = exact(initial), exact(rate), exact(target)
    lower, upper = (exact(target - 1) + mid) / 2, (mid + exact(target + 1)) / 2
    def output(magnitude):
        change = r * exact(magnitude)
        # A product of two binary32 operands is exact in binary64 here.
        return before + (exact(word(float(change))) if split else change)
    def first(threshold, inclusive):
        lo, stop = 0, 0x7f800000  # nonnegative finite f32 magnitudes only
        while lo < stop:
            m = (lo + stop) // 2
            y = output(m)
            if y > threshold or (inclusive and y == threshold):
                stop = m
            else:
                lo = m + 1
        return lo
    even = target % 2 == 0
    first_mag, last_mag = first(lower, even), first(upper, not even) - 1
    require(first_mag <= last_mag < 0x7f800000, "no finite gradient reaches this cell")
    magnitude = measured & 0x7fffffff
    nearest = max(first_mag, min(magnitude, last_mag))
    return {"nearest_to_zero_hex": f"{0x80000000 | first_mag:08x}",
            "farthest_from_zero_hex": f"{0x80000000 | last_mag:08x}",
            "minimum_f32_steps_from_measured": abs(magnitude - nearest),
            "minimum_relative_change": float(abs(exact(nearest) / exact(magnitude) - 1))}


def summarize(root, original, local):
    evidence = root / "recovered" / "evidence"
    trio = evidence / "trio"
    read = lambda p: decode_json(p.read_bytes())
    run, art = read(root / "run.json"), read(root / "artifact.json")
    receipt, admitted, roster = (read(evidence / name) for name in ("admission.json", "trio/host-admission.json", "roster.json"))
    require(run["status"] == "completed" and run["conclusion"] == "success" and run["run_attempt"] == 1
            and run["event"] == "workflow_dispatch" and run["head_branch"] == host.BRANCH
            and not host.mismatches(receipt["observed"]) and receipt["mismatches"] == []
            and admitted == {"observed": receipt["observed"], "mismatches": []}
            and receipt["observed"]["github_sha"] == run["head_sha"] and receipt["observed"]["run_id"] == str(run["id"])
            and roster == {"count": 1, "ids": [run["id"]]}
            and art["workflow_run"]["id"] == run["id"] and art["workflow_run"]["head_sha"] == run["head_sha"]
            and art["digest"] == "sha256:" + sha256(root / "download.zip"), "unbound hosted evidence")
    with zipfile.ZipFile(root / "download.zip") as z:
        require(z.namelist() == ["evidence.tgz.age"]
                and digest(z.read("evidence.tgz.age")) == sha256(root / "evidence.tgz.age"), "different ciphertext")
    # The age authentication/decryption is verified separately, without exposing its identity.
    with tarfile.open(root / "evidence.tgz", "r:gz") as tar:
        members = [m for m in tar if not m.isdir()]
        originals = {str(p.relative_to(evidence.parent)): p for p in evidence.rglob("*") if p.is_file()}
        require(len(members) == len(originals) and {m.name for m in members} == set(originals)
                and all(m.isfile() and not originals[m.name].is_symlink()
                        and digest(tar.extractfile(m).read()) == sha256(originals[m.name]) for m in members),
                "extracted evidence differs from the recovered archive")
    pre, result = read(trio / "preflight.json"), read(trio / "result.json")
    require(read(trio / "exit.json") == {"returncode": 0, "retries": 0} and pre["archive_transferred"] is False
            and pre["installation"]["all_original_files_match"] is True and pre["installation"]["wheel_sha256"] == WHEEL
            and pre["installation"]["native_sha256"] == NATIVE and sha256(original) == ARCHIVE, "different installed/input bytes")
    roles = [trio / role for role in ROLES]
    raw = [(p / "run-outcome.json").read_bytes() for p in roles]
    full, coords = compare_reports(*raw), compare(*(read_pair(p) for p in roles))
    times = [read(p / "host-call.json") for p in roles]
    overlap = min(t["end_ns"] for t in times[1:]) - max(t["start_ns"] for t in times[1:])
    require(result["full_reports"] == full == read(trio / "full-reports.json") and full["adapter"] == host.ADAPTER
            and full["workers"][0]["raw_equal"] and full["workers"][1]["raw_equal"] and result["coordinate"] == coords
            and not any(row["differences"] for row in coords["workers"]) and result["native_calls"] == 3
            and result["status"] == "no_difference_in_this_instrumented_trio"
            and times[0]["end_ns"] <= min(t["start_ns"] for t in times[1:])
            and overlap == result["worker_host_overlap_ns"] > 0, "unexpected trio outcome")
    reference = report_bytes(raw[0])
    bits = read_pair(roles[0])[1]["f32_bits"]
    comparisons = {}
    def compare_saved(name, data):
        diff = exact_report_difference_summary(data, reference)
        require(diff["complete"], "partial report comparison")
        final = word(decode_json(data)["cases"][1]["final_parameters"][0][4])
        comparisons[name] = {"raw_equal": diff["raw_equal"], "final_word": f"{final:08x}",
                             "differing_fields": {k: g["count"] for k, g in diff["groups"].items()}}
        return final
    with zipfile.ZipFile(original) as z:
        request = z.read("round-0/request.json")
        require(digest(request) == SELECTION["request_sha256"], "different historical request")
        req = decode_json(request)
        table = req["cases"][1]["parameters"][0]
        from inspect_retained import input_occurrences
        require(table["name"] == "token_embedding" and table["shape"] == [256,2]
                and word(table["values"][4]) == bits["weight_before"]
                and req["schema"] == "spiraltorch.byte_corpus.request.v1"
                and input_occurrences(req, 2) == [1, 0] and word(req["rate"]) == SELECTION["rate_f32_bits"],
                "different coordinate or SGD recipe")
        for name in ("expected", "actual-0", "actual-1"):
            compare_saved("historical_" + name, z.read("round-0/" + name + ".json"))
    local_raw = (local / "serial" / "run-outcome.json").read_bytes()
    compare_saved("local_M4_serial", report_bytes(local_raw))
    local_bits = read_pair(local / "serial")[1]["f32_bits"]
    outlier = int(comparisons["historical_actual-0"]["final_word"], 16)
    return {"schema": "spiraltorch.pr2301.hosted_embedding_trio.v1", "run": run, "artifact_id": art["id"],
            "provider_artifact_digest": art["digest"], "ciphertext_sha256": sha256(root/"evidence.tgz.age"),
            "recovered_tar_sha256": sha256(root/"evidence.tgz"), "environment": receipt["observed"], "wheel_sha256": WHEEL,
            "native_sha256": NATIVE, "request_sha256": SELECTION["request_sha256"], "archive_transferred": False,
            "status": result["status"], "native_calls": 3, "retries": 0, "worker_host_overlap_ns": overlap,
            "complete_report_sha256": dict(zip(ROLES, full["report_sha256"], strict=True)),
            "hosted_coordinate_hex": {k: f"{v:08x}" for k,v in bits.items()},
            "local_coordinate_hex": {k: f"{v:08x}" for k,v in local_bits.items()}, "comparisons": comparisons,
            "conditional_gradient_for_historical_outlier_if_only_sgd_input_changed": {
                k: inverse_negative_sgd(bits["weight_before"], SELECTION["rate_f32_bits"], outlier,
                                        bits["token_parameter_gradient"], split=split)
                for k,split in (("single_rounding", False), ("rounded_f32_product", True))},
            "historical_intermediate_values": "not recorded", "cause": "not established", "release_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("controller", "original_zip", "local_trio"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.controller, args.original_zip, args.local_trio), indent=2))
