"""Archive this failed qualification without changing its original success gate."""
import argparse
import math
from pathlib import Path
import runpy
import statistics

Q = runpy.run_path(str(Path(__file__).with_name("qualify.py")))


def failure_receipt(cases, teacher, resumed, timing):
    try:
        Q["validate"](cases, teacher, resumed, timing)
    except ValueError as error:
        # Unexpected/incomplete results must not be mislabeled as the observed logit failure.
        if str(error) != "logit gate failed":
            raise
    else:
        raise ValueError("this archiver is only for a failed numerical qualification")
    failures = []
    for group, rows in [("timing", timing["rows"]), ("teacher", [r for c in teacher for r in c["steps"]])]:
        for row in rows:
            value = row["max_logit_abs"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("non-finite/invalid numerical receipt")
            if type(row["argmax_exact"]) is not bool or type(row["within_gate"]) is not bool:
                raise ValueError("invalid result flags")
            passed = value <= Q["GATES"]["logit_atol"] and row["argmax_exact"]
            if passed != row["within_gate"]:
                raise ValueError("metric/decision mismatch")
            if not passed:
                failures.append({"group": group, **row})
    if not failures:
        raise ValueError("missing numerical failure")
    summaries = []
    for length in Q["TIMING_LENGTHS"]:
        for arm in Q["ARMS"]:
            rows = [r for r in timing["rows"] if r["length"] == length and r["arm"] == arm and not r["warmup"]]
            summaries.append({"length": length, "arm": arm, "repetitions": len(rows),
                "all_logit_gates_passed": all(r["within_gate"] for r in rows),
                **{key: statistics.median(r[key] for r in rows) for key in ["prefill_ms", "decode_ms", "total_ms"]}})
    return {"status": "UNQUALIFIED_LOGIT_GATE_FAILED", "failed_rows": failures,
            "medians_not_overall_qualified": summaries, "teacher_forcing": teacher,
            "resume": resumed, "timing": timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model-dir", "data", "checkpoint", "cohort", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.mode = "retain_failure"
    Q["checked"](args)
    cases = {p.stem: Q["load"](p) for p in (args.cohort / "native").glob("*.json")}
    teacher, resumed, timing = [Q["load"](args.cohort / name) for name in ["teacher.json", "resume.json", "timing.json"]]
    result = failure_receipt(cases, teacher, resumed, timing)
    runtime = Q["load"](args.cohort / "native-runtime.json")
    if runtime["extension_sha256"] != timing["extension_sha256"]:
        raise ValueError("extensions differ")
    Q["save"](args.output, {**result, "contract": Q["load"](args.cohort / "contract.json"), "native_runtime": runtime,
        "retention_script_sha256": Q["digest"](Path(__file__)),
        "cases": [{"case": name, "generated_tokens": len(case["generated"]),
                   "final_checkpoint_id": case["pin"], "raw_sha256": Q["digest"](args.cohort / "native" / (name + ".json"))}
                  for name, case in sorted(cases.items())],
        "raw_sha256": {name: Q["digest"](args.cohort / name) for name in ["contract.json", "teacher.json", "resume.json", "timing.json"]},
        "claim_ceiling": "native cache mechanics pass; the complete PyTorch timing comparison fails its sealed raw-logit gate; no quality or general performance claim"})


if __name__ == "__main__":
    main()
