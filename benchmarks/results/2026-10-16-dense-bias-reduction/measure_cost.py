"""Bounded paired process-cost diagnostic, not kernel or training throughput."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    binaries = {"before": args.raw / "probe-before", "exact": args.raw / "probe-exact-matched"}
    reports = {"before": args.raw / "native-before.json", "exact": args.raw / "native-exact-matched.json"}
    inputs = [args.study / "request.json", args.study / "checkpoint.json"]
    observed_inputs = [sha(path.read_bytes()) for path in inputs]
    if observed_inputs != ["b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c",
                           "9e8ae028c81705e05935e70ba8386e9db602ebfa4603d78ce874670f7b6c491c"]:
        raise ValueError("pinned inputs changed")
    hashes = {name: sha(path.read_bytes()) for name, path in binaries.items()}
    expected = {name: sha(path.read_bytes()) for name, path in reports.items()}
    # Require matched target features and profile, not just similarly named executables.
    builds = {}
    for name, filename in [("before", "build-before.jsonl"), ("exact", "build-exact-matched.jsonl")]:
        records = [json.loads(line) for line in (args.raw / filename).read_text().splitlines()]
        artifact, = [v for v in records if v.get("target", {}).get("name") == "resident_byte_gradient_probe"]
        builds[name] = {"features": artifact["features"], "profile": artifact["profile"]}
    if builds["before"] != builds["exact"]:
        raise ValueError("mismatched build controls")
    output = args.raw / "timing.json"
    # Reserve before execution; a partial/failed run remains visible, never overwritten.
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        rows = []
        orders = [("warmup", ["before", "exact"]),
                  ("measured", ["before", "exact", "exact", "before", "before", "exact", "exact", "before"])]
        for phase, order in orders:
            for name in order:
                started = time.perf_counter()
                result = subprocess.run([str(binaries[name]), *(str(v) for v in inputs)],
                                        capture_output=True, timeout=120, check=False)
                elapsed = time.perf_counter() - started
                matched = result.returncode == 0 and sha(result.stdout) == expected[name]
                rows.append({"phase": phase, "variant": name, "seconds": elapsed,
                             "exit_code": result.returncode, "report_matches": matched,
                             "stdout_sha256": sha(result.stdout), "stderr_sha256": sha(result.stderr)})
                if not matched:
                    json.dump({"passed": False, "rows": rows}, stream, indent=2)
                    raise ValueError("timed process changed expected report")
        unchanged = hashes == {name: sha(path.read_bytes()) for name, path in binaries.items()}
        unchanged &= observed_inputs == [sha(path.read_bytes()) for path in inputs]
        medians = {name: statistics.median(r["seconds"] for r in rows if r["phase"] == "measured" and r["variant"] == name)
                   for name in binaries}
        json.dump({"schema": "spiraltorch.dense_bias_reduction.process_cost.v1", "passed": unchanged,
                   "scope": "3 restored models x 4 backward-only batches per process; startup, pipeline compilation, JSON and readbacks included; not kernel throughput",
                   "binary_sha256": hashes, "build_controls": builds,
                   "request_checkpoint_sha256": observed_inputs, "rows": rows,
                   "median_seconds": medians, "median_ratio_exact_over_before": medians["exact"] / medians["before"],
                   "limitations": ["four measured processes per variant on one host", "not isolated from OS noise", "no timestamps or register/spill measurements", "no PyTorch speed comparison"]},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
        if not unchanged:
            raise ValueError("identity drift during timing")


if __name__ == "__main__":
    main()
