"""Verify synthetic cross-wheel parity and recompute paired timings without importing a model."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[3]
FILES = ("attempt.json", "protocol.json", "runs.jsonl", "result.json")
METHODS = ("old_report", "new_report", "new_id", "new_chunk")
ORDER = ((0,1,3,2), (1,2,0,3), (2,3,1,0), (3,0,2,1)) * 2
POLICIES = {
    "plain": {"selection": "greedy"},
    "sample32": {"selection": "sample", "top_k": 32, "top_p": .9,
                 "distribution": {"kind": "softmax", "temperature": .85}},
    "zspace32": {"selection": "greedy", "top_k": 32, "top_p": .9,
                 "distribution": {"kind": "zspace", "config": {"curvature": -1.,
                                   "repression_strength": 2., "last_token_repression": .5}}},
}


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    for name in FILES:
        path = root / name
        check(not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= 524_288, "receipt size or type")
    a, p, r = (json.loads((root / name).read_bytes()) for name in (FILES[0], FILES[1], FILES[3]))
    runs = [json.loads(line) for line in (root / FILES[2]).read_bytes().splitlines()]
    check(all(r[key] == sha(root / name) for key, name in zip(
        ("attempt_sha256", "protocol_sha256", "runs_sha256"), FILES)), "receipt digest")
    check(p["schema"] == "spiraltorch.synthetic_exact_decoding_bench.v1" and p["policies"] == POLICIES and
          p["script_sha256"] == sha(ROOT / "tools/bench_native_decoding.py") and
          p["fixture_sha256"] == sha(ROOT / "bindings/st-py/tests/test_native_causal_lm.py"), "wrong design or source")
    check((a["wall_seconds"], a["worker_wall_seconds"], a["worker_cpu_seconds"], a["retries"]) == (240,40,35,0)
          and 0 < r["elapsed_seconds"] <= a["wall_seconds"], "wrong budget")
    check(p["shape"] == dict(width=16,heads=2,layers=2,positions=32,vocabulary=50257) and
          p["max_new_tokens"] == 8 and p["prompts"] == [[12,4,26,81,17,2,63,482],[7,900,4,63,18,33,401,2,9,6,111]]
          and p["warmup"] == "each worker and policy once", "different workloads")
    check(p["thread_env"] == dict.fromkeys(("RAYON_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"), "1"), "different thread counts")
    installed = p["installed"]
    for side in ("old","new"):
        w = installed[side]
        check(w["all_original_files_match"] is True and w["python_bytecode_absent"] is True and
              w["wheel_sha256"] == a[side+"_wheel"] and re.fullmatch("[a-f0-9]{64}", w["native_sha256"])
              and re.fullmatch("[a-f0-9]{64}", w["wheel_sha256"]), "wheel admission")
    check(installed["old"]["native_sha256"] != installed["new"]["native_sha256"], "same binary")
    order = [[METHODS[i] for i in row] for row in ORDER]
    check(r["cells"] == len(runs) == 32 and p["order"] == order and
          [(v["index"],v["method"]) for v in runs] == [(i,m) for i,row in enumerate(order) for m in row],
          "unbalanced or missing worker")
    stats = {}
    for case in POLICIES:
        first = runs[0]["cases"][case]
        times = {m: [] for m in METHODS}
        for cell in runs:
            v, method = cell["cases"][case], cell["method"]
            check(cell["base"] == runs[0]["base"] and set(cell["cases"]) == set(POLICIES), "different model or coverage")
            check(v["histories"] == first["histories"] and len(v["histories"]) == 2 and
                  all(len(h) == 8 and all(type(x) is int and 0 <= x < 50257 for x in h) for h in v["histories"])
                  and v["finish"] == ["length","length"], "token or stopping parity")
            check(v["checkpoints"] == first["checkpoints"] and len(v["checkpoints"]) == 2 and
                  all(re.fullmatch("sha256:[a-f0-9]{64}", x) for x in v["checkpoints"]), "checkpoint parity")
            if method.endswith("report"):
                check(v["reports_sha256"] == first["reports_sha256"] and
                      re.fullmatch("[a-f0-9]{64}", first["reports_sha256"]), "report parity")
            else:
                check(v["reports_sha256"] is None, "unexpected report")
            check(type(v["ns"]) is int and 0 < v["ns"] < 40_000_000_000, "duration")
            times[method].append(v["ns"])
        stats[case] = {"median_ms": {m: median(v)/1e6 for m,v in times.items()}}
        for label, x, y in (("report_old_over_new",0,1), ("id_old_over_new",0,2),
                            ("chunk_old_over_new",0,3), ("chunk_new_report_over_ids",1,3)):
            pairs = [u/v for u,v in zip(times[METHODS[x]], times[METHODS[y]], strict=True)]
            stats[case][label] = {"pairs": pairs, "median": median(pairs)}
    check(stats == r["ratios"], "paired median or intermediate timings")
    return {"workers": 32, "policy_measurements": 96, "model_executions_by_verifier": 0,
            "median_old_report_over_new_chunk": {k: v["chunk_old_over_new"]["median"] for k,v in stats.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--export", type=Path, help="copy only the four verified receipts into a new directory")
    args = parser.parse_args()
    result = verify(args.directory)
    if args.export:
        args.export.mkdir(exist_ok=False)
        for name in FILES:
            shutil.copyfile(args.directory / name, args.export / name)
        verify(args.export)
    print(json.dumps(result, sort_keys=True))
