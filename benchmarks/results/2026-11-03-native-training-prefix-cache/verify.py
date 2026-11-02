"""Recompute paired timings and resource/accounting checks without training or model weights."""
import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok,why):
    if not ok: raise ValueError(why)


def verify(root):
    proto=json.loads((root/"protocol.json").read_bytes())
    result=json.loads((root/"result.json").read_bytes())
    runs=[json.loads(line) for line in (root/"runs.jsonl").read_bytes().splitlines()]
    check(result["protocol_sha256"]==sha(root/"protocol.json") and result["runs_sha256"]==sha(root/"runs.jsonl"),"receipt pins")
    paths={"script_sha256":"tools/bench_native_training_prefix_cache.py","fixture_sha256":"bindings/st-py/tests/test_native_causal_lm.py",
           "arm_constructor_sha256":"benchmarks/results/2026-11-03-native-factorial-answer-training/design.py"}
    check(all(proto[k]==sha(ROOT/f) for k,f in paths.items()) and proto["installed"]["all_original_files_match"],"different source/build")
    check(proto["shape"]==dict(width=64,heads=4,layers=6,positions=32,vocabulary=97) and
          proto["arms"]==["affine","saturation","projection","both"] and proto["layer"]==5 and
          (proto["documents"],proto["context"],proto["batch_windows"],proto["epochs"])==(12,25,2,4),"different schedule")
    schedule=[["plain","cached"]]+[["plain","cached"] if i%2==0 else ["cached","plain"] for i in range(6)]
    check([proto["warmup"],*proto["order"]]==schedule and len(runs)==14,"unexpected timing sequence")
    ratios,fill,repeats=[],[],[]
    for i,methods in enumerate(schedule):
        pair={}
        for j,method in enumerate(methods):
            r=runs[2*i+j]
            check(r["pair"]==i and r["method"]==method and len(r["epoch_ns"])==4 and all(t>0 for t in r["epoch_ns"])
                  and sum(r["epoch_ns"])==r["total_ns"] and r["trace_sha256"]==result["all_trace_sha256"]
                  and r["checkpoint_ids"]==result["all_checkpoint_ids"] and len(r["checkpoint_ids"])==4,"trace, duration or checkpoint")
            if method=="cached":
                check(r["cache"]==dict(max_values=18432,max_entries=6,entries=6,retained_values=18432,hits=18,insertions=6,
                      capacity_bypasses=0,no_frozen_block=0,skipped_block_calls=90,last_action="hit"),"unexpected cache accounting")
            else:
                check(r["cache"] is None,"plain baseline had a cache")
            pair[method]=r
        if i:
            a,b=pair["plain"],pair["cached"]
            ratios.append(a["total_ns"]/b["total_ns"])
            fill.append(a["epoch_ns"][0]/b["epoch_ns"][0])
            repeats.append(sum(a["epoch_ns"][1:])/sum(b["epoch_ns"][1:]))
    for values,name,median in [(ratios,"paired_ratios","median_total_ratio"),(fill,"fill_ratios","median_fill_ratio"),
                               (repeats,"repeated_ratios","median_repeated_ratio")]:
        check(result[name]==values and result[median]==statistics.median(values),"paired speed ratio or summary")
    return dict(timed_pairs=6,warmup_pairs=1,equal_trace_and_checkpoint_runs=14,
                median_total_ratio=statistics.median(ratios),median_fill_ratio=statistics.median(fill),
                median_repeated_ratio=statistics.median(repeats),model_executions_by_verifier=0)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    ap.add_argument("--export",type=Path,help="copy the three verified receipts into a new directory")
    args=ap.parse_args()
    report=verify(args.directory)
    if args.export is not None:
        args.export.mkdir(exist_ok=False)
        for name in ("protocol.json","runs.jsonl","result.json"):
            shutil.copyfile(args.directory/name,args.export/name)
        verify(args.export)
    print(json.dumps(report,sort_keys=True))
