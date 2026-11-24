"""Render all base/order/teacher/template outcomes from the pinned result; no model calls."""
import argparse
from pathlib import Path
import csv
import hashlib
import json
import sys

def table(folder,pin,stream):
    raw=(folder/"result.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError("foreign result")
    doc=json.loads(raw)
    cols=("base","order","teacher","template","mode")
    out=csv.writer(stream,delimiter="\t",lineterminator="\n")
    out.writerow(cols+("pairs","initial_both","both","initial_correct","correct","initial_mean_min","mean_min",
                        "capacity_ratio","common_gain","opposing_train_pairs","train_pairs","over_cap"))
    mech={(c["base"],c["teacher"],c["template"],c["mode"]):c for c in doc["mechanism"]["cells"]}
    seen=set()
    for c in doc["transfer"]["cells"]:
        key=tuple(c[k] for k in cols)
        if key in seen:raise ValueError("duplicate cell")
        seen.add(key);m=mech[(key[0],*key[2:])]
        out.writerow(key+(c["pairs"],c["both_before"],c["both"],c["raw"]["before"],c["raw"]["correct"],
                          c["mean_min_before"],c["mean_min"],m["ratio"],m["gain"],m["opposing"],m["pairs"],m["over_cap"]))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path);p.add_argument("pin");p.add_argument("output",type=Path,nargs="?")
    a=p.parse_args()
    if a.output:
        with a.output.open("x",newline="") as f:table(a.folder,a.pin,f)
    else:table(a.folder,a.pin,sys.stdout)
