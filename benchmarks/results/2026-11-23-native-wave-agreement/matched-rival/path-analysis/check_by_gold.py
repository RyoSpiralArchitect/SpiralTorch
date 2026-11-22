"""Independent identity: all opposite-subject terms equal 2*dot(sum_is,sum_are)."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path


def check(path, report_path):
    report = json.loads(report_path.read_bytes())
    archive = path/report["kind"]
    plan = (archive/"plan.json").read_bytes()
    if hashlib.sha256(plan).hexdigest()!=report["plan"]:
        raise ValueError("plan hash mismatch")
    width = json.loads(plan)["shape"][1]
    root = archive/"attempt"
    def read_state(ident):
        raw = (root/"states"/(ident.split(":")[1]+".json")).read_bytes()
        if "sha256:"+hashlib.sha256(raw).hexdigest()!=ident:
            raise ValueError("state hash mismatch")
        params = {t["name"]:t["values"] for t in json.loads(raw)["placements"][0]["tensors"]}
        return array("f",params["gate"]+params["bias"])
    maximum = 0.
    for route,r in report["routes"].items():
        acc, lengths = {}, []
        for line in (root/(route+".jsonl")).read_bytes().splitlines():
            record = json.loads(line)
            v = record["step"]
            before,after = read_state(v["origin_id"]), read_state(v["result_id"])
            delta = [b-a for a,b in zip(before, after)]
            lengths.append(math.sqrt(math.fsum(x*x for x in delta)))
            group = acc.setdefault(record["gold"],[0.]*(2*width))
            for j,x in enumerate(delta):
                group[j]+=x
        if len(acc)!=2 or len(lengths)!=r["n"]:
            raise ValueError("expected both gold verbs and the complete path")
        left,right = acc.values()
        for name,start in (("gate",0),("bias",width)):
            direct=2*math.fsum(left[i]*right[i] for i in range(start,start+width))/math.fsum(lengths)**2
            binned = r["gram"]["flip_flip"][name]+r["gram"]["flip_same"][name]
            maximum=max(maximum, abs(direct-binned))
            if abs(direct-binned)>1e-12:
                raise ValueError("gold-token grouping disagrees with subject-flip bins")
    print("routes=%d max_abs_block_error=%.17g"%(len(report["routes"]), maximum))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("report",type=Path)
    args=p.parse_args()
    check(args.archive,args.report)
