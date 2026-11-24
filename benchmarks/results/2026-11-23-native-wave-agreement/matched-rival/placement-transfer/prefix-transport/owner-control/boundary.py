"""Post-hoc observed progress toward an existing wrong conflict decision, not a reachability bound."""
import argparse
import csv
import runpy
from pathlib import Path
from types import SimpleNamespace

v = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("verify.py"))))


def progress(before, after, gold):
    signed = (1-2*gold)*before
    if signed >= 0:
        return None
    # Positive means toward the tie, 1 exactly reaches it, negative moves away.
    return (1-2*gold)*(after-before)/(-signed)


def table(root, previous):
    sp,zs,counts,_ = v.archive(root,previous)
    for b,t,o in v.p.starts(sp):
        origin = zs[v.s.key(b,t,o)]
        for order in sp["orders"]:
            z = zs[v.s.key(b,t,o,"head",order)]
            gains = []
            for row in sp["test"]:
                if row["head"] == row["attractor"]:
                    continue
                key = row["key"]
                start = origin[key][7]-origin[key][8]
                end = z[key][7]-z[key][8]
                g = progress(start,end,row["head"])
                if g is not None:
                    gains.append((g,key,start,end))
            best = max(gains) if gains else (None,)*4
            yield dict(base=b,rotation=t,owner=o,order=order,initially_wrong_conflicts=len(gains),
                       moved_toward=sum(g>0 for g,*_ in gains),reached_tie_or_passed=sum(g>=1 for g,*_ in gains),
                       best_fraction=best[0],best_key=best[1],original_gap=best[2],final_gap=best[3])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","previous","output"):
        cli.add_argument(name,type=Path)
    q=cli.parse_args()
    rows=list(table(q.root,q.previous))
    with q.output.open("x",encoding="ascii",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
