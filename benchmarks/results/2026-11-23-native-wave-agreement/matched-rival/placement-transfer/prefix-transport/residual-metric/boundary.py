"""Descriptive only: how much of each initially wrong held-out gap did each saved head fit close?"""
import argparse
import csv
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

v=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("study.py"))))


def fraction(before,after):
    if before>=0: return None
    return (after-before)/(-before)


def read(root,checked,output):
    exact=v.k.read(checked)
    sp=v.k.read(root/"spec.json")
    manifest=(root/"complete.json").read_bytes()
    v.need(v.d.digest(manifest)==exact["manifest"] and v.d.digest((root/"spec.json").read_bytes())==exact["spec"],
           "wrong raw journal")
    ids=json.loads(manifest)["files"]
    def logits(name):
        f="logits/"+name+".json"
        return json.loads(v.sig.pinned(root/"attempt"/f,ids[f]))["rows"]
    with output.open("x",encoding="ascii",newline="") as out:
        writer=csv.writer(out,delimiter="\t",lineterminator="\n")
        writer.writerow(("base","order","rotation","rule","key","conflict","initial_margin","final_margin","closed_fraction"))
        for b in sp["bases"]:
            for t in v.TURNS:
                orig=logits(v.s.key(b,t,"gate"))
                for rule in v.RULES:
                    for order in sp["orders"]:
                        z=logits(v.s.key(b,t,rule,"head",order))
                        for r in sp["test"]:
                            key=r["key"]
                            a,b1=[(1-2*r["head"])*(rows[key][7]-rows[key][8]) for rows in (orig,z)]
                            f=fraction(a,b1)
                            if f is not None:
                                v.need(math.isfinite(f) and (b1>=0)==(f>=1.),"gap ratio")
                                writer.writerow((b,order,t,rule,key,int(r["head"]!=r["attractor"]),a,b1,f))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("root","checked","output"): cli.add_argument(key,type=Path)
    read(**vars(cli.parse_args()))
