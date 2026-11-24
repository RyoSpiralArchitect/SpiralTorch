"""Evidence-only readback; keep the first failed verifier, every gate and the untouched model journal."""
import argparse
import csv
import json
import math
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import runpy
import struct
from types import FunctionType, SimpleNamespace

HERE=Path(__file__).resolve().parent
v1=SimpleNamespace(**runpy.run_path(str(HERE/"verify.py")))
need,d=v1.need,v1.d
V1="622cad81df95df4d8d3e0e0956ae008e90e718a3276183d36a7ed73e60bc0102"


def nearest32(x):
    """Round an exact, finite, in-range rational to binary32, ties to even."""
    if not x: return 0.
    if x<0: return -nearest32(-x)
    initial,=struct.unpack("<I",struct.pack("<f",float(x)))
    def candidate(bits):
        v,=struct.unpack("<f",struct.pack("<I",bits))
        return abs(Fraction(v)-x),bits%2,v
    choices=[candidate(i) for i in range(max(initial-1,0),min(initial+1,0x7f7fffff)+1)]
    return min(choices)[2]


@lru_cache(maxsize=1024)
def means(raw,n,turn):
    need(len(raw)==3*n and n>0 and turn in v1.v.TURNS,"invalid raw calibration row")
    sums=[Fraction() for _ in range(3)]
    for i in range(n):
        x,y,z=map(Fraction,raw[i*3:i*3+3])
        if turn=="quarter_turn":
            for _ in range((n-1-i)%4): x,y=-y,x
        for j,item in enumerate((x,y,z)):
            sums[j]+=item
    return tuple(nearest32(value/n) for value in sums)


def delivered(r,g):
    m=means(tuple(r["site_input"]),r["shape"][0],r["rotation"])
    return [d.f32(x+d.f32(r["strength"]*d.f32(y*z))) for x,y,z in zip(r["site_input"][-3:],m,g)]


def audit(root,parent,signal,result,table):
    need(d.digest((HERE/"verify.py").read_bytes())==V1, "unexpected version of the original verifier")
    with (root/"attempt/physical.jsonl").open("rb") as f:
        steps=[(r["cell"],r["i"]) for r in map(json.loads,f)]
    differences=[]
    calls=0
    coords=0
    def oracle(examples,gate):
        nonlocal calls,coords
        old=v1.expected_outputs(examples,gate)
        reference=[delivered(r,gate) for r in examples]
        cell,i=steps[calls//2]
        side=("before","after")[calls%2]
        calls+=1
        for n,(a,b) in enumerate(zip(old,reference)):
            for j,(x,y) in enumerate(zip(a,b)):
                coords+=1
                if x!=y:
                    differences.append(dict(cell=cell,i=i,side=side,train_row=n,channel=j,
                                            divide_first=x,divide_last=y))
        return reference
    # Create a separate reader with ONLY the witness oracle replaced. The hash-pinned
    # v1 module, its failure and its stricter-than-RMS exact-equality assertion stay intact.
    old_read=v1.read
    read=FunctionType(old_read.__code__,dict(old_read.__globals__,expected_outputs=oracle),old_read.__name__)
    sp,z,counts,high,metrics,manifest=read(root,parent,signal)
    need(calls==2*len(steps) and coords==3*counts["native_residual_rows"],"not every native row rechecked")
    rows,gates=v1.analyze(sp,z)
    with table.open("x",newline="",encoding="ascii") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return d.save(result,dict(schema="spiraltorch.residual_metric_exact_readback.v1",
                              spec=d.digest((root/"spec.json").read_bytes()),manifest=manifest,
                              pinned_reader=V1,reader=d.digest(Path(__file__).read_bytes()),
                              coordinates=coords,changed_oracle_coordinates=differences,
                              counts=counts,worst=high,moments=metrics,by_base=gates,rows=rows))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","parent","signal","result","table"): cli.add_argument(name,type=Path)
    print(audit(**vars(cli.parse_args())))
