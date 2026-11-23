"""Read-only native positive control: a designed activation, not a language model."""
import itertools
import json
import math
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

r = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("run.py"))))
p, need = r.p, r.need


def close(obs, expected):
    need(len(obs) == len(expected) and all(math.isclose(x,y,rel_tol=2e-6,abs_tol=1e-9)
                                        for x,y in zip(obs,expected)), "unexpected forward or VJP")


def check(turn, amp, h, t, output, axes):
    sign = {"identity":0,"quarter_turn":1,"inverse_quarter_turn":-1}[turn]
    summary = [(h if sign else h+t)*amp/6, sign*t*amp/6, 0.]
    close(output,[x/32 for x in summary])
    need(len(axes)==2, "missing final-axis VJP")
    for axis,v in enumerate(axes):
        src = (1/192,(not sign)/192) if axis==0 else (0.,sign/192)
        close(v["number_input_derivatives"],src)
        close(v["gate"],[x/4 if j==axis else 0. for j,x in enumerate(summary)])
        close(v["bias"],[.25 if j==axis else 0. for j in range(3)])
    if sign:
        need(h*output[0]>0 and sign*t*output[1]>0,"fixed role readouts failed")
    elif h != t:
        close(output,[0.]*3)


def run(output_path):
    need(not output_path.exists(), "write-once control")
    st = r.native()
    rows = []
    for turn in p.u.TURNS:
        obj = p.artifact("sha256:"+"0"*64,turn,"gate",[0.,0.,0.,.125,.125,0.])
        runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(obj)))
        need(runtime.execution_backend=="rust_f32_cpu","unexpected execution backend")
        for amp,h,t in itertools.product((1.,.5),(-1,1),(-1,1)):
            x = [0.]*18
            x[3],x[12] = h*amp,t*amp
            batch = runtime.forward_sequence(p.s.SITE,x,[1,6])
            final = batch.output()[-3:]
            axes = []
            for axis in (0,1):
                upstream = [0.]*18
                upstream[15+axis] = 1.
                dx, grad = batch.vjp(upstream)
                axes.append(dict(number_input_derivatives=[dx[3],dx[12]],**grad))
            check(turn,amp,h,t,final,axes)
            rows.append(dict(rotation=turn,amplitude=amp,head=h,attractor=t,final=final,axes=axes))
    return p.d.save(output_path,dict(schema="spiraltorch.fixed_role_activation.v1",extension=p.EXTENSION,
                                     forward_calls=len(rows),local_vjps=2*len(rows),updates=0,
                                     code=p.d.digest(Path(__file__).read_bytes()),
                                     protocol=p.d.digest(Path(__file__).with_name("ROLE_CONTROL.md").read_bytes()),rows=rows))


if __name__=="__main__":
    print(run(Path(sys.argv[1])))
