"""One CPU training attempt per split; never overwrite a predecessor or an output."""
import argparse
import json
import math
from itertools import product
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
old = SimpleNamespace(**runpy.run_path(str(HERE.parent / "owner-control/run.py")))
sig = SimpleNamespace(**runpy.run_path(str(HERE.parent / "prefix-signal/study.py")))
p, d, k, s, u, need = old.p, old.d, old.k, old.s, old.u, old.need
RULES = ("uniform", "diagonal", "cyclic")
TURNS = ("identity", "quarter_turn")
TARGET = .25 * (2/256) / math.sqrt(3)
SIGNALS = {"formal": "40fa48fcc32477e1306fc27c16d16cad95c474359915746eac7f426881badcfe",
           "rehearsal": "71292b65608b6baaaeb6eb0dfb7d160fef0fa51792150319fda660e292d57578"}


def plan(kind, parent, signal, ext):
    pin = sig.pinned
    sp = json.loads(pin(parent/"spec.json", p.PARENT[kind][0]))
    manifest = json.loads(pin(parent/"complete.json", p.PARENT[kind][1]))
    pin(signal/"attempt/rows.jsonl", SIGNALS[kind])
    for b in sp["bases"]:
        pin(parent/("base-%d.safetensors"%b), sp["weights"][str(b)])
        ref = "frozen-%d.json"%b
        pin(parent/"attempt"/ref, manifest["files"][ref])
    for name in ("hosts.json", "steps.jsonl"):
        pin(parent/"attempt"/name, manifest["files"][name])
    need(len(ext)==64 and all(c in "0123456789abcdef" for c in ext), "extension digest")
    repo = HERE.parents[6]
    sources = [HERE/n for n in ("PROTOCOL.md","study.py","verify.py","campaign.py")]
    sources += [HERE.parent / n for n in ("owner-control/design.py","owner-control/run.py","prefix-signal/study.py")]
    sources += [repo / n for n in (
        "crates/st-nn/src/geometry_runtime/prefix_transport.rs","crates/st-nn/src/causal_lm/decoder.rs",
        "crates/st-nn/src/causal_lm/geometry_cohort.rs","crates/st-nn/src/causal_lm/geometry_cohort/prefix_metric.rs",
        "bindings/st-py/src/causal_lm/prefix_metric.rs","bindings/st-py/src/causal_lm.rs")]
    return dict(schema="spiraltorch.residual_metric_study.v1", kind=kind, parent_spec=p.PARENT[kind][0],
                parent_manifest=p.PARENT[kind][1], train_signals=SIGNALS[kind], extension=ext,
                sources={f.relative_to(repo).as_posix():d.digest(f.read_bytes()) for f in sources},
                protocol=d.digest((HERE/"PROTOCOL.md").read_bytes()), rules=list(RULES), turns=list(TURNS),
                target_rms=TARGET, max_parameter_l2=16, **{x:sp[x] for x in ("bases","orders","epochs","train","test","weights")})


def traces(signal, sp):
    bank = {}
    for line in sig.pinned(signal/"attempt/rows.jsonl", sp["train_signals"]).splitlines():
        row = json.loads(line)
        if row["turn"] not in TURNS:
            continue
        key = (row["base"],row["turn"],row["key"])
        need(key not in bank, "duplicate calibration reference")
        bank[key] = row["trace"]
    need(set(bank) == set(product(sp["bases"],TURNS,[r["key"] for r in sp["train"]])), "calibration/reference split drift")
    return bank


def moments(examples):
    # Separate oracle: replay each archived f32 site input, without the native metric or labels.
    xs = []
    for rec in examples:
        n,w = rec["shape"]
        v = [0.]*w
        for t in range(n):
            if rec["rotation"] == "quarter_turn":
                v[0],v[1] = -v[1],v[0]
            v = [a+b for a,b in zip(v,rec["site_input"][t*w:t*w+w])]
        xs.append([rec["strength"]*d.f32(x/n) for x in v])
    return [math.fsum(x*x for x in col)/len(xs) for col in zip(*xs)]


def native_rows(st, model, examples):
    runtime = st.NativeGeometryRuntime(model.geometry_artifact())
    return [runtime.forward_sequence(s.SITE, rec["site_input"], (1,rec["shape"][0])).output[-3:]
            for rec in examples]


def rms_rows(before, after):
    return math.sqrt(math.fsum((y-x)**2 for a,b in zip(before,after) for x,y in zip(a,b))/(3*len(before)))


def run(root, parent, signal, pin):
    raw = sig.pinned(root/"spec.json", pin)
    sp = json.loads(raw)
    need(raw == d.encoded(plan(sp["kind"],parent,signal,sp["extension"])), "plan drift")
    st = sig.native(sp["extension"])
    out = root/"attempt"
    out.mkdir(mode=0o700)
    for part in ("weights","logits"):
        (out/part).mkdir()
    rows = sp["train"] + sp["test"]
    hosts = k.read(parent/"attempt/hosts.json")
    anchors = p.anchor_rows(parent)
    bank = traces(signal,sp)
    with (out/"steps.jsonl").open("xb") as journal, (out/"physical.jsonl").open("xb") as physical:
        for b in sp["bases"]:
            host = hosts[str(b)]
            data = sig.pinned(parent/("base-%d.safetensors"%b),sp["weights"][str(b)])
            base = st.NativeCausalLm.from_gpt2(s.fixtures_cached()[0]["config"],st.SafeTensorArchive.from_bytes(data)).with_readout_accumulator("f32")
            need(base.base_model_identity == host["original_base"], "base mismatch")
            frozen = base.with_lora_json(host["lora_weights"]).freeze_lora()
            ref = k.read(parent/"attempt"/("frozen-%d.json"%b))
            for turn in TURNS:
                start = s.key(b,turn,"gate")
                zero = sig.perturb(frozen,host,turn,[0.]*3,st)
                orig, v0, z = old.snapshot(out,start,zero,host,turn,"gate",rows)
                old.same_logits(z,ref)
                examples = [bank[b,turn,r["key"]] for r in sp["train"]]
                need(all(r["geometry_artifact_id"]==orig for r in examples),"trace owner")
                metric = zero.calibrate_prefix_residual_metric([r["prompt"] for r in sp["train"]])
                report = metric.report()
                d.save(out/("metric-"+start+".json"),report)
                q = moments(examples)
                need(report["samples"]==len(examples) and report["anchor"]==orig and
                     all(math.isclose(x,y,rel_tol=1e-14,abs_tol=0.) for x,y in zip(report["second_moments"],q)) and
                     report["base_model_identity"]==host["base"] and report["frozen_lora_artifact_id"]==host["frozen_lora"],
                     "calibration moments/owner")
                for rule,teacher,order in product(RULES,d.TEACHERS,sp["orders"]):
                    m,ident,vec = zero,orig,v0
                    name = s.key(b,turn,rule,teacher,order)
                    pairs = k.schedule(sp["train"],order,"pair")*sp["epochs"]
                    for i,pair in enumerate(pairs):
                        handles, members = zip(*(u.old.handle(m,r,r[teacher],1.) for r in pair))
                        rec = dict(cell=name,i=i,keys=[r["key"] for r in pair],members=members)
                        p.check_members(rec,pair,teacher,z,anchors[s.key(b,turn,teacher,order)])
                        next_model,step = m.step_prefix_residual_rms(metric,handles,rule,target_rms=TARGET,max_parameter_l2=16.)
                        after,v,_ = old.current(next_model,host,turn,"gate")
                        rec.update(id=after,values=v,report=step)
                        journal.write(d.encoded(rec)); journal.flush()
                        validate_step(rec,vec,ident,rule,q,report["id"])
                        if i in (0,len(pairs)-1):
                            x,y = native_rows(st,m,examples),native_rows(st,next_model,examples)
                            physical.write(d.encoded(dict(cell=name,i=i,before=x,after=y))); physical.flush()
                            need(abs(rms_rows(x,y)-step["realized_rms"])<=2e-6, "actual native residual motion")
                        m,ident,vec = next_model,after,v
                    end,_,_ = old.snapshot(out,name,m,host,turn,"gate",rows)
                    need(end==ident and zero.geometry_artifact_id==orig, "endpoint/origin alias")
    d.save(root/"complete.json",dict(spec=pin,files=k.identities(out)))


def validate_step(rec, v, ident, rule, q, metric_id):
    r = rec["report"]
    need(r["schema"]=="spiraltorch.prefix_residual_step.v1" and r["pre_geometry_id"]==ident and
         r["post_geometry_id"]==rec["id"]!=ident and r["metric_id"]==metric_id and
         r["rule"]==rule and r["members"]==r["total_target_tokens"]==len(rec["members"])==2 and
         r["target_rms"]==TARGET, "unbound or incomplete update")
    gs = [p.section(m["v"],"gate") for m in rec["members"]]
    g = k.avg(gs)
    delta = k.sub(p.coords(rec["values"],"gate"),p.coords(v,"gate"))
    damping = math.fsum(q)/(16*len(q))
    direction = [(x if rule=="uniform" else x/(q[(j+(rule=="cyclic"))%3]+damping)) if q[j] else 0.
                 for j,x in enumerate(g)]
    norm = lambda dx: math.sqrt(math.fsum(v*x*x for v,x in zip(q,dx))/len(q))
    expected = [-x*TARGET/norm(direction) for x in direction]
    pred = [k.dot(g,delta) for g in gs]
    need(k.norm(k.sub(delta,expected))<=1e-4*k.norm(expected) and k.norm(delta)<=16 and
         math.isclose(r["requested_parameter_l2"],k.norm(expected),rel_tol=1e-12) and
         math.isclose(r["realized_parameter_l2"],k.norm(delta),rel_tol=1e-12) and
         math.isclose(r["realized_rms"],norm(delta),rel_tol=2e-12) and abs(norm(delta)/TARGET-1)<=1e-4 and
         math.fsum(pred)<0 and len(r["member_loss_dots"])==2 and
         all(math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-14) for x,y in zip(pred,r["member_loss_dots"])) and
         math.isclose(r["mean_input_loss"],math.fsum(m["loss"] for m in rec["members"])/2,rel_tol=1e-7),
         "wrong residual length, direction or descent")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("prepare","run"))
    cli.add_argument("kind",choices=list(SIGNALS))
    for key in ("root","parent","signal"):
        cli.add_argument(key,type=Path)
    cli.add_argument("pin")
    a=cli.parse_args()
    if a.mode=="prepare":
        a.root.mkdir(mode=0o700)
        print(d.save(a.root/"spec.json",plan(a.kind,a.parent,a.signal,a.pin)))
    else:
        need(k.read(a.root/"spec.json")["kind"]==a.kind,"wrong worker kind")
        run(a.root,a.parent,a.signal,a.pin)
