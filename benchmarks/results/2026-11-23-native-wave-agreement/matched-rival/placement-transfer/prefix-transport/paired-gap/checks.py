"""Numerical checks using saved values, independently of the native objective and optimizer."""
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

x = SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name("design.py"))))
d, v, k, c, need = x.d, x.v, x.k, x.c, x.need


def numeric(a,b):
    return math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-14)


def unit(tr,r,host,origin):
    need((tr["schema"],tr["site"],tr["rotation"],tr["shape"],tr["strength"],tr["readout_accumulator"]) ==
         ("spiraltorch.last_mlp_prefix_margin_signal.v1",x.s.SITE,c.TURN,[len(r["prompt"]),3],.25,"f32"),
         "unexpected unit-margin trace")
    need((tr["base_model_identity"],tr["frozen_lora_artifact_id"],tr["geometry_artifact_id"],tr["lora_artifact_id"]) ==
         (host["base"],host["frozen_lora"],origin,None), "incorrect trace owner")
    v.sig.vectors(tr)
    q = tr["contrast"]
    need((q["target_id"],q["alternative_id"],q["prompt_tokens"]) == (7,8,len(r["prompt"]))
         and q["target_minus_alternative"] == q["target_logit"]-q["alternative_logit"], "invalid margin trace")
    means = c.exact.means(tuple(tr["site_input"]),len(r["prompt"]),c.TURN)
    for gate,bias,u,m in zip(tr["gate_gradient"],tr["bias_gradient"],tr["upstream"],means):
        need(bias == d.f32(.25*u) and math.isclose(gate,m*bias,rel_tol=2e-6,abs_tol=2e-8),
             "unit parameter chain rule")


def metric(report, examples, host, origin):
    q = v.moments(examples)
    need(report["samples"] == len(examples) and report["anchor"] == origin and
         report["base_model_identity"] == host["base"] and report["frozen_lora_artifact_id"] == host["frozen_lora"]
         and len(report["second_moments"]) == 3 and
         all(a > 0 and math.isclose(a,b,rel_tol=1e-14) for a,b in zip(q,report["second_moments"])),
         "wrong calibrated response")
    return q


def center(traces):
    a,b = [tr["contrast"] for tr in traces]
    return math.fsum((a["target_logit"],-a["alternative_logit"],b["target_logit"],-b["alternative_logit"]))/2


def reference(z, mode, c0):
    (a,b),(c,d0) = z
    plus,minus = a-b,c-d0
    rank_gap = math.fsum((a,-b,-c,d0))/2
    mid = math.fsum((a,-b,c,-d0))/2
    sp = lambda t: max(t,0.)+math.log1p(math.exp(-abs(t)))
    def sig(t):
        v0 = math.exp(-abs(t))
        return (1. if t >= 0 else v0)/(1.+v0)
    w = sig(-rank_gap)
    binary, rank = (sp(-plus)+sp(minus))/2, sp(-rank_gap)
    drift = math.fsum((.5*a,-.5*b,.5*c,-.5*d0,-c0)) if mode=="anchored_rank" else None
    penalty = drift**2/2 if drift is not None else 0.
    coeff = ([-sig(-plus)/2,sig(minus)/2] if mode=="binary" else
             [(-w+(drift or 0.))/2,(w+(drift or 0.))/2])
    return dict(gaps=[plus,minus],half_difference=rank_gap,center=mid,center_drift=drift,
                binary_loss=binary,rank_loss=rank,center_penalty=penalty,
                loss=binary if mode=="binary" else rank+penalty,row_derivatives=coeff)


def gradient(r, pair, arm, traces, host, ident, first):
    g, report = r["g"],r["objective"]
    need(g["geometry"]==ident and g["base"]==host["base"] and g["shape"]==[2,9] and g["tokens"]==2
         and r["keys"]==[a["key"] for a in pair], "different pair, count, or owner")
    c0 = center(traces)
    objective = dict(kind=arm)
    if arm=="anchored_rank":
        objective.update(reference_center=c0,stiffness=1.)
    need(report["schema"]=="spiraltorch.paired_token_gap.v1" and report["tokens"]==[7,8]
         and report["objective"]==objective, "incorrect objective or anchor")
    z = report["logits"]
    need(len(z)==2 and all(len(p)==2 and all(math.isfinite(t) and t==d.f32(t) for t in p) for p in z),
         "invalid paired raw logits")
    ref = reference(z,arm,c0)
    for key, val in ref.items():
        observed = report[key]
        if isinstance(val,list):
            good = len(val)==len(observed) and all(numeric(a,b) for a,b in zip(val,observed))
        else:
            good = observed is None if val is None else numeric(val,observed)
        need(good, "independent paired objective: "+key)
    need(g["loss"]==d.f32(ref["loss"]) and set(g["params"])=={"gate","bias"} and
         all(len(a)==3 and all(math.isfinite(t) and t==d.f32(t) for t in a) for a in g["params"].values()),
         "wrong sealed loss or gradient")
    if first:
        for row,tr in zip(z,traces):
            need(row==[tr["contrast"][key] for key in ("target_logit","alternative_logit")],
                 "first two-lane readout does not match origins")
        for block,vector in g["params"].items():
            for j,actual in enumerate(vector):
                target = math.fsum(coef*tr[block+"_gradient"][j] for coef,tr in zip(ref["row_derivatives"],traces))
                need(abs(actual-target)<=2e-7+3e-4*abs(target), "packed gradient vs two unit-margin seeds")


def step(r, before, ident, q, metric_id):
    st, g = r["step"],r["g"]
    need(st["schema"]=="spiraltorch.prefix_residual_step.v1" and st["pre_geometry_id"]==ident and
         st["post_geometry_id"]==r["id"]!=ident and st["metric_id"]==metric_id and
         (st["rule"],st["members"],st["total_target_tokens"],st["target_rms"])==("diagonal",1,2,v.TARGET),
         "invalid geometry step identity")
    after = r["values"]
    need(len(before)==len(after)==6 and before[:3]==after[:3]==[0.]*3 and
         all(math.isfinite(z) and d.f32(z)==z for z in after), "fixed bias or gate layout changed")
    delta = [y-a for a,y in zip(before[3:],after[3:])]
    damp = math.fsum(q)/48
    grad = g["params"]["gate"]
    direction = [a/(z+damp) for a,z in zip(grad,q)]
    rms = lambda xs: math.sqrt(math.fsum(z*a*a for z,a in zip(q,xs))/3)
    denom = rms(direction)
    need(denom>0,"no descent direction")
    expect = [-a*v.TARGET/denom for a in direction]
    dot = math.fsum(a*b for a,b in zip(grad,delta))
    need(k.norm(k.sub(delta,expect))<=1e-4*k.norm(expect) and k.norm(delta)<=16 and
         numeric(k.norm(delta),st["realized_parameter_l2"]) and numeric(k.norm(expect),st["requested_parameter_l2"]) and
         numeric(rms(delta),st["realized_rms"]) and abs(rms(delta)/v.TARGET-1)<=1e-4 and dot<0 and
         len(st["member_loss_dots"])==1 and numeric(st["member_loss_dots"][0],dot) and
         numeric(st["mean_input_loss"],g["loss"]), "stored step length, direction, loss or descent")


def witness(w,before,after,traces,receipt):
    a,b = w["before"],w["after"]
    need(len(a)==len(b)==len(traces) and c.same_rows(a,c.outputs(traces,before[3:],"identity")) and
         c.same_rows(b,c.outputs(traces,after[3:],"identity")) and
         abs(v.rms_rows(a,b)-receipt["realized_rms"])<=2e-6,"native first/last residual movement")
