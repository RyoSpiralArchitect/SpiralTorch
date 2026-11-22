"""Frozen factorial design and no-framework measurements."""
from array import array
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
OLD = runpy.run_path(str(HERE.parent/"probe.py"))
P, N = OLD["P"], OLD["N"]
need, digest, encoded = P["need"], P["digest"], P["encoded"]
ARMS = ("wave_full", "wave_rival", "affine_full", "affine_rival")
DISTANCE = 1 / 256
TRAIN = (("teacher","teachers","pupil","pupils"), ("doctor","doctors","nurse","nurses"),
         ("singer","singers","fan","fans"), ("actor","actors","critic","critics"),
         ("driver","drivers","officer","officers"), ("farmer","farmers","worker","workers"))
HELD = (("pilot","pilots","soldier","soldiers"), ("lawyer","lawyers","judge","judges"),
        ("artist","artists","writer","writers"), ("player","players","coach","coaches"))


def sources():
    return {name:P["file_hash"](HERE/name) for name in
            ("design.py","run.py","../probe.py","../protocol.py","../native.py")}


def lexemes(groups):
    return {w for g in groups for w in g}


def cases(groups, ids):
    result = []
    for sg,pl,near_sg,near_pl in groups:
        for h,subject in enumerate((sg,pl)):
            for n,near in enumerate((near_sg,near_pl)):
                words = ("The",subject,"that","the",near,"met")
                result.append(dict(key="%s.%d%d" % (sg,h,n), pair=sg, plural=bool(h), conflict=h!=n,
                                   text=" ".join(words), prefix=[ids[w] for w in words],
                                   gold=ids[("is","are")[h]], rival=ids[("are","is")[h]]))
    return result


def plan(kind, model):
    if kind == "gpt2":
        need(model is not None, "local GPT-2 weights required")
        for name,pin in P["PINS"].items():
            need(P["file_hash"](model/name) == pin, "wrong input: "+name)
        train,held,seeds,shape = TRAIN,HELD,(17,31,73),[12,768,12,50257]
        need(not lexemes(train+held) & lexemes(P["TRAIN"]+P["TEST"]), "noun reused from the completed pilot")
        words = sorted(lexemes(train+held) | {"The","that","the","met","is","are"})
        vocab = json.loads((model/"vocab.json").read_bytes())
        ranks = {tuple(line.split()): i for i,line in enumerate((model/"merges.txt").read_text().splitlines()[1:])}
        ids = {w:P["single_word_id"](w if w=="The" else chr(288)+w,vocab,ranks) for w in words}
    else:
        need(kind == "rehearsal" and model is None, "wrong model scope")
        train,held,seeds,shape = (("aa","aas","bb","bbs"),),(("cc","ccs","dd","dds"),),(11,),[2,16,2,32]
        ids = {w:i for i,w in enumerate(sorted(lexemes(train+held) | {"The","that","the","met","is","are"}),1)}
    need(not lexemes(train) & lexemes(held) and len(set(ids.values())) == len(ids), "lexical split or token collision")
    tr,ev = cases(train,ids),cases(held,ids)
    order = [sorted(range(len(tr)), key=lambda i: digest(("relative/%s/%s"%(seed,tr[i]["key"])).encode()))
             for seed in seeds]
    return dict(schema="spiraltorch.relative_agreement_matched_rival.v1", kind=kind, shape=shape, strength=.25,
                train=tr, evaluation=ev, order=order, seeds=list(seeds), arms=list(ARMS),
                target_l2=DISTANCE, tolerance=1e-4, updates=len(tr)*len(seeds)*len(ARMS),
                scored_rows=len(ev)*(2+len(ARMS)*len(seeds)), sources=sources(),
                model_pins=P["PINS"] if kind=="gpt2" else None,
                extension_pin=P["EXTENSION"] if kind=="gpt2" else None, seconds=600 if kind=="gpt2" else 60)


def f32(xs):
    a = array("f",xs)
    need(sys.byteorder=="little" and a.itemsize==4, "little-endian binary32 required")
    return a


def length(xs):
    return math.sqrt(math.fsum(x*x for x in xs))


def delta(y,x):
    need(len(x)==len(y),"parameter layout mismatch")
    return [b-a for a,b in zip(x,y)]


def binary_loss(margin):
    return math.log1p(math.exp(-abs(margin))) + max(0.,-margin)


def check_objective(kind, loss, report):
    need(math.isfinite(loss) and loss>=0., "invalid supervised loss")
    if kind=="full":
        need(report is None, "unexpected declared rival on full CE")
    else:
        need(kind=="rival" and (report["mix"],report["target_tokens"],report["contrasted_tokens"],
             report["rival_ids"])==(1.,1,1,1), "not one explicitly supervised binary contrast")
        m = report["mean_target_minus_declared_rival"]
        need(math.isfinite(m) and math.isfinite(report["full_ce"]) and report["full_ce"]>=0 and
             math.isclose(binary_loss(m),report["declared_ce"],rel_tol=2e-7,abs_tol=2e-12) and
             math.isclose(report["declared_ce"],loss,rel_tol=2e-7,abs_tol=2e-12), "wrong binary loss")


def check_step(before, after, grad, trace, spec):
    target = spec["target_l2"]
    dx = delta(after,before)
    need(len(grad)==len(dx)==trace["trainable_values"]==2*spec["shape"][1] and
         trace["method"]=="gradient_ray_f32.v1" and trace["requested_l2"]==target, "wrong step contract")
    gnorm, step = length(grad),length(dx)
    need(math.isfinite(gnorm) and gnorm>0., "zero or nonfinite direction")
    dot = math.fsum(g*x for g,x in zip(grad,dx))
    err = length(x/target + g/gnorm for x,g in zip(dx,grad))
    for key,value in (("gradient_l2",gnorm),("realized_l2",step),("gradient_dot_step",dot),("relative_vector_error",err)):
        need(math.isclose(trace[key],value,rel_tol=1e-8,abs_tol=5e-15), "native receipt disagrees: "+key)
    need(dot<0 and abs(step/target-1)<=spec["tolerance"] and err<=spec["tolerance"], "unmatched step")
    return step,err


def contrast(spec, scored):
    rows = spec["evaluation"]
    keys = {r["key"] for r in rows}
    need(all(set(scores)==keys for scores in scored.values()), "wrong evaluation keys")
    def means(a,b,group):
        subset = [r["key"] for r in rows if group=="all" or r["conflict"]==(group=="conflict")]
        x,y = scored[a],scored[b]
        def avg(f):
            return math.fsum(f(x[k])-f(y[k]) for k in subset)/len(subset)
        return dict(n=len(subset),correct=sum(x[k]["margin"]>0 for k in subset),
                    ties=sum(x[k]["margin"]==0 for k in subset), margin=avg(lambda r:r["margin"]),
                    nll=avg(lambda r:r["nll"]), conditional_logp=avg(lambda r:-binary_loss(r["margin"])),
                    pair_logp=avg(lambda r:-r["nll"]+binary_loss(r["margin"])))
    all_orders=[]
    for seed in spec["seeds"]:
        effects={}
        for arm in ARMS:
            kind = arm.split("_")[0]
            effects[arm]={g:means("%s-%s"%(seed,arm),"initial-"+kind,g) for g in ("all","conflict","congruent")}
        d = {k:effects[k+"_rival"]["all"]["margin"]-effects[k+"_full"]["all"]["margin"] for k in ("wave","affine")}
        all_orders.append(dict(seed=seed,arms=effects,rival_minus_full=d,interaction=d["wave"]-d["affine"]))
    gates = {k:all(o["rival_minus_full"][k]>0 and o["arms"][k+"_rival"]["all"]["margin"]>0
                   for o in all_orders) for k in ("wave","affine")}
    return dict(orders=all_orders, objective_advantage=gates,
                wave_specific=gates["wave"] and all(o["interaction"]>0 for o in all_orders))
