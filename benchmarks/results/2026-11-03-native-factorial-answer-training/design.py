"""Frozen split, trainable owners, and decision rule for the four-arm real-learning comparison."""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"2026-11-03-native-wave-affine-extraction"
if hashlib.sha256((PREV/"protocol.py").read_bytes()).hexdigest() != "11d81ac0145ff488fcd6db4b5003177e251da86554341b870d07fe439efe3548":
    raise ValueError("predecessor generator changed")
sys.path.insert(0,str(PREV))
import protocol as p  # noqa: E402
old, check, pin = p.old, p.old.require, p.pin
from native_answer_generation import factorial_effects, paired_lines  # noqa: E402

ARMS = ("affine","saturation","projection","both")
ORDERS = p.ORDERS
TRAIN, EVAL, EPOCHS = 120, 72, 4
UPDATES = TRAIN * EPOCHS
MAX_NEW, NL, EOS, CACHE = 6, 198, 50256, 2_097_152
BUILD = HERE.parent/"2026-11-03-native-wave-factorial"/"build.json"
check(old.sha(BUILD)=="308224752015471801fdf12a137defd30b30ef57c0c42119f6d9655971e88358","changed operator wheel")
WHEEL, NATIVE = (old.read(BUILD)[k] for k in ("wheel_sha256","native_sha256"))
PAIRS = {"projection_minus_affine":(0,2),"saturation_minus_affine":(0,1),"both_minus_affine":(0,3),
         "both_minus_projection":(2,3),"both_minus_saturation":(1,3)}
PRIMARY = "projection_minus_affine"
MARGIN = math.log1p(.001)


def excluded_keys():
    old_split, spec = p.freeze()
    check(pin(spec)=="7ba48e87df11570ac065e1e5251cfe23e8f91a33af6430e002b0d01c7685a7e2","different older split")
    keys = p.previous_keys() | {r["key"] for rs in old_split.values() for r in rs}
    check(len(keys)==288,"wrong predecessor exclusion")
    return keys


def records():
    excluded = excluded_keys()
    pool = sorted({"-".join(map(str, t)) for t in itertools.product(range(8), repeat=3)} - excluded)
    slots = [(split,f,v,r) for split,reps in (("train",5),("eval",3))
             for f in range(3) for v in range(8) for r in range(reps)]
    edges = {s:sorted((k for k in pool if int(k.split("-")[s[1]])==s[2]),
                      key=lambda k:pin(["field-factor-training-v1",s,k])) for s in slots}
    owner = {}
    # Reroute assignments within this single predeclared matching; never redraw a split.
    def match(s, seen):
        for k in edges[s]:
            if k in seen:
                continue
            seen.add(k)
            if k not in owner or match(owner[k],seen):
                owner[k]=s
                return True
        return False
    check(all(match(s,set()) for s in slots) and len(owner)==TRAIN+EVAL,"cannot fill balanced disjoint slots")
    out = {"train":[],"eval":[]}
    for key, (split,f,v,rep) in sorted(owner.items()):
        values = {field:old.VALUES[j][int(k)] for j,(field,k) in enumerate(zip(old.FIELDS,key.split("-"),strict=True))}
        field = old.FIELDS[f]
        position,template = (f+v+rep)%3,(f+v+rep)%2
        display = [n for n in old.FIELDS if n!=field]
        if template:
            display.reverse()
        display.insert(position,field)
        if template:
            prompt="Record: "+"; ".join(n+" = "+values[n] for n in display)+f".\nReturn the {field}.\nAnswer:"
        else:
            prompt=" | ".join(n+": "+values[n] for n in display)+f"\nWhat is the {field}?\nAnswer:"
        out[split].append(dict(key=key,field=field,template=template,position=position,prompt=prompt,answer=" "+values[field]))
    for k,n in (("train",TRAIN),("eval",EVAL)):
        rs=out[k]
        counts=Counter((r["field"],r["answer"]) for r in rs)
        check(len(rs)==n and len(counts)==24 and set(counts.values())=={n//24}
              and Counter(r["template"] for r in rs)=={i:n//2 for i in range(2)}
              and Counter(r["position"] for r in rs)=={i:n//3 for i in range(3)}, "balanced roster")
    return out


def indices(rows, order):
    check(order in ORDERS,"unregistered training order")
    return sorted(range(len(rows)),key=lambda i:pin(["factorial-training-order-v1",order,rows[i]["key"]]))


def freeze(rows):
    return dict(schema="spiraltorch.factorial_answer_training.v1", arms=list(ARMS), excluded=288,
                excluded_keys_sha256=pin(sorted(excluded_keys())), sizes={"train":TRAIN,"eval":EVAL},
                text_sha256={k:pin(v) for k,v in rows.items()},
                order_sha256={o:pin([rows["train"][i]["key"] for i in indices(rows["train"],o)]) for o in ORDERS},
                epochs=EPOCHS, updates_per_fit=UPDATES, parameters_per_arm=4608, learning_rate=p.LR,
                path="transformer.h.11.attn", strength=p.STRENGTH, rank=p.RANK, alpha=p.ALPHA,
                train_mask="answer_and_one_LF", nll_mask="answer_only", primary=PRIMARY, nll_margin=MARGIN,
                generation=dict(max_new_tokens=MAX_NEW,stop_ids=[NL,EOS],cache_values=CACHE,strict_first_line=True),
                endpoints=[0,UPDATES], base_id=old.BASE_ID, wheel_sha256=WHEEL, native_sha256=NATIVE)


def origin(st,base,width,layer):
    lora=base.with_lora(layer,p.RANK,p.ALPHA)
    weights=json.loads(lora.lora_to_json())
    check(len(weights["a"])==len(weights["b"])==width*p.RANK and any(weights["a"]) and not any(weights["b"]),"LoRA origin")
    affine=p.geometry(st,base.base_model_identity,"affine",width=width,layer=layer)
    both=p.geometry(st,base.base_model_identity,"wave_fixed",width=width,layer=layer)
    states=[affine]
    for stage in (dict(kind="saturation",saturation=1.,porosity=.05),dict(kind="projection",curvature=-1.,log_radius=0.)):
        r=json.loads(affine.state_json())
        r["placements"][0]["recipe"].update(schema="spiraltorch.wave_stage_adapter.v1",stage=stage)
        states.append(st.GeometryAdapterArtifact.create_json(json.dumps(r)))
    states.append(both)
    for state in states:
        tensors,n=p.parameter_state(state)
        check(n==2*width and all(not any(v) for v in tensors.values()),"unmatched capacity or origin")
    return [lora.with_geometry(s) for s in states]


def first_step(models):
    states=[p.parameter_state(m.geometry_artifact())[0] for m in models]
    check(len(states)==4 and states[3].get("log_radius")==[0.] and all(set(s)=={"gate","bias"} for s in states[:3]),"frozen radius")
    differences={}
    for name in ("gate","bias"):
        groups=list(zip(*(s[name] for s in states),strict=True))
        check(any(row[0]!=0. for row in groups) and all(math.isclose(row[0],v,abs_tol=2e-8,rel_tol=2e-6)
              for row in groups for v in row), "unmatched first update: "+name)
        differences[name]=max(max(row)-min(row) for row in groups)
    return differences


def decision(results):
    check(len(results)==len(ORDERS),"incomplete order replication")
    pairs=[(r["nll_contrasts"][PRIMARY]["answer_weighted_delta_nll"],r["line_contrasts"][PRIMARY]["delta"]) for r in results]
    if all(d < -MARGIN and n > 0 for d,n in pairs):
        return "projection_joint_advantage_on_this_split"
    if all(d > MARGIN and n < 0 for d,n in pairs):
        return "affine_joint_advantage_on_this_split"
    return "no_prespecified_joint_advantage_not_equivalence"


if __name__=="__main__":
    print(json.dumps(freeze(records()),sort_keys=True))
