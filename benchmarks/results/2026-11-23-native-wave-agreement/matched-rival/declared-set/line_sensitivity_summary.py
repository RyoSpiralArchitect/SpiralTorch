"""Recompute the published table and a labelled column-norm bound without a decoder."""
from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import sys

T,H,D = 1/65536,1/256,1/8


def column_bound(matrix, large, limiting, radius, cap):
    """Norm bound for *linear* maps with these column lengths (in sqrt(KL/T))."""
    n=len(matrix[0])
    assert matrix[large][limiting]>0 and all(len(row)==n and all(v>=0 for v in row) for row in matrix)
    remaining=[radius*math.sqrt(math.fsum(row[p]**2 for j,row in enumerate(matrix) if j!=large)) for p in range(n)]
    xmax=min(radius,(math.sqrt(cap)+remaining[limiting])/matrix[large][limiting])
    bounds=[min(cap,(xmax*matrix[large][p]+remaining[p])**2) for p in range(n)]
    return dict(dominant_coordinate_bound_in_probe_units=xmax, upper_per_anchor_over_T=bounds,
                upper_mean_over_T=math.fsum(bounds)/n)


def reduce(result, journal, wasm):
    blob,log,wasm_blob=(p.read_bytes() for p in (result,journal,wasm))
    raw,client=json.loads(blob),json.loads(wasm_blob)
    assert raw["schema"]=="spiraltorch.synthetic_line_sensitivity.v1" and raw["pretrained_calls"]==client["pretrained_calls"]==0
    assert raw["previous"]=="20d430ff3734fb1440bcab0b38163e82112850e9c997b50fc00f52048e9d18bb"
    rows=defaultdict(list)
    for r in map(json.loads,log.splitlines()):
        assert len(rows[r["case"]])==r["cursor"]
        rows[r["case"]].append(r["attempt"])
    keys={"/".join([str(i),m,b,s]) for i in range(3) for m,b,s in
          product(("lora","geometry","joint") if i==0 else ("geometry",),("parameter_block","coordinate"),("gradient_norm","envelope"))}
    assert set(raw["cases"])==set(rows)==keys and client["cells"]==len(keys)
    counts=dict(attempts=0,updates=0,refusals=0,raw=0,probes=0,scalar_trials=0,anchor_readouts=0,
                independent_initial_probes=0,same_state_raw_checks=0)
    main,starts,maps,bounds={},{},{},{}
    for key,attempts in sorted(rows.items()):
        offset,mode,basis,selector=key.split("/")
        c=raw["cases"][key]
        assert len(attempts)==len(c["events"]) and 0<len(attempts)<=(1 if int(offset) else 6)
        assert c["completed"]==sum(a["status"]=="applied" for a in attempts)
        trace=[]
        for i,(a,e) in enumerate(zip(attempts,c["events"])):
            counts["attempts"]+=1
            ok=a["status"]=="applied"
            k=a["step"]["anchor_kl"] if ok else a["refusal"]
            assert ok==e["applied"] and e["trials"]==len(k["trials"]) and e["readouts"]==k["anchor_readouts"]
            assert e["raw"]==("raw_miss" not in k) and e["loss_case"]==(i+int(offset))%3
            pin=a["step"]["plan_id"] if ok else a["plan_id"]
            assert pin=="sha256:"+sha256(c["plan"].encode()).hexdigest()
            counts["updates" if ok else "refusals"]+=1
            counts["same_state_raw_checks"]+=int(e["prior_checked"])
            counts["anchor_readouts"]+=e["readouts"]; counts["scalar_trials"]+=len(k["trials"])
            if ok:
                assert all(math.isclose(x,y,rel_tol=1e-10,abs_tol=3e-15) for x,y in zip(k["per_anchor_kl"],e["kl"]))
                assert .98*T<=math.fsum(e["kl"])/3<=T and max(e["kl"])<=2*T
            else:
                assert i+1==len(attempts) and k["reason"]==e["reason"]
            if e["raw"]:
                counts["raw"]+=1
                assert "sensitivity" not in k and ok
                trace.append("raw");continue
            info=k["sensitivity"]
            assert info["basis"]==basis and info["select"]==selector
            assert info["selected_axis"]==e["selected"]
            n=sum(ax["probe_per_anchor_kl"] is not None for ax in info["axes"])
            assert n==e["probes"] and info["probe_readouts"]==3*n
            counts["probes"]+=n; counts["independent_initial_probes"]+=e["initial_probe_checks"]
            counts["scalar_trials"]+=len(k["raw_miss"]["trials"])
            assert k["anchor_readouts"]==k["raw_miss"]["anchor_readouts"]+3*n+3*len(k["trials"])
            ax=info["axes"][e["selected"]] if e["selected"] is not None else None
            label=None if ax is None else ax["tensor"]+("" if ax["coordinate"] is None else "["+str(ax["coordinate"])+"]")
            trace.append(("accepted " if ok else "refused ")+str(label))
            if i==0:
                cap_safe=[t["mean_kl"]/T for t in k["trials"] if t["peak"]["kl"]<=2*T]
                starts[key]=dict(selected=label,applied=ok,probe_count=n,
                    predicted_fraction=ax and ax["prediction"] and ax["prediction"]["target_fraction"],
                    terminal_mean_over_T=k["mean_kl"]/T if ok else None,
                    maximum_tested_cap_safe_mean_over_T=max(cap_safe,default=None))
                if mode=="geometry" and basis=="coordinate" and selector=="envelope":
                    coords=info["axes"]
                    assert [(a["tensor"],a["coordinate"]) for a in coords]==list(product(("bias","gate"),range(3)))
                    values=[[v/T for v in a["probe_per_anchor_kl"]] for a in coords]
                    # Exploratory: strongest measured coordinate, its highest-KL prompt.
                    j=max(range(6),key=lambda j:max(values[j]))
                    lim=max(range(3),key=values[j].__getitem__)
                    bounds[offset]=dict(coordinate=coords[j]["tensor"]+"["+str(coords[j]["coordinate"])+"]",prompt=lim,
                        **column_bound([[math.sqrt(v) for v in row] for row in values],j,lim,D/H,2))
                    if offset=="0":
                        maps={a["tensor"]+"["+str(a["coordinate"])+"]":v for a,v in zip(coords,values)}
        if offset=="0":
            main[key[2:]]=dict(completed=c["completed"],events=trace,final_training_losses=c["scores"])
    assert all(counts[k]==client[k] for k in ("attempts","updates","raw","refusals","probes"))
    return dict(schema="spiraltorch.synthetic_line_sensitivity_summary.v1",counts=counts,
                main=main, first_decisions=starts, geometry_probe_KL_over_T=maps,
                posthoc_linear_model_bounds_not_finite_certificates=bounds, wasm=client,
                result_sha256=sha256(blob).hexdigest(),journal_sha256=sha256(log).hexdigest(),
                wasm_readout_sha256=sha256(wasm_blob).hexdigest(),previous_journal_sha256=raw["previous"])


if __name__=="__main__":
    if len(sys.argv)!=4 or not __debug__:
        sys.exit("NATIVE_RESULT RAW_JOURNAL WASM_READOUT (assertions required)")
    print(json.dumps(reduce(*map(Path,sys.argv[1:])),sort_keys=True,indent=2))
