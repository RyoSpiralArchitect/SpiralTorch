"""Audit native owner trajectories, Node and rendered Chrome; do not execute a decoder."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[4]/"bindings"/"st-wasm"/"tests"


def summarize(raw, node, dom):
    data = json.loads(raw)
    ref = json.loads((HERE/"line_owner_distance_results.json").read_bytes())
    capture = runpy.run_path(str(TESTS/"native_line_session_capture.py"))["Capture"]()
    capture.feed(dom)
    assert capture.passed
    browser = json.loads("".join(capture.chunks["result"]))["line_anchor_kl"]
    wasm = [node["line_anchor_kl"], browser]
    assert [x["checkpoint_source"] for x in wasm] == ["native", "wasm"]
    assert all((x["independently_checked_steps"],x["resumed_steps"],x["pretrained_calls"]) == (18,12,0)
               for x in wasm)
    modes = ("lora","geometry","joint")
    target = data["target_kl"]
    assert data["schema"] == "spiraltorch.synthetic_owner_anchor_kl.v1"
    assert target == 1/65536 and data["lower_target_ratio"] == 0.98 and data["ceiling_l2"] == 1/8
    assert data["in_sample"] and data["pretrained_calls"] == 0 and data["before"] == ref["before"]
    assert data["training_anchor_ids"] == [[1,2],[2,4],[2,1,3]]
    assert set(data["cases"]) == {p+"/"+m for m in modes for p in ("kl","ray")}
    assert all(set(x["cases"]) == set(modes) for x in wasm)
    summary = {k:data[k] for k in ("before","training_anchor_ids","in_sample","pretrained_calls",
                                     "target_kl","lower_target_ratio","ceiling_l2","host")}
    summary.update(schema="spiraltorch.owner_anchor_kl_comparison.v1",
                   native_raw_sha256=hashlib.sha256(raw).hexdigest(), cases={})
    max_loss, max_kl = 0., 0.
    for name,r in sorted(data["cases"].items()):
        policy, mode = name.split("/")
        steps = r["steps"]
        assert len(steps)==6 and len(r["after"])==3 and r["refusal"] is None
        assert r["status"] == ("matched" if policy == "kl" else "control")
        path = math.fsum(s["parameter_l2"] for s in steps)
        assert path==r["parameter_path"] and 0 < r["endpoint_l2"] <= path+1e-12
        assert len(r["endpoint_per_anchor_kl"])==3
        assert math.isclose(math.fsum(r["endpoint_per_anchor_kl"])/3, r["endpoint_kl"], abs_tol=1e-15)
        probes = sum(len(s.get("trials",[])) for s in steps)
        reads = sum(s.get("anchor_readouts",0) for s in steps)
        for s in steps:
            assert len(s["per_anchor_kl"])==3 and min(s["per_anchor_kl"])>=0
            assert math.isclose(math.fsum(s["per_anchor_kl"])/3,s["mean_kl"],abs_tol=1e-15)
            if policy=="kl":
                trials=s["trials"]
                assert 1 <= len(trials) <= 20 and trials[0]["requested_l2"] == 1/8
                assert math.isclose(trials[-1]["mean_kl"],s["mean_kl"],abs_tol=1e-15)
                assert s["anchor_readouts"] == 3*(len(trials)+1)
                assert 0.98 <= s["mean_kl"]/target <= 1
        if policy=="ray":
            old=ref["cases"][name]
            assert r["after"]==old["after"] and path==old["path_l2"] and r["endpoint_l2"]==old["endpoint_l2"]
            assert [s["loss"] for s in steps]==old["pre_update_losses"] and probes==reads==0
        else:
            for client in wasm:
                c = client["cases"][mode]
                assert c["trials"] == probes and len(c["steps"])==6
                assert abs(c["parameter_path"]-path)<3e-6 and abs(c["endpoint_kl"]-r["endpoint_kl"])<1e-8
                for a,b in zip(steps,c["steps"]):
                    assert 0.98 <= b["mean_kl"]/target <= 1
                    max_loss=max(max_loss,abs(a["loss"]-b["loss"]))
                    max_kl=max(max_kl,abs(a["mean_kl"]-b["mean_kl"]))
        summary["cases"][name] = {k:r[k] for k in ("status","parameter_path","endpoint_l2","endpoint_kl","after")}
        summary["cases"][name].update(mean_kl_by_step=[s["mean_kl"] for s in steps],
            per_anchor_kl_by_step=[s["per_anchor_kl"] for s in steps],
            pre_update_losses=[s["loss"] for s in steps], trials=probes, internal_anchor_readouts=reads)
    assert max_loss<=3e-6 and max_kl<=1e-8
    summary["wasm"] = dict(checked_steps_per_client=18, resumed_steps_per_client=12,
                            max_abs_pre_update_loss_error=max_loss, max_abs_step_mean_kl_error=max_kl)
    return summary


if __name__ == "__main__":
    if len(sys.argv)!=5:
        sys.exit("NATIVE_RESULT NODE_JSON_WITH_NATIVE_BRIDGE RENDERED_CHROME_DOM NEW_OUTPUT")
    raw, node, dom, out = map(Path,sys.argv[1:])
    result = summarize(raw.read_bytes(),json.loads(node.read_bytes()),dom.read_text())
    with out.open("x") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
