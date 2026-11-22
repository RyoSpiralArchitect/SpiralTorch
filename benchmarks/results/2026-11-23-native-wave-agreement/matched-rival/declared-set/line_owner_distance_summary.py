"""Audit measured synthetic runs and reduce them without importing an ML package."""
import json
import math
from pathlib import Path
import sys


def summary(native, node, chrome):
    modes = ("lora", "geometry", "joint")
    result = {"schema":"spiraltorch.synthetic_line_owner_distance.v1",
              "host":native["host"], "before":native["before"], "requested_l2":native["target_l2"],
              "in_sample":True, "pretrained_calls":0, "cases":{}}
    d = result["requested_l2"]
    assert native["host"]=="ordinary/wave" and d==1/256
    assert set(native["cases"]) == {p+"/"+m for p in ("sgd","ray") for m in modes}
    joint_shares, relative_errors = [], []
    for policy, key in [("sgd","line_owners"),("ray","line_ray")]:
        js = node[key]
        assert js == chrome[key] and (js["matched_steps"],js["native_resumed_steps"]) == (18,12)
        for mode in modes:
            name = policy+"/"+mode
            row = native["cases"][name]
            assert [s["update"] for s in row["steps"]] == list(range(1,7))
            path = math.fsum(s["displacement"]["total"] for s in row["steps"])
            observed = dict(path_l2=path,endpoint_l2=row["endpoint_distance"]["total"],
                            centered_logit_rms=row["centered_logit_rms"])
            assert path==row["path_l2"] and observed["endpoint_l2"] <= path + 1e-12
            assert all(abs(v-js["measurements"][mode][k])<=3e-6 for k,v in observed.items())
            for s in row["steps"]:
                motion = s["displacement"]
                assert math.isclose(math.hypot(motion["lora"],motion["geometry"]),motion["total"],abs_tol=1e-14)
                assert mode!="geometry" or motion["lora"]==0
                assert mode!="lora" or motion["geometry"]==0
                ray = s["ray"]
                if policy == "sgd":
                    assert ray is None
                    continue
                assert ray["requested_l2"]==d and ray["mode"]==mode and ray["gradient_dot_step"]<0
                assert abs(motion["total"]/d-1)<=1e-4 and ray["relative_vector_error"]<=1e-4
                relative_errors.append(ray["relative_vector_error"])
                if mode=="joint":
                    joint_shares.append((motion["lora"]/motion["total"])**2)
            assert len(row["after"])==3
            result["cases"][name]=dict(observed, after=row["after"], pre_update_losses=[s["loss"] for s in row["steps"]])
    assert len(relative_errors)==18 and len(joint_shares)==6
    result["max_native_ray_vector_error"]=max(relative_errors)
    result["joint_lora_squared_motion_share_range"]=[min(joint_shares),max(joint_shares)]
    result["wasm_ray"]={k:node["line_ray"][k] for k in ("max_native_loss_error","max_native_endpoint_error","max_relative_vector_error","fixed_steps","refusals")}
    return result


if __name__=="__main__":
    if len(sys.argv)!=5: sys.exit("NATIVE_RESULT NODE_REPORT CHROME_EXTRACTED_REPORT NEW_OUTPUT")
    raw, node, chrome = (json.loads(Path(p).read_bytes()) for p in sys.argv[1:4])
    with Path(sys.argv[4]).open("x") as f:
        json.dump(summary(raw,node,chrome),f,sort_keys=True,allow_nan=False,indent=2)
        f.write("\n")
