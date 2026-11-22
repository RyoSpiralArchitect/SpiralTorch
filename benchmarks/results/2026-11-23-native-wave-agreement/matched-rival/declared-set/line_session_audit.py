"""Verify saved complete-line handoffs without loading weights or running SGD."""
import hashlib
import json
from pathlib import Path
import struct
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def word(value):
    return struct.unpack("<I",struct.pack("<f",value))[0]


def error(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max(error(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max(error(x,y) for x,y in zip(a,b))
    return abs(a-b)


def audit(plan,reference,bridge,node,browser,returned,out):
    files=dict(plan=plan,direct=reference,native_bridge=bridge,node_download=node,
               chrome_download=browser,native_take_back=returned)
    raw={k:path.read_bytes() for k,path in files.items()}
    assert raw["node_download"]==raw["chrome_download"]
    ref,b,c,back=(json.loads(raw[k]) for k in ("direct","native_bridge","chrome_download","native_take_back"))
    plan_id="sha256:"+digest(raw["plan"])
    assert b["plan_id"]==c["plan_id"]==plan_id and (b["cut"],b["return_at"],b["total"])==(2,4,6)
    assert ref.keys()==b["cases"].keys()==c["cases"].keys()==back.keys() and len(ref)==4
    cases={}
    for host,r in ref.items():
        n,wasm,cpu=b["cases"][host],c["cases"][host],back[host]
        assert len(r["steps"])==6 and r["endpoint"]==n["endpoint"]
        assert [s["loss"] for s in r["steps"]]==n["losses"]
        prev=r["origin"]
        for step in r["steps"]:
            assert step["before"]==prev and step["scores"]["loss"]==step["loss"]
            prev=step["after"]
        assert prev==n["final_id"]
        for bundle,cut in ((n,2),(wasm,4)):
            snap=json.loads(bundle["payload"])
            assert "sha256:"+digest(bundle["payload"].encode())==bundle["id"]
            assert "sha256:"+digest(snap["geometry_json"].encode())==snap["geometry_id"]
            assert snap["updates"]==cut and snap["plan_id"]==plan_id
            assert snap["origin_geometry_id"]==r["origin"] and snap["base_id"]==r["base"]
        assert json.loads(n["payload"])["geometry_id"]==r["steps"][1]["after"]
        assert cpu["prefix_pin"]==wasm["id"] and len(wasm["suffix_losses"])==2
        ulps=[abs(word(x)-word(y)) for x,y in zip(wasm["suffix_losses"],n["losses"][4:])]
        # All losses are nonnegative for these objectives: unsigned word distances are ULPs.
        assert all(v>=0 for v in wasm["suffix_losses"]+n["losses"])
        params=error(wasm["endpoint"]["parameters"],r["endpoint"]["parameters"])
        logits=error(wasm["endpoint"]["logits"],r["endpoint"]["logits"])
        assert max(params,logits,cpu["native_final_error"],cpu["suffix_loss_error"])<=3e-6
        cases[host]=dict(native_prefix=n["id"],browser_prefix=wasm["id"],return_payload_bytes=len(wasm["payload"].encode()),
                         returned_geometry_id_exact=cpu["result_id"]==prev,wasm_parameter_error=params,wasm_logit_error=logits,
                         returned_native_endpoint_error=cpu["native_final_error"],suffix_loss_ulps=ulps)
    result=dict(schema="spiraltorch.artificial_line_session_transfer.v1",plan_id=plan_id,updates_per_host=6,
                transfer_cursors=[2,4],source_sha256={k:digest(v) for k,v in raw.items()},cases=cases)
    with out.open("x") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    if len(sys.argv)!=8:
        sys.exit("PLAN DIRECT NATIVE_BRIDGE NODE_DOWNLOAD CHROME_DOWNLOAD NATIVE_TAKE_BACK NEW_RESULT")
    audit(*map(Path,sys.argv[1:]))
