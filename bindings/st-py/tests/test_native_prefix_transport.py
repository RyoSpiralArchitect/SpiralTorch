"""Cross-client, generated decoder controls; optional write-once reference for WASM."""
import base64
import json
import math
from pathlib import Path
import sys
import unittest
import spiraltorch as st

SITE = "transformer.h.1.mlp"
TURNS = ("identity", "quarter_turn", "inverse_quarter_turn")


def reference():
    fixture = json.loads((Path(__file__).resolve().parents[2] / "st-wasm/tests/native_causal_lm_cache_fixture.json").read_text())
    case, = [c for c in fixture["cases"] if c["name"]=="ordinary/wave"]
    base=st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(base64.b64decode(fixture["weights"],validate=True)))
    prompt=[3,1,9,5,2,10]; suffix=[7]; out={}
    for turn in TURNS:
        record=json.loads(case["geometry"]);p,=record["placements"];p["path"]=SITE
        p["recipe"]=dict(schema="spiraltorch.prefix_transport_adapter.v1",features=3,strength=.25,rotation=turn,max_values=64)
        for t in p["tensors"]: t["values"]=[.25,-.5,.75] if t["name"]=="gate" else [0.]*3
        art=st.GeometryAdapterArtifact.create_json(json.dumps(record))
        m=base.with_geometry(art)
        g=m.geometry_token_menu_gradients(prompt,[7,8],gold_index=0,mix=1.)
        after=m.step_adaptation_sgd(g,"geometry",1/32)
        raw=after.geometry_artifact()
        restored=base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.to_json(),raw.artifact_id,base.base_model_identity))
        old=m.forward(prompt,[1,6],last_only=True).values()
        new=restored.forward(prompt,[1,6],last_only=True).values()
        full=after.forward(prompt+suffix,[1,7],last_only=True).values()
        ctx=st.NativeCausalLmContext(restored,prompt,max_cache_values=90)
        fork=ctx.with_suffix(suffix)
        vals=fork.logits().values()
        if old==new or new!=after.forward(prompt,[1,6],last_only=True).values() or (ctx.cache_values,fork.cache_values)!=(78,90):
            raise AssertionError("training, reload or compact cache")
        if len(vals)!=11 or len(full)!=11 or not all(math.isfinite(v) and abs(v-u)<3e-6 for v,u in zip(vals,full)):
            raise AssertionError("cached and full logits differ")
        out[turn]=dict(origin=art.to_json(),origin_id=art.artifact_id,grads=g.parameter_gradients()[SITE],
                       loss=g.loss,trained=new,suffix=full)
    return out


class PrefixTransport(unittest.TestCase):
    def test_trained_prefix_transport_cache(self):
        r=reference()
        self.assertEqual(len({tuple(v["grads"]["gate"]) for v in r.values()}),3)


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--reference":
        with Path(sys.argv[2]).open("x") as f: json.dump(reference(),f,sort_keys=True,allow_nan=False)
    else: unittest.main()
