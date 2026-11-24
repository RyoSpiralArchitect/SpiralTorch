"""A unit logit-margin adjoint, not CE and not four independently evaluated decoders."""
from array import array
import json
import math
from pathlib import Path
import runpy
import sys
import unittest

for name in ("torch", "numpy", "transformers", "safetensors"):
    if __name__ == "__main__":
        sys.modules[name] = None
import spiraltorch as st

T = runpy.run_path(str(Path(__file__).with_name("test_native_wave_factorial.py")))
CFG, SITE, KINDS = T["CFG"], T["SITE"], T["KINDS"]


def unpack(buf):
    a = array("f")
    a.frombytes(buf)
    if sys.byteorder!="little":
        a.byteswap()
    return a.tolist()


class WaveMargin(unittest.TestCase):
    def setUp(self):
        self.base = st.NativeCausalLm.from_gpt2(*T["fixture"]()).with_readout_accumulator("f32_wide_backward")
        self.artifacts = {k:T["attach"](self.base,k).geometry_artifact() for k in KINDS}
        self.ids, self.gold, self.rival = [1,4,3], 5, 7

    def test_four_standalone_maps_match_captured_row_and_arriving_gradient(self):
        xs=[]
        for factual, art in self.artifacts.items():
            m=self.base.with_geometry(art)
            r=m.last_attention_wave_margin_pullbacks(self.ids,self.gold,self.rival,CFG)
            self.assertEqual((r["schema"],r["objective"],r["factual"],r["geometry_artifact_id"],r["lora_artifact_id"]),
                             ("spiraltorch.wave_margin_parameter_pullbacks.v1","target_logit_minus_alternative_logit",
                              factual,art.artifact_id,None))
            self.assertEqual(r["contrast"],m.next_token_contrast(self.ids,self.gold,self.rival))
            xs.append(r["site_input"])
            for kind,alt in self.artifacts.items():
                local=st.NativeGeometryRuntime(alt).forward(SITE,unpack(r["site_input"]),1)
                ref=local.vjp_parameters(unpack(r["upstream"]))
                for p in ("gate","bias"):
                    self.assertEqual(unpack(r["arms"][kind][p]),ref[p])
            reverse=m.last_attention_wave_margin_pullbacks(self.ids,self.rival,self.gold,CFG)
            self.assertEqual(reverse["site_input"],r["site_input"])
            for k in KINDS:
                for p in ("gate","bias"):
                    self.assertEqual(unpack(reverse["arms"][k][p]),[-g for g in unpack(r["arms"][k][p])])
        self.assertEqual(len(set(xs)),1)

    def test_central_difference_of_actual_decoder_margin_without_lora(self):
        for k in KINDS:
            art=self.artifacts[k]
            m=self.base.with_geometry(art)
            r=m.last_attention_wave_margin_pullbacks(self.ids,self.gold,self.rival,CFG)
            g={p:unpack(buf) for p,buf in r["arms"][k].items()}
            norm=math.sqrt(math.fsum(v*v for vec in g.values() for v in vec))
            self.assertGreater(norm,1e-5)
            def perturbed(eps):
                state=json.loads(art.to_json())
                for p in state["placements"][0]["tensors"]:
                    if p["name"] in g:
                        p["values"]=[v+eps*d/norm for v,d in zip(p["values"],g[p["name"]])]
                changed=st.GeometryAdapterArtifact.create_json(json.dumps(state))
                return self.base.with_geometry(changed).next_token_contrast(self.ids,self.gold,self.rival)["target_minus_alternative"]
            numeric=(perturbed(.005)-perturbed(-.005))/.01
            self.assertTrue(math.isclose(norm,numeric,rel_tol=.006,abs_tol=4e-6),(k,norm,numeric))

    def test_admission_precedes_prompt_copy(self):
        class Poison:
            def __len__(self): return 3
            def __getitem__(self, _): raise AssertionError("read prompt before factual/config admission")
        m=self.base.with_geometry(self.artifacts["both"])
        for model,g,r,c in ((self.base,1,2,CFG),(m,1,1,CFG),(m,1,11,CFG),(m,1,2,dict(CFG,log_radius=.5))):
            with self.assertRaises((ValueError,TypeError)):
                model.last_attention_wave_margin_pullbacks(Poison(),g,r,c)


if __name__=="__main__":
    unittest.main()
