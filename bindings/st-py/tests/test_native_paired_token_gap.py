"""The Python surface calls the same two-lane Rust objective as the sealed update."""
import base64
import copy
import json
import math
from pathlib import Path
import unittest
import spiraltorch as st

class PairedGap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f=json.loads((Path(__file__).resolve().parents[2]/"st-wasm/tests/native_causal_lm_cache_fixture.json").read_bytes())
        case,=[c for c in f["cases"] if c["name"]=="ordinary/wave"]
        cls.base=st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(base64.b64decode(f["weights"])))
        cls.rec=json.loads(case["geometry"])
        p,=cls.rec["placements"];p["path"]="transformer.h.1.mlp"
        p["recipe"]=dict(schema="spiraltorch.prefix_transport_adapter.v1",features=3,strength=.25,max_values=64,rotation="quarter_turn")
        for t in p["tensors"]:
            t["requires_grad"]=t["name"]=="gate"
            t["values"]=[.25,-.5,.125] if t["requires_grad"] else [0.]*3
        cls.first=[3,1,9,5,2,10];cls.second=[3,2,10]

    def model(self,rec=None):
        return self.base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(self.rec if rec is None else rec)))

    def test_three_objectives_pack_each_rows_own_last_token(self):
        m=self.model()
        ts=[m.last_mlp_prefix_margin_signal(p,7,8) for p in (self.first,self.second)]
        for mode,kw in [("binary",{}),("rank",{}),("anchored_rank",dict(reference_center=.75,stiffness=2.))]:
            with self.subTest(mode=mode):
                g=m.geometry_paired_token_gap_gradients(self.first,self.second,[7,8],mode,**kw)
                r=g.paired_token_gap_report()
                self.assertEqual(r,m.paired_token_gap(self.first,self.second,[7,8],mode,**kw))
                self.assertEqual((g.window_shape,g.target_tokens,g.geometry_artifact_id),([2,7],2,m.geometry_artifact_id))
                self.assertIsNone(g.context_token_preference_report())
                self.assertEqual(r["logits"],[[t["contrast"]["target_logit"],t["contrast"]["alternative_logit"]] for t in ts])
                for k,v in g.parameter_gradients()["transformer.h.1.mlp"].items():
                    for j,x in enumerate(v):
                        expected=sum(w*t[k+"_gradient"][j] for w,t in zip(r["row_derivatives"],ts))
                        self.assertAlmostEqual(x,expected,delta=2e-7+3e-4*abs(expected))
        controls=[m.geometry_token_menu_gradients(p,[7,8],gold_index=i,mix=1.) for i,p in enumerate((self.first,self.second))]
        self.assertIsNone(controls[0].paired_token_gap_report())
        g=m.geometry_paired_token_gap_gradients(self.first,self.second,[7,8],"binary")
        self.assertAlmostEqual(g.loss,sum(x.loss for x in controls)/2,places=6)
        for k,v in g.parameter_gradients()["transformer.h.1.mlp"].items():
            for j,x in enumerate(v):
                expected=sum(s.parameter_gradients()["transformer.h.1.mlp"][k][j] for s in controls)/2
                self.assertAlmostEqual(x,expected,delta=2e-7+3e-4*abs(expected))

    def test_anchored_gate_matches_independent_full_decoder_difference_and_sealed_step(self):
        m=self.model();opts=dict(reference_center=.75,stiffness=2.)
        g=m.geometry_paired_token_gap_gradients(self.first,self.second,[7,8],"anchored_rank",**opts)
        weights=g.parameter_gradients()["transformer.h.1.mlp"]["gate"]
        def loss(j,h):
            rec=copy.deepcopy(self.rec)
            gate,=[t for t in rec["placements"][0]["tensors"] if t["name"]=="gate"]
            gate["values"][j]+=h;model=self.model(rec)
            a,b=[model.next_token_contrast(p,7,8)["target_minus_alternative"] for p in (self.first,self.second)]
            return math.log1p(math.exp((b-a)/2))+((a+b)/2-.75)**2
        for j,x in enumerate(weights):
            approx=(loss(j,1/128)-loss(j,-1/128))*64
            self.assertAlmostEqual(x,approx,delta=2e-5+.02*abs(x))
        metric=m.calibrate_prefix_residual_metric([self.first,self.second])
        nxt,report=m.step_prefix_residual_rms(metric,[g],"diagonal",target_rms=.0001,max_parameter_l2=16.)
        self.assertEqual((report["members"],report["total_target_tokens"]),(1,2))
        self.assertNotEqual(nxt.geometry_artifact_id,m.geometry_artifact_id)
        with self.assertRaisesRegex(ValueError,"owner|belong|snapshot|identity"):
            nxt.step_prefix_residual_rms(metric,[g],"diagonal",target_rms=.0001,max_parameter_l2=16.)

    def test_rejects_before_payload_copy_and_retains_score_symmetry(self):
        class NoCopy:
            def __len__(self):return 4
            def __getitem__(self,i):raise AssertionError("copied tokens before admission")
        m=self.model()
        for ids,mode,kw in [([7,7],"rank",{}),([7,80],"binary",{}),([7,8],"unknown",{}),
                            ([7,8],"rank",{"reference_center":1.}),([7,8],"anchored_rank",{}),
                            ([7,8],"anchored_rank",dict(reference_center=float("nan"),stiffness=1.))]:
            with self.assertRaises((ValueError,TypeError)):
                m.geometry_paired_token_gap_gradients(NoCopy(),NoCopy(),ids,mode,**kw)
        with self.assertRaises(ValueError):
            m.paired_token_gap(self.first,self.first,[7,8],"rank")
        a=m.paired_token_gap(self.first,self.second,[7,8],"anchored_rank",reference_center=.5,stiffness=2.)
        b=m.paired_token_gap(self.second,self.first,[8,7],"anchored_rank",reference_center=-.5,stiffness=2.)
        self.assertEqual(a["loss"],b["loss"])
        self.assertEqual(a["row_derivatives"],[-x for x in reversed(b["row_derivatives"])])

if __name__=="__main__":unittest.main()
