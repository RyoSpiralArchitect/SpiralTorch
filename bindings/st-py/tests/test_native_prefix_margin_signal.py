"""Raw next-token margin, not CE or a derivative with respect to token IDs."""
import base64
import copy
import json
from pathlib import Path
import unittest

import spiraltorch as st


class PrefixMarginSignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[2]/"st-wasm/tests/native_causal_lm_cache_fixture.json"
        f=json.loads(path.read_text())
        case,=[c for c in f["cases"] if c["name"]=="ordinary/wave"]
        cls.base=st.NativeCausalLm.from_gpt2(case["config"],st.SafeTensorArchive.from_bytes(base64.b64decode(f["weights"])))
        cls.rec=json.loads(case["geometry"])
        p,=cls.rec["placements"]
        p["path"]="transformer.h.1.mlp"
        p["recipe"]=dict(schema="spiraltorch.prefix_transport_adapter.v1",features=3,strength=.25,
                         max_values=64,rotation="quarter_turn")
        for t in p["tensors"]:
            t["requires_grad"]=t["name"]=="gate"
            t["values"]=[.25,-.5,.125] if t["requires_grad"] else [0.]*3

    def model(self,record):
        return self.base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record)))

    def test_factual_and_swapped_ports(self):
        for policy in ("f32","f32_wide_backward","f64"):
            m=self.model(self.rec).with_readout_accumulator(policy)
            ids=[3,1,9,5,2,10]
            t=m.last_mlp_prefix_margin_signal(ids,7,8)
            self.assertEqual(t["contrast"],m.next_token_contrast(ids,7,8))
            self.assertEqual(t["shape"],[6,3])
            self.assertEqual(t["readout_accumulator"],policy)
            self.assertEqual(t["geometry_artifact_id"],m.geometry_artifact_id)
            neg=m.last_mlp_prefix_margin_signal(ids,8,7)
            self.assertEqual(t["site_input"],neg["site_input"])
            for j,g in enumerate(t["gate_gradient"]):
                self.assertAlmostEqual(g,-neg["gate_gradient"][j],places=6)
                self.assertAlmostEqual(g,sum(t["source_features"][j::3])*t["bias_gradient"][j],places=6)

    def test_budget_and_unsupported_downstream_site_fail_before_token_copy(self):
        class NoCopy:
            def __len__(self):
                return 6
            def __getitem__(self,i):
                raise AssertionError("copied IDs before checking model placement or site budget")
        small=copy.deepcopy(self.rec)
        small["placements"][0]["recipe"]["max_values"]=3
        with self.assertRaises(ValueError):
            self.model(small).last_mlp_prefix_margin_signal(NoCopy(),7,8)
        block=copy.deepcopy(self.rec)
        block["placements"][0]["path"]="transformer.h.1"
        m=self.model(block)
        with self.assertRaisesRegex(ValueError,"downstream block"):
            m.last_mlp_prefix_margin_signal(NoCopy(),7,8)


if __name__=="__main__":
    unittest.main()
