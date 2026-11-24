"""Adjacent prefix changes share a native derivative, calibrator and append path."""
import copy
import json
from pathlib import Path
import runpy
import unittest

Fixture=runpy.run_path(str(Path(__file__).with_name("test_native_paired_token_gap.py")))["PairedGap"]

class PrefixInnovation(unittest.TestCase):
    def test_sources_owners_and_cold_restore(self):
        Fixture.setUpClass()
        f=Fixture();old=f.model()
        rec=copy.deepcopy(f.rec)
        rec["placements"][0]["recipe"]["schema"]="spiraltorch.prefix_transport_adapter.v3"
        m=f.model(rec)
        p=f.first
        sig=m.last_mlp_prefix_margin_signal(p,7,8)
        self.assertEqual(sig["source_encoding"],"adjacent_difference")
        self.assertEqual(sig["source_features"][:3],[0.]*3)
        self.assertNotIn("source_encoding",old.last_mlp_prefix_margin_signal(p,7,8))
        for j,g in enumerate(sig["gate_gradient"]):
            self.assertAlmostEqual(g,sum(sig["source_features"][j::3])*sig["bias_gradient"][j],delta=2e-7)
        prompts=[p,f.second]
        metric=m.calibrate_prefix_residual_metric(prompts)
        grad=m.geometry_paired_token_gap_gradients(*prompts,[7,8],"weakest_lane")
        with self.assertRaises(ValueError):
            m.step_prefix_residual_rms(old.calibrate_prefix_residual_metric(prompts),[grad],"diagonal",
                                       target_rms=.0001,max_parameter_l2=16.)
        nxt,_=m.step_prefix_residual_rms(metric,[grad],"diagonal",target_rms=.0001,max_parameter_l2=16.)
        restored=f.model(json.loads(nxt.geometry_artifact().to_json()))
        self.assertEqual(restored.forward(p,[1,len(p)],last_only=True).values(),
                         nxt.forward(p,[1,len(p)],last_only=True).values())

if __name__=="__main__":
    unittest.main()
