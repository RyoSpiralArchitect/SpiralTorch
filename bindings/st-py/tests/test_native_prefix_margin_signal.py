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

    def test_train_only_metric_survives_gate_motion_but_not_stale_gradients(self):
        m = self.model(self.rec)
        rows = [[3,1,9], [2,6,9]]
        metric = m.calibrate_prefix_residual_metric(rows)
        self.assertIsInstance(metric, st.NativePrefixResidualMetric)
        self.assertNotEqual(metric.report()["id"], m.calibrate_prefix_residual_metric(rows[::-1]).report()["id"])
        gs = [m.geometry_token_menu_gradients(r,[7,8],gold_index=0,mix=1.) for r in rows]
        for rule in ("uniform","diagonal","cyclic"):
            nxt, step = m.step_prefix_residual_rms(metric,gs,rule,target_rms=.0001,max_parameter_l2=16.)
            self.assertAlmostEqual(step["realized_rms"],.0001,places=8)
            self.assertLess(sum(step["member_loss_dots"]),0.)
            self.assertEqual(nxt.calibrate_prefix_residual_metric(rows).report(),metric.report())
            with self.assertRaises(ValueError):
                nxt.step_prefix_residual_rms(metric,gs,rule,target_rms=.0001,max_parameter_l2=16.)
        with self.assertRaises(ValueError):
            m.step_prefix_residual_rms(metric,gs,"uniform",target_rms=.0001,max_parameter_l2=1e-14)
        other = m.with_readout_accumulator("f64")
        with self.assertRaisesRegex(ValueError,"owner mismatch"):
            other.step_prefix_residual_rms(metric,gs,"uniform",target_rms=.0001,max_parameter_l2=16.)
        different = copy.deepcopy(self.rec)
        different["placements"][0]["recipe"]["rotation"] = "identity"
        with self.assertRaisesRegex(ValueError,"owner mismatch"):
            self.model(different).step_prefix_residual_rms(metric,gs,"uniform",target_rms=.0001,max_parameter_l2=16.)

    def test_calibration_rejects_non_gate_owners_and_large_batches(self):
        both = copy.deepcopy(self.rec)
        for t in both["placements"][0]["tensors"]:
            t["requires_grad"] = True
        with self.assertRaises(ValueError):
            self.model(both).calibrate_prefix_residual_metric([[3]])
        m = self.model(self.rec)
        for invalid in ([], [[]], [[1]]*257, [[1000000]]):
            with self.assertRaises(ValueError):
                m.calibrate_prefix_residual_metric(invalid)
        with self.assertRaises(TypeError):
            st.NativePrefixResidualMetric()


if __name__=="__main__":
    unittest.main()
