"""Native-only attention value read, supplied-source partials, and optimizer ownership."""
import copy
from pathlib import Path
import runpy
import unittest
import spiraltorch as st

fixture=runpy.run_path(str(Path(__file__).with_name("test_native_paired_token_gap.py")))["PairedGap"]

class AttentionValueGate(unittest.TestCase):
    def test_actual_read_and_training_boundary(self):
        fixture.setUpClass();f=fixture();p=f.first
        anchors=set()
        for route in ({"kind":"attention"},{"kind":"uniform"},{"kind":"fixed","position":2}):
            rec=copy.deepcopy(f.rec);recipe=rec["placements"][0]["recipe"]
            del recipe["rotation"]
            recipe.update(schema="spiraltorch.attention_value_gate.v1",route=route)
            model=f.model(rec);sig=model.last_mlp_value_read_signal(p,7,8)
            self.assertEqual(sig["route"],route)
            self.assertEqual(sig["shape"],[len(p),3])
            rt=st.NativeGeometryRuntime(model.geometry_artifact());site=sig["site"]
            with self.assertRaises(ValueError):rt.forward_sequence(site,sig["site_input"],[1,1])
            batch=rt.forward_value_read_sequence(site,sig["site_input"],sig["source"],[1,1])
            self.assertEqual(batch.vjp_input([1.,2.,3.]),[1.,2.,3.])
            self.assertEqual(batch.vjp_value_source([1.,2.,3.]),[.0625,-.25,.09375])
            self.assertEqual(batch.vjp_parameters(sig["upstream"])["gate"],sig["gate_gradient"])
            metric=model.calibrate_prefix_residual_metric([p,f.second]);anchors.add(metric.report()["id"])
            g=model.geometry_paired_token_gap_gradients(p,f.second,[7,8],"weakest_lane")
            nxt,step=model.step_prefix_residual_rms(metric,[g],"diagonal",target_rms=.0001,max_parameter_l2=16.)
            self.assertEqual(step["post_geometry_id"],nxt.geometry_artifact_id)
        self.assertEqual(len(anchors),3)

if __name__=="__main__":unittest.main()
