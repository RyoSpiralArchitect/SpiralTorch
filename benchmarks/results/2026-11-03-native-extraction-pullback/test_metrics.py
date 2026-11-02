"""Synthetic, no-pretrained-model rehearsal of the production receipt and reductions."""
import json
import math
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as m  # noqa: E402


class Measures(unittest.TestCase):
    def test_pool_squares_not_per_document_ratios_or_average_gradient(self):
        def site(n, x, delta, dy, correction, g):
            return dict(elements=n, input_rms=x, residual_rms=delta, wave=None, parameter_gradient_l2={"gate":g},
                        pullback=dict(upstream_rms=dy, correction_rms=correction, input_adjoint_rms=dy+correction))
        r = m.reduce_sites([site(1,1.,2.,2.,1.,3.), site(3,2.,1.,1.,3.,4.)])
        self.assertAlmostEqual(r["relative_residual_rms"], math.sqrt(7/13))
        self.assertAlmostEqual(r["relative_correction_rms"], math.sqrt(28/7))
        self.assertEqual(r["document_rms_parameter_l2"]["gate"], math.sqrt(12.5))

    def test_real_binding_receipt_is_admitted_and_crossed_owners_or_mask_rejected(self):
        import spiraltorch as st
        fixture = runpy.run_path(str(m.ROOT / "bindings/st-py/tests/test_native_causal_lm.py"))["fixture"]
        geometry = runpy.run_path(str(m.PREVIOUS.parent / "protocol.py"))["geometry"]
        config, weights = fixture()
        base = st.NativeCausalLm.from_gpt2(config, weights)
        with patch.multiple(m, BASE=base.base_model_identity, WIDTH=4, SITE="transformer.h.1.attn"):
            for arm in m.ARMS:
                g = geometry(st, m.BASE, arm, width=4, layer=1)
                record = json.loads(g.state_json())
                for t in record["placements"][0]["tensors"]:
                    t["values"] = [0.12] if t["name"] == "log_radius" else [.6, -.1, .2, .4]
                model = base.with_lora(1, 2, 4.).with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record)))
                grad = model.adaptation_loss_gradients([1,2,3,4], [1,4], labels=[-100,-100,3,4], mode="joint", observe_geometry=True)
                obs = json.loads(grad.observation_json())
                row = dict(order=m.ORDERS[0], arm=arm, index=0, key="synthetic", observation=obs)
                meta = dict(owners=[model.lora_artifact_id,model.geometry_artifact_id], log_radius=.12)
                args = (row,m.ORDERS[0],arm,0,meta)
                s = m.validate_row(*args, ["synthetic",4,2])
                self.assertGreater(s["pullback"]["correction_rms"], 0.)
                if arm == "wave_learned":
                    self.assertNotEqual(s["log_radius_gradient"], 0.)
                with self.assertRaisesRegex(ValueError, "provenance"):
                    m.validate_row(*args, ["synthetic",4,1])
                meta["owners"].reverse()
                with self.assertRaisesRegex(ValueError, "provenance"):
                    m.validate_row(*args, ["synthetic",4,2])


if __name__ == "__main__":
    unittest.main()
