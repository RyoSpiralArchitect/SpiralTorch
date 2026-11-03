"""Torch-free factor controls: real joint SGD, checkpoint restore, cached inference and local VJPs."""
import json
from pathlib import Path
import runpy
import unittest

import spiraltorch as st

fixture = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))["fixture"]
CFG = dict(curvature=-1., saturation=1., porosity=.05, log_radius=0.)
SITE = "transformer.h.1.attn"
KINDS = ("affine", "saturation", "projection", "both")
IDS, MASK = [1, 2, 3, 4], [-100, -100, 3, 4]


def attach(base, kind):
    tensors = [dict(name=name, shape=[4], role="parameter", requires_grad=True, values=values)
               for name, values in (("gate", [1.3, -.8, 2., .4]), ("bias", [.25, 1.8, -.1, .6]))]
    common = dict(features=4, strength=.25)
    if kind == "affine":
        recipe = dict(schema="spiraltorch.channel_affine_adapter.v1", max_values=1024, **common)
    elif kind == "both":
        recipe = dict(schema="spiraltorch.wave_gate_adapter.v2", learnable_radius=False,
                      kernel=dict(curvature=-1., saturation=1., porosity=.05, max_values=1024), **common)
        tensors.append(dict(name="log_radius", shape=[], values=[0.], role="buffer", requires_grad=False))
    else:
        stage = dict(kind=kind, **({"saturation":1., "porosity":.05} if kind=="saturation" else {"curvature":-1., "log_radius":0.}))
        recipe = dict(schema="spiraltorch.wave_stage_adapter.v1", stage=stage, max_values=1024, **common)
    state = st.GeometryAdapterArtifact.create_json(json.dumps(dict(schema="spiraltorch.geometry_adapter_artifact.v1",
            base_model_identity=base.base_model_identity, policy="full_prefix_f32_no_cache.v1", training=True,
            placements=[dict(path=SITE, training=True, recipe=recipe, tensors=tensors)])))
    return base.with_lora(1,2,4.).with_geometry(state)


class Factorial(unittest.TestCase):
    def setUp(self):
        self.base = st.NativeCausalLm.from_gpt2(*fixture())

    def test_same_hidden_and_factual_dy_not_four_independent_loss_gradients(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                model = attach(self.base, kind)
                g = model.adaptation_loss_gradients(IDS,[1,4],labels=MASK,mode="joint",observe_geometry=True)
                old = json.loads(g.observation_json())
                new = json.loads(model.wave_factorial_observation_json(IDS,[1,4],CFG,labels=MASK))
                r = new["sites"][SITE].pop("wave_factorial")
                self.assertEqual(new["schema"], "spiraltorch.native_causal_lm_adaptation_observation.v2")
                new["schema"] = old["schema"]
                self.assertEqual(old,new)
                self.assertEqual((r["factual"], r["elements"],r["gate_outside"]), (kind,12,2))
                self.assertEqual((r["factual_output_max_abs_diff"],r["factual_adjoint_max_abs_diff"]),(0.,0.))
                self.assertGreater(r["affine_outside"], 0)
                self.assertLess(r["affine_on_outside"]["residual_sq"], r["affine"]["residual_sq"])
                self.assertEqual(len({r[k]["residual_sq"] for k in KINDS}),4)
                site = old["sites"][SITE]
                self.assertAlmostEqual(r[kind]["residual_sq"], site["residual_rms"]**2 * 12, places=12)
                self.assertAlmostEqual(r[kind]["correction_sq"], site["pullback"]["correction_rms"]**2 * 12, places=12)

    def test_partial_adapters_train_reload_and_run_cached_inference(self):
        for kind in ("saturation","projection"):
            model = attach(self.base,kind)
            session = st.NativeCausalLmTrainingSession(model,[IDS],4,mode="joint",learning_rate=.02,labels=[MASK])
            step = session.step()
            self.assertNotEqual(step.pre_geometry_artifact_id,step.post_geometry_artifact_id)
            self.assertNotEqual(step.pre_lora_artifact_id,step.post_lora_artifact_id)
            payload,pin = session.checkpoint_with_id()
            restored = st.NativeCausalLm.from_session_checkpoint_at(self.base,payload,pin,update=1,recorded_data_id=session.data_id,
                       post_lora_id=step.post_lora_artifact_id,post_geometry_id=step.post_geometry_artifact_id)
            ctx = st.NativeCausalLmContext(restored, IDS[:2])
            cached = ctx.append(IDS[2]).values()
            full = session.model.forward(IDS[:3],[1,3],last_only=True).values()
            self.assertLess(max(abs(a-b) for a,b in zip(cached,full,strict=True)),3e-6)

    def test_reject_incompatible_source_and_inert_knobs_before_reading_tokens(self):
        class Poison:
            def __len__(self):
                raise AssertionError("foreign tokens were read")
        model = attach(self.base,"both")
        for cfg in (dict(CFG,log_radius=.5),dict(CFG,unknown=1),dict(CFG,curvature=True)):
            with self.assertRaises((ValueError, TypeError)):
                model.wave_factorial_observation_json(Poison(),[1,4],cfg)
        with self.assertRaises(ValueError):
            self.base.wave_factorial_observation_json(Poison(),[1,4],CFG)
        for kind,extra in (("saturation", {"log_radius":0.}),("projection",{"porosity":.1})):
            state = json.loads(attach(self.base,kind).geometry_artifact().state_json())
            state["placements"][0]["recipe"]["stage"].update(extra)
            with self.assertRaises(ValueError):
                st.GeometryAdapterArtifact.create_json(json.dumps(state))


if __name__ == "__main__":
    unittest.main()
