"""Installed native optimizer, checkpoint and LM tests with foreign ML imports forbidden."""
from array import array
import hashlib
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "peft", "numpy", "accelerate", "safetensors"}:
            raise AssertionError("native AdamW must not import " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

geometry = runpy.run_path(str(Path(__file__).with_name("test_native_geometry_runtime.py")))
lm = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
artifact = geometry["artifact"]
KINDS = ("affine", "wave", "fixed", "learned", "elliptic", "anchored", "causal", "gated", "topos", *geometry["FRACTIONAL_KINDS"])


class NativeGeometryAdamWTests(unittest.TestCase):
    def test_every_recipe_updates_and_resumes_exactly(self):
        x = array("f", [.2, -.3, .4, .5, .1, -.2])
        for kind in KINDS:
            with self.subTest(kind=kind):
                original = artifact(kind)
                runtime = st.NativeGeometryRuntime(original)
                optimizer = st.NativeGeometryAdamW(original, max_gradient_norm=.1)
                for _ in range(4):
                    batch = runtime.forward_sequence_buffer("site", x, (1, 2))
                    upstream = array("f", [2 * v / len(x) for v in batch.output])
                    before = optimizer.to_json()
                    next_runtime, next_state = runtime.step_adamw_buffer(batch, upstream, optimizer, .01, weight_decay=.02)
                    self.assertEqual(optimizer.to_json(), before)
                    runtime, optimizer = next_runtime, next_state
                saved = runtime.artifact()
                self.assertNotEqual(saved.artifact_id, original.artifact_id)
                self.assertEqual(optimizer.geometry_artifact_id, saved.artifact_id)
                self.assertEqual(optimizer.revision, 4)
                self.assertAlmostEqual(optimizer.max_gradient_norm, .1)
                state = optimizer.to_json()
                self.assertEqual(optimizer.checkpoint_id, "sha256:" + hashlib.sha256(state.encode()).hexdigest())
                restored_artifact = st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, saved.base_model_identity)
                restored = st.NativeGeometryRuntime(restored_artifact)
                resumed = st.NativeGeometryAdamW.from_json(state, optimizer.checkpoint_id, restored_artifact)
                self.assertEqual(resumed.to_json(), state)
                a = runtime.step_adamw(runtime.forward_sequence("site", x, (1, 2)), [.1] * 6, optimizer, .01)
                b = restored.step_adamw(restored.forward_sequence("site", x, (1, 2)), [.1] * 6, resumed, .01)
                self.assertEqual(a[0].artifact().to_json(), b[0].artifact().to_json())
                self.assertEqual(a[1].to_json(), b[1].to_json())
                fixed = {t["name"]: t["values"] for t in json.loads(original.state_json())["placements"][0]["tensors"] if not t["requires_grad"]}
                for tensor in json.loads(saved.state_json())["placements"][0]["tensors"]:
                    if tensor["name"] in fixed:
                        self.assertEqual(tensor["values"], fixed[tensor["name"]])

    def test_invalid_config_checkpoint_and_updates_leave_pair_unchanged(self):
        saved = artifact(frozen_gate=True)
        runtime = st.NativeGeometryRuntime(saved)
        optimizer = st.NativeGeometryAdamW(saved)
        self.assertIn("NativeGeometryAdamW", st.__all__)
        self.assertIsNone(optimizer.max_gradient_norm)
        for name in ("beta1", "beta2", "epsilon", "max_gradient_norm"):
            with self.assertRaises((TypeError, ValueError)):
                st.NativeGeometryAdamW(saved, **{name: True})
            with self.assertRaises(ValueError):
                st.NativeGeometryAdamW(saved, **{name: float("nan")})
        for name, value in (("beta1", 1.), ("beta2", -.1), ("epsilon", 0.), ("max_gradient_norm", 0.)):
            with self.assertRaises(ValueError):
                st.NativeGeometryAdamW(saved, **{name: value})
        batch = runtime.forward("site", [.2, -.3, .4], 1)
        before = optimizer.to_json()
        for rate in (-.1, float("nan"), float("inf"), True):
            with self.assertRaises((TypeError, ValueError)):
                runtime.step_adamw(batch, [.1] * 3, optimizer, rate)
        for decay in (-.1, float("nan"), True):
            with self.assertRaises((TypeError, ValueError)):
                runtime.step_adamw(batch, [.1] * 3, optimizer, .01, weight_decay=decay)
        with self.assertRaises(ValueError):
            runtime.step_adamw(batch, [float("nan")] * 3, optimizer, 0.)
        same = runtime.step_adamw(batch, [.1] * 3, optimizer, 0., weight_decay=.1)
        self.assertEqual(same[0].artifact().to_json(), saved.to_json())
        self.assertEqual(same[1].to_json(), before)
        changed = runtime.step_adamw(batch, [.1] * 3, optimizer, .01)
        for payload, expected, target in ((before + " ", optimizer.checkpoint_id, saved),
                                          (before, optimizer.checkpoint_id, changed[0].artifact())):
            with self.assertRaises(ValueError):
                st.NativeGeometryAdamW.from_json(payload, expected, target)
        with self.assertRaises(ValueError):
            changed[0].step_adamw(batch, [.1] * 3, changed[1], .01)
        fresh = changed[0].forward("site", [.2, -.3, .4], 1)
        with self.assertRaises(ValueError):
            changed[0].step_adamw(fresh, [.1] * 3, optimizer, .01)
        self.assertEqual(runtime.artifact().artifact_id, saved.artifact_id)
        self.assertEqual(optimizer.to_json(), before)

    def test_language_loss_adamw_and_checkpoint_pair_without_torch(self):
        config, weights = lm["fixture"]()
        base = st.NativeCausalLm.from_gpt2(config, weights)
        model = base.with_geometry(lm["artifact"](base, trainable=True))
        optimizer = st.NativeGeometryAdamW(model.geometry_artifact(), max_gradient_norm=.1)
        windows = [1, 2, 3, 4, 5, 6, 7, 8]
        first = model.geometry_loss_gradients(windows, [2, 4])
        for _ in range(20):
            gradient = model.geometry_loss_gradients(windows, [2, 4])
            model, optimizer = model.step_geometry_adamw(gradient, optimizer, .05, weight_decay=.01)
        self.assertLess(model.next_token_loss(windows, [2, 4]), first.loss - 1e-4)
        self.assertIsNone(base.geometry_artifact_id)
        self.assertEqual(model.base_model_identity, base.base_model_identity)
        saved = model.geometry_artifact()
        restored = base.with_geometry(st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, base.base_model_identity))
        resumed = st.NativeGeometryAdamW.from_json(optimizer.to_json(), optimizer.checkpoint_id, restored.geometry_artifact())
        a = model.step_geometry_adamw(model.geometry_loss_gradients(windows, [2, 4]), optimizer, .02)
        b = restored.step_geometry_adamw(restored.geometry_loss_gradients(windows, [2, 4]), resumed, .02)
        self.assertEqual(a[0].geometry_artifact().to_json(), b[0].geometry_artifact().to_json())
        self.assertEqual(a[1].to_json(), b[1].to_json())
        with self.assertRaises(ValueError):
            model.step_geometry_adamw(first, optimizer, .02)
        with self.assertRaises((TypeError, ValueError)):
            model.step_geometry_adamw(model.geometry_loss_gradients(windows, [2, 4]), optimizer, True)

    def test_resume_rejects_impossible_native_placement_clocks(self):
        saved = artifact("learned")
        original = json.loads(st.NativeGeometryAdamW(saved).to_json())
        for advance_first in (False, True):
            state = json.loads(json.dumps(original))
            state["revision"] = 1
            if advance_first:
                state["slots"][0]["steps"] = 1
            payload = json.dumps(state)
            digest = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
            with self.assertRaises(ValueError):
                st.NativeGeometryAdamW.from_json(payload, digest, saved)


if __name__ == "__main__":
    unittest.main()
