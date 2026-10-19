"""Native microbatch reduction and update tests with foreign ML imports forbidden."""
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "accelerate", "safetensors"}:
            raise AssertionError("native accumulation must not import " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
MICROBATCHES = [([1, 2, 3], [1, 3]), ([4, 5, 6, 7, 2, 4, 6, 8], [2, 4]), ([3, 2, 1, 4, 5], [1, 5])]


class NativeCausalLmAccumulationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(self.config, self.weights)
        self.model = self.base.with_geometry(fixtures["artifact"](self.base, trainable=True))

    def test_uniform_windows_match_full_batch(self):
        self.assertIn("NativeCausalLmAccumulator", st.__all__)
        with self.assertRaises(TypeError):
            st.NativeCausalLmAccumulator()
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8]
        full = self.model.geometry_loss_gradients(windows, [3, 4])
        state = self.model.geometry_accumulator()
        state.add(self.model.geometry_loss_gradients(windows[:4], [1, 4]))
        state.accumulate_loss(self.model, windows[4:], [2, 4])
        self.assertEqual((state.microbatches, state.target_tokens), (2, 9))
        self.assertAlmostEqual(state.loss, full.loss, delta=2e-6)
        for path, params in full.parameter_gradients().items():
            for name, values in params.items():
                for a, b in zip(values, state.parameter_gradients()[path][name]):
                    self.assertAlmostEqual(a, b, delta=1e-7)
        a = json.loads(state.step_sgd(self.model, .1).geometry_artifact().state_json())
        b = json.loads(self.model.step_geometry_sgd(full, .1).geometry_artifact().state_json())
        for left, right in zip(a["placements"][0]["tensors"], b["placements"][0]["tensors"]):
            for x, y in zip(left["values"], right["values"]):
                self.assertAlmostEqual(x, y, delta=1e-7)

    def test_ragged_targets_weight_means_and_direct_path_matches_sealed_batches(self):
        direct, sealed = self.model.geometry_accumulator(), self.model.geometry_accumulator()
        losses, batches, counts = [], [], []
        for windows, shape in MICROBATCHES:
            gradients = self.model.geometry_loss_gradients(windows, shape)
            sealed.add(gradients)
            losses.append(direct.accumulate_loss(self.model, windows, shape))
            batches.append(gradients.parameter_gradients())
            counts.append(shape[0] * (shape[1] - 1))
        self.assertEqual(counts, [2, 6, 4])
        self.assertEqual((direct.microbatches, direct.target_tokens), (3, 12))
        self.assertEqual(direct.loss, sealed.loss)
        self.assertAlmostEqual(direct.loss, sum(x*n for x, n in zip(losses, counts)) / 12, places=12)
        self.assertEqual(direct.parameter_gradients(), sealed.parameter_gradients())
        for path, params in direct.parameter_gradients().items():
            for name, values in params.items():
                for i, value in enumerate(values):
                    expected = sum(batch[path][name][i]*n for batch, n in zip(batches, counts)) / 12
                    self.assertAlmostEqual(value, expected, delta=1e-7)
        detached = direct.parameter_gradients()
        detached.clear()
        self.assertTrue(direct.parameter_gradients())
        self.assertEqual(direct.geometry_artifact_id, self.model.geometry_artifact_id)
        self.assertEqual(direct.base_model_identity, self.base.base_model_identity)

    def test_empty_invalid_and_stale_inputs_do_not_modify_means(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError("token input must not be inspected before native admission")

        with self.assertRaises(ValueError):
            self.base.geometry_accumulator()
        state = self.model.geometry_accumulator()
        self.assertIsNone(state.loss)
        for action in (state.parameter_gradients, lambda: state.step_sgd(self.model, .1),
                       lambda: state.step_adamw(self.model, st.NativeGeometryAdamW(self.model.geometry_artifact()), .01)):
            with self.assertRaises(ValueError):
                action()
        state.accumulate_loss(self.model, *MICROBATCHES[0])
        before = state.loss, state.target_tokens, state.microbatches, state.parameter_gradients()
        for ids, shape in [([1, 11, 3], [1, 3]), ([1, 2], [1, 3]), ([True, 2, 3], [1, 3])]:
            with self.assertRaises((TypeError, ValueError)):
                state.accumulate_loss(self.model, ids, shape)
        limited = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=11).with_geometry(self.model.geometry_artifact())
        with self.assertRaises(ValueError):
            state.accumulate_loss(limited, Unreadable(), [1, 3])
        updated = state.step_sgd(self.model, .2)
        with self.assertRaises(ValueError):
            state.accumulate_loss(updated, Unreadable(), [1, 3])
        with self.assertRaises(ValueError):
            state.add(updated.geometry_loss_gradients(*MICROBATCHES[0]))
        with self.assertRaises(ValueError):
            state.step_sgd(updated, .2)
        with self.assertRaises(TypeError):
            state.add({})
        self.assertEqual(before, (state.loss, state.target_tokens, state.microbatches, state.parameter_gradients()))

    def test_one_adamw_step_and_resume_at_update_boundary(self):
        state = self.model.geometry_accumulator()
        for windows, shape in MICROBATCHES:
            state.accumulate_loss(self.model, windows, shape)
        optimizer = st.NativeGeometryAdamW(self.model.geometry_artifact(), max_gradient_norm=.0001)
        model, optimizer = state.step_adamw(self.model, optimizer, .01, weight_decay=.02)
        self.assertEqual(optimizer.revision, 1)
        self.assertEqual({s["steps"] for s in json.loads(optimizer.to_json())["slots"]}, {1})
        self.assertEqual(state.microbatches, 3)
        self.assertIsNone(self.base.geometry_artifact_id)
        saved = model.geometry_artifact()
        restored = self.base.with_geometry(st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, self.base.base_model_identity))
        resumed = st.NativeGeometryAdamW.from_json(optimizer.to_json(), optimizer.checkpoint_id, restored.geometry_artifact())
        a, b = model.geometry_accumulator(), restored.geometry_accumulator()
        for windows, shape in MICROBATCHES:
            a.accumulate_loss(model, windows, shape)
            b.accumulate_loss(restored, windows, shape)
        self.assertEqual(a.parameter_gradients(), b.parameter_gradients())
        left, left_opt = a.step_adamw(model, optimizer, .01)
        right, right_opt = b.step_adamw(restored, resumed, .01)
        self.assertEqual(left.geometry_artifact().to_json(), right.geometry_artifact().to_json())
        self.assertEqual(left_opt.to_json(), right_opt.to_json())
        with self.assertRaises((TypeError, ValueError)):
            a.step_adamw(model, optimizer, True)


if __name__ == "__main__":
    unittest.main()
