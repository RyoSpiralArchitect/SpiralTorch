"""Masked/smoothed causal learning without Torch, including group continuation."""
import importlib.abc
import json
import math
from pathlib import Path
import runpy
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "accelerate", "safetensors"}:
            raise AssertionError("native objective must not import " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
BATCHES = [
    ([1, 2, 3], [1, 3], [-100, -100, 5]),
    ([4, 5, 6, 7, 2, 4, 6, 8], [2, 4], [9, 2, -100, 3, -100, -100, -100, -100]),
    ([3, 2, 1, 4, 5], [1, 5], [-100, -100, -100, 8, 6]),
]


class NativeCausalLmObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(self.config, self.weights)
        self.model = self.base.with_geometry(fixtures["artifact"](self.base, trainable=True))

    def test_explicit_loss_and_default_equivalence(self):
        ids, shape, labels = BATCHES[1]
        logits = self.model.forward(ids[:3] + ids[4:7], [2, 3]).values()
        shifted = labels[1:4] + labels[5:8]
        for smoothing in (0., .2, 1.):
            losses = []
            for i, target in enumerate(shifted):
                if target == -100:
                    continue
                row = logits[i*11:(i+1)*11]
                maximum = max(row)
                log_z = maximum + math.log(math.fsum(math.exp(x-maximum) for x in row))
                losses.append(log_z - (1-smoothing)*row[target] - smoothing*math.fsum(row)/11)
            expected = math.fsum(losses) / len(losses)
            batch = self.model.geometry_loss_gradients(ids, shape, labels=labels, label_smoothing=smoothing)
            self.assertAlmostEqual(batch.loss, expected, delta=2e-6)
            self.assertEqual(batch.target_tokens, 2)
            self.assertEqual(batch.label_smoothing, smoothing)
            self.assertEqual(batch.loss, self.model.next_token_loss(ids, shape, labels=labels, label_smoothing=smoothing))
        a = self.model.geometry_loss_gradients(ids, shape)
        b = self.model.geometry_loss_gradients(ids, shape, labels=ids, label_smoothing=0.)
        self.assertEqual(a.loss, b.loss)
        self.assertEqual(a.parameter_gradients(), b.parameter_gradients())

    def test_active_targets_weight_loss_and_gradient(self):
        mean = self.model.geometry_accumulator(label_smoothing=.2)
        other = self.model.geometry_accumulator(label_smoothing=.2)
        gradients = []
        for ids, shape, labels in BATCHES:
            g = self.model.geometry_loss_gradients(ids, shape, labels=labels, label_smoothing=.2)
            mean.add(g)
            other.accumulate_loss(self.model, ids, shape, labels=labels)
            gradients.append(g)
        self.assertEqual([g.target_tokens for g in gradients], [1, 2, 2])
        self.assertEqual((mean.microbatches, mean.target_tokens, mean.label_smoothing), (3, 5, .2))
        self.assertEqual(mean.loss, other.loss)
        self.assertEqual(mean.parameter_gradients(), other.parameter_gradients())
        self.assertAlmostEqual(mean.loss, sum(g.loss*g.target_tokens for g in gradients)/5, places=12)
        for path, parameters in mean.parameter_gradients().items():
            for name, values in parameters.items():
                for i, value in enumerate(values):
                    expected = sum(g.parameter_gradients()[path][name][i]*g.target_tokens for g in gradients)/5
                    self.assertAlmostEqual(value, expected, delta=1e-7)

    def test_invalid_labels_objectives_and_budgets(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError("foreign IDs/labels touched before admission")

        for smoothing in (True, -.1, 1.1, math.nan, math.inf):
            for call in (lambda: self.model.geometry_accumulator(label_smoothing=smoothing),
                         lambda: self.model.next_token_loss(Unreadable(), [1, 3], label_smoothing=smoothing),
                         lambda: self.model.geometry_loss_gradients(Unreadable(), [1, 3], label_smoothing=smoothing)):
                with self.assertRaises((TypeError, ValueError)):
                    call()
        for labels in ([-100]*3, [1, -100, -100], [1, 2], [1, -99, 3], [11, 2, 3], [1, 2, 11], [True, 2, 3], [1, 2., 3]):
            for call in (self.model.next_token_loss, self.model.geometry_loss_gradients):
                with self.assertRaises((TypeError, ValueError)):
                    call([1, 2, 3], [1, 3], labels=labels)
        limited = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=11).with_geometry(self.model.geometry_artifact())
        for call in (limited.next_token_loss, limited.geometry_loss_gradients):
            with self.assertRaises(ValueError):
                call(Unreadable(), [1, 3], labels=Unreadable())
        mean = limited.geometry_accumulator()
        with self.assertRaises(ValueError):
            mean.accumulate_loss(limited, Unreadable(), [1, 3], labels=Unreadable())

    def test_objective_mismatch_and_empty_selection_do_not_change_mean(self):
        mean = self.model.geometry_accumulator(label_smoothing=.2)
        ids, shape, labels = BATCHES[0]
        mean.accumulate_loss(self.model, ids, shape, labels=labels)
        before = mean.loss, mean.target_tokens, mean.microbatches, mean.parameter_gradients()
        with self.assertRaises(ValueError):
            mean.add(self.model.geometry_loss_gradients(ids, shape, labels=labels))
        for bad in ([-100]*3, [1, -99, 3], [1, 2]):
            with self.assertRaises(ValueError):
                mean.accumulate_loss(self.model, ids, shape, labels=bad)
        self.assertEqual(before, (mean.loss, mean.target_tokens, mean.microbatches, mean.parameter_gradients()))

    def test_masked_group_adamw_continues_exactly_with_same_objective(self):
        def accumulate(model):
            mean = model.geometry_accumulator(label_smoothing=.2)
            for ids, shape, labels in BATCHES:
                mean.accumulate_loss(model, ids, shape, labels=labels)
            return mean
        optimizer = st.NativeGeometryAdamW(self.model.geometry_artifact(), max_gradient_norm=.001)
        mean = accumulate(self.model)
        model, optimizer = mean.step_adamw(self.model, optimizer, .01)
        saved = model.geometry_artifact()
        restored_artifact = st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, self.base.base_model_identity)
        restored = self.base.with_geometry(restored_artifact)
        resumed = st.NativeGeometryAdamW.from_json(optimizer.to_json(), optimizer.checkpoint_id, restored_artifact)
        a, b = accumulate(model), accumulate(restored)
        self.assertEqual(a.loss, b.loss)
        self.assertEqual(a.parameter_gradients(), b.parameter_gradients())
        left, lo = a.step_adamw(model, optimizer, .01)
        right, ro = b.step_adamw(restored, resumed, .01)
        self.assertEqual(left.geometry_artifact().to_json(), right.geometry_artifact().to_json())
        self.assertEqual(lo.to_json(), ro.to_json())
        self.assertEqual({s["steps"] for s in json.loads(lo.to_json())["slots"]}, {2})
        self.assertIsNone(self.base.geometry_artifact_id)


if __name__ == "__main__":
    unittest.main()
