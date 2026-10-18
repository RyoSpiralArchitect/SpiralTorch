"""Native repetition learning: foreign ML imports are forbidden, not mocked."""
import copy
from collections import UserDict
import importlib.abc
import json
import math
from pathlib import Path
import runpy
import sys
from types import MappingProxyType
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
            raise AssertionError("native repetition imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
SOURCES = (
    {"kind": "prior_continuation", "ngram_order": 2},
    {"kind": "model_topk_history", "proposal_top_k": 11},
    {"kind": "model_topk_periodic", "proposal_top_k": 11},
)
WINDOWS = [1, 2, 1, 2, 1, 3, 4, 5, 6, 7, 8, 9]


def recipe(source=SOURCES[0], strength=.7):
    return {"kind": "native_v1", "config": {"strength": strength, "candidate_source": dict(source),
            "context_window": 16, "max_candidates_per_position": 3}}


class NativeRepetitionTests(unittest.TestCase):
    def setUp(self):
        config, weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(config, weights)
        self.model = self.base.with_geometry(fixtures["artifact"](self.base, trainable=True))

    def test_prior_and_periodic_losses_equal_explicit_probability_at_selected_row(self):
        ids = WINDOWS[:6]
        labels = [-100] * 5 + [3]
        values = self.model.forward(ids[:-1], [1, 5], last_only=True).values()
        maximum = max(values)
        probability = math.exp(values[2] - maximum) / sum(math.exp(v - maximum) for v in values)
        expected = -math.log1p(-probability)
        ce = self.model.next_token_loss(ids, [1, 6], labels=labels)
        for source in (SOURCES[0], SOURCES[2]):
            config = recipe(source)
            batch = self.model.geometry_loss_gradients(ids, [1, 6], labels=labels, repetition=config)
            self.assertEqual((batch.target_tokens, batch.repetition_positions), (1, 1))
            self.assertEqual(batch.causal_loss, ce)
            self.assertAlmostEqual(batch.repetition_loss, expected, delta=1e-7)
            self.assertAlmostEqual(batch.loss, ce + .7 * expected, delta=2e-7)
            self.assertEqual(batch.loss, self.model.next_token_loss(ids, [1, 6], labels=labels, repetition=config))
            self.assertEqual(json.loads(batch.repetition_json()), config)
            self.assertNotEqual(batch.parameter_gradients(),
                                self.model.geometry_loss_gradients(ids, [1, 6], labels=labels).parameter_gradients())

    def test_all_sources_reduce_separate_normalizers_across_masked_microbatches(self):
        labels = list(WINDOWS)
        labels[7:9] = [-100, -100]
        for source in SOURCES:
            for arithmetic in ("f32", "f64"):
                with self.subTest(source=source, arithmetic=arithmetic):
                    model = self.model.with_readout_accumulator(arithmetic)
                    config = recipe(source)
                    full = model.geometry_loss_gradients(WINDOWS, [2, 6], labels=labels,
                                                        label_smoothing=.1, repetition=config)
                    acc = model.geometry_accumulator(label_smoothing=.1, repetition=config)
                    self.assertIsNone(acc.loss)
                    for start in (0, 6):
                        acc.accumulate_loss(model, WINDOWS[start:start+6], [1, 6], labels=labels[start:start+6])
                    self.assertEqual(acc.target_tokens, 8)
                    self.assertEqual(acc.repetition_positions, full.repetition_positions)
                    self.assertGreater(acc.repetition_positions, 0)
                    self.assertAlmostEqual(acc.loss, full.loss, delta=3e-7)
                    self.assertAlmostEqual(acc.causal_loss, full.causal_loss, delta=3e-7)
                    self.assertAlmostEqual(acc.repetition_loss, full.repetition_loss, delta=3e-7)
                    self.assertEqual(json.loads(acc.repetition_json()), config)
                    for path, group in full.parameter_gradients().items():
                        for name, values in group.items():
                            for a, b in zip(values, acc.parameter_gradients()[path][name]):
                                self.assertAlmostEqual(a, b, delta=2e-7)
                    before = (acc.loss, acc.parameter_gradients(), acc.microbatches, acc.repetition_positions)
                    with self.assertRaises(ValueError):
                        acc.add(model.geometry_loss_gradients(WINDOWS, [2, 6], labels=labels, label_smoothing=.1))
                    self.assertEqual(before, (acc.loss, acc.parameter_gradients(), acc.microbatches, acc.repetition_positions))

    def test_zero_strength_is_exact_noop_and_empty_candidate_batch_stays_zero(self):
        expected = self.model.geometry_loss_gradients(WINDOWS, [2, 6])
        for source in SOURCES:
            config = recipe(source, 0.)
            actual = self.model.geometry_loss_gradients(WINDOWS, [2, 6], repetition=config)
            self.assertEqual(actual.loss.hex(), expected.loss.hex())
            self.assertEqual(actual.parameter_gradients(), expected.parameter_gradients())
            self.assertEqual((actual.repetition_loss, actual.repetition_positions), (0., 0))
        self.assertIsNone(json.loads(expected.repetition_json()))
        acc = self.model.geometry_accumulator(repetition=recipe())
        acc.accumulate_loss(self.model, [4, 5, 6], [1, 3])
        self.assertEqual((acc.repetition_loss, acc.repetition_positions), (0., 0))

    def test_two_active_microbatches_do_not_ce_weight_the_auxiliary_mean(self):
        rows = [WINDOWS[:6], [4, 5, 4, 6, 4, 7]]
        labels = [list(rows[0]), [-100, -100, -100, 6, 4, 7]]
        batches = [self.model.geometry_loss_gradients(ids, [1, 6], labels=mask, repetition=recipe())
                   for ids, mask in zip(rows, labels)]
        ce = [self.model.geometry_loss_gradients(ids, [1, 6], labels=mask) for ids, mask in zip(rows, labels)]
        self.assertEqual([b.target_tokens for b in batches], [5, 3])
        self.assertEqual([b.repetition_positions for b in batches], [1, 2])
        acc = self.model.geometry_accumulator(repetition=recipe())
        for b in batches:
            acc.add(b)
        expected = (5*ce[0].loss + 3*ce[1].loss)/8 + .7*(batches[0].repetition_loss + 2*batches[1].repetition_loss)/3
        self.assertAlmostEqual(acc.loss, expected, delta=1e-7)
        wrong = (5*batches[0].loss + 3*batches[1].loss)/8
        self.assertGreater(abs(acc.loss - wrong), 1e-6)
        for path, group in acc.parameter_gradients().items():
            for name, values in group.items():
                for i, actual in enumerate(values):
                    a, b = (g.parameter_gradients()[path][name][i] for g in batches)
                    ca, cb = (g.parameter_gradients()[path][name][i] for g in ce)
                    self.assertAlmostEqual(actual, (5*ca + 3*cb)/8 + ((a-ca) + 2*(b-cb))/3, delta=1e-7)

    def test_recipe_admission_precedes_token_extraction_and_bad_labels_are_atomic(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError("read tokens before admission")
        invalid = [recipe(strength=-1.), recipe(strength=float("nan")), recipe(strength=True),
                   recipe({"kind": "model_topk_history", "proposal_top_k": 12}),
                   {"kind": "native_v2", "config": recipe()["config"]},
                   {**recipe(), "unrecognized": True}, UserDict(recipe()), MappingProxyType(recipe())]
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises((ValueError, TypeError)):
                    self.model.next_token_loss(Unreadable(), [1, 6], repetition=config)
                with self.assertRaises((ValueError, TypeError)):
                    self.model.geometry_loss_gradients(Unreadable(), [1, 6], repetition=config)
                with self.assertRaises((ValueError, TypeError)):
                    self.model.geometry_accumulator(repetition=config)
        acc = self.model.geometry_accumulator(repetition=recipe())
        acc.accumulate_loss(self.model, WINDOWS[:6], [1, 6])
        before = (acc.loss, acc.parameter_gradients(), acc.repetition_positions, acc.microbatches)
        bad = WINDOWS[:6]
        bad[-1] = 10
        with self.assertRaisesRegex(ValueError, "labels must equal"):
            acc.accumulate_loss(self.model, WINDOWS[:6], [1, 6], labels=bad)
        self.assertEqual(before, (acc.loss, acc.parameter_gradients(), acc.repetition_positions, acc.microbatches))

    def test_trainer_recipe_is_owned_and_restores_identical_updates(self):
        for source in SOURCES:
            config = recipe(source)
            docs = [WINDOWS]
            optimizer = st.NativeGeometryAdamW(self.model.geometry_artifact(), max_gradient_norm=.1)
            trainer = st.NativeCausalLmTrainer(self.model, optimizer, docs, 6, learning_rate=.01,
                                             accumulation_steps=2, epochs=3, repetition=config)
            saved_config = copy.deepcopy(config)
            config["config"]["strength"] = 3.
            self.assertEqual(json.loads(trainer.config_json())["repetition"], saved_config)
            first = trainer.step()
            self.assertGreater(first.repetition_positions, 0)
            self.assertGreater(first.repetition_loss, 0.)
            self.assertEqual(json.loads(first.to_json())["repetition_positions"], first.repetition_positions)
            restored = st.NativeCausalLmTrainer.from_json(self.base, docs, trainer.to_json(), trainer.checkpoint_id)
            while not trainer.exhausted:
                self.assertEqual(trainer.step().to_json(), restored.step().to_json())
                self.assertEqual(trainer.to_json(), restored.to_json())
            self.assertEqual(trainer.model.base_model_identity, self.base.base_model_identity)
            self.assertNotEqual(trainer.model.geometry_artifact_id, self.model.geometry_artifact_id)
            bad = [list(WINDOWS)]
            bad[0][5] = 10
            with self.assertRaisesRegex(ValueError, "labels must equal"):
                st.NativeCausalLmTrainer(self.model, optimizer, docs, 6, labels=bad,
                                        learning_rate=.01, repetition=saved_config)

    def test_later_work_over_budget_batch_is_rejected_during_construction(self):
        config, weights = fixtures["fixture"](positions=1024)
        base = st.NativeCausalLm.from_gpt2(config, weights)
        state = json.loads(fixtures["artifact"](base, trainable=True).state_json())
        state["placements"][0]["recipe"]["max_values"] = 4096
        model = base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
        optimizer = st.NativeGeometryAdamW(model.geometry_artifact())
        objective = recipe(SOURCES[2])
        objective["config"]["context_window"] = 1024
        sparse = [-100] * 1023 + [1]
        st.NativeCausalLmTrainer(model, optimizer, [[1] * 1024], 1024, labels=[sparse],
                                learning_rate=.01, repetition=objective)
        with self.assertRaisesRegex(ValueError, "work"):
            st.NativeCausalLmTrainer(model, optimizer, [[1] * 1024] * 2, 1024,
                                    labels=[sparse, [1] * 1024], learning_rate=.01, repetition=objective)


if __name__ == "__main__":
    unittest.main()
