"""Whole-corpus native scoring; foreign ML imports forbidden, including during FT."""
import ast
import contextlib
import importlib
import importlib.abc
import io
import json
import math
from pathlib import Path
import runpy
import struct
import sys
import tempfile
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "peft", "accelerate", "safetensors"}:
            raise AssertionError("native evaluation imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
DOCS = [[], [1], [2, 3], [1, 2, 3, 4, 5, 6, 7, 8], [4, 3, 2, 1]]
SPANS = [(2, 0, 2), (3, 0, 4), (3, 3, 7), (3, 6, 8), (4, 0, 4)]


class NativeCausalLmEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixtures["fixture"]()
        self.base = st.NativeCausalLm.from_gpt2(self.config, self.weights)
        self.model = self.base.with_geometry(fixtures["artifact"](self.base, trainable=True))

    def test_all_targets_tails_and_document_boundaries_match_scalar_nll(self):
        scores = []
        for model in (self.base, self.model):
            terms = []
            for doc, start, end in SPANS:
                ids = DOCS[doc][start:end]
                logits = model.forward(ids[:-1], [1, len(ids)-1]).values()
                for i, target in enumerate(ids[1:]):
                    row = logits[i*11:(i+1)*11]
                    maximum = max(row)
                    terms.append(maximum - row[target] + math.log(math.fsum(math.exp(v-maximum) for v in row)))
            report = model.evaluate_documents(DOCS, 4)
            self.assertEqual((report.target_tokens, report.window_count, report.ignored_targets), (11, 5, 0))
            self.assertAlmostEqual(report.nll, math.fsum(terms)/len(terms), delta=3e-6)
            self.assertAlmostEqual(report.perplexity, math.exp(report.nll), places=12)
            row = json.loads(report.to_json())
            self.assertEqual(row["geometry_artifact_id"], model.geometry_artifact_id)
            for batch in (2, 8):
                grouped = model.evaluate_documents(DOCS, 4, microbatch_windows=batch)
                self.assertAlmostEqual(grouped.nll, report.nll, delta=3e-6)
                self.assertEqual(grouped.data_id, report.data_id)
            scores.append(report.nll)
        self.assertNotEqual(*scores)

    def test_selected_targets_are_weighted_not_batch_means_and_zero_windows_skip(self):
        labels = [[-100]*len(d) for d in DOCS]
        labels[3][4:7] = [1, 9, 0]
        labels[4][2] = 7
        score = self.model.evaluate_documents(DOCS, 4, labels=labels, microbatch_windows=2)
        a = self.model.next_token_loss(DOCS[3][3:7], [1, 4], labels=labels[3][3:7])
        b = self.model.next_token_loss(DOCS[4], [1, 4], labels=labels[4])
        self.assertAlmostEqual(score.nll, (3*a+b)/4, delta=3e-6)
        self.assertGreater(abs(score.nll-(a+b)/2), 1e-4)
        self.assertEqual((score.target_tokens, score.ignored_targets, score.skipped_windows), (4, 7, 3))
        labels[1][0] = 4
        other = self.model.evaluate_documents(DOCS, 4, labels=labels, microbatch_windows=2)
        self.assertEqual(other.nll, score.nll)
        self.assertNotEqual(other.data_id, score.data_id)
        labels[1][0] = 99
        with self.assertRaisesRegex(ValueError, "labels"):
            self.model.evaluate_documents(DOCS, 4, labels=labels)
        with self.assertRaisesRegex(ValueError, "no selected"):
            self.model.evaluate_documents(DOCS, 4, labels=[[-100]*len(d) for d in DOCS])

    def test_validation_precedes_token_extraction_and_checks_every_unused_value(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError("data touched before config validation")
        class Sized:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
            def __getitem__(self, _):
                raise AssertionError("data touched before length admission")
        for context, batch in ((1, 1), (4, 0), (True, 1), (4, True)):
            with self.assertRaises((ValueError, TypeError)):
                self.model.evaluate_documents(Unreadable(), context, microbatch_windows=batch)
        with self.assertRaisesRegex(ValueError, "document count"):
            self.model.evaluate_documents(Sized(1_048_577), 4)
        with self.assertRaisesRegex(ValueError, "token count"):
            self.model.evaluate_documents([Sized(16_777_217)], 4)
        with self.assertRaisesRegex(ValueError, "position"):
            self.model.evaluate_documents([Sized(4)], 10)
        for docs in ([[], [1]], [[True, 2]], [[1, 11]], [[99], [1, 2]]):
            with self.assertRaises((ValueError, TypeError)):
                self.model.evaluate_documents(docs, 4)
        for labels in ([], [[1]], [[True, 2]], [[-1, 2]]):
            with self.assertRaises((ValueError, TypeError)):
                self.model.evaluate_documents([[1, 2]], 4, labels=labels)

    def test_selected_readout_budget_and_checked_index_transport(self):
        limited = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=11)
        ids = [[1, 2, 3, 4]]
        labels = [[-100, -100, -100, 2]]
        self.assertEqual(limited.evaluate_documents(ids, 4, labels=labels).target_tokens, 1)
        with self.assertRaisesRegex(ValueError, "logits"):
            limited.evaluate_documents(ids, 4)
        labels[0][2] = 3
        with self.assertRaisesRegex(ValueError, "logits"):
            limited.evaluate_documents(ids, 4, labels=labels)
        class Indexed:
            def __init__(self, values):
                self.values = values
            def __len__(self):
                return len(self.values)
            def __getitem__(self, index):
                assert 0 <= index < len(self.values)
                return self.values[index]
            def __iter__(self):
                raise AssertionError("unbounded iteration")
        expected = self.model.evaluate_documents(DOCS, 4)
        observed = self.model.evaluate_documents(Indexed([Indexed(d) for d in DOCS]), 4)
        self.assertEqual(observed.to_json(), expected.to_json())

    def test_later_batch_activation_attention_and_geometry_budgets_precede_ids(self):
        class UnreadIds:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
            def __getitem__(self, _):
                raise AssertionError("IDs copied before admitting every batch")
        for limits in ({"max_activation_elements": 20}, {"max_attention_pairs": 15}):
            limited = st.NativeCausalLm.from_gpt2(self.config, self.weights, **limits)
            with self.assertRaises(ValueError):
                limited.evaluate_documents([UnreadIds(2), UnreadIds(4)], 4)
        record = json.loads(self.model.geometry_artifact().state_json())
        record["placements"][0]["recipe"]["max_values"] = 8
        limited = self.base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record)))
        with self.assertRaises(ValueError):
            limited.evaluate_documents([UnreadIds(2), UnreadIds(4)], 4)

    def test_eval_during_training_is_read_only_and_does_not_change_exact_resume(self):
        def fresh():
            return st.NativeCausalLmTrainer(self.model, st.NativeGeometryAdamW(self.model.geometry_artifact()),
                [[1, 2, 3, 4], [2, 3, 4, 5]], 4, learning_rate=.01, epochs=2, label_smoothing=.2)
        reference, observed = fresh(), fresh()
        initial_score = observed.model.evaluate_documents(DOCS, 4).nll
        while not observed.exhausted:
            before = observed.to_json()
            score = observed.model.evaluate_documents(DOCS, 4)
            self.assertEqual(observed.to_json(), before)
            self.assertEqual(score.to_json(), observed.model.evaluate_documents(DOCS, 4).to_json())
            self.assertEqual(observed.step().to_json(), reference.step().to_json())
            self.assertEqual(observed.to_json(), reference.to_json())
            observed = st.NativeCausalLmTrainer.from_json(self.base, [[1, 2, 3, 4], [2, 3, 4, 5]],
                                                        observed.to_json(), observed.checkpoint_id)
        self.assertNotEqual(initial_score, observed.model.evaluate_documents(DOCS, 4).nll)

    def test_result_export_stub_and_immutability(self):
        self.assertIn("NativeCausalLmEvaluation", st.__all__)
        stub = ast.parse(Path(st.__file__).with_name("__init__.pyi").read_text())
        classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
        methods = {node.name for node in classes["NativeCausalLm"].body if isinstance(node, ast.FunctionDef)}
        self.assertIn("evaluate_documents", methods)
        self.assertIn("NativeCausalLmEvaluation", classes)
        with self.assertRaises(TypeError):
            st.NativeCausalLmEvaluation()
        result = self.model.evaluate_documents(DOCS, 4)
        with self.assertRaises(AttributeError):
            result.nll = 0.
        with self.assertRaises(AttributeError):
            result.readout_accumulator = "f64"

    def test_repetition_training_keeps_evaluation_pure_nll_and_readout_policy(self):
        docs = [[1, 2, 1, 2, 1, 3], [4, 5, 4, 6, 4, 7]]
        repetition = {"kind": "native_v1", "config": {"strength": .7,
            "candidate_source": {"kind": "prior_continuation", "ngram_order": 2},
            "context_window": 16, "max_candidates_per_position": 3}}
        data_ids = []
        for arithmetic in ("f32", "f64"):
            with self.subTest(arithmetic=arithmetic):
                base = self.base.with_readout_accumulator(arithmetic)
                model = self.model.with_readout_accumulator(arithmetic)

                def fresh():
                    return st.NativeCausalLmTrainer(model,
                        st.NativeGeometryAdamW(model.geometry_artifact(), max_gradient_norm=.1),
                        docs, 6, learning_rate=.01, microbatch_windows=2, epochs=3,
                        label_smoothing=.2, repetition=repetition)

                reference, observed = fresh(), fresh()
                while not observed.exhausted:
                    before = observed.to_json()
                    score = observed.model.evaluate_documents(docs, 6, microbatch_windows=2)
                    plain = observed.model.next_token_loss(docs[0] + docs[1], [2, 6])
                    self.assertEqual(score.nll, plain)
                    self.assertEqual(score.readout_accumulator, arithmetic)
                    self.assertEqual(json.loads(score.to_json())["readout_accumulator"], arithmetic)
                    self.assertEqual(observed.to_json(), before)
                    step = observed.step()
                    self.assertGreater(step.repetition_loss, 0.)
                    self.assertGreater(step.repetition_positions, 0)
                    self.assertEqual(step.to_json(), reference.step().to_json())
                    self.assertEqual(observed.to_json(), reference.to_json())
                    observed = st.NativeCausalLmTrainer.from_json(base, docs, observed.to_json(),
                                                                observed.checkpoint_id)
                data_ids.append(score.data_id)
                # Custom evaluation targets remain legal despite the training
                # objective requiring labels to match the original token IDs.
                labels = [[-100] * 5 + [9], [-100] * 5 + [10]]
                selected = observed.model.evaluate_documents(docs, 6, labels=labels, microbatch_windows=2)
                expected = observed.model.next_token_loss(docs[0] + docs[1], [2, 6],
                                                          labels=labels[0] + labels[1])
                self.assertEqual(selected.nll, expected)
                self.assertEqual(selected.target_tokens, 2)
        self.assertEqual(*data_ids)

    def test_packaged_periodic_evaluation_preserves_training_and_resumed_checkpoints(self):
        repetition = {"kind": "native_v1", "config": {"strength": .7,
            "candidate_source": {"kind": "model_topk_history", "proposal_top_k": 11},
            "context_window": 16, "max_candidates_per_position": 3}}
        for objective in (None, repetition):
            with self.subTest(repetition=objective is not None):
                self._check_packaged_periodic_evaluation(objective)

    def _check_packaged_periodic_evaluation(self, repetition):
        example = importlib.import_module("examples.native_geometry_corpus_training")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            header, payload = {}, bytearray()
            for name in self.weights.names():
                tensor = self.weights.tensor(name)
                values = tensor.values()
                start = len(payload)
                payload.extend(struct.pack(f"<{len(values)}f", *values))
                header[name] = {"dtype": "F32", "shape": tensor.shape, "data_offsets": [start, len(payload)]}
            raw = json.dumps(header).encode()
            model_path = model_dir / "model.safetensors"
            model_path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
            (model_dir / "config.json").write_text(self.config)
            base = st.NativeCausalLm.from_gpt2(self.config, st.SafeTensorArchive(model_path))
            artifact = fixtures["artifact"](base, trainable=True)
            geometry = root / "geometry.json"
            geometry.write_text(artifact.to_json())
            data = root / "data.json"
            data.write_text(json.dumps({"token_documents": [[1, 2, 3, 4], [2, 3, 4, 5]]}))
            heldout = root / "heldout.json"
            heldout.write_text(json.dumps({"token_documents": DOCS}))
            recipe = root / "recipe.json"
            settings = dict(context_length=4, microbatch_windows=1, accumulation_steps=1,
                epochs=2, learning_rate=.01, weight_decay=0., label_smoothing=.2, max_gradient_norm=.1)
            if repetition is not None:
                settings["repetition"] = repetition
            recipe.write_text(json.dumps(settings))
            common = ["--model-dir", str(model_dir), "--data-json", str(data), "--save-every", "1"]
            fresh = ["--recipe-json", str(recipe), "--artifact", str(geometry), "--artifact-id", artifact.artifact_id]
            evaluation = ["--eval-data-json", str(heldout), "--eval-every", "1"]
            runs = (("plain", []), ("scored", evaluation),
                    ("interval", [*evaluation, "--eval-every", "3"]),
                    ("default", ["--eval-data-json", str(heldout), "--save-every", "2"]),
                    ("zero", [*evaluation, "--stop-after", "0"]),
                    ("part", [*evaluation, "--stop-after", "2"]))
            for name, extra in runs:
                with contextlib.redirect_stdout(io.StringIO()):
                    example.main([*common, *fresh, *extra, "--output", str(root / name)])
            part = json.loads((root / "part/result.json").read_text())
            pin = part["checkpoints"][-1]["checkpoint_id"]
            with contextlib.redirect_stdout(io.StringIO()):
                example.main([*common, *evaluation, "--resume", str(root / "part/checkpoint-000002.json"),
                              "--checkpoint-id", pin, "--eval-every", "3", "--output", str(root / "resumed")])
            expected = (root / "plain/checkpoint-000004.json").read_bytes()
            if repetition is not None:
                training = [json.loads(row) for row in (root / "plain/steps.jsonl").read_text().splitlines()]
                self.assertEqual(len(training), 4)
                self.assertTrue(all(row["repetition_positions"] > 0 and row["repetition_loss"] > 0.
                                    for row in training))
            for name in ("scored", "interval", "default", "resumed"):
                self.assertEqual((root / name / "checkpoint-000004.json").read_bytes(), expected)
            for name, steps in (("scored", [0, 1, 2, 3, 4]), ("interval", [0, 3, 4]),
                                ("default", [0, 2, 4]), ("resumed", [2, 3, 4]), ("zero", [0])):
                rows = [json.loads(line) for line in (root / name / "evaluation.jsonl").read_text().splitlines()]
                self.assertEqual([row["completed_updates"] for row in rows], steps)
                self.assertTrue(all(row["evaluation"]["target_tokens"] == 11 for row in rows))
            heldout.write_text(json.dumps({"token_documents": DOCS, "labels": [[-100]*len(d) for d in DOCS]}))
            with self.assertRaisesRegex(ValueError, "no selected"):
                example.main([*common, *fresh, *evaluation, "--output", str(root / "rejected")])
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
