"""Owned native KV + causal-geometry context with no Torch/NumPy runtime imports."""
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "safetensors", "accelerate"}:
            raise AssertionError("native context imported " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
geometry = runpy.run_path(str(Path(__file__).with_name("test_native_geometry_runtime.py")))
KINDS = (None, "affine", "wave", "fixed", "learned", "elliptic", "anchored", "causal", "gated", "topos", *geometry["FRACTIONAL_KINDS"])


def pair(kind=None, path="transformer.h.0.attn", **limits):
    config, weights = fixtures["fixture"](width=3, heads=1)
    base = st.NativeCausalLm.from_gpt2(config, weights, **limits)
    if kind is None:
        return base, base
    record = json.loads(geometry["artifact"](kind).state_json())
    record["base_model_identity"] = base.base_model_identity
    record["placements"][0]["path"] = path
    if kind == "affine":
        record["placements"][0]["recipe"]["max_values"] = 96
    return base, base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(record)))


class NativeCausalLmContextTests(unittest.TestCase):
    def test_cached_line_lengths_precede_token_reads_and_refusals_preserve_root(self):
        _, model = pair("wave")
        ctx = st.NativeCausalLmContext(model, [1, 2], max_cache_values=1024)
        start = ctx.logits().values()
        class Oversize:
            def __len__(self):
                return 4096
            def __getitem__(self, i):
                raise AssertionError("unadmitted tokens must not be accessed")
        for lines, stop in [([Oversize()], 0), ([[1], [1]], 0), ([[1, True]], 0), ([[1]], False), ([[0]], 0)]:
            with self.assertRaises((ValueError, TypeError)):
                ctx.score_lines(lines, stop_id=stop)
        report = ctx.score_lines([[1], [1, 2]], stop_id=0)
        self.assertEqual(report["scores"]["target_counts"], [2, 3])
        self.assertEqual(report["decoded_content_tokens"], 2)
        self.assertEqual(report["decoder_chunks"], 1)
        self.assertEqual(ctx.token_ids, [1, 2])
        self.assertEqual(ctx.logits().values(), start)

    def test_line_tiles_validate_before_foreign_tokens_and_match_full_vocabulary(self):
        _, model = pair("history_gain", max_logit_elements=88)
        ctx = st.NativeCausalLmContext(model, [1,2], max_cache_values=1024)
        class Untouchable:
            def __len__(self):
                raise AssertionError("invalid tile must be refused before examining menu")
        for cap in (0, 10, 89, True, 11.5):
            with self.assertRaises((ValueError, TypeError)):
                ctx.score_lines(Untouchable(), stop_id=0, max_tile_values=cap)
        lines = [[1], [1,2,3], [1,2,4], [1,5], [7,8]]
        a = ctx.score_lines(lines, stop_id=0)
        b = ctx.score_lines(lines, stop_id=0, max_tile_values=11)
        self.assertEqual((a["readout_batches"], b["readout_batches"]), (6, 7))
        self.assertEqual((a["peak_projected_logit_values"], b["peak_projected_logit_values"]), (22, 11))
        self.assertEqual((b["scanned_logit_values"], b["retained_edge_logits"]), (88,12))
        self.close(a["scores"]["line_nlls"], b["scores"]["line_nlls"])
        self.assertEqual(a["scores"]["winner_index"], b["scores"]["winner_index"])
        with self.assertRaises(ValueError):
            ctx.score_lines(lines+[[9]], stop_id=0, max_tile_values=11)
        self.assertEqual(ctx.token_ids,[1,2])

    def close(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a, b, delta=3e-6)

    def test_all_recipes_and_sites_match_full_prefix_outputs(self):
        for kind in KINDS:
            for path in ("transformer.h.0.attn", "transformer.h.0.mlp", "transformer.h.0"):
                with self.subTest(kind=kind, path=path):
                    _, model = pair(kind, path)
                    context = st.NativeCausalLmContext(model, [1, 2], max_cache_values=1024)
                    self.assertEqual(context.geometry_artifact_id, model.geometry_artifact_id)
                    self.assertEqual(context.base_model_identity, model.base_model_identity)
                    fork = context.with_suffix([3, 4, 5])
                    tail = fork.with_suffix([6, 7])
                    self.assertEqual(context.token_ids, [1, 2])
                    self.assertEqual(fork.token_ids, [1, 2, 3, 4, 5])
                    self.assertEqual(tail.geometry_artifact_id, context.geometry_artifact_id)
                    for token in (3, 4, 5, 6, 7):
                        ids = context.token_ids
                        self.close(context.logits().values(), model.forward(ids, [1, len(ids)], last_only=True).values())
                        result = context.append(token)
                        if token in (5, 7):
                            child = fork if token == 5 else tail
                            self.assertEqual(child.cache_values, context.cache_values)
                            self.close(result.values(), child.logits().values())
                        self.assertEqual(result.values(), context.logits().values())
                        history = 1 if kind in ("causal", "gated", *geometry["FRACTIONAL_KINDS"]) else 0
                        self.assertEqual(context.cache_values, (4 + history) * len(context.token_ids) * 3)
                    self.close(context.logits().values(), model.forward(context.token_ids, [1, 7], last_only=True).values())
                    self.assertEqual(context.execution_backend, "rust_f32_cpu_prepacked_kv")
        self.assertIn("NativeCausalLmContext", st.__all__)

    def test_owned_input_and_failure_atomicity(self):
        _, model = pair()
        prompt = [1, 2]
        context = st.NativeCausalLmContext(model, prompt, max_cache_values=36)
        prompt[0] = 5
        self.assertEqual(context.token_ids, [1, 2])
        before = context.logits().values()
        for token in (True, -1, 11):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                context.append(token)
            self.assertEqual(context.token_ids, [1, 2])
            self.assertEqual(context.logits().values(), before)
        context.append(3)
        before = context.logits().values()
        with self.assertRaisesRegex(ValueError, "cache"):
            context.append(4)
        self.assertEqual(context.token_ids, [1, 2, 3])
        self.assertEqual(context.logits().values(), before)

    def test_admission_precedes_value_extraction(self):
        _, model = pair()
        class Unreadable:
            def __len__(self):
                return 2
            def __getitem__(self, key):
                raise AssertionError("over-budget prompt accessed")
        for budget in (True, 0, 1):
            with self.assertRaises((ValueError, TypeError)):
                st.NativeCausalLmContext(model, Unreadable(), max_cache_values=budget)
        for config in ({"kind": "kv_cache", "max_values": 1}, {"kind": "kv_cache", "max_values": True},
                       {"kind": "kv_cache"}, {"kind": "torch"}):
            with self.assertRaises(ValueError):
                st.NativeCausalLmGenerator(model, Unreadable(), json.dumps({"max_new_tokens": 5, "execution": config}))

    def test_cached_attention_has_linear_admission_not_full_prefix_fallback(self):
        _, small = pair(max_attention_pairs=6)
        _, full = pair()
        context = st.NativeCausalLmContext(small, [1])
        context.append(2)
        actual = context.append(3)
        with self.assertRaisesRegex(ValueError, "attention"):
            small.forward([1, 2, 3], [1, 3], last_only=True)
        self.close(actual.values(), full.forward([1, 2, 3], [1, 3], last_only=True).values())
        with self.assertRaisesRegex(ValueError, "attention"):
            context.append(4)

    def test_suffix_admission_precedes_extraction_and_failure_keeps_fork_usable(self):
        _, m = pair(max_attention_pairs=12)
        root = st.NativeCausalLmContext(m, [1, 2])
        class Unreadable:
            def __len__(self): return 2
            def __getitem__(self, key): raise AssertionError("over-budget suffix read")
        with self.assertRaisesRegex(ValueError, "attention"):
            root.with_suffix(Unreadable())
        for bad in ([], [True], [11]):
            with self.assertRaises((ValueError, TypeError)):
                root.with_suffix(bad)
        fork = root.with_suffix([4])
        self.assertEqual(fork.token_ids, [1, 2, 4])
        self.assertEqual(root.token_ids, [1, 2])

    def test_every_recipe_cached_generation_resumes_exactly(self):
        for kind in KINDS:
            for control in (False, True):
                with self.subTest(kind=kind, control=control):
                    _, model = pair(kind)
                    config = {"max_new_tokens": 5, "seed": 17, "execution": {"kind": "kv_cache", "max_values": 1024},
                              "decoding": {"selection": "sample", "top_k": 7, "top_p": .8}}
                    if control:
                        config["decoding"]["distribution"] = {"kind": "zspace", "config": {"curvature": -1.}}
                    generator = st.NativeCausalLmGenerator(model, [1, 2], json.dumps(config))
                    generator.step()
                    generator.step()
                    restored = st.NativeCausalLmGenerator.from_json(model, generator.to_json(), generator.checkpoint_id)
                    while not generator.finished:
                        self.assertEqual(generator.step().to_json(), restored.step().to_json())
                        self.assertEqual(generator.to_json(), restored.to_json())

    def test_failed_cached_control_leaves_history_and_checkpoint_unchanged(self):
        _, model = pair()
        generator = st.NativeCausalLmGenerator(model, [1, 1], json.dumps({"max_new_tokens": 3,
            "execution": {"kind": "kv_cache", "max_values": 1024}, "decoding": {"selection": "sample",
            "distribution": {"kind": "zspace", "config": {"repression_strength": sys.float_info.max}}}}))
        before = generator.to_json()
        for _ in range(2):
            with self.assertRaises(ValueError):
                generator.step()
            self.assertEqual(generator.to_json(), before)

    def test_full_prefix_v1_default_serialization_does_not_gain_execution_fields(self):
        _, model = pair()
        generator = st.NativeCausalLmGenerator(model, [1, 2], '{"max_new_tokens": 2}')
        self.assertNotIn("execution", json.loads(generator.config_json()))
        self.assertNotIn("execution", json.loads(generator.to_json())["config"])


if __name__ == "__main__":
    unittest.main()
