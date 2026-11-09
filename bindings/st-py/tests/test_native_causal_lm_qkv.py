"""Real native QKV updates without foreign ML packages or array libraries."""
import ast
import importlib.abc
import inspect
import json
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class NoForeignML(importlib.abc.MetaPathFinder):
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "safetensors", "accelerate"}:
            self.attempts.append(fullname)
            raise AssertionError("QKV path must not import " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st
from test_native_causal_lm import fixture, artifact

if __name__ == "__main__" and NoForeignML.attempts:
    raise AssertionError("optional import swallowed: " + repr(NoForeignML.attempts))


class NativeQkvTests(unittest.TestCase):
    def setUp(self):
        self.model = st.NativeCausalLm.from_gpt2(*fixture())
        self.model = self.model.with_geometry(artifact(self.model))
        self.qs, self.ys = [[1, 2], [3, 4, 5], [6, 7, 8], [1]], [3, 6, 9, 2]

    def new(self):
        return st.NativeCausalLmQkvAdamW(self.model, 0)

    def test_optional_elliptic_exports_resolve_only_when_requested(self):
        names = ("EllipticWarpFunction", "elliptic_warp_autograd",
                 "elliptic_warp_features", "elliptic_warp_partial")
        fake = SimpleNamespace(**{name: object() for name in names})
        original = {name: st.__dict__[name] for name in names if name in st.__dict__}
        try:
            for name in names:
                st.__dict__.pop(name, None)
            with patch.object(st, "import_module", return_value=fake) as loader:
                for name in names:
                    self.assertIs(getattr(st, name), getattr(fake, name))
                    self.assertIs(getattr(st, name), getattr(fake, name))
                self.assertEqual(loader.call_count, len(names))
                loader.assert_called_with("spiraltorch.elliptic")
        finally:
            for name in names:
                st.__dict__.pop(name, None)
            st.__dict__.update(original)

    def test_export_gradient_and_zero_smoothing_compatibility(self):
        self.assertIn("NativeCausalLmQkvAdamW", st.__all__)
        a, b = self.new(), self.new()
        before = a.checkpoint_with_id()
        g = a.answer_gradients(self.qs, self.ys, max_rows=2)
        self.assertEqual(tuple(g["gradient_shape"]), (12, 4))
        self.assertEqual(len(g["gradient_f32le"]), 48 * 4)
        self.assertEqual(a.checkpoint_with_id(), before)
        h = a.answer_gradients(self.qs, self.ys, max_rows=2, label_smoothing=0.)
        self.assertEqual(g["gradient_f32le"], h["gradient_f32le"])
        self.assertEqual(g["gold_nll"], h["objective_loss"])
        self.assertNotIn("gold_nll", h)
        a.step_answers(self.qs, self.ys, max_rows=2, learning_rate=.01)
        b.step_answers(self.qs, self.ys, max_rows=2, learning_rate=.01, label_smoothing=0.)
        self.assertEqual(a.checkpoint_with_id(), b.checkpoint_with_id())
        self.assertEqual((a.steps, a.layer), (1, 0))

    def test_global_gradient_clipping_is_opt_in_native_and_checkpointed(self):
        plain = self.new()
        self.assertIsNone(plain.max_gradient_norm)
        explicit = st.NativeCausalLmQkvAdamW(self.model, 0, max_gradient_norm=None)
        self.assertEqual(plain.checkpoint_with_id(), explicit.checkpoint_with_id())
        g = plain.answer_gradients(self.qs, self.ys, max_rows=2, label_smoothing=.1)
        values = struct.unpack("<48f", g["gradient_f32le"])
        norm = sum(x*x for x in values)**.5
        self.assertGreater(norm, 1e-6)
        a = st.NativeCausalLmQkvAdamW(self.model, 0, epsilon=.1, max_gradient_norm=norm/4)
        self.assertEqual(a.serving_id, plain.serving_id)
        self.assertEqual(a.answer_gradients(self.qs, self.ys, max_rows=2, label_smoothing=.1), g)
        kw = dict(max_rows=2, learning_rate=.03, label_smoothing=.1)
        report = a.step_answers(self.qs, self.ys, **kw)
        clip = report["gradient_clip"]
        self.assertAlmostEqual(clip["before_l2"], norm)
        self.assertLessEqual(clip["after_l2"], a.max_gradient_norm * (1+1e-6))
        self.assertTrue(clip["factors"])
        b = st.NativeCausalLmQkvAdamW.from_bytes(self.model, *a.checkpoint_with_id())
        self.assertEqual(a.max_gradient_norm, b.max_gradient_norm)
        self.assertEqual(a.step_answers(self.qs, self.ys, **kw), b.step_answers(self.qs, self.ys, **kw))
        self.assertEqual(a.checkpoint_with_id(), b.checkpoint_with_id())
        before = a.checkpoint_with_id()
        a.step_answers(self.qs, self.ys, learning_rate=0.)
        self.assertEqual(a.checkpoint_with_id(), before)
        self.assertNotIn("gradient_clip", plain.step_answers(self.qs, self.ys, learning_rate=.01))
        for invalid in (True, 0., -1., float("nan"), float("inf")):
            with self.assertRaises((TypeError, ValueError)):
                st.NativeCausalLmQkvAdamW(self.model, 0, max_gradient_norm=invalid)

    def test_smoothed_step_and_resume_are_exact(self):
        a = self.new()
        initial = a.last_logits_f32le(self.qs[1])
        kw = dict(max_rows=2, learning_rate=.03, label_smoothing=.1)
        report = a.step_answers(self.qs, self.ys, **kw)
        self.assertEqual(report["label_smoothing"], .1)
        self.assertNotIn("gold_nll", report)
        b = st.NativeCausalLmQkvAdamW.from_bytes(self.model, *a.checkpoint_with_id())
        self.assertEqual(a.serving_id, b.serving_id)
        self.assertEqual(a.step_answers(self.qs, self.ys, **kw), b.step_answers(self.qs, self.ys, **kw))
        self.assertEqual(a.checkpoint_with_id(), b.checkpoint_with_id())
        self.assertEqual(a.last_logits_f32le(self.qs[1]), b.last_logits_f32le(self.qs[1]))
        self.assertNotEqual(initial, a.last_logits_f32le(self.qs[1]))
        original = self.model.forward(self.qs[1], [1, 3], last_only=True).values()
        self.assertEqual(initial, struct.pack("<11f", *original))

    def test_rejections_and_zero_rate_never_change_state(self):
        a = self.new()
        state = a.checkpoint_with_id()
        a.step_answers(self.qs, self.ys, learning_rate=0., label_smoothing=.1)
        self.assertEqual(a.checkpoint_with_id(), state)
        for bad in [dict(learning_rate=True), dict(label_smoothing=True), dict(label_smoothing=float("nan")),
                    dict(label_smoothing=1.1), dict(max_rows=0), dict(max_rows=True)]:
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                a.step_answers(self.qs, self.ys, **(dict(learning_rate=.01) | bad))
            self.assertEqual(a.checkpoint_with_id(), state)
        for qs, ys in [([[True]], [1]), ([[1]], [True]), ([[999]], [1]), (self.qs, []),
                       ([[1], []], [1, 2])]:
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                a.step_answers(qs, ys, learning_rate=.01)
            self.assertEqual(a.checkpoint_with_id(), state)
        raw, pin = state
        with self.assertRaises(ValueError):
            st.NativeCausalLmQkvAdamW.from_bytes(self.model, raw[:-1], pin)
        other = st.NativeCausalLm.from_gpt2(*fixture())
        with self.assertRaises(ValueError):
            st.NativeCausalLmQkvAdamW.from_bytes(other, raw, pin)

    def test_shape_admission_precedes_reading_token_values(self):
        class Unreadable:
            def __len__(self): return 9
            def __getitem__(self, index): raise AssertionError("token copied before admission")
        a = self.new()
        with self.assertRaises(ValueError):
            a.answer_gradients([Unreadable()], [2])
        with self.assertRaises(ValueError):
            a.last_logits_f32le(Unreadable())
        class TooMany:
            def __len__(self): return st.NativeCausalLmQkvAdamW.MAX_ANSWER_SEQUENCES + 1
            def __getitem__(self, index): raise AssertionError("roster copied before admission")
        with self.assertRaises(ValueError):
            a.answer_gradients(TooMany(), [])

    def test_cached_projection_keeps_old_snapshot_after_training(self):
        for lag in (0, 1):
            for arithmetic in ("f32", "f32_wide_backward", "f64"):
                with self.subTest(lag=lag, arithmetic=arithmetic):
                    config, weights = fixture()
                    if lag:
                        config = json.loads(config)
                        config.update(model_type="spiraltorch_gpt2_key_lag_v1",
                                      spiraltorch_key_lag={"layer": 0, "lag": lag})
                        config = json.dumps(config)
                    model = st.NativeCausalLm.from_gpt2(config, weights)
                    model = model.with_lora(1, 2, 1.).with_geometry(
                        artifact(model, path="transformer.h.1.attn")
                    )
                    model = model.with_readout_accumulator(arithmetic)
                    a = st.NativeCausalLmQkvAdamW(model, 0)
                    a.step_answers(self.qs, self.ys, learning_rate=.03, label_smoothing=.1)
                    frozen = st.NativeCausalLmQkvAdamW.from_bytes(model, *a.checkpoint_with_id())
                    prefix = a.prefill([1, 2, 3], max_cache_values=128)
                    self.assertIsInstance(prefix, st.NativeCausalLmProjectionPrefix)
                    self.assertIn("NativeCausalLmProjectionPrefix", st.__all__)
                    self.assertEqual(prefix.serving_id, a.serving_id)
                    original = prefix.logits_f32le()
                    self.assertEqual(original, a.last_logits_f32le([1, 2, 3]))
                    a.step_answers(self.qs, self.ys, learning_rate=.03, label_smoothing=.1)
                    self.assertNotEqual(a.serving_id, prefix.serving_id)
                    for tail in ([4, 5], [7, 8]):
                        branch = prefix
                        for token in tail:
                            branch = branch.branch(token)
                            self.assertEqual(branch.logits_f32le(), frozen.last_logits_f32le(branch.token_ids))
                            self.assertEqual(branch.serving_id, frozen.serving_id)
                        self.assertEqual(branch.token_ids, [1, 2, 3] + tail)
                    self.assertEqual(prefix.token_ids, [1, 2, 3])
                    self.assertEqual(prefix.logits_f32le(), original)
                    fresh = a.prefill([1, 2, 3])
                    self.assertEqual(fresh.logits_f32le(), a.last_logits_f32le([1, 2, 3]))
                    self.assertNotEqual(fresh.logits_f32le(), original)
                    self.assertEqual((prefix.cache_values, prefix.max_cache_values), (48, 128))

    def test_cache_admission_and_failed_branch_preserve_prefix(self):
        class Unreadable:
            def __len__(self): return 3
            def __getitem__(self, index): raise AssertionError("token read before cache admission")
        a = self.new()
        before = a.checkpoint_with_id()
        for budget in (0, 47, True):
            with self.assertRaises((TypeError, ValueError)):
                a.prefill(Unreadable(), max_cache_values=budget)
        prefix = a.prefill([1, 2, 3], max_cache_values=48)
        state = (prefix.serving_id, prefix.token_ids, prefix.logits_f32le(), prefix.cache_values)
        for token in (True, -1, 999, 4):
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                prefix.branch(token)
            self.assertEqual(state, (prefix.serving_id, prefix.token_ids, prefix.logits_f32le(), prefix.cache_values))
        full = a.prefill([1] * 8)
        with self.assertRaises(ValueError):
            full.branch(2)
        self.assertEqual(full.token_ids, [1] * 8)
        self.assertEqual(a.checkpoint_with_id(), before)

    def test_generator_snapshot_resume_and_old_model_rejection(self):
        a = self.new()
        a.step_answers(self.qs, self.ys, learning_rate=.03)
        saved = a.checkpoint_with_id()
        frozen = st.NativeCausalLmQkvAdamW.from_bytes(self.model, *saved)
        cfg = dict(max_new_tokens=4, seed=81)
        full = a.generator([1, 2, 3], json.dumps(cfg))
        cache = a.generator([1, 2, 3], json.dumps(dict(
            cfg, execution=dict(kind="kv_cache", max_values=96))))
        self.assertEqual(full.projection_id, a.serving_id)
        self.assertEqual(full.step().to_json(), cache.step().to_json())
        payload, pin = cache.to_json(), cache.checkpoint_id
        resumed = frozen.restore_generator(payload, pin)
        a.step_answers(self.qs, self.ys, learning_rate=.03)
        with self.assertRaises(ValueError):
            a.restore_generator(payload, pin)
        with self.assertRaises(ValueError):
            st.NativeCausalLmGenerator.from_json(self.model, payload, pin)
        while not full.finished:
            expected = full.step().to_json()
            self.assertEqual(cache.step().to_json(), expected)
            self.assertEqual(resumed.step().to_json(), expected)
            self.assertEqual(cache.to_json(), resumed.to_json())
        self.assertEqual(cache.finish_reason, "length")
        self.assertEqual(cache.projection_id, frozen.serving_id)
        self.assertEqual(frozen.checkpoint_with_id(), saved)

    def test_generation_admission_precedes_foreign_token_access(self):
        class Unreadable:
            def __len__(self): return 3
            def __getitem__(self, index): raise AssertionError("prompt read before admission")
        a = self.new()
        state = a.checkpoint_with_id()
        for cfg in [dict(max_new_tokens=7), dict(max_new_tokens=4,
                    execution=dict(kind="kv_cache", max_values=95))]:
            with self.assertRaises(ValueError):
                a.generator(Unreadable(), json.dumps(cfg))
        with self.assertRaises((TypeError, ValueError)):
            a.generator([True], json.dumps(dict(max_new_tokens=1)))
        self.assertEqual(a.checkpoint_with_id(), state)

    def test_generation_type_stubs_match_the_native_interface(self):
        tree = ast.parse(Path(st.__file__).with_suffix(".pyi").read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        methods = {node.name: node for node in classes["NativeCausalLmQkvAdamW"].body
                   if isinstance(node, ast.FunctionDef)}
        for name in ("generator", "restore_generator"):
            node = methods[name]
            self.assertEqual(ast.unparse(node.returns), "NativeCausalLmGenerator")
            declared = tuple(arg.arg for arg in node.args.posonlyargs + node.args.args)
            native = tuple(inspect.signature(getattr(st.NativeCausalLmQkvAdamW, name)).parameters)
            self.assertEqual(declared, native)
        properties = {node.name: node for node in classes["NativeCausalLmGenerator"].body
                      if isinstance(node, ast.FunctionDef)}
        owner = properties["projection_id"]
        self.assertEqual(ast.unparse(owner.returns), "str | None")
        self.assertEqual([ast.unparse(d) for d in owner.decorator_list], ["property"])
        self.assertTrue(hasattr(st.NativeCausalLmGenerator, "projection_id"))

    def test_qkv_sampled_zspace_preserves_policy_and_resumes_for_each_readout(self):
        for lag in (0, 1):
            for arithmetic in ("f32", "f32_wide_backward", "f64"):
                with self.subTest(lag=lag, arithmetic=arithmetic):
                    config, weights = fixture()
                    if lag:
                        config = json.loads(config)
                        config.update(model_type="spiraltorch_gpt2_key_lag_v1",
                                      spiraltorch_key_lag={"layer": 0, "lag": lag})
                        config = json.dumps(config)
                    model = st.NativeCausalLm.from_gpt2(config, weights)
                    model = model.with_lora(1, 2, 1.).with_geometry(
                        artifact(model, path="transformer.h.1.attn")
                    ).with_readout_accumulator(arithmetic)
                    trainer = st.NativeCausalLmQkvAdamW(model, 0)
                    trainer.step_answers(self.qs, self.ys, learning_rate=.03)
                    saved = trainer.checkpoint_with_id()
                    policy = dict(max_new_tokens=4, seed=132, decoding=dict(
                        selection="sample", top_k=6, top_p=.8,
                        distribution=dict(kind="zspace", config=dict(curvature=-1.))))
                    full = trainer.generator([1, 2, 3], json.dumps(policy))
                    cached = trainer.generator([1, 2, 3], json.dumps(dict(
                        policy, execution=dict(kind="kv_cache", max_values=96))))
                    while not full.finished:
                        resumed = trainer.restore_generator(cached.to_json(), cached.checkpoint_id)
                        report = full.step().to_json()
                        self.assertEqual(cached.step().to_json(), report)
                        self.assertEqual(resumed.step().to_json(), report)
                        self.assertEqual(cached.to_json(), resumed.to_json())
                        self.assertEqual(json.loads(report)["zspace_contract"],
                                         "spiraltorch.zspace_generation_control.v1")
                        self.assertEqual(cached.projection_id, trainer.serving_id)
                    self.assertEqual(trainer.checkpoint_with_id(), saved)


if __name__ == "__main__":
    unittest.main()
