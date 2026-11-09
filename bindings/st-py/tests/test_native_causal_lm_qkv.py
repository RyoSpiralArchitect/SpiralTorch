"""Real native QKV updates without foreign ML packages or array libraries."""
import importlib.abc
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


if __name__ == "__main__":
    unittest.main()
