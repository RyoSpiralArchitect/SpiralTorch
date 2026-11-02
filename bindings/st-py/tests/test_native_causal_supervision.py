"""Source-span ownership, with real SGD restricted to a small synthetic fixture."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import spiraltorch as st

ROOT = Path(__file__).resolve().parents[3]
CASE = json.loads((ROOT / "crates/st-nn/tests/fixtures/causal_target_alignment.json").read_bytes())
EXAMPLE = ROOT / "bindings/st-py/examples/native_training_session.py"
FIXTURES = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_session.py")))


class Codec:
    def __init__(self):
        self.encodes = 0
        self.decodes = []

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens and text == CASE["text"]
        self.encodes += 1
        return SimpleNamespace(ids=CASE["input"]["token_ids"], offsets=CASE["input"]["token_offsets"])

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        self.decodes.append(tuple(ids))
        cat, kanji = "\U0001f408".encode(), "\u732b".encode()
        pieces = [b".", b"q:", cat[:2], cat[2:], " \u2192".encode(), kanji[:1], kanji[1:],
                  b"!", b"\n", b"x:", b"4"]
        return b"".join(pieces[i] for i in ids).decode()


class SupervisionTests(unittest.TestCase):
    def test_matched_lora_frozen_and_trainable_wave_score_the_same_answers_as_independent_steps(self):
        base, _ = FIXTURES["owners"]()
        lora = base.with_lora(1, 2, 4.)
        state = json.loads(FIXTURES["fixtures"]["artifact"](base, path="transformer.h.1.attn", trainable=True).state_json())
        site = state["placements"][0]
        site["recipe"] = {"schema": "spiraltorch.wave_gate_adapter.v2", "features": 4,
                          "strength": .25, "learnable_radius": True,
                          "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 1024}}
        site["tensors"].append({"name": "log_radius", "shape": [], "values": [0.], "requires_grad": True, "role": "parameter"})
        geom = st.GeometryAdapterArtifact.create_json(json.dumps(state))
        both = lora.with_geometry(geom)
        ids, labels = st.encode_supervised_document(Codec(), CASE["text"], CASE["input"]["target_spans"])
        session = st.NativeCausalLmTrainingSession
        def new(model, mode):
            return session(model, [ids], 6, labels=[labels], learning_rate=.1,
                           mode=mode, batch_windows=2, epochs=2, window_policy="covering_targets")
        control = new(lora, "lora")
        arms = [control, control.fork_variant(both, "lora"), control.fork_variant(both, "joint")]
        scalar = [new(lora,"lora"), new(both,"lora"), new(both,"joint")]
        for update in (1,2):
            depth, steps = session.step_matched(arms)
            self.assertEqual(depth,1)
            self.assertEqual([s.target_tokens for s in steps], [5]*3)
            self.assertEqual([s.to_json() for s in steps], [s.step().to_json() for s in scalar])
            self.assertEqual([s.checkpoint_with_id() for s in arms], [s.checkpoint_with_id() for s in scalar])
            if update == 1:
                self.assertEqual(steps[1].loss, steps[2].loss)
                arms = [session.from_json(base, [ids], *s.checkpoint_with_id(), labels=[labels]) for s in arms]
        self.assertIsNone(session.step_matched(arms))
        self.assertEqual(arms[1].model.geometry_artifact_id, geom.artifact_id)
        self.assertNotEqual(arms[2].model.geometry_artifact_id, geom.artifact_id)
        scored = st.NativeCausalLm.evaluate_variants([s.model for s in arms], [ids], 6, labels=[labels])
        self.assertEqual([v.to_json() for v in scored.evaluations],
                         [s.model.evaluate_documents([ids], 6, labels=[labels]).to_json() for s in scalar])

    def test_matched_binding_refuses_aliases_bad_rosters_and_mixed_cursors(self):
        base, model = FIXTURES["owners"]()
        a = FIXTURES["session"](model)
        b = a.fork_variant(model, "lora")
        session = st.NativeCausalLmTrainingSession
        class Huge:
            def __len__(self):
                return session.MAX_MATCHED_VARIANTS + 1
            def __getitem__(self, i):
                raise AssertionError("oversized roster was visited")
        for roster in (Huge(), [], [a], [a,a], [a,b,b], [a,base]):
            with self.assertRaises((TypeError, ValueError)):
                session.step_matched(roster)
        self.assertEqual((a.completed_updates,b.completed_updates), (0,0))
        b.step()
        pins = (a.checkpoint_id,b.checkpoint_id)
        with self.assertRaisesRegex(ValueError,"cursor"):
            session.step_matched([a,b])
        self.assertEqual((a.checkpoint_id,b.checkpoint_id), pins)
        with self.assertRaisesRegex(ValueError,"unadvanced"):
            b.fork_variant(model, "joint")

    def test_unshifted_multiturn_labels_match_rust_and_train_exactly(self):
        self.assertTrue({"encode_supervised_document", "CausalTargetAlignment"} <= set(st.__all__))
        plan = st.CausalTargetAlignment(**CASE["input"])
        self.assertEqual({k: getattr(plan, k) for k in CASE["expected"]}, CASE["expected"])
        codec = Codec()
        ids, labels = st.encode_supervised_document(codec, CASE["text"], CASE["input"]["target_spans"])
        self.assertEqual((ids, labels), (plan.token_ids, plan.labels))
        self.assertEqual(codec.encodes, 1)
        self.assertEqual(codec.decodes, [tuple(ids[a:b]) for a,b,_,_ in plan.runs])
        base, model = FIXTURES["owners"]()
        def session(mask):
            return st.NativeCausalLmTrainingSession(model, [ids], 6, labels=[mask], batch_windows=2,
                                                  learning_rate=.1, window_policy="covering_targets")
        actual, manual = session(labels), session(CASE["expected"]["labels"])
        step = actual.step()
        self.assertEqual(step.target_tokens, 5)
        self.assertEqual(step.to_json(), manual.step().to_json())
        self.assertEqual(actual.checkpoint_with_id(), manual.checkpoint_with_id())
        restored = st.NativeCausalLmTrainingSession.from_json(base, [ids], *actual.checkpoint_with_id(), labels=[labels])
        self.assertEqual(restored.checkpoint_id, manual.checkpoint_id)

    def test_boundary_errors_and_nonlossless_decoding_cannot_produce_training_input(self):
        for changes in ({"target_spans": [[1,3]]}, {"target_spans": [[0,2]]}, {"token_offsets": [[0,1]]},
                        {"source_length": True}, {"target_spans": [[5.,7]]}, {"token_ids": [False]*11}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError, OverflowError)):
                st.CausalTargetAlignment(**{**CASE["input"], **changes})
        codec = Codec()
        codec.decode = lambda ids, *, skip_special_tokens: "normalized"
        with self.assertRaisesRegex(ValueError, "exact source"):
            st.encode_supervised_document(codec, CASE["text"], CASE["input"]["target_spans"])
        codec = Codec()
        with self.assertRaisesRegex(ValueError, "span count"):
            st.encode_supervised_document(codec, CASE["text"], [])
        self.assertEqual(codec.encodes, 0)

    def test_admits_declared_sizes_before_reading_foreign_elements(self):
        class Unread:
            def __len__(self):
                return st.CausalTargetAlignment.MAX_TOKENS + 1

            def __getitem__(self, index):
                raise AssertionError("inadmissible sequence was visited")
        with self.assertRaisesRegex(ValueError, "bounded token IDs"):
            st.CausalTargetAlignment(4, Unread(), Unread(), [[2,3]])

        class LargeEncoding(Unread):
            @property
            def ids(self):
                raise AssertionError("oversized IDs were materialized")

        for attr, value in (("padding", {"length": 10**12}), ("truncation", {"max_length": 1}),
                            ("model", SimpleNamespace(dropout=.25))):
            codec = Codec()
            setattr(codec, attr, value)
            with self.assertRaisesRegex(ValueError, "disable tokenizer"):
                st.encode_supervised_document(codec, "qa", [[1,2]])
            self.assertEqual(codec.encodes, 0)
        with self.assertRaisesRegex(ValueError, "token count"):
            st.encode_supervised_document(SimpleNamespace(encode=lambda *a, **kw: LargeEncoding()), "qa", [[1,2]])

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "optional local text codec")
    def test_pinned_local_codec_cli_trains_reopens_and_refuses_unsafe_inputs(self):
        from tokenizers import Tokenizer, decoders, models

        codec = Tokenizer(models.BPE({"_":0, "q":1, ":":2, "a":3, ".":4, "x":5, "qa":6}, [("q", "a")]))
        codec.decoder = decoders.Fuse()
        cfg, weights = FIXTURES["fixtures"]["fixture"](raw_weights=True)
        _, model = FIXTURES["owners"]()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"config").write_text(cfg)
            (root/"weights").write_bytes(weights)
            (root/"geom").write_text(model.geometry_artifact().to_json())
            (root/"codec").write_text(codec.to_str())
            pin = hashlib.sha256((root/"codec").read_bytes()).hexdigest()
            text = {"examples":[{"prompt":"q:", "completion":"a."}, {"text":"x:a.", "target_spans":[[2,4]]}]}
            (root/"text").write_text(json.dumps(text))
            (root/"ids").write_text(json.dumps({"token_documents":[[1,2,3,4],[5,2,3,4]],
                                                "labels":[[-100,-100,3,4],[-100,-100,3,4]]}))
            def run(data, out, *opts):
                args = [str(EXAMPLE), "--config", str(root/"config"), "--weights", str(root/"weights"),
                        "--documents", str(root/data), "--output", str(root/out), *opts]
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()) as stream:
                    runpy.run_path(str(EXAMPLE), run_name="__main__")
                return json.loads(stream.getvalue())
            raw = ["--tokenizer-json",str(root/"codec"),"--tokenizer-sha256",pin]
            train = ["--geometry",str(root/"geom"),"--context","6","--batch-windows","2"]
            a = run("text","one",*raw,*train)
            b = run("ids","manual","--window-policy","covering_targets",*train)
            self.assertEqual(a,b)
            self.assertEqual(a["reports"][0]["target_tokens"],4)
            self.assertEqual((root/"one").read_bytes(), (root/"manual").read_bytes())
            again = run("text","reopen",*raw,"--resume",str(root/"one"),"--expected-id",a["checkpoint_id"],"--updates","0")
            self.assertEqual(again["checkpoint_id"],a["checkpoint_id"])
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                run("text","bad",*raw[:-1],"0"*64,*train)
            self.assertIn("differs from pinned SHA-256",err.getvalue())
            (root/"text").write_text(json.dumps({"examples":[{"prompt":":q","completion":"a."}]}))
            with self.assertRaisesRegex(ValueError, "example 0: token crosses"):
                run("text","bad",*raw,*train)
            self.assertFalse((root/"bad").exists())


if __name__ == "__main__":
    unittest.main()
