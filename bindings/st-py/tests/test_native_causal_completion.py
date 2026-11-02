"""Paired text/ID ingress through synthetic native SFT, scoring and prompt-only generation."""
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import spiraltorch as st

ROOT = Path(__file__).resolve().parents[3]
CASE = json.loads((ROOT / "crates/st-nn/tests/fixtures/causal_completion.json").read_bytes())
SPEC, EXPECTED = CASE["input"], CASE["expected"]
TEXT = CASE["text"]
P, A = SPEC["source"]["answer_span"]
OWNERS = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_session.py")))["owners"]


class Codec:
    def __init__(self):
        self.calls = []
        cat, answer = TEXT[1].encode(), TEXT[4].encode()
        self.pieces = [b"",b"Q",cat[:1],cat[1:],b":",b" ",answer[:1],answer[1:],b".",b"\n",b":",b"."]

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        self.calls.append(text)
        n = {TEXT:9, TEXT[:P]:4, TEXT[:A]:8}[text]
        return SimpleNamespace(ids=SPEC["training_ids"][:n],offsets=SPEC["source"]["offsets"][:n])

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return b"".join(self.pieces[i] for i in ids).decode()


def prepare(codec=None, terminator="\n"):
    return st.encode_supervised_completion(codec or Codec(), TEXT[:P], TEXT[P:A], terminator=terminator)


class NativeCompletionTests(unittest.TestCase):
    def test_public_text_and_id_views_match_shared_wire_fixture(self):
        self.assertTrue({"CausalCompletion","encode_supervised_completion"} <= set(st.__all__))
        codec = Codec()
        view = prepare(codec)
        self.assertEqual(codec.calls,[TEXT,TEXT[:P],TEXT[:A]])
        ids = st.CausalCompletion(SPEC["prompt_ids"],SPEC["prompt_answer_ids"],SPEC["training_ids"])
        for item in (ids,view):
            self.assertEqual(json.loads(item.to_json()),EXPECTED)
            self.assertEqual(item.prompt_ids,SPEC["prompt_ids"])
            self.assertEqual(item.training_document(),(EXPECTED["training_ids"],EXPECTED["training_labels"]))
            self.assertEqual(item.evaluation_document(),(SPEC["prompt_answer_ids"],EXPECTED["training_labels"][:item.answer_end]))
            local = item.prompt_ids
            local.append(0)
            self.assertEqual(item.prompt_ids,SPEC["prompt_ids"])
        codec = Codec()
        no_stop = prepare(codec,"")
        self.assertEqual(codec.calls,[TEXT[:A],TEXT[:P]])
        self.assertEqual(no_stop.evaluation_document(),no_stop.training_document())

    def test_independent_text_may_decode_identically_but_use_different_ids(self):
        for changed in (TEXT[:P],TEXT[:A]):
            class Alias(Codec):
                def encode(self,text,**kw):
                    result = super().encode(text,**kw)
                    if text == changed:
                        result.ids[-1] = 10 if text == TEXT[:P] else 11
                    return result
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError,"prefix"):
                prepare(Alias())
        class RewritesJoin(Codec):
            def decode(self,ids,**kw):
                value = super().decode(ids,**kw)
                return value.strip() if len(ids)==9 else value
        with self.assertRaisesRegex(ValueError,"joined"):
            prepare(RewritesJoin())

    def test_rejects_bad_bounds_before_foreign_elements_and_ambiguous_alignment(self):
        class Unread:
            def __len__(self): return (1<<20)+1
            def __getitem__(self,i): raise AssertionError("oversize sequence read")
        for p,a,t in [([],[],[]),([1],[1],[1,2]),([1],[1,2],Unread()),([True],[1,2],[1,2]),
                      ([0],[0,2**32],[0,2**32]),([1],[1,2],[1,-1])]:
            with self.assertRaises((ValueError,OverflowError,TypeError)):
                st.CausalCompletion(p,a,t)
        alignment = st.CausalTargetAlignment(4,[1,2,3,4],[[0,1],[1,2],[2,3],[3,4]],[[1,3]])
        for p,a in (([1,2],[1,2,3]),([1],[1,2]),([1],[1,2,3,4])):
            with self.assertRaisesRegex(ValueError,"single answer"):
                alignment.completion(p,a)
        codec = Codec()
        for p,a,t in [("","a",""),("q","",""),("q","a",None),("q","a"*(1<<20),"")]:
            with self.assertRaises(ValueError):
                st.encode_supervised_completion(codec,p,a,terminator=t)
        self.assertFalse(codec.calls)

    def test_native_joint_update_eval_and_generation_use_the_three_distinct_views(self):
        example = prepare()
        base, model = OWNERS()
        def train(ids,labels):
            session = st.NativeCausalLmTrainingSession(model,[ids],9,mode="joint",learning_rate=.01,epochs=1,
                                                       labels=[labels],window_policy="covering_targets")
            self.assertEqual(session.step().target_tokens,5)
            return session
        actual = train(*example.training_document())
        manual = train(EXPECTED["training_ids"],EXPECTED["training_labels"])
        self.assertEqual((actual.data_id, actual.to_json()),(manual.data_id, manual.to_json()))
        restored = st.NativeCausalLm.from_session_checkpoint_at(base,actual.to_json(),actual.checkpoint_id,update=1,
                    recorded_data_id=actual.data_id,post_lora_id=actual.model.lora_artifact_id,
                    post_geometry_id=actual.model.geometry_artifact_id)
        ids,labels = example.evaluation_document()
        result = st.NativeCausalLm.evaluate_variants_by_answer([restored], [ids], 9, labels=[labels])
        self.assertEqual(result.evaluations[0].target_tokens,4)
        self.assertEqual((result.answer(0)["start"],result.answer(0)["end"]),(4,8))
        cfg = '{"max_new_tokens":3,"execution":{"kind":"kv_cache","max_values":1024}}'
        gen = st.NativeCausalLmGenerator(restored,example.prompt_ids,cfg)
        independent = st.NativeCausalLmGenerator(restored,Codec().encode(TEXT[:P],add_special_tokens=False).ids,cfg)
        self.assertEqual(gen.token_ids,SPEC["prompt_ids"])
        self.assertEqual(gen.advance_ids(3),independent.advance_ids(3))
        self.assertEqual(gen.checkpoint_id,independent.checkpoint_id)
        runner = runpy.run_path(str(ROOT/"bindings/st-py/examples/native_training_session.py"))
        docs,masks = runner["text_documents"]({"examples":[dict(prompt=TEXT[:P],completion=TEXT[P:A],terminator="\n")]},Codec())
        self.assertEqual((docs,masks),([EXPECTED["training_ids"]],[EXPECTED["training_labels"]]))

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"),"optional local text codec")
    def test_real_local_bpe_rejects_merges_at_either_boundary(self):
        from tokenizers import Tokenizer, decoders, models
        for pair in (("q","a"),("a",".")):
            codec = Tokenizer(models.BPE({x:i for i,x in enumerate(["x","q","a",".","".join(pair)])},[pair]))
            codec.decoder = decoders.Fuse()
            with self.subTest(pair=pair),self.assertRaisesRegex(ValueError,"crosses"):
                st.encode_supervised_completion(codec,"xq","a",terminator=".")
        codec = Tokenizer(models.BPE({"x":0,"q":1,"a":2,".":3},[]))
        codec.decoder = decoders.Fuse()
        item = st.encode_supervised_completion(codec,"xq","a",terminator=".")
        self.assertEqual(item.prompt_ids,[0,1])
        self.assertEqual(item.training_document(),([0,1,2,3],[-100,-100,2,3]))
        self.assertEqual(item.evaluation_document(),([0,1,2],[-100,-100,2]))


if __name__ == "__main__":
    unittest.main()
