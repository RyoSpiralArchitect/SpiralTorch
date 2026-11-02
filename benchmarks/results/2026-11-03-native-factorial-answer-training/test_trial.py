"""Only synthetic weights; include all four training and generation routes."""
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as study  # noqa: E402
from native_answer_generation import greedy_completions, paired_lines, score_line  # noqa: E402
d,t=study.d,study.trial


class NoForeign:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in d.old.FORBIDDEN|{"tokenizers"}:
            raise AssertionError("foreign ML or tokenizer in synthetic fit: "+fullname)


sys.meta_path.insert(0,NoForeign())
import spiraltorch as st  # noqa: E402


class Trial(unittest.TestCase):
    def test_disjoint_split_and_permutations(self):
        rows=d.records()
        frozen=d.freeze(rows)
        self.assertEqual(d.pin(frozen),study.SPEC_SHA)
        keys={r["key"] for rs in rows.values() for r in rs}
        self.assertEqual(len(keys),192)
        self.assertFalse(keys & d.excluded_keys())
        self.assertEqual(512-len(keys | d.excluded_keys()),32)
        self.assertEqual(len(set(frozen["order_sha256"].values())),3)

    def test_line_requires_emitted_delimiter_and_exact_content_not_tokenization(self):
        for text,exact,prefix in [(" red\n",True,True),(" red\n\nmore",True,True),(" red",False,True),
                                  (" red clay\n",False,True),("red\n",False,False),(" red\r\n",False,True),
                                  ("\n red\n",False,False),(" red<|endoftext|>",False,True)]:
            self.assertEqual(score_line(text," red"),dict(exact=exact,terminated="\n" in text,prefix_exact=prefix))
        self.assertEqual(paired_lines([True,False],[False,True]),dict(reference=1,candidate=1,gained=1,lost=1,delta=0))
        with self.assertRaises(ValueError):
            paired_lines([True],[1])

    def test_no_promotion_without_both_quality_endpoints_in_each_order(self):
        def result(delta,lines):
            return dict(nll_contrasts={d.PRIMARY:dict(answer_weighted_delta_nll=delta)},line_contrasts={d.PRIMARY:dict(delta=lines)})
        self.assertEqual(d.decision([result(-.002,1)]*3),"projection_joint_advantage_on_this_split")
        self.assertEqual(d.decision([result(.002,-1)]*3),"affine_joint_advantage_on_this_split")
        for rows in ([result(-.002,0)]*3,[result(-.0001,2)]*3,[result(.002,-1)]*2+[result(-.002,1)]):
            self.assertEqual(d.decision(rows),"no_prespecified_joint_advantage_not_equivalence")

    def test_two_real_synthetic_epochs_four_checkpoints_and_greedy_streams(self):
        fixture=runpy.run_path(str(d.old.ROOT/"bindings/st-py/tests/test_native_causal_lm.py"))["fixture"]
        base=st.NativeCausalLm.from_gpt2(*fixture(width=4,layers=2,positions=12,vocabulary=13))
        models=d.origin(st,base,4,1)
        train=dict(keys=["a","b","c"],token_documents=[[1,2,3,10],[2,4,6,7,10],[2,3,4,8,10]],
                   labels=[[-100,-100,3,10],[-100,-100,6,7,10],[-100,-100,-100,8,10]])
        ev=dict(keys=train["keys"],token_documents=[x[:-1] for x in train["token_documents"]],
                labels=[x[:-1] for x in train["labels"]])
        prompts=dict(keys=train["keys"],prompt_ids=[[1,2],[2,4],[2,3,4]])
        rows=[dict(key=k,answer=" sample") for k in train["keys"]]
        kw=dict(context=8,layer=1,stop_ids=[10,12],cap=3,cache=2048)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=root/"initial"
            first.mkdir()
            start=t.endpoint(st,models,ev,prompts,first,time.time()+60,**kw)
            loc=root/"order"
            restored,_=t.fit_order(st,base,models,train,[1,0,2],loc,time.time()+60,context=8,layer=1,epochs=2)
            self.assertEqual(len(study.readout.lines(loc/"steps.jsonl")),6)
            self.assertEqual([x["targets"] for x in t.read(loc/"training.json")],[7,7])
            self.assertTrue(all(e["update"]==6 for e in t.read(loc/"checkpoints.json")))
            self.assertEqual(start,t.endpoint(st,restored,ev,prompts,loc,time.time()+60,**kw))
            for dest in (first,loc):
                study.readout.decode_endpoint(dest,rows,lambda ids:" sample\n" if ids[0]%2 else "other")
                result=study.readout.describe(dest,rows)
                self.assertEqual(result["arms"]["both"]["content_tokens"],4)
                self.assertTrue(math.isfinite(result["factorial_nll"]["interaction"]))
                self.assertEqual(len(study.readout.lines(dest/"lines.jsonl")),12)
            self.assertTrue(all(x["tied"]==3 for x in t.read(first/"scores.json")["contrasts"].values()))
            with self.assertRaises(FileExistsError):
                t.fit_order(st,base,models,train,[1,0,2],loc,time.time()+60,context=8,layer=1,epochs=2)


if __name__=="__main__":
    unittest.main()
