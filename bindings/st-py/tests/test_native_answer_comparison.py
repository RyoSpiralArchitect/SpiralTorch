"""A deliberately constructed weighting counterexample, not a quality study."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import struct
import tempfile
import unittest

import spiraltorch as st

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))["fixture"]
DOCS = [[1,1], [1,1], [1]*9, [], [1], [1,1,1]]
LABELS = [[-100,1], [-100,1], [-100]+[0]*8, [], [0], [-100]*3]


def counterexample():
    # Every real input is ID 1 with a zero embedding. Tied head row 0 reads
    # [1,-1]. The affine intervention raises only logit 0; the base has ties.
    cfg, raw = FIXTURE(width=2, heads=1, layers=1, positions=4, vocabulary=2, raw_weights=True)
    header_end = 8 + struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:header_end])
    weights = bytearray(raw[:header_end]) + bytearray(len(raw)-header_end)
    for name, value in header.items():
        start, stop = value["data_offsets"]
        if name == "wte.weight":
            numbers = [1.,-1.,0.,0.]
        elif name.endswith(".weight") and "ln_" in name:
            numbers = [1.]*((stop-start)//4)
        else:
            continue
        struct.pack_into("<"+"f"*len(numbers), weights, header_end+start, *numbers)
    base = st.NativeCausalLm.from_gpt2(cfg, st.SafeTensorArchive.from_bytes(bytes(weights)))
    bare = base.with_lora(0,1,1.)
    state = st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema":"spiraltorch.geometry_adapter_artifact.v1","policy":"full_prefix_f32_no_cache.v1",
        "base_model_identity":base.base_model_identity,"training":False,"placements":[{
            "path":"transformer.h.0.attn","training":False,
            "recipe":{"schema":"spiraltorch.channel_affine_adapter.v1","features":2,"strength":1.,"max_values":1024},
            "tensors":[{"name":name,"shape":[2],"values":v,"requires_grad":False,"role":"parameter"}
                       for name,v in [("gate",[0.,0.]),("bias",[1.,-1.])]]}]}))
    return cfg, bytes(weights), bare, bare.with_geometry(state)


def pinned_inputs(root):
    cfg, weights, bare, candidate = counterexample()
    (root/"config.json").write_text(cfg)
    (root/"model.safetensors").write_bytes(weights)
    (root/"eval.json").write_text(json.dumps({"token_documents":DOCS,"labels":LABELS}))
    def arm(name, model, training_docs):
        session = st.NativeCausalLmTrainingSession(model, training_docs, 4, mode="lora", learning_rate=.1,
                                                  window_policy="covering_targets")
        ck, boundary = root/(name+".ck"), root/(name+".json")
        pin = session.save_checkpoint_new(ck)
        boundary.write_text(json.dumps({"checkpoint_id":pin,"completed_updates":session.completed_updates,
                                        "recorded_data_id":session.data_id,"post_lora_id":model.lora_artifact_id,
                                        "post_geometry_id":model.geometry_artifact_id}))
        return ["--variant",name,str(ck),pin,str(boundary)]
    args = ["--model-dir",str(root),"--documents",str(root/"eval.json"),
            "--documents-sha256",hashlib.sha256((root/"eval.json").read_bytes()).hexdigest(),
            "--reference","bare","--context","4","--microbatch-windows","2"]
    return args, arm("bare", bare, [[1,1]]), arm("affine", candidate, [[1,1]]), arm("unmatched", candidate, [[0,1]])


class AnswerComparisonTests(unittest.TestCase):
    def test_weightings_have_opposite_signs_even_when_top1_is_unchanged(self):
        _, _, bare, candidate = counterexample()
        detail = st.NativeCausalLm.evaluate_variants_by_answer([bare,candidate,bare],DOCS,4,
                                                              labels=LABELS,microbatch_windows=2)
        self.assertIsInstance(detail,st.NativeCausalLmAnswerComparison)
        for m in (st,st._rs):
            self.assertIn("NativeCausalLmAnswerComparison",m.__all__)
        self.assertEqual((len(detail),detail.document_count,detail.scored_documents),(3,6,3))
        self.assertEqual([(v["document"],v["start"],v["end"]) for v in map(detail.answer,range(3))],
                         [(0,1,2),(1,1,2),(2,1,9)])
        corpus = st.NativeCausalLm.evaluate_variants([bare,candidate,bare],DOCS,4,labels=LABELS,microbatch_windows=2)
        self.assertEqual(json.loads(detail.to_json())["comparison"],json.loads(corpus.to_json()))
        c = detail.contrast(0,1)
        v0,v1 = candidate.forward([1],[1,1]).values()
        long_delta = math.log1p(math.exp(v1-v0))-math.log(2.)
        short_delta = long_delta+v0-v1
        self.assertAlmostEqual(c["token_weighted_delta_nll"],(8*long_delta+2*short_delta)/10,places=10)
        self.assertAlmostEqual(c["answer_weighted_delta_nll"],(long_delta+2*short_delta)/3,places=10)
        self.assertLess(c["token_weighted_delta_nll"],0.)
        self.assertGreater(c["answer_weighted_delta_nll"],0.)
        self.assertEqual((c["improved"],c["regressed"],c["tied"],c["target_tokens"]),(1,2,0,10))
        self.assertEqual((c["candidate_top1_correct"],c["reference_top1_correct"]),(8,8))
        self.assertEqual((c["candidate_all_correct_answers"],c["reference_all_correct_answers"]),(1,1))
        self.assertEqual(detail.contrast(0,2)["tied"],3)
        with self.assertRaises(TypeError):
            detail.answer(False)
        with self.assertRaises(IndexError):
            detail.answer(3)
        for a,b in [(0,0),(0,3),(True,1)]:
            with self.assertRaises((ValueError,TypeError)):
                detail.contrast(a,b)

    def test_bounds_and_owners_are_checked_before_foreign_values(self):
        _,_,bare,_ = counterexample()
        class Unread:
            def __len__(self):
                return 2
            def __getitem__(self,i):
                raise AssertionError("inadmissible corpus read")
        score = st.NativeCausalLm.evaluate_variants_by_answer
        with self.assertRaisesRegex(ValueError,"readout"):
            score([bare,bare.with_readout_accumulator("f64")],Unread(),4)
        limits = st.NativeCausalLmAnswerComparison
        docs = [Unread()]*(limits.MAX_SCORE_CELLS//limits.MAX_VARIANTS+1)
        with self.assertRaisesRegex(ValueError,"score budget"):
            score([bare]*limits.MAX_VARIANTS,docs,4)
        with self.assertRaisesRegex(ValueError,"no selected targets"):
            score([bare,bare],[[1,1]],4,labels=[[0,-100]])

    def test_cli_checks_pinned_files_and_boundaries_without_training(self):
        run = runpy.run_path(str(ROOT/"bindings/st-py/examples/native_answer_comparison.py"))["main"]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args,a,b,unmatched=pinned_inputs(root)
            out=root/"result"
            with contextlib.redirect_stdout(io.StringIO()):
                run([*args,*a,*b,"--output",str(out)])
            raw=out.read_bytes()
            result=json.loads(raw)
            self.assertEqual(result["contrasts"]["affine"]["improved"],1)
            self.assertEqual([v["checkpoint_id"] for v in result["variants"]],[a[3],b[3]])
            self.assertFalse(result["training_documents_verified"])
            self.assertFalse(result["split_independence_verified"])
            self.assertNotIn("token_documents",raw.decode())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run([*args,*a,*b,"--output",str(out)])
            self.assertEqual(out.read_bytes(),raw)
            other=root/"forbidden"
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                run([*args,*a,*unmatched,"--output",str(other)])
            self.assertIn("recorded training corpus",err.getvalue())
            (root/"eval.json").write_bytes((root/"eval.json").read_bytes()+b" ")
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                run([*args,*a,*b,"--output",str(other)])
            self.assertIn("pinned SHA-256",err.getvalue())
            (root/"eval.json").write_bytes((root/"eval.json").read_bytes()[:-1])
            (root/"affine.ck").write_bytes((root/"affine.ck").read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"differs from pinned identity"):
                run([*args,*a,*b,"--output",str(other)])
            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
