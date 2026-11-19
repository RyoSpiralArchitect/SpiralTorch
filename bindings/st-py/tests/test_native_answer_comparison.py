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
FIXTURES = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm.py")))
FIXTURE = FIXTURES["fixture"]
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
    def test_cli_rejects_ambiguous_pinned_corpus_before_opening_model(self):
        run = runpy.run_path(str(ROOT / "bindings/st-py/examples/native_answer_comparison.py"))["main"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, output = root / "eval.json", root / "result"
            for raw, reason in (
                (b'{"labels":[[1]],"token_documents":[[1]],"labels":[[1]]}', "duplicate field"),
                (b'{"labels":[[1]],"token_documents":[[]]}', "labels must match token document shape"),
                (b'{"labels":[[1]],"token_documents":[[true]]}', "invalid type"),
            ):
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    args = [
                        "--model-dir", str(root / "missing"), "--documents", str(path),
                        "--documents-sha256", hashlib.sha256(raw).hexdigest(),
                        "--conditional-variant", "after", str(root / "missing.ck"), "unused",
                        "--context", "8", "--reference", "source", "--output", str(output),
                    ]
                    with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                        run(args)
                    self.assertIn("invalid labeled evaluation corpus", err.getvalue())
                    self.assertIn(reason, err.getvalue())
                    self.assertFalse(output.exists())

    def test_conditional_snapshot_scores_match_independent_full_answers(self):
        source = st.NativeCausalLm.from_gpt2(*FIXTURE())
        axis, guard, rows, pairs, schedule = FIXTURES["conditional_session_inputs"](source)
        kind = st.NativeCausalLmConditionalAnswerSession
        session = kind(axis, guard, rows, schedule, length=.01)
        initial = kind.serving_from_checkpoint(source, *session.checkpoint_with_id())
        self.assertTrue(session.step()["report"]["step"]["committed"])
        trained = kind.serving_from_checkpoint(source, *session.checkpoint_with_id())
        documents = [[5, 4, 8, 2, 7], [3, 2, 1, 4]]
        labels = [[-100, -100, 8, 2, 7], [-100, 2, -100, 4]]
        score = st.NativeCausalLmConditionalServingSnapshot.evaluate_variants_by_answer
        report = score(source, [initial, trained], documents, 8, labels=labels, microbatch_windows=2)
        self.assertIsInstance(report, st.NativeCausalLmConditionalAnswerComparison)
        self.assertIn("NativeCausalLmConditionalAnswerComparison", st.__all__)
        self.assertEqual((len(report), report.shared_prefix_layers), (3, 1))
        self.assertEqual((report.document_count, report.scored_documents), (2, 2))
        self.assertEqual([r.target_tokens for r in report.evaluations], [5] * 3)
        self.assertEqual(len({r.data_id for r in report.evaluations}), 1)
        self.assertEqual(report.contrast(0, 1)["tied"], 3)
        self.assertGreater(abs(report.contrast(1, 2)["token_weighted_delta_nll"]), 1e-6)
        raw = json.loads(report.to_json())
        self.assertEqual(raw["base_model_identity"], source.base_model_identity)
        self.assertEqual(raw["snapshots"], [
            dict(variant=i, checkpoint_id=s.checkpoint_id, projection_id=s.projection_id,
                 layer=1, site="output", position=s.position, total_batches=s.total_batches)
            for i, s in enumerate((initial, trained), 1)
        ])

        for i, (doc_index, start, end) in enumerate([(0, 2, 5), (1, 1, 2), (1, 3, 4)]):
            answer = report.answer(i)
            self.assertEqual((answer["document"], answer["start"], answer["end"]), (doc_index, start, end))
            doc, targets = documents[doc_index], labels[doc_index]
            saved = answer["scores"]
            self.assertEqual(saved[0], saved[1])
            for j, snapshot in enumerate((initial, trained), 1):
                independent = [
                    st.score_token_logits_f32le(snapshot.prefill(doc[:t]).logits_f32le(), targets[t])
                    for t in range(start, end)
                ]
                self.assertAlmostEqual(saved[j]["nll"], math.fsum(s["nll"] for s in independent) / (end-start), places=6)
                self.assertEqual(saved[j]["top1_correct"], sum(s["target_id"] == s["greedy_id"] for s in independent))

        early = source.rank_one_projection(
            "output", layer=0, left_f64le=struct.pack("<4d", 0., .6, -.8, 0.),
            right_f64le=struct.pack("<4d", .03, -.02, .01, .04),
        )
        early_session = kind(early, early.prepare_retention(rows, pairs), rows, schedule, length=.01)
        early_snapshot = kind.serving_from_checkpoint(source, *early_session.checkpoint_with_id())
        mixed = score(source, [trained, early_snapshot], documents, 8, labels=labels, microbatch_windows=2)
        self.assertEqual(mixed.shared_prefix_layers, 0)
        self.assertEqual([mixed.answer(i)["scores"][1] for i in range(len(mixed))],
                         [report.answer(i)["scores"][2] for i in range(len(report))])
        self.assertGreater(abs(mixed.contrast(0, 2)["token_weighted_delta_nll"]), 1e-6)
        for i in range(len(mixed)):
            row = mixed.answer(i)
            doc = row["document"]
            expected = [
                st.score_token_logits_f32le(early_snapshot.prefill(documents[doc][:t]).logits_f32le(),
                                            labels[doc][t])["nll"]
                for t in range(row["start"], row["end"])
            ]
            self.assertAlmostEqual(row["scores"][2]["nll"], math.fsum(expected) / len(expected), places=6)

        class Unreadable:
            def __len__(self):
                raise AssertionError("corpus read before source admission")

        for wrong in (source.with_readout_accumulator("f64"), source.with_lora(0, 2, 4.)):
            with self.assertRaisesRegex(ValueError, "exact frozen source"):
                score(wrong, [trained], Unreadable(), 8)
        with self.assertRaisesRegex(ValueError, "1 to 15 snapshots"):
            score(source, [trained] * 16, Unreadable(), 8)

    def test_cli_scores_conditional_checkpoints_without_training_inputs(self):
        run = runpy.run_path(str(ROOT / "bindings/st-py/examples/native_answer_comparison.py"))["main"]
        config, weights = FIXTURE(raw_weights=True)
        source = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive.from_bytes(weights))
        source = source.with_readout_accumulator("f64").with_lora(0, 2, 4.)
        artifact = FIXTURES["artifact"](source, path="transformer.h.0.attn", trainable=True)
        source = source.with_geometry(artifact)
        projection, guard, rows, pairs, schedule = FIXTURES["conditional_session_inputs"](source)
        kind = st.NativeCausalLmConditionalAnswerSession
        session = kind(projection, guard, rows, schedule, length=.01)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(config, encoding="utf-8")
            (root / "model.safetensors").write_bytes(weights)
            (root / "lora").write_text(source.lora_to_json(), encoding="utf-8")
            (root / "geometry").write_text(artifact.to_json(), encoding="utf-8")
            docs, labels = [[5, 4, 8, 2, 7]], [[-100, -100, 8, 2, 7]]
            corpus = json.dumps(dict(token_documents=docs, labels=labels)).encode()
            (root / "evaluation").write_bytes(corpus)
            first = session.save_checkpoint_new(root / "before.ck")
            session.step()
            later = session.save_checkpoint_new(root / "after.ck")
            args = ["--documents", str(root / "evaluation"), "--documents-sha256", hashlib.sha256(corpus).hexdigest(),
                    "--context", "8", "--reference", "before", "--source-readout", "f64",
                    "--source-lora", str(root / "lora"), "--source-geometry", str(root / "geometry"),
                    "--source-geometry-id", artifact.artifact_id, "--conditional-variant", "before",
                    str(root / "before.ck"), first, "--conditional-variant", "after", str(root / "after.ck")]
            out = root / "result"
            with contextlib.redirect_stdout(io.StringIO()):
                run([*args, later, "--model-dir", str(root), "--output", str(out)])
            report = json.loads(out.read_bytes())
            self.assertEqual(report["schema"], "spiraltorch.pinned_conditional_answer_comparison.v1")
            self.assertEqual([v["name"] for v in report["variants"]], ["source", "before", "after"])
            self.assertEqual([v["checkpoint_id"] for v in report["variants"][1:]], [first, later])
            self.assertEqual(set(report["contrasts"]), {"source", "after"})
            self.assertFalse(report["training_documents_verified"])
            self.assertFalse(report["split_independence_verified"])
            self.assertNotIn(b"token_documents", out.read_bytes())
            bad = root / "bad"
            with self.assertRaisesRegex(ValueError, "conditional checkpoint differs from pinned identity"):
                run([*args, "sha256:" + "0" * 64, "--model-dir", str(root / "missing"), "--output", str(bad)])
            self.assertFalse(bad.exists())

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
