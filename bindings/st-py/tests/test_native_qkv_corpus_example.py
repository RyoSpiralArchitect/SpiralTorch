"""File-based native QKV training/resume with small synthetic models only."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import spiraltorch as st
from spiraltorch import native_qkv_runs
import test_native_qkv_generation_example as generation_fixture


class NativeQkvCorpusExampleTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parents[1]/"examples"
        self.main=runpy.run_path(str(root/"native_qkv_corpus_training.py"))["main"]
        self.generate=runpy.run_path(str(root/"native_geometry_generation.py"))["main"]

    def invoke(self, args, main=None):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            (main or self.main)([str(x) for x in args])
        return json.loads(output.getvalue())

    def prepare(self,root,source="geometry",arithmetic="f32"):
        # Reuse the verified frozen geometry/LoRA fixture; do not use its old QKV snapshot.
        helper=generation_fixture.NativeQkvGenerationExampleTests()
        model,_,generation_args=helper.prepare(root,source=source,arithmetic=arithmetic)
        common=generation_args[:-4]
        docs=[[1,2,3,4,5],[2,4,6]]
        labels=[[-100,-100,3,4,5],[-100,4,6]]
        (root/"data.json").write_text(json.dumps(dict(token_documents=docs,labels=labels)))
        recipe=dict(layer=0,optimizer=dict(beta1=.9,beta2=.999,epsilon=1e-8,max_gradient_norm=.01),
                    training=dict(context_length=4,batch_windows=2,max_rows=2,epochs=2,
                                  learning_rate=.03,weight_decay=.02,label_smoothing=.1))
        (root/"recipe.json").write_text(json.dumps(recipe))
        return model,common,docs,labels,recipe

    def test_file_training_matches_manual_and_resume_then_existing_generation(self):
        for source in ("geometry","session"):
            for arithmetic in ("f32","f32_wide_backward","f64"):
                with self.subTest(source=source,arithmetic=arithmetic), tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp)
                    model,common,docs,labels,recipe=self.prepare(root,source,arithmetic)
                    originals={p:p.read_bytes() for p in root.iterdir() if p.is_file()}
                    base=[*common,"--data-json",root/"data.json","--save-every",2]
                    full=self.invoke([*base,"--recipe-json",root/"recipe.json","--output",root/"full"])
                    self.assertEqual(native_qkv_runs.verify_qkv_run(root/"full"),full)
                    part=self.invoke([*base,"--recipe-json",root/"recipe.json","--stop-after",1,"--output",root/"part"])
                    self.assertEqual(native_qkv_runs.verify_qkv_run(root/"part"),part)
                    checkpoint=root/"part"/part["final_checkpoint"]["file"]
                    resumed=self.invoke([*base,"--resume",checkpoint,"--checkpoint-id",part["final_checkpoint"]["checkpoint_id"],
                                         "--output",root/"resumed"])
                    self.assertEqual(native_qkv_runs.verify_qkv_run(root/"resumed"),resumed)
                    manual=st.NativeCausalLmQkvTrainer(st.NativeCausalLmQkvAdamW(model,0,**recipe["optimizer"]),
                        docs,labels=labels,**recipe["training"])
                    expected=[]
                    while (step:=manual.step()) is not None:
                        expected.append(step)
                    logs=lambda folder:[json.loads(s) for s in (root/folder/"steps.jsonl").read_text().splitlines()]
                    self.assertEqual(logs("full"),expected)
                    self.assertEqual(logs("part")+logs("resumed"),expected)
                    self.assertEqual(full["final_checkpoint"]["checkpoint_id"],manual.checkpoint_with_id()[1])
                    self.assertEqual(full["final_checkpoint"],resumed["final_checkpoint"])
                    self.assertEqual((root/"full/qkv.bin").read_bytes(),(root/"resumed/qkv.bin").read_bytes())
                    self.assertEqual((root/"full"/full["final_checkpoint"]["file"]).read_bytes(),manual.checkpoint_with_id()[0])
                    projection=manual.optimizer_snapshot()
                    self.assertEqual((root/"full/qkv.bin").read_bytes(),projection.checkpoint_with_id()[0])
                    policy=dict(max_new_tokens=2,seed=71,execution=dict(kind="kv_cache",max_values=96),
                                decoding=dict(selection="sample",distribution=dict(kind="zspace",config=dict(curvature=-1.))))
                    (root/"generate.json").write_text(json.dumps(policy))
                    self.invoke([*common,"--qkv-checkpoint",root/"full/qkv.bin","--qkv-checkpoint-id",full["qkv"]["checkpoint_id"],
                        "--prompt-json",root/"prompt.json","--recipe-json",root/"generate.json","--output",root/"generated"],self.generate)
                    direct=projection.generator([1,2,3],json.dumps(policy))
                    direct.advance_ids(2)
                    self.assertEqual((root/"generated/generation.json").read_text(),direct.to_json())
                    self.assertEqual(originals,{p:p.read_bytes() for p in originals})
                    if os.name=="posix":
                        self.assertEqual((root/"full").stat().st_mode&0o777,0o700)
                        self.assertTrue(all(p.stat().st_mode&0o777==0o600 for p in (root/"full").iterdir()))

    def test_bad_pin_recipe_data_or_output_never_creates_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model,common,docs,labels,recipe=self.prepare(root)
            trainer=st.NativeCausalLmQkvTrainer(st.NativeCausalLmQkvAdamW(model,0),docs,4,labels=labels,learning_rate=.03)
            path=root/"resume.bin"
            pin=trainer.save_checkpoint_new(path)
            base=[*common,"--data-json",root/"data.json"]
            with self.assertRaises(ValueError), patch.object(st,"SafeTensorArchive",side_effect=AssertionError("loaded model before pin")):
                self.invoke([*base,"--resume",path,"--checkpoint-id","sha256:"+"0"*64,"--output",root/"bad"])
            self.assertFalse((root/"bad").exists())
            labels[0][0]=1
            (root/"data.json").write_text(json.dumps(dict(token_documents=docs,labels=labels)))
            with self.assertRaises(ValueError):
                self.invoke([*base,"--resume",path,"--checkpoint-id",pin,"--output",root/"changed"])
            self.assertFalse((root/"changed").exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke([*base,"--recipe-json",root/"recipe.json","--resume",path,"--checkpoint-id",pin,"--output",root/"mixed"])
            self.assertFalse((root/"mixed").exists())
            existing=root/"existing"; existing.mkdir(); (existing/"keep").write_bytes(b"keep")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke([*base,"--recipe-json",root/"recipe.json","--output",existing])
            self.assertEqual((existing/"keep").read_bytes(),b"keep")

    def test_log_error_preserves_published_boundary_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model,_,docs,labels,recipe=self.prepare(root)
            trainer=st.NativeCausalLmQkvTrainer(st.NativeCausalLmQkvAdamW(model,0),docs,labels=labels,**recipe["training"])
            calls=[]
            append=native_qkv_runs._append
            def fail_second(stream,record):
                calls.append(record)
                if len(calls)==2:
                    raise OSError("injected index failure after native checkpoint publication")
                return append(stream,record)
            with patch.object(native_qkv_runs,"_append",side_effect=fail_second), self.assertRaises(OSError):
                native_qkv_runs.run_qkv_corpus(trainer,root/"failed",save_every=2)
            self.assertEqual(trainer.updates,2)
            self.assertEqual(len(calls),2)
            self.assertEqual((root/"failed/checkpoint-000000002.bin").read_bytes(),trainer.checkpoint_with_id()[0])
            self.assertTrue((root/"failed/checkpoint-000000000.bin").exists())
            self.assertFalse((root/"failed/result.json").exists())
            self.assertFalse((root/"failed/qkv.bin").exists())
            with self.assertRaises(OSError):
                native_qkv_runs.verify_qkv_run(root/"failed")
            state=trainer.checkpoint_with_id()
            for stop,every in ((True,2),(1,2),(99,2),(2,0),(2,True),(2,1025)):
                with self.assertRaises(ValueError):
                    native_qkv_runs.run_qkv_corpus(trainer,root/"invalid",stop_after=stop,save_every=every)
                self.assertFalse((root/"invalid").exists())
                self.assertEqual(trainer.checkpoint_with_id(),state)


if __name__=="__main__":
    unittest.main()
