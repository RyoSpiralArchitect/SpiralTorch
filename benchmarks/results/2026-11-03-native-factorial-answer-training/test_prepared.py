"""Read-back tests require no pretrained weights, and never change a prepared study artifact."""
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify as v  # noqa: E402


class Admission(unittest.TestCase):
    def test_prepared_receipts_verify_independently_and_reject_changed_denominators(self):
        root=v.d.HERE/"prepared-receipts"
        meta,rows=v.verify_inputs(root)
        self.assertEqual((len(rows["train"]),len(rows["eval"])),(120,72))
        with tempfile.TemporaryDirectory() as tmp:
            copy=Path(tmp)/"copy"
            shutil.copytree(root,copy)
            meta["lengths"]["train"][0][3]+=1
            (copy/"manifest.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,"input freeze"):
                v.verify_inputs(copy)

    def test_multi_epoch_order_and_mask_chain_are_checked_from_a_synthetic_journal(self):
        import spiraltorch as st
        fixture=runpy.run_path(str(v.d.old.ROOT/"bindings/st-py/tests/test_native_causal_lm.py"))["fixture"]
        base=st.NativeCausalLm.from_gpt2(*fixture(width=4,layers=2,positions=12,vocabulary=13))
        models=v.d.origin(st,base,4,1)
        train=dict(keys=["a","b","c"],token_documents=[[1,2,3,10],[4,5,6,7,10],[2,3,4,8,10]],
                   labels=[[-100,-100,3,10],[-100,-100,6,7,10],[-100,-100,-100,8,10]])
        order=v.d.ORDERS[0]
        rows=[dict(key=k) for k in train["keys"]]
        stats=[[k,len(mask),mask.count(-100),len(mask)-mask.count(-100)] for k,mask in zip(train["keys"],train["labels"])]
        origin=[v.st.trial.owners(m) for m in models]
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(v.d,TRAIN=3,UPDATES=6,EPOCHS=2),patch.object(v.d.p,"LAYER",1):
            root=Path(tmp)
            models,data_id=v.st.trial.fit_order(st,base,models,train,v.d.indices(rows,order),root/order,time.time()+60,context=8,layer=1,epochs=2)
            self.assertEqual(v.training(root,order,origin,rows,stats,None,width=4),([v.st.trial.owners(m) for m in models],data_id))
            path=root/order/"steps.jsonl"
            data=v.r.lines(path)
            data[3]["reports"][0]["epoch"]=0
            path.write_text("".join(json.dumps(row)+"\n" for row in data))
            with self.assertRaisesRegex(ValueError,"update lineage"):
                v.training(root,order,origin,rows,stats,None,width=4)

    def test_no_unlisted_archive_member_can_be_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with zipfile.ZipFile(root/"x.zip","w") as z:
                z.writestr("../unexpected","test")
            with self.assertRaisesRegex(ValueError,"allowlist"):
                v.open_archive(root/"x.zip",root)
            self.assertFalse((root/"unexpected").exists())


if __name__=="__main__":
    unittest.main()
