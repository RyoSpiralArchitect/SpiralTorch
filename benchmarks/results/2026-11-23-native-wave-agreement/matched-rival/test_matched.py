"""Unit contracts, plus an optional independent oracle on disjoint tiny weights."""
from collections import Counter
import math
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

R=runpy.run_path(str(Path(__file__).with_name("run.py")))
D,P,N=R["D"],R["P"],R["N"]


class MatchedTests(unittest.TestCase):
    def setUp(self):
        self.spec=D["plan"]("rehearsal",None)

    def test_fresh_nouns_and_balanced_relative_clause(self):
        train,held=D["TRAIN"],D["HELD"]
        past=D["lexemes"](P["TRAIN"]+P["TEST"])
        self.assertFalse(D["lexemes"](train)&D["lexemes"](held))
        self.assertFalse(past&D["lexemes"](train+held))
        ids={w:i for i,w in enumerate(sorted(D["lexemes"](train+held)|{"The","that","the","met","is","are"}))}
        for groups,size in ((train,24),(held,16)):
            rows=D["cases"](groups,ids)
            self.assertEqual(len({tuple(x["prefix"]) for x in rows}),size)
            self.assertEqual(Counter((x["plural"],x["conflict"]) for x in rows),
                             Counter({(a,b):size//4 for a in (False,True) for b in (False,True)}))
            for x in rows:
                self.assertEqual(len(x["prefix"]),6)
                self.assertEqual(x["prefix"],[ids[w] for w in x["text"].split()])
                self.assertEqual(x["prefix"][-1],ids["met"])
                self.assertEqual(x["gold"],ids["are" if x["plural"] else "is"])

    def test_objective_and_mask(self):
        r=self.spec["train"][0]
        m=Mock(geometry_artifact_id="a",base_model_identity="b")
        g=Mock(geometry_artifact_id="a",base_model_identity="b",lora_artifact_id=None,
               window_shape=[1,7],target_tokens=1,loss=math.log(2),readout_accumulator="f32_wide_backward")
        g.declared_rival_report.return_value=dict(mix=1.,target_tokens=1,contrasted_tokens=1,rival_ids=1,
            mean_target_minus_declared_rival=0.,loss=g.loss,declared_ce=g.loss,full_ce=3.)
        m.geometry_declared_rival_gradients.return_value=g
        R["gradient"](m,r,"rival")
        m.geometry_declared_rival_gradients.assert_called_once_with(r["prefix"]+[r["gold"]],[1,7],
            labels=[-100]*6+[r["gold"]],mix=1.,contrasts=[(6,r["gold"],[r["rival"]])])
        g.declared_rival_report.return_value["mean_target_minus_declared_rival"]=2.
        with self.assertRaisesRegex(ValueError,"binary loss"):
            R["gradient"](m,r,"rival")
        g.declared_rival_report.return_value=None
        m.adaptation_loss_gradients.return_value=g
        R["gradient"](m,r,"full")
        m.adaptation_loss_gradients.assert_called_once_with(r["prefix"]+[r["gold"]],[1,7],
                                                          mode="geometry",labels=[-100]*6+[r["gold"]])

    def test_actual_distance_and_direction_not_only_native_claim(self):
        target=D["DISTANCE"]
        before=[0.]*32
        after=[-target]+before[1:]
        grad=[2.]+before[1:]
        trace=dict(trainable_values=32,method="gradient_ray_f32.v1",requested_l2=target,
                   gradient_l2=2.,realized_l2=target,gradient_dot_step=-2*target,relative_vector_error=0.)
        self.assertEqual(D["check_step"](before,after,grad,trace,self.spec),(target,0.))
        for bad,other in (([-2*target]+after[1:],grad),(after,[-2.]+grad[1:])):
            with self.assertRaises(ValueError):
                D["check_step"](before,bad,other,trace,self.spec)

    def test_interaction_is_not_any_objective_gain(self):
        scores={}
        for s,val in (("initial-wave",2.),("initial-affine",1.),("11-wave_full",1.9),("11-wave_rival",2.1),
                      ("11-affine_full",.9),("11-affine_rival",1.4)):
            scores[s]={r["key"]:dict(margin=val,nll=4.) for r in self.spec["evaluation"]}
        v=D["contrast"](self.spec,scores)
        self.assertEqual(v["objective_advantage"],dict(wave=True,affine=True))
        self.assertFalse(v["wave_specific"])
        self.assertAlmostEqual(v["orders"][0]["interaction"],-.3)
        del scores["11-wave_rival"][self.spec["evaluation"][0]["key"]]
        with self.assertRaises(ValueError):
            D["contrast"](self.spec,scores)

    def test_receipt_budget_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch("shutil.disk_usage",return_value=Mock(free=0)), patch("subprocess.Popen") as call:
                with self.assertRaisesRegex(ValueError,"admission"):
                    R["launch"](self.spec,root,None,"p")
                with self.assertRaises(FileExistsError):
                    R["launch"](self.spec,root,None,"p")
                call.assert_not_called()
            pin=D["digest"](D["encoded"](self.spec))
            R["save"](root/"attempt/receipt.json",dict(plan=pin,min_free=R["FLOOR"],seconds=1.))
            with self.assertRaisesRegex(ValueError,"receipt"):
                R["verify"](self.spec,root,pin)

    def test_only_local_content_ids(self):
        for bad in ("garbage","sha256:"+"."*64,"sha256:"+"0"*63,"SHA256:"+"0"*64):
            with self.assertRaises(ValueError):
                R["blob"](Path("."),bad)


def oracle():
    for key in ("torch","numpy","transformers","safetensors"):
        if key in sys.modules:
            raise RuntimeError("unexpected package")
        sys.modules[key]=None
    import spiraltorch as st
    spec=D["plan"]("rehearsal",None)
    base=N["tiny"](st).with_forward_readout_storage("original_packed").with_readout_accumulator("f32_wide_backward")
    row=spec["train"][0]
    for kind in ("wave","affine"):
        model=N["geometry"](st,base,kind,spec["shape"],spec["strength"])
        xs=row["prefix"]
        raw=model.forward(xs,[1,len(xs)],last_only=True).values_buffer()
        metrics=P["metrics"](memoryview(raw).cast("f"),row["gold"],row["rival"])
        exp={"full":metrics["nll"],"rival":D["binary_loss"](metrics["margin"])}
        assert exp["full"]>exp["rival"]
        for obj in ("full","rival"):
            g,_=R["gradient"](model,row,obj)
            assert math.isclose(g.loss,exp[obj],rel_tol=2e-6,abs_tol=1e-6),(kind,obj,g.loss,exp[obj])
    print("tiny Rust CE and declared binary loss match independent logits (2 geometries, 4 checks)")


if __name__=="__main__":
    if sys.argv[1:]==["--native-oracle"]:
        oracle()
    else:
        unittest.main()
