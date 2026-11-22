"""No model or optional dependencies needed for these experimental controls."""
from collections import Counter
import json
from pathlib import Path
import runpy
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
driver = runpy.run_path(str(ROOT/"probe.py"))
p, native = driver["P"], driver["N"]


class AgreementTests(unittest.TestCase):
    def setUp(self):
        self.plan = p["build_plan"]("rehearsal", driver["sources"]())

    def test_real_lexical_split_balance_and_length(self):
        train, test = p["TRAIN"], p["TEST"]
        t, e = ({w for g in groups for w in g} for groups in (train,test))
        self.assertEqual((len(t),len(e),len(t & e)),(24,16,0))
        ids = {w:i for i,w in enumerate(sorted(t | e | {"The","near","the","is","are"}))}
        for groups, size in ((train,24),(test,16)):
            rows = p["cases"](groups,ids)
            self.assertEqual(len({r["key"] for r in rows}),size)
            self.assertEqual(len({tuple(r["prefix"]) for r in rows}),size)
            self.assertEqual(set(Counter((r["plural"],r["conflict"]) for r in rows).values()), {size//4})
            for r in rows:
                self.assertEqual(len(r["prefix"]),5)
                self.assertEqual((r["gold"],r["rival"]), (ids["are"],ids["is"]) if r["plural"] else (ids["is"],ids["are"]))

    def test_bpe_replays_merge_ranks_including_repeated_pairs(self):
        get = p["single_word_id"]
        ranks = {("a","a"):0,("aa","aa"):1,("G","aaaa"):2}
        self.assertEqual(get("Gaaaa",{"Gaaaa":8},ranks),8)
        with self.assertRaisesRegex(ValueError,"one token"):
            get("Gab",{"Gab":3},ranks)

    def test_margin_and_nll_ignore_shared_shift(self):
        metric = p["metrics"]
        a, b = metric([0.25,1.,-2.],1,0), metric([512.25,513.,510.],1,0)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["margin"],0.75)
        self.assertEqual(metric([1.,1.],1,0)["margin"],0.)
        with self.assertRaises(ValueError):
            metric([0.,float("nan")],0,1)

    def test_within_arm_change_not_final_score(self):
        names = driver["stages"](self.plan)
        scores = {n:{r["key"]:dict(margin=m,nll=2.) for r in self.plan["evaluation"]}
                  for n,m in zip(names,[5.,0.,5.,5.5,0.7])}
        s = p["summarize"](self.plan,scores)
        self.assertFalse(s["pilot_positive"])
        self.assertAlmostEqual(s["per_order"][0]["wave_minus_affine"],-.2)
        scores[names[-1]].pop(self.plan["evaluation"][0]["key"])
        with self.assertRaisesRegex(ValueError,"mismatched"):
            p["summarize"](self.plan,scores)

    def test_gradient_mask_order_and_frozen_control(self):
        example = self.plan["train"][0]
        before = Mock(geometry_artifact_id="a",base_model_identity="base")
        after = Mock(geometry_artifact_id="b",base_model_identity="base")
        grad = Mock(target_tokens=1,loss=.5,geometry_artifact_id="a",base_model_identity="base")
        before.geometry_loss_gradients.return_value = grad
        before.step_geometry_sgd.return_value = after
        same, log = native["train_one"](before,[example],[0],.025,True)
        self.assertIs(same,before)
        before.step_geometry_sgd.assert_not_called()
        before.geometry_loss_gradients.assert_called_once_with(example["prefix"]+[example["gold"]],[1,6],
                                                                labels=[-100]*5+[example["gold"]])
        self.assertEqual(log[0]["before"],log[0]["after"])
        new, log = native["train_one"](before,[example],[0],.025,False)
        self.assertIs(new,after)
        self.assertEqual(log[0]["after"],"b")
        grad.target_tokens = 2
        with self.assertRaisesRegex(ValueError,"gradient"):
            native["train_one"](before,[example],[0],.025,False)

    def test_plan_pin_scope_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            driver["save"](d/"plan.json",self.plan)
            pin = p["file_hash"](d/"plan.json")
            self.assertEqual(driver["plan_from"](d,pin,"rehearsal"),self.plan)
            for wrong_pin,kind in ((pin,"gpt2"),("0"*64,"rehearsal")):
                with self.assertRaises(ValueError):
                    driver["plan_from"](d,wrong_pin,kind)
            with self.assertRaises(FileExistsError):
                driver["save"](d/"plan.json",self.plan)

    def test_disk_failure_retains_attempt_and_never_launches(self):
        low = patch("shutil.disk_usage", return_value=Mock(free=0))
        with tempfile.TemporaryDirectory() as tmp, low, patch("subprocess.Popen") as proc:
            d = Path(tmp)
            with self.assertRaisesRegex(ValueError,"space"):
                driver["supervise"](self.plan,d,"pin",None)
            self.assertFalse(driver["load"](d/"attempt/failure.json")["retry"])
            with self.assertRaises(FileExistsError):
                driver["supervise"](self.plan,d,"pin",None)
            proc.assert_not_called()

    def test_readback_rejects_false_fixed_control_and_swapped_step(self):
        plan = self.plan
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            identity = {"base":"fixture","extension":"fixture"}
            driver["save"](dest/"identity.json",identity)
            pins = {}
            for name in driver["stages"](plan):
                # Use the same native construction recipe without loading any model.
                fake = Mock()
                fake.GeometryAdapterArtifact.create_json.side_effect = json.loads
                base = Mock(base_model_identity="fixture")
                base.with_geometry.side_effect = lambda x:x
                kind = "affine" if "affine" in name else "wave"
                raw = p["encoded"](native["geometry"](fake,base,kind,plan["shape"],plan["strength"]))
                driver["write"](dest/(name+".geometry.json"),raw)
                pins[name] = "sha256:"+p["digest"](raw)
                driver["write"](dest/(name+".f32le"),struct.pack("<128f",*range(128)))
                if not name.startswith("initial"):
                    hist = [dict(before=pins[name],after=pins[name],ce=1.,key=plan["train"][i]["key"])
                            for i in plan["order"][0]]
                    driver["save"](dest/(name+".steps.json"),hist)
            driver["save"](dest/"weights.json",pins)
            self.assertFalse(driver["score"](plan,dest)["pilot_positive"])
            original = Path.read_bytes
            def corrupted(path):
                data = original(path)
                if path.name == "11-wave_fixed.f32le":
                    return bytes([data[0] ^ 1]) + data[1:]
                return data
            with patch.object(Path,"read_bytes",corrupted), self.assertRaisesRegex(ValueError,"fixed forward bytes"):
                driver["score"](plan,dest)
            def swapped(path):
                data = original(path)
                if path.name.endswith("wave_sgd.steps.json"):
                    return p["encoded"](list(reversed(json.loads(data))))
                return data
            with patch.object(Path,"read_bytes",swapped), self.assertRaisesRegex(ValueError,"training chain"):
                driver["score"](plan,dest)


if __name__ == "__main__":
    unittest.main()
