"""Negative controls for the paired design, including isolated edits to toy copies."""
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest

E=runpy.run_path(str(Path(__file__).resolve().with_name("experiment.py")))


class Design(unittest.TestCase):
    def test_high_absolute_margin_is_not_a_gain(self):
        p=dict(evaluation=[dict(key="a",conflict=True),dict(key="b",conflict=False)],seeds=[1,2],min_effect=.01)
        def scores(values):
            return {k:dict(margin=v,nll=10-v) for k,v in zip(("a","b"),values)}
        s={k:scores(v) for k,v in {"initial-r1":(1,0),"initial-r4":(4,5),
            "1-r1":(1,1),"1-r4":(5,6),"2-r1":(1,1),"2-r4":(4.1,5.1)}.items()}
        self.assertGreater(s["2-r4"]["a"]["margin"],s["2-r1"]["a"]["margin"])
        out=E["contrast"](p,s)
        self.assertEqual([x["passed"] for x in out["orders"]],[True,False])
        self.assertFalse(out["advance"])
        self.assertEqual(out["orders"][0]["arms"]["r1"]["all"]["initial_ties"],1)

    def test_no_advance_for_smaller_deterioration(self):
        p=dict(evaluation=[dict(key="a",conflict=True),dict(key="b",conflict=False)],seeds=[1],min_effect=.01)
        s={k:{t:dict(margin=v,nll=10-v) for t in ("a","b")} for k,v in
           {"initial-r1":1,"initial-r4":1,"1-r1":0,"1-r4":.5}.items()}
        o=E["contrast"](p,s)
        self.assertGreater(o["orders"][0]["paired_gain"]["all"],.01)
        self.assertFalse(o["advance"])

    def test_pairs_are_structurally_balanced(self):
        words=set(E["D"]["lexemes"](E["HELD"]))|{"The","that","the","met","is","are"}
        ids={w:i+1 for i,w in enumerate(sorted(words))}
        rows=E["D"]["cases"](E["HELD"],ids)
        self.assertEqual(len(rows),16)
        for pair in E["HELD"]:
            self.assertEqual({(x["plural"],x["conflict"]) for x in rows if x["pair"]==pair[0]},
                             {(False,False),(False,True),(True,False),(True,True)})


def tamper(toy):
    p=E["load"](toy/"plan.json")
    assert p["kind"]=="rehearsal"
    art=E["base64"].b64decode(p["initial"]["r1"])
    try:
        E["weights"](art,p,"r4")
    except ValueError:
        pass
    else:
        raise AssertionError("unit radius accepted as radius four")
    checks=("tape","endpoint","direction","anchor","truncated-logits")
    for which in checks:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"copy"
            shutil.copytree(toy/"attempt",root)
            stage="11-r1"
            if which=="tape":
                a=root/(stage+".g32")
                raw=bytearray(a.read_bytes())
                raw[0]^=1
                a.write_bytes(raw)
            elif which=="endpoint":
                a=root/"sealed.json"
                sealed=E["load"](a)
                sealed[stage]=sealed["11-r4"]
                a.write_bytes(E["enc"](sealed))
            elif which=="direction":
                a=root/(stage+".jsonl")
                rows=list(map(E["json"].loads,a.read_bytes().splitlines()))
                rows[0]["step"]["gradient_dot_step"]*=-1
                a.write_bytes(b"".join(map(E["enc"],rows)))
            elif which=="truncated-logits":
                a=root/(stage+".f32le")
                a.write_bytes(a.read_bytes()[:-4])
            changed_plan=E["json"].loads(E["enc"](p))
            if which=="anchor":
                changed_plan["anchors"]["11"][0][1]=p["pins"]["r4"]
            try:
                E["replay"](changed_plan,root)
            except ValueError as exc:
                print(which, str(exc))
            else:
                raise AssertionError("accepted "+which)


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--tamper-toy":
        tamper(Path(sys.argv[2]))
    else:
        unittest.main()
