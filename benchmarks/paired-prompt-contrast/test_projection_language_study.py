import math
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from projection_language_audit import PIN, displacement, f32, get, objective
from projection_language_fixture import STOP, V, prepare, roster
from field_completions import fit_menus


class ProjectionLanguageStudyTests(unittest.TestCase):
    def test_readback_cli_imports_in_an_isolated_file_only_environment(self):
        root=Path(__file__).resolve().parent
        script=root/"projection_language_audit.py"
        command=[sys.executable,"-I","-S","-B",str(script)]
        sandbox=shutil.which("sandbox-exec")
        if sys.platform=="darwin" and sandbox:
            files=[script,*(root/(n+".py") for n in ("field_candidates","field_completions",
                                                      "completion_evidence","projection_language_fixture"))]
            allowed=" ".join("(literal "+json.dumps(str(p))+")" for p in files)
            policy=("(version 1)(allow default)(deny network*)(deny file-read-data (subpath "+
                    json.dumps(str(root))+"))(allow file-read-data "+allowed+")")
            command=[sandbox,"-p",policy,*command]
        result=subprocess.run(command,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,1)
        self.assertEqual(result.stderr.strip(),"usage: projection_language_audit.py FIXTURE TRAINED SCORED NEW_DIR")

    def test_seal_regenerates_and_never_overwrites_old_inputs(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/"fresh"
            self.assertEqual(prepare(root),PIN)
            with self.assertRaises(FileExistsError):
                prepare(root)
            seen=get(root/"fixture.json",PIN)
            self.assertEqual(set(seen["files"]),{"model.safetensors","config.json","train.json","queries.json","truth.json"})
            with self.assertRaises(ValueError):
                get(root/"train.json",seen["files"]["truth.json"])

    def test_templates_disjoint_labels_not_in_queries_train_only_complete_support(self):
        pairs,qs,truth=roster()
        train={tuple(p[k]["prompt"]) for p in pairs for k in ("short","long")}
        self.assertEqual((len(train),len(qs),len(truth)),(36,18,18))
        self.assertTrue(train.isdisjoint(tuple(q["prompt"]) for q in qs))
        menus=fit_menus(((p["field"],tuple(p[k]["answer"])) for p in pairs for k in ("short","long")),V,STOP)
        for i,(q,g) in enumerate(zip(qs,truth,strict=True)):
            self.assertEqual(set(q),{"slot","field","prompt"})
            self.assertEqual((q["slot"],g["slot"]),(i,i))
            self.assertIn(tuple(g["answer"]),menus[q["field"]].choices)
            self.assertEqual(len(menus[q["field"]].choices),6)

    def test_declared_mix_applies_only_to_first_answer_and_lf_is_scored(self):
        head=[0.]*V;head[2]=2.;head[14]=-.5
        body=[0.]*V;body[STOP]=-3.
        def bits(xs):return struct.unpack(f"<{V}I",struct.pack(f"<{V}f",*xs))
        def cost(xs,t):return math.log(sum(math.exp(x) for x in xs))-xs[t]
        short=objective([bits(head),bits(body)],[2],14)
        long=objective([bits(head),bits(body),bits(body)],[14,2],2)
        first_s=.5*cost(head,2)+.5*math.log1p(math.exp(-2.5))
        first_l=.5*cost(head,14)+.5*math.log1p(math.exp(2.5))
        self.assertEqual(short[0],f32((first_s+cost(body,STOP))/2))
        self.assertEqual(long[0],f32((first_l+cost(body,2)+cost(body,STOP))/3))
        self.assertNotAlmostEqual((short[0]+long[0])/2,(short[0]*2+long[0]*3)/5)

    def test_refusal_witness_distinguishes_zero_harm_and_safe_directions(self):
        _,z=displacement([1.,1.],[[0.,0.],[0.,0.]],[.5,.5],.25)
        self.assertFalse(z["representable"])
        gs=[[4.,0.],[-1.,1.]]
        _,equal=displacement([0.,0.],gs,[.5,.5],None)
        self.assertFalse(equal["both"])
        ns=[4.,math.sqrt(2.)]
        _,guarded=displacement([0.,0.],gs,[ns[1]/sum(ns),ns[0]/sum(ns)],equal["l2"])
        self.assertTrue(guarded["both"] and guarded["representable"])
