import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shared_prefix_readback import choice, report
from shared_prefix_study import fit, obj, prepare, roster
from projection_language_audit import f32


class SharedPrefixStudyTests(unittest.TestCase):
    def test_frozen_fixture_uses_shared_content_and_disjoint_prompts(self):
        train,qs,gold=roster()
        self.assertEqual(fit(train),[[23],[23,2],[23,14]])
        self.assertEqual(len({tuple(x["prompt"]) for x in train+qs}),24)
        self.assertEqual({tuple(sorted(q)) for q in qs},{("prompt","slot")})
        self.assertEqual([x["answer"] for x in train],[x["answer"] for x in gold])
        with TemporaryDirectory() as tmp:
            f=Path(tmp)/"fixture";pin=prepare(f)
            self.assertEqual(pin,"5bde5bd7ee6022810a6f5aaf9b63111a229e4377ab530e70d7c6b9bc2cfe1ddf")
            manifest=obj(f/"fixture.json",pin)
            self.assertEqual(obj(f/"train.json",manifest["files"]["train"]),train)
            with self.assertRaises(FileExistsError):
                prepare(f)

    def test_conditional_decision_does_not_depend_on_common_first_token(self):
        menu=[[23],[23,2],[23,14]];suffix=[.25,1.25,3.];gold=1
        w,relative,prob=choice(suffix,gold)
        for shared in (0.,10.,100.):
            line=[s+shared for s in suffix];ce=line[gold]/3
            row=dict(gold_index=gold,mix=.5,stop_id=1,shared_content_tokens=1,target_counts=[2,3,3],
                line_nlls=line,suffix_nlls=suffix,winner_index=w,menu_nll=relative,gold_mean_nll=ce,
                menu_probabilities=prob,rival_mass=prob[0]+prob[2],loss=f32(.5*(ce+relative)))
            report(row,menu,gold,.5)
            with self.assertRaises(ValueError):
                report({**row,"loss":relative+.1},menu,gold,.5)
        self.assertEqual(w,0)
        self.assertGreater(choice([0.,40.],0)[1],0.)
        self.assertAlmostEqual(sum(prob),1.)
