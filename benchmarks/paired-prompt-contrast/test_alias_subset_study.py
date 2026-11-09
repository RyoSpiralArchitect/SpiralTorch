import math
import unittest
from pathlib import Path
from unittest.mock import patch
from alias_subset_study import metrics, open_model, roster, MENU


class AliasStudyTests(unittest.TestCase):
    def test_model_builder_receives_only_the_existing_weight_and_config_pins(self):
        with patch("alias_subset_study.model") as builder:
            open_model(None,Path("."),{"config.json":"config-hash","weights":"weight-hash","truth.json":"not-for-model"})
            builder.assert_called_once_with(None,Path("."),{"config":"config-hash","weights":"weight-hash"})

    def test_disjoint_prompts_and_balanced_positive_exposure(self):
        train,queries,truth=roster()
        self.assertEqual((len(train),len(queries),len(truth)),(18,12,12))
        self.assertEqual(len({tuple(d["prompt"]) for d in train+queries}),30)
        self.assertTrue(all(set(q)=={"slot","prompt"} for q in queries))
        self.assertEqual(MENU,sorted(MENU))
        self.assertEqual({len(line) for line in MENU},{2})
        chosen=[]
        for step in range(36):
            i=step%18; m=train[i]["meaning"]
            chosen.append([m,m+3][(i//3+step//18)%2])
        self.assertEqual([chosen.count(k) for k in range(6)],[6]*6)
        self.assertEqual([v["meaning"] for v in truth].count(1),4)

    def test_worse_absolute_alias_mass_does_not_become_a_quality_win(self):
        a=metrics([4.]*6,0); b=metrics([5.,8.,8.,6.,8.,8.],0)
        self.assertLess(b["conditional"],a["conditional"])
        self.assertGreater(b["absolute"],a["absolute"])
        self.assertGreater(b["worst_alias"],a["worst_alias"])
        self.assertAlmostEqual(a["conditional"],math.log(3.))
        self.assertTrue(b["both_beat_negatives"])
