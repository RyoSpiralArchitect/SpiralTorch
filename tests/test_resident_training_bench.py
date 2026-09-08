"""Dependency-light admission tests; they do not claim GPU execution."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("training_bench",Path(__file__).resolve().parents[1]/"tools/bench_resident_training_vs_torch.py")
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Admission(unittest.TestCase):
    def test_recipe_matrix_is_frozen_and_bounded(self):
        values=bench.recipes()
        self.assertEqual(len(values),9)
        self.assertEqual({v["seed"] for v in values},{17,29,43})
        self.assertEqual({(tuple(v["shape"]),v["depth"]) for v in values},
                         {((2,16,32),2),((4,16,64),8),((4,32,128),16)})
        self.assertTrue(all(v["steps"]==8 for v in values))

    def test_only_valid_complete_intervals_admitted(self):
        valid=dict(status="passed",cadence="deferred",steps=8,losses=[1.]*8,elapsed_ms=1.)
        bench.validate_sample(valid,"deferred",8)
        for key,value in (("status","error"),("cadence","immediate"),("steps",7),("losses",[1.]*7),
                          ("elapsed_ms",True),("elapsed_ms",0),("elapsed_ms",float("nan")),("elapsed_ms",float("inf"))):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                bench.validate_sample(dict(valid,**{key:value}),"deferred",8)

    def test_finite_loss_trajectory_not_just_end_loss(self):
        bench.close_values([1.,.9],[1.,.9])
        for values in ([1.],[2.,.9],[float("nan"),.9],[1.,float("inf")]):
            with self.assertRaises(ValueError): bench.close_values(values,[1.,.9])


if __name__ == "__main__": unittest.main()
