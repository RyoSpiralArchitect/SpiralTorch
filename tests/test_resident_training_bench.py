"""Dependency-light admission tests; they do not claim GPU execution."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("training_bench",Path(__file__).resolve().parents[1]/"tools/bench_resident_training_vs_torch.py")
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
spec=importlib.util.spec_from_file_location("training_bench_validation",Path(__file__).resolve().parents[1]/"tools/validate_resident_training_bench.py")
validation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


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


class Revalidation(unittest.TestCase):
    def row(self):
        config=bench.recipes()[0]
        row=dict(config=config,fixture=dict(config=config),samples=[],captures={},fingerprints={})
        for cadence in ("immediate","deferred"):
            for block in range(10):
                order=["baseline","candidate"] if (block+config["seed"])%2==0 else ["candidate","baseline"]
                row["samples"].append(dict(cadence=cadence,block=block,warmup=block<2,order=order,
                                           times_ms=dict(baseline=2.,candidate=1.)))
            for lane in ("baseline","candidate"):
                row["fingerprints"][lane]="a"*64
                row["captures"][cadence+"_"+lane]=dict(status="passed",cadence=cadence,steps=8,
                    losses=[1.]*8,elapsed_ms=1.,initial_loss=1.,state_sha256="a"*64)
        return row

    def test_recomputes_all_samples_and_rejects_selection(self):
        row=self.row()
        result=validation.summarize_case(row,row["config"],("baseline","candidate"))
        self.assertEqual(result["immediate"]["baseline_over_candidate"],2.)
        self.assertEqual(result["deferred"]["lanes"]["candidate"]["retained"],8)
        for key,value in (("warmup",False),("block",2),("order",["baseline","candidate"]),
                          ("times_ms",dict(baseline=float("nan"),candidate=1.))):
            bad=copy.deepcopy(row)
            bad["samples"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validation.summarize_case(bad,bad["config"],("baseline","candidate"))
        row["samples"].pop()
        with self.assertRaises(ValueError): validation.summarize_case(row,row["config"],("baseline","candidate"))

    def test_rejects_changed_fingerprint_or_published_median(self):
        for key,value in (("state_sha256","b"*64),("losses",[float("inf")]*8)):
            row=self.row()
            row["captures"]["deferred_candidate"][key]=value
            with self.assertRaises(ValueError): validation.summarize_case(row,row["config"],("baseline","candidate"))
        row=self.row()
        row["summary"]={c:dict(baseline=3.,candidate=1.) for c in ("immediate","deferred")}
        with self.assertRaises(ValueError): validation.summarize_case(row,row["config"],("baseline","candidate"))

    def test_every_browser_interval_requires_ordered_start_and_finish(self):
        row=self.row()
        events=[]
        for sample in row["samples"]:
            for lane in sample["order"]:
                identity=dict(config=row["config"],cadence=sample["cadence"],block=sample["block"],lane=lane)
                events.extend([dict(identity,stage="sample_started"),dict(identity,stage="sample_finished",
                    elapsed_ms=sample["times_ms"][lane],setup_ms=0.,state_sha256="a"*64)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"progress.jsonl"
            path.write_text("".join(json.dumps(e)+"\n" for e in events))
            self.assertEqual(validation.validate_progress(path,[row]),40)
            for bad in (events[:-1],events[1:],events+events[-2:]):
                path.write_text("".join(json.dumps(e)+"\n" for e in bad))
                with self.assertRaises(ValueError): validation.validate_progress(path,[row])

    def test_rejects_wrong_or_dirty_build_source(self):
        source=dict(commit="commit",tree="tree",tracked_dirty=False)
        manifest=dict(pkg=dict(name="st-core"),git=dict(commit="commit",tree="tree",dirty=False))
        validation.manifest_binding(manifest,source)
        for key,value in (("commit","different"),("tree","different"),("dirty",True)):
            bad=copy.deepcopy(manifest)
            bad["git"][key]=value
            with self.assertRaises(ValueError): validation.manifest_binding(bad,source)


if __name__ == "__main__": unittest.main()
