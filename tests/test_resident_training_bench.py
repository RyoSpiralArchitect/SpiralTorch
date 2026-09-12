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
    def test_optimizer_history_and_recipe_cannot_be_relabelled(self):
        value=dict(status="passed",learner=True,cadence="deferred",steps=8,completed_updates=8,
            accepted_updates=list(range(2,10)),acceptance="guarded_receipts",losses=[],initial_loss=.2,final_loss=.1,elapsed_ms=1.,
            learner_optimizer="clipped_topos_ema",momentum_damping=.5,grad_clip_max_norm=1/1024,
            state=dict(parameters=[[1.],[2.]],momentum=[[.1],[.2]]))
        check=lambda v:bench.validate_sample(v,"deferred",8,learner=True,learner_optimizer="clipped_topos_ema")
        check(value)
        for key,replacement in (("learner_optimizer",None),("momentum_damping",.6),("grad_clip_max_norm",None)):
            with self.subTest(key=key),self.assertRaises(ValueError):check(dict(value,**{key:replacement}))
        for history in ([],[[.1]],[[.1],[float("nan")]],[[.1],[True]]):
            with self.assertRaises(ValueError):check(dict(value,state=dict(parameters=[[1.],[2.]],momentum=history)))
        with self.assertRaises(ValueError):check(dict(value,state=dict(parameters=[[1.],[2.]])))
        with self.assertRaises(ValueError):bench.validate_sample(value,"deferred",8,learner=True)
        for mode in ("topos_ema","clipped_topos_ema"):
            self.assertTrue(all(c["learner_optimizer"]==mode for c in bench.recipes(True,"standard",mode)))
        with self.assertRaises(ValueError):bench.recipes(False,"standard","topos_ema")
        with self.assertRaises(ValueError):bench.recipes(True,"standard","adam")

    def test_seed_fusion_cannot_change_model_or_hide_its_execution(self):
        base=dict(config=dict(graph=True,shape=[2,1],steps=8),plan_json="unchanged",
            input=[1.],target=[2.],learning_rate=.01,kernel="register_2x2",accumulation="sequential",adapter={})
        candidate=copy.deepcopy(base);candidate["config"]["fuse_learner_seeds"]=True
        bench.match_seed_fixture(base,candidate)
        for key,value in (("plan_json","different"),("input",[3.]),("learning_rate",.2)):
            bad=copy.deepcopy(candidate);bad[key]=value
            with self.assertRaises(ValueError):bench.match_seed_fixture(base,bad)
        for selected in (False,1,None):
            bad=copy.deepcopy(candidate);bad["config"]["fuse_learner_seeds"]=selected
            with self.assertRaises(ValueError):bench.match_seed_fixture(base,bad)
        value=dict(status="passed",learner=True,cadence="deferred",steps=8,completed_updates=8,
            accepted_updates=list(range(2,10)),acceptance="guarded_receipts",losses=[],initial_loss=.2,final_loss=.1,elapsed_ms=1.)
        with self.assertRaises(ValueError):bench.validate_sample(value,"deferred",8,learner=True,seed_fusion=True)
        bench.validate_sample(dict(value,fused_learner_seeds=True),"deferred",8,learner=True,seed_fusion=True)
        with self.assertRaises(ValueError):bench.validate_sample(dict(value,fused_learner_seeds=1),"deferred",8,learner=True,seed_fusion=True)

    def test_learner_requires_actual_receipts_not_loss_or_torch_surrogates(self):
        value=dict(status="passed",learner=True,cadence="deferred",steps=8,completed_updates=8,
                   accepted_updates=list(range(2,10)),acceptance="guarded_receipts",losses=[],
                   initial_loss=.2,final_loss=.1,elapsed_ms=1.)
        bench.validate_sample(value,"deferred",8,learner=True,lane="candidate")
        for key,invalid in (("learner",False),("accepted_updates",list(range(1,9))),
                            ("completed_updates",7),("acceptance","synchronized_only"),
                            ("final_loss",float("nan")),("losses",[.1]*8)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                bench.validate_sample(dict(value,**{key:invalid}),"deferred",8,learner=True,lane="candidate")
        with self.assertRaises(ValueError):bench.validate_sample(value,"deferred",8,learner=True,lane="torch")
        with self.assertRaises(ValueError):bench.validate_sample(value,"deferred",8)
        torch=dict(value,acceptance="synchronized_only");del torch["accepted_updates"]
        bench.validate_sample(torch,"deferred",8,learner=True,lane="torch")
        with self.assertRaises(ValueError):bench.validate_sample(torch,"deferred",8,learner=True,lane="baseline")

    def test_learner_oracle_checks_both_vjps_and_all_updated_weights(self):
        value=dict(loss=.1,prediction=[.2],input_gradients=[[.1],[.3]],
                   raw_gradients=[[[.1],[.2]],[[.3],[.4]]],parameters=[[.5],[.6]])
        self.assertEqual(bench.learner_reference.compare(value,value),0.)
        for key,replacement in (("input_gradients",[[.1]]),("raw_gradients",[[[.1],[.2]],[[.3],[100.]]]),
                                ("parameters",[[float("nan")],[.6]])):
            with self.subTest(key=key),self.assertRaises(ValueError):
                bench.learner_reference.compare(dict(value,**{key:replacement}),value)
        state=dict(value,momentum=[[.1],[.2]])
        self.assertEqual(bench.learner_reference.compare(state,state),0.)
        for bad in (value,dict(state,momentum=[[.1],[100.]]),dict(state,momentum=[[.1]])):
            with self.assertRaises(ValueError):bench.learner_reference.compare(bad,state)

    def test_fusion_equivalence_rejects_residual_rebinding_or_changed_math(self):
        source = dict(schema="spiraltorch.nn.inference_plan.v2", input_shape=[2, 1],
            parameters=[dict(role="gain", shape=[1], values=[2.])], stages=[
                dict(kind="pointwise", parameters=[0], steps=[dict(op="multiply", rhs=1)]),
                dict(kind="pointwise", parameters=[], steps=[dict(op="relu", rhs=None)])])
        fused = copy.deepcopy(source)
        fused["stages"] = [dict(kind="pointwise", parameters=[0],
            steps=[dict(op="multiply", rhs=1), dict(op="relu", rhs=None)])]
        check = lambda a,b: bench.graph_reference.require_fusion_equivalent(json.dumps(a), json.dumps(b))
        check(source, fused)
        for key,value in (("op","gelu"),("rhs",0)):
            bad=copy.deepcopy(fused); bad["stages"][0]["steps"][1][key]=value
            with self.assertRaises(ValueError): check(source,bad)
        bad=copy.deepcopy(fused); bad["parameters"][0]["values"]=[3.]
        with self.assertRaises(ValueError): check(source,bad)
        source["stages"][1]["steps"]=[dict(op="add",rhs=0)]
        fused["stages"][0]["steps"][1]=dict(op="add",rhs=0)
        with self.assertRaises(ValueError): check(source,fused)

    def test_wide_matrix_is_explicit_bounded_and_separate(self):
        values=bench.recipes(True,"wide")
        self.assertEqual(len(values),9)
        self.assertEqual({v["seed"] for v in values},{17,29,43})
        self.assertEqual({(tuple(v["shape"]),v["depth"]) for v in values},
                         {((4,64,64),8),((2,128,128),8),((2,64,256),4)})
        self.assertTrue(all(v["graph"] is True and v["steps"]==8 for v in values))
        for v in values:
            self.assertLessEqual(v["shape"][0]*v["shape"][1]*v["shape"][2],32768)
        for graph,matrix in ((False,"wide"),(True,"typo")):
            with self.subTest(graph=graph,matrix=matrix),self.assertRaises(ValueError):
                bench.recipes(graph,matrix)

    def test_graph_recipes_include_two_pass_unbroadcast(self):
        values=bench.recipes(True)
        self.assertEqual(len(values),9)
        self.assertEqual({(tuple(v["shape"]),v["depth"]) for v in values},
                         {((2,16,32),2),((2,129,32),4),((4,32,64),8)})
        self.assertTrue(all(v["graph"] is True and v["steps"]==8 for v in values))
        self.assertTrue(all("graph" not in v for v in bench.recipes()))

    def test_graph_oracle_checks_every_parameter_and_gradient(self):
        value=dict(loss=1., prediction=[.1], input_gradient=[.2], parameters=[[.3],[.4]],
                   raw_gradients=[[.5],[.6]],effective_gradients=[[.5],[.6]])
        self.assertEqual(bench.graph_reference.compare(value,value),0.)
        for key in ("parameters","raw_gradients","effective_gradients"):
            for replacement in ([[.3]], [[float("nan")],[.4]], [[.3],[100.]]):
                bad=dict(value,**{key:replacement})
                with self.subTest(key=key),self.assertRaises(ValueError):
                    bench.graph_reference.compare(bad,value)

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
    def test_per_interval_optimizer_cannot_be_relabelled(self):
        row=self.row()
        row["samples"][4]["learner_optimizers"]={"baseline":None,"candidate":"topos_ema"}
        with self.assertRaisesRegex(ValueError,"optimizer"):
            validation.summarize_case(row,row["config"],("baseline","candidate"))

    def test_per_interval_seed_policy_cannot_be_relabelled(self):
        row=self.row()
        row["samples"][4]["fused_learner_seeds"]={"baseline":False,"candidate":True}
        with self.assertRaisesRegex(ValueError,"seed fusion"):
            validation.summarize_case(row,row["config"],("baseline","candidate"))

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
