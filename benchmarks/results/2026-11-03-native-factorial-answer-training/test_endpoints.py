"""Rehearse the production read-back using generated weights and invented token IDs only."""
import copy
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as v
from endpoint_check import EndpointPlan, rounded_batch_bounds


def decode(ids):
    return "".join("\n" if x == 10 else f" {x}" for x in ids)


class SyntheticEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spiraltorch as st
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        fixture = runpy.run_path(str(v.d.old.ROOT / "bindings/st-py/tests/test_native_causal_lm.py"))["fixture"]
        base = st.NativeCausalLm.from_gpt2(*fixture(width=4, layers=2, positions=12, vocabulary=13))
        cls.base = base.base_model_identity
        models = v.d.origin(st,base,4,1)
        cls.origin = models
        train = dict(keys=["a","b","c"], token_documents=[[1,2,3,10],[2,4,6,7,10],[2,3,4,8,10]],
                     labels=[[-100,-100,3,10],[-100,-100,6,7,10],[-100,-100,-100,8,10]])
        docs = [[1,2,3,4,5],[2,5,6,4,7],[5,1,7,4,3],[5,4,1,9,8,6],[6,3,5,4,8,2]]
        starts = [3,4,1,3,5]
        keys = [f"x{i}" for i in range(len(docs))]
        cls.rows = [dict(key=k,answer=" 3") for k in keys]
        cls.content = dict(keys=keys, token_documents=docs, labels=[[-100]*s+doc[s:] for s,doc in zip(starts,docs)])
        cls.stats = [(k,len(doc),s,len(doc)-s) for k,doc,s in zip(keys,docs,starts)]
        cls.plan = EndpointPlan.build(cls.stats,10,2,content=cls.content,vocabulary=13)
        prompts = dict(keys=keys,prompt_ids=[doc[:s] for doc,s in zip(docs,starts)])
        init,final = cls.root/"initial",cls.root/"trained"
        init.mkdir()
        deadline = time.time()+60
        restored,_ = v.st.trial.fit_order(st,base,models,train,[1,0,2],final,deadline,context=8,layer=1,epochs=2)
        cls.owner = {}
        for path,arms in ((init,models),(final,restored)):
            cls.owner[path.name] = [v.st.trial.owners(m) for m in arms]
            v.st.trial.endpoint(st,arms,cls.content,prompts,path,deadline,context=10,layer=1,stop_ids=[10,12],cap=3,cache=2048)
            v.r.decode_endpoint(path,cls.rows,decode)
        cls.raw = v.read(final/"scores.json")["scores"]

    def readback(self, name, **overrides):
        options = dict(base_id=self.base,layer=1,stops={10,12},vocabulary=13,cap=3,decode=decode,initial=name=="initial")
        options.update(overrides)
        return v.verified_endpoint(self.root/name,self.rows,self.plan,self.owner[name],**options)

    def test_common_origin_and_all_reloaded_endpoints(self):
        self.assertEqual(self.plan.groups,((0,2),(2,3),(3,5)))
        self.assertEqual([v[3] for v in self.stats],[2,1,4,3,1])
        self.assertNotEqual(self.owner["initial"],self.owner["trained"])
        for name in ("initial","trained"):
            result,ids,count = self.readback(name)
            self.assertEqual((ids,count),((self.plan.data_id,self.plan.plan_id),20))
            self.assertTrue(all(a["content_tokens"] == 11 for a in result["arms"].values()))
        public = EndpointPlan.build(self.stats,10,2)
        self.assertIsNone(public.data_id)
        public.verify_scores(self.raw,4)
        with self.assertRaisesRegex(ValueError,"decoded generation"):
            self.readback("trained", decode=lambda ids:"not the decoded tokens")

    def test_wrong_aggregate_coverage_and_extra_cells_fail(self):
        for key,value in (("nll",self.raw["comparison"]["evaluations"][2]["nll"]+1),("perplexity",1.0),
                          ("window_count",4),("skipped_windows",1),("microbatches",4),("target_tokens",12),("ignored_targets",0)):
            with self.subTest(key=key):
                doc = copy.deepcopy(self.raw)
                doc["comparison"]["evaluations"][2][key] = value
                with self.assertRaises(ValueError):
                    self.plan.verify_scores(doc,4)
        for key,value in (("nll",-1),("nll",float("nan")),("top1_correct",True)):
            doc = copy.deepcopy(self.raw)
            doc["answers"][0]["scores"][1][key] = value
            with self.assertRaisesRegex(ValueError,"answer cell"):
                self.plan.verify_scores(doc,4)
        doc = copy.deepcopy(self.raw)
        doc["answers"][0]["scores"].append(doc["answers"][0]["scores"][0])
        with self.assertRaisesRegex(ValueError,"alignment"):
            self.plan.verify_scores(doc,4)

    def test_same_lengths_do_not_bind_different_prompt_or_labels(self):
        for change in ("prompt","label","plan","digest"):
            with self.subTest(change=change):
                data = copy.deepcopy(self.content)
                doc = copy.deepcopy(self.raw)
                plan = self.plan
                if change == "prompt":
                    data["token_documents"][0][0] += 1
                    plan = EndpointPlan.build(self.stats,10,2,content=data)
                    self.assertEqual(plan.plan_id,self.plan.plan_id)
                    self.assertNotEqual(plan.data_id,self.plan.data_id)
                elif change == "label":
                    data["labels"][0][-1] += 1
                    with self.assertRaisesRegex(ValueError,"mask"):
                        EndpointPlan.build(self.stats,10,2,content=data)
                    continue
                else:
                    for e in doc["comparison"]["evaluations"]:
                        e["plan_id" if change=="plan" else "data_id"] = "sha256:"+"0"*64
                with self.assertRaisesRegex(ValueError,"plan or content ID"):
                    plan.verify_scores(doc,4)

    def test_token_mean_not_macro_or_unrounded_f64(self):
        macro = math.fsum(a["scores"][2]["nll"] for a in self.raw["answers"])/len(self.stats)
        exact = math.fsum(a["scores"][2]["nll"]*s[3] for a,s in zip(self.raw["answers"],self.stats))/11
        aggregate = self.raw["comparison"]["evaluations"][2]["nll"]
        self.assertNotEqual(macro,aggregate)
        self.assertNotEqual(exact,aggregate)
        for wrong in (macro, exact):
            doc = copy.deepcopy(self.raw)
            doc["comparison"]["evaluations"][2].update(nll=wrong,perplexity=math.exp(wrong))
            with self.assertRaisesRegex(ValueError,"contradicts answer cells"):
                self.plan.verify_scores(doc,4)

    def test_swapped_aggregate_columns_of_distinct_native_predictions(self):
        import spiraltorch as st
        arms = []
        for i,model in enumerate(self.origin):
            state = json.loads(model.geometry_artifact().state_json())
            bias = next(t for t in state["placements"][0]["tensors"] if t["name"]=="bias")
            bias["values"][i] = 3.0
            arms.append(model.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state))))
        raw = json.loads(st.NativeCausalLm.evaluate_variants_by_answer(
            arms,self.content["token_documents"],10,labels=self.content["labels"],microbatch_windows=2).to_json())
        self.plan.verify_scores(raw,4)
        ev = raw["comparison"]["evaluations"]
        self.assertEqual(len({e["nll"] for e in ev}),4)
        for field in ("nll","perplexity"):
            ev[0][field],ev[2][field] = ev[2][field],ev[0][field]
        with self.assertRaisesRegex(ValueError,"contradicts answer cells"):
            self.plan.verify_scores(raw,4)

    def test_rounding_boundaries_and_legitimate_missing_perplexity(self):
        self.assertEqual(rounded_batch_bounds(1+2**-24,3),(1.,1.+2**-23))
        self.assertEqual(rounded_batch_bounds(2**-150,3),(0.,2**-149))
        doc = copy.deepcopy(self.raw)
        for a in doc["answers"]:
            a["scores"][0]["nll"] = 1000.0
        e = doc["comparison"]["evaluations"][0]
        e.update(nll=1000.0,perplexity=None)
        self.plan.verify_scores(doc,4)
        e["perplexity"] = 1.0
        with self.assertRaisesRegex(ValueError,"perplexity"):
            self.plan.verify_scores(doc,4)


class WorkerBudget(unittest.TestCase):
    def test_both_workers_share_the_original_deadline_and_remaining_cpu(self):
        prepared = dict(cpu_seconds=1.,wall_seconds=2.)
        attempt = dict(started_epoch=100.,deadline=3697.,limit_seconds=3600,retries=0)
        end = dict(finished_epoch=160.,cpu_seconds=48.,active_wall_seconds=62.,exit_code=0)
        phases = {"fit-launch": dict(cpu_limit=3594,started_epoch=100.1,deadline=3697.,prior_cpu=1.1),
                  "fit-worker": dict(cpu_limit=3594,started_epoch=100.5),
                  "grade-launch": dict(cpu_limit=3525,started_epoch=146.,deadline=3697.,prior_cpu=35.),
                  "grade-worker": dict(cpu_limit=3525,started_epoch=147.)}
        for mutation in ("none","cap","deadline","prior_cpu","order","excess","attempt","missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                files,meta = copy.deepcopy(phases),dict(attempt)
                if mutation == "cap":
                    files["fit-worker"]["cpu_limit"] += 1
                elif mutation == "deadline":
                    files["grade-launch"]["deadline"] += 1
                elif mutation == "prior_cpu":
                    files["grade-launch"]["prior_cpu"] = .5
                elif mutation == "order":
                    files["grade-launch"]["started_epoch"] = 99.
                elif mutation == "excess":
                    files["grade-worker"]["cpu_limit"] = files["grade-launch"]["cpu_limit"] = 3566
                elif mutation == "attempt":
                    meta["deadline"] += 1
                elif mutation == "missing":
                    del files["fit-worker"]
                for key,value in files.items():
                    (root/(key+".json")).write_text(json.dumps(value))
                if mutation == "none":
                    v.verify_fit_budget(root,prepared,meta,end)
                else:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        v.verify_fit_budget(root,prepared,meta,end)


if __name__ == "__main__":
    unittest.main()
