import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("publish", Path(__file__).with_name("publish.py"))
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


def fixture():
    studies, rows = [], []
    for width in (16,32):
        for seed in (11,29,47):
            name = f"w{width}-seed{seed}"
            study = dict(name=name,width=width,seed=seed,artifacts=[],comparison=dict(numerical_checks_passed=seed != 47))
            for arm in publish.ARMS:
                for cursor in (0,256,512,1024):
                    artifact = dict(case_name=f"seed{seed}_{arm}",cursor=cursor,artifact_id=f"{name}-{arm}-{cursor}",selected=cursor == 1024)
                    study["artifacts"].append(artifact)
                    generated = [[1,2,1],[2,1,2]]
                    row = dict(**artifact,study=name,width=width,seed=seed,numerical_checks_passed=seed != 47,
                               fresh_loss_exact_f32=True,generation_full_forward_oracle=True,
                               generation=dict(generated_bytes=generated),repetition=publish.qualify.repetition(generated),
                               space_fraction=0.,development=dict(mean_ce=3.))
                    if cursor:
                        row["probe"] = dict(mean_ce=2.,batch_losses=[2.]*32,target_bytes=2048)
                    rows.append(row)
            studies.append(study)
    return dict(studies=studies), dict(cases=rows,baselines=[dict(cursor=c,development=dict(bigram_ce=2.5),probe=dict(bigram_ce=2.5)) for c in (0,256,512,1024)])


class PublicationTests(unittest.TestCase):
    def test_failure_kept_but_raw_parameter_values_not_published(self):
        original = dict(studies=[dict(comparison=dict(numerical_checks_passed=False,error="case.qkv.bias: 0.01 != 0.02"))])
        result = publish.public_selection(original,"original-hash")
        comparison = result["studies"][0]["comparison"]
        self.assertFalse(comparison["numerical_checks_passed"])
        self.assertEqual(comparison["error"],"case.qkv.bias: strict scalar tolerance exceeded")
        self.assertNotIn("0.01",str(result))
        self.assertEqual(original["studies"][0]["comparison"]["error"],"case.qkv.bias: 0.01 != 0.02")

    def test_redaction_falls_back_without_a_safe_label(self):
        for message in ("0.01 != 0.02","value=0.01: 0.02 != 0.03","0.01: 0.02 != 0.03"):
            original = dict(studies=[dict(comparison=dict(numerical_checks_passed=False,error=message))])
            public = publish.public_selection(original,"private")
            self.assertEqual(public["studies"][0]["comparison"]["error"],"strict scalar tolerance exceeded")
            self.assertNotIn("0.01",str(public))

    def test_every_seed_including_numerical_failure_is_retained(self):
        selection, result = fixture()
        publish.validate_evaluation(selection,result)
        summary = publish.summarize(result)
        self.assertEqual(len(summary["means"]),24)
        self.assertEqual(len(summary["paired_deltas"]),24)
        for group in summary["means"]:
            self.assertEqual(group["seeds"],3)
            self.assertEqual(group["numerical_qualified_seeds"],2)
            self.assertEqual(group["development"]["mean_ce_minus_bigram"],.5)
            if group["cursor"]:
                self.assertEqual(group["probe"]["seeds_beating_bigram"],3)

    def test_missing_or_relabelled_outputs_reject(self):
        selection, original = fixture()
        mutations = [lambda r:r["cases"].pop(),lambda r:r["cases"].reverse(),
                     lambda r:r["cases"][0].__setitem__("width",32),
                     lambda r:r["cases"][0].__setitem__("artifact_id","foreign"),
                     lambda r:r["cases"][0].__setitem__("numerical_checks_passed",False),
                     lambda r:r["cases"][1].pop("probe"),
                     lambda r:r["cases"][0]["repetition"].__setitem__("max_constant_run",999)]
        for mutate in mutations:
            result = copy.deepcopy(original)
            mutate(result)
            with self.assertRaises(ValueError):
                publish.validate_evaluation(selection,result)

    def test_baseline_change_cannot_invent_model_superiority(self):
        repo = Path(__file__).resolve().parents[3]
        protocol = publish.qualify.module(repo/"tools/byte_corpus_study.py")
        helper = publish.qualify.helpers(repo)
        probe = bytes(range(256))*10
        request = dict(config=dict(steps=32),train_documents=[list(range(64))],
                       train_batches=[[(0,0),(0,1)]],validation_documents=[list(range(128))],
                       validation_batches=[[(0,0),(0,32)]])
        available = protocol.windows([probe],32)
        starts = [available[i*len(available)//64][1] for i in range(64)]
        rows = [probe[s:s+33] for s in starts]
        valid = [bytes(range(33)),bytes(range(32,65))]
        result = dict(probe=dict(path="docs/byte_corpus_query_controls.md",revision=publish.qualify.launch.REVISION,
                                 sha256=publish.digest(probe),starts=starts,previously_inspected=True),
                      prompt_sha256=publish.digest(b"".join(row[:32] for row in valid)),
                      baselines=[dict(cursor=stop,development=helper.baselines(request,stop,valid),
                                      probe=helper.baselines(request,stop,rows)) for stop in publish.qualify.STOPS])
        with patch.object(publish.subprocess,"check_output",return_value=probe):
            publish.validate_baselines(repo,request,result,protocol)
            result["baselines"][1]["development"]["bigram_ce"] = 1000.
            with self.assertRaisesRegex(ValueError,"baselines"):
                publish.validate_baselines(repo,request,result,protocol)


if __name__ == "__main__":
    unittest.main()
