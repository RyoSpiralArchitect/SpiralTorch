import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout

SPEC = importlib.util.spec_from_file_location("diagnose", Path(__file__).with_name("diagnose.py"))
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


class DiagnosticsTests(unittest.TestCase):
    def test_uniform_distribution(self):
        result = DIAG.score([0.] * 512, [0, 31])
        self.assertAlmostEqual(result["mean_ce"], math.log(256))
        self.assertAlmostEqual(result["mean_entropy"], math.log(256))
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["top1_space"], 0)

    def test_stable_softmax_and_lowest_tie(self):
        scores = [10000.] * 256
        result = DIAG.score(scores, [0])
        self.assertAlmostEqual(result["mean_ce"], math.log(256))
        self.assertEqual(result["top1_correct"], 1)

    def test_large_finite_offsets_preserve_uniform_ce(self):
        for offset in (float(2**56), -float(2**56), float(2**127)):
            with self.subTest(offset=offset):
                result = DIAG.score([offset] * 256, [0])
                self.assertAlmostEqual(result["mean_ce"], math.log(256))

    def test_shape_and_finite_guards(self):
        with self.assertRaises(ValueError):
            DIAG.score([0.] * 255, [0])
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                DIAG.packed([value])
            with self.assertRaises(ValueError):
                DIAG.score([value] * 256, [0])

    def test_prefix_reversal_preserves_final_input_and_target(self):
        window = bytes(range(33))
        result = window[:31][::-1] + window[31:]
        self.assertEqual(result[-2:], window[-2:])
        self.assertEqual(result[:31], bytes(range(30, -1, -1)))
        self.assertNotEqual(result, window)

    def test_context_shift_changes_logits_not_distribution(self):
        result = DIAG.context_pair([0.] * 256, [7.] * 256)
        self.assertEqual(result["logits_rms"], 7.)
        self.assertEqual(result["mean_total_variation"], 0.)

    def test_windows_reject_truncation(self):
        self.assertEqual(DIAG.rows([list(range(33))], [(0, 0)], 32), [bytes(range(33))])
        with self.assertRaises(ValueError):
            DIAG.rows([list(range(33))], [(0, 1)], 32)

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            DIAG.write(path, dict(value=1))
            with self.assertRaises(FileExistsError):
                DIAG.write(path, dict(value=2))

    def test_reject_same_engine(self):
        record = dict(schema="spiraltorch.byte_corpus.frozen_diagnosis.v1", engine="native")
        with self.assertRaisesRegex(ValueError, "engines"):
            DIAG.require_engines(record, record)
        DIAG.require_engines(record, dict(record, engine="torch"))

    def test_reject_summary_tampering_with_unchanged_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = [0.] * 768
            raw = DIAG.packed(values)
            name = "case-0-cohort-0.f32le"
            DIAG.write(path/name,raw)
            record = dict(name="tiny",file=name,sha256=DIAG.sha(raw),
                          metrics=DIAG.score(values[:512],[1,2]),
                          original_last=DIAG.score(values[:256],[2]),
                          reversed_prefix_last=DIAG.score(values[:256],[2]),
                          context_sensitivity=DIAG.context_pair(values[:256],values[:256]))
            args = (path,("tiny",[[0,1,2]],[[(0,0)]]),0,0,1,2,1)
            self.assertEqual(DIAG.verified_scores(record,*args),values)
            record["metrics"]["mean_ce"] += 1
            with self.assertRaisesRegex(ValueError,"summary"):
                DIAG.verified_scores(record,*args)

    def test_baselines_do_not_fit_future_or_validation_labels(self):
        request = dict(config=dict(steps=2), train_documents=[[1,2,3],[255,255,255]], train_batches=[[(0,0)]])
        a = DIAG.fitted_baselines(request,[("test",[[255,255,255]],[[(0,0)]])],1)
        self.assertEqual(a["training_target_presentations"],2)
        self.assertAlmostEqual(a["baselines"][0]["unigram_ce"],math.log(258))
        self.assertAlmostEqual(a["baselines"][0]["bigram_ce"],math.log(256))

    def test_future_scheduled_batches_do_not_fit_baseline(self):
        request = dict(config=dict(steps=2), train_documents=[[1,2,3],[255,255,255]],
                       train_batches=[[(0,0)],[(1,0)]])
        cohorts = [("test",[[255,255,255]],[[(0,0)]])]
        early = DIAG.fitted_baselines(request,cohorts,1)
        self.assertEqual(early["training_target_presentations"],2)
        self.assertAlmostEqual(early["baselines"][0]["unigram_ce"],math.log(258))
        late = DIAG.fitted_baselines(request,cohorts,2)
        self.assertEqual(late["training_target_presentations"],4)
        self.assertLess(late["baselines"][0]["unigram_ce"],early["baselines"][0]["unigram_ce"])

    def test_invalid_baseline_cursor_is_rejected(self):
        request = dict(train_batches=[[]])
        for cursor in (0,-1,2,True,1.0):
            with self.subTest(cursor=cursor):
                with self.assertRaisesRegex(ValueError,"cursor"):
                    DIAG.fitted_baselines(request,[],cursor)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        archive = self.root/"archive"
        (archive/"input").mkdir(parents=True)
        documents = [list(range(33)),list(range(33,66))]
        batches = [[[0,0],[1,0]]]
        selection = dict(name="synthetic")
        request = dict(config=dict(batch=2,steps=32),train_documents=documents,
                       train_batches=batches*128,validation_documents=documents,
                       validation_batches=batches)
        req = archive/"input/request.json"
        DIAG.write(req,request)
        model = json.dumps(dict(synthetic=True))
        saved = archive/"browser-checkpoint-128.json"
        DIAG.write(saved,dict(request_sha256=DIAG.sha(req.read_bytes()),completed_updates=128,
                             cases=[dict(name="synthetic",model_json=model)]))
        self.gate = self.root/"gate.json"
        DIAG.write(self.gate,dict(documents=documents,cohorts=[dict(name=name,batches=batches)
                   for name in ("first","second","third")]))
        self.plan = self.root/"plan.json"
        self.plan_value = dict(archive=str(archive),gate=str(self.gate),selections=[selection],
                context_batches_per_cohort=1,generation_bytes=32,parity=dict(atol=3e-6,rtol=5e-5),
                sources={str(p):DIAG.sha(p.read_bytes()) for p in (req,saved,self.gate)})
        DIAG.write(self.plan,self.plan_value)
        cohorts = DIAG.bound_inputs(self.plan_value)[4]
        baselines = DIAG.fitted_baselines(request,cohorts,128)
        self.reports = []
        self.destinations = []
        for engine in ("native","torch"):
            destination = self.root/engine
            destination.mkdir()
            self.destinations.append(destination)
            record = dict(selection=selection,model_json_sha256=DIAG.sha(model.encode()),
                          generated_bytes=[[0]*32,[0]*32],cohorts=[])
            for ni, (name,docs,selected) in enumerate(cohorts):
                values = [0.]*(2*32*256)
                last = [0.]*(2*256)
                targets = [value for row in docs for value in row[1:]]
                end_targets = [row[-1] for row in docs]
                raw = DIAG.packed(values+last)
                filename = f"case-0-cohort-{ni}.f32le"
                DIAG.write(destination/filename,raw)
                record["cohorts"].append(dict(name=name,file=filename,sha256=DIAG.sha(raw),
                    metrics=DIAG.score(values,targets),original_last=DIAG.score(last,end_targets),
                    reversed_prefix_last=DIAG.score(last,end_targets),
                    context_sensitivity=DIAG.context_pair(last,last)))
            self.reports.append(dict(schema="spiraltorch.byte_corpus.frozen_diagnosis.v1",
                engine=engine,plan_sha256=DIAG.sha(self.plan.read_bytes()),cases=[record],**baselines))

    def compare(self):
        for path, report in zip(self.destinations,self.reports):
            (path/"result.json").write_text(json.dumps(report))
        output = self.root/"comparison.json"
        with redirect_stdout(io.StringIO()):
            DIAG.compare(SimpleNamespace(plan=self.plan,native=self.destinations[0],
                                         reference=self.destinations[1],output=output))
        return json.loads(output.read_bytes())

    def test_complete_comparison_passes(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"][0]["errors"]),4)

    def test_identical_invalid_generations_cannot_pass(self):
        invalid = [[],[[0]*32],[[0]*31,[0]*32],[[0]*33,[0]*32]]
        invalid += [[[value]*32,[0]*32] for value in (-1,256,999,True,0.0,"0")]
        for rows in invalid:
            with self.subTest(rows=rows):
                for report in self.reports:
                    report["cases"][0]["generated_bytes"] = rows
                with self.assertRaisesRegex(ValueError,"generation|generated byte"):
                    self.compare()

    def test_cohort_source_shortening_cannot_skip_report_tail(self):
        gate = json.loads(self.gate.read_bytes())
        gate["cohorts"].pop()
        self.gate.write_text(json.dumps(gate))
        self.plan_value["sources"][str(self.gate)] = DIAG.sha(self.gate.read_bytes())
        self.plan.write_text(json.dumps(self.plan_value))
        for report in self.reports:
            report["plan_sha256"] = DIAG.sha(self.plan.read_bytes())
            report["cases"][0]["cohorts"][-1]["file"] = "does-not-exist.f32le"
        with self.assertRaisesRegex(ValueError,"three named cohorts"):
            self.compare()

    def test_report_cohort_shortening_is_rejected(self):
        for report in self.reports:
            report["cases"][0]["cohorts"].pop()
        with self.assertRaisesRegex(ValueError,"cohort coverage"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
