import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

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

    def test_shape_and_finite_guards(self):
        with self.assertRaises(ValueError):
            DIAG.score([0.] * 255, [0])
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                DIAG.packed([value])

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
        a = DIAG.fitted_baselines(request,[("test",[[255,255,255]],[[(0,0)]])])
        self.assertEqual(a["training_target_presentations"],2)
        self.assertAlmostEqual(a["baselines"][0]["unigram_ce"],math.log(258))
        self.assertAlmostEqual(a["baselines"][0]["bigram_ce"],math.log(256))


if __name__ == "__main__":
    unittest.main()
