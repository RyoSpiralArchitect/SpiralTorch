"""Model-free scoring checks for the prospective runner."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("qkv_rival_pilot_run", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def scores(nll, top1):
    return {"answers": [
        {"document": i, "start": 3, "end": 5,
         "scores": [{"nll": 1.0, "top1_correct": 1},
                    {"nll": nll if i == 0 else 2.0, "top1_correct": top1 if i == 0 else 1}]}
        for i in range(48)
    ]}


class ScoringTests(unittest.TestCase):
    def test_first_fit_input_error_consumes_attempt(self):
        with TemporaryDirectory() as root:
            args = SimpleNamespace(output=Path(root), manifest_pin="not-the-manifest",
                                   deadline=1)
            with self.assertRaises(FileNotFoundError):
                runner.fit(args)
            attempt = args.output / "fit-attempt.json"
            self.assertEqual(json.loads(attempt.read_bytes())["manifest_sha256"],
                             "not-the-manifest")
            with self.assertRaises(FileExistsError):
                runner.fit(args)

    def test_native_qkv_answer_wrapper_is_unwrapped(self):
        class Result:
            document_count = 48
            scored_documents = 48

            def to_json(self):
                return json.dumps({"schema": "spiraltorch.causal_lm_qkv_answer_comparison.v1",
                                   "serving_id": "sha256:fixture", "layer": 11, "steps": 0,
                                   "comparison": scores(2.0, 1)})

        class Optimizer:
            def evaluate_against_source_by_answer(self, *args, **kwargs):
                return Result()

        data = {"keys": list(range(48)), "token_documents": [], "labels": []}
        self.assertEqual(runner.scored(Optimizer(), None, data), scores(2.0, 1))

    def test_teacher_forced_answer_and_nll_contrast(self):
        ordinary, rival = scores(2.0, 1), scores(1.5, 3)
        ordinary["answers"][0]["end"] = rival["answers"][0]["end"] = 6
        outcome = runner.contrast(ordinary, rival)
        self.assertEqual(outcome["all_top1_delta"], 1)
        self.assertEqual(outcome["ordinary_token_top1"], 48)
        self.assertEqual(outcome["rival_token_top1"], 50)
        self.assertEqual((outcome["improved_nll"], outcome["tied_nll"],
                          outcome["regressed_nll"]), (1, 47, 0))
        self.assertAlmostEqual(outcome["token_weighted_delta_nll"], -1.5 / 97)
        self.assertAlmostEqual(outcome["answer_weighted_delta_nll"], -0.5 / 48)

    def test_source_mismatch_is_not_a_contrast(self):
        ordinary, rival = scores(2.0, 1), scores(1.5, 2)
        rival["answers"][0]["scores"][0]["nll"] = 1.1
        with self.assertRaisesRegex(ValueError, "unmatched source"):
            runner.contrast(ordinary, rival)


if __name__ == "__main__":
    unittest.main()
