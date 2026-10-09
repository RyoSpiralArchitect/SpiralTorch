import copy
import unittest

import byte_corpus_study as study
import verify_byte_corpus_calibrated_controls as controls
from test_byte_corpus_calibration import fixture as calibrated_fixture
from test_byte_corpus_study import metric_fixture


def fixture():
    raw, _, full = calibrated_fixture()
    source, _, retained = metric_fixture()
    first = copy.deepcopy(full)
    first["schema"] = "spiraltorch.byte_corpus.partial.v4"
    for c in first["cases"]:
        c["training"] = c["training"][:1]
        c["validation"] = c["validation"][:1]
    return [raw, source, full, retained, first]


class FirstUpdateControls(unittest.TestCase):
    def test_coordinated_fabrications_do_not_replace_request_constraints(self):
        for mutation in ("truncated", "evaluation", "initialization", "seed", "scalar_count"):
            args = fixture()
            full, retained, first = args[2:]
            if mutation == "truncated":
                for report in (full, retained):
                    for case in report["cases"]:
                        case["training"] = case["training"][:1]
            elif mutation == "evaluation":
                for report in (full, retained, first):
                    for case in report["cases"]:
                        case["validation"] = [dict(case["validation"][0], revision=1)]
            else:
                field, value = {"initialization": ("bias_initialization", study.ORIGINAL),
                                "seed": ("seed", 23), "scalar_count": ("parameter_scalars", True)}[mutation]
                for report in (full, first):
                    report["cases"][5][field] = value
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                controls.verify(*args)

    def test_all_three_first_update_pairs_and_five_inherited_arms(self):
        result = controls.verify(*fixture())
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["pairs"]), 3)
        self.assertEqual(len(result["inherited_full_reports_exact"]), 5)

    def test_mutations_reject(self):
        for mutation in ("inherited", "extra_eval", "missing", "revision", "core", "freeze", "embedding", "no_learning", "identity", "metadata"):
            args = fixture()
            first = args[-1]
            if mutation == "inherited":
                args[2]["cases"][0]["final_parameters"][0][0] += 1e-7
            elif mutation == "extra_eval":
                first["cases"][0]["validation"].append(dict(first["cases"][0]["validation"][0], revision=1))
            elif mutation == "missing":
                first["cases"][0]["final_parameters"].pop()
            elif mutation == "revision":
                first["cases"][0]["training"][0]["revision"] = 0
            elif mutation in ("core", "freeze"):
                first["cases"][6]["final_parameters"][0 if mutation == "core" else 6][0] += 1e-7
            elif mutation == "embedding":
                request = study.decode_json(args[0])
                for i in (5, 6):
                    first["cases"][i]["final_parameters"][0] = request["cases"][i]["parameters"][0]["values"]
            elif mutation == "no_learning":
                request = study.decode_json(args[0])
                for i, p in enumerate(request["cases"][5]["parameters"]):
                    if p["name"].startswith("geometry."):
                        first["cases"][5]["final_parameters"][i] = p["values"]
            elif mutation == "identity":
                first["cases"][6]["bias_initialization"] = study.ORIGINAL
            else:
                first["bias_calibration"]["train_batch_indices"] = [1, 0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                controls.verify(*args)


if __name__ == "__main__":
    unittest.main()
