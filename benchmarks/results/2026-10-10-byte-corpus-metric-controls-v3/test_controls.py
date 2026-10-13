import copy
import importlib.util
from pathlib import Path
import unittest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("controls", Path(__file__).with_name("verify_controls.py"))
F = load("fixtures", C.ROOT / "tools/test_byte_corpus_study.py")


def fixture():
    raw, _, baseline = F.metric_fixture()
    request = C.S.decode_json(raw)
    old = copy.deepcopy(request)
    old["schema"] = C.S.REQUEST_V2
    old["cases"] = [{k: v for k, v in row.items() if k != "pair_metric"}
                    for row in old["cases"] if row.get("pair_metric") != C.S.FLAT]
    first = copy.deepcopy(baseline)
    first["schema"] = "spiraltorch.byte_corpus.partial.v3"
    for row in first["cases"]:
        row["training"] = row["training"][:1]
        row["validation"] = row["validation"][:1]
    return [request, old, baseline, first, C.S.digest(raw)]


class ControlsTests(unittest.TestCase):
    def test_both_metrics_and_lineage(self):
        result = C.verify(*fixture())
        self.assertEqual(result["existing_cases_identical"], 3)
        self.assertEqual(len(result["pairs"]), 2)
        self.assertTrue(result["passed"])

    def test_changed_inputs_and_first_step_fail(self):
        for variant in ("schedule", "old_init", "new_init", "prefix", "frozen_weight",
                        "backbone", "no_embedding_update", "no_geometry_update", "metric",
                        "missing_case", "count", "runtime", "extra_evaluation", "coordinated_extra_evaluation"):
            args = fixture()
            request, old, baseline, first, _ = args
            if variant == "schedule":
                old["rate"] += .01
            elif variant == "old_init":
                old["cases"][0]["parameters"][0]["values"][0] += .01
            elif variant == "new_init":
                request["cases"][3]["parameters"][2]["values"][0] += .01
            elif variant == "prefix":
                first["cases"][1]["training"][0]["ce"] += .01
            elif variant == "frozen_weight":
                first["cases"][4]["final_parameters"][2][0] += .01
            elif variant == "backbone":
                first["cases"][1]["final_parameters"][0][0] += .01
            elif variant == "no_embedding_update":
                for i in (1, 2):
                    first["cases"][i]["final_parameters"][0] = request["cases"][i]["parameters"][0]["values"]
            elif variant == "no_geometry_update":
                for i, p in enumerate(request["cases"][1]["parameters"]):
                    if p["name"].startswith("geometry."):
                        first["cases"][1]["final_parameters"][i] = p["values"]
            elif variant == "metric":
                first["cases"][3]["pair_metric"] = C.S.POINCARE
            elif variant == "missing_case":
                first["cases"].pop()
            elif variant == "count":
                first["cases"][4]["trainable_parameter_scalars"] += 1
            elif variant == "runtime":
                first["adapter"] = "different"
            elif variant == "coordinated_extra_evaluation":
                point = copy.deepcopy(baseline["cases"][0]["validation"][0])
                point["revision"] = 1
                baseline["cases"][0]["validation"].insert(1, point)
                first["cases"][0]["validation"].append(point)
            else:
                first["cases"][0]["validation"].append(copy.deepcopy(baseline["cases"][0]["validation"][-1]))
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                C.verify(*args)


if __name__ == "__main__":
    unittest.main()
