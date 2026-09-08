"""Admission-only tests for client evidence replay; no claimed GPU execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("client_validator",
    Path(__file__).resolve().parents[1] / "tools/validate_resident_training_clients_vs_torch.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ReportPath:
    def __init__(self, report):
        self.raw = json.dumps(report).encode()

    def read_bytes(self):
        return self.raw

    def __str__(self):
        return "synthetic-admission-fixture"


def fixtures():
    cases = [dict(seed=seed, shape=list(shape), kernel=k, accumulation=a)
             for seed, shape in sorted(validator.RECIPES) for k, a in sorted(validator.POLICIES)]
    plan = "synthetic-bytes-for-digest-admission-only"
    digest = hashlib.sha256(plan.encode()).hexdigest()
    python = dict(schema="spiraltorch.nn.training_client_fixture.v1", status="passed", vjps=cases,
        learning=[dict(seed=seed, steps=128, learning_rate=.2, shape=[2, 16, 4],
                       half_plan_json=plan, half_plan_sha256=digest) for seed in (17, 29, 43)])
    browser = dict(schema="spiraltorch.nn.training_browser_fixture.v1", status="passed", cpu_only=False,
        vjps=copy.deepcopy(cases), nonfinite_cases=[{} for _ in range(6)],
        learning=[dict(seed=seed, steps=64, learning_rate=.2, plan_json=plan,
                       plan_sha256=digest, source_plan_sha256=digest) for seed in (17, 29, 43)])
    return python, browser


class Admission(unittest.TestCase):
    def reject_mutation(self, mutate):
        python, browser = fixtures()
        mutate(python, browser)
        with self.assertRaises(ValueError):
            validator.load_pair(ReportPath(python), ReportPath(browser))

    def test_recipe_and_input_digests_are_retained(self):
        python, browser = fixtures()
        a, b = ReportPath(python), ReportPath(browser)
        actual_python, actual_browser, sources = validator.load_pair(a, b)
        self.assertEqual(actual_python, python)
        self.assertEqual(actual_browser, browser)
        self.assertEqual([s["sha256"] for s in sources], [hashlib.sha256(p.raw).hexdigest() for p in (a, b)])

    def test_cpu_only_failed_wrong_schema_or_missing_guards(self):
        for key, value in (("cpu_only", True), ("status", "error"), ("schema", "unknown"), ("nonfinite_cases", [])):
            with self.subTest(key=key):
                self.reject_mutation(lambda p, b: b.update({key: value}))

    def test_missing_or_duplicate_vjp_is_not_full_matrix(self):
        for side in (0, 1):
            self.reject_mutation(lambda p, b: (p, b)[side]["vjps"].pop())
            self.reject_mutation(lambda p, b: (p, b)[side]["vjps"].__setitem__(0, (p, b)[side]["vjps"][1]))

    def test_recipe_or_weight_lineage_drift(self):
        for key, value in (("steps", 63), ("learning_rate", .1), ("source_plan_sha256", "unknown"),
                           ("plan_json", "changed"), ("seed", 999)):
            with self.subTest(key=key):
                self.reject_mutation(lambda p, b: b["learning"][0].update({key: value}))
        self.reject_mutation(lambda p, b: p["learning"][0].update(half_plan_json="changed"))
        self.reject_mutation(lambda p, b: p["learning"][0].update(steps=127))
        self.reject_mutation(lambda p, b: p["learning"][0].update(seed=29))


if __name__ == "__main__":
    unittest.main()
