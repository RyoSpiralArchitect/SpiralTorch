import copy
import hashlib
from pathlib import Path
import runpy
import unittest

protocol = runpy.run_path(str(Path(__file__).with_name("qualify.py")))


def fixture():
    cases, resumed, reference = {}, [], {"cases": []}
    for owner in ("base", "learned"):
        for offset in (0, 128, 512):
            for kind in protocol["CONFIGS"]:
                name = f"{owner}-{offset}-{kind}"
                steps = [{"step": i, "token_id": 1} for i in range(1, 9)]
                cases[name] = {"generated": [1]*8, "prompt": [2]*8, "tokens": [2]*8+[1]*8, "steps": steps,
                               "payload": "checkpoint", "pin": "sha256:" + hashlib.sha256(b"checkpoint").hexdigest()}
                resumed.append({"case": name, "steps_exact": True, "checkpoint_exact": True, "continued_tokens": 5})
                if kind != "zspace":
                    reference["cases"].append({"case": name, "steps": [{"step": i, "probability_abs": 0.,
                        "expected_probability": .5, "entropy_abs": 0., "greedy_exact": True,
                        "candidate_count_exact": True, "within_gate": True} for i in range(1, 9)]})
    return cases, resumed, reference


class QualificationTests(unittest.TestCase):
    def test_complete_valid_fixture(self):
        protocol["validate_results"](*fixture())

    def test_missing_duplicate_or_truncated_rows_fail(self):
        for edit in [lambda c, r, t: r.pop(), lambda c, r, t: r.append(r[0]),
                     lambda c, r, t: t["cases"].pop(), lambda c, r, t: t["cases"][0]["steps"].pop(),
                     lambda c, r, t: c.pop(next(iter(c)))]:
            values = fixture()
            edit(*values)
            with self.assertRaises(ValueError):
                protocol["validate_results"](*values)

    def test_failed_nonfinite_and_boolean_metrics_fail(self):
        for key, value in [("probability_abs", .1), ("entropy_abs", float("nan")), ("probability_abs", True),
                           ("expected_probability", 2.), ("greedy_exact", False), ("within_gate", False), ("step", 2)]:
            values = fixture()
            values[2]["cases"][0]["steps"][0][key] = value
            with self.assertRaises(ValueError):
                protocol["validate_results"](*values)

    def test_bad_resume_and_sequence_accounting_fail(self):
        original = fixture()
        for edit in [lambda c, r, t: r[0].update(continued_tokens=4), lambda c, r, t: r[0].update(steps_exact=False),
                     lambda c, r, t: next(iter(c.values())).update(pin="bad"),
                     lambda c, r, t: next(iter(c.values()))["steps"][0].update(token_id=2)]:
            values = copy.deepcopy(original)
            edit(*values)
            with self.assertRaises(ValueError):
                protocol["validate_results"](*values)


if __name__ == "__main__":
    unittest.main()
