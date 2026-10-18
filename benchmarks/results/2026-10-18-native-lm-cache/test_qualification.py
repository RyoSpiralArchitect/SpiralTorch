"""Publication refuses incomplete or failed measurements; no ML dependencies."""
import copy
import hashlib
from pathlib import Path
import runpy
import unittest

q = runpy.run_path(str(Path(__file__).with_name("qualify.py")))
retain = runpy.run_path(str(Path(__file__).with_name("retain_failure.py")))["failure_receipt"]


def fixture():
    names = [f"{owner}-{offset}-{kind}" for owner in ["base", "learned"]
             for offset in [0, 128, 512] for kind in q["CONFIGS"]]
    case = {"generated": [1]*8, "prompt": [2]*8, "tokens": [2]*8+[1]*8,
            "steps": [{"step": i, "token_id": 1} for i in range(1, 9)],
            "payload": "test", "pin": "sha256:" + hashlib.sha256(b"test").hexdigest()}
    cases = {name: copy.deepcopy(case) for name in names}
    metric = {"max_logit_abs": 1e-5, "argmax_exact": True, "within_gate": True}
    teacher = [{"owner": owner, "length": length,
                "steps": [{"step": i, **metric} for i in range(9)]}
               for owner in ["base", "learned"] for length in q["LENGTHS"]]
    resumed = [{"mode": mode, "case": name, "before_exact": True, "steps_exact": True,
                "checkpoint_exact": True, "continued_tokens": 5}
               for mode in ["cached", "old_full_prefix"] for name in names]
    timing = {"rows": [{"arm": arm, "length": length, "round": i, "warmup": i < 2,
                         "prefill_ms": .1, "decode_ms": .2, "total_ms": .3, **metric}
                        for arm in q["ARMS"] for length in q["TIMING_LENGTHS"] for i in range(9)]}
    return cases, teacher, resumed, timing


class PublicationTests(unittest.TestCase):
    def test_complete_fixture(self):
        q["validate"](*fixture())

    def test_missing_or_duplicate_conditions(self):
        for index in (1, 2, 3):
            for duplicate in (False, True):
                args = fixture()
                rows = args[index]["rows"] if index == 3 else args[index]
                rows.pop()
                if duplicate:
                    rows.append(copy.deepcopy(rows[0]))
                with self.assertRaises(ValueError):
                    q["validate"](*args)

    def test_false_or_nonfinite_metrics(self):
        for key, value in [("max_logit_abs", float("nan")), ("max_logit_abs", float("inf")),
                           ("max_logit_abs", True), ("max_logit_abs", .1),
                           ("argmax_exact", False), ("within_gate", False),
                           ("decode_ms", 0), ("total_ms", .4), ("warmup", False)]:
            args = fixture()
            args[3]["rows"][0][key] = value
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_incomplete_teacher_and_resume(self):
        args = fixture()
        args[1][0]["steps"].pop()
        with self.assertRaises(ValueError):
            q["validate"](*args)
        for key, value in [("checkpoint_exact", False), ("before_exact", False),
                           ("steps_exact", False), ("continued_tokens", 4)]:
            args = fixture()
            args[2][0][key] = value
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_changed_pin_and_step_identity(self):
        for key, value in [("pin", "sha256:wrong"), ("tokens", [3])]:
            args = fixture()
            next(iter(args[0].values()))[key] = value
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_retention_cannot_turn_failure_into_pass_or_hide_bad_rows(self):
        with self.assertRaises(ValueError):
            retain(*fixture())
        args = fixture()
        args[3]["rows"][0].update(max_logit_abs=.001, within_gate=False)
        result = retain(*args)
        self.assertEqual(result["status"], "UNQUALIFIED_LOGIT_GATE_FAILED")
        self.assertEqual(len(result["failed_rows"]), 1)
        for key, value in [("max_logit_abs", float("nan")), ("within_gate", True)]:
            changed = copy.deepcopy(args)
            changed[3]["rows"][-1][key] = value
            if key == "within_gate":
                changed[3]["rows"][-1]["max_logit_abs"] = .1
            with self.assertRaises(ValueError):
                retain(*changed)


if __name__ == "__main__":
    unittest.main()
