"""Controls with known coefficients, ties and before-plan refusal."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

M = runpy.run_path(str(Path(__file__).with_name("agreement_readout.py")))


def design(bias, subject, attractor, interaction, group="x"):
    rows, margins = [], {}
    for s in (-1, 1):
        for a in (-1, 1):
            key = "%s/%s/%s" % (group, s, a)
            rows.append(dict(key=key, pair=group, plural=s == 1, conflict=s != a,
                             gold=int(s == 1), rival=int(s == -1)))
            margins[key] = s * (bias + s*subject + a*attractor + s*a*interaction)
    return rows, margins


class AgreementReadout(unittest.TestCase):
    def test_constant_verb_prior_changes_choices_not_subject_response(self):
        b = M["describe"](*design(0, 1, 0, 0))
        c = M["describe"](*design(7, 1, 0, 0))
        self.assertEqual(b["pair_outcomes"]["both_correct"], 2)
        self.assertEqual((c["pair_outcomes"]["both_correct"], c["pair_outcomes"]["always_plural"]), (0, 2))
        self.assertEqual((b["orders"]["correct_order"], c["orders"]["correct_order"]), (2, 2))
        self.assertEqual(b["coefficients"]["subject"], c["coefficients"]["subject"])
        self.assertEqual(c["coefficients"], dict(verb_bias=7, subject=1, attractor=0, interaction=0))

    def test_primary_can_advance_without_any_new_correct_pair(self):
        rows, before = design(-10, 0, 0, 0)
        _, after = design(-10, .125, 0, 0)
        scores = lambda vals: {k: dict(margin=m, nll=5.) for k, m in vals.items()}
        s = {k: scores(before) for k in ("initial-r1", "initial-r4", "1-r1")}
        s["1-r4"] = scores(after)
        primary = M["E"]["contrast"](dict(evaluation=rows, seeds=[1], min_effect=.01), s)
        self.assertTrue(primary["advance"])
        self.assertEqual(primary["orders"][0]["paired_gain"]["all"], .125)
        for values in (before, after):
            self.assertEqual(M["describe"](rows, values)["pair_outcomes"]["always_singular"], 2)

    def test_interaction_can_reverse_one_fixed_attractor_despite_positive_mean(self):
        r = M["describe"](*design(0, 1, 0, 4))
        self.assertEqual((r["pair_outcomes"]["both_correct"], r["pair_outcomes"]["both_wrong"]), (1, 1))
        self.assertEqual(r["orders"], dict(correct_order=1, reverse_order=1, tied=0))
        self.assertEqual(r["quartets"]["x"]["by_attractor"]["singular"]["switch"], -6)
        self.assertEqual(r["coefficients"]["subject"], 1)

    def test_attractor_only_and_exact_ties(self):
        r = M["describe"](*design(0, 0, 3, 0))
        self.assertEqual(r["coefficients"], dict(verb_bias=0, subject=0, attractor=3, interaction=0))
        self.assertEqual(r["orders"]["tied"], 2)
        for bias, sub, ties in [(0, 0, 4), (1, 1, 2)]:
            r = M["describe"](*design(bias, sub, 0, 0))
            self.assertEqual((r["pair_outcomes"]["boundary_tie"], r["pairwise_gold_ties"]), (2, ties))

    def test_aggregation_keeps_lexical_groups_and_all_factor_signs(self):
        a, x = design(2, -1, 3, -4, group="one")
        b, y = design(-4, 3, 7, 2, group="two")
        r = M["describe"]((b+a)[::-1], dict(x, **y))
        self.assertEqual(r["coefficients"], dict(verb_bias=-1, subject=1, attractor=5, interaction=-1))
        self.assertEqual(r["quartets"]["one"]["coefficients"]["subject"], -1)
        self.assertEqual(r["quartets"]["two"]["coefficients"]["subject"], 3)

    def test_incomplete_duplicate_misoriented_and_nonfinite_rows_rejected(self):
        rows, margins = design(0, 1, 0, 0)
        for bad, values in [(rows[:-1], margins), (rows + rows[:1], margins),
                            ([dict(rows[0], gold=1), *rows[1:]], margins),
                            (rows, dict(margins, extra=1)),
                            (rows, {**margins, rows[0]["key"]: float("nan")})]:
            with self.assertRaises(ValueError):
                M["describe"](bad, values)

    def test_failed_or_absent_attempt_and_wrong_result_rejected_before_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "attempt"
            with self.assertRaises(FileNotFoundError) as e:
                M["completed"](root, "gpt2", "0"*64)
            self.assertEqual(Path(e.exception.filename), attempt / "receipt.json")
            attempt.mkdir()
            receipt = dict(success=False, retry=False, exit_code=0, seconds_limit=600, elapsed=1, min_free=5<<30)
            (attempt / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "successful bounded attempt"):
                M["completed"](root, "gpt2", "0"*64)
            receipt["success"] = True
            (attempt / "receipt.json").write_text(json.dumps(receipt))
            (attempt / "result.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "wrong sealed result"):
                M["completed"](root, "gpt2", "0"*64)
            self.assertFalse((root / "plan.json").exists())


if __name__ == "__main__":
    unittest.main()
