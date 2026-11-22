"""Controls with known coefficients, ties and before-plan refusal."""
import json
import math
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
                             gold=int(s == 1), rival=int(s == -1), prefix=[0, s+2, 4, 5, a+7, 10]))
            margins[key] = s * (bias + s*subject + a*attractor + s*a*interaction)
    return rows, margins


class AgreementReadout(unittest.TestCase):
    def test_full_vocab_loss_separates_pair_choice_from_coverage(self):
        row = M["vocabulary_row"]
        a = row([0., 0., math.log(2)], 0, 1)
        self.assertAlmostEqual(a["pair_mass"], .5)
        self.assertAlmostEqual(a["choice_nll"], math.log(2))
        self.assertAlmostEqual(a["gold_nll"], math.log(4))
        self.assertEqual(a["greedy"], "other")
        x, y, z = [row(v, 0, 1) for v in ([0.,2.,-3.], [2.,0.,-3.], [2.,0.,8.])]
        self.assertEqual(x["pair_mass"], y["pair_mass"])
        self.assertGreater(x["choice_nll"], y["choice_nll"])
        self.assertEqual(y["choice_nll"], z["choice_nll"])
        self.assertLess(y["coverage_nll"], z["coverage_nll"])
        self.assertEqual(z, row([66.,64.,72.], 0, 1))

    def test_extreme_gold_logit_two_class_vocab_underflow_and_top_tie(self):
        row = M["vocabulary_row"]
        r = row([-1e30, 0., 0.], 0, 1)
        self.assertEqual(r["gold_nll"] - r["choice_nll"], 0.)
        self.assertAlmostEqual(r["coverage_nll"], math.log(2))
        self.assertAlmostEqual(r["pair_mass"], .5)
        self.assertEqual((r["greedy"], r["top_tie"]), ("rival", True))
        z = row([0., 0.], 0, 1)
        self.assertEqual((z["coverage_nll"], z["pair_mass"], z["greedy"], z["top_tie"]), (0.,1.,"gold",True))
        extreme = row([0.,0.,1000.],0,1)
        self.assertEqual(extreme["pair_mass"], 0.)
        self.assertAlmostEqual(extreme["coverage_nll"], 1000.-math.log(2))
        for values, a, b in [([0., float("nan")],0,1), ([1.],0,0), ([1.,2.],0,2)]:
            with self.assertRaises(ValueError):
                row(values,a,b)

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
                            ([dict(rows[0], prefix=[9]+rows[0]["prefix"][1:]), *rows[1:]], margins),
                            ([dict(rows[0], prefix=rows[2]["prefix"]), *rows[1:]], margins),
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
