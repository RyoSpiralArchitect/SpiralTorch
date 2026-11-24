"""Analytic checks; no native model is imported."""
import math
from pathlib import Path
import runpy
import unittest

A = runpy.run_path(str(Path(__file__).with_name("analyze.py")))


def example(pairs, vector):
    labels = [(str(p), h, a) for p in range(pairs) for h in (False, True) for a in (False, True)]
    steps = [vector(int(p), 1 if h else -1, 1 if a else -1) for p,h,a in labels]
    return labels, steps, [math.fsum(col) for col in zip(*steps)]


class PathTests(unittest.TestCase):
    def test_opposite_subjects_cancel_gate_not_bias(self):
        r = A["describe"](*example(6, lambda p,h,a: [h, 1]))
        self.assertEqual(r["n"], 24)
        self.assertAlmostEqual(r["retention_squared"]["total"], .5)
        self.assertAlmostEqual(r["retention_squared"]["gate"], 0.)
        self.assertAlmostEqual(r["gram"]["diagonal"]["total"], 1/24)
        self.assertEqual([r["pairs"][k]["n"] for k in A["BINS"]], [60,72,72,72])
        for k in ("flip_same","flip_flip"):
            self.assertAlmostEqual(r["gram"][k]["total"], 0.)
            self.assertLess(r["gram"][k]["gate"], 0)
            self.assertGreater(r["gram"][k]["bias"], 0)
        for k in A["FLIPS"]:
            self.assertEqual(r["matched"][k]["n"], 6)
            self.assertAlmostEqual(r["matched"][k]["mean_cosine"], 0. if k.startswith("subject") else 1.)

    def test_known_factor_contrasts_and_residual(self):
        def vec(p,h,a):
            return [2, 3*h, 4*a, 5*h*a+(6 if p else -6)]
        labels, steps, end = example(2, vec)
        r = A["describe"](labels, steps, end)
        for k,val,block in (("common",4,"gate"),("subject",9,"gate"),("attractor",16,"bias"),
                            ("interaction",25,"bias"),("within_cell",36,"bias")):
            self.assertAlmostEqual(r["energy_fraction"][k]["total"], val/90)
            self.assertAlmostEqual(r["energy_fraction"][k][block], val/90)
        reverse = A["describe"](labels[::-1], steps[::-1], end)
        self.assertEqual(r, reverse)

    def test_angular_flip_and_gap_gates_are_distinct(self):
        wave = A["describe"](*example(1, lambda p,h,a: [h, 0]))
        affine = A["describe"](*example(1, lambda p,h,a: [h, 1]))
        gap = A["difference"](wave, affine)
        self.assertAlmostEqual(gap["gap"]["total"], .5)
        self.assertAlmostEqual(gap["opposite_subject"]["total"], .5)
        self.assertTrue(gap["opposite_majority"] and gap["stronger_subject_reversal"])
        self.assertFalse(A["difference"](affine, wave)["opposite_majority"])
        self.assertIsNone(wave["pairs"]["same_same"]["mean_cosine"])
        attractor = A["describe"](*example(1, lambda p,h,a: [a, 0]))
        self.assertAlmostEqual(attractor["matched"]["subject_at_pl"]["mean_cosine"], 1.)
        self.assertAlmostEqual(attractor["matched"]["attractor_at_sg"]["mean_cosine"], -1.)
        self.assertFalse(A["difference"](attractor, affine)["stronger_subject_reversal"])

    def test_endpoint_invariant_is_not_an_attribution(self):
        # Exchanging the roles of subject and attractor leaves the endpoint intact.
        x = A["describe"](*example(1, lambda p,h,a: [h, 1]))
        y = A["describe"](*example(1, lambda p,h,a: [a, 1]))
        self.assertEqual(x["retention_squared"], y["retention_squared"])
        self.assertNotEqual(x["gram"]["flip_same"], y["gram"]["flip_same"])
        self.assertNotEqual(x["energy_fraction"]["subject"], y["energy_fraction"]["subject"])

    def test_invalid_or_incomplete_paths(self):
        labels, steps, end = example(2, lambda p,h,a: [a, 1])
        for changed in ((labels, steps, [0,0]), (labels[:-1], steps[:-1], end),
                        ([labels[0]]+labels[:-1], steps, end), (labels, [[math.nan,0]]+steps[1:], end),
                        (labels, [[0,0]]*len(steps), [0,0]), (labels, steps, [0,0,1])):
            with self.assertRaises(ValueError):
                A["describe"](*changed)

    def test_input_roles_exclude_predictions_held_out_report_and_other_runs(self):
        role = lambda p: A["input_role"](Path(p), "gpt2")
        self.assertEqual(role("gpt2/attempt/17-wave_rival.g32"), "gradient_tape")
        self.assertEqual(role("gpt2/attempt/states/example.json"), "state")
        for p in ("gpt2/attempt/17-wave_full.f32le", "gpt2/attempt/worker-result.json",
                  "rehearsal/plan.json", "RESULTS.md", "gpt2/model.safetensors"):
            with self.assertRaises(ValueError):
                role(p)


if __name__=="__main__":
    unittest.main()
