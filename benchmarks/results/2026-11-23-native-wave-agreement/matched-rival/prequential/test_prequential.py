import math
from pathlib import Path
import runpy
import struct
import unittest

T = runpy.run_path(str(Path(__file__).with_name("study.py")))
M, D = T["M"], T["D"]


def coords(*values):
    return M["coords"](struct.pack("<%sf"%len(values), *values))


class Prequential(unittest.TestCase):
    def test_signed_displacement_and_cancellation(self):
        move = M["displacement"](coords(2., -1., 4.), coords(1., 1., 4.))
        self.assertEqual(M["tangent"](coords(.5, .25, 3.), move), 0.)
        self.assertEqual(M["tangent"](coords(.5, .5, 3.), move), -.5)
        with self.assertRaises(ValueError):
            M["tangent"](coords(1.), move)

    def test_smallest_f32_movement(self):
        x = 2.**-149
        self.assertEqual(M["tangent"](coords(-x), M["displacement"](coords(x), coords(0.))), -x*x)
        with self.assertRaises(ValueError):
            coords(float("inf"))

    def test_controls_match_both_bits_and_exclude_own_family(self):
        rows = [dict(key=str(k), prefix=[k], pair=str(k//4), plural=bool(k%4//2), conflict=bool(k%2))
                for k in range(12)]
        self.assertEqual(M["controls"](rows)["1"], ["5", "9"])
        self.assertEqual(M["controls"](rows[:4])["1"], [])
        for bad in (rows[:-1], rows+[rows[0]], [dict(r, prefix=[0]) for r in rows]):
            with self.assertRaises(ValueError):
                M["controls"](bad)

    def test_null_signs_rmse_and_strict_ties(self):
        rows = [dict(key=str(i), prior=i+1, seed=1, change=y, tangent=p, zero=0., lexical=None)
                for i, (y, p) in enumerate([(1., -1.), (-2., -1.), (0., 0.)])]
        result = M["errors"](rows, "tangent")
        self.assertEqual((result["sign_same"], result["sign_opposite"], result["sign_tied"]), (1, 1, 1))
        self.assertAlmostEqual(result["rmse"], math.sqrt(5/3))
        self.assertEqual(result["rmse"], M["errors"](rows, "zero")["rmse"])
        self.assertIsNone(M["errors"](rows, "lexical"))
        rows[1]["lexical"] = 0.
        with self.assertRaises(ValueError):
            M["errors"](rows, "lexical")

    def test_distinct_prefix_weighting(self):
        rows = [dict(host=h, seed=s, prior=i, key=k, margin=1., margin0=1., change=x, tangent=x,
                     lexical=None, zero=0., rival_minus_full_nll=x)
                for h in ("affine", "both") for s, i, k, x in
                [(1, 1, "a", 3.), (1, 2, "b", 0.), (2, 1, "a", 3.), (2, 2, "c", 0.)]]
        q = M["summarize"](rows, [1, 2])["affine"]
        self.assertEqual(q["all_visits"]["observed_mean"], 1.5)
        self.assertEqual(q["equal_prefix"]["mean_change"], 1.)
        self.assertTrue(q["beats_zero_each_order"])
        self.assertIsNone(q["beats_lexical_each_order"])

    def test_incoming_state_not_outgoing_and_broken_chain(self):
        state = {k: coords(a, 0.) for k, a in (("initial", 1.), ("next", .75), ("last", .5))}
        tape = struct.pack("<ff", 1., 0.)
        cases = [dict(key="case"+str(i), prefix=[i], gold=1) for i in (0, 1)]
        trace = dict(method="gradient_ray_f32.v1", requested_l2=.25, realized_l2=.25, trainable_values=2,
                     gradient_l2=1., gradient_dot_step=-.25, relative_vector_error=0.)
        log = [dict(key=c["key"], position=1, gold=1, rival=None, objective="full", loss=.5, declared=None,
                    grad_sha256=T["sha"](tape), step=dict(trace, origin_id=u, result_id=v))
               for c, u, v in zip(cases, ("initial", "next"), ("next", "last"))]
        def parse():
            read = {"stage.jsonl": b"".join(T["enc"](r) for r in log), "stage.g32": tape+tape}.__getitem__
            return list(T["checked_path"](read, {"stage":"last"}, "stage", "full", cases, "initial",
                                          state.__getitem__, dict(shape=[1, 1], target_l2=.25, tolerance=1e-4)))
        self.assertEqual([x[1][0][0] for x in parse()], [1., .75])
        log[1]["step"]["origin_id"] = "last"
        with self.assertRaises(ValueError):
            parse()

    def test_only_observed_anchor_and_loss_decomposition(self):
        start = dict(host="affine", key="a", seed=1, prior=0, margin=-1., full_nll_rival=5., full_nll_ordinary=5.)
        later = dict(start, seed=2, prior=2, margin=1., full_nll_rival=4.75, full_nll_ordinary=4.5)
        records = T["anchor_changes"]([later, dict(later, key="no-initial"), start])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["within_pair"], -1.)
        self.assertEqual(records[0]["coverage"], .75)
        self.assertEqual(records[0]["full_ordinary"], -.5)


if __name__ == "__main__":
    unittest.main()
