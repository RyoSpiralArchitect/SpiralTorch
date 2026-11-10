import unittest

from alias_common_trajectory import endpoints
from alias_common_trajectory_audit import decision, read_scores, summarize, validate_step
from alias_subset_study import metrics


def path(costs, *, eps=5e-6):
    points = [[dict(cost=v) for v in row] for row in costs]
    steps = [dict(length=.002, min_normalized_margin=.1, cosine_to_first=1.,
                  predicted=[b - a for a, b in zip(x, y, strict=True)])
             for x, y in zip(costs[:-1], costs[1:], strict=True)]
    return summarize(points, steps, eps)


class TrajectoryTests(unittest.TestCase):
    def test_one_reversal_cannot_be_hidden_by_better_endpoint(self):
        common = path([[1., 1., 1.], [.9, .9, .9], [.8, .8, .8]])
        rebound = path([[1., 1., 1.], [.9, .9, .75], [.85, .85, .8]])
        equal = path([[1., 1., 1.], [.8, .8, 1.2], [.7, .7, 1.3]])
        g = decision(dict(common=common, initial_ray=rebound, equal=equal), 2, 5e-6)
        self.assertEqual((rebound["class_descents"], rebound["first_later_harm"]), ([2, 2, 1], 2))
        self.assertTrue(g["common_avoids_later_ray_harm"])
        self.assertTrue(g["equal_harms_idle_at_end"])
        self.assertTrue(g["common_better_worst_gain"])
        self.assertFalse(decision(dict(common=rebound, initial_ray=rebound, equal=equal), 2, 5e-6)["common_all_groups_every_step"])

    def test_neutral_is_not_harm_and_larger_gains_are_a_separate_claim(self):
        small = path([[1.] * 3, [.9] * 3, [.8, .8, .9]])
        good = path([[1.] * 3, [.9] * 3, [.8] * 3])
        self.assertEqual(small["first_nonstrict"], 2)
        self.assertIsNone(small["first_later_harm"])
        g = decision(dict(common=good, initial_ray=small, equal=small), 2, 5e-6)
        self.assertTrue(g["common_better_worst_gain"])
        self.assertFalse(g["common_avoids_later_ray_harm"])
        self.assertFalse(g["equal_harms_idle_at_end"])

    def test_weaker_alias_identity_is_fixed_not_reranked(self):
        lines = [7., 5., 5., 3., 5., 5.]
        m = metrics(lines, 0)
        scores = read_scores(dict(lines=lines, cost=m["conditional"], absolute=m["absolute"]), 0, 3)
        self.assertEqual((scores["baseline_weak"], scores["worst_alias"]), (3., 7.))
        with self.assertRaises(ValueError):
            read_scores(dict(lines=lines, cost=m["conditional"] + .02, absolute=m["absolute"]), 0, 3)

    def test_ray_checks_direction_not_only_step_length(self):
        start, before, after = [0., 0.], [.002, 0.], [.004, 0.]
        samples = [dict(v=[1., 1.])] * 3
        pl = dict(radius=.002, groups=[[0], [1], [2]])
        r = dict(origin_id="o", direction_id="f", result_id="sha256:p", trainable_values=64,
                 requested_l2=.004, realized_l2=.004, direction_l2=.002, relative_vector_error=0.)
        step = dict(post="p", length=.002, receipt=r)
        args = ("initial_ray", 1, "o", "f", {}, step, before)
        self.assertEqual(validate_step(*args, after, start, before, samples, pl)["cosine_to_first"], 1.)
        with self.assertRaises(ValueError):
            validate_step(*args, [.002, .002], start, before, samples, pl)
        r["direction_id"] = "foreign"
        with self.assertRaises(ValueError):
            validate_step(*args, after, start, before, samples, pl)

    def test_incomplete_previous_scores_refuse_to_freeze(self):
        base = dict(source="initial", policy="common", radius=1)
        raw = [dict(**base, event="proposal", pin="p", receipt={})]
        raw.extend(dict(**base, event="score", slot=i, cost=0., weak=0., absolute=0.) for i in range(18))
        self.assertEqual(endpoints(raw, "initial", "common")["artifact"], "p")
        raw[-1]["slot"] = 16
        with self.assertRaises(ValueError):
            endpoints(raw, "initial", "common")
