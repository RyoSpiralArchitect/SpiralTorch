import unittest

from alias_protection import control_point
from alias_protection_audit import certify, decision, summarize, token_costs


class ProtectionTests(unittest.TestCase):
    def test_four_unequal_raw_mean_norms_require_different_weights(self):
        groups, delta = [], [-.01] * 4 + [0.] * 60
        for i in range(4):
            v = [0.] * 64
            v[i] = 2. ** i
            groups.append([dict(v=v, loss=.5)])
        r = dict(unit_group_weights=[.25] * 4, raw_group_weights=[8/15, 4/15, 2/15, 1/15],
                 group_mean_losses=[.5] * 4, group_gradient_l2=[1., 2., 4., 8.],
                 group_loss_dots=[-.01, -.02, -.04, -.08], predicted_min_unit_margin=.5,
                 applied_min_unit_margin=.5, hull_norm=.5)
        self.assertAlmostEqual(certify(groups, delta, r)["min_margin"], .5)
        with self.assertRaises(ValueError):
            certify(groups, [-x for x in delta], r)
        with self.assertRaises(ValueError):
            certify(groups, delta, dict(r, raw_group_weights=r["unit_group_weights"]))
        with self.assertRaises(ValueError):
            certify(groups, delta, dict(r, unit_group_weights=[.5, .25, .25, 0.]))

    def test_replay_requires_the_original_gradient_and_all_line_scores(self):
        r = dict(slot=0, cost=.2, absolute=.9, lines=[1,2,3,4,5,6], loss=.2, v=[1.])
        self.assertEqual(control_point([r]), control_point([dict(r, singletons=[{"index": 5}])]))
        self.assertNotEqual(control_point([r]), control_point([dict(r, v=[2.])]))
        self.assertNotEqual(control_point([r]), control_point([dict(r, lines=r["lines"][::-1])]))

    def test_word_and_stop_are_not_interchangeable_between_aliases(self):
        terms = [1., 2., 4., 8., 16.]
        row = dict(terms=terms, lines=[0., 0., 7., 0., 0., 25.])
        self.assertEqual(token_costs(row, 1e-7), terms)
        with self.assertRaises(ValueError):
            token_costs(dict(row, terms=[1., 2., 16., 8., 4.]), 1e-7)

    def test_primary_controls_and_prompt_level_benefit_are_separate(self):
        keys = ["on", "off", "idle", "standby_line", "standby_menu_line", "standby_within_accepted"]
        a, b, c = [{k: v for k in keys} for v in (2., 1.9, 1.8)]
        c["standby_within_accepted"] = 2.1
        steps = [dict(min_margin=.2, unit_weights=[.2,.2,.3,.3])] * 2
        s = summarize([a,b,c], steps, [-.1] * 5 + [.01], [0.,0.,0.,.1,-.3], 5e-6)
        other = dict(s, changes=dict(s["changes"], standby_line=-.2))
        g = decision(dict(guard_standby=s, three=other, guard_idle=other), 2, 5e-6)
        self.assertTrue(g["primary"])
        self.assertFalse(any(g[k] for k in ("beats_both_controls", "all_six_benefit", "standby_word_gains", "standby_share_gains")))
        b["standby_menu_line"] = 2.1
        miss = summarize([a,b,c], steps, [-.1] * 6, [0.] * 5, 5e-6)
        self.assertEqual((miss["jointly_descending"], miss["first_missed_step"]), (1, 1))
