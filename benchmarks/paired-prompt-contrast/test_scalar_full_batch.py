import copy
import unittest

from scalar_binding_readback import pair_means
from scalar_full_batch import FLOOR, TRIALS, batch, prior
from scalar_full_batch_audit import check_step, cycle, proposal


def observed(margin):
    pairs = [[margin, 0.0] for _ in range(32)]
    return dict(pairs=pairs, loss=pair_means([[p, n, 0.0, 0.0] for p, n in pairs])[0])


def report(cells):
    baseline = observed(0.0)
    deriv = [[0.0, 0.0, 1.0, 0.0]] * 32
    loss, g = pair_means(deriv)
    return dict(
        baseline=baseline,
        gradient_loss=loss,
        gradient=g,
        derivatives=deriv,
        before=0.0,
        trials=[
            dict(coefficient=c, required_decrease=max(FLOOR, -0.0001 * g * c), objective=observed(m))
            for c, m in cells
        ],
    )


class FullBatchTests(unittest.TestCase):
    def test_pair_budget_and_disjoint_training_views(self):
        train = prior.task.deck("train")
        pairs = batch(train)
        self.assertEqual(6 * len(pairs), 48 * 4)
        self.assertEqual({x["id"] for pair in pairs for x in pair}, {r["id"] for r in train})
        with self.assertRaises(ValueError):
            batch(train[:-1])

    def test_guard_must_accept_first_sufficient_drop_not_merely_any_drop(self):
        r = report([(0.08, -0.01), (0.04, 0.02)]) | dict(status="accepted", after=0.04, step=1)
        end = r["trials"][-1]["objective"]
        self.assertEqual(check_step("guarded", 0.0, 0, r, end), (False, 2))
        for bad in ("skipped", "wrong_step", "wrong_bound"):
            z = copy.deepcopy(r)
            if bad == "skipped":
                z["trials"][0]["objective"] = observed(0.1)
            if bad == "wrong_step":
                z["after"] = 0.08
            if bad == "wrong_bound":
                z["trials"][1]["required_decrease"] *= 2
            with self.assertRaises(ValueError):
                check_step("guarded", 0.0, 0, z, end)

    def test_ties_cannot_advance_and_no_unobserved_early_stop(self):
        r = report([(0.08 / (2**i), 0.0) for i in range(TRIALS)]) | dict(
            status="trials_exhausted", after=0.0, step=3
        )
        self.assertEqual(check_step("guarded", 0.0, 3, r, r["baseline"]), (True, TRIALS))
        for key, value in (("step", 4), ("trials", r["trials"][:-1]), ("after", 0.001)):
            with self.assertRaises(ValueError):
                check_step("guarded", 0.0, 3, r | {key: value}, r["baseline"])

    def test_fixed_step_caps_and_period_two(self):
        self.assertEqual(proposal(3.99, -5), 4.0)
        self.assertTrue(cycle([0.0, 0.08, 0.0, 0.08, 0.0]))
        for values in ([0.0, 0.08], [0.0, 0.0, 0.0, 0.0], [0.0, 0.08, 0.0, 0.04]):
            self.assertFalse(cycle(values))
