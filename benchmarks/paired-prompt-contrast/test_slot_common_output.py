import copy
import math
import unittest

from slot_common_output import H, paired, plane, task
from slot_common_readback import fold_metrics, test_gate as gate


class SlotCommonTests(unittest.TestCase):
    def test_no_held_out_edges_and_equal_group_sizes(self):
        train = task.deck("train")
        for parity in (0, 1):
            rr = [r for r in train if r["edge"] % 2 == parity]
            pp = paired(rr)
            self.assertEqual(len(pp), 16)
            self.assertEqual([sum(s == k for a, b, s, j in pp) for k in (0, 1)], [8, 8])
            self.assertEqual([sum(j == k for a, b, s, j in pp) for k in range(8)], [2] * 8)
            with self.assertRaises((ValueError, StopIteration)):
                paired(rr[:-1])

    def test_plane_control_preserves_input_major_orientation_and_rejects_opposites(self):
        v = [1.0, 0.0] + [0.0] * 30
        a = [1.0, 2.0] + [0.0] * 30
        b = [1.0, -2.0] + [0.0] * 30
        self.assertEqual(plane([a, b], [0, 1], v), v)
        self.assertIsNone(plane([a, [-x for x in a]], [0, 1], v))
        r = plane([a, a], [0, 1], v)
        self.assertAlmostEqual(r[0], 1 / math.sqrt(5))
        self.assertAlmostEqual(r[1], 2 / math.sqrt(5))

    def test_signed_response_is_not_a_token_flip(self):
        rows = [r for r in task.deck("train") if r["edge"] % 2 == 1]
        values = {}
        for r in rows:
            v = [-4.0] * 64
            slot = r["query"] ^ r["order"]
            v[2] = 2.0
            v[16] = (0.1 if slot == 0 else -0.1) * (1 if r["gold"] == 16 else -1)
            values[r["id"]] = v
        metrics = fold_metrics(rows, values)
        self.assertAlmostEqual(metrics["slot_means"][0], 0.2)
        self.assertAlmostEqual(metrics["slot_means"][1], -0.2)
        self.assertEqual((metrics["ranked_by_slot"], metrics["two_choice"]), ([8, 0], 16))

    def test_one_slot_or_one_edge_cannot_carry_the_gate(self):
        cell = dict(
            agrees=True,
            positive_edges=[3, 4],
            central=[0.002, 0.002],
            predicted=[0.002, 0.002],
            plus=[0.002, 0.002],
        )
        table = {str(h): {k: copy.deepcopy(cell) for k in ("fit", "check")} for h in H}
        self.assertTrue(gate(table, True))
        for k, val in (("positive_edges", [4, 2]), ("plus", [1.0, -1.0]), ("predicted", [1.0, 0.0])):
            t = copy.deepcopy(table)
            t[str(H[-1])]["check"][k] = val
            self.assertFalse(gate(t, True))
        table[str(H[0])]["fit"]["agrees"] = False
        self.assertFalse(gate(table, True))
