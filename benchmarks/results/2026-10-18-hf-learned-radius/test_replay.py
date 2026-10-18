"""Model-free declaration and exact-treatment checks."""
import copy
import json
import math
from pathlib import Path
import runpy
import unittest

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "replay.py"))
PLAN = json.loads((HERE / "plan.json").read_text())


def record(values=(0., 0., 0., 0.)):
    return {"placements": [{"path": f"layer.{index}", "training": True,
                            "recipe": {"schema": "spiraltorch.wave_gate_adapter.v2", "learnable_radius": False,
                                       "strength": .1, "features": 768, "kernel": {"curvature": -1.}},
                            "tensors": [{"name": "gate", "values": [0.]},
                                        {"name": "log_radius", "shape": [], "values": [value],
                                         "role": "buffer", "requires_grad": False}]}
                           for index, value in enumerate(values)]}


class ReplayTests(unittest.TestCase):
    def test_all_full_and_resume_cases_are_fixed_before_launch(self):
        cases = API["cases_for"](PLAN)
        self.assertEqual(len(cases), 14)
        self.assertEqual([c["seed"] for c in cases[:12]], [41] * 4 + [43] * 4 + [47] * 4)
        self.assertEqual(cases[12:], [{"arm": "learned_r1", "seed": 41, "phase": "resume"},
                                     {"arm": "learned_r4", "seed": 41, "phase": "resume"}])
        self.assertEqual(12 * 512 + 2 * 256, PLAN["total_optimizer_updates_including_resume_replays"])

    def test_changed_selection_or_budget_is_rejected(self):
        for field, value in (("schema", "changed"), ("arms", PLAN["arms"][:2]), ("seeds", [41]),
                             ("primary_contrasts", []), ("resume_cases", []), ("resume_from_step", 128),
                             ("trainable_values", {"fixed": 6148, "learned": 6148})):
            plan = copy.deepcopy(PLAN)
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                API["cases_for"](plan)

    def test_learned_treatment_changes_only_radius_trainability(self):
        original = record()
        before = copy.deepcopy(original)
        learned = API["learned_record"](original)
        self.assertEqual(original, before)
        for placement in learned["placements"]:
            self.assertTrue(placement["recipe"]["learnable_radius"])
            placement["recipe"]["learnable_radius"] = False
            radius = placement["tensors"][1]
            self.assertEqual((radius["role"], radius["requires_grad"]), ("parameter", True))
            radius.update(role="buffer", requires_grad=False)
        self.assertEqual(learned, original)

    def test_wrong_source_is_not_silently_treated(self):
        for change in (lambda r: r["placements"][0]["recipe"].update(learnable_radius=True),
                       lambda r: r["placements"][0]["recipe"].update(schema="other"),
                       lambda r: r["placements"][0]["tensors"][1].update(role="parameter")):
            invalid = record()
            change(invalid)
            with self.assertRaises(ValueError):
                API["learned_record"](invalid)

    def test_aggregate_radius_changes_do_not_publish_individual_weights(self):
        stats = API["radius_metrics"](record(), record((.1, -.2, .3, 0.)))
        self.assertEqual(stats["changed_sites"], 3)
        self.assertEqual(stats["increased_sites"], 2)
        self.assertAlmostEqual(stats["log_radius_delta_l2"], math.sqrt(.14))
        self.assertAlmostEqual(stats["radius_multiplier_geometric_mean"], math.exp(.05))
        self.assertNotIn("placements", stats)
        self.assertNotIn("values", stats)
        self.assertEqual(API["radius_metrics"](record(), record())["changed_sites"], 0)
        with self.assertRaises(ValueError):
            API["radius_metrics"](record(), record((float("nan"), 0., 0., 0.)))


if __name__ == "__main__":
    unittest.main()
