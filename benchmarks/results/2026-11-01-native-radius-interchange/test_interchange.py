from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import (
    ORIGINAL,
    PATHS,
    PROMPTS,
    ROSTER,
    divergence,
    effects,
    masked,
    project,
    swap_record,
)


def state(learned, radius):
    def site(path):
        return {
            "path": path,
            "training": False,
            "recipe": {
                "schema": "spiraltorch.wave_gate_adapter.v2",
                "learnable_radius": learned,
                "features": 1,
                "strength": 0.1,
            },
            "tensors": [
                {
                    "name": k,
                    "values": [float(learned)],
                    "role": "parameter",
                    "shape": [1],
                    "requires_grad": True,
                }
                for k in ("bias", "gate")
            ]
            + [
                {
                    "name": "log_radius",
                    "values": [radius],
                    "role": "parameter" if learned else "buffer",
                    "shape": [],
                    "requires_grad": learned,
                }
            ],
        }

    return {
        "schema": "spiraltorch.geometry_adapter_artifact.v1",
        "policy": "full_prefix_f32_no_cache.v1",
        "base_model_identity": ORIGINAL["base_id"],
        "training": False,
        "placements": [site(p) for p in PATHS],
    }


class Interchange(unittest.TestCase):
    def test_graft_is_reciprocal_and_preserves_receiver_not_donor_roles(self):
        fixed, learned = state(False, 1.0), state(True, 2.0)
        original = deepcopy([fixed, learned])
        for r, d in ((fixed, learned), (learned, fixed)):
            sham, n = swap_record(r, r)
            self.assertEqual((sham, n), (r, 0))
            out, n = swap_record(r, d)
            expected = deepcopy(r)
            for p, s in zip(expected["placements"], d["placements"], strict=True):
                p["tensors"][-1]["values"] = list(s["tensors"][-1]["values"])
            self.assertEqual((out, n), (expected, 4))
        self.assertEqual(original, [fixed, learned])

    def test_wrong_recipe_base_order_radius_owner_or_nan_fails(self):
        fixed, learned = state(False, 1.0), state(True, 2.0)
        for change in (
            lambda s: s.update(base_model_identity="wrong"),
            lambda s: s["placements"].reverse(),
            lambda s: s["placements"][1]["recipe"].update(strength=0.2),
            lambda s: s["placements"][1]["tensors"][-1].update(requires_grad=False),
            lambda s: s["placements"][2]["tensors"][-1].update(values=[float("nan")]),
        ):
            with self.subTest(change=change):
                bad = deepcopy(learned)
                change(bad)
                with self.assertRaises(ValueError):
                    swap_record(fixed, bad)

    def test_prompt_mask_and_rank_of_divergence(self):
        self.assertEqual(masked([1, 2], [1, 2, 3, 4]), [-100, -100, 3, 4])
        for whole in ([1, 3, 4], [1, 2], [1, 2] + [3] * 64):
            with self.assertRaises(ValueError):
                masked([1, 2], whole)
        self.assertIsNone(divergence([3, 4], [3, 4]))
        self.assertEqual(divergence([3, 4], [4, 3]), 1)
        self.assertEqual(divergence([3, 4], [3, 4, 5]), 3)

    def test_measured_contrasts_and_complete_only_projection(self):
        out = effects({"FF": 5.0, "FL": 3.0, "LF": 4.0, "LL": 3.0})
        self.assertEqual(
            (
                out["radius_on_fixed_recipient"],
                out["radius_on_learned_recipient"],
                out["interaction"],
            ),
            (-2.0, -1.0, 1.0),
        )
        self.assertEqual(
            out["other_at_fixed_radius"] + out["radius_on_learned_recipient"],
            out["total_LL_minus_FF"],
        )
        raw = {
            k: {
                "geometry_id": None,
                "source_checkpoint_ids": [],
                "changed_radius_sites": 0,
                "records": {
                    p: {
                        "generated": [50256],
                        "finish": "eos",
                        "nll": 1.0,
                        "targets": i + 1,
                        "data_id": p,
                        "raw_sha256": "x",
                        "steps_sha256": "y",
                    }
                    for i, p in enumerate(PROMPTS)
                },
            }
            for k in ROSTER
        }
        report = project(raw)
        self.assertFalse(report["both_radius_deltas_negative_at_every_seed"])
        self.assertEqual(report["generated_tokens"], 128)
        self.assertNotIn("generated", report["models"]["base"]["prompts"][PROMPTS[0]])
        bad = deepcopy(raw)
        bad[ROSTER[1]]["records"][PROMPTS[2]]["data_id"] = "other"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            project(bad)
        del raw[ROSTER[-1]]["records"][PROMPTS[0]]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            project(raw)
        del raw[ROSTER[-1]]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            project(raw)


if __name__ == "__main__":
    unittest.main()
