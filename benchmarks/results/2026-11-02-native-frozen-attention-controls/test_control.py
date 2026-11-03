import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_contract import (
    CASES,
    PARENT,
    STATES,
    BOOKS,
    check_step,
    contrasts,
    first_match,
    read,
)
from control_score import check_scalar, verify_section
from control_study import run


class FrozenControlTests(unittest.TestCase):
    def test_every_step_preserves_both_frozen_owners_and_first_lora_update(self):
        r = {
            "epoch": 0,
            "update": 1,
            "window_start": 0,
            "window_end": 1,
            "target_tokens": 63,
            "mode": "lora",
            "loss": 4.0,
            "pre_lora_artifact_id": "old",
            "post_lora_artifact_id": "new",
            "pre_geometry_artifact_id": "g",
            "post_geometry_artifact_id": "g",
        }
        self.assertEqual(check_step(r, 1, "old", "g"), "new")
        first_match(r, {**r, "mode": "joint", "post_geometry_artifact_id": "trained"})
        with self.assertRaisesRegex(ValueError, "gradient step"):
            first_match(r, {**r, "post_lora_artifact_id": "another"})
        for field, value in (
            ("post_geometry_artifact_id", None),
            ("pre_geometry_artifact_id", "different"),
            ("mode", "joint"),
            ("window_end", 2),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "lineage"),
            ):
                check_step({**r, field: value}, 1, "old", "g")

    def test_deltas_use_each_recipes_own_frozen_control(self):
        values = {
            name: (
                {
                    ("wave", "joint"): 4.0,
                    ("wave", "fixed"): 4.5,
                    ("affine", "joint"): 3.875,
                    ("affine", "fixed"): 5.0,
                }[tuple(name.split("-")[1].split(":"))]
            )
            for name in STATES
        }
        out = contrasts({book: {"sample": values} for book in BOOKS})["mansfield"]
        self.assertEqual(out["wave_learning_wins"], 3)
        self.assertEqual(
            out["mean_deltas"],
            {
                "joint_minus_fixed_wave": -0.5,
                "joint_minus_fixed_affine": -1.125,
                "fixed_wave_minus_fixed_affine": -0.5,
                "joint_wave_minus_joint_affine": 0.125,
                "interaction": 0.625,
            },
        )

    def test_public_anchor_and_returned_owners_are_enforced(self):
        parent = read(PARENT / "results.json")
        group = read(PARENT / "score_roster.json")["groups"]["mansfield-5"]
        fits = {
            case: {"lora_id": case, "geometry_id": "fixed-" + case} for case in CASES
        }
        rows = []
        for state in STATES:
            case, mode = state.split(":")
            owner = (fits if mode == "fixed" else parent["fits"])[case]
            nll = parent["section_nll"]["mansfield"]["5"][case]
            rows.append(
                {
                    **group["common"],
                    "lora_artifact_id": owner["lora_id"],
                    "geometry_artifact_id": owner["geometry_id"],
                    "nll": nll,
                    "perplexity": math.exp(nll),
                }
            )
        report = {
            "schema": "spiraltorch.native_causal_lm_evaluation_comparison.v1",
            "shared_prefix_layers": 8,
            "evaluations": rows,
        }
        self.assertEqual(
            len(verify_section("mansfield-5", report, fits, parent, group)), 12
        )
        scalar = {
            name: r
            for name, r in zip(STATES, rows, strict=True)
            if name.endswith(":fixed")
        }
        check_scalar(report, scalar)
        with self.assertRaises(ValueError):
            check_scalar(report, dict(list(scalar.items())[1:]))
        for field, val, which, err in (
            ("geometry_artifact_id", "missing", 1, "owners"),
            ("target_tokens", 755, 2, "coverage"),
            ("nll", 4.25, 0, "anchor"),
        ):
            changed = copy.deepcopy(report)
            changed["evaluations"][which][field] = val
            if field == "nll":
                changed["evaluations"][which]["perplexity"] = math.exp(val)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, err):
                verify_section("mansfield-5", changed, fits, parent, group)

    def test_existing_attempt_cannot_be_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = SimpleNamespace(
                python=Path(sys.executable), run=Path(tmp) / "registry" / "fixed"
            )
            a.run.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run(a, {})
            self.assertEqual(list(a.run.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
