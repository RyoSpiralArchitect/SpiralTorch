"""Model-free regressions for the declared comparison orchestration."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

HERE = Path(__file__).parent
API = runpy.run_path(str(HERE / "replay.py"))
PLAN = json.loads((HERE / "plan.json").read_text())


def state(value, radius=0.):
    return {"placements": [{"path": "mlp", "recipe": {"kind": "fixture"}, "tensors": [
        {"name": "gate", "role": "parameter", "requires_grad": True, "shape": [1], "values": [value]},
        {"name": "radius", "role": "buffer", "requires_grad": False, "shape": [], "values": [radius]},
    ]}]}


class ReplayTests(unittest.TestCase):
    def test_all_twelve_cases_declared_in_paired_seed_order(self):
        cases = API["cases_for"](PLAN)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({(c["arm"], c["seed"]) for c in cases}), 12)
        self.assertEqual([c["seed"] for c in cases], [41] * 4 + [43] * 4 + [47] * 4)

    def test_unimplemented_defaults_reject(self):
        for field, value in (("weight_decay", .1), ("generation_max_new_tokens", 25),
                             ("gradient_accumulation_steps", 2), ("seeds", [41])):
            plan = copy.deepcopy(PLAN)
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                API["cases_for"](plan)

    def test_command_uses_public_example_and_only_selected_documents(self):
        selected = {"train": [Path("train/a.txt")], "development": [Path("development/b.txt")],
                    "reserve": [Path("reserve/never.txt")]}
        preflight = {"tokenized_dataset_identity": "data-pin", "artifacts": {"ordinary": {"artifact_id": "pin"}}}
        command = API["command_for"](PLAN, preflight, selected, Path("model"), Path("artifacts"),
                                     Path("output"), {"arm": "ordinary", "seed": 41})
        self.assertIn(str(API["EXAMPLE"]), command)
        self.assertEqual(command[command.index("--max-steps") + 1], "512")
        self.assertEqual(command[command.index("--expected-dataset-id") + 1], "data-pin")
        self.assertIn(str(Path("development/b.txt").resolve()), command)
        self.assertNotIn("reserve/never.txt", command)
        for option in ("--model-dir", "--geometry-artifact", "--output", "--train-text", "--eval-text"):
            self.assertTrue(Path(command[command.index(option) + 1]).is_absolute(), option)

    def test_artifact_metrics_count_parameters_not_fixed_radius(self):
        values = API["state_metrics"](state(0., 1.), state(.25, 1.))
        self.assertEqual(values, [{"path": "mlp", "trainable_values": 1,
                                  "parameter_delta_l2": .25, "parameter_delta_max_abs": .25}])
        for final in (state(.25, 2.), state(0., 1.), state(float("nan"), 1.)):
            with self.assertRaises(ValueError):
                API["state_metrics"](state(0., 1.), final)

    def test_pinned_input_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "record.json"
            raw = b'{"value": 1}'
            path.write_bytes(raw)
            pin = API["identity"](raw)
            self.assertEqual(API["load_pinned"](path, pin), {"value": 1})
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                API["load_pinned"](path, pin)

    def test_window_publication_preserves_counts_not_window_arrays(self):
        summary = API["window_summary"]({"context_length": 128, "target_tokens": 254,
                                          "windows": [{"document": 0}, {"document": 1}]})
        self.assertEqual(summary, {"context_length": 128, "target_tokens": 254, "windows": 2})


if __name__ == "__main__":
    unittest.main()
