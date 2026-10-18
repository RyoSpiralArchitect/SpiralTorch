"""Model-free tests for saved calibration verification, not quality evidence."""

import copy
import hashlib
import json
from pathlib import Path
import runpy
import unittest

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / "verify_saved_state.py"))
POLICY = API["POLICY"]
PLAN = json.loads((HERE / "plan.json").read_text())


def layers(value):
    return [{"path": path, "relative_residual_rms": value} for path in POLICY["SITES"]]


def fixture():
    case = {"arm": "fixed_r1", "seed": 41, "mode": "calibrated_ordinary"}
    original = {"placements": [{"path": path, "recipe": {"strength": .1, "features": 768},
                                "tensors": [{"name": "gate", "values": [1., 2.]}]}
                               for path in POLICY["SITES"]]}
    candidate = copy.deepcopy(original)
    for placement in candidate["placements"]:
        placement["recipe"]["strength"] = POLICY["fitted_strength"](.1, .01, .1)
    target = {"candidate_artifact_id": "target", "calibration": {"layers": layers(.01)},
              "development": {"layers": layers(.01)}}
    record = dict(case, all_learned_tensors_unchanged=True, restored_first_window_logits_exact=True,
                  only_recipe_strengths_may_change=True, optimizer_updates=0, development_loss=4.,
                  source_artifact_id="source", candidate_artifact_id="candidate", target_artifact_id="target",
                  reference_loss_exact=None, calibration_stages=[
                      {"stage": f"calibration-before-layer-{i}", "layers": layers(.1)} for i in range(4)],
                  calibration={"stage": "calibration-final", "layers": layers(.01)},
                  development={"stage": "development", "layers": layers(.012)},
                  calibration_mismatch=POLICY["mismatch"](layers(.01), layers(.01)),
                  development_mismatch=POLICY["mismatch"](layers(.012), layers(.01)))
    return case, record, original, candidate, target, PLAN


class SavedStateTests(unittest.TestCase):
    def test_exact_rule_passes_without_changing_inputs_or_rejecting_dev_mismatch(self):
        args = fixture()
        before = copy.deepcopy(args)
        API["verify_case"](*args)
        self.assertEqual(args, before)

    def test_changed_weights_strength_identity_or_missing_stage_rejected(self):
        for change in ("weights", "strength", "target", "stage", "base_recipe"):
            args = fixture()
            record, candidate = args[1], args[3]
            if change == "weights":
                candidate["placements"][0]["tensors"][0]["values"][0] += .01
            elif change == "strength":
                candidate["placements"][0]["recipe"]["strength"] += .0001
            elif change == "target":
                record["target_artifact_id"] = "different"
            elif change == "stage":
                record["calibration_stages"].reverse()
            else:
                candidate["placements"][0]["recipe"]["features"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                API["verify_case"](*args)

    def test_failed_runtime_guards_nonfinite_ce_and_calibration_rejected(self):
        for key, value in (("optimizer_updates", False), ("optimizer_updates", 1),
                           ("restored_first_window_logits_exact", False),
                           ("all_learned_tensors_unchanged", False), ("development_loss", float("nan"))):
            args = fixture()
            args[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                API["verify_case"](*args)
        args = fixture()
        args[1]["calibration"]["layers"] = layers(.011)
        args[1]["calibration_mismatch"] = POLICY["mismatch"](layers(.011), layers(.01))
        with self.assertRaises(ValueError):
            API["verify_case"](*args)

    def test_reference_requires_unchanged_strength_and_identity(self):
        _, record, original, _, _, plan = fixture()
        case = {"arm": "fixed_r1", "seed": 41, "mode": "reference"}
        record.update(case, only_recipe_strengths_may_change=False, reference_loss_exact=True,
                      target_artifact_id=None, candidate_artifact_id="source", calibration_stages=[],
                      calibration_mismatch=None, development_mismatch=None)
        candidate = copy.deepcopy(original)
        API["verify_case"](case, record, original, candidate, None, plan)
        candidate["placements"][0]["recipe"]["strength"] = .09
        with self.assertRaises(ValueError):
            API["verify_case"](case, record, original, candidate, None, plan)

    def test_stage_identity_tracks_only_preceding_layer_changes(self):
        def identify(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        _, record, original, candidate, _, _ = fixture()
        current = copy.deepcopy(original)
        record["source_artifact_id"] = identify(current)
        for index, stage in enumerate(record["calibration_stages"]):
            stage["geometry_artifact_id"] = identify(current)
            current["placements"][index]["recipe"]["strength"] = candidate["placements"][index]["recipe"]["strength"]
        record["candidate_artifact_id"] = identify(candidate)
        for label in ("calibration", "development"):
            record[label]["geometry_artifact_id"] = identify(candidate)
        API["verify_stage_ids"](record, original, candidate, identify)
        for defect in ("missing", "swapped", "incorrect", "final", "source", "candidate"):
            changed = copy.deepcopy(record)
            if defect == "missing":
                del changed["calibration_stages"][0]["geometry_artifact_id"]
            elif defect == "swapped":
                changed["calibration_stages"][0]["geometry_artifact_id"] = record["calibration_stages"][1]["geometry_artifact_id"]
            elif defect == "incorrect":
                changed["calibration_stages"][2]["geometry_artifact_id"] = "wrong"
            elif defect == "final":
                changed["development"]["geometry_artifact_id"] = record["source_artifact_id"]
            else:
                changed[f"{defect}_artifact_id"] = "wrong"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                API["verify_stage_ids"](changed, original, candidate, identify)
        reference = dict(record, calibration_stages=[], candidate_artifact_id=record["source_artifact_id"])
        reference["calibration"] = {"geometry_artifact_id": record["source_artifact_id"]}
        reference["development"] = {"geometry_artifact_id": record["source_artifact_id"]}
        API["verify_stage_ids"](reference, original, original, identify)
        del reference["calibration"]["geometry_artifact_id"]
        with self.assertRaises(ValueError):
            API["verify_stage_ids"](reference, original, original, identify)


if __name__ == "__main__":
    unittest.main()
