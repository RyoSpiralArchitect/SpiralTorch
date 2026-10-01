"""Offline negative checks for the real-image trainer replay; stdlib only."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).with_name("run_vision_trainer_matched_learning.py")
spec = importlib.util.spec_from_file_location("trainer_matched", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        records = [dict(revision=i + 1, epoch=i // 2, accepted=True, sample_ids=[2 * (i % 2), 2 * (i % 2) + 1],
                        flips=[False, True], input_sha256=runner.sha(bytes([i])),
                        rate_bits=runner.bits(0.01), loss_bits=runner.bits(2. / (i + 1))) for i in range(4)]
        self.phases = {}
        states = dict(control=dict(initial="start", split="middle", final="end"),
                      prefix=dict(initial="start", final="middle"), resume=dict(initial="middle", final="end"))
        for pid, phase in enumerate(("control", "prefix", "resume"), 1):
            checkpoints = {}
            for key, value in states[phase].items():
                target = self.directory / f"{phase}-{key}.json"
                runner.write_json(target, dict(state=value))
                checkpoints[key] = runner.receipt(target)
            subset = records if phase == "control" else records[:1] if phase == "prefix" else records[1:]
            self.phases[phase] = dict(phase=phase, pid=pid, status="passed", contract=dict(dataset="frozen"),
                                      records=copy.deepcopy(subset), checkpoints=checkpoints)

    def verify(self):
        return runner.verify_replay(self.directory, self.phases, 4, 1, [0, 1, 2, 3], 2)

    def test_exact_trajectory_and_all_checkpoint_boundaries(self):
        result = self.verify()
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["split"], [1, 3])
        self.assertTrue(result["all_bound_checkpoints_exact"])
        self.assertTrue(result["all_batch_records_exact"])

    def test_missing_phase(self):
        del self.phases["resume"]
        with self.assertRaisesRegex(ValueError, "phase"):
            self.verify()

    def test_partial_trajectory(self):
        self.phases["resume"]["records"].pop()
        with self.assertRaisesRegex(ValueError, "updates"):
            self.verify()

    def test_changed_rate_is_not_hidden_by_matching_final_checkpoint(self):
        self.phases["resume"]["records"][0]["rate_bits"] += 1
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.verify()

    def test_changed_pixels_are_not_hidden_by_matching_labels(self):
        self.phases["resume"]["records"][0]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.verify()

    def test_changed_augmentation_choice(self):
        self.phases["resume"]["records"][0]["flips"] = [True, False]
        with self.assertRaisesRegex(ValueError, "trajectory"):
            self.verify()

    def test_rejected_update_and_nonfinite_values_fail(self):
        original = copy.deepcopy(self.phases)
        for key, value in (("accepted", False), ("accepted", 1), ("loss_bits", runner.bits(float("nan"))),
                           ("rate_bits", runner.bits(float("inf"))), ("rate_bits", runner.bits(0.))):
            with self.subTest(key=key, value=value):
                self.phases = copy.deepcopy(original)
                self.phases["resume"]["records"][0][key] = value
                with self.assertRaises(ValueError):
                    self.verify()

    def test_wrong_clock_or_sample_fails(self):
        original = copy.deepcopy(self.phases)
        for key, value in (("revision", 9), ("epoch", 9), ("sample_ids", [0, 0]), ("sample_ids", [2, 99])):
            with self.subTest(key=key):
                self.phases = copy.deepcopy(original)
                self.phases["resume"]["records"][0][key] = value
                with self.assertRaises(ValueError):
                    self.verify()

    def test_same_wrong_coverage_in_all_arms_still_fails(self):
        self.phases["control"]["records"][1]["sample_ids"] = [0, 1]
        self.phases["resume"]["records"][0]["sample_ids"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "each sample once"):
            self.verify()

    def test_changed_environment_or_same_process_fails(self):
        self.phases["resume"]["contract"]["dataset"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs"):
            self.verify()
        self.phases["resume"]["contract"]["dataset"] = "frozen"
        self.phases["resume"]["pid"] = self.phases["prefix"]["pid"]
        with self.assertRaisesRegex(ValueError, "distinct processes"):
            self.verify()

    def test_corrupted_checkpoint_bytes_fail(self):
        path = self.directory / self.phases["resume"]["checkpoints"]["final"]["file"]
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "fixity"):
            self.verify()

    def test_consistently_rehashed_wrong_final_checkpoint_fails(self):
        path = self.directory / self.phases["resume"]["checkpoints"]["final"]["file"]
        path.write_text("changed")
        self.phases["resume"]["checkpoints"]["final"] = runner.receipt(path)
        with self.assertRaisesRegex(ValueError, "final state"):
            self.verify()

    def test_checkpoint_path_escape_fails(self):
        self.phases["resume"]["checkpoints"]["final"]["file"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "filename"):
            self.verify()

    def test_artifacts_never_overwrite(self):
        path = self.directory / "new.json"
        runner.write_json(path, dict(status="passed"))
        with self.assertRaises(FileExistsError):
            runner.write_json(path, dict(status="replacement"))


if __name__ == "__main__":
    unittest.main()
