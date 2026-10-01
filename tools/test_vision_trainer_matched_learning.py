"""Offline negative checks for the real-image trainer replay; stdlib only."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
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
        parameters = [dict(name="w", shape=[1, 1], values=[0.125])]
        for pid, phase in enumerate(("control", "prefix", "resume"), 1):
            checkpoints = {}
            for key, value in states[phase].items():
                target = self.directory / f"{phase}-{key}.json"
                runner.write_json(target, dict(state=value, model=dict(classes=2,
                    backbone=dict(config={}, parameters=parameters), head=[])))
                checkpoints[key] = runner.receipt(target)
            subset = records if phase == "control" else records[:1] if phase == "prefix" else records[1:]
            self.phases[phase] = dict(phase=phase, pid=pid, status="passed", contract=dict(dataset="frozen"),
                                      records=copy.deepcopy(subset), checkpoints=checkpoints)
        reference_path = self.directory / "control-torch-final.json"
        runner.write_json(reference_path, dict(schema="spiraltorch.vision.torch_reference_weights.v1",
                                               classes=2, config={}, parameters=parameters))
        self.phases["control"]["reference_checkpoint"] = runner.receipt(reference_path)

    def verify(self):
        return runner.verify_replay(self.directory, self.phases, 4, 1, [0, 1, 2, 3], 2)

    def test_exact_trajectory_and_all_checkpoint_boundaries(self):
        result = self.verify()
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["split"], [1, 3])
        self.assertTrue(result["all_bound_checkpoints_exact"])
        self.assertTrue(result["all_batch_records_exact"])
        self.assertEqual(result["saved_reference_parameters"], dict(tensors=1, values=1, max_scaled_error=0.))

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

    def test_rehashed_but_wrong_reference_weights_fail(self):
        path = self.directory / "wrong-torch.json"
        runner.write_json(path, dict(schema="spiraltorch.vision.torch_reference_weights.v1", classes=2, config={},
                                    parameters=[dict(name="w", shape=[1, 1], values=[0.5])]))
        self.phases["control"]["reference_checkpoint"] = runner.receipt(path)
        with self.assertRaisesRegex(ValueError, "numerical bound"):
            self.verify()

    def test_missing_reference_parameter_fails(self):
        path = self.directory / "truncated-torch.json"
        runner.write_json(path, dict(schema="spiraltorch.vision.torch_reference_weights.v1", classes=2,
                                    config={}, parameters=[]))
        self.phases["control"]["reference_checkpoint"] = runner.receipt(path)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.verify()

    def test_artifacts_never_overwrite(self):
        path = self.directory / "new.json"
        runner.write_json(path, dict(status="passed"))
        with self.assertRaises(FileExistsError):
            runner.write_json(path, dict(status="replacement"))

    def cli_fixture(self):
        root = self.directory / "cli"
        directory = root / "seed-17"
        directory.mkdir(parents=True)
        for file in self.directory.glob("*.json"):
            shutil.copyfile(file, directory / file.name)
        records = self.phases["control"]["records"]
        for i, row in enumerate(records):
            row.update(sample_ids=list(range(10 * (i % 2), 10 * (i % 2) + 10)), flips=[False, True] * 5)
        self.phases["prefix"]["records"] = copy.deepcopy(records[:1])
        self.phases["resume"]["records"] = copy.deepcopy(records[1:])
        contract = dict(source_sha256=runner.source_hashes(),
            data=dict(train_indices=list(range(20)), test_indices=list(range(10)),
                      train_per_class=2, test_per_class=1, augmentation="rust_horizontal_flip_0.5"),
            config=dict(model_seed="17", shuffle_seed="17", batch_size=10,
                        learning_rate=dict(kind="constant", rate=0.01)),
            environment=dict(torch_device="mps"), augmentation_seed="17",
            horizontal_flip=True, schedule="constant")
        evaluation = dict(spiraltorch=dict(accuracy=0.5, loss=1.), torch=dict(accuracy=0.5, loss=1.))
        control = self.phases["control"]
        control.update(admission={}, initial_evaluation=evaluation,
                       epochs=[dict(evaluation=evaluation)] * 2, final_parameter_comparison=[])
        for phase, value in self.phases.items():
            value["contract"] = copy.deepcopy(contract)
            runner.write_json(directory / f"{phase}.json", value)
        run = {key: control[key] for key in ("contract", "admission", "initial_evaluation", "epochs",
                                            "final_parameter_comparison", "reference_checkpoint")}
        run.update(seed=17, checkpoints={phase: value["checkpoints"] for phase, value in self.phases.items()},
                   phase_receipts={phase: runner.receipt(directory / f"{phase}.json") for phase in self.phases},
                   replay=runner.verify_replay(directory, self.phases, 4, 1, list(range(20)), 10))
        summary = dict(schema="spiraltorch.vision.trainer_matched_learning.v1", status="passed", runs=[run],
                       recipe=dict(seeds=[17], epochs=2, train_per_class=2, test_per_class=1,
                                   batch_size=10, torch_device="mps",
                                   horizontal_flip=True, schedule="constant", rate=0.01, restart_at=1))
        runner.write_json(root / "summary.json", summary)
        return root, summary

    def call_cli(self, root):
        return subprocess.run([sys.executable, "-I", str(path.with_name("verify_vision_trainer_replay.py")),
                               str(root), "--output", str(root / "verification.json")],
                              capture_output=True, text=True, timeout=30)

    def test_standalone_verifier_recomputes_saved_result(self):
        root, _ = self.cli_fixture()
        result = self.call_cli(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "passed")

    def test_standalone_verifier_rejects_edited_public_metric(self):
        root, summary = self.cli_fixture()
        summary["runs"][0]["epochs"][0]["evaluation"]["torch"]["accuracy"] = 0.9
        (root / "summary.json").write_text(json.dumps(summary))
        result = self.call_cli(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from raw control", result.stderr)
        self.assertFalse((root / "verification.json").exists())

    def test_standalone_verifier_rejects_noop_augmentation(self):
        root, summary = self.cli_fixture()
        directory = root / "seed-17"
        for phase, value in self.phases.items():
            for row in value["records"]:
                row["flips"] = [False] * 10
            file = directory / f"{phase}.json"
            file.write_text(json.dumps(value))
            summary["runs"][0]["phase_receipts"][phase] = runner.receipt(file)
        summary["runs"][0]["replay"] = runner.verify_replay(directory, self.phases, 4, 1, list(range(20)), 10)
        (root / "summary.json").write_text(json.dumps(summary))
        result = self.call_cli(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("augmentation was absent", result.stderr)
        self.assertFalse((root / "verification.json").exists())

    def test_standalone_verifier_rejects_edited_recipe(self):
        root, original = self.cli_fixture()
        for key, value in (("torch_device", "cpu"), ("test_per_class", 2), ("train_per_class", 4),
                           ("batch_size", 5), ("horizontal_flip", False), ("schedule", "cosine"),
                           ("rate", 0.02), ("epochs", 3), ("restart_at", 2),
                           ("control_scale", 0.5), ("optimizer_feedback", True)):
            with self.subTest(key=key):
                summary = copy.deepcopy(original)
                summary["recipe"][key] = value
                (root / "summary.json").write_text(json.dumps(summary))
                result = self.call_cli(root)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((root / "verification.json").exists())

    def test_standalone_verifier_rejects_relabelled_seed(self):
        root, summary = self.cli_fixture()
        summary["recipe"]["seeds"] = [29]
        summary["runs"][0]["seed"] = 29
        (root / "seed-17").rename(root / "seed-29")
        (root / "summary.json").write_text(json.dumps(summary))
        result = self.call_cli(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recipe seed differs", result.stderr)
        self.assertFalse((root / "verification.json").exists())


if __name__ == "__main__":
    unittest.main()
