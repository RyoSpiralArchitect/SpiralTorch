"""Offline control-coverage and matched-ablation negative checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

spec = importlib.util.spec_from_file_location("ablation", Path(__file__).with_name("run_vision_feedback_ablation.py"))
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)
runner = ablation.runner


class ControlChecks(unittest.TestCase):
    def fixture(self, feedback=True):
        config = dict(learning_rate=dict(kind="constant", rate=0.01))
        if feedback:
            config["optimizer_feedback"] = dict(maximum_gate=1.)
        contract = dict(config=config, schedule="constant", intervention=dict(requested_scale=0.5,
            feedback_enabled=feedback, projection=dict(absolute_learning_rate_scale=0.5, source_step=1)))
        row = dict(revision=1, accepted=True, loss_bits=runner.bits(2.), rate_bits=runner.bits(0.005),
                   parameter_control=dict(absolute_learning_rate_scale=0.5, source_meta_step=1))
        if feedback:
            row["optimizer_feedback"] = dict(config=config["optimizer_feedback"],
                state=dict(control_step=1, observation_count=1, last_observation_step=1, last_loss=2., gate=0.5))
        return contract, row

    def test_fixed_and_feedback_control_coverage(self):
        for feedback in (False, True):
            contract, row = self.fixture(feedback)
            runner.verify_intervention(contract, [row])

    def test_missing_or_replaced_controls_fail(self):
        for key in ("parameter_control", "optimizer_feedback"):
            contract, row = self.fixture()
            del row[key]
            with self.assertRaises(ValueError):
                runner.verify_intervention(contract, [row])

    def test_loss_clock_and_config_are_actual_observations(self):
        for key, value in (("control_step", 2), ("observation_count", 0), ("last_loss", 1.),
                           ("last_observation_step", 0), ("gate", float("nan"))):
            contract, row = self.fixture()
            row["optimizer_feedback"]["state"][key] = value
            with self.assertRaises(ValueError):
                runner.verify_intervention(contract, [row])
        contract, row = self.fixture()
        row["optimizer_feedback"]["config"] = {}
        with self.assertRaisesRegex(ValueError, "config drift"):
            runner.verify_intervention(contract, [row])

    def test_wrong_rate_or_clamped_proposal_fails(self):
        contract, row = self.fixture()
        row["rate_bits"] = runner.bits(0.02)
        with self.assertRaisesRegex(ValueError, "actual rate"):
            runner.verify_intervention(contract, [row])
        contract["intervention"]["requested_scale"] = 0.1
        with self.assertRaisesRegex(ValueError, "clamped"):
            runner.verify_intervention(contract, [row])

    def test_plain_arm_cannot_hide_feedback(self):
        _, row = self.fixture()
        with self.assertRaisesRegex(ValueError, "unconfigured"):
            runner.verify_intervention({}, [row])

    def test_dose_uses_actual_rates_and_requires_finite_accepted_steps(self):
        rows = [dict(accepted=True, rate_bits=runner.bits(x)) for x in (0.01, 0.005, 0.0075)]
        self.assertEqual(ablation.dose_rate(rows), runner.unbits(runner.bits(0.0075)))
        for key, value in (("accepted", False), ("rate_bits", runner.bits(float("nan"))),
                           ("rate_bits", runner.bits(0.))):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                ablation.dose_rate(changed)

    def test_saved_verification_rejects_changed_orchestrator_before_loading_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.write_json(root / "summary.json", dict(schema="spiraltorch.vision.feedback_ablation.v1",
                status="passed", orchestrator_sha256="0" * 64, boundary=ablation.BOUNDARY))
            with self.assertRaisesRegex(ValueError, "orchestrator source"):
                ablation.verify_saved(types.SimpleNamespace(verify=root, source_ref=None, output=root / "verify.json"))
            self.assertFalse((root / "verify.json").exists())


class MatchedArms(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe = dict(torch_device="mps", train_per_class=2, test_per_class=1,
                           batch_size=10, epochs=2, restart_at=1, rate=0.01, control_scale=0.5)
        evaluation = dict(spiraltorch=dict(loss=1., accuracy=0.2), torch=dict(loss=1., accuracy=0.2))
        for arm in ablation.ARMS:
            root = self.root / arm
            raw = root / "seed-17"
            raw.mkdir(parents=True)
            values = [0.01, 0.005] if arm == "loss_feedback" else (
                [0.005] * 2 if arm == "fixed_proposal" else [0.0075] * 2 if arm == "dose_matched" else [0.01] * 2)
            records = [dict(revision=i + 1, epoch=i, accepted=True, sample_ids=[1, 2], flips=[False] * 2,
                            input_sha256="0" * 64, rate_bits=runner.bits(rate)) for i, rate in enumerate(values)]
            if arm == "loss_feedback":
                for row in records:
                    row["optimizer_feedback"] = dict(state=dict(gate=0.5, halted=False))
            runner.write_json(raw / "initial.json", dict(model={"weights": [0.]}, input={"cursor": 0}))
            contract = {key: {} for key in ("data", "dataset_sha256", "adapter", "source_sha256",
                                            "native_binary_sha256", "environment")}
            control = dict(contract=contract, records=records, epochs=[dict(evaluation=evaluation)],
                           initial_evaluation=evaluation, checkpoints=dict(initial=runner.receipt(raw / "initial.json")))
            runner.write_json(raw / "control.json", control)
            recipe = dict(self.recipe, seeds=[17], schedule="constant", horizontal_flip=False,
                control_scale=0.5 if arm in ("fixed_proposal", "loss_feedback") else None,
                optimizer_feedback=arm == "loss_feedback")
            if arm == "dose_matched":
                recipe["rate"] = ablation.dose_rate([dict(accepted=True, rate_bits=runner.bits(x)) for x in (0.01, 0.005)])
            run = dict(seed=17, contract=contract, initial_evaluation=evaluation, epochs=control["epochs"],
                       replay=dict(all_batch_records_exact=True, all_bound_checkpoints_exact=True))
            runner.write_json(root / "summary.json", dict(status="passed", recipe=recipe, runs=[run]))

    def compare(self):
        return ablation.compare_seed(self.root, self.recipe, 17)

    def change(self, arm, name, edit):
        path = self.root / arm / name
        value = json.loads(path.read_text())
        edit(value)
        path.write_text(json.dumps(value))

    def test_matched_initial_inputs_and_retrospective_dose(self):
        result = self.compare()
        self.assertTrue(result["identical_initial_model_and_inputs"])
        self.assertEqual(result["feedback"]["changed_rate_updates"], 1)
        self.assertEqual(result["dose_matched_rate"], runner.unbits(runner.bits(0.0075)))

    def test_changed_batches_fail_even_with_equal_scores(self):
        self.change("loss_feedback", "seed-17/control.json", lambda v: v["records"][0].update(sample_ids=[2, 1]))
        with self.assertRaisesRegex(ValueError, "batches"):
            self.compare()

    def test_changed_initial_model_fails_even_when_rehashed(self):
        arm = self.root / "fixed_proposal" / "seed-17"
        (arm / "different.json").write_text(json.dumps(dict(model={"weights": [1.]}, input={"cursor": 0})))
        self.change("fixed_proposal", "seed-17/control.json",
                    lambda v: v["checkpoints"].update(initial=runner.receipt(arm / "different.json")))
        with self.assertRaisesRegex(ValueError, "initial model"):
            self.compare()

    def test_wrong_dose_recipe_fails(self):
        self.change("dose_matched", "summary.json", lambda v: v["recipe"].update(rate=0.006))
        with self.assertRaisesRegex(ValueError, "recipe"):
            self.compare()

    def test_equal_total_but_nonconstant_dose_fails(self):
        def edit(value):
            value["records"][0]["rate_bits"] = runner.bits(0.01)
            value["records"][1]["rate_bits"] = runner.bits(0.005)
        self.change("dose_matched", "seed-17/control.json", edit)
        with self.assertRaisesRegex(ValueError, "not constant"):
            self.compare()

    def test_relabelled_arm_or_augmentation_fails(self):
        self.change("baseline", "summary.json", lambda v: v["recipe"].update(horizontal_flip=True))
        with self.assertRaisesRegex(ValueError, "recipe"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
