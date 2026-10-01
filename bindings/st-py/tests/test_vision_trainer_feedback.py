"""Rust-owned loss feedback and fresh-process continuation via the public client."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import spiraltorch as st

spec = importlib.util.spec_from_file_location(
    "vision_control_fixture", Path(__file__).with_name("test_vision_trainer_control.py"))
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
base = control.base


def configuration(scheduled):
    config = json.loads(base.configuration(scheduled))
    config["optimizer_feedback"] = {
        "warmup_observations": 1, "recovery_rate": 0.5,
        "relative_delta_ema_alpha": 1., "recovery_threshold": 0.,
    }
    return json.dumps(config)


def feedback_steps(trainer, count, reports):
    records = []
    for _ in range(count):
        record = control.controlled_steps(trainer, 1, reports)[0]
        state = json.loads(trainer.state_json())
        record["optimizer_feedback"] = state["optimizer_feedback"]
        feedback = record["optimizer_feedback"]["state"]
        assert feedback["control_step"] == state["accepted_updates"] + state["rejected_updates"]
        assert feedback["observation_count"] == state["accepted_updates"]
        records.append(record)
    return records


def child_run(scheduled, phase, output, prefix=None):
    device = st.WgpuTensorDevice.create()
    assert device.adapter_info()["device_type"] != "Cpu"
    dataset, pipeline = base.inputs()
    if phase == "resume":
        saved = json.loads(Path(prefix).read_text())
        trainer = st.ResidentVisionTrainer.from_checkpoint_json(
            device, dataset, base.dataset_id(), saved["checkpoint"], pipeline)
        reports = saved["control_reports"]
    else:
        trainer = st.ResidentVisionTrainer.create(
            device, dataset, base.dataset_id(), configuration(scheduled), pipeline)
        reports = control.controls()
    initial = base.checkpoint(trainer)
    records = feedback_steps(trainer, {"control": 100, "prefix": 37, "resume": 63}[phase], reports)
    with Path(output).open("x", encoding="utf-8") as handle:
        json.dump({"phase": phase, "scheduled": scheduled, "adapter": device.adapter_info(),
            "pid": os.getpid(), "control_reports": reports, "initial": initial,
            "records": records, "checkpoint": base.checkpoint(trainer)}, handle)


class Surface(unittest.TestCase):
    def test_plain_configuration_does_not_enable_feedback(self):
        self.assertNotIn("optimizer_feedback", json.loads(st.ResidentVisionTrainer.default_config_json()))
        self.assertIn("optimizer_feedback", json.loads(configuration(False)))

    @unittest.skipUnless(os.environ.get("SPIRALTORCH_VISION_FEEDBACK_BROWSER_DIR"), "browser observations")
    def test_browser_observations_replay_exactly_through_native_core(self):
        directory = Path(os.environ["SPIRALTORCH_VISION_FEEDBACK_BROWSER_DIR"])
        reports = control.controls()
        for schedule in ("constant", "cosine"):
            run = json.loads((directory / f"{schedule}-control.json").read_text())["result"]
            self.assertTrue(run["passed"] and run["feedback"])
            self.assertEqual(len(run["records"]), 100)
            config = run["state"]["optimizer_feedback"]["config"]
            state = st.zspace_optimizer_feedback_init(config)["state"]
            proposal = 1.
            for index, row in enumerate(run["records"]):
                if str(index) in reports:
                    proposal = st.zspace_parameter_control(reports[str(index)])["absolute_learning_rate_scale"]
                state = st.zspace_optimizer_feedback_control(config=config, state=state,
                    target_step=index + 1, proposed_learning_rate_scale=proposal)["state_after"]
                expected = row["optimizer_feedback"]["state"]
                if row["accepted"]:
                    state = st.zspace_optimizer_feedback_observe(config=config, state=state,
                        observation={"step": index + 1, "epoch": row["epoch"],
                                     "loss": expected["last_loss"]})["state_after"]
                self.assertEqual(state, expected, (schedule, index + 1))


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_feedback_history_and_updates_resume_in_fresh_processes(self):
        fixtures = []
        with tempfile.TemporaryDirectory(prefix="st-vision-feedback-") as directory:
            for scheduled in (False, True):
                results = {}
                for phase in ("control", "prefix", "resume"):
                    output = Path(directory, f"{int(scheduled)}-{phase}.json")
                    command = [sys.executable, "-I", __file__, "--child", str(int(scheduled)), phase, str(output)]
                    if phase == "resume":
                        command.append(str(Path(directory, f"{int(scheduled)}-prefix.json")))
                    run = subprocess.run(command, capture_output=True, text=True, timeout=180)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    results[phase] = json.loads(output.read_text())
                uninterrupted, prefix, resume = (results[p] for p in ("control", "prefix", "resume"))
                self.assertEqual(len({r["pid"] for r in results.values()}), 3)
                self.assertEqual(uninterrupted["records"], prefix["records"] + resume["records"])
                self.assertEqual(prefix["checkpoint"], resume["initial"])
                self.assertEqual(uninterrupted["checkpoint"], resume["checkpoint"])
                self.assertEqual(json.loads(resume["checkpoint"])["schema"], "spiraltorch.vision.training_checkpoint.v3")
                self.assertEqual(sum(r["accepted"] for r in uninterrupted["records"]), 90)
                states = [r["optimizer_feedback"]["state"] for r in uninterrupted["records"]]
                self.assertTrue(any(s["gate"] > 0. for s in states))
                self.assertEqual(states[-1]["observation_count"], 90)
                self.assertEqual(states[-1]["control_step"], 100)
                for before, after, record in zip(states, states[1:], uninterrupted["records"][1:]):
                    if not record["accepted"]:
                        self.assertEqual(before["last_loss"], after["last_loss"])
                        self.assertEqual(before["observation_count"], after["observation_count"])
                fixtures.append({"scheduled": scheduled, "config": json.loads(configuration(scheduled)),
                    "initial": uninterrupted["initial"], "prefix": prefix["checkpoint"],
                    "final": resume["checkpoint"], "records": uninterrupted["records"],
                    "adapter": uninterrupted["adapter"], "control_reports": uninterrupted["control_reports"]})
        output = os.environ.get("SPIRALTORCH_VISION_FEEDBACK_HANDOFF")
        if output:
            with Path(output).open("x", encoding="utf-8") as handle:
                json.dump({"schema": "vision_trainer_client_fixture.v1", "controlled": True,
                    "feedback": True, "dataset_sha256": base.dataset_id(),
                    "samples": base.samples(), "cases": fixtures}, handle)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_run(bool(int(sys.argv[2])), sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    else:
        unittest.main()
