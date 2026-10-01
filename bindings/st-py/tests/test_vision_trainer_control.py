"""Public Rust control, actual GPU updates and controlled process continuation."""
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
    "vision_trainer_fixture", Path(__file__).with_name("test_vision_trainer_clients.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def controls():
    """A prescribed control replay, not an adaptive policy or quality claim."""
    initial = st.zspace_meta_optimizer_init({"dimension": 2, "topos_control_gain": 1.})
    config, state = initial["config"], initial["state"]
    result = {}
    for attempt, scale in ((0, 0.5), (9, 1.25), (36, 0.75), (37, 1.), (64, 0.5)):
        report = st.zspace_meta_optimizer_step(config=config, state=state, observation={
            "gradient": [0.1, -0.2],
            "telemetry": {"topos.training_hints.learning_rate_scale": scale},
        })
        state = report["state_after"]
        result[str(attempt)] = report
    return result


def controlled_steps(trainer, count, reports):
    records = []
    for _ in range(count):
        state = json.loads(trainer.state_json())
        attempt = state["accepted_updates"] + state["rejected_updates"]
        report = reports.get(str(attempt))
        if report is not None:
            encoded = json.dumps(report)
            trainer.apply_zspace_meta_optimizer_report_json(encoded)
            saved = trainer.state_json()
            duplicate = json.loads(trainer.apply_zspace_meta_optimizer_report_json(encoded))
            assert duplicate["changed"] is False
            assert trainer.state_json() == saved
        before = json.loads(trainer.state_json())["parameter_control"]
        records.extend(base.steps(trainer, 1))
        assert json.loads(trainer.state_json())["parameter_control"] == before
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
            device, dataset, base.dataset_id(), base.configuration(scheduled), pipeline)
        reports = controls()
    initial = base.checkpoint(trainer)
    records = controlled_steps(trainer, {"control": 100, "prefix": 37, "resume": 63}[phase], reports)
    Path(output).write_text(json.dumps({"phase": phase, "scheduled": scheduled,
        "adapter": device.adapter_info(), "pid": os.getpid(), "control_reports": reports,
        "initial": initial, "records": records, "checkpoint": base.checkpoint(trainer)}), encoding="utf-8")


class Surface(unittest.TestCase):
    def test_report_entry_is_native(self):
        self.assertTrue(callable(st.ResidentVisionTrainer.apply_zspace_meta_optimizer_report_json))
        for attempt, report in controls().items():
            self.assertTrue(report["transition_validated"])
            self.assertGreaterEqual(int(attempt), 0)
            self.assertGreater(st.zspace_parameter_control(report)["source_step"], 0)


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SPIRALTORCH_VISION_CONTROL_BROWSER_DIR"), "controlled browser receipts")
    def test_browser_controlled_checkpoints_continue_in_native(self):
        directory = Path(os.environ["SPIRALTORCH_VISION_CONTROL_BROWSER_DIR"])
        device = st.WgpuTensorDevice.create()
        results = []
        for schedule in ("constant", "cosine"):
            prefix = json.loads((directory / f"{schedule}-prefix.json").read_text())["result"]
            control = json.loads((directory / f"{schedule}-control.json").read_text())["result"]
            self.assertTrue(prefix["passed"] and control["passed"] and control["controlled"])
            dataset, pipeline = base.inputs()
            trainer = st.ResidentVisionTrainer.from_checkpoint_json(
                device, dataset, base.dataset_id(), prefix["checkpoint"], pipeline)
            self.assertEqual(base.checkpoint(trainer), prefix["checkpoint"])
            self.assertEqual(controlled_steps(trainer, 63, controls()), control["records"][37:])
            actual, expected = json.loads(base.checkpoint(trainer)), json.loads(control["checkpoint"])
            self.assertEqual(actual["input"], expected["input"])
            self.assertEqual(actual["trainer"], expected["trainer"])
            a = actual["model"]["backbone"]["parameters"] + actual["model"]["head"]
            b = expected["model"]["backbone"]["parameters"] + expected["model"]["head"]
            self.assertEqual(len(a), len(b))
            errors = []
            for left, right in zip(a, b):
                self.assertEqual((left["name"], left["shape"]), (right["name"], right["shape"]))
                self.assertEqual(len(left["values"]), len(right["values"]))
                for value, reference in zip(left["values"], right["values"]):
                    error = abs(value - reference) / (1 + abs(reference))
                    self.assertLessEqual(error, 2e-4)
                    errors.append(error)
            results.append(dict(schedule=schedule, all_input_rate_control_records_exact=True,
                                parameter_tensors=len(a), parameter_values=len(errors), max_scaled_error=max(errors)))
        output = os.environ.get("SPIRALTORCH_VISION_CONTROL_REVERSE_REPORT")
        if output:
            with Path(output).open("x", encoding="utf-8") as handle:
                json.dump(results, handle)

    def test_controlled_fresh_process_resume(self):
        fixtures = []
        with tempfile.TemporaryDirectory(prefix="st-vision-control-") as directory:
            for scheduled in (False, True):
                results = {}
                for phase in ("control", "prefix", "resume"):
                    output = Path(directory, phase + ".json")
                    command = [sys.executable, "-I", __file__, "--child", str(int(scheduled)), phase, str(output)]
                    if phase == "resume":
                        command.append(str(Path(directory, "prefix.json")))
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    results[phase] = json.loads(output.read_text())
                control, prefix, resume = (results[p] for p in ("control", "prefix", "resume"))
                self.assertEqual(len({r["pid"] for r in results.values()}), 3)
                self.assertEqual(control["records"], prefix["records"] + resume["records"])
                self.assertEqual(prefix["checkpoint"], resume["initial"])
                self.assertEqual(control["checkpoint"], resume["checkpoint"])
                self.assertEqual(sum(r["accepted"] for r in control["records"]), 90)
                final = json.loads(control["checkpoint"])
                self.assertEqual(final["schema"], "spiraltorch.vision.training_checkpoint.v2")
                self.assertEqual(final["trainer"]["parameter_control"], {
                    "absolute_learning_rate_scale": 0.5, "source_meta_step": 5})
                fixtures.append({"scheduled": scheduled, "config": json.loads(base.configuration(scheduled)),
                    "initial": control["initial"], "prefix": prefix["checkpoint"],
                    "final": resume["checkpoint"], "records": control["records"],
                    "adapter": control["adapter"], "control_reports": control["control_reports"]})
        output = os.environ.get("SPIRALTORCH_VISION_CONTROL_HANDOFF")
        if output:
            with Path(output).open("x", encoding="utf-8") as handle:
                json.dump({"schema": "vision_trainer_client_fixture.v1", "controlled": True,
                    "dataset_sha256": base.dataset_id(), "samples": base.samples(), "cases": fixtures}, handle)

    def test_pending_stale_and_corrupt_control_leave_owner_unchanged(self):
        device = st.WgpuTensorDevice.create()
        dataset, pipeline = base.inputs()
        trainer = st.ResidentVisionTrainer.create(
            device, dataset, base.dataset_id(), base.configuration(True), pipeline)
        reports = controls()
        apply = trainer.apply_zspace_meta_optimizer_report_json
        apply(json.dumps(reports["9"]))
        current = base.checkpoint(trainer)
        damaged = json.loads(json.dumps(reports["36"]))
        damaged["topos_control"]["learning_rate_scale"] = 0.1
        for encoded in ("{}", "{", json.dumps(reports["0"]), json.dumps(damaged)):
            with self.assertRaises(ValueError):
                apply(encoded)
            self.assertEqual(base.checkpoint(trainer), current)
        trainer.submit_next()
        pending = trainer.state_json()
        with self.assertRaises(ValueError):
            apply(json.dumps(reports["36"]))
        self.assertEqual(trainer.state_json(), pending)
        trainer.settle()
        current = base.checkpoint(trainer)
        bad = json.loads(current)
        bad["schema"] = "spiraltorch.vision.training_checkpoint.v1"
        with self.assertRaises(ValueError):
            trainer.restore_checkpoint_json(json.dumps(bad))
        self.assertEqual(base.checkpoint(trainer), current)

    @unittest.skipUnless(os.environ.get("SPIRALTORCH_VISION_TRAINER_LEGACY_FIXTURE"), "previous build fixture")
    def test_previous_plain_checkpoints_roundtrip_and_continue(self):
        fixture = json.loads(Path(os.environ["SPIRALTORCH_VISION_TRAINER_LEGACY_FIXTURE"]).read_text())
        device = st.WgpuTensorDevice.create()
        for case in fixture["cases"]:
            for key in ("initial", "prefix", "final"):
                payload = case[key]
                dataset, pipeline = base.inputs()
                trainer = st.ResidentVisionTrainer.from_checkpoint_json(
                    device, dataset, base.dataset_id(), payload, pipeline)
                self.assertEqual(base.checkpoint(trainer), payload)
                self.assertNotIn("parameter_control", json.loads(trainer.state_json()))
                if key == "prefix":
                    self.assertEqual(base.steps(trainer, 63), case["records"][37:])
                    self.assertEqual(base.checkpoint(trainer), case["final"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_run(bool(int(sys.argv[2])), sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    else:
        unittest.main()
