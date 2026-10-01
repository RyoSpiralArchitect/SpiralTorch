"""Exercise the public Rust-owned trainer, including fresh-process continuation."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import spiraltorch as st


def samples(count=20):
    return [([((i * 13 + j * 7) % 101) / 128 for j in range(48)],
             2 if i == 7 else i % 2, str(i)) for i in range(count)]


def dataset_id():
    digest = hashlib.sha256()
    for pixels, target, label in samples():
        digest.update(struct.pack("<49f", *pixels, target))
        digest.update(label.encode("ascii") + b"\0")
    return digest.hexdigest()


def inputs(count=20):
    dataset = st.TensorVisionDataset("CIFAR10")
    for pixels, target, label in samples(count):
        dataset.push(st.ImageTensor(3, 4, 4, pixels),
                     target=st.Tensor(1, 1, [target]), label=label)
    pipeline = st.TransformPipeline(seed=29)
    pipeline.add_horizontal_flip(0.5)
    pipeline.add_normalize([0.5], [0.25])
    return dataset, pipeline


def configuration(scheduled):
    config = json.loads(st.ResidentVisionTrainer.default_config_json())
    config.update(num_classes=2, batch_size=2, model_seed="43", shuffle_seed="17", shuffle=True)
    config["model"].update(input_channels=3, input_hw=[4, 4], stage_dims=[2, 3],
                           stage_depths=[1, 1], patch_size=[2, 2], epsilon=0.001)
    config["learning_rate"] = (
        {"kind": "warmup_cosine", "state": {
            "base_lr": 0.002, "min_lr": 0.0001, "warmup_steps": 10,
            "total_steps": 100, "step": 0,
        }} if scheduled else {"kind": "constant", "rate": 0.001}
    )
    return json.dumps(config)


def checkpoint(trainer):
    return trainer.checkpoint_snapshot().read_json()


def bits(values):
    return list(struct.unpack("<" + "I" * len(values), struct.pack("<" + "f" * len(values), *values)))


def steps(trainer, count):
    records = []
    for _ in range(count):
        before = json.loads(trainer.state_json())
        submitted = trainer.submit_next()
        assert trainer.has_pending_update
        for action in (trainer.submit_next, trainer.checkpoint_snapshot):
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError("pending owner accepted reuse")
        pixels = submitted.images().snapshot().read_values()
        outcome = trainer.settle()
        assert not trainer.has_pending_update
        assert outcome.attempted_revision == submitted.attempted_revision
        assert outcome.accepted == ("7" not in submitted.labels())
        after = json.loads(trainer.state_json())
        if not outcome.accepted:
            assert before["learning_rate"] == after["learning_rate"]
        records.append({
            "revision": outcome.attempted_revision, "epoch": submitted.epoch,
            "rate_bits": bits([submitted.learning_rate])[0], "labels": submitted.labels(),
            "images_bits": bits(pixels), "accepted": outcome.accepted,
        })
    return records


def child_run(scheduled, phase, output, prefix=None):
    device = st.WgpuTensorDevice.create()
    assert device.adapter_info()["device_type"] != "Cpu"
    dataset, pipeline = inputs()
    if phase == "resume":
        saved = json.loads(Path(prefix).read_text())
        trainer = st.ResidentVisionTrainer.from_checkpoint_json(
            device, dataset, dataset_id(), saved["checkpoint"], pipeline)
    else:
        trainer = st.ResidentVisionTrainer.create(
            device, dataset, dataset_id(), configuration(scheduled), pipeline)
    initial = checkpoint(trainer)
    count = {"control": 100, "prefix": 37, "resume": 63}[phase]
    records = steps(trainer, count)
    result = {"phase": phase, "scheduled": scheduled, "dataset_sha256": dataset_id(),
              "adapter": device.adapter_info(), "initial": initial, "records": records,
              "checkpoint": checkpoint(trainer)}
    Path(output).write_text(json.dumps(result), encoding="utf-8")


class Surface(unittest.TestCase):
    def test_public_native_aliases_and_configuration(self):
        for name in ("ResidentVisionTrainer", "ResidentVisionSubmission",
                     "ResidentVisionStepOutcome", "VisionTrainingCheckpointSnapshot"):
            self.assertIs(getattr(st, name), getattr(st.vision, name))
            self.assertIn(name, st.__all__)
            with self.assertRaises(TypeError):
                getattr(st, name)()
        config = json.loads(configuration(True))
        self.assertEqual(config["model_seed"], "43")
        self.assertEqual(config["shuffle_seed"], "17")


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SPIRALTORCH_VISION_TRAINER_BROWSER_DIR"), "browser receipt opt-in")
    def test_browser_checkpoints_continue_in_native(self):
        directory = Path(os.environ["SPIRALTORCH_VISION_TRAINER_BROWSER_DIR"])
        device = st.WgpuTensorDevice.create()
        reports = []
        for schedule in ("constant", "cosine"):
            prefix = json.loads((directory / f"{schedule}-prefix.json").read_text())["result"]
            control = json.loads((directory / f"{schedule}-control.json").read_text())["result"]
            self.assertTrue(prefix["passed"] and control["passed"])
            self.assertEqual(len(prefix["records"]), 37)
            self.assertEqual(len(control["records"]), 100)
            dataset, pipeline = inputs()
            trainer = st.ResidentVisionTrainer.from_checkpoint_json(
                device, dataset, dataset_id(), prefix["checkpoint"], pipeline)
            self.assertEqual(checkpoint(trainer), prefix["checkpoint"])
            self.assertEqual(steps(trainer, 63), control["records"][37:])
            payload = checkpoint(trainer)
            actual, expected = json.loads(payload), json.loads(control["checkpoint"])
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
            reports.append({"schedule": schedule, "parameter_tensors": len(a),
                            "parameter_values": len(errors), "max_scaled_error": max(errors),
                            "checkpoint_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                            "browser_checkpoint_sha256": hashlib.sha256(control["checkpoint"].encode()).hexdigest()})
        output = os.environ.get("SPIRALTORCH_VISION_TRAINER_REVERSE_REPORT")
        if output:
            Path(output).write_text(json.dumps(reports), encoding="utf-8")

    def test_fresh_process_resume_constant_and_scheduled(self):
        fixtures = []
        with tempfile.TemporaryDirectory(prefix="st-vision-trainer-") as directory:
            for scheduled in (False, True):
                results = {}
                for phase in ("control", "prefix", "resume"):
                    output = Path(directory, phase + ".json")
                    command = [sys.executable, "-I", __file__, "--child", str(int(scheduled)), phase, str(output)]
                    if phase == "resume":
                        command.append(str(Path(directory, "prefix.json")))
                    subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
                    results[phase] = json.loads(output.read_text())
                control, prefix, resume = (results[p] for p in ("control", "prefix", "resume"))
                self.assertEqual(control["records"], prefix["records"] + resume["records"])
                self.assertEqual(prefix["checkpoint"], resume["initial"])
                self.assertEqual(control["checkpoint"], resume["checkpoint"])
                self.assertEqual(sum(r["accepted"] for r in control["records"]), 90)
                fixtures.append({"scheduled": scheduled, "config": json.loads(configuration(scheduled)),
                                 "initial": control["initial"], "prefix": prefix["checkpoint"],
                                 "final": resume["checkpoint"], "records": control["records"]})
        output = os.environ.get("SPIRALTORCH_VISION_TRAINER_HANDOFF")
        if output:
            Path(output).write_text(json.dumps({"schema": "vision_trainer_client_fixture.v1",
                "dataset_sha256": dataset_id(), "samples": samples(), "cases": fixtures}), encoding="utf-8")

    def test_templates_pending_restore_and_frozen_snapshots(self):
        device = st.WgpuTensorDevice.create()
        dataset, pipeline = inputs()
        template = pipeline.checkpoint_json()
        trainer = st.ResidentVisionTrainer.create(device, dataset, dataset_id(), configuration(True), pipeline)
        initial = checkpoint(trainer)
        frozen = trainer.checkpoint_snapshot()
        # An unrelated addition and a changed template must not enter this owner.
        dataset.push(st.ImageTensor(3, 4, 4, [0.] * 48), target=st.Tensor(1, 1, [0.]), label="new")
        pipeline.add_center_crop(2, 2)
        submitted = trainer.submit_next()
        with self.assertRaises(ValueError):
            trainer.restore_checkpoint_json(initial)
        trainer.settle()
        self.assertNotIn("new", submitted.labels())
        self.assertEqual(frozen.read_json(), initial)
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            frozen.read_json()
        current = checkpoint(trainer)
        corrupt = json.loads(current)
        corrupt["trainer"]["accepted_updates"] += 1
        with self.assertRaises(ValueError):
            trainer.restore_checkpoint_json(json.dumps(corrupt))
        self.assertEqual(checkpoint(trainer), current)
        trainer.restore_checkpoint_json(initial)
        self.assertEqual(checkpoint(trainer), initial)
        with self.assertRaises(ValueError):
            st.ResidentVisionTrainer.from_checkpoint_json(device, dataset, dataset_id(), initial, pipeline)
        fresh, matching = inputs()
        self.assertEqual(matching.checkpoint_json(), template)
        with self.assertRaises(ValueError):
            st.ResidentVisionTrainer.from_checkpoint_json(device, fresh, "f" * 64, initial, matching)
        del trainer
        self.assertEqual(len(submitted.images().snapshot().read_values()), 96)

    def test_no_pipeline_and_invalid_initial_configuration(self):
        device = st.WgpuTensorDevice.create()
        dataset, _ = inputs()
        trainer = st.ResidentVisionTrainer.create(device, dataset, dataset_id(), configuration(False))
        initial = checkpoint(trainer)
        restored = st.ResidentVisionTrainer.from_checkpoint_json(device, dataset, dataset_id(), initial)
        self.assertEqual(steps(trainer, 2), steps(restored, 2))
        self.assertEqual(checkpoint(trainer), checkpoint(restored))
        for field, value in (("model_seed", 43), ("shuffle_seed", "01"), ("batch_size", 3),
                             ("learning_rate", {"kind": "constant", "rate": -1.})):
            config = json.loads(configuration(False))
            config[field] = value
            with self.assertRaises(ValueError):
                st.ResidentVisionTrainer.create(device, dataset, dataset_id(), json.dumps(config))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_run(bool(int(sys.argv[2])), sys.argv[3], sys.argv[4],
                  sys.argv[5] if len(sys.argv) > 5 else None)
    else:
        unittest.main()
