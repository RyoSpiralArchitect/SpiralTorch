"""Mutation checks for the offline convolution diagnostic verifier."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("profile_verifier", Path(__file__).with_name("verify_vision_gpu_profile.py"))
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class ProfileChecks(unittest.TestCase):
    def steps(self):
        operations = []
        for index, kind in enumerate(["depthwise", "dense", "depthwise", "dense"]):
            passes = [dict(phase=phase, start_tick=str(index * 30 + i * 10),
                           end_tick=str(index * 30 + i * 10 + 5), elapsed_ns=10.)
                      for i, phase in enumerate(["input_vjp", "weight_vjp", "bias_vjp"])]
            operations.append(dict(kind=kind, input=[2, 1, 2, 2], upstream=[2, 1, 2, 2],
                                   timestamp_period_ns=2., passes=passes))
        return [dict(step=1, labels=["21", "20"], host_step_ns=500,
                     gpu=dict(schema="spiraltorch.convolution_vjp_gpu_profile.v1", operations=operations))]

    def check(self, steps, profiled=True):
        return VERIFIER.check_steps(steps, 2, [20, 21], [1, 0], profiled)

    def test_complete(self):
        geometry, rows = self.check(self.steps())
        self.assertEqual(len(geometry), 4)
        self.assertEqual(sum(rows[0]["convolution_pass_ns"]), 120.)

    def test_control(self):
        steps = self.steps()
        steps[0]["gpu"] = None
        self.assertEqual(self.check(steps, False)[1][0]["convolution_pass_ns"], [])
        with self.assertRaises(ValueError):
            self.check(self.steps(), False)

    def test_wrong_input_or_horizon(self):
        for key, value in [("labels", ["20", "21"]), ("step", 2), ("host_step_ns", float("nan"))]:
            steps = self.steps()
            steps[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(steps)
        with self.assertRaises(ValueError):
            self.check([])

    def test_missing_or_duplicate_operation(self):
        for operations in [self.steps()[0]["gpu"]["operations"][:3], [self.steps()[0]["gpu"]["operations"][0]] * 4]:
            steps = self.steps()
            steps[0]["gpu"]["operations"] = operations
            with self.assertRaises(ValueError):
                self.check(steps)

    def test_invalid_timestamp(self):
        for key, value in [("elapsed_ns", 11.), ("elapsed_ns", float("nan")), ("end_tick", "-1"), ("phase", "weight_vjp")]:
            steps = self.steps()
            steps[0]["gpu"]["operations"][0]["passes"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(steps)

    def test_geometry_drift(self):
        steps = self.steps()
        second = copy.deepcopy(steps[0])
        second.update(step=2, labels=[])
        second["gpu"]["operations"][0]["input"][1] = 3
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            VERIFIER.check_steps(steps + [dict(second, labels=["21", "20"])], 2, [20, 21], [1, 0, 1, 0], True)

    def fixture(self, parent):
        root, timing = parent / "profile", parent / "timing"
        name = "seed-17-batch-2-plain-0"
        retained = timing / f"{name}-candidate"
        (root / name).mkdir(parents=True)
        retained.mkdir(parents=True)
        initial = json.dumps(dict(input=dict(position=0, batch_size=2, order=list(range(32))))).encode()
        final, pixels, launcher = b"retained-checkpoint", b"retained-pixels", b"launcher"
        (root / "pixels.u8").write_bytes(pixels)
        (retained / "initial.json").write_bytes(initial)
        (retained / "spiraltorch-final.json").write_bytes(final)
        recipe = dict(seed=17, steps=16, warmup=3, profile_first=False, dataset_id="dataset",
                      initial_sha256=VERIFIER.sha(initial), expected_final_sha256=VERIFIER.sha(final),
                      dataset=dict(ids=list(range(32)), pixels_sha256=VERIFIER.sha(pixels)))
        recipe_blob = json.dumps(recipe).encode()
        (root / f"{name}.json").write_bytes(recipe_blob)
        reference = dict(status="passed", contract=dict(initial_checkpoint_sha256=VERIFIER.sha(initial),
                         dataset_sha256="dataset", data=dict(train_pixels_sha256=VERIFIER.sha(pixels), train_indices=list(range(32)))),
                         final_checkpoint=dict(sha256=VERIFIER.sha(final)), recipe=dict(steps=16, warmup=3))
        (retained / "result.json").write_text(json.dumps(reference))
        routes = []
        for profiled in [False, True]:
            steps = []
            for index in range(16):
                step = self.steps()[0]
                step.update(step=index + 1, labels=[str(index * 2), str(index * 2 + 1)])
                if not profiled:
                    step["gpu"] = None
                steps.append(step)
            routes.append(dict(profiled=profiled, steps=steps, final_sha256=VERIFIER.sha(final)))
            (root / name / ("profiled-final.json" if profiled else "control-final.json")).write_bytes(final)
        result = dict(schema="spiraltorch.vision.trainer_gpu_profile.v1", passed=True,
                      case_sha256=VERIFIER.sha(recipe_blob), batch_size=2, dataset_id="dataset", adapter="fixture",
                      initial_sha256=VERIFIER.sha(initial), expected_final_sha256=VERIFIER.sha(final),
                      exact_retained_checkpoint=True, records=routes)
        (root / name / "result.json").write_text(json.dumps(result))
        grid = dict(seeds=[17], batches=[2], modes=["plain"], repeats=1)
        summary = dict(schema="spiraltorch.vision.gpu_profile_sweep.v1", passed=True, requested=grid,
                       launcher_sha256=VERIFIER.sha(launcher), binary_sha256="0" * 64,
                       records=[dict(case=name, seed=17, batch_size=2, mode="plain", repeat=0, result=result)])
        (root / "summary.json").write_text(json.dumps(summary))
        return root, timing, grid, name, launcher

    def test_full_file_replay_and_tampering(self):
        for mutation in ["none", "checkpoint", "coverage", "source", "pixels", "worker"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, timing, grid, name, launcher = self.fixture(Path(directory))
                if mutation == "checkpoint":
                    (root / name / "profiled-final.json").write_bytes(b"different weight")
                elif mutation == "coverage":
                    summary = VERIFIER.read(root / "summary.json")
                    summary["records"].append(summary["records"][0])
                    (root / "summary.json").write_text(json.dumps(summary))
                elif mutation == "source":
                    launcher = b"wrong source"
                elif mutation == "pixels":
                    (root / "pixels.u8").write_bytes(b"different images")
                elif mutation == "worker":
                    result = VERIFIER.read(root / name / "result.json")
                    result["passed"] = False
                    (root / name / "result.json").write_text(json.dumps(result))
                with mock.patch.object(VERIFIER.subprocess, "check_output", return_value=launcher):
                    if mutation == "none":
                        verification, records, table = VERIFIER.verify(root, timing, grid, "fixture-source")
                        self.assertEqual(verification["exact_checkpoint_comparisons"], 2)
                        self.assertEqual(len(records), 1)
                        self.assertEqual(table[0]["median_convolution_vjp_ms"], 0.00012)
                    else:
                        with self.assertRaises(ValueError):
                            VERIFIER.verify(root, timing, grid, "fixture-source")


if __name__ == "__main__":
    unittest.main()
