"""Offline checks for timing boundaries and paired runtime comparison."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("timing", Path(__file__).with_name("bench_vision_trainer_vs_torch.py"))
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
spec = importlib.util.spec_from_file_location("verify_timing", Path(__file__).with_name("verify_vision_training_timing.py"))
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def record(runtime="baseline"):
    return dict(status="passed", recipe=dict(batch_size=16, steps=16, warmup=3,
        train_per_class=128, test_per_class=32, seed=17, mode="feedback", profile=False,
        framework_order=["spiraltorch", "torch"], rate=0.01, runtime=runtime, repeat=0),
        measurements={name: dict(elapsed_ns=1_000_000_000, phases_ns=[], examples_per_second=256.)
                      for name in ("spiraltorch", "torch")},
        final_comparison=[dict(max_scaled_error=1e-7)], final_checkpoint=dict(sha256="same"),
        contract={key: "same" for key in ("config", "data", "dataset_sha256", "initial_checkpoint_sha256",
                                         "source_sha256", "environment", "adapter")})


class TimingChecks(unittest.TestCase):
    def test_retained_parameter_checks_recompute_values_and_reject_nan(self):
        left = [dict(name="a", shape=[2], values=[1., 2.])]
        self.assertEqual(verify.compare_parameters(left, left)[0]["max_scaled_error"], 0.)
        for right in ([dict(name="a", shape=[2], values=[1., 3.])],
                      [dict(name="a", shape=[2], values=[1., float("nan")])],
                      [dict(name="b", shape=[2], values=[1., 2.])], []):
            with self.subTest(right=right), self.assertRaises(ValueError):
                verify.compare_parameters(left, right)

    def test_missing_whole_condition_cannot_pass(self):
        saved = dict(status="passed", schema="spiraltorch.vision.training_timing_sweep.v1",
                     requested=dict(seeds=[17], batches=[16], modes=["feedback"], repeats=1,
                                    runtimes=["baseline", "candidate"], steps=16, warmup=3, profile=False),
                     records=[record(), record("candidate")])
        verify.verify_coverage(saved)
        saved["records"].pop()
        with self.assertRaisesRegex(ValueError, "cases"):
            verify.verify_coverage(saved)

    def test_valid_pair_requires_exact_checkpoint(self):
        result = timing.summarize([record(), record("candidate")])
        self.assertEqual(result["pairs"][0]["candidate_over_baseline"], 1.)
        self.assertTrue(result["pairs"][0]["exact_checkpoint"])

    def test_partial_or_cross_epoch_intervals_fail(self):
        for key, value in (("batch_size", 3), ("steps", 81), ("warmup", 81), ("seed", -1),
                           ("rate", float("nan")), ("framework_order", ["torch", "torch"])):
            r = record()["recipe"]
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                timing.validate_recipe(r)

    def test_bad_duration_or_instrumentation_cannot_be_throughput(self):
        for key, value in (("elapsed_ns", 0), ("phases_ns", [[1, 2]]), ("examples_per_second", 1.)):
            r = record()
            r["measurements"]["torch"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                timing.summarize([r])

    def test_candidate_contract_or_checkpoint_drift_fails(self):
        for key in record()["contract"]:
            r = record("candidate")
            r["contract"][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract differs"):
                timing.summarize([record(), r])
        r = record("candidate")
        r["final_checkpoint"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            timing.summarize([record(), r])

    def test_missing_pair_or_duplicate_cannot_pass(self):
        r = record()
        r["recipe"]["repeat"] = 1
        with self.assertRaisesRegex(ValueError, "missing paired"):
            timing.summarize([record(), record("candidate"), r])
        with self.assertRaisesRegex(ValueError, "duplicate runtime"):
            timing.summarize([record(), record()])

    def test_failed_workers_and_mixed_boundaries_fail(self):
        r = record()
        r["status"] = "error"
        with self.assertRaises(ValueError):
            timing.summarize([r])
        r = copy.deepcopy(record("candidate"))
        r["recipe"]["profile"] = True
        with self.assertRaisesRegex(ValueError, "mixed measurement"):
            timing.summarize([record(), r])


if __name__ == "__main__":
    unittest.main()
