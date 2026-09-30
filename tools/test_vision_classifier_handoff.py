"""The handoff verifier must not pass truncated, non-finite or mismatched evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("handoff", Path(__file__).with_name("verify_vision_classifier_handoff.py"))
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffChecks(unittest.TestCase):
    def test_value_comparison_rejects_empty_short_nonfinite_and_wrong_values(self):
        for a, b in (([], []), ([1], [1, 2]), ([float("nan")], [0]), ([0], [float("inf")]),
                     ([True], [1]), ([0.1], [0])):
            with self.subTest(actual=a, expected=b), self.assertRaises(ValueError):
                handoff.compare_values(a, b, "test")
        self.assertEqual(handoff.compare_values([1], [1], "test")["max_scaled_error"], 0)

    def test_every_parameter_and_metadata_are_compared(self):
        state = dict(schema="test", classes=2, head=[dict(name="head", shape=[1], values=[0.5])],
                     backbone=dict(attempted_updates=7, parameters=[dict(name="stem", shape=[1], values=[1])]))
        self.assertEqual(handoff.compare_checkpoints(json.dumps(state), json.dumps(state))["parameters"], 2)
        for change in (lambda s: s["head"][0].update(values=[0.9]),
                       lambda s: s["head"][0].update(name="wrong"),
                       lambda s: s["head"][0].update(shape=[1, 1]),
                       lambda s: s["head"].clear(),
                       lambda s: s["backbone"].update(attempted_updates=6)):
            bad = copy.deepcopy(state)
            change(bad)
            with self.assertRaises(ValueError):
                handoff.compare_checkpoints(json.dumps(bad), json.dumps(state))

    def test_browser_fixture_requires_exact_recipe_and_guard_evidence(self):
        fixture = dict(schema="spiraltorch.vision.classifier_clients.v1", status="passed",
                       fixture_request="convnext-classifier-clients", page_errors=[],
                       within_runtime_resume="bitwise", accepted_revisions=[1, 2, 3, 4, 6],
                       rejected_revision=5, continued_revision=7, continuation_rate=0.01,
                       labels=[0, 1], parameters=24, rust_runtime_adapter=dict(backend="BrowserWebGpu"),
                       checks=["frozen_checkpoint", "bitwise_resume", "invalid_labels_preserve_all_weights",
                               "valid_retry", "retained_handles"], normalized_input=[0.] * 128,
                       final_logits=[0.] * 4, continued_logits=[0.] * 4, losses=[0.] * 4, continued_loss=0.)
        handoff.validate_fixture(fixture)
        for key, bad in (("status", "failed"), ("parameters", 23), ("continued_revision", 6),
                         ("page_errors", ["uncaught"]), ("checks", []), ("normalized_input", [0.]),
                         ("labels", [9, 9]), ("losses", [float("nan")] * 4)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                handoff.validate_fixture({**fixture, key: bad})


if __name__ == "__main__":
    unittest.main()
