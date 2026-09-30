"""Offline admission checks; no dataset download or GPU is required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


def module(filename):
    spec = importlib.util.spec_from_file_location(filename, Path(__file__).with_name(filename + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module("run_vision_matched_learning")
reference = module("vision_convnext_torch_reference")


def checkpoint():
    def parameter(name, shape, values):
        return dict(name=name, shape=shape, values=values)
    return dict(schema="spiraltorch.convnext.classifier_plain_sgd_checkpoint.v1", classes=2,
                backbone=dict(config=dict(input_channels=1, input_hw=[2, 2], stage_dims=[1],
                                          stage_depths=[0], patch_size=[1, 1], epsilon=1e-3, curvature=-1.),
                              parameters=[parameter("convnext.stem::weight", [1, 1], [1.]),
                                          parameter("convnext.stem::bias", [1, 1], [0.]),
                                          parameter("convnext.final_norm_gamma", [1, 4], [1., 2., 3., 4.]),
                                          parameter("convnext.final_norm_beta", [1, 4], [0., 0., 0., 0.])]),
                head=[parameter("convnext.classifier::weight", [1, 2], [0.4, -0.2]),
                      parameter("convnext.classifier::bias", [1, 2], [0.1, -0.1])])


class MatchedLearningChecks(unittest.TestCase):
    def test_reference_matches_flattened_norm_before_pooling(self):
        model = reference.ConvNeXtReference(json.dumps(checkpoint()))
        x = torch.tensor([0., 1., 2., 4.]).reshape(1, 1, 2, 2).requires_grad_(True)
        flat = x.reshape(1, 4)
        normalized = (flat - flat.mean(1, keepdim=True)) / torch.sqrt(flat.var(1, correction=0, keepdim=True) + model.epsilon)
        pooled = (normalized * torch.tensor([1., 2., 3., 4.])).mean(1, keepdim=True)
        expected = pooled @ torch.tensor([[0.4, -0.2]]) + torch.tensor([0.1, -0.1])
        torch.testing.assert_close(model(x), expected, rtol=1e-6, atol=1e-6)
        model(x).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(x.grad.shape, x.shape)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 1, 3, 3))

    def test_reference_rejects_wrong_schema_and_duplicate_roles(self):
        state = checkpoint()
        with self.assertRaises(ValueError):
            reference.ConvNeXtReference(json.dumps({**state, "schema": "wrong"}))
        state["head"][1]["name"] = state["head"][0]["name"]
        with self.assertRaises(ValueError):
            reference.ConvNeXtReference(json.dumps(state))

    def test_comparison_does_not_admit_missing_nonfinite_or_changed_values(self):
        for a, b in (([], []), ([1], [1, 2]), ([np.nan], [0]), ([0], [np.inf]), ([1], [0])):
            with self.assertRaises(ValueError):
                runner.compare(a, b, "test")

    def test_balanced_subset_has_exact_counts_and_replayable_order(self):
        labels = np.repeat(np.arange(10), 5)
        indices = runner.balanced_indices(labels, 3, 17)
        self.assertTrue(np.array_equal(indices, runner.balanced_indices(labels, 3, 17)))
        self.assertEqual(len(set(indices)), 30)
        self.assertEqual(np.bincount(labels[indices]).tolist(), [3] * 10)
        with self.assertRaises(ValueError):
            runner.balanced_indices(labels, 6, 17)

    def test_checkpoint_artifacts_are_exact_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            payload = json.dumps(checkpoint())
            receipt = runner.save_checkpoint(path, "initial.json", payload)
            self.assertEqual((path / receipt["file"]).read_text(), payload)
            self.assertEqual(receipt["bytes"], len(payload.encode()))
            with self.assertRaises(FileExistsError):
                runner.save_checkpoint(path, "initial.json", "replacement")


if __name__ == "__main__":
    unittest.main()
