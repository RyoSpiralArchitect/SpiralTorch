"""Fail-closed and fixed-state tests, runnable with python3 -I -B test_probe.py."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("gradient_probe", Path(__file__).with_name("probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Snapshot:
    def __init__(self, value):
        self.raw = json.dumps(value).encode()
        self.reads = 0

    def read_bytes(self):
        self.reads += 1
        # A reread would hash bytes other than those actually compared.
        return self.raw if self.reads == 1 else b'{"changed":true}'


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.request = {"config": {"width": 1}, "cases": [{"name": "ordinary",
                        "parameters": [{"name": "block.0.qkv.bias", "shape": [3]}]}]}
        self.saved = {"cases": [{"final_parameters": [[0., 1., 2.]]}]}
        rows = [{"batch_index": i, "loss": 1., "gradients": [[0.5, 0., -0.5]]} for i in probe.INDICES]
        self.native = {"schema": "spiraltorch.byte_gradient_probe.v1",
                       "request_sha256": probe.PINS["request"],
                       "cases": [{"name": "ordinary", "parameter_values": [[0., 1., 2.]],
                                  "checkpoint_unchanged": True, "probes": rows}]}
        self.reference = {"schema": "spiraltorch.byte_gradient_reference.v1", "pins": probe.PINS,
                          "cases": [{"name": "ordinary", "modes": {
                              mode: copy.deepcopy(rows) for mode in (
                                  "float32_legacy", "float32_module", "float64_module")}}]}

    def compare(self, native=None):
        paths = {"reference": Snapshot(self.reference), "before": Snapshot(self.native),
                 "after": Snapshot(self.native if native is None else native)}
        args = SimpleNamespace(request=None, saved=None, output=None, **paths)
        with patch.object(probe, "inputs", return_value=(self.request, self.saved)), patch.object(probe, "write") as write:
            probe.compare(args)
        return write.call_args.args[1], paths

    def test_hashes_are_bound_to_parsed_bytes(self):
        report, paths = self.compare()
        self.assertEqual(len(report["probes"]), 4)
        self.assertEqual(report["sources"], {name: probe.digest(path.raw) for name, path in paths.items()})
        self.assertTrue(all(path.reads == 1 for path in paths.values()))

    def test_rejects_shape_and_nonfinite_gradients(self):
        for values in ([0., 0.], [0., float("nan"), 0.]):
            bad = copy.deepcopy(self.native)
            bad["cases"][0]["probes"][0]["gradients"][0] = values
            with self.assertRaises(ValueError):
                self.compare(bad)

    def test_rejects_checkpoint_or_signed_zero_parameter_changes(self):
        bad = copy.deepcopy(self.native)
        bad["cases"][0]["checkpoint_unchanged"] = False
        with self.assertRaisesRegex(ValueError, "immutability"):
            self.compare(bad)
        bad = copy.deepcopy(self.native)
        bad["cases"][0]["parameter_values"][0][0] = -0.
        with self.assertRaisesRegex(ValueError, "parameter bits"):
            self.compare(bad)

    def test_rejects_coverage_schema_or_loss_drift(self):
        for mutation in (lambda v: v.update(schema="wrong"),
                         lambda v: v["cases"][0]["probes"].pop(),
                         lambda v: v["cases"][0]["probes"][0].update(loss=float("inf"))):
            bad = copy.deepcopy(self.native)
            mutation(bad)
            with self.assertRaises(ValueError):
                self.compare(bad)
        self.reference["cases"][0]["modes"].pop("float32_legacy")
        with self.assertRaisesRegex(ValueError, "mode coverage"):
            self.compare()

    def test_high_precision_key_shift_invariance(self):
        import torch
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(11)
        shapes = [(4,), (4,), (4, 12), (12,), (4, 4), (4,), (4,), (4,), (4, 8), (8,), (8, 4), (4,)]
        p = [(torch.randn(shape, generator=generator, dtype=torch.float64) * 0.1).requires_grad_() for shape in shapes]
        x = torch.randn((1, 3, 4), generator=generator, dtype=torch.float64)
        original = probe.high_precision_block(torch, x, p, 2, None)
        grad, = torch.autograd.grad(original.square().sum(), p[3])
        shifted = [v.detach().clone() for v in p]
        shifted[3][4:8] += 0.125
        changed = probe.high_precision_block(torch, x, shifted, 2, None)
        self.assertLess((original - changed).abs().max().item(), 1e-14)
        self.assertLess(grad[4:8].abs().max().item(), 1e-14)


if __name__ == "__main__":
    unittest.main()
