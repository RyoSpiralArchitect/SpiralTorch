"""Analytic witnesses for the optional reference graph, independent of receipts."""
import copy
import unittest
from unittest.mock import patch

import torch

import generate_resident_byte_decoder_torch_fixture as reference
import byte_projective_diagnostic_contract as contract
import diagnose_byte_residual_association as diagnostic


def parameters(large):
    vector = lambda v: torch.tensor([v], dtype=torch.float32, device="cpu", requires_grad=True)
    matrix = lambda v: torch.tensor([[v]], dtype=torch.float32, device="cpu", requires_grad=True)
    return [vector(1.), vector(0.), torch.zeros(1, 3, requires_grad=True),
        torch.tensor([0., 0., 1.], requires_grad=True), matrix(large), vector(-large),
        vector(1.), vector(0.), matrix(0.), vector(0.), matrix(0.), vector(0.)]


class ResidualAssociationReferenceTests(unittest.TestCase):
    def test_default_is_frozen_legacy_and_grouped_retains_residual(self):
        for exponent in (24, 48):
            for sign in (-1., 1.):
                with self.subTest(exponent=exponent, sign=sign):
                    p = parameters(sign * 2.**exponent)
                    x = torch.tensor([[[sign]]], dtype=torch.float32, device="cpu")
                    default = reference.block_forward(x, p, 1, None, None, False)
                    legacy = reference.block_forward(x, p, 1, None, None, False, module_residual=False)
                    grouped = reference.block_forward(x, p, 1, None, None, False, module_residual=True)
                    self.assertTrue(torch.equal(default, legacy))
                    self.assertEqual(legacy.item(), 0.)
                    self.assertEqual(grouped.item(), sign)

    def test_both_residual_and_projection_bias_receive_cotangent(self):
        p = parameters(2.**24)
        for grouped in (False, True):
            x = torch.ones(1, 1, 1, device="cpu", requires_grad=True)
            y = reference.block_forward(x, p, 1, None, None, False, module_residual=grouped)
            dx, db = torch.autograd.grad(y, (x, p[5]), torch.full_like(y, 2.))
            self.assertEqual(dx.item(), 2.)
            self.assertEqual(db.item(), 2.)

    def test_reference_does_not_modify_inputs_or_parameters(self):
        p = parameters(2.**24)
        before = [v.detach().clone() for v in p]
        x = torch.ones(1, 1, 1, device="cpu")
        for grouped in (False, True):
            reference.block_forward(x, p, 1, None, None, False, module_residual=grouped)
        self.assertEqual(x.item(), 1.)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(p, before)))


def receipt_pair():
    descriptors = [{"name": f"p{i}", "shape": [2], "values": [0., 0.]} for i in range(3)]
    case = {"name": "test", "parameters": descriptors, "output": [0.],
            "parameter_gradients": [[0., 0.] for _ in descriptors],
            "learning": {"rate": .125, "trace": [{"geometry_gradients": [[0., 0.]]}]}}
    actual = {"name": "test", "parameter_count": 3, "parameter_names": [p["name"] for p in descriptors],
              "parameter_shapes": [[2], [2], [2]], "output": [0.],
              "parameter_gradients": [[0., 0.] for _ in descriptors],
              "qualification_failures": [], "learning": {"steps": 16, "trace": [
                  {"revision": i, "loss": 1., "parameters": [[0., 0.] for _ in descriptors],
                   "geometry_gradients": [[0., 0.]]} for i in range(1, 17)]}}
    return {"cases": [case]}, {"schema": contract.REPORT_SCHEMA, "checks": [actual]}


class DiagnosticContractTests(unittest.TestCase):
    def test_complete_receipt_is_accepted(self):
        fixture, report = receipt_pair()
        contract.validate_report(fixture, report)

    def test_missing_or_extra_cases_are_rejected(self):
        fixture, report = receipt_pair()
        for checks in ([], report["checks"] * 2):
            with self.subTest(count=len(checks)), self.assertRaises(ValueError):
                contract.validate_report(fixture, {**report, "checks": checks})

    def test_wrong_identity_names_shapes_and_revision_are_rejected(self):
        fixture, original = receipt_pair()
        mutations = [
            lambda c: c.update(name="different"),
            lambda c: c.update(parameter_names=list(reversed(c["parameter_names"]))),
            lambda c: c.update(parameter_shapes=[[1, 2], [2], [2]]),
            lambda c: c.update(parameter_count=2),
            lambda c: c["learning"].update(steps=12),
            lambda c: c["learning"]["trace"][0].update(revision=2),
            lambda c: c["learning"]["trace"][0].update(revision=True),
        ]
        for index, mutate in enumerate(mutations):
            report = copy.deepcopy(original)
            mutate(report["checks"][0])
            with self.subTest(index=index), self.assertRaises(ValueError):
                contract.validate_report(fixture, report)

    def test_incomplete_and_nonfinite_arrays_are_rejected(self):
        fixture, original = receipt_pair()
        mutations = [
            lambda c: c["learning"]["trace"].pop(),
            lambda c: c["parameter_gradients"].pop(),
            lambda c: c["learning"]["trace"][0]["parameters"].pop(),
            lambda c: c["learning"]["trace"][0]["geometry_gradients"].pop(),
            lambda c: c["learning"]["trace"][0]["parameters"][0].pop(),
            lambda c: c["learning"]["trace"][0]["geometry_gradients"][0].append(0.),
            lambda c: c.update(output=[float("nan")]),
            lambda c: c["learning"]["trace"][0].update(loss=float("inf")),
            lambda c: c["parameter_gradients"][0].__setitem__(0, True),
        ]
        for index, mutate in enumerate(mutations):
            report = copy.deepcopy(original)
            mutate(report["checks"][0])
            with self.subTest(index=index), self.assertRaises(ValueError):
                contract.validate_report(fixture, report)

    def test_comparison_rejects_broadcasting_and_nonfinite_values(self):
        for actual, expected in [([1.], [1., 1.]), ([], []), ([float("nan")], [1.])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                diagnostic.compare(actual, expected)

    def test_legacy_replay_checks_survive_optimized_python(self):
        fixture, _ = receipt_pair()
        case = fixture["cases"][0]
        result = {"output": torch.zeros(1, dtype=torch.float64),
                  "gradients": [torch.zeros(2, dtype=torch.float64) for _ in range(3)],
                  "geometry_end": 3, "loss": 1.}
        for target in ("output", "parameter_gradients", "update"):
            altered = copy.deepcopy(case)
            if target == "output":
                altered["output"] = [1.]
            elif target == "parameter_gradients":
                altered["parameter_gradients"][0][0] = 1.
            else:
                altered["learning"]["trace"][0]["parameters"] = [[1., 1.] for _ in range(3)]
            with self.subTest(target=target), patch.object(diagnostic.same, "evaluate", return_value=result):
                with self.assertRaisesRegex(ValueError, "legacy .* replay mismatch"):
                    diagnostic.replay(altered, False)

    def test_unknown_failure_text_is_not_published(self):
        _, report = receipt_pair()
        case = report["checks"][0]
        case["query_controls_error"] = "unexpected payload"
        with self.assertRaises(ValueError):
            contract.inherited_failures(case)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
