"""Explicit Fisher-Rao identity must survive shared metric verification."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("flat_tests", Path(__file__).with_name("test_verify_byte_flat_metric.py"))
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)
verify = flat.module
METRIC = verify.METRICS["fisher_rao"]


def inputs():
    reference, report = flat.inputs()
    reference["schema"] = "spiraltorch.resident_byte_geometry_fisher_rao.torch_fixture.v1"
    report["schema"] = "spiraltorch.resident_byte_geometry_fisher_rao.validation.v1"
    report["wide_pullback_control"] = {
        "schema": "spiraltorch.fisher_rao_wide_pullback.v1", **verify.fisher_wide_reference()}
    for case, check in zip(reference["cases"], report["checks"]):
        case["config"]["causal_geometry"]["pair_metric"] = check["metric"] = METRIC
        for key, revision in (("checkpoint_json", 7), ("final_checkpoint_json", 16)):
            check["resume_trajectory"][key] = json.dumps(verify.expected_checkpoint(
                case, check["learning"]["trace"][revision - 1]["parameters"], revision, METRIC))
    return reference, report


class FisherVerifierTests(unittest.TestCase):
    def test_wide_boundary_is_required_and_checked(self):
        for mutation in (
            lambda a: a.pop("wide_pullback_control"),
            lambda a: a["wide_pullback_control"]["coordinates"].__setitem__(0, 0.),
            lambda a: a["wide_pullback_control"]["raw_gain"].__setitem__(0, float("inf")),
            lambda a: a["wide_pullback_control"].__setitem__("schema", "unknown"),
        ):
            reference, report = inputs()
            mutation(report)
            with self.assertRaises(ValueError):
                verify.verify(reference, report, kind="fisher_rao")

    def test_explicit_mode_and_no_default_relabel(self):
        self.assertTrue(verify.verify(*inputs(), kind="fisher_rao")["passed"])
        for args, kind in ((inputs(), "flat"), (flat.inputs(), "fisher_rao"), (inputs(), "unknown")):
            with self.assertRaises(ValueError):
                verify.verify(*args, kind=kind)

    def test_metric_is_bound_at_every_surface(self):
        for surface in ("reference", "report", "checkpoint_json", "final_checkpoint_json"):
            reference, report = inputs()
            if surface == "reference":
                reference["cases"][0]["config"]["causal_geometry"]["pair_metric"] = verify.METRIC
            elif surface == "report":
                report["checks"][0]["metric"] = verify.METRIC
            else:
                checkpoint = report["checks"][0]["resume_trajectory"]
                checkpoint[surface] = checkpoint[surface].replace(METRIC, verify.METRIC)
            with self.assertRaises(ValueError):
                verify.verify(reference, report, kind="fisher_rao")

    def test_no_gradient_or_resume_gate_relaxation(self):
        for mutation in (
            lambda r, a: a["checks"][0]["learning"]["trace"][7]["geometry_gradients"][0].__setitem__(0, 0.),
            lambda r, a: a["checks"][1]["resume_trajectory"]["trace"][9]["parameters"][0].__setitem__(0, .11 + 1e-7),
            lambda r, a: r["tolerance"].__setitem__("rtol", .1),
            lambda r, a: a["checks"][0]["output"].__setitem__(0, True),
            lambda r, a: a["checks"].pop(),
        ):
            reference, report = inputs()
            mutation(reference, report)
            with self.assertRaises(ValueError):
                verify.verify(reference, report, kind="fisher_rao")


if __name__ == "__main__":
    unittest.main()
