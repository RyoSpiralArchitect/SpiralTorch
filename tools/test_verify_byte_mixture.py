"""Fabricated records test acceptance boundaries, not learning correctness."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("flat_tests", Path(__file__).with_name("test_verify_byte_flat_metric.py"))
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)
verify = flat.module


def inputs():
    reference, report = flat.inputs()
    reference["schema"] = "spiraltorch.resident_byte_geometry_mixture.torch_fixture.v1"
    report["schema"] = "spiraltorch.resident_byte_geometry_mixture.validation.v1"
    report["mixture_guard"] = {"schema": "spiraltorch.byte_mixture_guard.v1", "fault_head": 1,
        "fault_component": 0, "parameter_tensors": 24, "finite_control": True,
        "zero_mass_control": True, "all_gradients_guarded": True, "atomic_rejection": True,
        "baseline_max_abs_bias": 3.}
    report["wide_pullback_control"] = {"schema": "spiraltorch.mixture_wide_pullback.v1",
        "checks": verify.mixture_wide_reference(), "nonfinite_final_rejected": True}
    for index, (case, check) in enumerate(zip(reference["cases"], report["checks"])):
        blocks, end = index + 1, 7 + index
        mixture = {"metrics": verify.MIXTURE_METRICS.copy(), "scales": copy.deepcopy(verify.MIXTURE_SCALES[:blocks])}
        del case["config"]["causal_geometry"]["pair_metric"]
        del check["metric"]
        case["config"]["causal_geometry"]["mixture"] = mixture
        check["mixture"] = copy.deepcopy(mixture)
        router = [{"name": f"geometry.mixture_logits.{i}", "shape": [2, 3], "values": [.1] * 6} for i in range(blocks)]
        case["parameters"][end:end] = copy.deepcopy(router)
        check["parameter_count"] += blocks
        check["checkpoint"]["parameter_count"] += blocks
        check["metric_controls"]["geometry_parameter_slots"][1] += blocks
        check["parameter_names"] = [p["name"] for p in case["parameters"]]
        check["parameter_shapes"] = [p["shape"] for p in case["parameters"]]
        for record in (case, check):
            record["parameter_gradients"][end:end] = [[1e-7] * 6 for _ in range(blocks)]
            record["mixture_weights"] = [[1. / 3] * 6 for _ in range(blocks)]
            for n, row in enumerate(record["learning"]["trace"], 1):
                row["parameters"][end:end] = [[.1 + n * .001] * 6 for _ in range(blocks)]
                row["geometry_gradients"] += [[1e-7] * 6 for _ in range(blocks)]
                row["mixture_weights"] = [[1. / 3] * 6 for _ in range(blocks)]
        check["resume_trajectory"]["exact_mixture_weights"] = True
        check["resume_trajectory"]["trace"] = copy.deepcopy(check["learning"]["trace"])
        for key, revision in (("checkpoint_json", 7), ("final_checkpoint_json", 16)):
            check["resume_trajectory"][key] = json.dumps(verify.expected_checkpoint(
                case, check["learning"]["trace"][revision - 1]["parameters"], revision, None))
    return reference, report


class MixtureVerifierTests(unittest.TestCase):
    def reject(self, mutate):
        reference, report = inputs()
        mutate(reference, report)
        with self.assertRaises((ValueError, KeyError)):
            verify.verify(reference, report, kind="mixture")

    def test_positive_and_no_legacy_relabel(self):
        self.assertTrue(verify.verify(*inputs(), kind="mixture")["passed"])
        for args, kind in ((inputs(), "flat"), (inputs(), "headwise"), (flat.inputs(), "mixture")):
            with self.assertRaises(ValueError):
                verify.verify(*args, kind=kind)

    def test_routes_and_scales_are_bound_on_both_surfaces(self):
        for key in ("metrics", "scales"):
            for surface in ("reference", "report", "both"):
                def change(r, a):
                    if surface in ("reference", "both"):
                        r["cases"][1]["config"]["causal_geometry"]["mixture"][key].reverse()
                    if surface in ("report", "both"):
                        a["checks"][1]["mixture"][key].reverse()
                self.reject(change)

    def test_small_router_component_cannot_hide_behind_large_neighbor(self):
        for coordinate in range(6):
            def change(r, a):
                for trace, value in ((r["cases"][0]["learning"]["trace"], 1e-8),
                                     (a["checks"][0]["learning"]["trace"], 0.)):
                    for row in trace:
                        row["geometry_gradients"][-1] = [1.] * 6
                        row["geometry_gradients"][-1][coordinate] = value
                a["checks"][0]["resume_trajectory"]["trace"] = copy.deepcopy(a["checks"][0]["learning"]["trace"])
            self.reject(change)

    def test_masses_are_complete_finite_normalized_and_exact_on_resume(self):
        self.reject(lambda r, a: a["checks"][1]["mixture_weights"].pop())
        for bad in (True, float("nan"), -1., 2.):
            self.reject(lambda r, a: a["checks"][0]["mixture_weights"][0].__setitem__(0, bad))
        def unnormalized(r, a):
            for record in (r["cases"][0], a["checks"][0]):
                record["mixture_weights"][0] = [.2] * 6
        self.reject(unnormalized)
        self.reject(lambda r, a: a["checks"][1]["resume_trajectory"]["trace"][8]["mixture_weights"][0].__setitem__(0, 1. / 3 + 1e-7))
        self.reject(lambda r, a: a["checks"][0]["resume_trajectory"].pop("exact_mixture_weights"))

    def test_checkpoint_binds_all_mixture_fields_and_values(self):
        for field in ("checkpoint_json", "final_checkpoint_json"):
            for change in (
                lambda cp: cp.__setitem__("schema", "spiraltorch.nn.byte_decoder_checkpoint.v3"),
                lambda cp: cp["model"]["geometry"].__setitem__("pair_metric", verify.METRIC),
                lambda cp: cp["model"]["geometry"]["mixture"]["metrics"].reverse(),
                lambda cp: cp["model"]["geometry"]["mixture"]["scales"][0].__setitem__(0, 2.),
                lambda cp: cp["model"]["geometry"]["mixture"]["raw_logits"][0].__setitem__(0, .107 + 1e-6),
            ):
                def mutate(r, a):
                    resume = a["checks"][0]["resume_trajectory"]
                    cp = json.loads(resume[field])
                    change(cp)
                    resume[field] = json.dumps(cp)
                self.reject(mutate)

    def test_full_guard_and_no_missing_router_slots(self):
        for k in ("finite_control", "zero_mass_control", "all_gradients_guarded", "atomic_rejection"):
            self.reject(lambda r, a: a["mixture_guard"].__setitem__(k, False))
        self.reject(lambda r, a: a["checks"][1]["parameter_gradients"].pop())
        self.reject(lambda r, a: a["checks"][0]["learning"]["trace"][0]["geometry_gradients"].pop())
        self.reject(lambda r, a: a["checks"][0]["parameter_gradients"][7].__setitem__(0, 0.))

    def test_wide_pullback_cannot_erase_small_signal_or_accept_final_overflow(self):
        self.reject(lambda r, a: a.pop("wide_pullback_control"))
        self.reject(lambda r, a: a["wide_pullback_control"].__setitem__("nonfinite_final_rejected", False))
        self.reject(lambda r, a: a["wide_pullback_control"]["checks"].pop())
        for index in range(3):
            self.reject(lambda r, a: a["wide_pullback_control"]["checks"][index].__setitem__("logits", [0., 0.]))
        self.reject(lambda r, a: a["wide_pullback_control"]["checks"][3]["logits"].__setitem__(0, 1e-8))
        self.reject(lambda r, a: a["wide_pullback_control"]["checks"][0]["components"].__setitem__(0, float("inf")))


if __name__ == "__main__":
    unittest.main()
