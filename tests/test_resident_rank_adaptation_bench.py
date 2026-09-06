"""Synthetic receipt validation only; these fixtures are not performance evidence."""
from copy import deepcopy
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_resident_rank_adaptation_vs_torch as bench


def fixture():
    request = dict(kind="topk", rows=2, cols=257, k=7, seed=17, policy="ucb",
                   scripts=["u2: true; rank_tile: 512;"], rounds=1)
    candidate = dict(index=0, spiralk_source_sha256=hashlib.sha256(request["scripts"][0].encode()).hexdigest(),
                     execution_signature="spiraltorch.rank_execution.v1/backend=wgpu/kind=topk/rows=2/cols=257/k=7/fallback=forbid/scope=declared_native/path=exact_2ce/tile=257",
                     plan={"fixture": "identity comparison only"})
    initial = dict(candidates=[candidate], observation_counts={"rank_plan_variant": {"0": 0}},
                   pending_selection_id=None)
    final = deepcopy(initial)
    final["observation_counts"]["rank_plan_variant"]["0"] = 1
    selection = dict(candidate_index=0, selection_id=1, **{key: candidate[key] for key in
                     ["execution_signature", "spiralk_source_sha256", "plan"]})
    observation = dict(candidate_index=0, selection_id=1, elapsed_ms=16.0,
                       credited=True, correctness_passed=True, candidate_quarantined=False,
                       reward=1 / 17, observation_counts=deepcopy(final["observation_counts"]))
    result = dict(request, status="passed", repetitions=16, initial=initial, rank_adaptation=final,
                  control_samples_per_op_ms=[[1.0] * 12],
                  observations=[dict(selection=selection, observation=observation, per_op_ms=1.0)])
    return result, request, [512]


class ResidentAdaptationAuditTest(unittest.TestCase):
    def test_suite_preserves_shapes_seeds_policies_and_default_control(self):
        cases = list(bench.requests(bench.audit.load_bench_module()))
        self.assertEqual(len(cases), 36)
        self.assertEqual({r["seed"] for r, _ in cases}, {17, 29, 43})
        self.assertEqual({r["policy"] for r, _ in cases}, {"ucb", "thompson_sampling"})
        self.assertEqual({(r["cols"], r["k"]) for r, _ in cases}, {(257, 7), (8193, 65)})
        for request, tiles in cases:
            self.assertEqual(set(tiles), {32, 128, 256, 512})
            self.assertEqual(len(request["input"]), request["rows"] * request["cols"])
            self.assertEqual(request["rounds"], 64)
            for script, tile in zip(request["scripts"], tiles):
                self.assertEqual(script, f"u2: true; rank_tile: {tile}; ctile: {tile};")

    def test_valid_completed_batch_receipt(self):
        bench.validate_native(*fixture())

    def test_wrong_execution_identity_or_source_fails(self):
        for key, value in [("execution_signature", "other/path=exact_2ce/tile=257"),
                           ("spiralk_source_sha256", "wrong"), ("index", False)]:
            result, request, tiles = fixture()
            result["initial"]["candidates"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_native(result, request, tiles)

    def test_bad_selection_or_timing_boundary_fails(self):
        changes = [("candidate_index", False), ("candidate_index", 1), ("selection_id", 2),
                   ("execution_signature", "wrong"), ("spiralk_source_sha256", "wrong"),
                   ("plan", {})]
        for key, value in changes:
            result, request, tiles = fixture()
            result["observations"][0]["selection"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_native(result, request, tiles)
        for elapsed in [-1.0, float("nan"), float("inf"), 1.0]:
            result, request, tiles = fixture()
            result["observations"][0]["observation"]["elapsed_ms"] = elapsed
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                bench.validate_native(result, request, tiles)

    def test_incorrect_or_uncredited_or_misrewarded_result_fails(self):
        for key, value in [("credited", False), ("correctness_passed", False),
                           ("candidate_quarantined", True), ("reward", 0.5),
                           ("observation_counts", {"rank_plan_variant": {"0": 2}})]:
            result, request, tiles = fixture()
            result["observations"][0]["observation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_native(result, request, tiles)

    def test_controls_cannot_credit_and_final_must_be_complete(self):
        for target, change in [("initial", {"observation_counts": {"rank_plan_variant": {"0": 1}}}),
                               ("rank_adaptation", {"pending_selection_id": 2}),
                               ("rank_adaptation", {"observation_counts": {"rank_plan_variant": {"0": 2}}}),
                               ("rank_adaptation", {"candidates": []})]:
            result, request, tiles = fixture()
            result[target].update(change)
            with self.subTest(target=target, change=change), self.assertRaises(ValueError):
                bench.validate_native(result, request, tiles)


if __name__ == "__main__":
    unittest.main()
