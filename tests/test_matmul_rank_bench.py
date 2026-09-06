"""Equal scores must not admit different cutoff indices into a matched benchmark."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_matmul_rank_vs_torch as bench
from bench_matmul_rank_vs_torch import require_canonical_indices


class CanonicalRankTest(unittest.TestCase):
    def test_matching_rows_pass(self):
        require_canonical_indices([[0, 1], [3, 4]], [[0, 1], [3, 4]])

    def test_different_tied_selection_fails(self):
        with self.assertRaisesRegex(RuntimeError, "canonical stable"):
            require_canonical_indices([[0, 2, 3]], [[0, 1, 3]])

    def test_different_order_or_shape_fails(self):
        for actual in [[[1, 0]], [[0]], [[0, 1], []]]:
            with self.assertRaises(RuntimeError):
                require_canonical_indices(actual, [[0, 1]])


class ReadbackProbeTest(unittest.TestCase):
    def test_probe_rejects_comparison_samples_or_changed_outputs(self):
        request = dict(rows=1, inner=1, cols=8, k=1, kind="topk", seed=17)
        comparison = dict(values=[7.], indices=[4], adapter={"name": "fixture"})
        result = dict(request, **comparison, status="passed", mode="probe_only",
                      samples_ms=None, readback_probe={"status": "passed"})
        for changes in ({}, {"samples_ms": {"host_bridge": [1.]}}, {"indices": [0]},
                        {"seed": 29}, {"mode": "comparison"}, {"readback_probe": None}):
            with self.subTest(changes=changes), patch.object(
                bench, "run_native_pass", return_value=[dict(result, **changes)]
            ) as native:
                if changes:
                    with self.assertRaisesRegex(RuntimeError, "diagnostic contract"):
                        bench.collect_readback_diagnostics(Path("fixture"), "", [request], [comparison])
                else:
                    report = bench.collect_readback_diagnostics(Path("fixture"), "", [request], [comparison])
                    self.assertEqual(report["results"], [result])
                native.assert_called_once_with(Path("fixture"), "", probe_only=True)

    def test_probe_is_separate_and_after_every_cuda_case(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                events = []
                torch = MagicMock()
                torch.__version__ = "fixture"
                torch.cuda.get_device_name.return_value = "fixture"
                torch.get_num_threads.return_value = 1
                reference = torch.tensor.return_value.reshape.return_value.__matmul__.return_value
                ordered = MagicMock()
                ordered.__getitem__.return_value.flatten.return_value.tolist.return_value = []
                ordered.__getitem__.return_value.tolist.return_value = []
                reference.sort.return_value = (ordered, ordered)
                torch.empty.return_value.__getitem__.return_value.cpu.return_value.tolist.return_value = []
                timing = MagicMock()

                def timed(*args):
                    events.append("cuda")
                    return {"torch": {"samples_ms": [1.]}}

                timing.paired_timings.side_effect = timed
                timing.summarize.return_value = {"median_ms": 1.}

                def native(image, payload):
                    events.append("native")
                    return [dict(json.loads(line), status="passed", adapter={"name": "fixture"},
                                 values=[], indices=[], samples_ms={"fixture": [1.]})
                            for line in payload.splitlines()]

                def probe(*args):
                    self.assertEqual(events, ["native"] + ["cuda"] * 18)
                    events.append("probe")
                    return {"results": []}

                with tempfile.TemporaryDirectory() as directory, \
                        patch.dict(sys.modules, {"torch": torch}), \
                        patch.object(bench, "require_uncontended_gpu"), \
                        patch.object(bench.os, "link"), \
                        patch.object(bench.audit, "source_identity", return_value={}), \
                        patch.object(bench.audit, "file_identity", return_value={}), \
                        patch.object(bench.audit, "read_native_build_identity", return_value={}), \
                        patch.object(bench.audit, "validate_source_binding", return_value={"valid": True}), \
                        patch.object(bench.audit, "load_bench_module", return_value=timing), \
                        patch.object(bench, "run_native_pass", side_effect=native) as native_pass, \
                        patch.object(bench, "collect_readback_diagnostics", side_effect=probe) as diagnostics:
                    report = bench.run(Path(directory) / "fixture", enabled)
                    self.assertEqual(len(report["cases"]), 18)
                    native_pass.assert_called_once()
                    self.assertEqual(diagnostics.call_count, int(enabled))
                    self.assertEqual("readback_diagnostics" in report, enabled)
                    self.assertEqual(events, ["native"] + ["cuda"] * 18 + (["probe"] if enabled else []))


if __name__ == "__main__":
    unittest.main()
