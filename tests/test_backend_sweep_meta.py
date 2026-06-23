import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import backend_sweep_meta  # noqa: E402
import audit_learning_backend_backlog  # noqa: E402
import compare_char_lm_runs  # noqa: E402
import run_gnn_band_trace_sweep  # noqa: E402
import run_gnn_threshold_grid  # noqa: E402
import run_char_lm_sweep  # noqa: E402
import run_lstm_scan_profile_grid  # noqa: E402
import summarize_char_lm_compare  # noqa: E402


class BackendSweepMetaTests(unittest.TestCase):
    def test_classify_failure_uses_compile_error_not_logged_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "process.log"
            log_path.write_text(
                "\n".join(
                    [
                        "$ cargo run -p st-nn --features wgpu --example model",
                        "   Compiling st-tensor v0.1.0",
                        "error[E0308]: mismatched types",
                        "error: could not compile `st-tensor` (lib)",
                    ]
                ),
                encoding="utf-8",
            )

            kind, detail = backend_sweep_meta.classify_failure(101, log_path)

        self.assertEqual(kind, "compile")
        self.assertEqual(detail, "error[E0308]: mismatched types")

    def test_preflight_skipped_run_record_writes_failure_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "backend-wgpu__seed-1"
            log_path = run_dir / "process.log"
            record = backend_sweep_meta.preflight_skipped_run_record(
                schema="st.test.sweep_failure.v1",
                backend="wgpu",
                seed=1,
                run_dir=run_dir,
                log_path=log_path,
                command=["cargo", "run"],
                preflight_failure={
                    "failure_kind": "signal",
                    "failure_detail": "signal:6",
                    "returncode": -6,
                    "log_path": str(root / "_preflight" / "process.log"),
                },
            )
            failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
            log_text = log_path.read_text(encoding="utf-8")

        self.assertTrue(record["failed"])
        self.assertTrue(record["skipped"])
        self.assertEqual(record["failure_kind"], "preflight_signal")
        self.assertEqual(record["returncode"], -6)
        self.assertEqual(failure["schema"], "st.test.sweep_failure.v1")
        self.assertEqual(failure["failure_detail"], "signal:6")
        self.assertIn("skipped after WGPU sweep preflight failure", log_text)

    def test_gnn_band_trace_sweep_grid_dry_run_writes_shape_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(root),
                        "--backends",
                        "cpu",
                        "--seeds",
                        "1,2",
                        "--epoch-values",
                        "1,2",
                        "--batch-values",
                        "1,2",
                        "--node-values",
                        "3",
                        "--feature-values",
                        "2",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["schema"], "st.gnn.band_trace_sweep.v2")
        self.assertEqual(manifest["grid"]["epochs"], [1, 2])
        self.assertEqual(manifest["grid"]["batch"], [1, 2])
        self.assertEqual(len(manifest["runs"]), 8)
        target = next(
            run
            for run in manifest["runs"]
            if run["epochs"] == 2 and run["batch"] == 2 and run["seed"] == 2
        )
        self.assertEqual(target["input_rows"], 6)
        self.assertIn("epochs-2__train-4__val-2__batch-2__nodes-3", target["name"])
        self.assertEqual(target["command"][target["command"].index("--epochs") + 1], "2")
        self.assertEqual(target["command"][target["command"].index("--batch") + 1], "2")
        self.assertEqual(target["command"][target["command"].index("--nodes") + 1], "3")

    def test_gnn_band_trace_sweep_roundtable_grid_dry_run_writes_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(root),
                        "--backends",
                        "cpu",
                        "--seeds",
                        "1",
                        "--lr-values",
                        "0.03,0.05",
                        "--top-k-values",
                        "1,2",
                        "--mid-k-values",
                        "1",
                        "--bottom-k-values",
                        "1",
                        "--here-tolerance-values",
                        "0.00001,0.001",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["grid"]["learning_rates"], [0.03, 0.05])
        self.assertEqual(manifest["grid"]["top_k"], [1, 2])
        self.assertEqual(manifest["grid"]["here_tolerance"], [0.00001, 0.001])
        self.assertEqual(len(manifest["runs"]), 8)
        target = next(
            run
            for run in manifest["runs"]
            if run["lr"] == 0.05 and run["top_k"] == 2 and run["here_tolerance"] == 0.001
        )
        self.assertIn("lr-0p05__top-2__mid-1__bottom-1__tol-0p001", target["name"])
        self.assertEqual(target["command"][target["command"].index("--lr") + 1], "0.05")
        self.assertEqual(target["command"][target["command"].index("--top-k") + 1], "2")
        self.assertEqual(
            target["command"][target["command"].index("--here-tolerance") + 1],
            "0.001",
        )

    def test_gnn_band_trace_compare_surfaces_validation_readout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = run_gnn_band_trace_sweep.write_compare(
                root,
                [
                    {
                        "backend": "cpu",
                        "seed": 9,
                        "epochs": 1,
                        "train_graphs": 4,
                        "validation_graphs": 3,
                        "batch": 2,
                        "nodes": 4,
                        "features": 2,
                        "lr": 0.03,
                        "top_k": 1,
                        "mid_k": 1,
                        "bottom_k": 1,
                        "here_tolerance": 0.00001,
                        "input_rows": 8,
                        "run_dir": str(root / "run"),
                        "log_path": str(root / "run" / "process.log"),
                        "returncode": 0,
                        "failure_kind": None,
                        "failure_detail": None,
                        "skipped": False,
                        "failed": False,
                        "gnn_summary": {
                            "trainer": {"best_score": 0.12},
                            "readout": {
                                "trace": {"graph_count": 2, "total_rows": 8},
                                "error": {
                                    "mean_squared_error": 0.20,
                                    "normalized_mean_squared_error": 1.25,
                                },
                            },
                            "validation_readout": {
                                "graph_count": 3,
                                "total_rows": 12,
                                "mean_squared_error": 0.15,
                                "normalized_mean_squared_error": 0.75,
                            },
                            "bands": {},
                        },
                        "trainer_summary": {"metrics": {}},
                        "run_meta": {},
                    }
                ],
            )

            report = path.read_text(encoding="utf-8")

        self.assertIn("validation_readout_mse", report)
        self.assertIn("validation_readout_nmse", report)
        self.assertIn("avg_validation_readout_mse", report)
        self.assertIn("avg_validation_readout_nmse", report)
        self.assertIn("0.150000", report)
        self.assertIn("0.750000", report)
        self.assertIn(
            "| cpu | 1 | 4 | 3 | 2 | 4 | 2 | 0.030000 | 1 | 1 | 1 | 0.000010 | 8 | 1 | 9 |",
            report,
        )

    def test_gnn_band_trace_compare_recommends_roundtable_candidates(self) -> None:
        def summary(
            *,
            top_k: int = 1,
            mid_k: int = 1,
            validation_mse: float,
            here_delta: float,
        ) -> dict:
            return {
                "backend": "cpu",
                "seed": 9,
                "epochs": 1,
                "train_graphs": 4,
                "validation_graphs": 3,
                "batch": 2,
                "nodes": 4,
                "features": 2,
                "lr": 0.03,
                "top_k": top_k,
                "mid_k": mid_k,
                "bottom_k": 1,
                "here_tolerance": 0.00001,
                "input_rows": 8,
                "run_dir": "run",
                "log_path": "run/process.log",
                "returncode": 0,
                "failure_kind": None,
                "failure_detail": None,
                "skipped": False,
                "failed": False,
                "gnn_summary": {
                    "trainer": {"best_score": validation_mse},
                    "readout": {
                        "trace": {"graph_count": 2, "total_rows": 8},
                        "error": {"mean_squared_error": validation_mse + 0.05},
                    },
                    "validation_readout": {
                        "graph_count": 3,
                        "total_rows": 12,
                        "mean_squared_error": validation_mse,
                    },
                    "bands": {
                        "above": {"band_pass_scales": {"max_abs_delta": 0.20 + top_k * 0.01}},
                        "here": {"band_pass_scales": {"max_abs_delta": here_delta}},
                        "beneath": {
                            "band_pass_scales": {
                                "max_abs_delta": 0.11 - top_k * 0.01 - (mid_k - 1) * 0.01
                            }
                        },
                    },
                },
                "trainer_summary": {"metrics": {}},
                "run_meta": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = run_gnn_band_trace_sweep.write_compare(
                root,
                [
                    summary(top_k=1, mid_k=1, validation_mse=0.16, here_delta=0.22),
                    summary(top_k=2, mid_k=1, validation_mse=0.15, here_delta=0.17),
                    summary(top_k=1, mid_k=2, validation_mse=0.155, here_delta=0.19),
                ],
            )

            report = path.read_text(encoding="utf-8")

        self.assertIn("## Top Validation Candidates", report)
        self.assertIn("## Roundtable Axis Deltas", report)
        self.assertIn("| 1 | cpu | 1 | 4 | 3 | 2 | 4 | 2 | 0.030000 | 2 |", report)
        self.assertIn(
            "| top_k | cpu | 1 | 4 | 3 | 2 | 4 | 2 | 0.030000 | 1 | 2 | 2 | 1 | 1 | 0.000010 | 0.150000 | - | -0.010000 | - | -0.0500 | 0.0100 | -0.0100 |",
            report,
        )
        self.assertIn(
            "| mid_k | cpu | 1 | 4 | 3 | 2 | 4 | 2 | 0.030000 | 1 | 2 | 1 | 2 | 1 | 0.000010 | 0.155000 | - | -0.005000 | - | -0.0300 | 0.0000 | -0.0100 |",
            report,
        )

    def test_gnn_band_trace_manifest_records_compare_summary(self) -> None:
        def summary(*, top_k: int, validation_mse: float, here_delta: float) -> dict:
            return {
                "name": f"backend-cpu__top-{top_k}__seed-9",
                "backend": "cpu",
                "seed": 9,
                "epochs": 1,
                "train_graphs": 4,
                "validation_graphs": 3,
                "batch": 2,
                "nodes": 4,
                "features": 2,
                "lr": 0.03,
                "top_k": top_k,
                "mid_k": 1,
                "bottom_k": 1,
                "here_tolerance": 0.00001,
                "input_rows": 8,
                "run_dir": "run",
                "log_path": "run/process.log",
                "returncode": 0,
                "failure_kind": None,
                "failure_detail": None,
                "skipped": False,
                "failed": False,
                "command": ["cargo", "run"],
                "gnn_summary": {
                    "trainer": {"best_score": validation_mse},
                    "readout": {
                        "trace": {"graph_count": 2, "total_rows": 8},
                        "error": {"mean_squared_error": validation_mse + 0.05},
                    },
                    "validation_readout": {
                        "graph_count": 3,
                        "total_rows": 12,
                        "mean_squared_error": validation_mse,
                    },
                    "bands": {
                        "above": {"band_pass_scales": {"max_abs_delta": 0.2}},
                        "here": {"band_pass_scales": {"max_abs_delta": here_delta}},
                        "beneath": {"band_pass_scales": {"max_abs_delta": 0.1}},
                    },
                },
                "trainer_summary": {"metrics": {}},
                "run_meta": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = run_gnn_band_trace_sweep.parse_args(
                [
                    "--run-root",
                    str(root),
                    "--backends",
                    "cpu",
                    "--seeds",
                    "9",
                    "--top-k-values",
                    "1,2",
                ]
            )
            grid = run_gnn_band_trace_sweep.grid_manifest(args)
            summaries = [
                summary(top_k=1, validation_mse=0.16, here_delta=0.22),
                summary(top_k=2, validation_mse=0.15, here_delta=0.17),
            ]
            manifest = run_gnn_band_trace_sweep.sweep_manifest(
                args,
                grid,
                summaries,
                {},
                summaries=summaries,
            )

        comparison = manifest["comparison"]
        self.assertEqual(comparison["schema"], "st.gnn.band_trace_compare.v1")
        self.assertEqual(comparison["successful_runs"], 2)
        self.assertEqual(comparison["group_count"], 2)
        self.assertEqual(comparison["top_validation_candidates"][0]["top_k"], 2)
        self.assertEqual(comparison["top_validation_candidates"][0]["seeds"], [9])
        top_k_delta = next(
            row for row in comparison["roundtable_axis_deltas"] if row["axis"] == "top_k"
        )
        self.assertEqual(top_k_delta["baseline_value"], 1)
        self.assertEqual(top_k_delta["value"], 2)
        self.assertAlmostEqual(top_k_delta["validation_mse_delta"], -0.01)
        self.assertAlmostEqual(top_k_delta["here_delta_delta"], -0.05)

    def test_gnn_band_trace_compare_surfaces_seed_stability_candidates(self) -> None:
        def summary(*, seed: int, top_k: int, validation_mse: float) -> dict:
            return {
                "name": f"backend-cpu__top-{top_k}__seed-{seed}",
                "backend": "cpu",
                "seed": seed,
                "epochs": 1,
                "train_graphs": 4,
                "validation_graphs": 3,
                "batch": 2,
                "nodes": 4,
                "features": 2,
                "lr": 0.03,
                "top_k": top_k,
                "mid_k": 1,
                "bottom_k": 1,
                "here_tolerance": 0.00001,
                "input_rows": 8,
                "run_dir": "run",
                "log_path": "run/process.log",
                "returncode": 0,
                "failure_kind": None,
                "failure_detail": None,
                "skipped": False,
                "failed": False,
                "command": ["cargo", "run"],
                "gnn_summary": {
                    "trainer": {"best_score": validation_mse},
                    "readout": {
                        "trace": {"graph_count": 2, "total_rows": 8},
                        "error": {"mean_squared_error": validation_mse + 0.05},
                    },
                    "validation_readout": {
                        "graph_count": 3,
                        "total_rows": 12,
                        "mean_squared_error": validation_mse,
                    },
                    "bands": {
                        "above": {"band_pass_scales": {"max_abs_delta": 0.2}},
                        "here": {"band_pass_scales": {"max_abs_delta": 0.1}},
                        "beneath": {"band_pass_scales": {"max_abs_delta": 0.1}},
                    },
                },
                "trainer_summary": {"metrics": {}},
                "run_meta": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summaries = [
                summary(seed=1, top_k=1, validation_mse=0.10),
                summary(seed=2, top_k=1, validation_mse=0.30),
                summary(seed=1, top_k=2, validation_mse=0.21),
                summary(seed=2, top_k=2, validation_mse=0.21),
            ]
            compare_path = run_gnn_band_trace_sweep.write_compare(root, summaries)
            comparison = run_gnn_band_trace_sweep.comparison_summary(
                summaries,
                run_root=root,
            )
            report = compare_path.read_text(encoding="utf-8")

        top_candidate = comparison["top_validation_candidates"][0]
        self.assertEqual(top_candidate["top_k"], 1)
        self.assertAlmostEqual(top_candidate["avg_validation_readout_mse"], 0.20)
        self.assertAlmostEqual(top_candidate["validation_readout_mse_stddev"], 0.10)
        self.assertAlmostEqual(top_candidate["validation_readout_mse_spread"], 0.20)
        self.assertEqual(top_candidate["validation_stability_status"], "volatile")

        stable_candidate = comparison["stable_validation_candidates"][0]
        self.assertEqual(stable_candidate["stability_rank"], 1)
        self.assertEqual(stable_candidate["validation_rank"], 2)
        self.assertEqual(stable_candidate["top_k"], 2)
        self.assertAlmostEqual(stable_candidate["validation_stability_score"], 0.21)
        self.assertEqual(stable_candidate["validation_stability_status"], "multi_seed_stable")
        self.assertIn("## Stable Validation Candidates", report)
        self.assertIn("validation_mse_stddev", report)
        self.assertIn("multi_seed_stable", report)
        self.assertIn("volatile", report)

    def test_gnn_band_trace_follow_up_from_replays_best_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 3,
                                    "train_graphs": 8,
                                    "validation_graphs": 5,
                                    "batch": 2,
                                    "nodes": 6,
                                    "features": 3,
                                    "lr": 0.04,
                                    "top_k": 2,
                                    "mid_k": 1,
                                    "bottom_k": 2,
                                    "here_tolerance": 0.001,
                                    "seeds": [9, 10],
                                    "avg_validation_readout_mse": 0.08,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source),
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["grid"]["epochs"], [3])
        self.assertEqual(manifest["grid"]["train_graphs"], [8])
        self.assertEqual(manifest["grid"]["validation_graphs"], [5])
        self.assertEqual(manifest["grid"]["batch"], [2])
        self.assertEqual(manifest["grid"]["nodes"], [6])
        self.assertEqual(manifest["grid"]["features"], [3])
        self.assertEqual(manifest["grid"]["learning_rates"], [0.04])
        self.assertEqual(manifest["grid"]["top_k"], [2])
        self.assertEqual(manifest["grid"]["bottom_k"], [2])
        self.assertEqual(manifest["grid"]["here_tolerance"], [0.001])
        self.assertEqual(manifest["seeds"], [9, 10])
        self.assertEqual(len(manifest["runs"]), 2)
        self.assertEqual(manifest["config"]["follow_up"]["rank"], 1)
        self.assertEqual(manifest["config"]["follow_up"]["candidate"]["top_k"], 2)
        lineage = manifest["config"]["follow_up"]["lineage"]
        self.assertEqual(lineage["schema"], "st.gnn.band_trace_follow_up_lineage.v1")
        self.assertEqual(lineage["generation"], 1)
        self.assertEqual(lineage["parent_generation"], 0)
        self.assertEqual(lineage["parent_sweep_path"], str(source / "sweep.json"))
        self.assertEqual(lineage["parent_run_root"], str(source))
        self.assertEqual(lineage["source_mode"], "auto")
        self.assertEqual(lineage["candidate_source"], "top-candidate")
        target = manifest["runs"][0]
        self.assertIn("epochs-3__train-8__val-5__batch-2__nodes-6", target["name"])
        self.assertEqual(target["command"][target["command"].index("--lr") + 1], "0.04")
        self.assertEqual(target["command"][target["command"].index("--top-k") + 1], "2")
        self.assertEqual(
            target["command"][target["command"].index("--here-tolerance") + 1],
            "0.001",
        )

    def test_gnn_band_trace_follow_up_keeps_explicit_axis_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 2,
                                    "train_graphs": 4,
                                    "validation_graphs": 3,
                                    "batch": 2,
                                    "nodes": 4,
                                    "features": 2,
                                    "lr": 0.03,
                                    "top_k": 2,
                                    "mid_k": 1,
                                    "bottom_k": 1,
                                    "here_tolerance": 0.00001,
                                    "seeds": [9],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source / "sweep.json"),
                        "--seeds",
                        "11,12",
                        "--top-k-values",
                        "1,2",
                        "--epoch-values",
                        "2,4",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["seeds"], [11, 12])
        self.assertEqual(manifest["grid"]["epochs"], [2, 4])
        self.assertEqual(manifest["grid"]["top_k"], [1, 2])
        self.assertEqual(manifest["grid"]["learning_rates"], [0.03])
        self.assertEqual(len(manifest["runs"]), 8)
        self.assertTrue(any(run["top_k"] == 1 for run in manifest["runs"]))
        self.assertTrue(any(run["epochs"] == 4 for run in manifest["runs"]))

    def test_gnn_band_trace_follow_up_auto_uses_promotion_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "follow_up_promotion": {
                                "schema": "st.gnn.band_trace_follow_up_promotion.v1",
                                "action": "keep_source",
                                "selected_origin": "source",
                                "selected_candidate": {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 1,
                                    "train_graphs": 4,
                                    "validation_graphs": 4,
                                    "batch": 2,
                                    "nodes": 4,
                                    "features": 2,
                                    "lr": 0.04,
                                    "top_k": 1,
                                    "mid_k": 1,
                                    "bottom_k": 1,
                                    "here_tolerance": 0.00001,
                                    "seeds": [7],
                                },
                            },
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 2,
                                    "train_graphs": 5,
                                    "validation_graphs": 5,
                                    "batch": 2,
                                    "nodes": 6,
                                    "features": 3,
                                    "lr": 0.08,
                                    "top_k": 3,
                                    "mid_k": 2,
                                    "bottom_k": 2,
                                    "here_tolerance": 0.001,
                                    "seeds": [9],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source),
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["config"]["follow_up"]["candidate_source"], "promotion")
        self.assertEqual(manifest["grid"]["epochs"], [1])
        self.assertEqual(manifest["grid"]["nodes"], [4])
        self.assertEqual(manifest["grid"]["learning_rates"], [0.04])
        self.assertEqual(manifest["grid"]["top_k"], [1])
        self.assertEqual(manifest["seeds"], [7])

    def test_gnn_band_trace_follow_up_can_force_top_candidate_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "follow_up_promotion": {
                                "schema": "st.gnn.band_trace_follow_up_promotion.v1",
                                "action": "keep_source",
                                "selected_origin": "source",
                                "selected_candidate": {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 1,
                                    "train_graphs": 4,
                                    "validation_graphs": 4,
                                    "batch": 2,
                                    "nodes": 4,
                                    "features": 2,
                                    "lr": 0.04,
                                    "top_k": 1,
                                    "mid_k": 1,
                                    "bottom_k": 1,
                                    "here_tolerance": 0.00001,
                                    "seeds": [7],
                                },
                            },
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 2,
                                    "train_graphs": 5,
                                    "validation_graphs": 5,
                                    "batch": 2,
                                    "nodes": 6,
                                    "features": 3,
                                    "lr": 0.08,
                                    "top_k": 3,
                                    "mid_k": 2,
                                    "bottom_k": 2,
                                    "here_tolerance": 0.001,
                                    "seeds": [9],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source),
                        "--follow-up-source",
                        "top-candidate",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["config"]["follow_up"]["candidate_source"], "top-candidate")
        self.assertEqual(manifest["grid"]["epochs"], [2])
        self.assertEqual(manifest["grid"]["nodes"], [6])
        self.assertEqual(manifest["grid"]["learning_rates"], [0.08])
        self.assertEqual(manifest["grid"]["top_k"], [3])
        self.assertEqual(manifest["seeds"], [9])

    def test_gnn_band_trace_follow_up_lineage_extends_parent_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "run_root": str(source),
                        "config": {
                            "follow_up": {
                                "lineage": {
                                    "schema": "st.gnn.band_trace_follow_up_lineage.v1",
                                    "generation": 2,
                                }
                            }
                        },
                        "comparison": {
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 1,
                                    "train_graphs": 4,
                                    "validation_graphs": 4,
                                    "batch": 2,
                                    "nodes": 4,
                                    "features": 2,
                                    "lr": 0.04,
                                    "top_k": 2,
                                    "mid_k": 1,
                                    "bottom_k": 1,
                                    "here_tolerance": 0.00001,
                                    "seeds": [7],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source),
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        lineage = manifest["config"]["follow_up"]["lineage"]
        self.assertEqual(lineage["generation"], 3)
        self.assertEqual(lineage["parent_generation"], 2)
        self.assertEqual(lineage["parent_sweep_path"], str(source / "sweep.json"))
        self.assertEqual(lineage["parent_run_root"], str(source))
        self.assertEqual(lineage["candidate_source"], "top-candidate")
        self.assertEqual(manifest["config"]["follow_up"]["source_mode"], "auto")

    def test_gnn_band_trace_follow_up_neighborhood_expands_selected_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 1,
                                    "train_graphs": 4,
                                    "validation_graphs": 4,
                                    "batch": 2,
                                    "nodes": 4,
                                    "features": 2,
                                    "lr": 0.04,
                                    "top_k": 2,
                                    "mid_k": 1,
                                    "bottom_k": 2,
                                    "here_tolerance": 0.001,
                                    "seeds": [9],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source),
                        "--follow-up-neighborhood",
                        "--follow-up-neighborhood-axes",
                        "lr,top_k,bottom_k",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["grid"]["learning_rates"], [0.03, 0.04, 0.05])
        self.assertEqual(manifest["grid"]["top_k"], [1, 2, 3])
        self.assertEqual(manifest["grid"]["mid_k"], [1])
        self.assertEqual(manifest["grid"]["bottom_k"], [1, 2, 3])
        self.assertEqual(manifest["grid"]["here_tolerance"], [0.001])
        self.assertEqual(len(manifest["runs"]), 27)
        neighborhood = manifest["config"]["follow_up"]["neighborhood"]
        self.assertTrue(neighborhood["enabled"])
        self.assertEqual(neighborhood["axes"], ["lr", "top_k", "bottom_k"])
        self.assertEqual(neighborhood["expanded"]["bottom_k"], [1, 2, 3])
        self.assertTrue(any(run["top_k"] == 3 for run in manifest["runs"]))
        self.assertTrue(any(run["lr"] == 0.03 for run in manifest["runs"]))

    def test_gnn_band_trace_follow_up_neighborhood_keeps_explicit_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            follow_up = root / "follow-up"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "top_validation_candidates": [
                                {
                                    "rank": 1,
                                    "backend": "cpu",
                                    "epochs": 1,
                                    "train_graphs": 4,
                                    "validation_graphs": 4,
                                    "batch": 2,
                                    "nodes": 4,
                                    "features": 2,
                                    "lr": 0.04,
                                    "top_k": 2,
                                    "mid_k": 1,
                                    "bottom_k": 1,
                                    "here_tolerance": 0.001,
                                    "seeds": [9],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_band_trace_sweep.main(
                    [
                        "--run-root",
                        str(follow_up),
                        "--follow-up-from",
                        str(source / "sweep.json"),
                        "--follow-up-neighborhood",
                        "--follow-up-neighborhood-axes",
                        "lr,top_k",
                        "--lr-values",
                        "0.02,0.03",
                        "--top-k-values",
                        "4",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((follow_up / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["grid"]["learning_rates"], [0.02, 0.03])
        self.assertEqual(manifest["grid"]["top_k"], [4])
        self.assertEqual(len(manifest["runs"]), 2)
        neighborhood = manifest["config"]["follow_up"]["neighborhood"]
        self.assertEqual(neighborhood["expanded"], {})

    def test_gnn_band_trace_follow_up_result_compares_to_source_candidate(self) -> None:
        def summary(*, top_k: int, validation_mse: float, validation_nmse: float) -> dict:
            return {
                "name": f"backend-cpu__top-{top_k}__seed-9",
                "backend": "cpu",
                "seed": 9,
                "epochs": 1,
                "train_graphs": 4,
                "validation_graphs": 3,
                "batch": 2,
                "nodes": 4,
                "features": 2,
                "lr": 0.03,
                "top_k": top_k,
                "mid_k": 1,
                "bottom_k": 1,
                "here_tolerance": 0.00001,
                "input_rows": 8,
                "run_dir": "run",
                "log_path": "run/process.log",
                "returncode": 0,
                "failure_kind": None,
                "failure_detail": None,
                "skipped": False,
                "failed": False,
                "command": ["cargo", "run"],
                "gnn_summary": {
                    "trainer": {"best_score": validation_mse},
                    "readout": {
                        "trace": {"graph_count": 2, "total_rows": 8},
                        "error": {
                            "mean_squared_error": validation_mse + 0.05,
                            "normalized_mean_squared_error": validation_nmse + 0.05,
                        },
                    },
                    "validation_readout": {
                        "graph_count": 3,
                        "total_rows": 12,
                        "mean_squared_error": validation_mse,
                        "normalized_mean_squared_error": validation_nmse,
                    },
                    "bands": {},
                },
                "trainer_summary": {"metrics": {}},
                "run_meta": {},
            }

        source_candidate = {
            "rank": 1,
            "backend": "cpu",
            "epochs": 1,
            "train_graphs": 4,
            "validation_graphs": 3,
            "batch": 2,
            "nodes": 4,
            "features": 2,
            "lr": 0.03,
            "top_k": 1,
            "mid_k": 1,
            "bottom_k": 1,
            "here_tolerance": 0.00001,
            "avg_validation_readout_mse": 0.16,
            "avg_validation_readout_nmse": 0.80,
            "avg_cpu_debt_ops": 10.0,
            "avg_step_ms_last": 5.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "sweep.json").write_text(
                json.dumps(
                    {
                        "run_root": str(source),
                        "config": {
                            "follow_up": {
                                "candidate_source": "promotion",
                                "source_mode": "auto",
                                "lineage": {
                                    "schema": "st.gnn.band_trace_follow_up_lineage.v1",
                                    "generation": 2,
                                    "parent_generation": 1,
                                },
                            }
                        },
                        "comparison": {
                            "follow_up_result": {
                                "schema": "st.gnn.band_trace_follow_up_result.v1",
                                "verdict": "matched",
                            },
                            "follow_up_promotion": {
                                "schema": "st.gnn.band_trace_follow_up_promotion.v1",
                                "action": "keep_source",
                                "selected_origin": "source",
                                "selected_schedule": "top=1 mid=1 bottom=1 here_tol=0.000010",
                                "selected_avg_validation_readout_mse": 0.16,
                                "selected_avg_validation_readout_nmse": 0.80,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = run_gnn_band_trace_sweep.parse_args(
                [
                    "--run-root",
                    str(root),
                    "--backends",
                    "cpu",
                    "--seeds",
                    "9",
                ]
            )
            args.follow_up_from = source / "sweep.json"
            args.follow_up_rank = 1
            args.follow_up_candidate = source_candidate
            args.follow_up_candidate_source = "top-candidate"
            args.follow_up_parent_sweep_path = source / "sweep.json"
            args.follow_up_parent_run_root = str(source)
            args.follow_up_parent_generation = 2
            follow_up = run_gnn_band_trace_sweep.follow_up_manifest(args)
            grid = run_gnn_band_trace_sweep.grid_manifest(args)
            summaries = [
                summary(top_k=1, validation_mse=0.17, validation_nmse=0.82),
                summary(top_k=2, validation_mse=0.15, validation_nmse=0.70),
            ]
            compare_path = run_gnn_band_trace_sweep.write_compare(
                root,
                summaries,
                follow_up=follow_up,
                follow_up_candidate=source_candidate,
                follow_up_fail_on_verdict=["regressed", "unknown"],
            )
            manifest = run_gnn_band_trace_sweep.sweep_manifest(
                args,
                grid,
                summaries,
                {},
                summaries=summaries,
            )
            report = compare_path.read_text(encoding="utf-8")
            script = root / "next_follow_up_command.sh"
            script_exists = script.exists()
            script_mode = script.stat().st_mode if script_exists else 0
            script_text = script.read_text(encoding="utf-8") if script_exists else ""

        result = manifest["comparison"]["follow_up_result"]
        self.assertEqual(result["schema"], "st.gnn.band_trace_follow_up_result.v1")
        self.assertEqual(result["verdict"], "improved")
        self.assertEqual(result["best_candidate"]["top_k"], 2)
        self.assertEqual(result["source_replay_candidate"]["top_k"], 1)
        self.assertEqual(result["source_replay_rank"], 2)
        self.assertAlmostEqual(result["validation_mse_delta"], -0.01)
        self.assertAlmostEqual(result["source_replay_validation_mse_delta"], 0.01)
        self.assertAlmostEqual(result["best_vs_source_replay_validation_mse_delta"], -0.02)
        self.assertAlmostEqual(result["validation_nmse_delta"], -0.10)
        self.assertAlmostEqual(result["source_replay_validation_nmse_delta"], 0.02)
        self.assertAlmostEqual(result["best_vs_source_replay_validation_nmse_delta"], -0.12)
        promotion = manifest["comparison"]["follow_up_promotion"]
        self.assertEqual(promotion["schema"], "st.gnn.band_trace_follow_up_promotion.v1")
        self.assertEqual(promotion["action"], "promote_best")
        self.assertEqual(promotion["selected_origin"], "best")
        self.assertAlmostEqual(promotion["selected_avg_validation_readout_nmse"], 0.70)
        self.assertAlmostEqual(promotion["validation_nmse_delta"], -0.10)
        self.assertEqual(promotion["selected_candidate"]["top_k"], 2)
        next_command = manifest["comparison"]["follow_up_next_command"]
        self.assertEqual(
            next_command["schema"],
            "st.gnn.band_trace_follow_up_next_command.v1",
        )
        self.assertEqual(next_command["promotion_action"], "promote_best")
        self.assertEqual(next_command["selected_origin"], "best")
        self.assertIn("--follow-up-from", next_command["command"])
        self.assertIn(str(root), next_command["command"])
        chain = manifest["comparison"]["follow_up_chain"]
        self.assertEqual(chain["schema"], "st.gnn.band_trace_follow_up_chain.v1")
        self.assertEqual(chain["generation"], 3)
        self.assertEqual(chain["parent_generation"], 2)
        self.assertEqual(chain["candidate_source"], "top-candidate")
        self.assertEqual(chain["chain_depth"], 2)
        self.assertEqual(chain["ancestor_count"], 1)
        self.assertEqual(chain["ancestors"][0]["generation"], 2)
        self.assertEqual(chain["ancestors"][0]["promotion_action"], "keep_source")
        self.assertAlmostEqual(chain["ancestors"][0]["selected_avg_validation_readout_nmse"], 0.80)
        self.assertEqual(chain["ancestors"][0]["verdict"], "matched")
        guidance = manifest["comparison"]["follow_up_chain_guidance"]
        self.assertEqual(
            guidance["schema"],
            "st.gnn.band_trace_follow_up_chain_guidance.v1",
        )
        self.assertEqual(guidance["action"], "repeat_with_fresh_seeds")
        self.assertEqual(guidance["current_verdict"], "improved")
        self.assertEqual(guidance["recent_verdicts"], ["improved", "matched"])
        self.assertEqual(guidance["improved_streak"], 1)
        self.assertEqual(guidance["non_improving_streak"], 0)
        self.assertEqual(guidance["candidate_validation_stability_status"], "single_seed_probe")
        guided_command = manifest["comparison"]["follow_up_guided_next_command"]
        self.assertEqual(
            guided_command["schema"],
            "st.gnn.band_trace_follow_up_guided_next_command.v1",
        )
        self.assertEqual(guided_command["guidance_action"], "repeat_with_fresh_seeds")
        self.assertIn("--follow-up-from", guided_command["command"])
        self.assertIn(str(root), guided_command["command"])
        self.assertIn("--follow-up-source", guided_command["command"])
        self.assertIn("auto", guided_command["command"])
        self.assertEqual(guided_command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])
        self.assertTrue(guided_command["requires_user_input"])
        self.assertEqual(guided_command["script_path"], str(root / "next_follow_up_command.sh"))
        self.assertIn("NEXT_RUN_ROOT=...", guided_command["script_usage"])
        self.assertIn("NEW_SEEDS=...", guided_command["script_usage"])
        self.assertTrue(script_exists)
        self.assertTrue(script_mode & 0o111)
        self.assertIn("Required placeholders: NEXT_RUN_ROOT, NEW_SEEDS", script_text)
        self.assertIn("${NEXT_RUN_ROOT:?", script_text)
        self.assertIn("${NEW_SEEDS:?", script_text)
        self.assertIn("## Follow-Up Result", report)
        self.assertIn("| 1 | 2 | 1 | improved |", report)
        self.assertIn("-0.010000", report)
        self.assertIn("source_replay_validation_mse", report)
        self.assertIn("## Follow-Up Gate", report)
        self.assertIn("| improved | no | regressed,unknown |", report)
        self.assertIn("## Follow-Up Promotion", report)
        self.assertIn("| promote_best | improved | best |", report)
        self.assertIn("## Next Follow-Up Command", report)
        self.assertIn("--follow-up-from", report)
        self.assertIn("NEXT_RUN_ROOT", report)
        self.assertIn("## Follow-Up Chain", report)
        self.assertIn("| 3 | 2 | auto | top-candidate | 2 |", report)
        self.assertIn("## Follow-Up Ancestors", report)
        self.assertIn("| 2 |", report)
        self.assertIn("## Follow-Up Chain Guidance", report)
        self.assertIn("| repeat_with_fresh_seeds | improved | improved,matched |", report)
        self.assertIn("single_seed_probe", report)
        self.assertIn("## Guided Next Follow-Up Command", report)
        self.assertIn("repeat_with_fresh_seeds", report)
        self.assertIn("next_follow_up_command.sh", report)

    def test_gnn_band_trace_follow_up_chain_guidance_widens_repeated_regression(self) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "regressed",
                    "promotion_action": "keep_source",
                }
            ],
        }
        result = {"schema": "st.gnn.band_trace_follow_up_result.v1", "verdict": "regressed"}

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)

        self.assertEqual(guidance["action"], "widen_neighborhood")
        self.assertEqual(guidance["current_verdict"], "regressed")
        self.assertEqual(guidance["recent_verdicts"], ["regressed", "regressed"])
        self.assertEqual(guidance["regressed_streak"], 2)
        self.assertEqual(guidance["non_improving_streak"], 2)
        self.assertIn("--follow-up-neighborhood", guidance["suggested_flags"])
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/chain"),
            guidance,
        )
        self.assertEqual(command["guidance_action"], "widen_neighborhood")
        self.assertIn("--follow-up-neighborhood", command["command"])
        self.assertIn("target/tmp/chain", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT"])

    def test_gnn_band_trace_follow_up_guidance_rechecks_volatile_improvement(self) -> None:
        chain = {"schema": "st.gnn.band_trace_follow_up_chain.v1", "ancestors": []}
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "best_candidate": {
                "validation_stability_status": "volatile",
                "validation_stability_score": 0.42,
                "validation_readout_mse_stddev": 0.10,
                "validation_readout_mse_spread": 0.20,
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/volatile improved"),
            guidance,
        )

        self.assertEqual(guidance["action"], "repeat_with_fresh_seeds")
        self.assertEqual(guidance["current_verdict"], "improved")
        self.assertEqual(guidance["candidate_validation_stability_status"], "volatile")
        self.assertAlmostEqual(guidance["candidate_validation_stability_score"], 0.42)
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_follow_up_guidance_widens_repeated_volatile_improvement(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "improved",
                    "promotion_action": "promote_best",
                }
            ],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "best_candidate": {
                "validation_stability_status": "volatile",
                "validation_stability_score": 0.42,
                "validation_readout_mse_stddev": 0.10,
                "validation_readout_mse_spread": 0.20,
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/repeated volatile improved"),
            guidance,
        )

        self.assertEqual(guidance["action"], "widen_stability_search")
        self.assertEqual(guidance["recent_verdicts"], ["improved", "improved"])
        self.assertEqual(guidance["improved_streak"], 2)
        self.assertEqual(guidance["candidate_validation_stability_status"], "volatile")
        self.assertIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])
        self.assertIn("NEW_SEEDS=...", command["script_usage"])

    def test_gnn_band_trace_follow_up_guidance_increases_budget_after_widened_volatility(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "neighborhood_enabled": True,
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "improved",
                    "promotion_action": "promote_best",
                }
            ],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "best_candidate": {
                "epochs": 1,
                "train_graphs": 4,
                "validation_graphs": 3,
                "validation_stability_status": "volatile",
                "validation_stability_score": 0.42,
                "validation_readout_mse_stddev": 0.10,
                "validation_readout_mse_spread": 0.20,
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/widened volatile improved"),
            guidance,
        )

        self.assertEqual(guidance["action"], "increase_sample_budget")
        self.assertTrue(guidance["neighborhood_enabled"])
        self.assertEqual(guidance["candidate_validation_stability_status"], "volatile")
        self.assertIn("--epoch-values", command["command"])
        self.assertIn("1,2", command["command"])
        self.assertIn("--train-graph-values", command["command"])
        self.assertIn("4,8", command["command"])
        self.assertIn("--validation-graph-values", command["command"])
        self.assertIn("3,6", command["command"])
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_follow_up_guidance_continues_stable_improvement(self) -> None:
        chain = {"schema": "st.gnn.band_trace_follow_up_chain.v1", "ancestors": []}
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "best_candidate": {
                "validation_stability_status": "multi_seed_stable",
                "validation_stability_score": 0.12,
                "validation_readout_mse_stddev": 0.001,
                "validation_readout_mse_spread": 0.002,
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/stable improved"),
            guidance,
        )

        self.assertEqual(guidance["action"], "continue_promotion")
        self.assertEqual(guidance["candidate_validation_stability_status"], "multi_seed_stable")
        self.assertNotIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT"])

    def test_gnn_band_trace_follow_up_guidance_explores_stable_repeated_improvement(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "promotion_action": "promote_best",
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "improved",
                    "promotion_action": "keep_source_stability_guard",
                }
            ],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "best_candidate": {
                "validation_stability_status": "multi_seed_stable",
                "validation_stability_score": 0.070,
                "validation_readout_mse_stddev": 0.001,
                "validation_readout_mse_spread": 0.002,
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/stable repeated improved"),
            guidance,
        )

        self.assertEqual(guidance["action"], "explore_stable_neighborhood")
        self.assertEqual(guidance["recent_verdicts"], ["improved", "improved"])
        self.assertEqual(guidance["improved_streak"], 2)
        self.assertEqual(guidance["candidate_validation_stability_status"], "multi_seed_stable")
        self.assertIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])
        self.assertIn("NEW_SEEDS=...", command["script_usage"])

    def test_gnn_band_trace_follow_up_guidance_confirms_stable_neighborhood_win(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "promotion_action": "promote_best",
            "neighborhood_enabled": True,
            "ancestors": [],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "best_candidate": {
                "validation_stability_status": "multi_seed_stable",
                "validation_stability_score": 0.055,
                "validation_readout_mse_stddev": 0.001,
                "validation_readout_mse_spread": 0.002,
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/stable neighborhood win"),
            guidance,
        )

        self.assertEqual(guidance["action"], "confirm_stable_promotion")
        self.assertTrue(guidance["neighborhood_enabled"])
        self.assertEqual(guidance["candidate_validation_stability_status"], "multi_seed_stable")
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_guided_next_command_marks_fresh_seed_placeholder(self) -> None:
        chain = {"schema": "st.gnn.band_trace_follow_up_chain.v1", "ancestors": []}
        result = {"schema": "st.gnn.band_trace_follow_up_result.v1", "verdict": "matched"}
        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)

        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/matched path"),
            guidance,
        )

        self.assertEqual(guidance["action"], "repeat_with_fresh_seeds")
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])
        self.assertTrue(command["requires_user_input"])
        self.assertIn("NEXT_RUN_ROOT=...", command["script_usage"])
        self.assertIn("NEW_SEEDS=...", command["script_usage"])
        self.assertIn("'target/tmp/matched path/next_follow_up_command.sh'", command["script_usage"])

    def test_gnn_band_trace_fresh_seed_regression_drops_seed_placeholder(self) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "matched",
                    "promotion_action": "keep_source",
                },
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "improved",
                    "promotion_action": "promote_best",
                },
            ],
        }
        result = {"schema": "st.gnn.band_trace_follow_up_result.v1", "verdict": "regressed"}

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/fresh seed regressed"),
            guidance,
        )

        self.assertEqual(guidance["action"], "keep_source")
        self.assertEqual(guidance["recent_verdicts"], ["regressed", "matched", "improved"])
        self.assertEqual(guidance["regressed_streak"], 1)
        self.assertEqual(guidance["non_improving_streak"], 2)
        self.assertEqual(command["guidance_action"], "keep_source")
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT"])
        self.assertIn("--follow-up-source", command["command"])
        self.assertIn("auto", command["command"])
        self.assertNotIn("NEW_SEEDS", command["command"])
        self.assertNotIn("NEW_SEEDS=...", command["script_usage"])

    def test_gnn_band_trace_regressed_neighborhood_reviews_seed_shift(self) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "neighborhood_enabled": True,
            "ancestors": [],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_replay_validation_mse_delta": 0.0021,
            "best_vs_source_replay_validation_mse_delta": -0.00001,
            "source_replay_validation_nmse_delta": 0.0012,
            "best_vs_source_replay_validation_nmse_delta": -0.00002,
            "source_candidate": {
                "validation_stability_status": "multi_seed_stable",
                "validation_readout_nmse_stddev": 0.004,
                "validation_readout_nmse_spread": 0.010,
            },
            "best_candidate": {
                "validation_stability_status": "watch_spread",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/seed shift neighborhood"),
            guidance,
        )

        self.assertEqual(guidance["action"], "review_seed_shift_neighborhood")
        self.assertTrue(guidance["source_replay_seed_shift"])
        self.assertAlmostEqual(guidance["source_replay_validation_mse_delta"], 0.0021)
        self.assertAlmostEqual(
            guidance["best_vs_source_replay_validation_mse_delta"],
            -0.00001,
        )
        self.assertAlmostEqual(guidance["source_replay_validation_nmse_delta"], 0.0012)
        self.assertAlmostEqual(
            guidance["best_vs_source_replay_validation_nmse_delta"],
            -0.00002,
        )
        self.assertAlmostEqual(guidance["candidate_validation_nmse_stddev"], 0.004)
        self.assertAlmostEqual(guidance["candidate_validation_nmse_spread"], 0.010)
        self.assertIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_seed_shift_volatility_increases_validation_budget(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "neighborhood_enabled": True,
            "ancestors": [],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_replay_validation_mse_delta": 0.070,
            "best_vs_source_replay_validation_mse_delta": -0.0001,
            "source_replay_validation_stability_status": "volatile",
            "best_validation_stability_status": "volatile",
            "source_candidate": {
                "validation_graphs": 6,
                "validation_stability_status": "watch_spread",
            },
            "best_candidate": {
                "validation_stability_status": "volatile",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/seed shift budget"),
            guidance,
        )

        self.assertEqual(guidance["action"], "increase_seed_shift_validation_budget")
        self.assertTrue(guidance["source_replay_seed_shift"])
        self.assertTrue(guidance["seed_shift_needs_validation_budget"])
        self.assertEqual(guidance["source_replay_validation_stability_status"], "volatile")
        self.assertEqual(guidance["best_validation_stability_status"], "volatile")
        self.assertIn("--validation-graph-values", command["command"])
        self.assertIn("6,12", command["command"])
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_seed_shift_volatility_without_neighborhood_increases_validation_budget(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "neighborhood_enabled": False,
            "ancestors": [],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_replay_validation_mse_delta": 0.007,
            "best_vs_source_replay_validation_mse_delta": -0.0001,
            "source_replay_validation_stability_status": "volatile",
            "best_validation_stability_status": "volatile",
            "source_candidate": {
                "validation_graphs": 4,
                "validation_stability_status": "volatile",
            },
            "best_candidate": {
                "validation_stability_status": "volatile",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/seed shift non-neighborhood budget"),
            guidance,
        )

        self.assertEqual(guidance["action"], "increase_seed_shift_validation_budget")
        self.assertTrue(guidance["source_replay_seed_shift"])
        self.assertTrue(guidance["seed_shift_needs_validation_budget"])
        self.assertIn("--validation-graph-values", command["command"])
        self.assertIn("4,8", command["command"])
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_target_scale_shift_reviews_validation_budget(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "neighborhood_enabled": False,
            "ancestors": [],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_replay_validation_mse_delta": 0.025,
            "best_vs_source_replay_validation_mse_delta": -0.00005,
            "source_replay_validation_nmse_delta": -0.017,
            "best_vs_source_replay_validation_nmse_delta": -0.0007,
            "source_candidate": {
                "validation_graphs": 6,
                "validation_stability_status": "watch_spread",
            },
            "best_candidate": {
                "validation_stability_status": "watch_spread",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/target scale shift"),
            guidance,
        )

        self.assertEqual(guidance["action"], "review_target_scale_shift")
        self.assertTrue(guidance["source_replay_seed_shift"])
        self.assertTrue(guidance["target_scale_seed_shift"])
        self.assertFalse(guidance["seed_shift_needs_validation_budget"])
        self.assertIn("--validation-graph-values", command["command"])
        self.assertIn("6,12", command["command"])
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_normalized_regression_reviews_tradeoff(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "neighborhood_enabled": False,
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "improved",
                    "promotion_action": "promote_best",
                }
            ],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "validation_mse_delta": -0.013,
            "validation_nmse_delta": 0.0048,
            "source_candidate": {
                "validation_graphs": 16,
                "validation_stability_status": "multi_seed_stable",
            },
            "best_candidate": {
                "validation_graphs": 16,
                "validation_stability_status": "multi_seed_stable",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/normalized tradeoff"),
            guidance,
        )

        self.assertEqual(guidance["action"], "review_normalized_tradeoff")
        self.assertTrue(guidance["normalized_regression"])
        self.assertFalse(guidance["target_scale_seed_shift"])
        self.assertIn("--validation-graph-values", command["command"])
        self.assertIn("16,32", command["command"])
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_repeated_seed_shift_regression_widens_with_fresh_seeds(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "regressed",
                    "promotion_action": "keep_source",
                }
            ],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_replay_validation_mse_delta": 0.080,
            "best_vs_source_replay_validation_mse_delta": -0.020,
            "source_replay_validation_stability_status": "watch_spread",
            "best_validation_stability_status": "multi_seed_stable",
            "source_candidate": {
                "validation_stability_status": "watch_spread",
            },
            "best_candidate": {
                "validation_stability_status": "multi_seed_stable",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/repeated seed shift"),
            guidance,
        )

        self.assertEqual(guidance["action"], "widen_seed_shift_neighborhood")
        self.assertEqual(guidance["regressed_streak"], 2)
        self.assertTrue(guidance["source_replay_seed_shift"])
        self.assertIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_persistent_seed_shift_regression_audits_seed_sensitivity(
        self,
    ) -> None:
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "ancestors": [
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "regressed",
                    "promotion_action": "keep_source",
                },
                {
                    "schema": "st.gnn.band_trace_follow_up_chain_ancestor.v1",
                    "verdict": "regressed",
                    "promotion_action": "keep_source",
                },
            ],
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_replay_validation_mse_delta": 0.110,
            "best_vs_source_replay_validation_mse_delta": -0.00003,
            "source_replay_validation_stability_status": "multi_seed_stable",
            "best_validation_stability_status": "multi_seed_stable",
            "source_candidate": {
                "validation_graphs": 6,
                "validation_stability_status": "watch_spread",
            },
            "best_candidate": {
                "validation_stability_status": "multi_seed_stable",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/seed sensitivity audit"),
            guidance,
        )

        self.assertEqual(guidance["action"], "audit_seed_sensitivity")
        self.assertEqual(guidance["regressed_streak"], 3)
        self.assertTrue(guidance["source_replay_seed_shift"])
        self.assertIn("--validation-graph-values", command["command"])
        self.assertIn("6,12", command["command"])
        self.assertNotIn("--follow-up-neighborhood", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])

    def test_gnn_band_trace_regressed_stable_candidate_requests_tradeoff_review(
        self,
    ) -> None:
        chain = {"schema": "st.gnn.band_trace_follow_up_chain.v1", "ancestors": []}
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "regressed",
            "source_validation_stability_status": "volatile",
            "best_validation_stability_status": "multi_seed_stable",
            "source_candidate": {
                "validation_stability_status": "volatile",
            },
            "best_candidate": {
                "validation_stability_status": "multi_seed_stable",
            },
        }

        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/stability tradeoff"),
            guidance,
        )

        self.assertEqual(guidance["action"], "review_stability_tradeoff")
        self.assertTrue(guidance["stability_tradeoff"])
        self.assertEqual(guidance["candidate_validation_stability_status"], "volatile")
        self.assertIn("--follow-up-source", command["command"])
        self.assertIn("top-candidate", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])
        self.assertIn("NEW_SEEDS=...", command["script_usage"])

    def test_gnn_band_trace_promotion_guard_keeps_stable_source(self) -> None:
        source_candidate = {
            "lr": 0.03,
            "top_k": 2,
            "mid_k": 1,
            "bottom_k": 1,
            "here_tolerance": 0.00001,
            "avg_validation_readout_mse": 0.072506,
            "validation_stability_status": "multi_seed_stable",
            "validation_stability_score": 0.020,
            "validation_readout_mse_stddev": 0.001,
            "validation_readout_mse_spread": 0.002,
        }
        volatile_best = {
            "lr": 0.03,
            "top_k": 3,
            "mid_k": 1,
            "bottom_k": 1,
            "here_tolerance": 0.00001,
            "avg_validation_readout_mse": 0.072355,
            "validation_stability_status": "volatile",
            "validation_stability_score": 0.110,
            "validation_readout_mse_stddev": 0.020,
            "validation_readout_mse_spread": 0.050,
        }
        result = {
            "schema": "st.gnn.band_trace_follow_up_result.v1",
            "verdict": "improved",
            "source_candidate": source_candidate,
            "best_candidate": volatile_best,
            "validation_mse_delta": -0.000151,
            "source_validation_stability_status": "multi_seed_stable",
            "best_validation_stability_status": "volatile",
            "source_validation_stability_score": 0.020,
            "best_validation_stability_score": 0.110,
        }

        promotion = run_gnn_band_trace_sweep.follow_up_promotion_record(result)
        self.assertIsNotNone(promotion)
        assert promotion is not None
        chain = {
            "schema": "st.gnn.band_trace_follow_up_chain.v1",
            "promotion_action": promotion["action"],
            "ancestors": [],
        }
        guidance = run_gnn_band_trace_sweep.follow_up_chain_guidance_record(chain, result)
        command = run_gnn_band_trace_sweep.follow_up_guided_next_command_record(
            Path("target/tmp/guarded stable source"),
            guidance,
        )

        self.assertEqual(promotion["action"], "keep_source_stability_guard")
        self.assertTrue(promotion["stability_guard"])
        self.assertEqual(promotion["selected_origin"], "source")
        self.assertEqual(promotion["selected_candidate"], source_candidate)
        self.assertEqual(guidance["action"], "keep_stable_source")
        self.assertEqual(guidance["promotion_action"], "keep_source_stability_guard")
        self.assertEqual(guidance["candidate_validation_stability_status"], "volatile")
        self.assertIn("--follow-up-source", command["command"])
        self.assertIn("auto", command["command"])
        self.assertIn("NEW_SEEDS", command["command"])
        self.assertEqual(command["placeholders"], ["NEXT_RUN_ROOT", "NEW_SEEDS"])
        self.assertIn("NEW_SEEDS=...", command["script_usage"])

    def test_gnn_band_trace_follow_up_gate_flags_requested_verdicts(self) -> None:
        summary = {
            "name": "backend-cpu__top-2__seed-9",
            "backend": "cpu",
            "seed": 9,
            "epochs": 1,
            "train_graphs": 4,
            "validation_graphs": 3,
            "batch": 2,
            "nodes": 4,
            "features": 2,
            "lr": 0.03,
            "top_k": 2,
            "mid_k": 1,
            "bottom_k": 1,
            "here_tolerance": 0.00001,
            "input_rows": 8,
            "run_dir": "run",
            "log_path": "run/process.log",
            "returncode": 0,
            "failure_kind": None,
            "failure_detail": None,
            "skipped": False,
            "failed": False,
            "command": ["cargo", "run"],
            "gnn_summary": {
                "trainer": {"best_score": 0.18},
                "readout": {
                    "trace": {"graph_count": 2, "total_rows": 8},
                    "error": {"mean_squared_error": 0.20},
                },
                "validation_readout": {
                    "graph_count": 3,
                    "total_rows": 12,
                    "mean_squared_error": 0.18,
                },
                "bands": {},
            },
            "trainer_summary": {"metrics": {}},
            "run_meta": {},
        }
        source_candidate = {
            "rank": 1,
            "backend": "cpu",
            "epochs": 1,
            "train_graphs": 4,
            "validation_graphs": 3,
            "batch": 2,
            "nodes": 4,
            "features": 2,
            "lr": 0.03,
            "top_k": 1,
            "mid_k": 1,
            "bottom_k": 1,
            "here_tolerance": 0.00001,
            "avg_validation_readout_mse": 0.16,
        }

        comparison = run_gnn_band_trace_sweep.comparison_summary(
            [summary],
            follow_up_candidate=source_candidate,
            follow_up_fail_on_verdict=["regressed", "unknown"],
        )

        self.assertEqual(comparison["follow_up_result"]["verdict"], "regressed")
        self.assertEqual(comparison["follow_up_gate"]["verdict"], "regressed")
        self.assertTrue(comparison["follow_up_gate"]["failed"])
        self.assertTrue(run_gnn_band_trace_sweep.follow_up_gate_failed(comparison))
        promotion = comparison["follow_up_promotion"]
        self.assertEqual(promotion["action"], "keep_source")
        self.assertEqual(promotion["selected_origin"], "source")
        self.assertEqual(promotion["selected_candidate"]["top_k"], 1)

    def test_gnn_threshold_grid_dry_run_writes_threshold_shape_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_gnn_threshold_grid.main(
                    [
                        "--run-root",
                        str(root),
                        "--thresholds",
                        "1,1024",
                        "--nodes",
                        "4",
                        "--features",
                        "2",
                        "--batches",
                        "1,2",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "9,10",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "grid.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["schema"], "st.gnn.tensor_util_threshold_grid.v2")
        self.assertEqual(manifest["thresholds"], [1, 1024])
        self.assertEqual(manifest["batches"], [1, 2])
        self.assertEqual(len(manifest["runs"]), 8)
        target = next(
            run
            for run in manifest["runs"]
            if run["threshold"] == 1024 and run["batch"] == 2 and run["seed"] == 10
        )
        self.assertEqual(target["input_rows"], 8)
        self.assertEqual(target["output_values"], 16)
        self.assertEqual(target["hidden_values"], 32)
        self.assertIn("threshold-1024__nodes-4__features-2__batch-2", target["name"])
        self.assertEqual(target["command"][target["command"].index("--batch") + 1], "2")
        self.assertEqual(target["command"][target["command"].index("--nodes") + 1], "4")
        self.assertEqual(target["command"][target["command"].index("--features") + 1], "2")

    def test_gnn_threshold_grid_compare_surfaces_validation_readout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = run_gnn_threshold_grid.write_compare(
                root,
                [
                    {
                        "threshold": 1024,
                        "nodes": 4,
                        "features": 2,
                        "batch": 2,
                        "backend": "cpu",
                        "seed": 9,
                        "input_rows": 8,
                        "output_values": 16,
                        "hidden_values": 32,
                        "run_dir": str(root / "run"),
                        "log_path": str(root / "run" / "process.log"),
                        "returncode": 0,
                        "failure_kind": None,
                        "failure_detail": None,
                        "skipped": False,
                        "failed": False,
                        "gnn_summary": {
                            "trainer": {"best_score": 0.12},
                            "readout": {
                                "trace": {"graph_count": 2, "total_rows": 8},
                                "error": {
                                    "mean_squared_error": 0.20,
                                    "normalized_mean_squared_error": 1.25,
                                },
                            },
                            "validation_readout": {
                                "graph_count": 3,
                                "total_rows": 12,
                                "mean_squared_error": 0.15,
                                "normalized_mean_squared_error": 0.75,
                            },
                            "bands": {},
                        },
                        "trainer_summary": {"metrics": {}},
                        "run_meta": {},
                    }
                ],
            )

            report = path.read_text(encoding="utf-8")

        self.assertIn("validation_readout_mse", report)
        self.assertIn("validation_readout_nmse", report)
        self.assertIn("avg_validation_readout_mse", report)
        self.assertIn("avg_validation_readout_nmse", report)
        self.assertIn("0.150000", report)
        self.assertIn("0.750000", report)
        self.assertIn("| 1024 | cpu | 4 | 2 | 2 | 1 | 16 | 32 |", report)

    def test_gnn_threshold_grid_preflight_skip_keeps_shape_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = run_gnn_threshold_grid.parse_args(
                [
                    "--run-root",
                    str(root),
                    "--thresholds",
                    "1024",
                    "--nodes",
                    "4",
                    "--features",
                    "2",
                    "--batches",
                    "2",
                    "--backends",
                    "wgpu",
                    "--seeds",
                    "9",
                    "--continue-on-error",
                ]
            )
            axes = next(run_gnn_threshold_grid.iter_axes(args))
            record = run_gnn_threshold_grid.preflight_skipped_run(
                args,
                axes,
                {
                    "failure_kind": "signal",
                    "failure_detail": "signal:6:empty_log",
                    "returncode": -6,
                    "log_path": str(root / "_preflight" / "backend-wgpu" / "process.log"),
                },
            )
            failure = json.loads(
                (Path(record["run_dir"]) / "failure.json").read_text(encoding="utf-8")
            )
            log_text = Path(record["log_path"]).read_text(encoding="utf-8")

        self.assertTrue(record["failed"])
        self.assertTrue(record["skipped"])
        self.assertEqual(record["failure_kind"], "preflight_signal")
        self.assertEqual(record["threshold"], 1024)
        self.assertEqual(record["input_rows"], 8)
        self.assertEqual(record["output_values"], 16)
        self.assertEqual(record["hidden_values"], 32)
        self.assertEqual(failure["schema"], "st.gnn.tensor_util_threshold_grid_failure.v1")
        self.assertEqual(failure["failure_detail"], "signal:6:empty_log")
        self.assertIn("skipped after WGPU sweep preflight failure", log_text)

    def test_char_lm_failed_runs_markdown_only_lists_failed_runs(self) -> None:
        markdown = run_char_lm_sweep.failed_runs_markdown(
            [
                {
                    "name": "cpu-run",
                    "backend": "cpu",
                    "run_status": "ok",
                    "failed": False,
                },
                {
                    "name": "wgpu-run",
                    "architecture": "finetune",
                    "backend": "wgpu",
                    "seed": 7,
                    "run_status": "failed",
                    "failed": True,
                    "returncode": -6,
                    "failure_kind": "preflight_signal",
                    "failure_detail": "signal:6",
                    "log_path": "target/tmp/wgpu-run/process.log",
                },
            ]
        )

        self.assertIn("## Failed Runs", markdown)
        self.assertIn("wgpu-run", markdown)
        self.assertIn("signal:6", markdown)
        self.assertNotIn("cpu-run", markdown)

    def test_char_lm_lstm_architecture_routes_finetune_with_recurrent_flag(self) -> None:
        settings = run_char_lm_sweep.SweepSettings(
            epochs=1,
            batches=1,
            batch=2,
            eval_samples=4,
            gen=4,
            early_stop_patience=0,
            steps=4,
            embed_dim=4,
            hidden=8,
            memory=None,
            lr=None,
            curvature=None,
            temperature=None,
            head_residual_scale=None,
            context_scale=None,
            self_score_scale=None,
            query_residual_scale=None,
            wave_kernel=None,
            wave_dilations=None,
            backend="cpu",
        )

        command = run_char_lm_sweep.build_command(
            cargo_bin="cargo",
            cargo_features=None,
            no_default_features=False,
            architecture="lstm",
            data_paths=[Path("demo.txt")],
            run_dir=Path("target/tmp/demo"),
            char_feature="token-bigram",
            head_prior="learned-unigram",
            seed=7,
            settings=settings,
            extra_args=[],
        )

        self.assertEqual(run_char_lm_sweep.EXAMPLES["lstm"], "modelzoo_llm_char_finetune")
        self.assertIn("modelzoo_llm_char_finetune", command)
        recurrent_index = command.index("--recurrent")
        self.assertEqual(command[recurrent_index + 1], "lstm")

    def test_char_lm_sweep_grid_values_fall_back_to_single_settings(self) -> None:
        self.assertEqual(
            run_char_lm_sweep.grid_values(None, 4, label="step-values"),
            [4],
        )
        self.assertEqual(
            run_char_lm_sweep.grid_values("4,6", None, label="step-values"),
            [4, 6],
        )

        with self.assertRaises(ValueError):
            run_char_lm_sweep.grid_values("4,0", None, label="step-values")

    def test_char_lm_sweep_float_grid_values_fall_back_to_single_settings(self) -> None:
        self.assertEqual(
            run_char_lm_sweep.float_grid_values(
                None,
                1.0,
                label="head-residual-scale-values",
            ),
            [1.0],
        )
        self.assertEqual(
            run_char_lm_sweep.float_grid_values(
                "0,0.5,2",
                None,
                label="head-residual-scale-values",
            ),
            [0.0, 0.5, 2.0],
        )
        self.assertEqual(
            run_char_lm_sweep.float_grid_values(
                None,
                0.0,
                label="head-residual-scale-values",
            ),
            [0.0],
        )

        with self.assertRaises(ValueError):
            run_char_lm_sweep.float_grid_values(
                "1,nan",
                None,
                label="head-residual-scale-values",
            )
        with self.assertRaises(ValueError):
            run_char_lm_sweep.float_grid_values(
                "-0.1",
                None,
                label="head-residual-scale-values",
            )

    def test_char_lm_sweep_compare_summary_accepts_bigram_lift_sort_metric(self) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--compare-summary-sort-metric",
                "final_bigram_logprob_lift",
            ]
        )
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(options.sort_metric, "final_bigram_logprob_lift")
        self.assertEqual(options.fail_on_route_statuses, ())
        self.assertEqual(options.fail_on_paired_quality_statuses, ())
        self.assertEqual(options.fail_on_efficiency_verdicts, ())
        self.assertEqual(options.fail_on_rank_min_promotion_decisions, ())
        self.assertEqual(options.fail_on_route_debt_decisions, ())
        self.assertEqual(options.extra_compare_paths, ())
        self.assertFalse(options.merge_evidence_sources)

        debt_args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--compare-summary-sort-metric",
                "final_bigram_rank_debt",
            ]
        )
        debt_options = run_char_lm_sweep.compare_summary_options_from_args(debt_args)

        self.assertEqual(debt_options.sort_metric, "final_bigram_rank_debt")

        route_debt_args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--compare-summary-sort-metric",
                "coherence_route_debt",
            ]
        )
        route_debt_options = run_char_lm_sweep.compare_summary_options_from_args(
            route_debt_args
        )

        self.assertEqual(route_debt_options.sort_metric, "coherence_route_debt")

    def test_char_lm_sweep_compare_summary_accepts_extra_merge_inputs(self) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--compare-summary-extra-compare-json",
                "previous/compare.json,confirm",
                "--compare-summary-extra-compare-json",
                "third/compare.json",
                "--compare-summary-merge-evidence-sources",
            ]
        )
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(
            options.extra_compare_paths,
            (
                Path("previous/compare.json"),
                Path("confirm"),
                Path("third/compare.json"),
            ),
        )
        self.assertTrue(options.merge_evidence_sources)

    def test_char_lm_sweep_manifest_records_compare_summary_extra_merge_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--dry-run",
                        "--no-wgpu-preflight",
                        "--compare-summary-extra-compare-json",
                        "previous/compare.json,confirm",
                        "--compare-summary-merge-evidence-sources",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["compare_summary"]["extra_compare_paths"],
            ["previous/compare.json", "confirm"],
        )
        self.assertTrue(manifest["compare_summary"]["merge_evidence_sources"])
        summary_command = manifest["compare_summary"]["command"]
        resolved_command_paths = {
            str(Path(part).resolve()) for part in summary_command if part.endswith(".json")
        }
        self.assertIn(str((root / "compare.json").resolve()), resolved_command_paths)
        self.assertIn("previous/compare.json", summary_command)
        self.assertIn("confirm", summary_command)
        self.assertIn("--merge-evidence-sources", summary_command)
        self.assertIn("--json-out", summary_command)
        self.assertEqual(
            manifest["compare_summary"]["command_cwd"],
            str(run_char_lm_sweep.REPO_ROOT),
        )
        self.assertEqual(
            str(Path(manifest["compare_summary"]["command_script_path"]).resolve()),
            str((root / "compare_summary_command.sh").resolve()),
        )

    def test_char_lm_sweep_bigram_guard_grid_dry_run_writes_names_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "bigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--bigram-topk-guard-values",
                        "0,0.05",
                        "--bigram-topk-guard-k",
                        "3",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["bigram_topk_guards"], [0.0, 0.05])
        self.assertEqual(manifest["bigram_topk_guard_k"], 3)
        self.assertEqual(len(manifest["runs"]), 2)
        guarded = next(run for run in manifest["runs"] if run["bigram_topk_guard"] == 0.05)
        self.assertIn("biguard-0.05__biguardk-3", guarded["name"])
        self.assertIn("--bigram-topk-guard", guarded["command"])
        self.assertIn("--bigram-topk-guard-k", guarded["command"])
        self.assertEqual(guarded["command"][guarded["command"].index("--bigram-topk-guard") + 1], "0.05")
        self.assertEqual(guarded["command"][guarded["command"].index("--bigram-topk-guard-k") + 1], "3")

    def test_char_lm_sweep_guarded_lstm_recipe_expands_sweetspot_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "guarded-lstm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "guarded-lstm")
        self.assertIn("guarded bigram sweet-spot", manifest["recipe_description"])
        self.assertEqual(
            manifest["recipe_defaults"]["bigram_topk_guard_values"],
            "0,0.1",
        )
        self.assertEqual(manifest["recipe_defaults"]["epoch_values"], "1,2")
        self.assertEqual(manifest["architectures"], ["finetune", "lstm"])
        self.assertEqual(manifest["head_priors"], ["bigram"])
        self.assertEqual(manifest["seeds"], [7, 11])
        self.assertEqual(manifest["shape_grid"]["steps"], [12])
        self.assertEqual(manifest["shape_grid"]["embed_dims"], [8])
        self.assertEqual(manifest["shape_grid"]["hidden"], [8])
        self.assertEqual(manifest["training_grid"]["epochs"], [1, 2])
        self.assertEqual(manifest["training_grid"]["batches"], [8, 16])
        self.assertEqual(manifest["head_residual_scales"], [2.0])
        self.assertEqual(manifest["bigram_topk_guards"], [0.0, 0.1])
        self.assertEqual(manifest["bigram_topk_guard_k"], 5)
        self.assertEqual(len(manifest["runs"]), 32)
        sweetspot = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "lstm"
            and run["seed"] == 11
            and run["epochs"] == 2
            and run["batches"] == 16
            and run["bigram_topk_guard"] == 0.1
        )
        self.assertIn("headresid-2", sweetspot["name"])
        self.assertIn("biguard-0.1__biguardk-5", sweetspot["name"])
        self.assertIn("epochs-2__batches-16", sweetspot["name"])
        self.assertEqual(
            sweetspot["command"][sweetspot["command"].index("--head-residual-scale") + 1],
            "2",
        )
        self.assertEqual(
            sweetspot["command"][sweetspot["command"].index("--bigram-topk-guard") + 1],
            "0.1",
        )

    def test_char_lm_sweep_guarded_lstm_recipe_preserves_overrides(self) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "guarded-lstm",
                "--architectures",
                "lstm",
                "--seeds",
                "3",
                "--batches-values",
                "4",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)

        self.assertEqual(args.architectures, "lstm")
        self.assertEqual(args.seeds, "3")
        self.assertEqual(args.batches_values, "4")
        self.assertEqual(args.head_priors, "bigram")
        self.assertEqual(args.bigram_topk_guard_values, "0,0.1")

    def test_char_lm_sweep_shape_winners_recipe_uses_promoted_lite_wave(
        self,
    ) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "no-prior-coherence-shape-winners",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)

        self.assertEqual(args.architectures, "scan,wave")
        self.assertEqual(args.context_scale, 2)
        self.assertEqual(args.query_residual_scale, 2)
        self.assertEqual(args.wave_kernel, 3)
        self.assertEqual(args.wave_dilations, "1")

    def test_char_lm_sweep_wave_promoted_recipe_expands_route_debt_grid(
        self,
    ) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "no-prior-coherence-wave-promoted",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(options.sort_metric, "coherence_route_debt")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "no-prior-coherence-wave-promoted",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "no-prior-coherence-wave-promoted")
        self.assertIn("promoted lite wave", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["wave"])
        self.assertEqual(manifest["head_priors"], ["none"])
        self.assertEqual(manifest["seeds"], [7, 13, 23])
        self.assertEqual(manifest["training_grid"]["epochs"], [10])
        self.assertEqual(manifest["training_grid"]["batches"], [32])
        self.assertEqual(manifest["coherence_grid"]["wave_kernels"], [3])
        self.assertEqual(manifest["coherence_grid"]["wave_dilations"], ["1"])
        self.assertEqual(
            manifest["recipe_defaults"]["compare_summary_sort_metric"],
            "coherence_route_debt",
        )
        self.assertEqual(len(manifest["runs"]), 3)
        run = manifest["runs"][0]
        self.assertIn("dil-1", run["name"])
        self.assertEqual(run["wave_dilations"], "1")
        self.assertEqual(run["command"][run["command"].index("--dilations") + 1], "1")

    def test_char_lm_sweep_wave_long_recipe_expands_long_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "no-prior-coherence-wave-long",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "no-prior-coherence-wave-long")
        self.assertIn("longer budget", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["wave"])
        self.assertEqual(manifest["head_priors"], ["none"])
        self.assertEqual(manifest["seeds"], [7, 13, 23])
        self.assertEqual(manifest["training_grid"]["epochs"], [16])
        self.assertEqual(manifest["training_grid"]["batches"], [64])
        self.assertEqual(manifest["settings"]["eval_samples"], 128)
        self.assertEqual(manifest["settings"]["early_stop_patience"], 6)
        self.assertEqual(manifest["settings"]["gen"], 128)
        self.assertEqual(manifest["coherence_grid"]["query_residual_scales"], [2.0])
        self.assertEqual(manifest["coherence_grid"]["wave_kernels"], [3])
        self.assertEqual(manifest["coherence_grid"]["wave_dilations"], ["1"])
        self.assertEqual(manifest["extra_args"], ["--mix-rms", "1.0"])
        self.assertEqual(len(manifest["runs"]), 3)
        run = manifest["runs"][0]
        self.assertIn("dil-1", run["name"])
        self.assertIn("epochs-16", run["name"])
        self.assertIn("batches-64", run["name"])
        self.assertEqual(run["command"][run["command"].index("--dilations") + 1], "1")
        self.assertEqual(run["command"][run["command"].index("--mix-rms") + 1], "1.0")

    def test_char_lm_sweep_wave_wide_corpus_recipe_accepts_docs_bundle(
        self,
    ) -> None:
        corpus_paths = [
            "models/samples/spiral_corpus_en",
            "models/samples/spiral_demo_en.txt",
            "models/README.md",
            "docs/getting-started.md",
            "docs/example-gallery.md",
            "docs/zspace_intro.md",
            "bindings/st-py/README.md",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        *corpus_paths,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "no-prior-coherence-wave-wide-corpus",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "no-prior-coherence-wave-wide-corpus")
        self.assertIn("widened corpus", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], corpus_paths)
        self.assertEqual(manifest["architectures"], ["wave"])
        self.assertEqual(manifest["head_priors"], ["none"])
        self.assertEqual(manifest["seeds"], [7, 13, 23])
        self.assertEqual(manifest["training_grid"]["epochs"], [16])
        self.assertEqual(manifest["training_grid"]["batches"], [64])
        self.assertEqual(manifest["settings"]["eval_samples"], 128)
        self.assertEqual(manifest["settings"]["early_stop_patience"], 6)
        self.assertEqual(manifest["settings"]["gen"], 128)
        self.assertEqual(manifest["coherence_grid"]["query_residual_scales"], [2.0])
        self.assertEqual(manifest["coherence_grid"]["wave_kernels"], [3])
        self.assertEqual(manifest["coherence_grid"]["wave_dilations"], ["1"])
        self.assertEqual(manifest["extra_args"], ["--mix-rms", "1.0"])
        self.assertEqual(len(manifest["runs"]), 3)

        run = manifest["runs"][0]
        self.assertIn("dil-1", run["name"])
        self.assertIn("epochs-16", run["name"])
        self.assertIn("batches-64", run["name"])
        self.assertEqual(run["command"][run["command"].index("--") + 1], corpus_paths[0])
        self.assertIn("models/README.md", run["command"])
        self.assertIn("bindings/st-py/README.md", run["command"])
        self.assertEqual(run["command"][run["command"].index("--dilations") + 1], "1")
        self.assertEqual(run["command"][run["command"].index("--mix-rms") + 1], "1.0")

    def test_char_lm_sweep_promoted_frontier_recipe_compares_scan_and_wave(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "no-prior-coherence-promoted-frontier",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "no-prior-coherence-promoted-frontier")
        self.assertIn("promoted lite wave", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["scan", "wave"])
        self.assertEqual(manifest["head_priors"], ["none"])
        self.assertEqual(manifest["seeds"], [7, 13, 23])
        self.assertEqual(manifest["training_grid"]["epochs"], [10])
        self.assertEqual(manifest["training_grid"]["batches"], [32])
        self.assertEqual(manifest["coherence_grid"]["context_scales"], [2.0])
        self.assertEqual(manifest["coherence_grid"]["query_residual_scales"], [2.0])
        self.assertEqual(manifest["coherence_grid"]["wave_kernels"], [3])
        self.assertEqual(manifest["coherence_grid"]["wave_dilations"], ["1"])
        self.assertEqual(
            manifest["architecture_extra_args"],
            {"scan": ["--mix-rms", "1.0"], "wave": ["--mix-rms", "1.0"]},
        )
        self.assertEqual(len(manifest["runs"]), 6)

        scan_runs = [run for run in manifest["runs"] if run["architecture"] == "scan"]
        wave_runs = [run for run in manifest["runs"] if run["architecture"] == "wave"]
        self.assertEqual(len(scan_runs), 3)
        self.assertEqual(len(wave_runs), 3)
        self.assertTrue(all("ctx-2" in run["name"] for run in scan_runs))
        self.assertTrue(all("dil-1" in run["name"] for run in wave_runs))
        self.assertTrue(
            all(
                run["command"][run["command"].index("--mix-rms") + 1] == "1.0"
                for run in [*scan_runs, *wave_runs]
            )
        )

    def test_char_lm_sweep_no_prior_coherence_frontier_recipe_expands_finalists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "no-prior-coherence-frontier",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "no-prior-coherence-frontier")
        self.assertIn("scan shape winner", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["head_priors"], ["none"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["training_grid"]["epochs"], [8])
        self.assertEqual(manifest["training_grid"]["batches"], [24])
        self.assertEqual(manifest["coherence_grid"]["context_scales"], [2.0])
        self.assertEqual(manifest["coherence_grid"]["query_residual_scales"], [2.0])
        self.assertEqual(manifest["coherence_grid"]["wave_kernels"], [3])
        self.assertEqual(manifest["coherence_grid"]["wave_dilations"], ["1", "1,2"])
        self.assertEqual(
            manifest["architecture_extra_args"],
            {"scan": ["--mix-rms", "1.0"], "wave": ["--mix-rms", "1.0"]},
        )
        self.assertEqual(len(manifest["runs"]), 8)

        lstm_runs = [run for run in manifest["runs"] if run["architecture"] == "lstm"]
        scan_runs = [run for run in manifest["runs"] if run["architecture"] == "scan"]
        wave_runs = [run for run in manifest["runs"] if run["architecture"] == "wave"]
        self.assertEqual(len(lstm_runs), 2)
        self.assertEqual(len(scan_runs), 2)
        self.assertEqual(len(wave_runs), 4)
        self.assertNotIn("--dilations", lstm_runs[0]["command"])
        self.assertNotIn("--context-scale", lstm_runs[0]["command"])
        self.assertNotIn("--mix-rms", lstm_runs[0]["command"])

        scan = scan_runs[0]
        self.assertIn("ctx-2", scan["name"])
        self.assertEqual(scan["command"][scan["command"].index("--context-scale") + 1], "2")
        self.assertEqual(
            scan["command"][scan["command"].index("--query-residual-scale") + 1],
            "2",
        )
        self.assertEqual(scan["command"][scan["command"].index("--mix-rms") + 1], "1.0")

        wave_dilations = sorted({run["wave_dilations"] for run in wave_runs})
        self.assertEqual(wave_dilations, ["1", "1,2"])
        self.assertTrue(all("--dilations" in run["command"] for run in wave_runs))
        self.assertTrue(
            all(
                run["command"][run["command"].index("--mix-rms") + 1] == "1.0"
                for run in wave_runs
            )
        )

    def test_char_lm_sweep_architecture_extra_args_apply_to_selected_arches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm,scan",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "none",
                        "--seeds",
                        "7",
                        "--architecture-extra-arg",
                        "scan:--mix-rms",
                        "--architecture-extra-arg",
                        "scan:1.0",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["architecture_extra_args"],
            {"scan": ["--mix-rms", "1.0"]},
        )
        lstm = next(run for run in manifest["runs"] if run["architecture"] == "lstm")
        scan = next(run for run in manifest["runs"] if run["architecture"] == "scan")
        self.assertNotIn("--mix-rms", lstm["command"])
        self.assertEqual(scan["command"][scan["command"].index("--mix-rms") + 1], "1.0")

    def test_char_lm_sweep_full_promotion_recipe_adds_rank_min_gate(
        self,
    ) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "hard-rank-band-adaptive-architecture-topk-schedule-scale-full",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(
            options.fail_on_rank_min_promotion_decisions,
            (
                "no_rank_min_evidence",
                "needs_tuning",
                "partial_promote_needs_tuning",
            ),
        )

    def test_char_lm_sweep_full_promotion_recipe_preserves_gate_override(
        self,
    ) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "hard-rank-band-adaptive-architecture-topk-schedule-scale-full",
                "--compare-summary-fail-on-rank-min-promotion-decision",
                "promote",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(options.fail_on_rank_min_promotion_decisions, ("promote",))

    def test_char_lm_sweep_wave_lite_confirm_recipe_adds_route_debt_gate(
        self,
    ) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "no-prior-coherence-wave-lite-confirm",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(
            options.fail_on_route_debt_decisions,
            ("no_route_debt_recommendation",),
        )
        self.assertEqual(options.sort_metric, "coherence_route_debt")

    def test_char_lm_sweep_wave_lite_confirm_recipe_preserves_route_debt_gate_override(
        self,
    ) -> None:
        args = run_char_lm_sweep.parse_args(
            [
                "data.txt",
                "--recipe",
                "no-prior-coherence-wave-lite-confirm",
                "--compare-summary-fail-on-route-debt-decision",
                "promote_lite_wave",
            ]
        )
        args = run_char_lm_sweep.apply_recipe_defaults(args)
        options = run_char_lm_sweep.compare_summary_options_from_args(args)

        self.assertEqual(options.fail_on_route_debt_decisions, ("promote_lite_wave",))

    def test_char_lm_sweep_val_start_hardness_recipe_expands_probe_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "val-start-hardness",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "val-start-hardness")
        self.assertIn("near-full validation slice", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["finetune", "lstm"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["shape_grid"]["steps"], [12])
        self.assertEqual(manifest["shape_grid"]["hidden"], [16])
        self.assertEqual(manifest["training_grid"]["epochs"], [10])
        self.assertEqual(manifest["training_grid"]["batches"], [24])
        self.assertEqual(manifest["training_grid"]["eval_samples"], [128])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(manifest["learning_rates"], [0.0025])
        self.assertEqual(manifest["head_residual_scales"], [0.5])
        self.assertEqual(len(manifest["runs"]), 12)
        middle = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "lstm"
            and run["seed"] == 13
            and run["validation_start_fraction"] == 0.5
        )
        self.assertIn("valstart-0.5", middle["name"])
        self.assertEqual(middle["command"][middle["command"].index("--eval-samples") + 1], "128")
        self.assertEqual(
            middle["command"][middle["command"].index("--val-start-fraction") + 1],
            "0.5",
        )

    def test_char_lm_sweep_hard_bigram_prior_recipe_expands_probe_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-bigram-prior",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-bigram-prior")
        self.assertIn("hardest validation-start slice", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["finetune", "lstm"])
        self.assertEqual(
            manifest["head_priors"],
            ["learned-unigram", "bigram", "learned-bigram"],
        )
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["shape_grid"]["steps"], [12])
        self.assertEqual(manifest["shape_grid"]["hidden"], [16])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0],
        )
        self.assertEqual(manifest["learning_rates"], [0.0025])
        self.assertEqual(manifest["head_residual_scales"], [0.5])
        self.assertEqual(len(manifest["runs"]), 12)
        bigram_run = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "lstm"
            and run["head_prior"] == "bigram"
            and run["seed"] == 13
        )
        self.assertIn("valstart-0", bigram_run["name"])
        self.assertEqual(
            bigram_run["command"][bigram_run["command"].index("--head-prior") + 1],
            "bigram",
        )
        self.assertEqual(
            bigram_run["command"][bigram_run["command"].index("--val-start-fraction") + 1],
            "0",
        )

    def test_char_lm_sweep_hard_bigram_guard_recipe_expands_probe_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-bigram-guard",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-bigram-guard")
        self.assertIn("bigram top-k guard", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["finetune", "lstm"])
        self.assertEqual(manifest["head_priors"], ["bigram", "learned-bigram"])
        self.assertEqual(manifest["bigram_topk_guards"], [0.0, 0.1])
        self.assertEqual(manifest["bigram_topk_guard_k"], 5)
        self.assertEqual(manifest["bigram_rank_guards"], [0.0, 0.05])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.0])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0],
        )
        self.assertEqual(len(manifest["runs"]), 32)
        guarded = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "lstm"
            and run["head_prior"] == "learned-bigram"
            and run["bigram_topk_guard"] == 0.1
            and run["bigram_rank_guard"] == 0.05
            and run["seed"] == 13
        )
        self.assertIn("head-learned-bigram", guarded["name"])
        self.assertIn("biguard-0.1__biguardk-5", guarded["name"])
        self.assertIn("bigrank-0.05__bigrankm-0", guarded["name"])
        self.assertIn("valstart-0", guarded["name"])
        self.assertEqual(
            guarded["command"][guarded["command"].index("--bigram-topk-guard") + 1],
            "0.1",
        )
        self.assertEqual(
            guarded["command"][guarded["command"].index("--bigram-topk-guard-k") + 1],
            "5",
        )
        self.assertEqual(
            guarded["command"][guarded["command"].index("--bigram-rank-guard") + 1],
            "0.05",
        )
        self.assertEqual(
            guarded["command"][guarded["command"].index("--bigram-rank-guard-margin") + 1],
            "0",
        )

    def test_char_lm_sweep_bigram_rank_guard_margin_grid_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-bigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--bigram-rank-guard-values",
                        "0.05",
                        "--bigram-rank-guard-margin-values",
                        "0,0.1",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["bigram_rank_guards"], [0.05])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.0, 0.1])
        self.assertEqual(len(manifest["runs"]), 2)
        margin_run = next(
            run for run in manifest["runs"] if run["bigram_rank_guard_margin"] == 0.1
        )
        self.assertIn("bigrank-0.05__bigrankm-0.1", margin_run["name"])
        self.assertEqual(
            margin_run["command"][margin_run["command"].index("--bigram-rank-guard") + 1],
            "0.05",
        )
        self.assertEqual(
            margin_run["command"][margin_run["command"].index("--bigram-rank-guard-margin") + 1],
            "0.1",
        )

    def test_char_lm_sweep_bigram_rank_guard_band_grid_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-bigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--bigram-rank-guard",
                        "0.1",
                        "--bigram-rank-guard-margin",
                        "0.05",
                        "--bigram-rank-guard-band-values",
                        "0,0.003",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["bigram_rank_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.05])
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.0, 0.003])
        self.assertEqual(len(manifest["runs"]), 2)
        band_run = next(
            run for run in manifest["runs"] if run["bigram_rank_guard_band"] == 0.003
        )
        self.assertIn("bigrank-0.1__bigrankm-0.05__bigrankband-0.003", band_run["name"])
        self.assertEqual(
            band_run["command"][band_run["command"].index("--bigram-rank-guard-band") + 1],
            "0.003",
        )

    def test_char_lm_sweep_bigram_rank_guard_min_candidates_grid_dry_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-bigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--bigram-topk-guard-k",
                        "5",
                        "--bigram-rank-guard",
                        "0.1",
                        "--bigram-rank-guard-band",
                        "0.003",
                        "--bigram-rank-guard-min-candidates-values",
                        "0,2",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.003])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 2])
        self.assertEqual(len(manifest["runs"]), 2)
        min_run = next(
            run for run in manifest["runs"] if run["bigram_rank_guard_min_candidates"] == 2
        )
        self.assertIn("bigrank-0.1__bigrankband-0.003__bigrankmin-2", min_run["name"])
        self.assertEqual(
            min_run["command"][
                min_run["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "2",
        )

    def test_char_lm_sweep_hard_rank_guard_local_recipe_expands_probe_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-guard-local",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-guard-local")
        self.assertIn("focused local rank-debt", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guards"], [0.0, 0.05, 0.1, 0.5])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.0, 0.05])
        self.assertEqual(len(manifest["runs"]), 8)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["bigram_rank_guard"] == 0.1
            and run["bigram_rank_guard_margin"] == 0.05
        )
        self.assertIn("bigrank-0.1__bigrankm-0.05", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-rank-guard") + 1],
            "0.1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-rank-guard-margin") + 1],
            "0.05",
        )

    def test_char_lm_sweep_hard_rank_band_local_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-local",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-band-local")
        self.assertIn("competitor bands", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.05])
        self.assertEqual(
            manifest["bigram_rank_guard_bands"],
            [0.0, 0.001, 0.003, 0.005, 0.01],
        )
        self.assertEqual(len(manifest["runs"]), 10)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13 and run["bigram_rank_guard_band"] == 0.003
        )
        self.assertIn("bigrank-0.1__bigrankm-0.05__bigrankband-0.003", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-rank-guard-band") + 1],
            "0.003",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_local_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-local",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-band-adaptive-local")
        self.assertIn("adaptive minimum competitor", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.003])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1, 2, 3])
        self.assertEqual(len(manifest["runs"]), 8)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13 and run["bigram_rank_guard_min_candidates"] == 2
        )
        self.assertIn("bigrankband-0.003__bigrankmin-2", candidate["name"])
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "2",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_confirm_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-band-adaptive-confirm")
        self.assertIn("confirm adaptive bigram-rank", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.003])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1, 2, 3])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(len(manifest["runs"]), 48)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 34
            and run["bigram_rank_guard_min_candidates"] == 3
            and run["validation_start_fraction"] == 1.0
        )
        self.assertIn("bigrankband-0.003__bigrankmin-3", candidate["name"])
        self.assertIn("valstart-1", candidate["name"])
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "3",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "1",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-band-adaptive-safe")
        self.assertIn("stable min=1", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.003])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(len(manifest["runs"]), 24)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 34
            and run["bigram_rank_guard_min_candidates"] == 1
            and run["validation_start_fraction"] == 1.0
        )
        self.assertIn("bigrankband-0.003__bigrankmin-1", candidate["name"])
        self.assertIn("valstart-1", candidate["name"])
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "1",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_arches_recipe_expands_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe-arches",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-band-adaptive-safe-arches")
        self.assertIn("LSTM to scan and wave", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.003])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 36)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 13
            and run["bigram_rank_guard_min_candidates"] == 1
            and run["validation_start_fraction"] == 1.0
        )
        self.assertEqual(candidate["example"], "modelzoo_llm_char_coherence_wave")
        self.assertIn("wave__feature-token-bigram", candidate["name"])
        self.assertIn("bigrankband-0.003__bigrankmin-1", candidate["name"])
        self.assertIn("valstart-1", candidate["name"])
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_corpus_recipe_expands_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe-corpus",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-band-adaptive-safe-corpus")
        self.assertIn("broader text/corpus", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], [corpus])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_rank_guard_bands"], [0.003])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 36)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "scan"
            and run["seed"] == 13
            and run["bigram_rank_guard_min_candidates"] == 1
            and run["validation_start_fraction"] == 0.5
        )
        self.assertEqual(candidate["example"], "modelzoo_llm_char_coherence_scan")
        self.assertEqual(candidate["command"][candidate["command"].index("--") + 1], corpus)
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_corpus_frontier_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe-corpus-frontier",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-safe-corpus-frontier",
        )
        self.assertIn("corpus val-start=0 frontier", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], [corpus])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 24)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["validation_start_fraction"], 0)
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "0",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_corpus_frontier_topk_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe-corpus-frontier-topk",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-safe-corpus-frontier-topk",
        )
        self.assertIn("stronger top-k", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], [corpus])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 21])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1, 0.2, 0.5])
        self.assertEqual(manifest["bigram_topk_guard_k"], 5)
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 36)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "scan"
            and run["seed"] == 21
            and run["bigram_topk_guard"] == 0.5
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["example"], "modelzoo_llm_char_coherence_scan")
        self.assertEqual(candidate["validation_start_fraction"], 0)
        self.assertIn("biguard-0.5__biguardk-5", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-topk-guard") + 1],
            "0.5",
        )
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "0",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_corpus_frontier_topk_confirm_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe-corpus-frontier-topk-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-safe-corpus-frontier-topk-confirm",
        )
        self.assertIn("top-k guard 0.5", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], [corpus])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_topk_guards"], [0.5])
        self.assertEqual(manifest["bigram_topk_guard_k"], 5)
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 24)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["example"], "modelzoo_llm_char_coherence_wave")
        self.assertEqual(candidate["bigram_topk_guard"], 0.5)
        self.assertIn("biguard-0.5__biguardk-5", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-topk-guard") + 1],
            "0.5",
        )
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "0",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_safe_corpus_topk_confirm_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-safe-corpus-topk-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-safe-corpus-topk-confirm",
        )
        self.assertIn("top-k guard 0.2", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], [corpus])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2])
        self.assertEqual(manifest["bigram_topk_guard_k"], 5)
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 72)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "scan"
            and run["seed"] == 21
            and run["validation_start_fraction"] == 0.5
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["example"], "modelzoo_llm_char_coherence_scan")
        self.assertEqual(candidate["bigram_topk_guard"], 0.2)
        self.assertIn("biguard-0.2__biguardk-5", candidate["name"])
        self.assertIn("valstart-0.5", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-topk-guard") + 1],
            "0.2",
        )
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "0.5",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_wave_topk_middle_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-wave-topk-middle",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-wave-topk-middle",
        )
        self.assertIn("middle top-k guard", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], [corpus])
        self.assertEqual(manifest["architectures"], ["wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_topk_guards"], [0.3, 0.4])
        self.assertEqual(manifest["bigram_topk_guard_k"], 5)
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(manifest["compare_summary"]["limit"], 12)
        self.assertEqual(len(manifest["runs"]), 48)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 34
            and run["validation_start_fraction"] == 1.0
            and run["bigram_topk_guard"] == 0.4
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["example"], "modelzoo_llm_char_coherence_wave")
        self.assertIn("wave__feature-token-bigram", candidate["name"])
        self.assertIn("biguard-0.4__biguardk-5", candidate["name"])
        self.assertIn("valstart-1", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-topk-guard") + 1],
            "0.4",
        )
        self.assertEqual(
            candidate["command"][
                candidate["command"].index("--bigram-rank-guard-min-candidates") + 1
            ],
            "1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--memory") + 1],
            "12",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_wave_topk_schedule_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-wave-topk-schedule-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-wave-topk-schedule-confirm",
        )
        self.assertIn("validation-start-aware", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.5])
        self.assertEqual(
            manifest["bigram_topk_guard_schedule"],
            [
                {"validation_start_fraction": 0.0, "bigram_topk_guard": 0.5},
                {"validation_start_fraction": 0.5, "bigram_topk_guard": 0.2},
                {"validation_start_fraction": 1.0, "bigram_topk_guard": 0.3},
            ],
        )
        self.assertEqual(manifest["bigram_rank_guard_min_candidates"], [0, 1])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(len(manifest["runs"]), 24)

        val0 = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 34
            and run["validation_start_fraction"] == 0.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        val05 = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 34
            and run["validation_start_fraction"] == 0.5
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        val1 = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 34
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(val0["bigram_topk_guard"], 0.5)
        self.assertEqual(val05["bigram_topk_guard"], 0.2)
        self.assertEqual(val1["bigram_topk_guard"], 0.3)
        self.assertIn("biguard-0.5__biguardk-5", val0["name"])
        self.assertIn("biguard-0.2__biguardk-5", val05["name"])
        self.assertIn("biguard-0.3__biguardk-5", val1["name"])
        self.assertEqual(
            val0["command"][val0["command"].index("--bigram-topk-guard") + 1],
            "0.5",
        )
        self.assertEqual(
            val05["command"][val05["command"].index("--bigram-topk-guard") + 1],
            "0.2",
        )
        self.assertEqual(
            val1["command"][val1["command"].index("--bigram-topk-guard") + 1],
            "0.3",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_architecture_topk_schedule_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-architecture-topk-schedule-confirm",
        )
        self.assertIn("architecture-aware", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.5])
        self.assertEqual(manifest["bigram_topk_guard_schedule"], None)
        self.assertEqual(
            manifest["bigram_topk_guard_arch_schedule"],
            [
                {
                    "architecture": "*",
                    "validation_start_fraction": 0.0,
                    "bigram_topk_guard": 0.2,
                },
                {
                    "architecture": "*",
                    "validation_start_fraction": 0.5,
                    "bigram_topk_guard": 0.2,
                },
                {
                    "architecture": "*",
                    "validation_start_fraction": 1.0,
                    "bigram_topk_guard": 0.2,
                },
                {
                    "architecture": "wave",
                    "validation_start_fraction": 0.0,
                    "bigram_topk_guard": 0.5,
                },
                {
                    "architecture": "wave",
                    "validation_start_fraction": 1.0,
                    "bigram_topk_guard": 0.3,
                },
            ],
        )
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(len(manifest["runs"]), 72)

        scan_val1 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "scan"
            and run["seed"] == 34
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        wave_val0 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["validation_start_fraction"] == 0.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        wave_val05 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["validation_start_fraction"] == 0.5
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        wave_val1 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(scan_val1["bigram_topk_guard"], 0.2)
        self.assertEqual(wave_val0["bigram_topk_guard"], 0.5)
        self.assertEqual(wave_val05["bigram_topk_guard"], 0.2)
        self.assertEqual(wave_val1["bigram_topk_guard"], 0.3)
        self.assertIn("biguard-0.2__biguardk-5", scan_val1["name"])
        self.assertIn("biguard-0.5__biguardk-5", wave_val0["name"])
        self.assertIn("biguard-0.2__biguardk-5", wave_val05["name"])
        self.assertIn("biguard-0.3__biguardk-5", wave_val1["name"])
        self.assertEqual(
            wave_val0["command"][
                wave_val0["command"].index("--bigram-topk-guard") + 1
            ],
            "0.5",
        )
        self.assertEqual(
            wave_val1["command"][
                wave_val1["command"].index("--bigram-topk-guard") + 1
            ],
            "0.3",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_architecture_topk_schedule_scale_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-scale",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-architecture-topk-schedule-scale",
        )
        self.assertIn("larger char-LM shape", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["shape_grid"]["steps"], [16])
        self.assertEqual(manifest["shape_grid"]["embed_dims"], [12])
        self.assertEqual(manifest["shape_grid"]["hidden"], [24])
        self.assertEqual(manifest["training_grid"]["epochs"], [12])
        self.assertEqual(manifest["training_grid"]["batches"], [32])
        self.assertEqual(manifest["training_grid"]["eval_samples"], [192])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.5])
        self.assertEqual(len(manifest["runs"]), 36)
        wave_val1 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 13
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        scan_val0 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "scan"
            and run["seed"] == 13
            and run["validation_start_fraction"] == 0.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(wave_val1["bigram_topk_guard"], 0.3)
        self.assertEqual(scan_val0["bigram_topk_guard"], 0.2)
        self.assertEqual(wave_val1["command"][wave_val1["command"].index("--steps") + 1], "16")
        self.assertEqual(wave_val1["command"][wave_val1["command"].index("--hidden") + 1], "24")
        self.assertEqual(wave_val1["command"][wave_val1["command"].index("--memory") + 1], "16")
        self.assertEqual(
            wave_val1["command"][wave_val1["command"].index("--bigram-topk-guard") + 1],
            "0.3",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_architecture_topk_schedule_scale_confirm_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-scale-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-architecture-topk-schedule-scale-confirm",
        )
        self.assertIn("complementary seeds", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [21, 34])
        self.assertEqual(manifest["shape_grid"]["steps"], [16])
        self.assertEqual(manifest["shape_grid"]["embed_dims"], [12])
        self.assertEqual(manifest["shape_grid"]["hidden"], [24])
        self.assertEqual(manifest["training_grid"]["epochs"], [12])
        self.assertEqual(manifest["training_grid"]["batches"], [32])
        self.assertEqual(manifest["training_grid"]["eval_samples"], [192])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.5])
        self.assertEqual(len(manifest["runs"]), 36)
        wave_val0 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["validation_start_fraction"] == 0.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        lstm_val1 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "lstm"
            and run["seed"] == 21
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(wave_val0["bigram_topk_guard"], 0.5)
        self.assertEqual(lstm_val1["bigram_topk_guard"], 0.2)
        self.assertEqual(
            wave_val0["command"][wave_val0["command"].index("--bigram-topk-guard") + 1],
            "0.5",
        )
        self.assertEqual(wave_val0["command"][wave_val0["command"].index("--steps") + 1], "16")
        self.assertEqual(wave_val0["command"][wave_val0["command"].index("--hidden") + 1], "24")
        self.assertEqual(wave_val0["command"][wave_val0["command"].index("--memory") + 1], "16")

    def test_char_lm_sweep_hard_rank_band_adaptive_architecture_topk_schedule_scale_full_recipe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = "models/samples/spiral_corpus_en"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        corpus,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-scale-full",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-architecture-topk-schedule-scale-full",
        )
        self.assertIn("full four-seed", manifest["recipe_description"])
        self.assertEqual(
            manifest["recipe_defaults"][
                "compare_summary_fail_on_rank_min_promotion_decision"
            ],
            "no_rank_min_evidence,needs_tuning,partial_promote_needs_tuning",
        )
        self.assertEqual(
            manifest["compare_summary"]["fail_on_rank_min_promotion_decisions"],
            [
                "no_rank_min_evidence",
                "needs_tuning",
                "partial_promote_needs_tuning",
            ],
        )
        self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["shape_grid"]["steps"], [16])
        self.assertEqual(manifest["shape_grid"]["embed_dims"], [12])
        self.assertEqual(manifest["shape_grid"]["hidden"], [24])
        self.assertEqual(manifest["training_grid"]["epochs"], [12])
        self.assertEqual(manifest["training_grid"]["batches"], [32])
        self.assertEqual(manifest["training_grid"]["eval_samples"], [192])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.5])
        self.assertEqual(len(manifest["runs"]), 72)
        wave_val1 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "wave"
            and run["seed"] == 34
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        scan_val05 = next(
            run
            for run in manifest["runs"]
            if run["architecture"] == "scan"
            and run["seed"] == 21
            and run["validation_start_fraction"] == 0.5
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(wave_val1["bigram_topk_guard"], 0.3)
        self.assertEqual(scan_val05["bigram_topk_guard"], 0.2)
        self.assertEqual(wave_val1["command"][wave_val1["command"].index("--steps") + 1], "16")
        self.assertEqual(wave_val1["command"][wave_val1["command"].index("--hidden") + 1], "24")
        self.assertEqual(wave_val1["command"][wave_val1["command"].index("--memory") + 1], "16")
        self.assertEqual(
            wave_val1["command"][wave_val1["command"].index("--bigram-topk-guard") + 1],
            "0.3",
        )

    def test_char_lm_sweep_hard_rank_band_adaptive_architecture_topk_schedule_corpus_widen_recipes(
        self,
    ) -> None:
        cases = [
            (
                "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen",
                [7, 13],
                36,
                "widened positional corpus",
            ),
            (
                "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-confirm",
                [21, 34],
                36,
                "complementary seeds",
            ),
            (
                "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-full",
                [7, 13, 21, 34],
                72,
                "full four-seed",
            ),
        ]
        corpus_paths = [
            "models/samples/spiral_corpus_en",
            "models/samples/spiral_demo_en.txt",
        ]

        for recipe, seeds, expected_runs, description_fragment in cases:
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with contextlib.redirect_stdout(io.StringIO()):
                    code = run_char_lm_sweep.main(
                        [
                            *corpus_paths,
                            "--run-root",
                            str(root),
                            "--recipe",
                            recipe,
                            "--dry-run",
                            "--no-wgpu-preflight",
                        ]
                    )
                manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertEqual(manifest["recipe"], recipe)
            self.assertIn(description_fragment, manifest["recipe_description"])
            if recipe.endswith("-full"):
                self.assertEqual(
                    manifest["recipe_defaults"][
                        "compare_summary_fail_on_rank_min_promotion_decision"
                    ],
                    "no_rank_min_evidence,needs_tuning,partial_promote_needs_tuning",
                )
            else:
                self.assertNotIn(
                    "compare_summary_fail_on_rank_min_promotion_decision",
                    manifest["recipe_defaults"],
                )
            self.assertEqual(manifest["data_paths"], corpus_paths)
            self.assertEqual(manifest["architectures"], ["lstm", "scan", "wave"])
            self.assertEqual(manifest["seeds"], seeds)
            self.assertEqual(manifest["shape_grid"]["steps"], [16])
            self.assertEqual(manifest["shape_grid"]["embed_dims"], [12])
            self.assertEqual(manifest["shape_grid"]["hidden"], [24])
            self.assertEqual(manifest["training_grid"]["epochs"], [12])
            self.assertEqual(manifest["training_grid"]["batches"], [32])
            self.assertEqual(manifest["training_grid"]["eval_samples"], [192])
            self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.5])
            self.assertEqual(len(manifest["runs"]), expected_runs)
            candidate = next(
                run
                for run in manifest["runs"]
                if run["architecture"] == "wave"
                and run["seed"] == seeds[-1]
                and run["validation_start_fraction"] == 1.0
                and run["bigram_rank_guard_min_candidates"] == 1
            )
            lstm_val_half = next(
                run
                for run in manifest["runs"]
                if run["architecture"] == "lstm"
                and run["seed"] == seeds[-1]
                and run["validation_start_fraction"] == 0.5
                and run["bigram_rank_guard_min_candidates"] == 1
            )
            scan_val0 = next(
                run
                for run in manifest["runs"]
                if run["architecture"] == "scan"
                and run["seed"] == seeds[-1]
                and run["validation_start_fraction"] == 0.0
                and run["bigram_rank_guard_min_candidates"] == 1
            )
            scan_val_half = next(
                run
                for run in manifest["runs"]
                if run["architecture"] == "scan"
                and run["seed"] == seeds[-1]
                and run["validation_start_fraction"] == 0.5
                and run["bigram_rank_guard_min_candidates"] == 1
            )
            wave_val_half = next(
                run
                for run in manifest["runs"]
                if run["architecture"] == "wave"
                and run["seed"] == seeds[-1]
                and run["validation_start_fraction"] == 0.5
                and run["bigram_rank_guard_min_candidates"] == 1
            )
            command = candidate["command"]
            example_separator = command.index("--")
            backend_flag = command.index("--backend")
            self.assertEqual(
                command[example_separator + 1 : example_separator + 1 + len(corpus_paths)],
                corpus_paths,
            )
            self.assertLess(example_separator + len(corpus_paths), backend_flag)
            self.assertEqual(candidate["bigram_topk_guard"], 0.3)
            self.assertEqual(command[command.index("--steps") + 1], "16")
            self.assertEqual(command[command.index("--hidden") + 1], "24")
            self.assertEqual(command[command.index("--memory") + 1], "16")
            self.assertEqual(
                command[command.index("--bigram-topk-guard") + 1],
                "0.3",
            )
            lstm_command = lstm_val_half["command"]
            self.assertEqual(lstm_val_half["bigram_topk_guard"], 0.5)
            self.assertEqual(
                lstm_command[lstm_command.index("--bigram-topk-guard") + 1],
                "0.5",
            )
            for promoted_candidate in (scan_val0, scan_val_half, wave_val_half):
                promoted_command = promoted_candidate["command"]
                self.assertEqual(promoted_candidate["bigram_topk_guard"], 0.5)
                self.assertEqual(
                    promoted_command[
                        promoted_command.index("--bigram-topk-guard") + 1
                    ],
                    "0.5",
                )

    def test_char_lm_sweep_corpus_widen_hotspot_recipe_targets_lstm_valstart_half(
        self,
    ) -> None:
        corpus_paths = [
            "models/samples/spiral_corpus_en",
            "models/samples/spiral_demo_en.txt",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        *corpus_paths,
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-hotspot",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["recipe"],
            "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen-hotspot",
        )
        self.assertIn("LSTM val-start=0.5 hotspot", manifest["recipe_description"])
        self.assertEqual(manifest["data_paths"], corpus_paths)
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["seeds"], [7, 13, 21, 34])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.3, 0.4, 0.5])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.5],
        )
        self.assertEqual(len(manifest["runs"]), 32)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 21
            and run["bigram_topk_guard"] == 0.4
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["architecture"], "lstm")
        self.assertEqual(candidate["validation_start_fraction"], 0.5)
        self.assertEqual(candidate["bigram_rank_guard_band"], 0.003)
        command = candidate["command"]
        example_separator = command.index("--")
        self.assertEqual(
            command[example_separator + 1 : example_separator + 1 + len(corpus_paths)],
            corpus_paths,
        )
        self.assertEqual(command[command.index("--bigram-topk-guard") + 1], "0.4")

    def test_char_lm_sweep_architecture_topk_schedule_allows_subset_smoke(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_corpus_en",
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen",
                        "--architectures",
                        "lstm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["bigram_topk_guards"], [0.2, 0.5])
        self.assertEqual(len(manifest["runs"]), 12)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13
            and run["validation_start_fraction"] == 1.0
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(candidate["architecture"], "lstm")
        self.assertEqual(candidate["bigram_topk_guard"], 0.2)
        self.assertNotIn("--memory", candidate["command"])
        val_half = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13
            and run["validation_start_fraction"] == 0.5
            and run["bigram_rank_guard_min_candidates"] == 1
        )
        self.assertEqual(val_half["bigram_topk_guard"], 0.5)
        self.assertEqual(
            val_half["command"][val_half["command"].index("--bigram-topk-guard") + 1],
            "0.5",
        )

    def test_char_lm_sweep_max_new_runs_limits_dry_run_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_corpus_en",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-band-adaptive-architecture-topk-schedule-corpus-widen",
                        "--architectures",
                        "lstm",
                        "--seeds",
                        "7,13",
                        "--dry-run",
                        "--no-wgpu-preflight",
                        "--max-new-runs",
                        "1",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["planned_runs"], 12)
        self.assertEqual(manifest["new_runs_started"], 1)
        self.assertTrue(manifest["run_limit_reached"])
        self.assertEqual(len(manifest["runs"]), 1)
        self.assertFalse(manifest["runs"][0]["skipped"])
        self.assertEqual(manifest["next_run_after_limit"]["index"], 2)

    def test_char_lm_sweep_skip_existing_max_new_runs_collects_resume_chunk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch resume chunk smoke", encoding="utf-8")
            existing = (
                root
                / "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7"
            )
            existing.mkdir(parents=True)
            (existing / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {"mean_nll": 2.9},
                        "validation_nll_delta": -0.1,
                    }
                ),
                encoding="utf-8",
            )
            (existing / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(8,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7,13,21",
                        "--skip-existing",
                        "--dry-run",
                        "--no-wgpu-preflight",
                        "--max-new-runs",
                        "1",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["planned_runs"], 3)
        self.assertEqual(manifest["new_runs_started"], 1)
        self.assertTrue(manifest["run_limit_reached"])
        self.assertEqual(len(manifest["runs"]), 2)
        self.assertTrue(manifest["runs"][0]["skipped"])
        self.assertEqual(manifest["runs"][0]["seed"], 7)
        self.assertFalse(manifest["runs"][1]["skipped"])
        self.assertEqual(manifest["runs"][1]["seed"], 13)
        self.assertEqual(manifest["next_run_after_limit"]["index"], 3)
        self.assertEqual(manifest["next_run_after_limit"]["seed"], 21)

    def test_char_lm_sweep_hard_rank_guard_confirm_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-guard-confirm",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-guard-confirm")
        self.assertIn("confirm the local rank-debt", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guards"], [0.0, 0.05, 0.1])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.05])
        self.assertEqual(len(manifest["runs"]), 6)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13
            and run["bigram_rank_guard"] == 0.1
            and run["bigram_rank_guard_margin"] == 0.05
        )
        self.assertIn("bigrank-0.1__bigrankm-0.05", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-rank-guard") + 1],
            "0.1",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-rank-guard-margin") + 1],
            "0.05",
        )

    def test_char_lm_sweep_hard_rank_guard_instability_map_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-rank-guard-instability-map",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-rank-guard-instability-map")
        self.assertIn("validation start positions", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guards"], [0.0, 0.05, 0.1])
        self.assertEqual(manifest["bigram_rank_guard_margins"], [0.05])
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(len(manifest["runs"]), 18)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13
            and run["validation_start_fraction"] == 0.5
            and run["bigram_rank_guard"] == 0.1
        )
        self.assertIn("bigrank-0.1__bigrankm-0.05", candidate["name"])
        self.assertIn("valstart-0.5", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--val-start-fraction") + 1],
            "0.5",
        )
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-rank-guard") + 1],
            "0.1",
        )

    def test_char_lm_sweep_hard_soft_guard_local_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-soft-guard-local",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-soft-guard-local")
        self.assertIn("softer full-bigram", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guards"], [0.0])
        self.assertEqual(manifest["bigram_soft_guards"], [0.0, 0.01, 0.05, 0.1])
        self.assertEqual(len(manifest["runs"]), 8)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13 and run["bigram_soft_guard"] == 0.05
        )
        self.assertIn("bigsoft-0.05", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-soft-guard") + 1],
            "0.05",
        )

    def test_char_lm_sweep_hard_soft_guard_micro_local_recipe_expands_probe_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--recipe",
                        "hard-soft-guard-micro-local",
                        "--dry-run",
                        "--no-wgpu-preflight",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["recipe"], "hard-soft-guard-micro-local")
        self.assertIn("very small full-bigram", manifest["recipe_description"])
        self.assertEqual(manifest["architectures"], ["lstm"])
        self.assertEqual(manifest["head_priors"], ["learned-bigram"])
        self.assertEqual(manifest["seeds"], [7, 13])
        self.assertEqual(manifest["bigram_topk_guards"], [0.1])
        self.assertEqual(manifest["bigram_rank_guards"], [0.0])
        self.assertEqual(
            manifest["bigram_soft_guards"],
            [0.0, 0.001, 0.003, 0.005, 0.01],
        )
        self.assertEqual(len(manifest["runs"]), 10)
        candidate = next(
            run
            for run in manifest["runs"]
            if run["seed"] == 13 and run["bigram_soft_guard"] == 0.003
        )
        self.assertIn("bigsoft-0.003", candidate["name"])
        self.assertEqual(
            candidate["command"][candidate["command"].index("--bigram-soft-guard") + 1],
            "0.003",
        )

    def test_char_lm_sweep_shape_grid_dry_run_writes_shape_manifest_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune,lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--step-values",
                        "4,6",
                        "--embed-dim",
                        "4",
                        "--hidden-values",
                        "8,12",
                        "--epochs",
                        "1",
                        "--batches",
                        "1",
                        "--batch",
                        "2",
                        "--eval-samples",
                        "4",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["shape_grid"]["steps"], [4, 6])
        self.assertEqual(manifest["shape_grid"]["embed_dims"], [4])
        self.assertEqual(manifest["shape_grid"]["hidden"], [8, 12])
        self.assertEqual(len(manifest["runs"]), 8)
        names = [run["name"] for run in manifest["runs"]]
        self.assertIn(
            "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__"
            "steps-6__embed-4__hidden-12__seed-7",
            names,
        )

    def test_char_lm_sweep_head_residual_grid_dry_run_writes_manifest_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "none,learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--head-residual-scale-values",
                        "0.5,2",
                        "--epochs",
                        "1",
                        "--batches",
                        "1",
                        "--batch",
                        "2",
                        "--eval-samples",
                        "4",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["head_residual_scales"], [0.5, 2.0])
        self.assertEqual(len(manifest["runs"]), 4)
        names = [run["name"] for run in manifest["runs"]]
        self.assertIn(
            "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__"
            "headresid-2__seed-7",
            names,
        )
        learned_scale_two = next(name for name in names if "headresid-2" in name)
        learned_run = next(run for run in manifest["runs"] if run["name"] == learned_scale_two)
        self.assertEqual(learned_run["head_residual_scale"], 2.0)
        self.assertIn("--head-residual-scale", learned_run["command"])

    def test_char_lm_sweep_bigram_head_priors_dry_run_writes_names_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "bigram,learned-bigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--epochs",
                        "1",
                        "--batches",
                        "1",
                        "--batch",
                        "2",
                        "--eval-samples",
                        "4",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["head_priors"], ["bigram", "learned-bigram"])
        self.assertEqual(len(manifest["runs"]), 2)
        names = [run["name"] for run in manifest["runs"]]
        self.assertIn(
            "lstm__feature-token-bigram__head-bigram__backend-cpu__seed-7",
            names,
        )
        learned_run = next(
            run for run in manifest["runs"] if run["head_prior"] == "learned-bigram"
        )
        head_prior_index = learned_run["command"].index("--head-prior")
        self.assertEqual(learned_run["command"][head_prior_index + 1], "learned-bigram")

    def test_char_lm_sweep_training_grid_dry_run_writes_manifest_names_and_commands(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--epoch-values",
                        "1,4",
                        "--batches-values",
                        "2,8",
                        "--batch",
                        "3",
                        "--eval-samples",
                        "4",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["training_grid"]["epochs"], [1, 4])
        self.assertEqual(manifest["training_grid"]["batches"], [2, 8])
        self.assertEqual(len(manifest["runs"]), 4)
        names = [run["name"] for run in manifest["runs"]]
        target_name = (
            "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__"
            "epochs-4__batches-8__seed-7"
        )
        self.assertIn(target_name, names)
        target_run = next(run for run in manifest["runs"] if run["name"] == target_name)
        self.assertEqual(target_run["epochs"], 4)
        self.assertEqual(target_run["batches"], 8)
        self.assertEqual(target_run["batch"], 3)
        epochs_index = target_run["command"].index("--epochs")
        batches_index = target_run["command"].index("--batches")
        batch_index = target_run["command"].index("--batch")
        self.assertEqual(target_run["command"][epochs_index + 1], "4")
        self.assertEqual(target_run["command"][batches_index + 1], "8")
        self.assertEqual(target_run["command"][batch_index + 1], "3")

    def test_char_lm_sweep_eval_samples_grid_dry_run_writes_manifest_names_and_commands(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--epochs",
                        "1",
                        "--batches",
                        "2",
                        "--batch",
                        "3",
                        "--eval-samples-values",
                        "32,64",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["training_grid"]["eval_samples"], [32, 64])
        self.assertEqual(len(manifest["runs"]), 2)
        names = [run["name"] for run in manifest["runs"]]
        target_name = (
            "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__"
            "eval-64__seed-7"
        )
        self.assertIn(target_name, names)
        target_run = next(run for run in manifest["runs"] if run["name"] == target_name)
        self.assertEqual(target_run["eval_samples"], 64)
        eval_index = target_run["command"].index("--eval-samples")
        self.assertEqual(target_run["command"][eval_index + 1], "64")

    def test_char_lm_sweep_lr_grid_dry_run_writes_manifest_names_and_commands(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--lr-values",
                        "0.01,0.02",
                        "--epochs",
                        "1",
                        "--batches",
                        "2",
                        "--batch",
                        "3",
                        "--eval-samples",
                        "4",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(manifest["learning_rates"], [0.01, 0.02])
        self.assertEqual(len(manifest["runs"]), 2)
        names = [run["name"] for run in manifest["runs"]]
        target_name = (
            "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__"
            "lr-0.02__seed-7"
        )
        self.assertIn(target_name, names)
        target_run = next(run for run in manifest["runs"] if run["name"] == target_name)
        self.assertEqual(target_run["lr"], 0.02)
        lr_index = target_run["command"].index("--lr")
        self.assertEqual(target_run["command"][lr_index + 1], "0.02")

    def test_char_lm_sweep_val_start_grid_dry_run_writes_manifest_names_and_commands(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        "models/samples/spiral_demo_en.txt",
                        "--run-root",
                        str(root),
                        "--architectures",
                        "lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--epochs",
                        "1",
                        "--batches",
                        "2",
                        "--batch",
                        "3",
                        "--eval-samples",
                        "4",
                        "--val-start-values",
                        "0,0.5,1",
                        "--gen",
                        "0",
                        "--dry-run",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            manifest["training_grid"]["validation_start_fractions"],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(len(manifest["runs"]), 3)
        names = [run["name"] for run in manifest["runs"]]
        target_name = (
            "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__"
            "valstart-0.5__seed-7"
        )
        self.assertIn(target_name, names)
        target_run = next(run for run in manifest["runs"] if run["name"] == target_name)
        self.assertEqual(target_run["validation_start_fraction"], 0.5)
        val_start_index = target_run["command"].index("--val-start-fraction")
        self.assertEqual(target_run["command"][val_start_index + 1], "0.5")

    def test_char_lm_compare_surfaces_recurrent_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_lstm",
                        "backend": "cpu",
                        "recurrent": "lstm",
                        "head_prior": "learned-unigram",
                        "head_residual_scale": 2.0,
                        "bigram_topk_guard": 0.05,
                        "bigram_topk_guard_k": 3,
                        "bigram_rank_guard": 0.025,
                        "bigram_rank_guard_margin": 0.01,
                        "bigram_rank_guard_band": 0.003,
                        "bigram_rank_guard_min_candidates": 2,
                        "bigram_rank_guard_coverage": {
                            "windows": 42,
                            "min_candidates": 2,
                            "mean_unbounded_candidates": 4.0,
                            "mean_band_candidates": 2.5,
                            "mean_guarded_candidates": 2.0,
                            "mean_effective_rank_band": 0.004,
                            "adaptive_fill_ratio": 0.25,
                            "mean_adaptive_filled_candidates": 0.5,
                            "zero_guarded_candidate_ratio": 0.1,
                            "mean_guarded_candidate_mass": 0.2,
                            "mean_band_to_unbounded_candidate_ratio": 0.625,
                            "mean_guarded_to_unbounded_topk_ratio": 0.5,
                        },
                        "validation_start_fraction_requested": 0.5,
                        "validation_start_fraction_actual": 0.5,
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {
                            "windows": 12,
                            "mean_nll": 2.5,
                            "mean_unigram_target_rank": 7.0,
                            "mean_target_rank_debt_vs_unigram": -1.0,
                            "mean_target_logprob_lift_vs_bigram": 0.3,
                            "mean_target_rank_lift_vs_bigram": 1.5,
                            "mean_bigram_target_rank": 5.5,
                            "mean_target_rank_debt_vs_bigram": -1.5,
                            "mean_kl_to_bigram": 0.02,
                            "mean_top5_overlap_with_bigram": 0.4,
                        },
                        "validation_nll_delta": -0.5,
                        "unigram_validation": {"windows": 12, "mean_nll": 2.7},
                        "bigram_validation": {"windows": 12, "mean_nll": 2.4},
                        "final_vs_unigram_nll_delta": -0.2,
                        "final_vs_bigram_nll_delta": 0.1,
                    }
                ),
                encoding="utf-8",
            )

            row, _ = compare_char_lm_runs.row_for(str(run_dir))
            table = compare_char_lm_runs.markdown_table([row])

        self.assertEqual(row["recurrent"], "lstm")
        self.assertEqual(row["mode"], "embedding(4,token-bigram)")
        self.assertEqual(row["head_resid"], "2.0000")
        self.assertEqual(row["bigram_guard"], "0.0500")
        self.assertEqual(row["bigram_guard_k"], "3")
        self.assertEqual(row["bigram_rank_guard"], "0.0250")
        self.assertEqual(row["bigram_rank_margin"], "0.0100")
        self.assertEqual(row["bigram_rank_band"], "0.0030")
        self.assertEqual(row["bigram_rank_min"], "2")
        self.assertEqual(row["rank_cov_windows"], "42")
        self.assertEqual(row["rank_cov_unbounded"], "4.0000")
        self.assertEqual(row["rank_cov_band"], "2.5000")
        self.assertEqual(row["rank_cov_min"], "2")
        self.assertEqual(row["rank_cov_guarded"], "2.0000")
        self.assertEqual(row["rank_cov_effective_band"], "0.0040")
        self.assertEqual(row["rank_cov_adaptive_fill_ratio"], "0.2500")
        self.assertEqual(row["rank_cov_filled"], "0.5000")
        self.assertEqual(row["rank_cov_zero_ratio"], "0.1000")
        self.assertEqual(row["rank_cov_mass"], "0.2000")
        self.assertEqual(row["rank_cov_band_ratio"], "0.6250")
        self.assertEqual(row["rank_cov_topk_ratio"], "0.5000")
        self.assertEqual(row["val_start"], "0.5000")
        self.assertEqual(row["val_start_actual"], "0.5000")
        self.assertEqual(row["final_windows"], "12")
        self.assertEqual(row["unigram_windows"], "12")
        self.assertEqual(row["bigram_windows"], "12")
        self.assertEqual(row["bigram_nll"], "2.4000")
        self.assertEqual(row["final_vs_bigram"], "0.1000")
        self.assertEqual(row["final_unigram_target_rank"], "7.00")
        self.assertEqual(row["final_unigram_target_rank_raw"], "7.00000000")
        self.assertEqual(row["final_unigram_rank_debt"], "-1.00")
        self.assertEqual(row["final_unigram_rank_debt_raw"], "-1.00000000")
        self.assertEqual(row["final_bigram_logprob_lift"], "0.3000")
        self.assertEqual(row["final_bigram_rank_lift"], "1.50")
        self.assertEqual(row["final_bigram_rank_lift_raw"], "1.50000000")
        self.assertEqual(row["final_bigram_target_rank"], "5.50")
        self.assertEqual(row["final_bigram_target_rank_raw"], "5.50000000")
        self.assertEqual(row["final_bigram_rank_debt"], "-1.50")
        self.assertEqual(row["final_bigram_rank_debt_raw"], "-1.50000000")
        self.assertEqual(row["final_kl_bigram"], "0.0200")
        self.assertEqual(row["final_top5_bigram_overlap"], "40.00%")
        self.assertEqual(row["final_top5_bigram_overlap_raw"], "40.00000000")
        self.assertIn("| run | arch | backend | recurrent |", table)
        self.assertIn("embedding(4,token-bigram)", table)
        self.assertIn("head_resid", table)
        self.assertIn("bigram_guard", table)
        self.assertIn("bigram_rank_guard", table)
        self.assertIn("bigram_rank_band", table)
        self.assertIn("bigram_rank_min", table)
        self.assertIn("rank_cov_guarded", table)
        self.assertIn("rank_cov_effective_band", table)
        self.assertIn("val_start", table)
        self.assertIn("bigram_nll", table)
        self.assertIn("final_bigram_logprob_lift", table)
        self.assertIn("final_bigram_rank_debt", table)

    def test_char_lm_sweep_render_compare_writes_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "lstm-run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_lstm",
                        "backend": "cpu",
                        "recurrent": "lstm",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {"windows": 8, "mean_nll": 2.5},
                        "unigram_validation": {"windows": 8, "mean_nll": 2.7},
                        "bigram_validation": {"windows": 8, "mean_nll": 2.4},
                        "validation_nll_delta": -0.5,
                        "final_vs_unigram_nll_delta": -0.2,
                        "final_vs_bigram_nll_delta": 0.1,
                        "best_vs_unigram_nll_delta": -0.2,
                        "best_vs_bigram_nll_delta": 0.1,
                    }
                ),
                encoding="utf-8",
            )

            output = run_char_lm_sweep.render_compare([run_dir], root, curves=False)
            summary_output = run_char_lm_sweep.render_compare_summary(root)
            compare_json = json.loads((root / "compare.json").read_text(encoding="utf-8"))
            compare_summary = json.loads(
                (root / "compare_summary.json").read_text(encoding="utf-8")
            )
            compare_summary_md_exists = (root / "compare_summary.md").exists()

        self.assertIsNotNone(output)
        self.assertIsNotNone(summary_output)
        self.assertEqual(compare_json["schema"], "st.char_lm.compare.v1")
        self.assertEqual(compare_json["runs"][0]["recurrent"], "lstm")
        self.assertEqual(compare_json["runs"][0]["final_windows"], "8")
        self.assertEqual(compare_json["aggregate_runs"][0]["final_windows_mean"], "8.0000")
        self.assertEqual(compare_json["aggregate_runs"][0]["bigram_windows_mean"], "8.0000")
        self.assertEqual(compare_json["aggregate_runs"][0]["final_nll_mean"], "2.5000")
        self.assertEqual(compare_json["aggregate_runs"][0]["unigram_nll_mean"], "2.7000")
        self.assertEqual(compare_json["aggregate_runs"][0]["bigram_nll_mean"], "2.4000")
        self.assertEqual(compare_json["aggregate_runs"][0]["best_vs_bigram_mean"], "0.1000")
        self.assertEqual(compare_json["aggregate_runs"][0]["route_status"], "-")
        self.assertEqual(compare_json["top_aggregate_runs"][0]["arch"], "llm_char_lstm")
        self.assertEqual(compare_summary["schema"], "st.char_lm.compare_summary.v1")
        self.assertEqual(compare_summary["rows"][0]["arch"], "llm_char_lstm")
        self.assertEqual(compare_summary["rows"][0]["final_windows_mean"], "8.0000")
        self.assertEqual(compare_summary["rows"][0]["bigram_nll_mean"], "2.4000")
        self.assertEqual(compare_summary["rows"][0]["best_vs_bigram_mean"], "0.1000")
        difficulty = compare_summary["baseline_difficulty_rows"]
        self.assertEqual(len(difficulty), 1)
        self.assertEqual(
            difficulty[0]["bigram_baseline_status"],
            "bigram_stronger_than_unigram",
        )
        self.assertEqual(difficulty[0]["model_vs_bigram_status"], "model_lags_bigram")
        self.assertTrue(compare_summary_md_exists)

    def test_char_lm_sweep_render_compare_summary_merges_extra_inputs(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "2.0000",
                    "best_nll_mean": "2.0000",
                }
            ],
        }
        extra_payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_coherence_scan",
                    "recurrent": "-",
                    "backend": "cpu",
                    "final_nll_mean": "1.9000",
                    "best_nll_mean": "1.9000",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra_root = root / "previous"
            extra_root.mkdir()
            (root / "compare.json").write_text(json.dumps(payload), encoding="utf-8")
            (extra_root / "compare.json").write_text(
                json.dumps(extra_payload),
                encoding="utf-8",
            )
            output = run_char_lm_sweep.render_compare_summary(
                root,
                options=run_char_lm_sweep.CompareSummaryOptions(
                    limit=8,
                    route_clean_only=False,
                    prefer_clean_route=True,
                    extra_compare_paths=(extra_root,),
                    merge_evidence_sources=True,
                ),
            )
            compare_summary = json.loads(
                (root / "compare_summary.json").read_text(encoding="utf-8")
            )
            command_script = root / "compare_summary_command.sh"
            command_script_text = command_script.read_text(encoding="utf-8")
            command_script_is_executable = bool(command_script.stat().st_mode & 0o111)

        self.assertIsNotNone(output)
        self.assertTrue(compare_summary["merge_evidence_sources"])
        self.assertEqual(len(compare_summary["sources"]), 2)
        self.assertIn(str(root / "compare.json"), compare_summary["sources"])
        self.assertIn(str(extra_root / "compare.json"), compare_summary["sources"])
        self.assertTrue(command_script_is_executable)
        self.assertIn(f"cd {run_char_lm_sweep.REPO_ROOT}", command_script_text)
        self.assertIn("compare_summary.md", command_script_text)
        self.assertIn("compare_summary.error.log", command_script_text)
        self.assertIn("--merge-evidence-sources", command_script_text)
        self.assertIn(str(extra_root), command_script_text)

    def test_char_lm_sweep_render_compare_summary_honors_route_gate_options(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "wgpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "no suitable WGPU adapter:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.1000",
                    "best_nll_mean": "1.1000",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "compare.json").write_text(json.dumps(payload), encoding="utf-8")
            output = run_char_lm_sweep.render_compare_summary(
                root,
                options=run_char_lm_sweep.CompareSummaryOptions(
                    limit=8,
                    route_clean_only=True,
                    prefer_clean_route=False,
                    fail_on_route_statuses=("scan_fallback",),
                ),
            )
            compare_summary = json.loads(
                (root / "compare_summary.json").read_text(encoding="utf-8")
            )
            error_text = (root / "compare_summary.error.log").read_text(encoding="utf-8")

        self.assertIsNone(output)
        self.assertEqual(compare_summary["rows"][0]["route_status"], "clean_route")
        self.assertEqual(compare_summary["route_status_counts"]["scan_fallback"], 1)
        self.assertEqual(compare_summary["selected_route_status_counts"]["scan_fallback"], 0)
        self.assertTrue(compare_summary["route_status_gate"]["failed"])
        self.assertEqual(
            compare_summary["route_status_gate"]["failures"],
            {"scan_fallback": 1},
        )
        self.assertIn("route status gate failed", error_text)

    def test_char_lm_sweep_render_compare_summary_honors_route_debt_gate_options(
        self,
    ) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_wave",
                    "recurrent": "wave",
                    "backend": "cpu",
                    "head_prior": "none",
                    "final_nll_mean": "2.0000",
                    "best_nll_mean": "2.0000",
                    "coherence_route_debt_mean": "10.0000",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "compare.json").write_text(json.dumps(payload), encoding="utf-8")
            output = run_char_lm_sweep.render_compare_summary(
                root,
                options=run_char_lm_sweep.CompareSummaryOptions(
                    limit=8,
                    route_clean_only=False,
                    prefer_clean_route=True,
                    fail_on_route_debt_decisions=("no_route_debt_recommendation",),
                ),
            )
            compare_summary = json.loads(
                (root / "compare_summary.json").read_text(encoding="utf-8")
            )
            command_script_text = (root / "compare_summary_command.sh").read_text(
                encoding="utf-8"
            )
            error_text = (root / "compare_summary.error.log").read_text(
                encoding="utf-8"
            )

        self.assertIsNone(output)
        route_debt_summary = compare_summary["route_debt_recommendation_summary"]
        self.assertEqual(route_debt_summary["decision"], "no_route_debt_recommendation")
        self.assertEqual(route_debt_summary["failed"], "true")
        self.assertEqual(
            route_debt_summary["fail_on_decisions"],
            "no_route_debt_recommendation",
        )
        self.assertIn("--fail-on-route-debt-decision", command_script_text)
        self.assertIn("route-debt decision gate failed", error_text)

    def test_char_lm_sweep_route_gate_failure_keeps_summary_paths_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch route gate smoke", encoding="utf-8")
            run_dir = (
                root
                / "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {"mean_nll": 2.9},
                        "validation_nll_delta": -0.1,
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--skip-existing",
                        "--compare-summary-fail-on-route-status",
                        "no_scan_route",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
            compare_summary_path_exists = Path(manifest["compare_summary_path"]).exists()
            compare_summary_json_path_exists = Path(
                manifest["compare_summary_json_path"]
            ).exists()
            compare_summary_error_path_exists = Path(
                manifest["compare_summary_error_path"]
            ).exists()

        self.assertEqual(code, 1)
        self.assertTrue(manifest["failed"])
        self.assertTrue(manifest["compare_summary_failed"])
        self.assertEqual(manifest["compare_summary_route_status_counts"]["no_scan_route"], 1)
        self.assertTrue(manifest["compare_summary_route_status_gate"]["failed"])
        self.assertEqual(
            manifest["compare_summary_route_status_gate"]["failures"],
            {"no_scan_route": 1},
        )
        self.assertTrue(compare_summary_path_exists)
        self.assertTrue(compare_summary_json_path_exists)
        self.assertTrue(compare_summary_error_path_exists)

    def test_char_lm_sweep_extra_compare_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch missing extra compare smoke", encoding="utf-8")
            run_dir = (
                root
                / "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {"mean_nll": 2.9},
                        "validation_nll_delta": -0.1,
                    }
                ),
                encoding="utf-8",
            )
            missing_extra = root / "missing-previous"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--skip-existing",
                        "--compare-summary-extra-compare-json",
                        str(missing_extra),
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
            error_path = Path(manifest["compare_summary_error_path"])
            error_text = error_path.read_text(encoding="utf-8")

        self.assertEqual(code, 1)
        self.assertTrue(manifest["failed"])
        self.assertTrue(manifest["compare_summary_failed"])
        self.assertEqual(
            manifest["compare_summary"]["extra_compare_paths"],
            [str(missing_extra)],
        )
        self.assertIn("missing-previous", error_text)

    def test_char_lm_sweep_extra_compare_sources_are_copied_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch extra compare smoke", encoding="utf-8")
            run_dir = (
                root
                / "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {"mean_nll": 2.9},
                        "validation_nll_delta": -0.1,
                    }
                ),
                encoding="utf-8",
            )
            extra_root = root / "previous"
            extra_root.mkdir()
            (extra_root / "compare.json").write_text(
                json.dumps(
                    {
                        "schema": "st.char_lm.compare.v1",
                        "aggregate_runs": [
                            {
                                "arch": "llm_char_lstm",
                                "recurrent": "lstm",
                                "backend": "cpu",
                                "final_nll_mean": "2.8000",
                                "best_nll_mean": "2.8000",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--skip-existing",
                        "--compare-summary-extra-compare-json",
                        str(extra_root),
                        "--compare-summary-merge-evidence-sources",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
            command_script_path = Path(manifest["compare_summary_command_path"])
            command_script_text = command_script_path.read_text(encoding="utf-8")
            command_script_is_executable = bool(
                command_script_path.stat().st_mode & 0o111
            )

        self.assertEqual(code, 0)
        self.assertFalse(manifest["failed"])
        self.assertFalse(manifest["compare_summary_failed"])
        self.assertEqual(
            manifest["compare_summary"]["command_script_path"],
            str(command_script_path),
        )
        self.assertEqual(
            manifest["compare_summary"]["command_cwd"],
            str(run_char_lm_sweep.REPO_ROOT),
        )
        self.assertEqual(
            manifest["compare_summary_command_cwd"],
            str(run_char_lm_sweep.REPO_ROOT),
        )
        self.assertTrue(command_script_is_executable)
        self.assertIn(f"cd {run_char_lm_sweep.REPO_ROOT}", command_script_text)
        self.assertIn("compare_summary.md", command_script_text)
        self.assertIn("compare_summary.error.log", command_script_text)
        self.assertIn("--merge-evidence-sources", command_script_text)
        self.assertIn(str(extra_root), command_script_text)
        self.assertTrue(manifest["compare_summary_merge_evidence_sources"])
        self.assertEqual(len(manifest["compare_summary_sources"]), 2)
        resolved_sources = {
            str(Path(source).resolve()) for source in manifest["compare_summary_sources"]
        }
        self.assertIn(str((root / "compare.json").resolve()), resolved_sources)
        self.assertIn(
            str((extra_root / "compare.json").resolve()),
            resolved_sources,
        )

    def test_char_lm_sweep_manifest_keeps_paired_recurrent_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch paired recurrent smoke", encoding="utf-8")
            run_payloads = [
                (
                    "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7",
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    },
                    100.0,
                    200.0,
                ),
                (
                    "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7",
                    {
                        "arch": "llm_char_lstm",
                        "backend": "cpu",
                        "recurrent": "lstm",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    },
                    50.0,
                    100.0,
                ),
            ]
            for name, run_payload, trace_ms, cpu_debt in run_payloads:
                run_dir = root / name
                run_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text(
                    json.dumps(run_payload),
                    encoding="utf-8",
                )
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "initial_validation": {"mean_nll": 3.0},
                            "final_validation": {"mean_nll": 2.9},
                            "validation_nll_delta": -0.1,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "trainer_trace_summary.json").write_text(
                    json.dumps(
                        {
                            "metrics": {
                                "step_time_ms": {
                                    "last": trace_ms,
                                    "mean": trace_ms,
                                    "max": trace_ms,
                                },
                                "tensor_ops_total": {"last": cpu_debt},
                                "tensor_op_backend_matmul_cpu": {"last": cpu_debt},
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune,lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--skip-existing",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        pairs = manifest["compare_summary_paired_recurrent_deltas"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_recurrent"], "lstm")
        self.assertEqual(pairs[0]["baseline_recurrent"], "spiral")
        recommendations = manifest["compare_summary_paired_recurrent_recommendations"]
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(
            recommendations[0]["recommendation"],
            "quality_neutral_cost_improved",
        )
        self.assertEqual(manifest["compare_summary_baseline_difficulty_rows"], [])

    def test_char_lm_sweep_manifest_keeps_rank_min_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch rank min promotion", encoding="utf-8")

            def write_run(seed: str, rank_min: int, debt: float) -> None:
                run_dir = root / (
                    "lstm__feature-token-bigram__head-learned-bigram__backend-cpu"
                    "__biguard-0.2__biguardk-5__bigrank-0.1__bigrankm-0.05"
                    f"__bigrankband-0.003__bigrankmin-{rank_min}__seed-{seed}"
                )
                run_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text(
                    json.dumps(
                        {
                            "arch": "llm_char_lstm",
                            "backend": "cpu",
                            "recurrent": "lstm",
                            "head_prior": "learned-bigram",
                            "char_feature": "token-bigram",
                            "mode": "embedding(12,token-bigram)",
                            "seed": int(seed),
                            "steps": 16,
                            "hidden": 24,
                            "embed_dim": 12,
                            "epochs": 12,
                            "batches": 32,
                            "batch": 4,
                            "eval_samples": 192,
                            "head_residual_scale": 0.5,
                            "bigram_topk_guard": 0.2,
                            "bigram_topk_guard_k": 5,
                            "bigram_rank_guard": 0.1,
                            "bigram_rank_guard_margin": 0.05,
                            "bigram_rank_guard_band": 0.003,
                            "bigram_rank_guard_min_candidates": rank_min,
                            "validation_start_fraction": 0.0,
                            "lr": 0.0025,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "initial_validation": {"mean_nll": 3.0},
                            "final_validation": {
                                "windows": 192,
                                "mean_nll": 2.7,
                                "mean_target_rank_debt_vs_bigram": debt,
                                "mean_target_rank_lift_vs_bigram": -debt,
                                "mean_top5_overlap_with_bigram": 0.95,
                            },
                            "bigram_validation": {
                                "windows": 192,
                                "mean_nll": 2.7,
                            },
                            "validation_nll_delta": -0.3,
                            "final_vs_bigram_nll_delta": 0.0,
                            "rank_guard_coverage": {
                                "windows": 192,
                                "mean_guarded_candidates": 1.5
                                if rank_min
                                else 1.0,
                                "zero_guarded_candidate_ratio": 0.0
                                if rank_min
                                else 0.5,
                                "mean_adaptive_filled_candidates": 0.5
                                if rank_min
                                else 0.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            for seed in ["7", "13"]:
                write_run(seed, 0, 3.0)
                write_run(seed, 1, 2.99)

            sweep_args = [
                str(data_path),
                "--run-root",
                str(root),
                "--architectures",
                "lstm",
                "--features",
                "token-bigram",
                "--head-priors",
                "learned-bigram",
                "--backends",
                "cpu",
                "--seeds",
                "7,13",
                "--bigram-topk-guard",
                "0.2",
                "--bigram-topk-guard-k",
                "5",
                "--bigram-rank-guard",
                "0.1",
                "--bigram-rank-guard-margin",
                "0.05",
                "--bigram-rank-guard-band",
                "0.003",
                "--bigram-rank-guard-min-candidates-values",
                "0,1",
                "--skip-existing",
                "--no-print-compare",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(sweep_args)
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                fail_code = run_char_lm_sweep.main(
                    sweep_args
                    + [
                        "--compare-summary-fail-on-rank-min-promotion-decision",
                        "promote",
                    ]
                )
            fail_manifest = json.loads(
                (root / "sweep.json").read_text(encoding="utf-8")
            )
            fail_error_text = (root / "compare_summary.error.log").read_text(
                encoding="utf-8"
            )

        self.assertEqual(code, 0)
        gate = manifest["compare_summary_bigram_rank_min_promotion_gate"]
        self.assertEqual(gate["decision"], "promote")
        self.assertEqual(gate["strict_promotions"], "1")
        self.assertEqual(gate["bounded_promotions"], "0")
        self.assertEqual(gate["non_promoted_rows"], "0")
        self.assertEqual(gate["recommendation_rows"], "1")
        self.assertEqual(fail_code, 1)
        fail_gate = fail_manifest["compare_summary_bigram_rank_min_promotion_gate"]
        self.assertTrue(fail_manifest["failed"])
        self.assertTrue(fail_manifest["compare_summary_failed"])
        self.assertEqual(fail_gate["decision"], "promote")
        self.assertEqual(fail_gate["failed"], "true")
        self.assertEqual(fail_gate["fail_on_decisions"], "promote")
        self.assertIn("rank-min promotion gate failed", fail_error_text)

    def test_char_lm_sweep_no_print_compare_keeps_artifacts_quietly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch quiet compare smoke", encoding="utf-8")
            run_dir = (
                root
                / "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "initial_validation": {"mean_nll": 3.0},
                        "final_validation": {"mean_nll": 2.9},
                        "validation_nll_delta": -0.1,
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--skip-existing",
                        "--no-print-compare",
                        "--quiet-runs",
                    ]
                )
            output = stdout.getvalue()
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
            compare_path_exists = Path(manifest["compare_path"]).exists()
            compare_summary_path_exists = Path(manifest["compare_summary_path"]).exists()

        self.assertEqual(code, 0)
        self.assertTrue(compare_path_exists)
        self.assertTrue(compare_summary_path_exists)
        self.assertIn("compare:", output)
        self.assertIn("compare_summary:", output)
        self.assertNotIn("[1/1]", output)
        self.assertNotIn("| run | arch | backend |", output)

    def test_char_lm_sweep_manifest_marks_paired_quality_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.txt"
            data_path.write_text("spiral torch paired gate smoke", encoding="utf-8")
            run_payloads = [
                (
                    "finetune__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7",
                    {
                        "arch": "llm_char_finetune",
                        "backend": "cpu",
                        "recurrent": "spiral",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    },
                    2.0,
                ),
                (
                    "lstm__feature-token-bigram__head-learned-unigram__backend-cpu__seed-7",
                    {
                        "arch": "llm_char_lstm",
                        "backend": "cpu",
                        "recurrent": "lstm",
                        "head_prior": "learned-unigram",
                        "char_feature": "token-bigram",
                        "mode": "embedding(4,token-bigram)",
                    },
                    2.1,
                ),
            ]
            for name, run_payload, final_nll in run_payloads:
                run_dir = root / name
                run_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text(
                    json.dumps(run_payload),
                    encoding="utf-8",
                )
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "initial_validation": {"mean_nll": 3.0},
                            "final_validation": {"mean_nll": final_nll},
                            "validation_nll_delta": final_nll - 3.0,
                        }
                    ),
                    encoding="utf-8",
                )

            with contextlib.redirect_stdout(io.StringIO()):
                code = run_char_lm_sweep.main(
                    [
                        str(data_path),
                        "--run-root",
                        str(root),
                        "--architectures",
                        "finetune,lstm",
                        "--features",
                        "token-bigram",
                        "--head-priors",
                        "learned-unigram",
                        "--backends",
                        "cpu",
                        "--seeds",
                        "7",
                        "--skip-existing",
                        "--compare-summary-fail-on-paired-quality-status",
                        "regressed",
                    ]
                )
            manifest = json.loads((root / "sweep.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        self.assertTrue(manifest["failed"])
        self.assertTrue(manifest["compare_summary_failed"])
        gate = manifest["compare_summary_paired_recurrent_gate"]
        self.assertTrue(gate["failed"])
        self.assertEqual(gate["fail_on_quality_statuses"], ["regressed"])
        self.assertEqual(gate["failures"]["quality_statuses"], {"regressed": 1})
        self.assertEqual(
            gate["failures"]["pairs"][0]["efficiency_verdict"],
            "candidate_quality_regressed",
        )

    def test_char_lm_compare_summary_flags_and_filters_scan_fallbacks(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "wgpu",
                    "steps": "3",
                    "hidden": "4",
                    "embed_dim": "4",
                    "runs": "1",
                    "final_nll_mean": "2.0000",
                    "best_nll_mean": "2.0000",
                    "final_vs_unigram_mean": "0.1000",
                    "final_vs_bigram_mean": "0.5000",
                    "cpu_debt_ops_mean": "30.0000",
                    "lstm_est_cpu_debt_ops_mean": "100.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "no suitable WGPU adapter:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "steps": "3",
                    "hidden": "4",
                    "embed_dim": "4",
                    "runs": "1",
                    "final_nll_mean": "2.1000",
                    "best_nll_mean": "2.1000",
                    "final_vs_unigram_mean": "0.2000",
                    "final_vs_bigram_mean": "0.6000",
                    "cpu_debt_ops_mean": "80.0000",
                    "lstm_est_cpu_debt_ops_mean": "100.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        rows = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=8,
            route_clean_only=False,
            prefer_clean_route=False,
        )
        clean_rows = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=8,
            route_clean_only=True,
            prefer_clean_route=False,
        )
        preferred = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
        )

        self.assertEqual(rows[0]["route_status"], "scan_fallback")
        self.assertEqual(clean_rows[0]["route_status"], "clean_route")
        self.assertEqual(clean_rows[0]["backend"], "cpu")
        self.assertEqual(clean_rows[0]["final_vs_bigram_mean"], "0.6000")
        self.assertEqual(preferred[0]["route_status"], "clean_route")

    def test_char_lm_compare_summary_surfaces_head_and_training_columns(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "head_resid": "2.0000",
                    "bigram_guard": "0.0500",
                    "bigram_guard_k": "3",
                    "bigram_rank_guard": "0.0250",
                    "bigram_rank_margin": "0.0100",
                    "bigram_rank_band": "0.0030",
                    "bigram_rank_min": "2",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "4",
                    "batches": "8",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.0200",
                    "runs": "1",
                    "rank_cov_windows_mean": "42.0000",
                    "rank_cov_unbounded_mean": "4.0000",
                    "rank_cov_band_mean": "2.5000",
                    "rank_cov_min_mean": "2.0000",
                    "rank_cov_guarded_mean": "2.0000",
                    "rank_cov_effective_band_mean": "0.0040",
                    "rank_cov_adaptive_fill_ratio_mean": "0.2500",
                    "rank_cov_filled_mean": "0.5000",
                    "rank_cov_zero_ratio_mean": "0.1000",
                    "rank_cov_mass_mean": "0.2000",
                    "rank_cov_band_ratio_mean": "0.6250",
                    "rank_cov_topk_ratio_mean": "0.5000",
                    "final_nll_mean": "2.8500",
                    "best_nll_mean": "2.8400",
                    "final_bigram_logprob_lift_mean": "0.1200",
                    "final_bigram_rank_lift_mean": "1.5000",
                    "final_bigram_target_rank_mean": "4.5000",
                    "final_bigram_rank_debt_mean": "-1.5000",
                    "final_kl_bigram_mean": "0.0200",
                    "final_top5_bigram_overlap_mean": "40.00%",
                    "trace_step_ms_mean_mean": "42.5000",
                    "trace_update_ratio_mean": "0.0075",
                    "cpu_debt_ops_mean": "616.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                }
            ],
        }

        rows = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
        )
        table = summarize_char_lm_compare.markdown_table(rows)

        self.assertEqual(rows[0]["head_prior"], "learned-unigram")
        self.assertEqual(rows[0]["head_resid"], "2.0000")
        self.assertEqual(rows[0]["bigram_guard"], "0.0500")
        self.assertEqual(rows[0]["bigram_guard_k"], "3")
        self.assertEqual(rows[0]["bigram_rank_guard"], "0.0250")
        self.assertEqual(rows[0]["bigram_rank_margin"], "0.0100")
        self.assertEqual(rows[0]["bigram_rank_band"], "0.0030")
        self.assertEqual(rows[0]["bigram_rank_min"], "2")
        self.assertEqual(rows[0]["rank_cov_min_mean"], "2.0000")
        self.assertEqual(rows[0]["rank_cov_guarded_mean"], "2.0000")
        self.assertEqual(rows[0]["rank_cov_effective_band_mean"], "0.0040")
        self.assertEqual(rows[0]["rank_cov_adaptive_fill_ratio_mean"], "0.2500")
        self.assertEqual(rows[0]["rank_cov_filled_mean"], "0.5000")
        self.assertEqual(rows[0]["rank_cov_band_ratio_mean"], "0.6250")
        self.assertEqual(rows[0]["mode"], "embedding(8,token-bigram)")
        self.assertEqual(rows[0]["epochs"], "4")
        self.assertEqual(rows[0]["batches"], "8")
        self.assertEqual(rows[0]["batch"], "4")
        self.assertEqual(rows[0]["eval_samples"], "32")
        self.assertEqual(rows[0]["final_bigram_logprob_lift_mean"], "0.1200")
        self.assertEqual(rows[0]["final_bigram_rank_lift_mean"], "1.5000")
        self.assertEqual(rows[0]["final_bigram_target_rank_mean"], "4.5000")
        self.assertEqual(rows[0]["final_bigram_rank_debt_mean"], "-1.5000")
        self.assertEqual(rows[0]["final_kl_bigram_mean"], "0.0200")
        self.assertEqual(rows[0]["final_top5_bigram_overlap_mean"], "40.00%")
        self.assertEqual(rows[0]["trace_step_ms_mean_mean"], "42.5000")
        self.assertEqual(rows[0]["trace_update_ratio_mean"], "0.0075")
        self.assertIn("head_prior", table)
        self.assertIn("bigram_guard", table)
        self.assertIn("bigram_rank_guard", table)
        self.assertIn("bigram_rank_band", table)
        self.assertIn("bigram_rank_min", table)
        self.assertIn("rank_cov_guarded_mean", table)
        self.assertIn("rank_cov_effective_band_mean", table)
        self.assertIn("epochs", table)
        self.assertIn("final_bigram_logprob_lift_mean", table)
        self.assertIn("final_bigram_rank_debt_mean", table)
        self.assertIn("final_top5_bigram_overlap_mean", table)
        self.assertIn("trace_step_ms_mean_mean", table)

    def test_char_lm_compare_summary_reports_baseline_difficulty_hotspots(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "eval_samples": "32",
                    "runs": "2",
                    "final_nll_mean": "2.9500",
                    "delta_nll_mean": "-0.0010",
                    "unigram_nll_mean": "2.9600",
                    "bigram_nll_mean": "3.0000",
                    "final_vs_bigram_mean": "-0.0500",
                    "best_vs_bigram_mean": "-0.0510",
                    "lstm_scan_backend_counts": "cpu:2",
                    "lstm_scan_fallback_counts": "none:2",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "eval_samples": "128",
                    "runs": "2",
                    "final_nll_mean": "2.9936",
                    "delta_nll_mean": "-0.0002",
                    "unigram_nll_mean": "2.9937",
                    "bigram_nll_mean": "2.9728",
                    "final_vs_bigram_mean": "0.0207",
                    "best_vs_bigram_mean": "0.0200",
                    "lstm_scan_backend_counts": "cpu:2",
                    "lstm_scan_fallback_counts": "none:2",
                },
            ],
        }

        rows = summarize_char_lm_compare.baseline_difficulty_rows(
            [("probe/compare.json", payload)],
            limit=2,
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            baseline_difficulty=rows,
        )

        self.assertEqual(rows[0]["eval_samples"], "128")
        self.assertEqual(rows[0]["bigram_vs_unigram_delta"], "-0.0209")
        self.assertEqual(
            rows[0]["bigram_baseline_status"],
            "bigram_stronger_than_unigram",
        )
        self.assertEqual(rows[0]["model_vs_bigram_status"], "model_lags_bigram")
        self.assertEqual(rows[0]["learning_status"], "loss_neutral")
        self.assertEqual(rows[1]["eval_samples"], "32")
        self.assertIn("## Baseline Difficulty Hotspots", report)

    def test_char_lm_compare_summary_can_rank_by_bigram_gap(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "final_vs_bigram_mean": "0.3000",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.1000",
                    "best_nll_mean": "1.1000",
                    "final_vs_bigram_mean": "-0.2000",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        by_final_nll = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_nll",
        )
        by_bigram_gap = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_vs_bigram",
        )

        self.assertEqual(by_final_nll[0]["final_vs_bigram_mean"], "0.3000")
        self.assertEqual(by_bigram_gap[0]["final_vs_bigram_mean"], "-0.2000")

    def test_char_lm_compare_summary_can_rank_by_bigram_lift_descending(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "final_bigram_logprob_lift_mean": "0.0100",
                    "final_bigram_rank_lift_mean": "0.2500",
                    "final_bigram_rank_debt_mean": "-0.2500",
                    "final_top5_bigram_overlap_mean": "20.00%",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.1000",
                    "best_nll_mean": "1.1000",
                    "final_bigram_logprob_lift_mean": "0.0300",
                    "final_bigram_rank_lift_mean": "1.5000",
                    "final_bigram_rank_debt_mean": "-1.5000",
                    "final_top5_bigram_overlap_mean": "80.00%",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        by_logprob_lift = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_bigram_logprob_lift",
        )
        by_rank_lift = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_bigram_rank_lift",
        )
        by_top5_overlap = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_top5_bigram_overlap",
        )
        by_rank_debt = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_bigram_rank_debt",
        )

        self.assertEqual(by_logprob_lift[0]["final_bigram_logprob_lift_mean"], "0.0300")
        self.assertEqual(by_rank_lift[0]["final_bigram_rank_lift_mean"], "1.5000")
        self.assertEqual(by_top5_overlap[0]["final_top5_bigram_overlap_mean"], "80.00%")
        self.assertEqual(by_rank_debt[0]["final_bigram_rank_debt_mean"], "-1.5000")

    def test_char_lm_compare_summary_pairs_spiral_and_lstm_deltas(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_finetune",
                    "recurrent": "spiral",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "head_resid": "0.5000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "8",
                    "batches": "16",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.02",
                    "runs": "3",
                    "final_nll_mean": "2.0000",
                    "delta_nll_mean": "-0.1000",
                    "final_vs_bigram_mean": "0.5000",
                    "trace_step_ms_mean_mean": "100.0000",
                    "cpu_debt_ops_mean": "200.0000",
                    "lstm_scan_backend_counts": "-",
                    "lstm_scan_fallback_counts": "-",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "head_resid": "0.5000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "8",
                    "batches": "16",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.02",
                    "runs": "3",
                    "final_nll_mean": "2.1000",
                    "final_vs_bigram_mean": "0.3000",
                    "trace_step_ms_mean_mean": "50.0000",
                    "cpu_debt_ops_mean": "100.0000",
                    "lstm_scan_backend_counts": "cpu:3",
                    "lstm_scan_fallback_counts": "none:3",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_recurrent_deltas(
            [("probe/compare.json", payload)]
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            paired_deltas=pairs,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_recurrent"], "lstm")
        self.assertEqual(pairs[0]["baseline_recurrent"], "spiral")
        self.assertEqual(pairs[0]["final_nll_delta"], "0.1000")
        self.assertEqual(pairs[0]["final_vs_bigram_delta"], "-0.2000")
        self.assertEqual(pairs[0]["trace_step_ms_delta"], "-50.0000")
        self.assertEqual(pairs[0]["trace_step_ms_ratio"], "0.5000")
        self.assertEqual(pairs[0]["cpu_debt_delta"], "-100.0000")
        self.assertEqual(pairs[0]["cpu_debt_ratio"], "0.5000")
        self.assertEqual(pairs[0]["quality_status"], "regressed")
        self.assertEqual(pairs[0]["latency_status"], "improved")
        self.assertEqual(pairs[0]["cpu_debt_status"], "improved")
        self.assertEqual(pairs[0]["efficiency_verdict"], "candidate_quality_regressed")
        self.assertEqual(pairs[0]["candidate_route_status"], "clean_route")
        self.assertEqual(pairs[0]["baseline_route_status"], "no_scan_route")
        self.assertEqual(
            summarize_char_lm_compare.paired_recurrent_recommendations(
                pairs,
                limit=8,
            ),
            [],
        )
        self.assertIn("## Paired Recurrent Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_guard_deltas(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "bigram",
                    "head_resid": "1.0000",
                    "bigram_guard": "0.0000",
                    "bigram_guard_k": "3",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "8",
                    "batches": "16",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.02",
                    "runs": "3",
                    "final_nll_mean": "2.1000",
                    "final_vs_bigram_mean": "0.2000",
                    "final_bigram_logprob_lift_mean": "0.0100",
                    "final_bigram_rank_lift_mean": "0.5000",
                    "final_top5_bigram_overlap_mean": "70.00%",
                    "lstm_scan_backend_counts": "cpu:3",
                    "lstm_scan_fallback_counts": "none:3",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "bigram",
                    "head_resid": "1.0000",
                    "bigram_guard": "0.0500",
                    "bigram_guard_k": "3",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "8",
                    "batches": "16",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.02",
                    "runs": "3",
                    "final_nll_mean": "2.0900",
                    "final_vs_bigram_mean": "0.1500",
                    "final_bigram_logprob_lift_mean": "0.0200",
                    "final_bigram_rank_lift_mean": "0.9000",
                    "final_top5_bigram_overlap_mean": "75.00%",
                    "lstm_scan_backend_counts": "cpu:3",
                    "lstm_scan_fallback_counts": "none:3",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_guard_deltas(
            [("probe/compare.json", payload)]
        )
        recommendations = summarize_char_lm_compare.paired_bigram_guard_recommendations(
            pairs,
            limit=8,
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_guard_deltas=pairs,
            bigram_guard_recommendations=recommendations,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_bigram_guard"], "0.0500")
        self.assertEqual(pairs[0]["baseline_bigram_guard"], "0.0000")
        self.assertEqual(pairs[0]["final_nll_delta"], "-0.0100")
        self.assertEqual(pairs[0]["final_vs_bigram_delta"], "-0.0500")
        self.assertEqual(pairs[0]["bigram_logprob_lift_delta"], "0.0100")
        self.assertEqual(pairs[0]["bigram_rank_lift_delta"], "0.4000")
        self.assertEqual(pairs[0]["top5_bigram_overlap_delta_pp"], "5.0000")
        self.assertEqual(pairs[0]["nll_status"], "improved")
        self.assertEqual(pairs[0]["bigram_gap_status"], "improved")
        self.assertEqual(pairs[0]["bigram_logprob_status"], "improved")
        self.assertEqual(pairs[0]["bigram_rank_status"], "improved")
        self.assertEqual(pairs[0]["top5_bigram_status"], "improved")
        self.assertEqual(pairs[0]["guard_verdict"], "guard_quality_and_topk_improved")
        self.assertEqual(pairs[0]["quality_status"], "improved")
        self.assertEqual(pairs[0]["candidate_route_status"], "clean_route")
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["rank"], "1")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "quality_and_topk_improved",
        )
        self.assertEqual(recommendations[0]["candidate_bigram_guard"], "0.0500")
        self.assertIn("## Bigram Guard Recommendations", report)
        self.assertIn("## Bigram Guard Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_rank_guard_deltas(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.0000",
                    "bigram_rank_margin": "0.0500",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "1",
                    "final_nll_mean": "3.0182",
                    "final_vs_bigram_mean": "-0.0010",
                    "final_bigram_logprob_lift_mean": "0.0010",
                    "final_bigram_rank_lift_mean": "-3.1500",
                    "final_bigram_rank_debt_mean": "3.1500",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.1000",
                    "bigram_rank_margin": "0.0500",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "1",
                    "final_nll_mean": "3.0182",
                    "final_vs_bigram_mean": "-0.0010",
                    "final_bigram_logprob_lift_mean": "0.0010",
                    "final_bigram_rank_lift_mean": "-3.1300",
                    "final_bigram_rank_debt_mean": "3.1300",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_guard_deltas(
            [("probe/compare.json", payload)]
        )
        recommendations = (
            summarize_char_lm_compare.paired_bigram_rank_guard_recommendations(
                pairs,
                limit=8,
            )
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_guard_deltas=pairs,
            bigram_rank_guard_recommendations=recommendations,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_bigram_rank_guard"], "0.1000")
        self.assertEqual(pairs[0]["baseline_bigram_rank_guard"], "0.0000")
        self.assertEqual(pairs[0]["final_nll_delta"], "0.0000")
        self.assertEqual(pairs[0]["final_vs_bigram_delta"], "0.0000")
        self.assertEqual(pairs[0]["bigram_logprob_lift_delta"], "0.0000")
        self.assertEqual(pairs[0]["bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(pairs[0]["bigram_rank_lift_delta"], "0.0200")
        self.assertEqual(pairs[0]["top5_bigram_overlap_delta_pp"], "0.0000")
        self.assertEqual(pairs[0]["nll_status"], "neutral")
        self.assertEqual(pairs[0]["bigram_gap_status"], "neutral")
        self.assertEqual(pairs[0]["bigram_logprob_status"], "neutral")
        self.assertEqual(pairs[0]["rank_debt_status"], "improved")
        self.assertEqual(pairs[0]["bigram_rank_status"], "improved")
        self.assertEqual(pairs[0]["top5_bigram_status"], "neutral")
        self.assertEqual(pairs[0]["guard_verdict"], "rank_guard_rank_improved")
        self.assertEqual(pairs[0]["quality_status"], "neutral")
        self.assertEqual(pairs[0]["rank_status"], "improved")
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["rank"], "1")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "rank_improved_quality_neutral",
        )
        self.assertEqual(
            recommendations[0]["candidate_bigram_rank_guard"],
            "0.1000",
        )
        self.assertIn("## Bigram Rank Guard Recommendations", report)
        self.assertIn("## Bigram Rank Guard Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_rank_guard_seed_deltas(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.1000",
            "bigram_guard_k": "5",
            "bigram_rank_margin": "0.0500",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0",
            "lr": "0.0025",
            "final_nll": "3.0182",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
        }
        payload = {
            "schema": "st.char_lm.compare.v1",
            "runs": [
                {
                    **base,
                    "seed": "7",
                    "bigram_rank_guard": "0.0000",
                    "final_bigram_rank_lift": "-3.1500",
                    "final_bigram_rank_debt": "3.1500",
                },
                {
                    **base,
                    "seed": "7",
                    "bigram_rank_guard": "0.1000",
                    "final_bigram_rank_lift": "-3.1300",
                    "final_bigram_rank_debt": "3.1300",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_rank_guard": "0.0000",
                    "final_bigram_rank_lift": "-2.6500",
                    "final_bigram_rank_debt": "2.6500",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_rank_guard": "0.1000",
                    "final_bigram_rank_lift": "-2.7000",
                    "final_bigram_rank_debt": "2.7000",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_guard_seed_deltas(
            [("probe/compare.json", payload)]
        )
        stability = summarize_char_lm_compare.bigram_rank_guard_stability_rows(pairs)
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_guard_seed_deltas=pairs,
            bigram_rank_guard_stability=stability,
        )

        self.assertEqual(len(pairs), 2)
        seed7 = next(pair for pair in pairs if pair["seed"] == "7")
        seed13 = next(pair for pair in pairs if pair["seed"] == "13")
        self.assertEqual(seed7["bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(seed7["bigram_rank_lift_delta"], "0.0200")
        self.assertEqual(seed7["guard_verdict"], "rank_guard_rank_improved")
        self.assertEqual(seed7["rank_status"], "improved")
        self.assertEqual(seed13["bigram_rank_debt_delta"], "0.0500")
        self.assertEqual(seed13["bigram_rank_lift_delta"], "-0.0500")
        self.assertEqual(seed13["guard_verdict"], "rank_guard_rank_regressed")
        self.assertEqual(seed13["rank_status"], "regressed")
        self.assertEqual(len(stability), 1)
        self.assertEqual(stability[0]["seed_pairs"], "2")
        self.assertEqual(stability[0]["rank_improved_seeds"], "1")
        self.assertEqual(stability[0]["rank_neutral_seeds"], "0")
        self.assertEqual(stability[0]["rank_regressed_seeds"], "1")
        self.assertEqual(stability[0]["mean_bigram_rank_debt_delta"], "0.0150")
        self.assertEqual(stability[0]["min_bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(stability[0]["max_bigram_rank_debt_delta"], "0.0500")
        self.assertEqual(stability[0]["mean_bigram_rank_lift_delta"], "-0.0150")
        self.assertEqual(stability[0]["stability_verdict"], "rank_guard_seed_mixed")
        self.assertIn("## Bigram Rank Guard Stability", report)
        self.assertIn("## Bigram Rank Guard Seed Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_rank_band_deltas(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.1000",
                    "bigram_rank_margin": "0.0500",
                    "bigram_rank_band": "0.0000",
                    "bigram_soft_guard": "0.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "2",
                    "final_nll_mean": "3.0181",
                    "final_vs_bigram_mean": "-0.0011",
                    "final_bigram_logprob_lift_mean": "0.0011",
                    "final_bigram_rank_lift_mean": "-2.9150",
                    "final_bigram_rank_debt_mean": "2.9150",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:2",
                    "lstm_scan_fallback_counts": "none:2",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.1000",
                    "bigram_rank_margin": "0.0500",
                    "bigram_rank_band": "0.0030",
                    "bigram_soft_guard": "0.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "2",
                    "final_nll_mean": "3.0181",
                    "final_vs_bigram_mean": "-0.0011",
                    "final_bigram_logprob_lift_mean": "0.0011",
                    "final_bigram_rank_lift_mean": "-2.9400",
                    "final_bigram_rank_debt_mean": "2.9400",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:2",
                    "lstm_scan_fallback_counts": "none:2",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_band_deltas(
            [("probe/compare.json", payload)]
        )
        recommendations = (
            summarize_char_lm_compare.paired_bigram_rank_band_recommendations(
                pairs,
                limit=8,
            )
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_band_deltas=pairs,
            bigram_rank_band_recommendations=recommendations,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_bigram_rank_band"], "0.0030")
        self.assertEqual(pairs[0]["baseline_bigram_rank_band"], "0.0000")
        self.assertEqual(pairs[0]["final_nll_delta"], "0.0000")
        self.assertEqual(pairs[0]["final_vs_bigram_delta"], "0.0000")
        self.assertEqual(pairs[0]["bigram_logprob_lift_delta"], "0.0000")
        self.assertEqual(pairs[0]["bigram_rank_debt_delta"], "0.0250")
        self.assertEqual(pairs[0]["bigram_rank_lift_delta"], "-0.0250")
        self.assertEqual(pairs[0]["top5_bigram_overlap_delta_pp"], "0.0000")
        self.assertEqual(pairs[0]["band_verdict"], "rank_band_alignment_regressed")
        self.assertEqual(pairs[0]["quality_status"], "neutral")
        self.assertEqual(pairs[0]["alignment_status"], "regressed")
        self.assertEqual(recommendations, [])
        self.assertIn("## Bigram Rank Band Deltas", report)
        self.assertNotIn("## Bigram Rank Band Recommendations", report)

    def test_char_lm_compare_summary_pairs_bigram_rank_band_seed_deltas(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.1000",
            "bigram_guard_k": "5",
            "bigram_rank_guard": "0.1000",
            "bigram_rank_margin": "0.0500",
            "bigram_soft_guard": "0.0000",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0",
            "lr": "0.0025",
            "final_nll": "3.0182",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
        }
        payload = {
            "schema": "st.char_lm.compare.v1",
            "runs": [
                {
                    **base,
                    "seed": "7",
                    "bigram_rank_band": "0.0000",
                    "final_bigram_rank_lift": "-3.1300",
                    "final_bigram_rank_debt": "3.1300",
                },
                {
                    **base,
                    "seed": "7",
                    "bigram_rank_band": "0.0030",
                    "final_bigram_rank_lift": "-3.2100",
                    "final_bigram_rank_debt": "3.2100",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_rank_band": "0.0000",
                    "final_bigram_rank_lift": "-2.7000",
                    "final_bigram_rank_debt": "2.7000",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_rank_band": "0.0030",
                    "final_bigram_rank_lift": "-2.6700",
                    "final_bigram_rank_debt": "2.6700",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_band_seed_deltas(
            [("probe/compare.json", payload)]
        )
        stability = summarize_char_lm_compare.bigram_rank_band_stability_rows(pairs)
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_band_seed_deltas=pairs,
            bigram_rank_band_stability=stability,
        )

        self.assertEqual(len(pairs), 2)
        seed7 = next(pair for pair in pairs if pair["seed"] == "7")
        seed13 = next(pair for pair in pairs if pair["seed"] == "13")
        self.assertEqual(seed7["bigram_rank_debt_delta"], "0.0800")
        self.assertEqual(seed7["bigram_rank_lift_delta"], "-0.0800")
        self.assertEqual(seed7["band_verdict"], "rank_band_alignment_regressed")
        self.assertEqual(seed7["alignment_status"], "regressed")
        self.assertEqual(seed13["bigram_rank_debt_delta"], "-0.0300")
        self.assertEqual(seed13["bigram_rank_lift_delta"], "0.0300")
        self.assertEqual(seed13["band_verdict"], "rank_band_alignment_improved")
        self.assertEqual(seed13["alignment_status"], "improved")
        self.assertEqual(len(stability), 1)
        self.assertEqual(stability[0]["seed_pairs"], "2")
        self.assertEqual(stability[0]["alignment_improved_seeds"], "1")
        self.assertEqual(stability[0]["alignment_neutral_seeds"], "0")
        self.assertEqual(stability[0]["alignment_regressed_seeds"], "1")
        self.assertEqual(stability[0]["mean_bigram_rank_debt_delta"], "0.0250")
        self.assertEqual(stability[0]["min_bigram_rank_debt_delta"], "-0.0300")
        self.assertEqual(stability[0]["max_bigram_rank_debt_delta"], "0.0800")
        self.assertEqual(stability[0]["mean_bigram_rank_lift_delta"], "-0.0250")
        self.assertEqual(stability[0]["stability_verdict"], "rank_band_seed_mixed")
        self.assertIn("## Bigram Rank Band Stability", report)
        self.assertIn("## Bigram Rank Band Seed Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_rank_min_deltas(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.1000",
                    "bigram_rank_margin": "0.0500",
                    "bigram_rank_band": "0.0030",
                    "bigram_rank_min": "0",
                    "bigram_soft_guard": "0.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "2",
                    "rank_cov_guarded_mean": "1.0738",
                    "rank_cov_zero_ratio_mean": "0.5743",
                    "rank_cov_filled_mean": "0.0000",
                    "final_nll_mean": "3.0181",
                    "final_vs_bigram_mean": "-0.0011",
                    "final_bigram_logprob_lift_mean": "0.0011",
                    "final_bigram_rank_lift_mean": "-2.9400",
                    "final_bigram_rank_debt_mean": "2.9400",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:2",
                    "lstm_scan_fallback_counts": "none:2",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.1000",
                    "bigram_rank_margin": "0.0500",
                    "bigram_rank_band": "0.0030",
                    "bigram_rank_min": "3",
                    "bigram_soft_guard": "0.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "2",
                    "rank_cov_guarded_mean": "3.1782",
                    "rank_cov_zero_ratio_mean": "0.0000",
                    "rank_cov_filled_mean": "2.1044",
                    "final_nll_mean": "3.0181",
                    "final_vs_bigram_mean": "-0.0011",
                    "final_bigram_logprob_lift_mean": "0.0011",
                    "final_bigram_rank_lift_mean": "-2.9200",
                    "final_bigram_rank_debt_mean": "2.9200",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:2",
                    "lstm_scan_fallback_counts": "none:2",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_min_deltas(
            [("probe/compare.json", payload)]
        )
        recommendations = (
            summarize_char_lm_compare.paired_bigram_rank_min_recommendations(
                pairs,
                limit=8,
            )
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_min_deltas=pairs,
            bigram_rank_min_recommendations=recommendations,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_bigram_rank_min"], "3")
        self.assertEqual(pairs[0]["baseline_bigram_rank_min"], "0")
        self.assertEqual(pairs[0]["rank_cov_zero_ratio_delta"], "-0.5743")
        self.assertEqual(pairs[0]["rank_cov_guarded_delta"], "2.1044")
        self.assertEqual(pairs[0]["rank_cov_filled_delta"], "2.1044")
        self.assertEqual(pairs[0]["bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(pairs[0]["bigram_rank_lift_delta"], "0.0200")
        self.assertEqual(pairs[0]["rank_cov_zero_status"], "improved")
        self.assertEqual(pairs[0]["rank_cov_guarded_status"], "improved")
        self.assertEqual(pairs[0]["min_verdict"], "rank_min_alignment_improved")
        self.assertEqual(pairs[0]["quality_status"], "neutral")
        self.assertEqual(pairs[0]["alignment_status"], "improved")
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["candidate_bigram_rank_min"], "3")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "alignment_improved_quality_neutral",
        )
        self.assertIn("## Bigram Rank Min Recommendations", report)
        self.assertIn("## Bigram Rank Min Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_rank_min_seed_deltas(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.1000",
            "bigram_guard_k": "5",
            "bigram_rank_guard": "0.1000",
            "bigram_rank_margin": "0.0500",
            "bigram_rank_band": "0.0030",
            "bigram_soft_guard": "0.0000",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0",
            "lr": "0.0025",
            "final_nll": "3.0182",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
        }
        payload = {
            "schema": "st.char_lm.compare.v1",
            "runs": [
                {
                    **base,
                    "seed": "7",
                    "bigram_rank_min": "0",
                    "rank_cov_guarded": "1.0738",
                    "rank_cov_zero_ratio": "0.5743",
                    "rank_cov_filled": "0.0000",
                    "final_bigram_rank_lift": "-3.2100",
                    "final_bigram_rank_debt": "3.2100",
                },
                {
                    **base,
                    "seed": "7",
                    "bigram_rank_min": "3",
                    "rank_cov_guarded": "3.1782",
                    "rank_cov_zero_ratio": "0.0000",
                    "rank_cov_filled": "2.1044",
                    "final_bigram_rank_lift": "-3.1400",
                    "final_bigram_rank_debt": "3.1400",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_rank_min": "0",
                    "rank_cov_guarded": "1.0738",
                    "rank_cov_zero_ratio": "0.5743",
                    "rank_cov_filled": "0.0000",
                    "final_bigram_rank_lift": "-2.6700",
                    "final_bigram_rank_debt": "2.6700",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_rank_min": "3",
                    "rank_cov_guarded": "3.1782",
                    "rank_cov_zero_ratio": "0.0000",
                    "rank_cov_filled": "2.1044",
                    "final_bigram_rank_lift": "-2.7000",
                    "final_bigram_rank_debt": "2.7000",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_min_seed_deltas(
            [("probe/compare.json", payload)]
        )
        stability = summarize_char_lm_compare.bigram_rank_min_stability_rows(pairs)
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_min_seed_deltas=pairs,
            bigram_rank_min_stability=stability,
        )

        self.assertEqual(len(pairs), 2)
        seed7 = next(pair for pair in pairs if pair["seed"] == "7")
        seed13 = next(pair for pair in pairs if pair["seed"] == "13")
        self.assertEqual(seed7["bigram_rank_debt_delta"], "-0.0700")
        self.assertEqual(seed7["bigram_rank_lift_delta"], "0.0700")
        self.assertEqual(seed7["rank_cov_zero_ratio_delta"], "-0.5743")
        self.assertEqual(seed7["min_verdict"], "rank_min_alignment_improved")
        self.assertEqual(seed7["alignment_status"], "improved")
        self.assertEqual(seed13["bigram_rank_debt_delta"], "0.0300")
        self.assertEqual(seed13["bigram_rank_lift_delta"], "-0.0300")
        self.assertEqual(seed13["rank_cov_zero_ratio_delta"], "-0.5743")
        self.assertEqual(seed13["min_verdict"], "rank_min_alignment_mixed")
        self.assertEqual(seed13["alignment_status"], "regressed")
        self.assertEqual(len(stability), 1)
        self.assertEqual(stability[0]["seed_pairs"], "2")
        self.assertEqual(stability[0]["alignment_improved_seeds"], "1")
        self.assertEqual(stability[0]["alignment_neutral_seeds"], "0")
        self.assertEqual(stability[0]["alignment_regressed_seeds"], "1")
        self.assertEqual(stability[0]["mean_rank_cov_zero_ratio_delta"], "-0.5743")
        self.assertEqual(stability[0]["mean_rank_cov_guarded_delta"], "2.1044")
        self.assertEqual(stability[0]["mean_rank_cov_filled_delta"], "2.1044")
        self.assertEqual(stability[0]["mean_bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(stability[0]["min_bigram_rank_debt_delta"], "-0.0700")
        self.assertEqual(stability[0]["max_bigram_rank_debt_delta"], "0.0300")
        self.assertEqual(stability[0]["stability_verdict"], "rank_min_seed_mixed")
        self.assertIn("## Bigram Rank Min Stability", report)
        self.assertIn("## Bigram Rank Min Seed Deltas", report)

    def test_char_lm_compare_summary_merges_rank_min_stability_sources(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.1000",
            "bigram_guard_k": "5",
            "bigram_rank_guard": "0.1000",
            "bigram_rank_margin": "0.0500",
            "bigram_rank_band": "0.0030",
            "bigram_soft_guard": "0.0000",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0",
            "lr": "0.0025",
            "final_nll": "3.0000",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
            "final_bigram_rank_lift": "-3.0000",
            "final_bigram_rank_debt": "3.0000",
        }

        def payload_for(seeds: list[str]) -> dict[str, object]:
            runs = []
            for seed in seeds:
                runs.extend(
                    [
                        {
                            **base,
                            "seed": seed,
                            "bigram_rank_min": "0",
                            "rank_cov_guarded": "1.0000",
                            "rank_cov_zero_ratio": "0.5000",
                            "rank_cov_filled": "0.0000",
                        },
                        {
                            **base,
                            "seed": seed,
                            "bigram_rank_min": "1",
                            "rank_cov_guarded": "1.5000",
                            "rank_cov_zero_ratio": "0.0000",
                            "rank_cov_filled": "0.5000",
                        },
                    ]
                )
            return {"schema": "st.char_lm.compare.v1", "runs": runs}

        pairs = summarize_char_lm_compare.paired_bigram_rank_min_seed_deltas(
            [
                ("left/compare.json", payload_for(["7", "13"])),
                ("right/compare.json", payload_for(["21", "34"])),
            ]
        )
        sourcewise = summarize_char_lm_compare.bigram_rank_min_stability_rows(
            pairs,
        )
        merged = summarize_char_lm_compare.bigram_rank_min_stability_rows(
            pairs,
            merge_sources=True,
        )
        recommendations = (
            summarize_char_lm_compare.bigram_rank_min_stable_recommendations(
                merged,
                limit=8,
            )
        )
        gate = summarize_char_lm_compare.bigram_rank_min_promotion_gate(
            merged,
            recommendations,
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_min_stability=merged,
            bigram_rank_min_stable_recommendations=recommendations,
            bigram_rank_min_promotion_gate=gate,
        )

        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(sourcewise), 2)
        self.assertEqual({row["seed_pairs"] for row in sourcewise}, {"2"})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "merged:2")
        self.assertEqual(merged[0]["evidence_source_count"], "2")
        self.assertEqual(
            merged[0]["evidence_sources"],
            "left/compare.json,right/compare.json",
        )
        self.assertEqual(merged[0]["seed_pairs"], "4")
        self.assertEqual(merged[0]["alignment_improved_seeds"], "4")
        self.assertEqual(merged[0]["alignment_regressed_seeds"], "0")
        self.assertEqual(
            merged[0]["stability_verdict"],
            "rank_min_seed_stably_improved",
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["source"], "merged:2")
        self.assertEqual(gate["decision"], "promote")
        self.assertEqual(gate["strict_promotions"], "1")
        self.assertEqual(gate["bounded_promotions"], "0")
        self.assertEqual(gate["non_promoted_rows"], "0")
        self.assertIn("merged:2", report)
        self.assertIn("## Bigram Rank Min Promotion Gate", report)

    def test_char_lm_compare_summary_recommends_bounded_rank_min_mixed_merge(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.1000",
            "bigram_guard_k": "5",
            "bigram_rank_guard": "0.1000",
            "bigram_rank_margin": "0.0500",
            "bigram_rank_band": "0.0030",
            "bigram_soft_guard": "0.0000",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0",
            "lr": "0.0025",
            "final_nll": "3.0000",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
        }

        def payload_for(seeds: list[str], regressed_seed: str) -> dict[str, object]:
            runs = []
            for seed in seeds:
                candidate_rank_debt = "3.0100" if seed == regressed_seed else "2.9900"
                candidate_rank_lift = "-3.0100" if seed == regressed_seed else "-2.9900"
                runs.extend(
                    [
                        {
                            **base,
                            "seed": seed,
                            "bigram_rank_min": "0",
                            "rank_cov_guarded": "1.0000",
                            "rank_cov_zero_ratio": "0.5000",
                            "rank_cov_filled": "0.0000",
                            "final_bigram_rank_lift": "-3.0000",
                            "final_bigram_rank_debt": "3.0000",
                        },
                        {
                            **base,
                            "seed": seed,
                            "bigram_rank_min": "1",
                            "rank_cov_guarded": "1.5000",
                            "rank_cov_zero_ratio": "0.0000",
                            "rank_cov_filled": "0.5000",
                            "final_bigram_rank_lift": candidate_rank_lift,
                            "final_bigram_rank_debt": candidate_rank_debt,
                        },
                    ]
                )
            return {"schema": "st.char_lm.compare.v1", "runs": runs}

        pairs = summarize_char_lm_compare.paired_bigram_rank_min_seed_deltas(
            [
                ("left/compare.json", payload_for(["7", "13"], "none")),
                ("right/compare.json", payload_for(["21", "34"], "34")),
            ]
        )
        sourcewise = summarize_char_lm_compare.bigram_rank_min_stability_rows(
            pairs,
        )
        merged = summarize_char_lm_compare.bigram_rank_min_stability_rows(
            pairs,
            merge_sources=True,
        )
        recommendations = (
            summarize_char_lm_compare.bigram_rank_min_stable_recommendations(
                merged,
                limit=8,
            )
        )
        gate = summarize_char_lm_compare.bigram_rank_min_promotion_gate(
            merged,
            recommendations,
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_min_stability=merged,
            bigram_rank_min_stable_recommendations=recommendations,
            bigram_rank_min_promotion_gate=gate,
        )

        self.assertEqual(len(sourcewise), 2)
        self.assertIn(
            "rank_min_seed_mixed",
            {row["stability_verdict"] for row in sourcewise},
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["seed_pairs"], "4")
        self.assertEqual(merged[0]["alignment_improved_seeds"], "3")
        self.assertEqual(merged[0]["alignment_regressed_seeds"], "1")
        self.assertEqual(merged[0]["max_bigram_rank_debt_delta"], "0.0100")
        self.assertEqual(
            merged[0]["stability_verdict"],
            "rank_min_seed_bounded_mixed",
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(
            recommendations[0]["recommendation"],
            "bounded_alignment_improved",
        )
        self.assertEqual(gate["decision"], "promote_with_bounded_watch")
        self.assertEqual(gate["strict_promotions"], "0")
        self.assertEqual(gate["bounded_promotions"], "1")
        self.assertEqual(gate["non_promoted_rows"], "0")
        self.assertIn("rank_min_seed_bounded_mixed:1", gate["verdict_counts"])
        self.assertIn("promote_with_bounded_watch", report)

    def test_char_lm_compare_summary_rank_min_prefers_raw_rank_metrics(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.2000",
            "bigram_guard_k": "5",
            "bigram_rank_guard": "0.1000",
            "bigram_rank_margin": "0.0500",
            "bigram_rank_band": "0.0030",
            "bigram_soft_guard": "0.0000",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0.5",
            "lr": "0.0025",
            "seed": "13",
            "rank_cov_guarded": "1.0000",
            "rank_cov_zero_ratio": "0.5000",
            "rank_cov_filled": "0.0000",
            "final_nll": "3.0000",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
        }
        payload = {
            "schema": "st.char_lm.compare.v1",
            "runs": [
                {
                    **base,
                    "bigram_rank_min": "0",
                    "final_bigram_rank_lift": "-3.00",
                    "final_bigram_rank_lift_raw": "-3.00000000",
                    "final_bigram_rank_debt": "3.00",
                    "final_bigram_rank_debt_raw": "3.00000000",
                    "final_top5_bigram_overlap_raw": "88.12500000",
                },
                {
                    **base,
                    "bigram_rank_min": "1",
                    "rank_cov_guarded": "1.5000",
                    "rank_cov_zero_ratio": "0.0000",
                    "rank_cov_filled": "0.5000",
                    "final_bigram_rank_lift": "-3.00",
                    "final_bigram_rank_lift_raw": "-3.00520833",
                    "final_bigram_rank_debt": "3.00",
                    "final_bigram_rank_debt_raw": "3.00520833",
                    "final_top5_bigram_overlap_raw": "87.91666667",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_rank_min_seed_deltas(
            [("compare.json", payload)]
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["bigram_rank_debt_delta"], "0.0052")
        self.assertEqual(pairs[0]["bigram_rank_lift_delta"], "-0.0052")
        self.assertEqual(pairs[0]["top5_bigram_overlap_delta_pp"], "-0.2083")
        self.assertEqual(pairs[0]["rank_debt_status"], "regressed")
        self.assertEqual(pairs[0]["top5_bigram_status"], "regressed")

    def test_char_lm_compare_summary_rank_min_stable_recommendations(
        self,
    ) -> None:
        stable_rows = [
            {
                "source": "probe/compare.json",
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "learned-bigram",
                "head_resid": "0.5000",
                "bigram_guard": "0.1000",
                "bigram_guard_k": "5",
                "bigram_rank_guard": "0.1000",
                "bigram_rank_margin": "0.0500",
                "bigram_rank_band": "0.0030",
                "bigram_soft_guard": "0.0000",
                "char_feature": "token-bigram",
                "mode": "embedding(8,token-bigram)",
                "steps": "12",
                "hidden": "16",
                "embed_dim": "8",
                "epochs": "10",
                "batches": "24",
                "batch": "4",
                "eval_samples": "128",
                "val_start": "0.5000",
                "lr": "0.0025",
                "candidate_bigram_rank_min": "1",
                "baseline_bigram_rank_min": "0",
                "seed_pairs": "4",
                "alignment_improved_seeds": "4",
                "alignment_neutral_seeds": "0",
                "alignment_regressed_seeds": "0",
                "mean_rank_cov_zero_ratio_delta": "-0.5249",
                "mean_rank_cov_guarded_delta": "0.5249",
                "mean_rank_cov_filled_delta": "0.5249",
                "mean_bigram_rank_debt_delta": "0.0000",
                "min_bigram_rank_debt_delta": "0.0000",
                "max_bigram_rank_debt_delta": "0.0000",
                "mean_bigram_rank_lift_delta": "0.0000",
                "mean_final_nll_delta": "0.0000",
                "mean_final_vs_bigram_delta": "0.0000",
                "mean_top5_bigram_overlap_delta_pp": "0.0000",
                "stability_verdict": "rank_min_seed_stably_improved",
            },
            {
                "source": "probe/compare.json",
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "learned-bigram",
                "head_resid": "0.5000",
                "bigram_guard": "0.1000",
                "bigram_guard_k": "5",
                "bigram_rank_guard": "0.1000",
                "bigram_rank_margin": "0.0500",
                "bigram_rank_band": "0.0030",
                "bigram_soft_guard": "0.0000",
                "char_feature": "token-bigram",
                "mode": "embedding(8,token-bigram)",
                "steps": "12",
                "hidden": "16",
                "embed_dim": "8",
                "epochs": "10",
                "batches": "24",
                "batch": "4",
                "eval_samples": "128",
                "val_start": "0.5000",
                "lr": "0.0025",
                "candidate_bigram_rank_min": "3",
                "baseline_bigram_rank_min": "0",
                "seed_pairs": "4",
                "alignment_improved_seeds": "3",
                "alignment_neutral_seeds": "0",
                "alignment_regressed_seeds": "1",
                "mean_rank_cov_zero_ratio_delta": "-0.5249",
                "mean_rank_cov_guarded_delta": "2.1190",
                "mean_rank_cov_filled_delta": "2.1190",
                "mean_bigram_rank_debt_delta": "0.0000",
                "min_bigram_rank_debt_delta": "-0.0400",
                "max_bigram_rank_debt_delta": "0.0500",
                "mean_bigram_rank_lift_delta": "0.0000",
                "mean_final_nll_delta": "0.0000",
                "mean_final_vs_bigram_delta": "0.0000",
                "mean_top5_bigram_overlap_delta_pp": "0.0000",
                "stability_verdict": "rank_min_seed_mixed",
            },
        ]

        recommendations = (
            summarize_char_lm_compare.bigram_rank_min_stable_recommendations(
                stable_rows,
                limit=8,
            )
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_rank_min_stable_recommendations=recommendations,
        )

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["candidate_bigram_rank_min"], "1")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "stable_alignment_improved",
        )
        self.assertIn("## Bigram Rank Min Stable Recommendations", report)

    def test_char_lm_compare_summary_rank_min_gate_reports_partial_promotion(
        self,
    ) -> None:
        rows = [
            {
                "stability_verdict": "rank_min_seed_stably_improved",
                "seed_pairs": "4",
            },
            {
                "stability_verdict": "rank_min_seed_mixed",
                "seed_pairs": "4",
            },
        ]
        recommendations = (
            summarize_char_lm_compare.bigram_rank_min_stable_recommendations(
                rows,
                limit=8,
            )
        )
        gate = summarize_char_lm_compare.bigram_rank_min_promotion_gate(
            rows,
            recommendations,
        )

        self.assertEqual(gate["decision"], "partial_promote_needs_tuning")
        self.assertEqual(gate["strict_promotions"], "1")
        self.assertEqual(gate["bounded_promotions"], "0")
        self.assertEqual(gate["non_promoted_rows"], "1")
        self.assertEqual(gate["recommendation_rows"], "1")

    def test_char_lm_compare_summary_pairs_bigram_soft_guard_deltas(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.0000",
                    "bigram_rank_margin": "0.0500",
                    "bigram_soft_guard": "0.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "1",
                    "final_nll_mean": "3.0182",
                    "final_vs_bigram_mean": "-0.0010",
                    "final_bigram_logprob_lift_mean": "0.0010",
                    "final_bigram_rank_lift_mean": "-3.1500",
                    "final_bigram_rank_debt_mean": "3.1500",
                    "final_top5_bigram_overlap_mean": "88.14%",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-bigram",
                    "head_resid": "0.5000",
                    "bigram_guard": "0.1000",
                    "bigram_guard_k": "5",
                    "bigram_rank_guard": "0.0000",
                    "bigram_rank_margin": "0.0500",
                    "bigram_soft_guard": "0.0500",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "16",
                    "embed_dim": "8",
                    "epochs": "10",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "128",
                    "val_start": "0",
                    "lr": "0.0025",
                    "runs": "1",
                    "final_nll_mean": "3.0181",
                    "final_vs_bigram_mean": "-0.0012",
                    "final_bigram_logprob_lift_mean": "0.0012",
                    "final_bigram_rank_lift_mean": "-3.1300",
                    "final_bigram_rank_debt_mean": "3.1300",
                    "final_top5_bigram_overlap_mean": "88.64%",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_soft_guard_deltas(
            [("probe/compare.json", payload)]
        )
        recommendations = (
            summarize_char_lm_compare.paired_bigram_soft_guard_recommendations(
                pairs,
                limit=8,
            )
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_soft_guard_deltas=pairs,
            bigram_soft_guard_recommendations=recommendations,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate_bigram_soft_guard"], "0.0500")
        self.assertEqual(pairs[0]["baseline_bigram_soft_guard"], "0.0000")
        self.assertEqual(pairs[0]["final_nll_delta"], "-0.0001")
        self.assertEqual(pairs[0]["final_vs_bigram_delta"], "-0.0002")
        self.assertEqual(pairs[0]["bigram_logprob_lift_delta"], "0.0002")
        self.assertEqual(pairs[0]["bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(pairs[0]["bigram_rank_lift_delta"], "0.0200")
        self.assertEqual(pairs[0]["top5_bigram_overlap_delta_pp"], "0.5000")
        self.assertEqual(pairs[0]["guard_verdict"], "soft_guard_alignment_improved")
        self.assertEqual(pairs[0]["quality_status"], "neutral")
        self.assertEqual(pairs[0]["alignment_status"], "improved")
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(
            recommendations[0]["recommendation"],
            "alignment_improved_quality_neutral",
        )
        self.assertEqual(
            recommendations[0]["candidate_bigram_soft_guard"],
            "0.0500",
        )
        self.assertIn("## Bigram Soft Guard Recommendations", report)
        self.assertIn("## Bigram Soft Guard Deltas", report)

    def test_char_lm_compare_summary_pairs_bigram_soft_guard_seed_deltas(
        self,
    ) -> None:
        base = {
            "arch": "llm_char_lstm",
            "recurrent": "lstm",
            "backend": "cpu",
            "head_prior": "learned-bigram",
            "head_resid": "0.5000",
            "bigram_guard": "0.1000",
            "bigram_guard_k": "5",
            "bigram_rank_guard": "0.0000",
            "bigram_rank_margin": "0.0500",
            "char_feature": "token-bigram",
            "mode": "embedding(8,token-bigram)",
            "steps": "12",
            "hidden": "16",
            "embed_dim": "8",
            "epochs": "10",
            "batches": "24",
            "batch": "4",
            "eval_samples": "128",
            "val_start": "0",
            "lr": "0.0025",
            "final_nll": "3.0182",
            "final_vs_bigram": "-0.0010",
            "final_top5_bigram_overlap": "88.14%",
        }
        payload = {
            "schema": "st.char_lm.compare.v1",
            "runs": [
                {
                    **base,
                    "seed": "7",
                    "bigram_soft_guard": "0.0000",
                    "final_bigram_rank_lift": "-3.1500",
                    "final_bigram_rank_debt": "3.1500",
                },
                {
                    **base,
                    "seed": "7",
                    "bigram_soft_guard": "0.0500",
                    "final_bigram_rank_lift": "-3.1300",
                    "final_bigram_rank_debt": "3.1300",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_soft_guard": "0.0000",
                    "final_bigram_rank_lift": "-2.6500",
                    "final_bigram_rank_debt": "2.6500",
                },
                {
                    **base,
                    "seed": "13",
                    "bigram_soft_guard": "0.0500",
                    "final_bigram_rank_lift": "-2.7000",
                    "final_bigram_rank_debt": "2.7000",
                },
            ],
        }

        pairs = summarize_char_lm_compare.paired_bigram_soft_guard_seed_deltas(
            [("probe/compare.json", payload)]
        )
        stability = summarize_char_lm_compare.bigram_soft_guard_stability_rows(pairs)
        report = summarize_char_lm_compare.markdown_report(
            [],
            bigram_soft_guard_seed_deltas=pairs,
            bigram_soft_guard_stability=stability,
        )

        self.assertEqual(len(pairs), 2)
        seed7 = next(pair for pair in pairs if pair["seed"] == "7")
        seed13 = next(pair for pair in pairs if pair["seed"] == "13")
        self.assertEqual(seed7["bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(seed7["bigram_rank_lift_delta"], "0.0200")
        self.assertEqual(seed7["guard_verdict"], "soft_guard_alignment_improved")
        self.assertEqual(seed7["alignment_status"], "improved")
        self.assertEqual(seed13["bigram_rank_debt_delta"], "0.0500")
        self.assertEqual(seed13["bigram_rank_lift_delta"], "-0.0500")
        self.assertEqual(seed13["guard_verdict"], "soft_guard_alignment_regressed")
        self.assertEqual(seed13["alignment_status"], "regressed")
        self.assertEqual(len(stability), 1)
        self.assertEqual(stability[0]["seed_pairs"], "2")
        self.assertEqual(stability[0]["alignment_improved_seeds"], "1")
        self.assertEqual(stability[0]["alignment_neutral_seeds"], "0")
        self.assertEqual(stability[0]["alignment_regressed_seeds"], "1")
        self.assertEqual(stability[0]["mean_bigram_rank_debt_delta"], "0.0150")
        self.assertEqual(stability[0]["min_bigram_rank_debt_delta"], "-0.0200")
        self.assertEqual(stability[0]["max_bigram_rank_debt_delta"], "0.0500")
        self.assertEqual(stability[0]["mean_bigram_rank_lift_delta"], "-0.0150")
        self.assertEqual(stability[0]["stability_verdict"], "soft_guard_seed_mixed")
        self.assertIn("## Bigram Soft Guard Stability", report)
        self.assertIn("## Bigram Soft Guard Seed Deltas", report)

    def test_char_lm_compare_summary_marks_bigram_guard_topk_mixed(self) -> None:
        verdict = summarize_char_lm_compare.bigram_guard_verdict(
            nll_status="improved",
            bigram_gap_status="neutral",
            bigram_logprob_status="improved",
            bigram_rank_status="regressed",
            top5_bigram_status="improved",
        )

        self.assertEqual(verdict, "guard_quality_improved_topk_mixed")

    def test_char_lm_compare_summary_recommends_topk_improved_bigram_guard(
        self,
    ) -> None:
        pairs = [
            {
                "source": "probe/compare.json",
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "bigram",
                "head_resid": "2.0000",
                "bigram_guard_k": "5",
                "char_feature": "token-bigram",
                "mode": "embedding(8,token-bigram)",
                "steps": "12",
                "hidden": "8",
                "embed_dim": "8",
                "epochs": "1",
                "batches": "8",
                "batch": "4",
                "eval_samples": "48",
                "lr": "0.02",
                "candidate_bigram_guard": "0.1000",
                "baseline_bigram_guard": "0.0000",
                "guard_verdict": "guard_topk_improved",
                "nll_status": "neutral",
                "bigram_gap_status": "neutral",
                "bigram_rank_status": "neutral",
                "top5_bigram_status": "improved",
                "final_nll_delta": "-0.0002",
                "final_vs_bigram_delta": "-0.0004",
                "bigram_logprob_lift_delta": "0.0004",
                "bigram_rank_lift_delta": "0.0000",
                "top5_bigram_overlap_delta_pp": "0.8200",
                "candidate_final_nll": "2.7651",
                "candidate_route_status": "clean_route",
                "baseline_route_status": "clean_route",
            },
            {
                "source": "probe/compare.json",
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "bigram",
                "head_resid": "1.0000",
                "bigram_guard_k": "5",
                "char_feature": "token-bigram",
                "mode": "embedding(8,token-bigram)",
                "steps": "12",
                "hidden": "8",
                "embed_dim": "8",
                "epochs": "1",
                "batches": "8",
                "batch": "4",
                "eval_samples": "48",
                "lr": "0.02",
                "candidate_bigram_guard": "0.0500",
                "baseline_bigram_guard": "0.0000",
                "guard_verdict": "guard_topk_mixed",
                "nll_status": "neutral",
                "bigram_gap_status": "neutral",
                "bigram_rank_status": "regressed",
                "top5_bigram_status": "improved",
                "final_nll_delta": "-0.0001",
                "final_vs_bigram_delta": "0.0000",
                "bigram_logprob_lift_delta": "0.0000",
                "bigram_rank_lift_delta": "-0.0100",
                "top5_bigram_overlap_delta_pp": "0.8200",
                "candidate_final_nll": "2.7665",
                "candidate_route_status": "clean_route",
                "baseline_route_status": "clean_route",
            },
        ]

        recommendations = summarize_char_lm_compare.paired_bigram_guard_recommendations(
            pairs,
            limit=8,
        )

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["candidate_bigram_guard"], "0.1000")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "topk_improved_quality_neutral",
        )

    def test_char_lm_compare_summary_marks_quality_neutral_cost_better_pair(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_finetune",
                    "recurrent": "spiral",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "head_resid": "0.5000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "8",
                    "batches": "16",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.02",
                    "runs": "3",
                    "final_nll_mean": "2.0000",
                    "final_vs_bigram_mean": "0.5000",
                    "trace_step_ms_mean_mean": "100.0000",
                    "cpu_debt_ops_mean": "200.0000",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "head_prior": "learned-unigram",
                    "head_resid": "0.5000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(8,token-bigram)",
                    "steps": "12",
                    "hidden": "8",
                    "embed_dim": "8",
                    "epochs": "8",
                    "batches": "16",
                    "batch": "4",
                    "eval_samples": "32",
                    "lr": "0.02",
                    "runs": "3",
                    "final_nll_mean": "2.0001",
                    "delta_nll_mean": "-0.1000",
                    "final_vs_bigram_mean": "0.4999",
                    "trace_step_ms_mean_mean": "50.0000",
                    "cpu_debt_ops_mean": "100.0000",
                    "lstm_scan_backend_counts": "cpu:3",
                    "lstm_scan_fallback_counts": "none:3",
                },
            ],
        }

        pair = summarize_char_lm_compare.paired_recurrent_deltas(
            [("probe/compare.json", payload)]
        )[0]
        recommendations = summarize_char_lm_compare.paired_recurrent_recommendations(
            [pair],
            limit=8,
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            paired_deltas=[pair],
            paired_recommendations=recommendations,
        )

        self.assertEqual(pair["quality_status"], "neutral")
        self.assertEqual(pair["candidate_learning_status"], "improved")
        self.assertEqual(pair["latency_status"], "improved")
        self.assertEqual(pair["cpu_debt_status"], "improved")
        self.assertEqual(
            pair["efficiency_verdict"],
            "candidate_quality_neutral_cost_better",
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["rank"], "1")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "quality_neutral_cost_improved",
        )
        self.assertIn("## Paired Recurrent Recommendations", report)

    def test_char_lm_compare_summary_recommends_wave_lite_route_debt(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_wave",
                    "recurrent": "wave",
                    "backend": "cpu",
                    "head_prior": "none",
                    "head_resid": "5.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(32,token-bigram)",
                    "wave_kernel": "3",
                    "wave_dilations": "1",
                    "steps": "32",
                    "hidden": "64",
                    "embed_dim": "32",
                    "epochs": "8",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "64",
                    "lr": "0.05",
                    "runs": "2",
                    "final_nll_mean": "3.6698",
                    "best_nll_mean": "3.6698",
                    "final_vs_bigram_mean": "0.1200",
                    "trace_step_ms_mean_mean": "189.3720",
                    "cpu_debt_ops_mean": "114.0000",
                    "coherence_route_status": "cpu_route",
                    "coherence_route_debt_mean": "10.0000",
                },
                {
                    "arch": "llm_char_wave",
                    "recurrent": "wave",
                    "backend": "cpu",
                    "head_prior": "none",
                    "head_resid": "5.0000",
                    "char_feature": "token-bigram",
                    "mode": "embedding(32,token-bigram)",
                    "wave_kernel": "3",
                    "wave_dilations": "1,2,4",
                    "steps": "32",
                    "hidden": "64",
                    "embed_dim": "32",
                    "epochs": "8",
                    "batches": "24",
                    "batch": "4",
                    "eval_samples": "64",
                    "lr": "0.05",
                    "runs": "2",
                    "final_nll_mean": "3.6698",
                    "best_nll_mean": "3.6698",
                    "final_vs_bigram_mean": "0.1200",
                    "trace_step_ms_mean_mean": "306.5770",
                    "cpu_debt_ops_mean": "172.0000",
                    "coherence_route_status": "cpu_route",
                    "coherence_route_debt_mean": "20.0000",
                },
            ],
        }

        recommendations = summarize_char_lm_compare.route_debt_recommendations(
            [("wave/compare.json", payload)],
            limit=8,
        )
        decision = summarize_char_lm_compare.route_debt_recommendation_summary(
            recommendations
        )
        report = summarize_char_lm_compare.markdown_report(
            [],
            route_debt_recommendations=recommendations,
            route_debt_summary=decision,
        )

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["candidate_wave_dilations"], "1")
        self.assertEqual(recommendations[0]["baseline_wave_dilations"], "1,2,4")
        self.assertEqual(recommendations[0]["quality_status"], "neutral")
        self.assertEqual(recommendations[0]["route_debt_status"], "improved")
        self.assertEqual(recommendations[0]["route_debt_ratio"], "0.5000")
        self.assertEqual(
            recommendations[0]["recommendation"],
            "quality_neutral_route_debt_lower",
        )
        self.assertEqual(decision["decision"], "promote_lite_wave")
        self.assertEqual(decision["failed"], "false")
        self.assertEqual(decision["fail_on_decisions"], "")
        self.assertEqual(decision["recommendation_rows"], "1")
        self.assertEqual(decision["top_candidate_wave_dilations"], "1")
        self.assertEqual(decision["top_route_debt_ratio"], "0.5000")
        self.assertIn("## Route Debt Decision", report)
        self.assertIn("## Route Debt Recommendations", report)

    def test_char_lm_compare_summary_excludes_learning_regressed_recommendation(
        self,
    ) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_finetune",
                    "recurrent": "spiral",
                    "backend": "cpu",
                    "final_nll_mean": "2.0000",
                    "delta_nll_mean": "0.0100",
                    "final_vs_bigram_mean": "0.5000",
                    "trace_step_ms_mean_mean": "100.0000",
                    "cpu_debt_ops_mean": "200.0000",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "2.0000",
                    "delta_nll_mean": "0.0100",
                    "final_vs_bigram_mean": "0.5000",
                    "trace_step_ms_mean_mean": "50.0000",
                    "cpu_debt_ops_mean": "100.0000",
                },
            ],
        }

        pair = summarize_char_lm_compare.paired_recurrent_deltas(
            [("probe/compare.json", payload)]
        )[0]
        recommendations = summarize_char_lm_compare.paired_recurrent_recommendations(
            [pair],
            limit=8,
        )

        self.assertEqual(pair["quality_status"], "neutral")
        self.assertEqual(pair["candidate_learning_status"], "regressed")
        self.assertEqual(pair["efficiency_verdict"], "candidate_quality_neutral_cost_better")
        self.assertEqual(recommendations, [])

    def test_char_lm_compare_summary_paired_delta_headers_are_unique(self) -> None:
        headers = summarize_char_lm_compare.PAIR_DELTA_HEADERS

        self.assertEqual(len(headers), len(set(headers)))
        self.assertIn("candidate_learning_status", headers)
        self.assertIn("baseline_learning_status", headers)

    def test_char_lm_compare_summary_cli_gate_checks_paired_quality(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_finetune",
                    "recurrent": "spiral",
                    "backend": "cpu",
                    "final_nll_mean": "2.0000",
                    "final_vs_bigram_mean": "0.5000",
                    "trace_step_ms_mean_mean": "100.0000",
                    "cpu_debt_ops_mean": "200.0000",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "2.1000",
                    "final_vs_bigram_mean": "0.3000",
                    "trace_step_ms_mean_mean": "50.0000",
                    "cpu_debt_ops_mean": "100.0000",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compare_path = root / "compare.json"
            json_out = root / "summary.json"
            compare_path.write_text(json.dumps(payload), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = summarize_char_lm_compare.main(
                    [
                        str(compare_path),
                        "--json-out",
                        str(json_out),
                        "--fail-on-paired-quality-status",
                        "regressed",
                    ]
                )
            summary = json.loads(json_out.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        gate = summary["paired_recurrent_gate"]
        self.assertTrue(gate["failed"])
        self.assertEqual(gate["fail_on_quality_statuses"], ["regressed"])
        self.assertEqual(gate["failures"]["quality_statuses"], {"regressed": 1})
        self.assertEqual(gate["failures"]["pairs"][0]["quality_status"], "regressed")
        self.assertIn("paired recurrent gate failed", stderr.getvalue())

    def test_char_lm_compare_summary_cli_gate_checks_rank_min_promotion(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "2.0000",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compare_path = root / "compare.json"
            json_out = root / "summary.json"
            compare_path.write_text(json.dumps(payload), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = summarize_char_lm_compare.main(
                    [
                        str(compare_path),
                        "--json-out",
                        str(json_out),
                        "--fail-on-rank-min-promotion-decision",
                        "no_rank_min_evidence",
                    ]
                )
            summary = json.loads(json_out.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        gate = summary["bigram_rank_min_promotion_gate"]
        self.assertEqual(gate["decision"], "no_rank_min_evidence")
        self.assertEqual(gate["failed"], "true")
        self.assertEqual(gate["fail_on_decisions"], "no_rank_min_evidence")
        self.assertIn("rank-min promotion gate failed", stderr.getvalue())

    def test_char_lm_compare_summary_cli_gate_checks_route_debt_decision(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_wave",
                    "recurrent": "wave",
                    "backend": "cpu",
                    "head_prior": "none",
                    "final_nll_mean": "2.0000",
                    "best_nll_mean": "2.0000",
                    "coherence_route_debt_mean": "10.0000",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compare_path = root / "compare.json"
            json_out = root / "summary.json"
            compare_path.write_text(json.dumps(payload), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = summarize_char_lm_compare.main(
                    [
                        str(compare_path),
                        "--json-out",
                        str(json_out),
                        "--fail-on-route-debt-decision",
                        "no_route_debt_recommendation",
                    ]
                )
            summary = json.loads(json_out.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        gate = summary["route_debt_recommendation_summary"]
        self.assertEqual(gate["decision"], "no_route_debt_recommendation")
        self.assertEqual(gate["failed"], "true")
        self.assertEqual(gate["fail_on_decisions"], "no_route_debt_recommendation")
        self.assertIn("route-debt decision gate failed", stderr.getvalue())

    def test_char_lm_compare_summary_does_not_penalize_non_lstm_no_scan_route(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "final_vs_bigram_mean": "0.3000",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
                {
                    "arch": "llm_char_finetune",
                    "recurrent": "spiral",
                    "backend": "cpu",
                    "final_nll_mean": "0.9000",
                    "best_nll_mean": "0.9000",
                    "final_vs_bigram_mean": "0.2000",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "-",
                    "lstm_scan_fallback_counts": "-",
                },
            ],
        }

        rows = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=2,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_nll",
        )

        self.assertEqual(rows[0]["arch"], "llm_char_finetune")
        self.assertEqual(rows[0]["route_status"], "no_scan_route")
        self.assertEqual(rows[1]["route_status"], "clean_route")

    def test_char_lm_compare_summary_penalizes_lstm_no_scan_route(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "0.9000",
                    "best_nll_mean": "0.9000",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "-",
                    "lstm_scan_fallback_counts": "-",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        rows = summarize_char_lm_compare.summarize_rows(
            payload,
            limit=2,
            route_clean_only=False,
            prefer_clean_route=True,
            sort_metric="final_nll",
        )

        self.assertEqual(rows[0]["route_status"], "clean_route")
        self.assertEqual(rows[1]["route_status"], "no_scan_route")

    def test_char_lm_compare_summary_merges_multiple_sources(self) -> None:
        fallback_payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "wgpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "no suitable WGPU adapter:1",
                }
            ],
        }
        clean_payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.1000",
                    "best_nll_mean": "1.1000",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                }
            ],
        }

        rows = summarize_char_lm_compare.summarize_compare_payloads(
            [("fallback/compare.json", fallback_payload), ("clean/compare.json", clean_payload)],
            limit=2,
            route_clean_only=False,
            prefer_clean_route=True,
        )

        self.assertEqual(rows[0]["route_status"], "clean_route")
        self.assertEqual(rows[0]["source"], "clean/compare.json")
        self.assertEqual(rows[1]["route_status"], "scan_fallback")
        counts = summarize_char_lm_compare.route_status_counts(rows)
        all_counts = summarize_char_lm_compare.route_status_counts_for_payloads(
            [("fallback/compare.json", fallback_payload), ("clean/compare.json", clean_payload)]
        )
        selected_clean = summarize_char_lm_compare.summarize_compare_payloads(
            [("fallback/compare.json", fallback_payload), ("clean/compare.json", clean_payload)],
            limit=1,
            route_clean_only=False,
            prefer_clean_route=True,
        )
        selected_counts = summarize_char_lm_compare.route_status_counts(selected_clean)
        report = summarize_char_lm_compare.markdown_report(rows)
        self.assertEqual(counts["rows"], 2)
        self.assertEqual(counts["clean_route"], 1)
        self.assertEqual(counts["scan_fallback"], 1)
        self.assertEqual(all_counts["scan_fallback"], 1)
        self.assertEqual(selected_counts["scan_fallback"], 0)
        self.assertIn("## Route Status Counts", report)
        self.assertIn("selected", report)
        self.assertIn("## Compare Summary", report)

    def test_char_lm_compare_summary_cli_gate_checks_all_candidates(self) -> None:
        payload = {
            "schema": "st.char_lm.compare.v1",
            "aggregate_runs": [
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "wgpu",
                    "final_nll_mean": "1.0000",
                    "best_nll_mean": "1.0000",
                    "cpu_debt_ops_mean": "10.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "no suitable WGPU adapter:1",
                },
                {
                    "arch": "llm_char_lstm",
                    "recurrent": "lstm",
                    "backend": "cpu",
                    "final_nll_mean": "1.1000",
                    "best_nll_mean": "1.1000",
                    "cpu_debt_ops_mean": "20.0000",
                    "lstm_scan_backend_counts": "cpu:1",
                    "lstm_scan_fallback_counts": "none:1",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compare_path = root / "compare.json"
            json_out = root / "summary.json"
            compare_path.write_text(json.dumps(payload), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = summarize_char_lm_compare.main(
                    [
                        str(compare_path),
                        "--route-clean-only",
                        "--json-out",
                        str(json_out),
                        "--fail-on-route-status",
                        "scan_fallback",
                    ]
                )
            summary = json.loads(json_out.read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        self.assertEqual(summary["route_status_counts"]["scan_fallback"], 1)
        self.assertEqual(summary["selected_route_status_counts"]["scan_fallback"], 0)
        self.assertTrue(summary["route_status_gate"]["failed"])
        self.assertEqual(summary["sort_metric"], "final_nll")
        self.assertEqual(summary["route_status_gate"]["fail_on"], ["scan_fallback"])
        self.assertEqual(summary["route_status_gate"]["failures"], {"scan_fallback": 1})
        self.assertIn("route status gate failed", stderr.getvalue())

    def test_char_lm_compare_summary_resolves_directories_and_recursive_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct"
            nested = root / "nested" / "child"
            direct.mkdir()
            nested.mkdir(parents=True)
            (direct / "compare.json").write_text("{}", encoding="utf-8")
            (nested / "compare.json").write_text("{}", encoding="utf-8")

            direct_paths = summarize_char_lm_compare.resolve_compare_paths(
                [direct, direct / "compare.json"],
                recursive=False,
            )
            recursive_paths = summarize_char_lm_compare.resolve_compare_paths(
                [root],
                recursive=True,
            )

        self.assertEqual(len(direct_paths), 1)
        self.assertEqual(direct_paths[0].name, "compare.json")
        self.assertEqual(len(recursive_paths), 2)
        self.assertTrue(all(path.name == "compare.json" for path in recursive_paths))

    def test_char_lm_compare_aggregates_recurrent_groups(self) -> None:
        rows = [
            {
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "head_resid": "1.0000",
                "bigram_guard": "0.0500",
                "bigram_guard_k": "3",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "val_start": "0.5000",
                "val_start_actual": "0.5000",
                "final_windows": "4",
                "unigram_windows": "4",
                "bigram_windows": "4",
                "final_nll": "2.0000",
                "delta_nll": "-0.1000",
                "final_vs_bigram": "0.2000",
                "final_bigram_logprob_lift": "0.1000",
                "final_bigram_rank_lift": "1.00",
                "final_bigram_rank_lift_raw": "1.12500000",
                "final_bigram_target_rank": "4.00",
                "final_bigram_target_rank_raw": "4.12500000",
                "final_bigram_rank_debt": "-1.00",
                "final_bigram_rank_debt_raw": "-1.12500000",
                "final_kl_bigram": "0.0100",
                "final_top5_bigram_overlap": "20.00%",
                "final_top5_bigram_overlap_raw": "20.12500000",
                "lstm_est_cpu_debt_ops": "10",
                "lstm_est_gate_wgpu_ops": "2",
                "lstm_scan_backend": "wgpu",
                "lstm_scan_fallback": "-",
            },
            {
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "head_resid": "1.0000",
                "bigram_guard": "0.0500",
                "bigram_guard_k": "3",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "val_start": "0.5000",
                "val_start_actual": "0.5000",
                "final_windows": "8",
                "unigram_windows": "8",
                "bigram_windows": "8",
                "final_nll": "4.0000",
                "delta_nll": "-0.3000",
                "final_vs_bigram": "0.4000",
                "final_bigram_logprob_lift": "0.3000",
                "final_bigram_rank_lift": "3.00",
                "final_bigram_rank_lift_raw": "3.37500000",
                "final_bigram_target_rank": "6.00",
                "final_bigram_target_rank_raw": "6.37500000",
                "final_bigram_rank_debt": "-3.00",
                "final_bigram_rank_debt_raw": "-3.37500000",
                "final_kl_bigram": "0.0300",
                "final_top5_bigram_overlap": "60.00%",
                "final_top5_bigram_overlap_raw": "60.37500000",
                "lstm_est_cpu_debt_ops": "14",
                "lstm_est_gate_wgpu_ops": "4",
                "lstm_scan_backend": "cpu",
                "lstm_scan_fallback": "no suitable WGPU adapter",
            },
        ]

        aggregate = compare_char_lm_runs.aggregate_rows(rows)
        table = compare_char_lm_runs.aggregate_table(rows)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["runs"], "2")
        self.assertEqual(aggregate[0]["final_windows_mean"], "6.0000")
        self.assertEqual(aggregate[0]["unigram_windows_mean"], "6.0000")
        self.assertEqual(aggregate[0]["bigram_windows_mean"], "6.0000")
        self.assertEqual(aggregate[0]["val_start"], "0.5000")
        self.assertEqual(aggregate[0]["val_start_actual_mean"], "0.5000")
        self.assertEqual(aggregate[0]["final_nll_mean"], "3.0000")
        self.assertEqual(aggregate[0]["bigram_guard"], "0.0500")
        self.assertEqual(aggregate[0]["bigram_guard_k"], "3")
        self.assertEqual(aggregate[0]["final_nll_std"], "1.0000")
        self.assertEqual(aggregate[0]["delta_nll_mean"], "-0.2000")
        self.assertEqual(aggregate[0]["final_vs_bigram_mean"], "0.3000")
        self.assertEqual(aggregate[0]["final_bigram_logprob_lift_mean"], "0.2000")
        self.assertEqual(aggregate[0]["final_bigram_rank_lift_mean"], "2.0000")
        self.assertEqual(aggregate[0]["final_bigram_rank_lift_raw_mean"], "2.25000000")
        self.assertEqual(aggregate[0]["final_bigram_target_rank_mean"], "5.0000")
        self.assertEqual(
            aggregate[0]["final_bigram_target_rank_raw_mean"],
            "5.25000000",
        )
        self.assertEqual(aggregate[0]["final_bigram_rank_debt_mean"], "-2.0000")
        self.assertEqual(
            aggregate[0]["final_bigram_rank_debt_raw_mean"],
            "-2.25000000",
        )
        self.assertEqual(aggregate[0]["final_kl_bigram_mean"], "0.0200")
        self.assertEqual(aggregate[0]["final_top5_bigram_overlap_mean"], "40.00%")
        self.assertEqual(
            aggregate[0]["final_top5_bigram_overlap_raw_mean"],
            "40.25000000",
        )
        self.assertEqual(aggregate[0]["lstm_est_cpu_debt_ops_mean"], "12.0000")
        self.assertEqual(aggregate[0]["lstm_est_gate_wgpu_ops_mean"], "3.0000")
        self.assertEqual(aggregate[0]["route_status"], "scan_fallback")
        self.assertEqual(aggregate[0]["lstm_scan_backend_counts"], "cpu:1,wgpu:1")
        self.assertEqual(
            aggregate[0]["lstm_scan_fallback_counts"],
            "no suitable WGPU adapter:1,none:1",
        )
        self.assertIn("## Aggregate Runs", table)
        self.assertIn("scan_fallback", table)
        self.assertIn("final_bigram_logprob_lift_mean", table)
        self.assertIn("final_bigram_rank_debt_mean", table)
        self.assertIn("final_top5_bigram_overlap_mean", table)
        self.assertIn("lstm_scan_backend_counts", table)

    def test_char_lm_compare_keeps_head_residual_scale_as_aggregate_key(self) -> None:
        rows = [
            {
                "arch": "llm_char_finetune",
                "recurrent": "spiral",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "head_resid": "1.0000",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "final_nll": "2.0000",
                "delta_nll": "0.0000",
            },
            {
                "arch": "llm_char_finetune",
                "recurrent": "spiral",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "head_resid": "2.0000",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "final_nll": "4.0000",
                "delta_nll": "0.0000",
            },
        ]

        aggregate = compare_char_lm_runs.aggregate_rows(rows)

        self.assertEqual(len(aggregate), 2)
        self.assertEqual([row["head_resid"] for row in aggregate], ["1.0000", "2.0000"])

    def test_char_lm_compare_aggregate_hides_lstm_scan_counts_without_route(self) -> None:
        rows = [
            {
                "arch": "llm_char_finetune",
                "recurrent": "-",
                "backend": "cpu",
                "head_prior": "none",
                "head_resid": "-",
                "char_feature": "token",
                "mode": "embedding(4,token)",
                "final_nll": "2.0000",
                "delta_nll": "0.0000",
            }
        ]

        aggregate = compare_char_lm_runs.aggregate_rows(rows)
        table = compare_char_lm_runs.aggregate_table(rows)

        self.assertEqual(aggregate[0]["lstm_scan_backend_counts"], "-")
        self.assertEqual(aggregate[0]["lstm_scan_fallback_counts"], "-")
        self.assertEqual(aggregate[0]["route_status"], "-")
        self.assertNotIn("lstm_scan_backend_counts", table)

    def test_char_lm_compare_ranks_top_aggregate_runs_by_final_nll(self) -> None:
        rows = [
            {
                "arch": "llm_char_finetune",
                "recurrent": "spiral",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "steps": "4",
                "hidden": "8",
                "embed_dim": "4",
                "epochs": "1",
                "batches": "1",
                "batch": "2",
                "eval_samples": "4",
                "lr": "0.02",
                "final_nll": "4.0000",
                "best_nll": "4.0000",
            },
            {
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "steps": "4",
                "hidden": "8",
                "embed_dim": "4",
                "epochs": "1",
                "batches": "1",
                "batch": "2",
                "eval_samples": "4",
                "lr": "0.02",
                "final_nll": "3.0000",
                "best_nll": "3.0000",
                "lstm_scan_backend": "cpu",
                "lstm_scan_fallback": "no suitable WGPU adapter",
            },
        ]

        aggregate = compare_char_lm_runs.aggregate_rows(rows)
        ranked = compare_char_lm_runs.ranked_aggregate_rows(aggregate)
        table = compare_char_lm_runs.aggregate_table(rows)

        self.assertEqual(ranked[0]["arch"], "llm_char_lstm")
        self.assertIn("## Top Aggregate Runs", table)
        top_index = table.index("## Top Aggregate Runs")
        self.assertLess(
            table.index("llm_char_lstm", top_index),
            table.index("llm_char_finetune", top_index),
        )
        self.assertIn("lstm_scan_fallback_counts", table[top_index:])
        self.assertIn("no suitable WGPU adapter:1", table[top_index:])
        self.assertIn("scan_fallback", table[top_index:])

    def test_char_lm_compare_tie_breaks_top_aggregate_runs_by_latency(self) -> None:
        rows = [
            {
                "arch": "llm_char_finetune",
                "recurrent": "spiral",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "steps": "12",
                "hidden": "8",
                "embed_dim": "4",
                "epochs": "1",
                "batches": "1",
                "batch": "2",
                "eval_samples": "4",
                "lr": "0.02",
                "final_nll": "3.0000",
                "best_nll": "3.0000",
                "trace_step_ms_mean": "50.0000",
                "cpu_debt_ops": "1000",
            },
            {
                "arch": "llm_char_lstm",
                "recurrent": "lstm",
                "backend": "cpu",
                "head_prior": "learned-unigram",
                "char_feature": "token-bigram",
                "mode": "embedding(4,token-bigram)",
                "steps": "12",
                "hidden": "8",
                "embed_dim": "4",
                "epochs": "1",
                "batches": "1",
                "batch": "2",
                "eval_samples": "4",
                "lr": "0.02",
                "final_nll": "3.0000",
                "best_nll": "3.0000",
                "trace_step_ms_mean": "25.0000",
                "cpu_debt_ops": "500",
                "lstm_scan_backend": "cpu",
                "lstm_scan_fallback": "-",
            },
        ]

        ranked = compare_char_lm_runs.ranked_aggregate_rows(
            compare_char_lm_runs.aggregate_rows(rows)
        )

        self.assertEqual(ranked[0]["arch"], "llm_char_lstm")

    def test_backend_audit_reports_failed_sweep_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "backend-wgpu__seed-9"
            run_dir.mkdir()
            (run_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "schema": "st.test.sweep_failure.v1",
                        "backend": "wgpu",
                        "run_status": "failed",
                        "failed": True,
                        "returncode": -6,
                        "failure_kind": "preflight_signal",
                        "failure_detail": "signal:6",
                        "run_dir": str(run_dir),
                        "log_path": str(run_dir / "process.log"),
                    }
                ),
                encoding="utf-8",
            )
            (root / "sweep.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "name": "cpu-run",
                                "backend": "cpu",
                                "run_status": "ok",
                                "failed": False,
                            },
                            {
                                "name": "wgpu-run",
                                "backend": "wgpu",
                                "run_status": "failed",
                                "failed": True,
                                "returncode": -6,
                                "failure_kind": "preflight_signal",
                                "failure_detail": "signal:6",
                                "run_dir": str(run_dir),
                                "log_path": str(run_dir / "process.log"),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = audit_learning_backend_backlog.render_report(
                run_root=root,
                source_roots=(),
                include_tests=False,
                include_cpu_runs=False,
                max_candidates=10,
            )

        self.assertIn("## Failed Runs", report)
        self.assertIn("backend-wgpu__seed-9", report)
        self.assertIn("preflight_signal", report)
        self.assertIn("signal:6", report)
        self.assertNotIn("cpu-run", report)

    def test_backend_residual_columns_separate_control_plane_cpu_ops(self) -> None:
        summary = {
            "metrics": {
                "tensor_ops_total": {"last": 5},
                "tensor_op_backend_distributed_trainer_sync_step_cpu": {"last": 2},
                "tensor_op_backend_onebit_allreduce_hook_cpu": {"last": 1},
                "tensor_op_backend_graph_flow_layer_begin_cpu": {"last": 1},
                "tensor_op_backend_matmul_cpu": {"last": 1},
            }
        }

        columns = backend_sweep_meta.backend_residual_columns(summary)
        row = backend_sweep_meta.backend_residual_row(summary)

        self.assertEqual(len(row), len(backend_sweep_meta.BACKEND_RESIDUAL_HEADERS))
        self.assertEqual(columns["cpu_control_ops"], "3")
        self.assertEqual(
            columns["cpu_control_top"],
            "distributed_trainer_sync_step:2,onebit_allreduce_hook:1",
        )
        self.assertEqual(columns["cpu_trace_ops"], "1")
        self.assertEqual(columns["cpu_trace_top"], "graph_flow_layer_begin:1")
        self.assertEqual(columns["cpu_debt_ops"], "1")
        self.assertEqual(columns["cpu_debt_top"], "matmul:1")

    def test_backend_residual_columns_subtract_wgpu_runtime_fallbacks(self) -> None:
        summary = {
            "metrics": {
                "tensor_ops_total": {"last": 4},
                "tensor_op_backend_matmul_naive": {"last": 3},
                "tensor_op_backend_scale_cpu": {"last": 1},
                "tensor_op_backend_lstm_forward_input_projection_naive": {"last": 3},
                "tensor_op_backend_lstm_forward_gate_activation_cpu": {"last": 2},
                "tensor_op_backend_wgpu_runtime_fallback_matmul_naive": {"last": 2},
            }
        }

        columns = backend_sweep_meta.backend_residual_columns(summary)

        self.assertEqual(columns["cpu_runtime_fallback_ops"], "2")
        self.assertEqual(columns["cpu_runtime_fallback_top"], "matmul:2")
        self.assertEqual(columns["cpu_debt_ops"], "4")
        self.assertEqual(
            columns["cpu_debt_top"],
            "lstm_forward_gate_activation:2,matmul:1,scale:1",
        )

    def test_char_lm_compare_surfaces_lstm_gate_wgpu_estimated_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            trace_path = run_dir / "trainer_trace.jsonl"
            records = [
                {
                    "idx": 1,
                    "event": {
                        "kind": "Custom",
                        "data": {
                            "event_type": "TensorOpMeta",
                            "data": {
                                "op_name": "lstm_forward",
                                "data": {
                                    "gate_activation_backend": "wgpu",
                                    "estimated_gate_activation_ops": 10,
                                },
                            },
                        },
                    },
                },
                {
                    "idx": 2,
                    "event": {
                        "kind": "Custom",
                        "data": {
                            "event_type": "TensorOpMeta",
                            "data": {
                                "op_name": "lstm_backward",
                                "data": {
                                    "gate_activation_backend": "wgpu",
                                    "bptt_backend": "wgpu",
                                    "estimated_gate_activation_ops": 6,
                                    "estimated_bptt_ops": 12,
                                    "estimated_bptt_wgpu_ops": 12,
                                    "bptt_scan_backend": "wgpu",
                                    "bptt_scan_kernel": "lstm_backward_scan.wgsl",
                                    "bptt_scan_lowering": (
                                        "wgpu_single_workgroup_hidden_parallel_recurrence"
                                    ),
                                },
                            },
                        },
                    },
                },
            ]
            trace_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            columns = compare_char_lm_runs.trace_learning_op_columns(run_dir)
            route_columns = compare_char_lm_runs.trace_lstm_scan_route_columns(run_dir)

        self.assertEqual(columns["lstm_fwd_gate_wgpu"], "1")
        self.assertEqual(columns["lstm_bwd_gate_wgpu"], "1")
        self.assertEqual(columns["lstm_est_fwd_gate_wgpu_ops"], "10")
        self.assertEqual(columns["lstm_est_bwd_gate_wgpu_ops"], "6")
        self.assertEqual(columns["lstm_est_gate_wgpu_ops"], "16")
        self.assertEqual(columns["lstm_est_bptt_wgpu_ops"], "12")
        self.assertEqual(columns["lstm_est_cpu_debt_ops"], "0")
        self.assertEqual(route_columns["lstm_scan_backend"], "wgpu")
        self.assertEqual(route_columns["lstm_scan_kernel"], "lstm_backward_scan.wgsl")
        self.assertEqual(
            route_columns["lstm_scan_lowering"],
            "wgpu_single_workgroup_hidden_parallel_recurrence",
        )
        self.assertEqual(route_columns["lstm_scan_fallback"], "-")

    def test_lstm_scan_profile_grid_json_fields_are_machine_readable(self) -> None:
        row = {
            "trace_summary": {"last_loss": 0.125},
            "trainer_summary": {
                "metrics": {
                    "step_time_ms": {"last": 2.5},
                    "tensor_ops_total": {"last": 10},
                    "tensor_backend_wgpu": {"last": 7},
                    "tensor_backend_fallbacks": {"last": 1},
                    "lstm_estimated_cpu_debt_ops": {"last": 0},
                    "lstm_estimated_bptt_wgpu_ops": {"last": 120},
                    "lstm_backward_estimated_bptt_ops": {"last": 120},
                    "lstm_backward_bptt_scan_runtime_requested": {"last": 1},
                    "lstm_backward_bptt_scan_runtime_available": {"last": 1},
                    "lstm_backward_bptt_scan_runtime_unavailable": {"last": 0},
                    "lstm_backward_bptt_scan_elapsed_us": {"last": 40},
                    "lstm_backward_bptt_scan_hidden_values": {"last": 12},
                    "lstm_backward_bptt_scan_gate_values": {"last": 48},
                    "lstm_backward_bptt_scan_recurrent_weight_values": {"last": 64},
                    "lstm_backward_bptt_scan_kernel_dispatches": {"last": 1},
                    "lstm_backward_bptt_scan_serial_steps": {"last": 3},
                    "lstm_backward_bptt_scan_workgroup_size": {"last": 64},
                    "lstm_backward_bptt_scan_parallel_lanes": {"last": 64},
                    "lstm_backward_estimated_bptt_ops_per_scan_step": {"last": 40},
                }
            },
            "run_meta": {
                "backend_runtime": {
                    "requested_backend_status": "kernel_wired",
                    "requested_backend_kernels_wired": True,
                }
            },
            "scan_route": {
                "backend": "wgpu",
                "kernel": "lstm_backward_scan.wgsl",
                "lowering": "wgpu_single_workgroup_hidden_parallel_recurrence",
                "fallback_reason": None,
            },
        }

        profile = run_lstm_scan_profile_grid.scan_profile_fields(row)

        self.assertEqual(profile["last_loss"], 0.125)
        self.assertEqual(profile["lstm_scan_us"], 40.0)
        self.assertEqual(profile["lstm_scan_gate_values"], 48.0)
        self.assertEqual(profile["lstm_scan_dispatches"], 1.0)
        self.assertEqual(profile["lstm_scan_workgroup"], 64.0)
        self.assertEqual(profile["lstm_scan_parallel_lanes"], 64.0)
        self.assertEqual(profile["lstm_scan_parallel_axis"], "hidden")
        self.assertEqual(profile["lstm_scan_backend"], "wgpu")
        self.assertEqual(profile["lstm_scan_kernel"], "lstm_backward_scan.wgsl")
        self.assertEqual(
            profile["lstm_scan_lowering"],
            "wgpu_single_workgroup_hidden_parallel_recurrence",
        )
        self.assertIsNone(profile["lstm_scan_fallback"])
        self.assertEqual(profile["lstm_est_bptt_ops_per_us"], 3.0)
        self.assertEqual(profile["backend_status"], "kernel_wired")
        self.assertEqual(profile["backend_kernels"], "yes")

        average = dict(
            zip(
                run_lstm_scan_profile_grid.AVERAGE_HEADERS,
                run_lstm_scan_profile_grid.average_values(
                    [row],
                    steps=3,
                    hidden=2,
                    backend="wgpu",
                ),
            )
        )
        self.assertEqual(average["lstm_scan_backend_counts"], "wgpu:1")
        self.assertEqual(average["lstm_scan_fallback_counts"], "none:1")

    def test_lstm_scan_profile_grid_extracts_latest_scan_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            trace_path = run_dir / "trainer_trace.jsonl"
            records = [
                {
                    "idx": 1,
                    "event": {
                        "kind": "Custom",
                        "data": {
                            "event_type": "TensorOpMeta",
                            "data": {
                                "op_name": "lstm_backward",
                                "data": {
                                    "bptt_scan_backend": "cpu",
                                    "bptt_scan_kernel": "lstm_backward_scan.cpu_fused_loop",
                                    "bptt_scan_lowering": "host_reverse_recurrent_scan",
                                },
                            },
                        },
                    },
                },
                {
                    "idx": 2,
                    "event": {
                        "kind": "Custom",
                        "data": {
                            "event_type": "TensorOpMeta",
                            "data": {
                                "op_name": "lstm_backward",
                                "data": {
                                    "bptt_scan_backend": "cpu",
                                    "bptt_scan_kernel": "lstm_backward_scan.cpu_fused_loop",
                                    "bptt_scan_lowering": "host_reverse_recurrent_scan",
                                    "bptt_scan_fallback_reason": "wgpu context unavailable",
                                },
                            },
                        },
                    },
                },
            ]
            trace_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            route = run_lstm_scan_profile_grid.latest_lstm_scan_route(run_dir)

        self.assertEqual(route["backend"], "cpu")
        self.assertEqual(route["kernel"], "lstm_backward_scan.cpu_fused_loop")
        self.assertEqual(route["lowering"], "host_reverse_recurrent_scan")
        self.assertEqual(route["fallback_reason"], "wgpu context unavailable")

    def test_backend_audit_trace_rows_surface_lstm_gate_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "backend-cpu__seed-1"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"backend": "cpu"}),
                encoding="utf-8",
            )
            records = [
                {
                    "idx": 1,
                    "event": {
                        "kind": "Custom",
                        "data": {
                            "event_type": "TensorOpMeta",
                            "data": {
                                "op_name": "lstm_forward",
                                "data": {
                                    "gate_activation_backend": "cpu",
                                    "estimated_gate_activation_ops": 8,
                                },
                            },
                        },
                    },
                },
                {
                    "idx": 2,
                    "event": {
                        "kind": "Custom",
                        "data": {
                            "event_type": "TensorOpMeta",
                            "data": {
                                "op_name": "lstm_backward",
                                "data": {
                                    "gate_activation_backend": "wgpu",
                                    "bptt_backend": "wgpu",
                                    "estimated_gate_activation_ops": 4,
                                    "estimated_bptt_ops": 16,
                                    "estimated_bptt_wgpu_ops": 16,
                                    "bptt_scan_runtime_requested": True,
                                    "bptt_scan_runtime_available": True,
                                },
                            },
                        },
                    },
                },
            ]
            (run_dir / "trainer_trace.jsonl").write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                encoding="utf-8",
            )

            rows = audit_learning_backend_backlog._trace_rows(root, include_cpu_runs=True)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["lstm_est_cpu_debt_ops"], "8")
        self.assertEqual(row["lstm_est_gate_cpu_debt_ops"], "8")
        self.assertEqual(row["lstm_est_gate_wgpu_ops"], "4")
        self.assertEqual(row["lstm_est_bptt_cpu_debt_ops"], "0")
        self.assertEqual(row["lstm_est_bptt_wgpu_ops"], "16")
        self.assertEqual(row["lstm_scan_rt_req"], "1")
        self.assertEqual(row["lstm_scan_rt_ok"], "1")


if __name__ == "__main__":
    unittest.main()
