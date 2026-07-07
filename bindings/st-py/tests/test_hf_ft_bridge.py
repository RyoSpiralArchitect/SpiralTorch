from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import spiraltorch as st
import spiraltorch.runtime_imports as runtime_imports
from spiraltorch import hf_ft


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "hf_gpt2_finetune_bridge.py"
)
SWEEP_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "hf_gpt2_finetune_sweep.py"
)


def load_bridge_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_bridge_test",
        EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_sweep_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_sweep_test",
        SWEEP_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = ["text"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])

    def train_test_split(self, *, test_size, seed):
        del seed
        test_count = max(1, int(round(len(self.rows) * float(test_size))))
        test_count = min(test_count, len(self.rows) - 1)
        return {
            "train": FakeDataset(self.rows[:-test_count]),
            "test": FakeDataset(self.rows[-test_count:]),
        }


class FakeDatasets:
    def __init__(self):
        self.calls = []

    def load_dataset(self, dataset_format, *, data_files):
        self.calls.append((dataset_format, data_files))
        return {
            "train": FakeDataset(
                [
                    {"text": "alpha spiral"},
                    {"text": "beta zspace"},
                    {"text": "gamma torch"},
                    {"text": "delta local"},
                ]
            )
        }


class HuggingFaceFineTuneBridgeTest(unittest.TestCase):
    def test_rust_dependency_report_names_required_surfaces(self) -> None:
        report = hf_ft.hf_gpt2_finetune_rust_dependency_report()

        self.assertEqual(
            report["row_type"],
            "hf_gpt2_finetune_rust_dependency_report",
        )
        self.assertIn("st-tensor", report["rust_surface_crates"])
        self.assertIn("st-nn", report["rust_surface_crates"])
        self.assertIn("st-backend-wgpu", report["rust_surface_crates"])
        self.assertIn("transformers", report["python_package_label"])
        self.assertIn("pyarrow", report["python_package_label"])

    def test_preflight_defaults_require_hf_gpt2_ft_and_report_backends(self) -> None:
        available = set(hf_ft.HF_GPT2_FT_REQUIRED_PYTHON_PACKAGES)

        def fake_import(name: str):
            if name not in available:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            module = types.ModuleType(name)
            module.__version__ = f"{name}-test"
            return module

        def fake_describe(backends, *, continue_on_error=True):
            self.assertTrue(continue_on_error)
            return {
                "reports": [
                    {
                        "backend": backend,
                        "requested_backend": backend,
                        "runtime_ready": True,
                        "runtime_status": "ready",
                    }
                    for backend in runtime_imports.csv_values(backends)
                ]
            }

        with mock.patch.object(
            runtime_imports.importlib,
            "import_module",
            side_effect=fake_import,
        ):
            report = hf_ft.hf_gpt2_finetune_preflight_report(
                describe_runtime_devices=fake_describe,
            )

        self.assertTrue(report["runtime_import_preflight_passed"])
        self.assertEqual(report["required_runtime_import_presets"], "hf-gpt2-ft")
        self.assertEqual(report["runtime_device_report_backends"], "wgpu,cpu")
        self.assertEqual(report["runtime_device_report_ready_backends"], "wgpu,cpu")
        self.assertEqual(report["hf_model_name"], "gpt2")
        self.assertEqual(report["hf_dataset_name"], "wikitext")

    def test_preflight_can_report_missing_without_requiring_preset(self) -> None:
        def fake_import(name: str):
            if name in {"transformers", "torch", "tokenizers"}:
                return types.ModuleType(name)
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        with mock.patch.object(
            runtime_imports.importlib,
            "import_module",
            side_effect=fake_import,
        ):
            report = hf_ft.hf_gpt2_finetune_preflight_report(
                runtime_device_backends=[],
                require_hf_gpt2_ft=False,
                describe_runtime_devices=lambda backends, **_: {"reports": []},
            )

        self.assertTrue(report["runtime_import_preflight_passed"])
        self.assertEqual(report["required_runtime_import_presets"], "none")
        self.assertIn("datasets", report["runtime_imports_failed"])
        self.assertIn("peft", report["runtime_import_failed_install_hints"])

    def test_empty_token_probe_is_honest_without_native_runtime(self) -> None:
        probe = hf_ft.hf_gpt2_finetune_zspace_probe([])

        self.assertEqual(probe["zspace_probe_status"], "missing_tokens")
        self.assertEqual(probe["zspace_probe_observed_token_count"], 0)

    def test_default_zspace_probe_topos_uses_full_native_constructor(self) -> None:
        captured = {}

        class FakeTopos:
            def __init__(self, *args):
                captured["args"] = args

        fake_st = types.SimpleNamespace(OpenCartesianTopos=FakeTopos)
        topos = hf_ft._hf_gpt2_ft_default_topos(
            fake_st,
            curvature=-0.2,
            observed_token_count=6,
        )

        self.assertIsInstance(topos, FakeTopos)
        self.assertEqual(captured["args"], (-0.2, 1e-3, 1.0, 6, 64))

    def test_projected_tensor_values_flattens_native_tolist_rows(self) -> None:
        class FakeTensor:
            def tolist(self):
                return [[1, 2.5], [3, 4]]

        self.assertEqual(
            hf_ft._projected_tensor_values(FakeTensor()),
            [1.0, 2.5, 3.0, 4.0],
        )

    def test_zspace_probe_runs_through_projector_with_default_topos(self) -> None:
        captured = {}

        class FakeTensor:
            def __init__(self, rows, cols, data):
                self.rows = rows
                self.cols = cols
                self._data = [float(value) for value in data]

            def data(self):
                return list(self._data)

        class FakeTopos:
            def __init__(self, curvature, tolerance, saturation, max_depth, max_volume):
                captured["topos"] = (
                    curvature,
                    tolerance,
                    saturation,
                    max_depth,
                    max_volume,
                )
                self._curvature = curvature

            def curvature(self):
                return self._curvature

        class FakeEncoder:
            def __init__(self, curvature, frequency):
                captured["encoder"] = (curvature, frequency)

        class FakeProjector:
            def __init__(self, topos, encoder, *, strength):
                captured["projector_strength"] = strength

            def forward(self, tensor):
                return FakeTensor(
                    tensor.rows,
                    tensor.cols,
                    [value + 0.5 for value in tensor.data()],
                )

        fake_nn = types.ModuleType("spiraltorch.nn")
        fake_nn.ZSpaceProjector = FakeProjector

        def fake_hypergrad_topos(
            *,
            curvature,
            tolerance,
            saturation,
            max_depth,
            max_volume,
        ):
            return FakeTopos(curvature, tolerance, saturation, max_depth, max_volume)

        with (
            mock.patch.object(st, "Tensor", FakeTensor, create=True),
            mock.patch.object(st, "hypergrad_topos", fake_hypergrad_topos, create=True),
            mock.patch.object(st, "LanguageWaveEncoder", FakeEncoder, create=True),
            mock.patch.dict(sys.modules, {"spiraltorch.nn": fake_nn}),
        ):
            probe = hf_ft.hf_gpt2_finetune_zspace_probe(
                [1, 2, 3],
                dim=4,
                vocab_size=100,
                curvature=-0.2,
                frequency=0.7,
                strength=0.5,
                require=True,
            )

        self.assertEqual(probe["zspace_probe_status"], "ok")
        self.assertEqual(captured["topos"], (-0.2, 1e-3, 1.0, 3, 64))
        self.assertEqual(captured["encoder"], (-0.2, 0.7))
        self.assertEqual(captured["projector_strength"], 0.5)
        self.assertGreater(probe["zspace_probe_delta_l2"], 0.0)

    def test_corpus_file_report_records_lightweight_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            val_path = Path(tmp) / "val.txt"
            train_path.write_text("alpha\nbeta\n", encoding="utf-8")
            val_path.write_text("gamma\n", encoding="utf-8")
            report = hf_ft.hf_gpt2_finetune_corpus_file_report(
                train_files=[train_path],
                validation_files=[val_path],
                dataset_format="text",
                text_column="text",
            )

        self.assertEqual(report["dataset_source"], "local_files")
        self.assertEqual(report["file_count"], 2)
        self.assertEqual(report["train_file_count"], 1)
        self.assertEqual(report["validation_file_count"], 1)
        self.assertEqual(report["missing_files"], "none")
        self.assertTrue(report["all_files_available"])
        self.assertIsInstance(report["fingerprint"], str)
        self.assertGreater(report["total_bytes"], 0)

    def test_corpus_scan_report_streams_line_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            train_path.write_text(
                "alpha spiral\n\nbeta zspace\ngamma torch\n",
                encoding="utf-8",
            )
            report = hf_ft.hf_gpt2_finetune_corpus_scan_report(
                train_files=[train_path],
                dataset_format="text",
                text_column="text",
                sample_line_limit=2,
            )

        self.assertEqual(report["row_type"], "hf_gpt2_finetune_corpus_scan_report")
        self.assertEqual(report["dataset_source"], "local_files")
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["line_count"], 4)
        self.assertEqual(report["nonempty_line_count"], 3)
        self.assertEqual(report["empty_line_count"], 1)
        self.assertEqual(report["scan_truncated_files"], "none")
        self.assertEqual(report["scan_error_files"], "none")
        self.assertGreater(report["rough_gpt2_token_estimate"], 0)
        self.assertEqual(
            report["files"][0]["sample_texts"],
            ["alpha spiral", "beta zspace"],
        )

    def test_dataset_fit_report_flags_train_and_eval_readiness(self) -> None:
        ready = hf_ft.hf_gpt2_finetune_dataset_fit_report(
            raw_train_rows=142,
            raw_eval_rows=20,
            tokenized_train_rows=20,
            tokenized_eval_rows=3,
            block_size=64,
        )
        too_small = hf_ft.hf_gpt2_finetune_dataset_fit_report(
            raw_train_rows=4,
            raw_eval_rows=1,
            tokenized_train_rows=1,
            tokenized_eval_rows=0,
            block_size=32,
        )
        empty_train = hf_ft.hf_gpt2_finetune_dataset_fit_report(
            raw_train_rows=1,
            tokenized_train_rows=0,
            block_size=512,
        )

        self.assertEqual(ready["verdict"], "train_eval_ready")
        self.assertTrue(ready["train_ready"])
        self.assertTrue(ready["eval_ready"])
        self.assertEqual(too_small["verdict"], "train_ready_eval_unusable")
        self.assertTrue(too_small["eval_dropped_empty"])
        self.assertIn("tokenized_eval_too_small", too_small["warnings"])
        self.assertEqual(empty_train["verdict"], "not_trainable")
        self.assertFalse(empty_train["train_ready"])

    def test_generation_report_records_text_and_token_delta(self) -> None:
        report = hf_ft.hf_gpt2_finetune_generation_report(
            stage="after_train",
            prompt="SpiralTorch is",
            generated_text="SpiralTorch is learning geometry.",
            generated_continuation_text=" learning geometry.",
            input_token_count=3,
            output_token_count=8,
            max_new_tokens=16,
        )
        error_report = hf_ft.hf_gpt2_finetune_generation_report(
            stage="before_train",
            prompt="SpiralTorch is",
            max_new_tokens=16,
            error="RuntimeError: unavailable",
        )

        self.assertEqual(report["row_type"], "hf_gpt2_finetune_generation_report")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["new_token_count"], 5)
        self.assertEqual(report["generated_continuation_text"], " learning geometry.")
        self.assertIsInstance(report["generated_text_sha256"], str)
        self.assertEqual(error_report["status"], "error")
        self.assertEqual(error_report["generated_text_sha256"], None)

    def test_eval_report_records_loss_perplexity_and_status(self) -> None:
        report = hf_ft.hf_gpt2_finetune_eval_report(
            stage="after_train",
            metrics={"eval_loss": 2.0, "eval_runtime": 0.5, "_private": "drop"},
        )
        skipped = hf_ft.hf_gpt2_finetune_eval_report(
            stage="before_train",
            skipped_reason="eval_dataset_unavailable",
        )
        errored = hf_ft.hf_gpt2_finetune_eval_report(
            stage="after_train",
            error="RuntimeError: unavailable",
        )

        self.assertEqual(report["row_type"], "hf_gpt2_finetune_eval_report")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["eval_loss"], 2.0)
        self.assertAlmostEqual(report["eval_perplexity"], math.exp(2.0))
        self.assertEqual(report["metric_count"], 2)
        self.assertIn("eval_loss", report["metric_keys"])
        self.assertNotIn("_private", report["metrics"])
        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(errored["status"], "error")

    def test_run_card_summary_and_comparison_surface_ft_deltas(self) -> None:
        base_card = {
            "row_type": "hf_gpt2_finetune_run_card",
            "model_name": "gpt2",
            "dataset_name": "local-files",
            "dataset_source": "local_files",
            "dataset_format": "text",
            "block_size": 16,
            "raw_train_rows": 8,
            "raw_eval_rows": 2,
            "tokenized_train_rows": 5,
            "tokenized_eval_rows": 2,
            "load_status": "ok",
            "model_saved": True,
            "dataset_fit_report": {
                "verdict": "train_eval_ready",
                "warnings": "none",
                "train_ready": True,
                "eval_ready": True,
            },
            "eval_before_train": hf_ft.hf_gpt2_finetune_eval_report(
                stage="before_train",
                metrics={"eval_loss": 2.0},
            ),
            "eval_after_train": hf_ft.hf_gpt2_finetune_eval_report(
                stage="after_train",
                metrics={"eval_loss": 1.5},
            ),
            "generation_before_train": hf_ft.hf_gpt2_finetune_generation_report(
                stage="before_train",
                prompt="SpiralTorch is",
                generated_text="SpiralTorch is quiet.",
                generated_continuation_text=" quiet.",
                input_token_count=3,
                output_token_count=5,
            ),
            "generation_after_train": hf_ft.hf_gpt2_finetune_generation_report(
                stage="after_train",
                prompt="SpiralTorch is",
                generated_text="SpiralTorch is learning geometry.",
                generated_continuation_text=" learning geometry.",
                input_token_count=3,
                output_token_count=7,
            ),
            "trainer_metrics": {"train_loss": 1.4, "train_runtime": 3.0},
            "trainer_trace_summary": {
                "trace_event_count": 4,
                "trace_last_loss": 1.4,
                "trace_min_eval_loss": 1.5,
                "trace_duration_s": 42.0,
                "trace_log_steps_per_second_min": 0.25,
                "trace_log_steps_per_second_mean": 0.5,
                "trace_log_steps_per_second_max": 1.0,
                "trace_eval_runtime_max": 3.5,
                "trace_eval_loss_series": "0=2.0,3=1.5",
            },
            "corpus_scan_report": {
                "line_count": 10,
                "rough_gpt2_token_estimate": 128,
            },
        }
        weaker_card = dict(base_card)
        weaker_card["eval_after_train"] = hf_ft.hf_gpt2_finetune_eval_report(
            stage="after_train",
            metrics={"eval_loss": 1.9},
        )
        weaker_card["generation_after_train"] = base_card["generation_before_train"]

        with tempfile.TemporaryDirectory() as tmp:
            strong_path = Path(tmp) / "strong.json"
            weak_path = Path(tmp) / "weak.json"
            hf_ft.write_hf_gpt2_finetune_run_card(base_card, strong_path)
            hf_ft.write_hf_gpt2_finetune_run_card(weaker_card, weak_path)
            loaded = hf_ft.load_hf_gpt2_finetune_run_card(strong_path)
            summary = hf_ft.summarize_hf_gpt2_finetune_run_card(
                strong_path,
                run_label="strong",
            )
            comparison = hf_ft.compare_hf_gpt2_finetune_run_cards(
                [strong_path, weak_path],
                run_labels=["strong", "weak"],
            )
            sweep_report_path = Path(tmp) / "sweep-report.json"
            sweep_report = {
                "row_type": "hf_gpt2_finetune_sweep_report",
                "dry_run": False,
                "run_count": 2,
                "attempted_run_count": 2,
                "completed_run_count": 2,
                "failed_run_count": 0,
                "skipped_run_count": 0,
                "comparison": comparison,
                "runs": [
                    {
                        "name": "strong",
                        "run_card": str(strong_path),
                        "returncode": 0,
                    },
                    {
                        "name": "weak",
                        "run_card": str(weak_path),
                        "returncode": 0,
                    },
                ],
            }
            sweep_report_path.write_text(json.dumps(sweep_report), encoding="utf-8")
            loaded_sweep = hf_ft.load_hf_gpt2_finetune_sweep_report(
                sweep_report_path,
            )
            sweep_summary = hf_ft.summarize_hf_gpt2_finetune_sweep_report(
                sweep_report_path,
                top_n=1,
            )
            sweep_lines = hf_ft.summarize_hf_gpt2_finetune_sweep_report_lines(
                sweep_report,
                top_n=1,
            )

        self.assertEqual(loaded["model_name"], "gpt2")
        self.assertEqual(summary["row_type"], "hf_gpt2_finetune_run_card_summary")
        self.assertEqual(summary["run_label"], "strong")
        self.assertEqual(summary["dataset_fit_verdict"], "train_eval_ready")
        self.assertEqual(summary["eval_loss_delta"], -0.5)
        self.assertTrue(summary["eval_loss_improved"])
        self.assertTrue(summary["generation_continuation_changed"])
        self.assertEqual(summary["trainer_train_loss"], 1.4)
        self.assertEqual(summary["trace_event_count"], 4)
        self.assertEqual(comparison["run_count"], 2)
        self.assertEqual(comparison["best_eval_after_run_label"], "strong")
        self.assertEqual(comparison["best_eval_loss_delta_run_label"], "strong")
        self.assertEqual(comparison["eval_loss_improved_count"], 2)
        self.assertEqual(comparison["generation_changed_count"], 1)
        self.assertEqual(loaded_sweep["run_count"], 2)
        self.assertEqual(
            sweep_summary["row_type"],
            "hf_gpt2_finetune_sweep_report_summary",
        )
        self.assertEqual(sweep_summary["status"], "complete")
        self.assertEqual(sweep_summary["selected_run_label"], "strong")
        self.assertEqual(sweep_summary["selected_reason"], "best_eval_loss_delta")
        self.assertEqual(sweep_summary["top_runs"][0]["run_label"], "strong")
        self.assertEqual(sweep_summary["top_runs"][0]["trainer_runtime"], 3.0)
        self.assertEqual(
            sweep_summary["top_runs"][0]["trace_log_steps_per_second_mean"],
            0.5,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0]["trace_eval_loss_series"],
            "0=2.0,3=1.5",
        )
        self.assertIn("selected=strong", sweep_lines[1])
        self.assertIn("trainer_sps=None", sweep_lines[2])
        self.assertIn("trace_sps_mean=0.5", sweep_lines[2])
        self.assertIn("eval_series=0=2.0,3=1.5", sweep_lines[2])

        trace_card = dict(base_card)
        trace_card["eval_after_train"] = hf_ft.hf_gpt2_finetune_eval_report(
            stage="after_train",
            skipped_reason="final_step_eval_already_requested",
        )
        trace_card["trainer_trace_summary"] = {
            "trace_event_count": 4,
            "trace_last_loss": 1.4,
            "trace_last_eval_loss": 1.45,
            "trace_min_eval_loss": 1.45,
        }
        trace_summary = hf_ft.summarize_hf_gpt2_finetune_run_card(
            trace_card,
            run_label="trace",
        )
        trace_comparison = hf_ft.compare_hf_gpt2_finetune_run_cards(
            [trace_card, weaker_card],
            run_labels=["trace", "weak"],
        )

        self.assertIsNone(trace_summary["eval_after_loss"])
        self.assertEqual(trace_summary["effective_eval_after_loss"], 1.45)
        self.assertEqual(
            trace_summary["effective_eval_after_loss_source"],
            "trainer_trace_last_eval_loss",
        )
        self.assertAlmostEqual(trace_summary["eval_loss_delta"], -0.55)
        self.assertTrue(trace_summary["eval_loss_improved"])
        self.assertEqual(trace_comparison["best_eval_after_run_label"], "trace")
        self.assertEqual(trace_comparison["best_eval_after_loss"], 1.45)
        self.assertEqual(
            trace_comparison["best_eval_after_loss_source"],
            "trainer_trace_last_eval_loss",
        )

    def test_sweep_example_builds_grid_and_writes_dry_run_report(self) -> None:
        module = load_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            validation_path = Path(tmp) / "validation.txt"
            train_path.write_text("alpha spiral\nbeta zspace\n", encoding="utf-8")
            validation_path.write_text("gamma eval\n", encoding="utf-8")
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--out-dir",
                    str(out_dir),
                    "--train-file",
                    str(train_path),
                    "--validation-file",
                    str(validation_path),
                    "--corpus-scan",
                    "--corpus-scan-max-bytes-per-file",
                    "128",
                    "--block-size-values",
                    "8,16",
                    "--learning-rate-values",
                    "0.001,0.0005",
                    "--max-step-values",
                    "1",
                    "--seed-values",
                    "7,13",
                    "--generation-prompt",
                    "SpiralTorch is",
                    "--generation-do-sample",
                    "--generation-temperature",
                    "0.8",
                    "--generation-top-k",
                    "12",
                    "--eval-before-train",
                    "--eval-after-train-policy",
                    "skip-if-final-step-eval",
                    "--max-eval-blocks",
                    "64",
                    "--eval-accumulation-steps",
                    "8",
                    "--dataloader-num-workers",
                    "2",
                    "--dataloader-pin-memory",
                    "false",
                    "--runtime-device-backend",
                    "wgpu",
                    "--require-wgpu-ready",
                    "--zspace-probe",
                ]
            )
            runs = module.build_sweep_runs(args)
            report = module.run_sweep(args)
            stored_report = json.loads((out_dir / "sweep-report.json").read_text())

        self.assertEqual(len(runs), 8)
        self.assertEqual(report["run_count"], 8)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["skipped_run_count"], 8)
        self.assertEqual(report["reused_run_count"], 0)
        self.assertEqual(report["summary"]["status"], "planned")
        self.assertEqual(stored_report["row_type"], "hf_gpt2_finetune_sweep_report")
        self.assertEqual(stored_report["summary"]["run_count"], 8)
        first_command = runs[0]["command"]
        self.assertIn("--train", first_command)
        self.assertIn("--corpus-scan", first_command)
        self.assertIn("--generation-do-sample", first_command)
        self.assertIn("--eval-before-train", first_command)
        self.assertIn("--eval-after-train-policy", first_command)
        self.assertIn("skip-if-final-step-eval", first_command)
        self.assertIn("--max-eval-blocks", first_command)
        self.assertIn("64", first_command)
        self.assertIn("--eval-accumulation-steps", first_command)
        self.assertIn("8", first_command)
        self.assertIn("--dataloader-num-workers", first_command)
        self.assertIn("2", first_command)
        self.assertIn("--dataloader-pin-memory", first_command)
        self.assertIn("false", first_command)
        self.assertIn("--runtime-device-backend", first_command)
        self.assertIn("--require-wgpu-ready", first_command)
        self.assertIn("--zspace-probe", first_command)
        self.assertIn("bs8", runs[0]["name"])
        self.assertIn("seed13", runs[-1]["name"])

    def test_sweep_example_compares_completed_run_cards(self) -> None:
        module = load_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            train_path.write_text("alpha spiral\nbeta zspace\n", encoding="utf-8")
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--train-file",
                    str(train_path),
                    "--validation-fraction",
                    "0.5",
                    "--block-size-values",
                    "8",
                    "--learning-rate-values",
                    "0.001",
                    "--max-step-values",
                    "1",
                    "--seed-values",
                    "7,13",
                ]
            )

            def fake_run(command, *, check=False):
                del check
                run_card_path = Path(command[command.index("--run-card") + 1])
                seed = int(command[command.index("--seed") + 1])
                loss = 1.8 if seed == 7 else 1.3
                hf_ft.write_hf_gpt2_finetune_run_card(
                    {
                        "row_type": "hf_gpt2_finetune_run_card",
                        "model_name": "gpt2",
                        "dataset_name": "local-files",
                        "dataset_source": "local_files",
                        "dataset_fit_report": {
                            "verdict": "train_eval_ready",
                            "train_ready": True,
                            "eval_ready": True,
                        },
                        "eval_after_train": hf_ft.hf_gpt2_finetune_eval_report(
                            stage="after_train",
                            metrics={"eval_loss": loss},
                        ),
                    },
                    run_card_path,
                )
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                report = module.run_sweep(args)

        self.assertEqual(report["attempted_run_count"], 2)
        self.assertEqual(report["completed_run_count"], 2)
        self.assertEqual(report["failed_run_count"], 0)
        self.assertEqual(report["reused_run_count"], 0)
        self.assertEqual(report["comparison"]["run_count"], 2)
        self.assertIn("seed13", report["comparison"]["best_eval_after_run_label"])
        self.assertIn("seed13", report["summary"]["selected_run_label"])

    def test_sweep_example_resume_existing_reuses_successful_run_cards(self) -> None:
        module = load_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            train_path.write_text("alpha spiral\nbeta zspace\n", encoding="utf-8")
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--resume-existing",
                    "--out-dir",
                    str(out_dir),
                    "--train-file",
                    str(train_path),
                    "--validation-fraction",
                    "0.5",
                    "--block-size-values",
                    "8",
                    "--learning-rate-values",
                    "0.001",
                    "--max-step-values",
                    "1",
                    "--seed-values",
                    "7,13",
                ]
            )
            runs = module.build_sweep_runs(args)
            reusable_card = Path(runs[0]["run_card"])
            hf_ft.write_hf_gpt2_finetune_run_card(
                {
                    "row_type": "hf_gpt2_finetune_run_card",
                    "model_name": "gpt2",
                    "dataset_name": "local-files",
                    "dataset_source": "local_files",
                    "load_status": "ok",
                    "dataset_fit_report": {
                        "verdict": "train_eval_ready",
                        "train_ready": True,
                        "eval_ready": True,
                    },
                    "eval_after_train": hf_ft.hf_gpt2_finetune_eval_report(
                        stage="after_train",
                        metrics={"eval_loss": 1.7},
                    ),
                },
                reusable_card,
            )

            def fake_run(command, *, check=False):
                del check
                seed = int(command[command.index("--seed") + 1])
                self.assertEqual(seed, 13)
                run_card_path = Path(command[command.index("--run-card") + 1])
                hf_ft.write_hf_gpt2_finetune_run_card(
                    {
                        "row_type": "hf_gpt2_finetune_run_card",
                        "model_name": "gpt2",
                        "dataset_name": "local-files",
                        "dataset_source": "local_files",
                        "load_status": "ok",
                        "dataset_fit_report": {
                            "verdict": "train_eval_ready",
                            "train_ready": True,
                            "eval_ready": True,
                        },
                        "eval_after_train": hf_ft.hf_gpt2_finetune_eval_report(
                            stage="after_train",
                            metrics={"eval_loss": 1.2},
                        ),
                    },
                    run_card_path,
                )
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                report = module.run_sweep(args)

        self.assertEqual(report["attempted_run_count"], 1)
        self.assertEqual(report["reused_run_count"], 1)
        self.assertEqual(report["completed_run_count"], 2)
        self.assertEqual(report["runs"][0]["status"], "reused")
        self.assertEqual(report["runs"][1]["status"], "completed")
        self.assertEqual(report["comparison"]["run_count"], 2)
        self.assertIn("seed13", report["summary"]["selected_run_label"])
        self.assertEqual(report["summary"]["reused_run_count"], 1)

    def test_example_local_corpus_reports_attach_scan_to_cards(self) -> None:
        module = load_bridge_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            train_path.write_text("alpha\nbeta\n", encoding="utf-8")
            args = types.SimpleNamespace(
                train_file=[train_path],
                validation_file=[],
                dataset_format="text",
                text_column="text",
                validation_fraction=0.0,
                corpus_scan=True,
                corpus_scan_max_bytes_per_file=0,
                corpus_scan_sample_lines=1,
            )
            file_report = module._corpus_file_report(args)
            scan_report = module._corpus_scan_report(args)
            card = module._attach_local_corpus_reports(
                {},
                args,
                corpus_file_report=file_report,
                corpus_scan_report=scan_report,
            )

        self.assertEqual(card["dataset_source"], "local_files")
        self.assertEqual(card["corpus_file_report"]["file_count"], 1)
        self.assertEqual(card["corpus_scan_report"]["line_count"], 2)
        self.assertEqual(
            card["corpus_scan_report"]["files"][0]["sample_texts"],
            ["alpha"],
        )

    def test_example_allow_remote_temporarily_overrides_offline_env(self) -> None:
        module = load_bridge_example()
        args = types.SimpleNamespace(allow_remote=True)
        with mock.patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        ):
            report = module._hf_remote_access_report(args)
            self.assertTrue(report["offline_env_overridden"])
            with module._hf_remote_access(args):
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")
                self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "0")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_example_training_arguments_kwargs_follow_signature(self) -> None:
        module = load_bridge_example()

        class MinimalTrainingArguments:
            def __init__(
                self,
                output_dir=None,
                per_device_train_batch_size=1,
                eval_strategy="no",
            ):
                pass

        args = types.SimpleNamespace(
            output_dir="runs/gpt2",
            train=True,
            num_train_epochs=1.0,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=5,
            seed=13,
            max_steps=1,
            eval_steps=1,
        )
        raw = module._raw_training_arguments_kwargs(
            args,
            has_eval=True,
            cls=MinimalTrainingArguments,
        )
        filtered = module._filter_training_arguments_kwargs(
            MinimalTrainingArguments,
            raw,
        )

        self.assertIn("overwrite_output_dir", raw)
        self.assertNotIn("overwrite_output_dir", filtered)
        self.assertIn("eval_strategy", filtered)
        self.assertIn(
            "overwrite_output_dir",
            module._dropped_training_arguments_kwargs(MinimalTrainingArguments, raw),
        )

    def test_example_training_arguments_include_eval_runtime_knobs(self) -> None:
        module = load_bridge_example()

        class RuntimeTrainingArguments:
            def __init__(
                self,
                output_dir=None,
                dataloader_pin_memory=None,
                dataloader_num_workers=None,
                eval_accumulation_steps=None,
            ):
                pass

        args = types.SimpleNamespace(
            output_dir="runs/gpt2",
            train=True,
            num_train_epochs=1.0,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=5,
            seed=13,
            max_steps=10,
            eval_steps=5,
            eval_accumulation_steps=4,
            dataloader_num_workers=2,
            _resolved_dataloader_pin_memory=False,
        )
        raw = module._raw_training_arguments_kwargs(
            args,
            has_eval=True,
            cls=RuntimeTrainingArguments,
        )
        filtered = module._filter_training_arguments_kwargs(
            RuntimeTrainingArguments,
            raw,
        )

        self.assertFalse(filtered["dataloader_pin_memory"])
        self.assertEqual(filtered["dataloader_num_workers"], 2)
        self.assertEqual(filtered["eval_accumulation_steps"], 4)

    def test_example_dataloader_pin_memory_auto_prefers_cuda_only(self) -> None:
        module = load_bridge_example()

        class FakeCuda:
            def __init__(self, available):
                self.available = available

            def is_available(self):
                return self.available

        false_args = types.SimpleNamespace(dataloader_pin_memory="auto")
        true_args = types.SimpleNamespace(dataloader_pin_memory="true")
        explicit_false_args = types.SimpleNamespace(dataloader_pin_memory="false")

        self.assertFalse(
            module._resolve_dataloader_pin_memory(
                types.SimpleNamespace(cuda=FakeCuda(False)),
                false_args,
            )
        )
        self.assertTrue(
            module._resolve_dataloader_pin_memory(
                types.SimpleNamespace(cuda=FakeCuda(True)),
                false_args,
            )
        )
        self.assertTrue(
            module._resolve_dataloader_pin_memory(types.SimpleNamespace(), true_args)
        )
        self.assertFalse(
            module._resolve_dataloader_pin_memory(
                types.SimpleNamespace(),
                explicit_false_args,
            )
        )

    def test_example_eval_after_train_policy_skips_duplicate_final_eval(self) -> None:
        module = load_bridge_example()
        args = types.SimpleNamespace(
            train=True,
            max_steps=160,
            eval_steps=40,
            no_eval_after_train=False,
            eval_after_train_policy="skip-if-final-step-eval",
        )

        self.assertEqual(
            module._eval_after_train_skipped_reason(args, has_eval=True),
            "final_step_eval_already_requested",
        )
        args.max_steps = 161
        self.assertIsNone(module._eval_after_train_skipped_reason(args, has_eval=True))
        args.eval_after_train_policy = "never"
        self.assertEqual(
            module._eval_after_train_skipped_reason(args, has_eval=True),
            "eval_after_train_policy_never",
        )

    def test_example_limit_tokenized_eval_dataset_records_before_after(self) -> None:
        module = load_bridge_example()
        dataset = FakeDataset([{"text": str(index)} for index in range(5)])
        limited, before, after = module._limit_tokenized_eval_dataset(
            dataset,
            types.SimpleNamespace(max_eval_blocks=2),
        )

        self.assertEqual(before, 5)
        self.assertEqual(after, 2)
        self.assertEqual(len(limited), 2)

    def test_example_local_corpus_loader_uses_data_files_and_split(self) -> None:
        module = load_bridge_example()
        fake_datasets = FakeDatasets()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            train_path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
            args = types.SimpleNamespace(
                train_file=[train_path],
                validation_file=[],
                dataset_format="text",
                text_column="text",
                validation_fraction=0.25,
                seed=13,
            )
            raw_train, raw_eval, report = module._load_raw_datasets(
                fake_datasets,
                args,
            )

        self.assertEqual(fake_datasets.calls[0][0], "text")
        self.assertEqual(fake_datasets.calls[0][1], {"train": [str(train_path)]})
        self.assertEqual(len(raw_train), 3)
        self.assertEqual(len(raw_eval), 1)
        self.assertEqual(report["dataset_source"], "local_files")
        self.assertEqual(module._preflight_dataset_name(args), "local-files")
        self.assertEqual(module._preflight_dataset_config(args), "text")

    def test_example_trainer_eval_report_wraps_evaluate(self) -> None:
        module = load_bridge_example()

        class FakeTrainer:
            def __init__(self):
                self.evaluate_calls = 0

            def evaluate(self):
                self.evaluate_calls += 1
                return {"eval_loss": 1.25, "eval_samples_per_second": 2.0}

        trainer = FakeTrainer()
        report = module._trainer_eval_report(
            trainer,
            stage="before_train",
            eval_dataset_available=True,
        )
        skipped = module._trainer_eval_report(
            trainer,
            stage="after_train",
            eval_dataset_available=False,
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["eval_loss"], 1.25)
        self.assertEqual(trainer.evaluate_calls, 1)
        self.assertEqual(skipped["status"], "skipped")

    def test_example_prepare_special_tokens_compat_drops_batch_size(self) -> None:
        module = load_bridge_example()

        class FakeModel:
            def __init__(self):
                self.calls = []

            def _prepare_special_tokens(
                self,
                generation_config,
                kwargs_has_attention_mask=None,
                device=None,
            ):
                self.calls.append(
                    (generation_config, kwargs_has_attention_mask, device)
                )
                return "prepared"

        model = FakeModel()
        with self.assertRaises(TypeError):
            model._prepare_special_tokens("cfg", batch_size=1)

        with module._prepare_special_tokens_batch_size_compat(model) as installed:
            self.assertTrue(installed)
            self.assertEqual(
                model._prepare_special_tokens("cfg", device="cpu", batch_size=1),
                "prepared",
            )

        self.assertEqual(model.calls, [("cfg", None, "cpu")])
        with self.assertRaises(TypeError):
            model._prepare_special_tokens("cfg", batch_size=1)

    def test_example_generation_sample_restores_model_state(self) -> None:
        module = load_bridge_example()

        class FakeTensor:
            def __init__(self, values):
                self.values = list(values)
                self.shape = (len(self.values),)

            def to(self, device):
                self.device = device
                return self

            def __len__(self):
                return len(self.values)

        class FakeBatchTensor(FakeTensor):
            def __init__(self, values):
                super().__init__(values)
                self.shape = (1, len(self.values))

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 99

            def __call__(self, prompt, return_tensors=None):
                self.last_prompt = prompt
                self.last_return_tensors = return_tensors
                return {"input_ids": FakeBatchTensor([1, 2, 3])}

            def decode(self, tokens, *, skip_special_tokens=True):
                self.last_skip_special_tokens = skip_special_tokens
                return "SpiralTorch is learning geometry."

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            def no_grad(self):
                return FakeNoGrad()

        class FakeParam:
            device = "fake-device"

        class FakeModel:
            def __init__(self):
                self.training = True
                self.generate_kwargs = None

            def parameters(self):
                return iter([FakeParam()])

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return [FakeTensor([1, 2, 3, 4, 5])]

        args = types.SimpleNamespace(
            generation_prompt="SpiralTorch is",
            generation_max_new_tokens=8,
            generation_do_sample=False,
            generation_temperature=1.0,
            generation_top_k=0,
        )
        tokenizer = FakeTokenizer()
        model = FakeModel()
        report = module._generation_sample(
            FakeTorch(),
            tokenizer,
            model,
            args,
            stage="before_train",
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["input_token_count"], 3)
        self.assertEqual(report["output_token_count"], 5)
        self.assertEqual(report["new_token_count"], 2)
        self.assertEqual(report["generated_continuation_text"], " learning geometry.")
        self.assertTrue(model.training)
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 8)
        self.assertFalse(model.generate_kwargs["do_sample"])

    def test_trainer_trace_event_round_trip_and_summary(self) -> None:
        args = types.SimpleNamespace(
            output_dir="runs/gpt2",
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=1.0,
        )
        state = types.SimpleNamespace(
            global_step=3,
            epoch=0.25,
            max_steps=10,
            log_history=[{"loss": 1.8}],
        )
        control = types.SimpleNamespace(
            should_training_stop=False,
            should_evaluate=True,
            should_save=False,
        )

        row = hf_ft.hf_gpt2_finetune_trainer_trace_event(
            "log",
            args=args,
            state=state,
            control=control,
            logs={"loss": 1.7, "learning_rate": 4.5e-5},
            run_id="demo",
        )
        eval_row = hf_ft.hf_gpt2_finetune_trainer_trace_event(
            "evaluate",
            args=args,
            state=state,
            control=control,
            metrics={"eval_loss": 1.6},
            run_id="demo",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/trace.jsonl"
            hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, path)
            hf_ft.write_hf_gpt2_finetune_trainer_trace_event(eval_row, path)
            loaded = hf_ft.load_hf_gpt2_finetune_trainer_trace(path)
            summary = hf_ft.summarize_hf_gpt2_finetune_trainer_trace(path)

        self.assertEqual([item["event"] for item in loaded], ["log", "evaluate"])
        self.assertEqual(summary["trace_event_count"], 2)
        self.assertEqual(summary["trace_max_global_step"], 3)
        self.assertEqual(summary["trace_last_loss"], 1.7)
        self.assertEqual(summary["trace_min_eval_loss"], 1.6)

    def test_trainer_trace_summary_reports_throughput_and_eval_series(self) -> None:
        rows = [
            {
                "event": "log",
                "global_step": 10,
                "time_unix_s": 100.0,
                "metrics": {"loss": 2.0, "learning_rate": 5e-5},
            },
            {
                "event": "evaluate",
                "global_step": 10,
                "time_unix_s": 110.0,
                "metrics": {"eval_loss": 1.8, "eval_runtime": 4.0},
            },
            {
                "event": "log",
                "global_step": 30,
                "time_unix_s": 140.0,
                "metrics": {"loss": 1.5, "learning_rate": 4e-5},
            },
            {
                "event": "evaluate",
                "global_step": 30,
                "time_unix_s": 145.0,
                "metrics": {"eval_loss": 1.6, "eval_runtime": 6.0},
            },
        ]

        summary = hf_ft.summarize_hf_gpt2_finetune_trainer_trace(rows)

        self.assertEqual(summary["trace_duration_s"], 45.0)
        self.assertEqual(summary["trace_log_interval_count"], 1)
        self.assertEqual(summary["trace_log_steps_per_second_min"], 0.5)
        self.assertEqual(summary["trace_log_steps_per_second_mean"], 0.5)
        self.assertEqual(summary["trace_log_steps_per_second_max"], 0.5)
        self.assertEqual(summary["trace_eval_loss_series"], "10=1.8,30=1.6")
        self.assertEqual(summary["trace_eval_runtime_min"], 4.0)
        self.assertEqual(summary["trace_eval_runtime_mean"], 5.0)
        self.assertEqual(summary["trace_eval_runtime_max"], 6.0)
        self.assertEqual(
            summary["trace_eval_loss_points"],
            [
                {
                    "step": 10,
                    "eval_loss": 1.8,
                    "eval_runtime": 4.0,
                    "time_unix_s": 110.0,
                },
                {
                    "step": 30,
                    "eval_loss": 1.6,
                    "eval_runtime": 6.0,
                    "time_unix_s": 145.0,
                },
            ],
        )

    def test_run_card_summary_supplements_trace_telemetry_from_jsonl(self) -> None:
        card = {
            "row_type": "hf_gpt2_finetune_run_card",
            "model_name": "gpt2",
            "dataset_name": "local-files",
            "eval_before_train": hf_ft.hf_gpt2_finetune_eval_report(
                stage="before_train",
                metrics={"eval_loss": 2.0},
            ),
            "eval_after_train": hf_ft.hf_gpt2_finetune_eval_report(
                stage="after_train",
                skipped_reason="final_step_eval_already_requested",
            ),
            "trainer_trace_summary": {
                "trace_event_count": 1,
                "trace_last_loss": 2.0,
                "trace_last_eval_loss": 1.4,
            },
        }
        rows = [
            {
                "event": "log",
                "global_step": 1,
                "time_unix_s": 10.0,
                "metrics": {"loss": 2.0},
            },
            {
                "event": "log",
                "global_step": 5,
                "time_unix_s": 18.0,
                "metrics": {"loss": 1.5},
            },
            {
                "event": "evaluate",
                "global_step": 5,
                "time_unix_s": 20.0,
                "metrics": {"eval_loss": 1.4, "eval_runtime": 3.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/trace.jsonl"
            for row in rows:
                hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, path)
            card["trainer_trace_jsonl"] = path
            card_path = Path(tmp) / "run-card.json"
            hf_ft.write_hf_gpt2_finetune_run_card(card, card_path)
            summary = hf_ft.summarize_hf_gpt2_finetune_run_card(card)
            sweep_summary = hf_ft.summarize_hf_gpt2_finetune_sweep_report(
                {
                    "row_type": "hf_gpt2_finetune_sweep_report",
                    "dry_run": False,
                    "run_count": 1,
                    "completed_run_count": 1,
                    "failed_run_count": 0,
                    "comparison": {
                        "summaries": [
                            {
                                "run_label": "demo",
                                "run_card_path": str(card_path),
                                "effective_eval_after_loss": 1.4,
                            }
                        ],
                    },
                },
                top_n=1,
            )

        self.assertEqual(summary["trace_event_count"], 1)
        self.assertEqual(summary["trace_last_loss"], 2.0)
        self.assertEqual(summary["trace_duration_s"], 10.0)
        self.assertEqual(summary["trace_log_steps_per_second_mean"], 0.5)
        self.assertEqual(summary["trace_eval_runtime_max"], 3.0)
        self.assertEqual(summary["trace_eval_loss_series"], "5=1.4")
        self.assertEqual(
            sweep_summary["top_runs"][0]["trace_eval_loss_series"],
            "5=1.4",
        )

    def test_trainer_trace_callback_writes_jsonl_with_fake_transformers(self) -> None:
        fake_transformers = types.ModuleType("transformers")

        class TrainerCallback:
            pass

        fake_transformers.TrainerCallback = TrainerCallback
        args = types.SimpleNamespace(output_dir="runs/gpt2")
        state = types.SimpleNamespace(global_step=0, epoch=0.0, max_steps=1)
        control = types.SimpleNamespace()
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/trace.jsonl"
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                callback = hf_ft.hf_gpt2_finetune_trainer_trace_callback(
                    path,
                    run_id="fake-run",
                )
            callback.on_train_begin(args, state, control)
            callback.on_log(args, state, control, logs={"loss": 2.0})
            rows = hf_ft.load_hf_gpt2_finetune_trainer_trace(path)

        self.assertEqual([row["event"] for row in rows], ["train_begin", "log"])
        self.assertEqual(rows[0]["run_id"], "fake-run")
        self.assertEqual(rows[1]["metrics"], {"loss": 2.0})
        self.assertEqual(callback.event_count, 2)

    def test_top_level_exports_hf_ft_helpers(self) -> None:
        self.assertIs(
            st.hf_gpt2_finetune_preflight_report,
            hf_ft.hf_gpt2_finetune_preflight_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_rust_dependency_report,
            hf_ft.hf_gpt2_finetune_rust_dependency_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_corpus_scan_report,
            hf_ft.hf_gpt2_finetune_corpus_scan_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_dataset_fit_report,
            hf_ft.hf_gpt2_finetune_dataset_fit_report,
        )
        self.assertIn("hf_ft", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_file_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_scan_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_dataset_fit_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_eval_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_generation_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_preflight_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_trainer_trace_callback", st.__all__)
        self.assertIn("compare_hf_gpt2_finetune_run_cards", st.__all__)
        self.assertIn("load_hf_gpt2_finetune_run_card", st.__all__)
        self.assertIn("load_hf_gpt2_finetune_sweep_report", st.__all__)
        self.assertIn("summarize_hf_gpt2_finetune_run_card", st.__all__)
        self.assertIn("summarize_hf_gpt2_finetune_sweep_report", st.__all__)
        self.assertIn("summarize_hf_gpt2_finetune_sweep_report_lines", st.__all__)
        self.assertIs(
            st.hf_gpt2_finetune_eval_report,
            hf_ft.hf_gpt2_finetune_eval_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_generation_report,
            hf_ft.hf_gpt2_finetune_generation_report,
        )
        self.assertIs(
            st.compare_hf_gpt2_finetune_run_cards,
            hf_ft.compare_hf_gpt2_finetune_run_cards,
        )
        self.assertIs(
            st.load_hf_gpt2_finetune_run_card,
            hf_ft.load_hf_gpt2_finetune_run_card,
        )
        self.assertIs(
            st.load_hf_gpt2_finetune_sweep_report,
            hf_ft.load_hf_gpt2_finetune_sweep_report,
        )
        self.assertIs(
            st.summarize_hf_gpt2_finetune_run_card,
            hf_ft.summarize_hf_gpt2_finetune_run_card,
        )
        self.assertIs(
            st.summarize_hf_gpt2_finetune_sweep_report,
            hf_ft.summarize_hf_gpt2_finetune_sweep_report,
        )
        self.assertIs(
            st.summarize_hf_gpt2_finetune_sweep_report_lines,
            hf_ft.summarize_hf_gpt2_finetune_sweep_report_lines,
        )
        self.assertIs(
            st.summarize_hf_gpt2_finetune_trainer_trace,
            hf_ft.summarize_hf_gpt2_finetune_trainer_trace,
        )


if __name__ == "__main__":
    unittest.main()
