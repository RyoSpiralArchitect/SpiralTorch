from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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
SCALE_UP_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "hf_gpt2_finetune_scale_up.py"
)
TRACE_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_trace_summary.py"
)
RUN_STATUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_run_status.py"
)
STATUS_HISTORY_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_status_history_summary.py"
)
WAIT_LAUNCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_wait_launch.py"
)
WAIT_LAUNCH_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_wait_launch_summary.py"
)
MONITOR_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_monitor_snapshot.py"
)
MILESTONE_CAPTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_milestone_capture.py"
)
MILESTONE_RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_finetune_milestone_runtime.py"
)
GENERATION_CURVE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_ft_generation_curve.py"
)
CHECKPOINT_GENERATION_CONTROL_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_ft_checkpoint_generation_control.py"
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


def load_trace_summary_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_trace_summary_test",
        TRACE_SUMMARY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_run_status_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_run_status_test",
        RUN_STATUS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_status_history_summary_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_status_history_summary_test",
        STATUS_HISTORY_SUMMARY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_wait_launch_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_wait_launch_test",
        WAIT_LAUNCH_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_wait_launch_summary_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_wait_launch_summary_test",
        WAIT_LAUNCH_SUMMARY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_monitor_snapshot_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_monitor_snapshot_test",
        MONITOR_SNAPSHOT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_milestone_capture_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_milestone_capture_test",
        MILESTONE_CAPTURE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_milestone_runtime_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_milestone_runtime_test",
        MILESTONE_RUNTIME_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_generation_curve_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_ft_generation_curve_test",
        GENERATION_CURVE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_checkpoint_generation_control_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_ft_checkpoint_generation_control_test",
        CHECKPOINT_GENERATION_CONTROL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
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


def load_scale_up_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_scale_up_test",
        SCALE_UP_PATH,
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

    def test_generation_curve_report_joins_eval_trace_and_control_sweeps(self) -> None:
        def sweep_report(
            *,
            model_name: str,
            prompt: str,
            baseline_loop: float,
            controlled_loop: float,
            top_changes: int,
        ) -> dict[str, object]:
            baseline = {
                "name": "baseline-greedy",
                "kind": "baseline",
                "status": "ok",
                "config": {},
                "generation": hf_ft.hf_gpt2_finetune_generation_report(
                    stage="baseline",
                    prompt=prompt,
                    generated_text=f"{prompt} loop loop loop",
                    generated_continuation_text=" loop loop loop",
                ),
                "repetition": {
                    "loop_score": baseline_loop,
                    "unique_word_ratio": 0.25,
                },
            }
            controlled = {
                "name": "zt3-rs1p25-lr0-k64",
                "kind": "zspace_repression_softmax",
                "status": "ok",
                "config": {
                    "top_k": 64,
                    "curvature": -0.04,
                    "temperature": 1.0,
                    "entropy_target": 3.0,
                    "entropy_tolerance": 1.0e-4,
                    "entropy_gain": 0.5,
                    "min_temperature": 0.7,
                    "max_temperature": 2.4,
                    "repression_window": 16,
                    "repression_strength": 1.25,
                    "last_token_repression": 0.0,
                    "ngram_size": 3,
                    "ngram_window": 96,
                    "ngram_repression_strength": 0.75,
                    "ngram_decay": 0.9,
                    "mask_non_top_k": True,
                    "use_native_zspace": True,
                },
                "generation": hf_ft.hf_gpt2_finetune_generation_report(
                    stage="controlled",
                    prompt=prompt,
                    generated_text=f"{prompt} clean geometry {model_name}",
                    generated_continuation_text=f" clean geometry {model_name}",
                    generation_control={
                        "status": "ok",
                        "calls": 4,
                        "backend": "spiraltorch_zspace_softmax",
                        "top_token_changed_count": top_changes,
                    },
                ),
                "repetition": {
                    "loop_score": controlled_loop,
                    "unique_word_ratio": 0.75,
                },
            }
            return {
                "row_type": "hf_gpt2_zspace_generation_control_sweep",
                "status": "complete",
                "dry_run": False,
                "model_name": model_name,
                "prompt": prompt,
                "run_count": 2,
                "runs": [baseline, controlled],
            }

        card = {
            "row_type": "hf_gpt2_finetune_run_card",
            "model_name": "gpt2",
            "output_dir": "runs/latest-ft",
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "trainer_trace_summary": {
                "trace_eval_loss_series": "0=4.0,384=3.2,512=3.1",
                "trace_eval_loss_points": [
                    {"step": 0, "eval_loss": 4.0},
                    {"step": 384, "eval_loss": 3.2},
                    {"step": 512, "eval_loss": 3.1},
                ],
            },
        }
        sweeps = {
            "base": sweep_report(
                model_name="gpt2",
                prompt="SpiralTorch is",
                baseline_loop=9.0,
                controlled_loop=5.0,
                top_changes=3,
            ),
            "ckpt384": sweep_report(
                model_name="runs/latest-ft/checkpoint-384",
                prompt="SpiralTorch is",
                baseline_loop=80.0,
                controlled_loop=4.0,
                top_changes=11,
            ),
            "final": sweep_report(
                model_name="runs/latest-ft",
                prompt="SpiralTorch is",
                baseline_loop=90.0,
                controlled_loop=2.0,
                top_changes=14,
            ),
        }

        report = hf_ft.hf_gpt2_finetune_generation_curve_report(
            card,
            sweeps,
            top_n=3,
        )
        lines = hf_ft.hf_gpt2_finetune_generation_curve_lines(report, top_n=2)
        direct_lines = hf_ft.hf_gpt2_finetune_generation_curve_lines(
            card,
            sweeps,
            top_n=1,
        )

        self.assertEqual(report["row_type"], "hf_gpt2_finetune_generation_curve")
        self.assertEqual(report["curve_model_count"], 3)
        self.assertEqual(report["recommended_model_name"], "runs/latest-ft")
        self.assertEqual(report["recommended_step"], 512)
        rows = {str(row["model_name"]): row for row in report["curve_rows"]}
        self.assertEqual(rows["gpt2"]["step"], 0)
        self.assertEqual(rows["gpt2"]["eval_loss"], 4.0)
        self.assertEqual(rows["runs/latest-ft/checkpoint-384"]["step"], 384)
        self.assertEqual(
            rows["runs/latest-ft/checkpoint-384"]["eval_loss"],
            3.2,
        )
        self.assertTrue(rows["runs/latest-ft/checkpoint-384"]["is_checkpoint"])
        self.assertEqual(rows["runs/latest-ft"]["step"], 512)
        self.assertEqual(rows["runs/latest-ft"]["eval_loss"], 3.1)
        self.assertTrue(rows["runs/latest-ft"]["is_final_output"])
        self.assertIn("eval_loss_series=0=4.0,384=3.2,512=3.1", lines[0])
        self.assertIn("recommend=runs/latest-ft", lines[0])
        self.assertIn("model=runs/latest-ft/checkpoint-384", " ".join(lines))
        self.assertIn("zspace_generation_control_compare", " ".join(lines))
        self.assertIn("hf_gpt2_ft_generation_curve", direct_lines[0])

    def test_generation_curve_example_accepts_live_trace_without_run_card(self) -> None:
        module = load_generation_curve_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "runs" / "live-ft"
            run_dir.mkdir(parents=True)
            trace_jsonl = run_dir / "trainer-trace.jsonl"
            trace_rows = [
                {
                    "event": "evaluate",
                    "global_step": 0,
                    "metrics": {"eval_loss": 4.0, "eval_runtime": 1.0},
                    "time_unix_s": 1.0,
                },
                {
                    "event": "evaluate",
                    "global_step": 6144,
                    "metrics": {"eval_loss": 3.2, "eval_runtime": 1.2},
                    "time_unix_s": 2.0,
                },
            ]
            trace_jsonl.write_text(
                "\n".join(json.dumps(row) for row in trace_rows) + "\n",
                encoding="utf-8",
            )
            prompt = "A tokenless fine-tuning stack should"
            sweep = {
                "row_type": "hf_gpt2_zspace_generation_control_sweep",
                "status": "complete",
                "dry_run": False,
                "model_name": str(run_dir / "checkpoint-6144"),
                "prompt": prompt,
                "run_count": 2,
                "runs": [
                    {
                        "name": "baseline-sample",
                        "kind": "baseline",
                        "status": "ok",
                        "config": {},
                        "generation": hf_ft.hf_gpt2_finetune_generation_report(
                            stage="baseline",
                            prompt=prompt,
                            generated_text=f"{prompt} loop loop loop",
                            generated_continuation_text=" loop loop loop",
                        ),
                        "repetition": {"loop_score": 2.0},
                    },
                    {
                        "name": "zt3-rs1-lr1-k64",
                        "kind": "zspace_repression_softmax",
                        "status": "ok",
                        "config": {
                            "top_k": 64,
                            "curvature": -0.04,
                            "temperature": 1.0,
                            "entropy_target": 3.0,
                            "entropy_tolerance": 1.0e-4,
                            "entropy_gain": 0.5,
                            "min_temperature": 0.7,
                            "max_temperature": 2.4,
                            "repression_window": 16,
                            "repression_strength": 1.0,
                            "last_token_repression": 1.0,
                            "ngram_size": 0,
                            "ngram_window": 0,
                            "ngram_repression_strength": 0.0,
                            "ngram_decay": 1.0,
                            "mask_non_top_k": True,
                            "use_native_zspace": True,
                        },
                        "generation": hf_ft.hf_gpt2_finetune_generation_report(
                            stage="controlled",
                            prompt=prompt,
                            generated_text=f"{prompt} clean geometry",
                            generated_continuation_text=" clean geometry",
                            generation_control={
                                "status": "ok",
                                "calls": 4,
                                "backend": "spiraltorch_zspace_softmax",
                                "top_token_changed_count": 7,
                            },
                        ),
                        "repetition": {"loop_score": 0.0},
                    },
                ],
            }
            sweep_path = run_dir / "prompt-tokenless-ft-checkpoint-6144-sweep.json"
            sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
            out = run_dir / "generation-curve.json"
            lines_out = run_dir / "generation-curve.txt"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = module.main(
                    [
                        str(sweep_path),
                        "--label",
                        "tokenless-6144",
                        "--trainer-trace-jsonl",
                        str(trace_jsonl),
                        "--run-dir",
                        str(run_dir),
                        "--out",
                        str(out),
                        "--lines-out",
                        str(lines_out),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("hf_gpt2_ft_generation_curve_json", stdout.getvalue())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["row_type"], "hf_gpt2_finetune_generation_curve")
            self.assertEqual(payload["recommended_step"], 6144)
            self.assertEqual(payload["recommended_eval_loss"], 3.2)
            self.assertIn("eval_loss_series=0=4.0,6144=3.2", lines_out.read_text())

    def test_checkpoint_generation_control_can_plan_curve_output(self) -> None:
        module = load_checkpoint_generation_control_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "runs" / "live-ft"
            run_dir.mkdir(parents=True)
            previous_sweep = run_dir / "checkpoint-6144-sweep.json"
            previous_sweep.write_text("{}", encoding="utf-8")
            trace_jsonl = run_dir / "trainer-trace.jsonl"
            trace_jsonl.write_text("{}", encoding="utf-8")
            curve_out = run_dir / "curve.json"
            curve_lines = run_dir / "curve.txt"
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-8192",
                    "--label-prefix",
                    "finewebedu-final",
                    "--compare-with-sweep",
                    str(previous_sweep),
                    "--compare-with-label",
                    "finewebedu-6144",
                    "--curve-trainer-trace-jsonl",
                    str(trace_jsonl),
                    "--curve-out",
                    str(curve_out),
                    "--curve-lines-out",
                    str(curve_lines),
                    "--dry-run",
                ]
            )

            report = module.run_checkpoint_generation_control(args)

            self.assertEqual(report["status"], "planned")
            self.assertIn("curve", report)
            curve = report["curve"]
            self.assertEqual(curve["status"], "planned")
            command = [str(part) for part in curve["command"]]
            self.assertIn(str(previous_sweep), command)
            self.assertIn(str(trace_jsonl), command)
            self.assertIn(str(curve_out), command)
            self.assertIn(str(curve_lines), command)
            self.assertIn("finewebedu-6144", command)
            self.assertTrue(
                any(
                    part.endswith("checkpoint-8192-generation-control-sweep.json")
                    for part in command
                )
            )

    def test_inference_distortion_handoff_report_flattens_recommendation(self) -> None:
        sweep_report = {
            "row_type": "zspace_inference_distortion_sweep",
            "status": "complete",
            "prompt": "SpiralTorch is",
            "runtime": {
                "api_provider": "fake",
                "api_model": "fake-distorted-api",
                "local_model": "runs/gpt2-small-zspace-ft",
            },
            "runtime_preflight": {
                "status": "ok",
                "runtime_ready": True,
                "ready_backends": ["wgpu"],
                "missing_ready_backends": [],
            },
            "runs": [
                {
                    "name": "strong",
                    "status": "ok",
                    "probe_path": "/tmp/missing-strong-probe.json",
                    "config": {
                        "desire_pressure": 0.8,
                        "desire_stability": 0.45,
                        "psi_total": 0.7,
                        "coherence": 0.5,
                        "distortion_strength": 1.0,
                        "base_temperature": 0.7,
                        "base_top_p": 0.95,
                        "include_penalties": True,
                    },
                }
            ],
            "comparison": {
                "row_type": "zspace_inference_distortion_probe_comparison",
                "recommended_probe": "strong",
                "recommended_reason": "highest_effect_score_lowest_risk_tiebreak",
                "top_probes": [
                    {
                        "label": "strong",
                        "effect_score": 0.91,
                        "risk_score": 0.22,
                        "api_compatibility_score": 0.84,
                        "api_provider": "fake",
                        "runtime_preflight_status": "ok",
                        "runtime_ready": True,
                        "runtime_ready_backends": ["wgpu"],
                        "runtime_missing_ready_backends": [],
                        "geometry_status": "ok",
                        "geometry_backend": "native_zspace_eval_with_derivative_stable",
                        "geometry_value_l2": 7.5,
                        "geometry_derivative_l2": 12.5,
                        "api_request_dropped_key_count": 2,
                        "api_request_dropped_keys": [
                            "frequency_penalty",
                            "presence_penalty",
                        ],
                        "api_request_retry_dropped_key_count": 1,
                        "api_request_retry_dropped_keys": ["temperature"],
                        "api_request_sent_keys": ["temperature", "top_p"],
                    }
                ],
            },
        }

        handoff = hf_ft.hf_gpt2_finetune_inference_distortion_handoff_report(
            sweep_report,
        )
        handoff_lines = hf_ft.hf_gpt2_finetune_inference_distortion_handoff_lines(
            handoff,
        )

        self.assertEqual(handoff["status"], "ok")
        self.assertEqual(handoff["prompt"], "SpiralTorch is")
        self.assertEqual(handoff["recommended_probe"], "strong")
        self.assertEqual(handoff["api_provider"], "fake")
        self.assertEqual(handoff["api_model"], "fake-distorted-api")
        self.assertEqual(handoff["runtime_preflight_status"], "ok")
        self.assertTrue(handoff["runtime_ready"])
        self.assertEqual(handoff["runtime_ready_backends"], ["wgpu"])
        self.assertEqual(handoff["geometry_status"], "ok")
        self.assertEqual(handoff["geometry_derivative_l2"], 12.5)
        self.assertEqual(
            handoff["recommended_runtime_adapter_kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(
            handoff["recommended_runtime_adapter_context_origin"],
            "zspace:inference_distortion",
        )
        self.assertEqual(handoff["recommended_runtime_adapter_context_weight"], 1.0)
        self.assertEqual(
            handoff["recommended_runtime_adapter"]["kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertIn("temperature", handoff["recommended_runtime_adapter_request"])
        self.assertEqual(handoff["recommended_api_compatibility_score"], 0.84)
        self.assertEqual(handoff["desire_pressure"], 0.8)
        self.assertEqual(handoff["psi_total"], 0.7)
        self.assertTrue(handoff["include_penalties"])
        self.assertEqual(handoff["api_request_dropped_key_count"], 2)
        self.assertEqual(
            handoff["api_request_dropped_keys"],
            ["frequency_penalty", "presence_penalty"],
        )
        self.assertEqual(handoff["api_request_retry_dropped_key_count"], 1)
        self.assertEqual(handoff["api_request_retry_dropped_keys"], ["temperature"])
        self.assertEqual(handoff["api_request_sent_keys"], ["temperature", "top_p"])
        self.assertIn("--desire-pressure", handoff["recommended_probe_cli_args"])
        self.assertIn("api_compat=0.84", handoff_lines[0])
        self.assertIn("runtime=ok", handoff_lines[0])
        self.assertIn("geom=12.5", handoff_lines[0])
        self.assertIn(
            "adapter=spiraltorch.zspace_inference_distortion_adapter",
            handoff_lines[0],
        )
        self.assertIn("retry_dropped=1", handoff_lines[0])

    def test_inference_distortion_handoff_report_accepts_probe_artifact(self) -> None:
        probe_report = {
            "row_type": "zspace_inference_distortion_probe",
            "prompt": "SpiralTorch direct probe",
            "runtime": {
                "api_provider": "fake",
                "api_model": "fake-distorted-api",
                "local_model": "runs/gpt2-small-zspace-ft",
            },
            "runtime_preflight": {
                "status": "ok",
                "runtime_ready": True,
                "ready_backends": ["wgpu"],
                "missing_ready_backends": [],
            },
            "config": {
                "desire_pressure": 0.83,
                "desire_stability": 0.41,
                "psi_total": 0.73,
                "coherence": 0.52,
                "distortion_strength": 1.1,
                "base_temperature": 0.7,
                "base_top_p": 0.95,
                "include_penalties": True,
            },
            "adapter": {
                "request": {"temperature": 1.08, "top_p": 0.76},
                "logits_processor_kwargs": {
                    "repression_strength": 1.7,
                    "ngram_repression_strength": 0.9,
                },
                "activation_hook": {
                    "name_contains": ["attn", "mlp"],
                    "intervention_scale": 0.91,
                },
            },
            "geometry_probe": {
                "status": "ok",
                "backend": "native_zspace_eval_with_derivative_stable",
                "value_l2": 8.5,
                "derivative_l2": 13.5,
            },
            "local_hf": {
                "status": "ok",
                "changed": True,
                "generation_control": {
                    "status": "ok",
                    "backend": "spiraltorch_zspace_softmax",
                    "top_token_changed_count": 6,
                },
                "activation_report": {"status": "ok", "event_count": 12},
            },
            "api": {
                "provider": "fake",
                "model": "fake-distorted-api",
                "text": "Fake direct probe response",
                "request_filter": {
                    "dropped_key_count": 1,
                    "dropped_keys": ["presence_penalty"],
                    "sent_keys": ["temperature", "top_p"],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            probe_path = Path(tmp) / "direct-probe.json"
            probe_path.write_text(json.dumps(probe_report), encoding="utf-8")
            handoff = hf_ft.hf_gpt2_finetune_inference_distortion_handoff_report(
                probe_path,
            )
            handoff_lines = (
                hf_ft.hf_gpt2_finetune_inference_distortion_handoff_lines(
                    probe_path,
                    replay_arg_limit=24,
                )
            )
            preflight_lines = hf_ft.hf_gpt2_finetune_summary_lines(
                {
                    "hf_model_name": "gpt2",
                    "hf_dataset_name": "local-files",
                    "hf_dataset_config": "text",
                    "hf_train_split": "train",
                    "hf_text_column": "text",
                    "hf_gpt2_ft_rust_surfaces": "st-tensor,st-nn",
                    "hf_gpt2_ft_python_packages": "transformers,torch",
                    "inference_distortion_handoff": handoff,
                    "trainer_telemetry_requested": False,
                    "trainer_telemetry_enabled": True,
                    "trainer_telemetry_auto_reason": "inference_distortion_handoff",
                    "trainer_telemetry_prefix": "hf_ft",
                }
            )

        self.assertEqual(handoff["status"], "ok")
        self.assertEqual(handoff["source_kind"], "probe")
        self.assertEqual(handoff["runtime_preflight_status"], "ok")
        self.assertTrue(handoff["runtime_ready"])
        self.assertEqual(handoff["geometry_derivative_l2"], 13.5)
        self.assertEqual(
            handoff["recommended_runtime_adapter_kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(
            handoff["recommended_runtime_adapter"]["context_partial"]["origin"],
            "zspace:inference_distortion",
        )
        self.assertEqual(
            handoff["source_row_type"],
            "zspace_inference_distortion_probe",
        )
        self.assertEqual(handoff["recommended_probe"], "direct-probe")
        self.assertIn("status=ok", handoff_lines[0])
        self.assertIn("probe=direct-probe", handoff_lines[0])
        self.assertIn("hf_gpt2_ft_inference_handoff_replay", handoff_lines[1])
        self.assertIn("--generation-repression-strength", handoff_lines[1])
        self.assertIn("hf_gpt2_ft_inference_handoff_generation", handoff_lines[2])
        self.assertIn("--generation-from-inference-distortion", handoff_lines[2])
        self.assertTrue(
            any("probe=direct-probe" in line for line in preflight_lines)
        )
        self.assertTrue(
            any(
                "hf_gpt2_ft_trainer_telemetry" in line
                and "auto=inference_distortion_handoff" in line
                for line in preflight_lines
            )
        )
        self.assertEqual(handoff["prompt"], "SpiralTorch direct probe")
        self.assertEqual(handoff["desire_pressure"], 0.83)
        self.assertEqual(handoff["psi_total"], 0.73)
        self.assertEqual(
            handoff["recommended_request"],
            {"temperature": 1.08, "top_p": 0.76},
        )
        self.assertEqual(
            handoff["recommended_processor_kwargs"]["repression_strength"],
            1.7,
        )
        self.assertIn(
            "--generation-zspace-softmax",
            handoff["recommended_bridge_cli_args"],
        )
        self.assertIn(
            "--generation-repression-strength",
            handoff["recommended_bridge_cli_args"],
        )
        self.assertIn("1.7", handoff["recommended_bridge_cli_args"])
        self.assertIn(
            "--generation-ngram-repression-strength",
            handoff["recommended_bridge_cli_args"],
        )
        self.assertIn(
            "--inference-distortion-probe",
            handoff["recommended_source_cli_args"],
        )
        self.assertIn(str(probe_path), handoff["recommended_source_cli_args"])
        self.assertIn(
            "--generation-from-inference-distortion",
            handoff["recommended_generation_handoff_cli_args"],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            handoff["recommended_generation_handoff_cli_display"],
        )
        self.assertIn(
            "--inference-distortion-probe",
            handoff["recommended_explicit_generation_bridge_cli_args"],
        )
        self.assertIn(
            "--generation-repression-strength",
            handoff["recommended_explicit_generation_bridge_cli_args"],
        )
        self.assertIn(
            "--generation-repression-strength",
            handoff["recommended_explicit_generation_bridge_cli_display"],
        )
        self.assertEqual(
            handoff["recommended_activation_hook"]["name_contains"],
            ["attn", "mlp"],
        )
        self.assertEqual(handoff["api_request_dropped_keys"], ["presence_penalty"])

    def test_inference_distortion_runtime_plan_accepts_nested_artifacts(self) -> None:
        handoff = {
            "row_type": "hf_gpt2_finetune_inference_distortion_handoff",
            "status": "ok",
            "source_kind": "probe",
            "recommended_probe": "distort-002",
            "recommended_runtime_adapter": {
                "kind": "spiraltorch.zspace_inference_distortion_adapter",
                "request": {"temperature": 1.05, "top_p": 0.82},
                "context_partial": {
                    "origin": "zspace:inference_distortion",
                    "weight": 1.0,
                    "metrics": {"speed": 0.7, "memory": 0.4, "stability": 0.5},
                    "telemetry": {"zspace.distortion.energy": 0.4},
                },
            },
            "recommended_runtime_adapter_kind": (
                "spiraltorch.zspace_inference_distortion_adapter"
            ),
            "recommended_runtime_adapter_request": {
                "temperature": 1.05,
                "top_p": 0.82,
            },
            "recommended_request": {"temperature": 0.9, "top_p": 0.95},
        }

        plan = hf_ft.hf_gpt2_finetune_inference_distortion_runtime_plan(
            handoff,
            request={"model": "fake-api", "temperature": 0.1},
        )
        adapter = hf_ft.hf_gpt2_finetune_inference_distortion_runtime_adapter(
            handoff,
        )
        request_kwargs = hf_ft.hf_gpt2_finetune_inference_distortion_request_kwargs(
            handoff,
            request={"model": "fake-api"},
        )

        self.assertEqual(
            plan["kind"],
            "spiraltorch.hf_gpt2_finetune_inference_distortion_runtime_plan",
        )
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["request"]["model"], "fake-api")
        self.assertEqual(plan["request"]["temperature"], 1.05)
        self.assertEqual(plan["request_overrides"]["top_p"], 0.82)
        self.assertEqual(
            plan["runtime_adapter"]["context_partial"]["origin"],
            "zspace:inference_distortion",
        )
        self.assertEqual(
            adapter["kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(request_kwargs["temperature"], 1.05)
        calls: list[tuple[str, dict[str, object]]] = []
        runtime = st.ApiLLMZSpaceRuntime([0.1, -0.2, 0.3, -0.4])

        def fake_api(prompt: str, **request_kwargs: object) -> dict[str, object]:
            calls.append((prompt, dict(request_kwargs)))
            return {
                "model": "fake-api",
                "output_text": "FT handoff runtime plan reached API inference.",
                "usage": {"total_tokens": 9},
            }

        trace = runtime.call(
            fake_api,
            "Route the FT handoff.",
            runtime_adapter=plan,
            context_prompt=True,
        )

        self.assertEqual(calls[0][1]["temperature"], 1.05)
        self.assertIn("origin=zspace:inference_distortion", calls[0][0])
        self.assertEqual(trace.telemetry["api_llm.total_tokens"], 9.0)

        with tempfile.TemporaryDirectory() as tmp:
            run_card_path = Path(tmp) / "run-card.json"
            sweep_path = Path(tmp) / "sweep-report.json"
            run_card_path.write_text(
                json.dumps(
                    {
                        "row_type": "hf_gpt2_finetune_run_card",
                        "inference_distortion_handoff": handoff,
                    }
                ),
                encoding="utf-8",
            )
            sweep_path.write_text(
                json.dumps(
                    {
                        "row_type": "hf_gpt2_finetune_sweep_report",
                        "inference_distortion_handoff": handoff,
                    }
                ),
                encoding="utf-8",
            )

            from_run_card = hf_ft.hf_gpt2_finetune_inference_distortion_runtime_plan(
                run_card_path,
            )
            from_sweep = hf_ft.hf_gpt2_finetune_inference_distortion_request_kwargs(
                sweep_path,
            )

        self.assertEqual(from_run_card["recommended_probe"], "distort-002")
        self.assertEqual(
            from_run_card["runtime_adapter_kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(from_sweep["temperature"], 1.05)

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
        inference_handoff = {
            "row_type": "hf_gpt2_finetune_inference_distortion_handoff",
            "status": "ok",
            "source_path": "runs/zspace-inference-distortion-sweep/sweep-report.json",
            "recommended_probe": "distort-002",
            "recommendation_reason": "highest_effect_score_lowest_risk_tiebreak",
            "recommended_effect_score": 0.88,
            "recommended_risk_score": 0.21,
            "recommended_api_compatibility_score": 0.84,
            "desire_pressure": 0.8,
            "desire_stability": 0.45,
            "psi_total": 0.7,
            "coherence": 0.5,
            "api_provider": "fake",
            "api_model": "fake-distorted-api",
            "recommended_runtime_adapter": {
                "kind": "spiraltorch.zspace_inference_distortion_adapter",
                "request": {"temperature": 1.05, "top_p": 0.82},
                "context_partial": {
                    "origin": "zspace:inference_distortion",
                    "weight": 1.0,
                    "metrics": {"speed": 0.7, "memory": 0.4, "stability": 0.5},
                    "telemetry": {"zspace.distortion.energy": 0.4},
                },
            },
            "recommended_runtime_adapter_kind": (
                "spiraltorch.zspace_inference_distortion_adapter"
            ),
            "recommended_runtime_adapter_request": {
                "temperature": 1.05,
                "top_p": 0.82,
            },
            "recommended_runtime_adapter_context_origin": (
                "zspace:inference_distortion"
            ),
            "recommended_runtime_adapter_context_weight": 1.0,
            "runtime_preflight_status": "ok",
            "runtime_ready": True,
            "runtime_ready_backends": ["wgpu"],
            "runtime_missing_ready_backends": [],
            "geometry_status": "ok",
            "geometry_backend": "native_zspace_eval_with_derivative_stable",
            "geometry_value_l2": 7.5,
            "geometry_derivative_l2": 12.5,
            "api_request_dropped_key_count": 2,
            "api_request_dropped_keys": ["frequency_penalty", "presence_penalty"],
            "api_request_retry_dropped_key_count": 1,
            "api_request_retry_dropped_keys": ["temperature"],
            "api_request_sent_keys": ["temperature", "top_p"],
            "recommended_bridge_cli_args": [
                "--generation-zspace-softmax",
                "--generation-repression-strength",
                "1.7",
            ],
            "recommended_generation_handoff_cli_args": [
                "--inference-distortion-sweep-report",
                "runs/zspace-inference-distortion-sweep/sweep-report.json",
                "--generation-from-inference-distortion",
            ],
            "recommended_explicit_generation_bridge_cli_args": [
                "--inference-distortion-sweep-report",
                "runs/zspace-inference-distortion-sweep/sweep-report.json",
                "--generation-zspace-softmax",
                "--generation-repression-strength",
                "1.7",
            ],
        }
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
                generation_control={
                    "status": "ok",
                    "calls": 7,
                    "top_token_changed_count": 2,
                    "temperature_min": 0.7,
                    "temperature_max": 1.2,
                    "entropy_min": 2.8,
                    "entropy_max": 4.1,
                    "backend": "spiraltorch_zspace_softmax",
                },
            ),
            "trainer_metrics": {"train_loss": 1.4, "train_runtime": 3.0},
            "trainer_telemetry_requested": False,
            "trainer_telemetry_enabled": True,
            "trainer_telemetry_auto_reason": "inference_distortion_handoff",
            "inference_distortion_handoff": inference_handoff,
            "generation_from_inference_distortion": True,
            "generation_from_inference_distortion_applied": {
                "status": "ok",
                "source_kind": "probe",
                "recommended_probe": "distort-002",
                "applied_arg_count": 6,
                "processor_kwargs": {
                    "top_k": 64,
                    "temperature": 1.05,
                    "entropy_target": 3.4,
                    "repression_strength": 1.7,
                    "ngram_repression_strength": 0.9,
                },
            },
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
                "trace_inference_distortion_telemetry_count": 4,
                "trace_last_inference_distortion_risk_score": 0.21,
                "trace_last_inference_distortion_api_compatibility_score": 0.84,
                "trace_last_inference_distortion_api_request_dropped_key_count": 2,
                "trace_last_inference_distortion_api_request_retry_dropped_key_count": 1,
                "trace_last_inference_distortion_logits_repression_strength": 1.7,
                "trace_last_inference_distortion_logits_ngram_repression_strength": 0.9,
                "trace_last_inference_distortion_include_penalties": 1.0,
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
                "inference_distortion_handoff": inference_handoff,
                "comparison": comparison,
                "runs": [
                    {
                        "name": "strong",
                        "run_dir": str(Path(tmp) / "strong-run"),
                        "run_card": str(strong_path),
                        "trainer_trace_jsonl": str(Path(tmp) / "strong-trace.jsonl"),
                        "command": [
                            "python",
                            "bridge",
                            "--run-card",
                            str(strong_path),
                        ],
                        "command_display": f"python bridge --run-card {strong_path}",
                        "returncode": 0,
                        "status": "completed",
                    },
                    {
                        "name": "weak",
                        "run_card": str(weak_path),
                        "returncode": 0,
                        "status": "completed",
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
        self.assertEqual(summary["generation_after_control_status"], "ok")
        self.assertEqual(summary["generation_after_control_calls"], 7)
        self.assertEqual(
            summary["generation_after_control_top_token_changed_count"],
            2,
        )
        self.assertEqual(
            summary["generation_after_control_backend"],
            "spiraltorch_zspace_softmax",
        )
        self.assertTrue(summary["generation_from_inference_distortion"])
        self.assertEqual(
            summary["generation_from_inference_distortion_status"],
            "ok",
        )
        self.assertEqual(
            summary["generation_from_inference_distortion_probe"],
            "distort-002",
        )
        self.assertEqual(
            summary["generation_from_inference_distortion_applied_arg_count"],
            6,
        )
        self.assertEqual(
            summary["generation_from_inference_distortion_entropy_target"],
            3.4,
        )
        self.assertEqual(
            summary["generation_from_inference_distortion_repression_strength"],
            1.7,
        )
        self.assertEqual(
            summary[
                "generation_from_inference_distortion_ngram_repression_strength"
            ],
            0.9,
        )
        bridge_cli_args = summary[
            "generation_from_inference_distortion_bridge_cli_args"
        ]
        self.assertIn("--generation-zspace-softmax", bridge_cli_args)
        self.assertIn("--generation-zspace-entropy-target", bridge_cli_args)
        self.assertIn("3.4", bridge_cli_args)
        self.assertIn("--generation-repression-strength", bridge_cli_args)
        self.assertIn("1.7", bridge_cli_args)
        self.assertIn("--generation-ngram-repression-strength", bridge_cli_args)
        self.assertIn("0.9", bridge_cli_args)
        self.assertEqual(summary["trainer_train_loss"], 1.4)
        self.assertFalse(summary["trainer_telemetry_requested"])
        self.assertTrue(summary["trainer_telemetry_enabled"])
        self.assertEqual(
            summary["trainer_telemetry_auto_reason"],
            "inference_distortion_handoff",
        )
        self.assertAlmostEqual(summary["distortion_pressure_index"], 0.3366666667)
        self.assertAlmostEqual(
            summary["distortion_adjusted_eval_loss"],
            1.5336666667,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_risk_score"],
            0.21,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_api_request_retry_dropped_key_count"],
            1,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_logits_repression_strength"],
            1.7,
        )
        self.assertEqual(summary["inference_distortion_handoff_status"], "ok")
        self.assertEqual(
            summary["inference_distortion_recommended_probe"],
            "distort-002",
        )
        self.assertEqual(summary["inference_distortion_effect_score"], 0.88)
        self.assertEqual(summary["inference_distortion_api_compatibility_score"], 0.84)
        self.assertEqual(summary["inference_distortion_desire_pressure"], 0.8)
        self.assertEqual(summary["inference_distortion_psi_total"], 0.7)
        self.assertEqual(summary["inference_distortion_api_provider"], "fake")
        self.assertEqual(
            summary["inference_distortion_runtime_adapter_kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(
            summary["inference_distortion_runtime_adapter_context_origin"],
            "zspace:inference_distortion",
        )
        self.assertEqual(
            summary["inference_distortion_runtime_adapter_request_temperature"],
            1.05,
        )
        self.assertEqual(summary["inference_distortion_runtime_preflight_status"], "ok")
        self.assertTrue(summary["inference_distortion_runtime_ready"])
        self.assertEqual(summary["inference_distortion_runtime_ready_backends"], "wgpu")
        self.assertEqual(summary["inference_distortion_geometry_status"], "ok")
        self.assertEqual(summary["inference_distortion_geometry_derivative_l2"], 12.5)
        self.assertEqual(
            summary["inference_distortion_api_request_dropped_key_count"],
            2,
        )
        self.assertEqual(
            summary["inference_distortion_api_request_dropped_keys"],
            "frequency_penalty,presence_penalty",
        )
        self.assertEqual(
            summary["inference_distortion_api_request_retry_dropped_key_count"],
            1,
        )
        self.assertEqual(
            summary["inference_distortion_api_request_retry_dropped_keys"],
            "temperature",
        )
        self.assertIn(
            "--generation-repression-strength",
            summary["inference_distortion_bridge_cli_args"],
        )
        self.assertEqual(summary["inference_distortion_handoff_line_count"], 3)
        self.assertTrue(
            any(
                "probe=distort-002" in line
                for line in summary["inference_distortion_handoff_lines"]
            )
        )
        self.assertEqual(summary["inference_distortion_replay_arg_count"], 3)
        self.assertIn(
            "--generation-repression-strength",
            summary["inference_distortion_replay_cli_preview"],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            summary["inference_distortion_generation_handoff_cli_preview"],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            summary["inference_distortion_generation_handoff_cli_display"],
        )
        self.assertIn(
            "--generation-repression-strength",
            summary["inference_distortion_explicit_generation_bridge_cli_preview"],
        )
        self.assertIn(
            "--generation-repression-strength",
            summary["inference_distortion_explicit_generation_bridge_cli_display"],
        )
        self.assertEqual(summary["trace_event_count"], 4)
        self.assertEqual(comparison["run_count"], 2)
        self.assertEqual(comparison["best_eval_after_run_label"], "strong")
        self.assertEqual(comparison["best_eval_loss_delta_run_label"], "strong")
        self.assertEqual(comparison["best_distortion_adjusted_run_label"], "strong")
        self.assertEqual(comparison["eval_loss_improved_count"], 2)
        self.assertEqual(comparison["generation_changed_count"], 1)
        self.assertEqual(comparison["generation_from_inference_distortion_count"], 2)
        self.assertEqual(loaded_sweep["run_count"], 2)
        self.assertEqual(
            sweep_summary["row_type"],
            "hf_gpt2_finetune_sweep_report_summary",
        )
        self.assertEqual(sweep_summary["status"], "complete")
        self.assertEqual(sweep_summary["selected_run_label"], "strong")
        self.assertEqual(sweep_summary["selected_reason"], "best_eval_loss_delta")
        self.assertEqual(sweep_summary["scale_up_candidate_label"], "strong")
        self.assertEqual(
            sweep_summary["scale_up_candidate_reason"],
            "lowest_distortion_adjusted_eval_loss",
        )
        self.assertAlmostEqual(
            sweep_summary["scale_up_candidate_distortion_pressure_index"],
            0.3366666667,
        )
        self.assertEqual(sweep_summary["scale_up_candidate_status"], "completed")
        self.assertEqual(
            sweep_summary["scale_up_candidate_run_card"],
            str(strong_path),
        )
        self.assertEqual(
            sweep_summary["scale_up_candidate_trainer_trace_jsonl"],
            str(Path(tmp) / "strong-trace.jsonl"),
        )
        self.assertEqual(
            sweep_summary["scale_up_candidate_command"],
            ["python", "bridge", "--run-card", str(strong_path)],
        )
        self.assertEqual(sweep_summary["selected_run_status"], "completed")
        self.assertEqual(sweep_summary["selected_run_card"], str(strong_path))
        self.assertIn(
            "--generation-repression-strength",
            sweep_summary["top_runs"][0][
                "generation_from_inference_distortion_bridge_cli_args"
            ],
        )
        self.assertEqual(
            sweep_summary["selected_trainer_trace_jsonl"],
            str(Path(tmp) / "strong-trace.jsonl"),
        )
        self.assertEqual(
            sweep_summary["selected_command"],
            ["python", "bridge", "--run-card", str(strong_path)],
        )
        self.assertEqual(
            sweep_summary["inference_distortion_api_request_dropped_key_count"],
            2,
        )
        self.assertEqual(
            sweep_summary["inference_distortion_runtime_preflight_status"],
            "ok",
        )
        self.assertEqual(
            sweep_summary["inference_distortion_runtime_adapter_kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(
            sweep_summary["inference_distortion_runtime_adapter_request_temperature"],
            1.05,
        )
        self.assertEqual(
            sweep_summary["inference_distortion_geometry_derivative_l2"],
            12.5,
        )
        self.assertIn(
            "--generation-repression-strength",
            sweep_summary["inference_distortion_bridge_cli_args"],
        )
        self.assertEqual(sweep_summary["inference_distortion_handoff_line_count"], 3)
        self.assertIn(
            "--generation-repression-strength",
            sweep_summary["inference_distortion_replay_cli_preview"],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            sweep_summary["inference_distortion_generation_handoff_cli_preview"],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            sweep_summary["inference_distortion_generation_handoff_cli_display"],
        )
        self.assertEqual(sweep_summary["top_runs"][0]["run_label"], "strong")
        self.assertIn(
            "--generation-repression-strength",
            sweep_summary["top_runs"][0]["inference_distortion_bridge_cli_args"],
        )
        self.assertEqual(sweep_summary["top_runs"][0]["trainer_runtime"], 3.0)
        self.assertFalse(
            sweep_summary["top_runs"][0]["trainer_telemetry_requested"]
        )
        self.assertTrue(sweep_summary["top_runs"][0]["trainer_telemetry_enabled"])
        self.assertEqual(
            sweep_summary["top_runs"][0]["trainer_telemetry_auto_reason"],
            "inference_distortion_handoff",
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "generation_after_control_top_token_changed_count"
            ],
            2,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0]["generation_after_control_backend"],
            "spiraltorch_zspace_softmax",
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "generation_from_inference_distortion_status"
            ],
            "ok",
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "generation_from_inference_distortion_repression_strength"
            ],
            1.7,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0]["trace_log_steps_per_second_mean"],
            0.5,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0]["trace_eval_loss_series"],
            "0=2.0,3=1.5",
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "trace_last_inference_distortion_risk_score"
            ],
            0.21,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "trace_last_inference_distortion_api_request_retry_dropped_key_count"
            ],
            1,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "trace_last_inference_distortion_logits_repression_strength"
            ],
            1.7,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0]["inference_distortion_recommended_probe"],
            "distort-002",
        )
        self.assertEqual(
            sweep_summary["top_runs"][0]["inference_distortion_effect_score"],
            0.88,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "inference_distortion_api_compatibility_score"
            ],
            0.84,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "inference_distortion_api_request_dropped_key_count"
            ],
            2,
        )
        self.assertEqual(
            sweep_summary["top_runs"][0][
                "inference_distortion_api_request_retry_dropped_key_count"
            ],
            1,
        )
        self.assertIn("selected=strong", sweep_lines[1])
        self.assertTrue(
            any("hf_gpt2_ft_sweep_selected" in line for line in sweep_lines)
        )
        self.assertTrue(
            any("api_dropped=2" in line for line in sweep_lines)
        )
        top_line = next(line for line in sweep_lines if "hf_gpt2_ft_sweep_top" in line)
        self.assertIn("trainer_sps=None", top_line)
        self.assertIn("trace_sps_mean=0.5", top_line)
        self.assertIn("adjusted=1.533666", top_line)
        self.assertIn("pressure=0.336666", top_line)
        self.assertIn("telemetry=True", top_line)
        self.assertIn("telemetry_auto=inference_distortion_handoff", top_line)
        self.assertIn("infer_trace_risk=0.21", top_line)
        self.assertIn("infer_trace_retry_drop=1", top_line)
        self.assertIn("infer_trace_repress=1.7", top_line)
        self.assertIn("eval_series=0=2.0,3=1.5", top_line)
        self.assertIn("infer_probe=distort-002", top_line)
        self.assertIn("gen_infer=ok", top_line)
        self.assertIn("gen_repress=1.7", top_line)
        self.assertIn("gen_entropy=3.4", top_line)
        self.assertIn("zcontrol_changed=2", top_line)
        self.assertIn("zcontrol_backend=spiraltorch_zspace_softmax", top_line)
        self.assertTrue(
            any("hf_gpt2_ft_sweep_scale_up" in line for line in sweep_lines)
        )

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

    def test_distortion_adjusted_scale_up_prefers_safer_nearby_run(self) -> None:
        def run_card(eval_loss: float, *, trace: dict[str, float]) -> dict[str, object]:
            return {
                "row_type": "hf_gpt2_finetune_run_card",
                "model_name": "gpt2",
                "dataset_name": "local-files",
                "dataset_fit_report": {
                    "verdict": "train_eval_ready",
                    "train_ready": True,
                    "eval_ready": True,
                },
                "eval_after_train": hf_ft.hf_gpt2_finetune_eval_report(
                    stage="after_train",
                    metrics={"eval_loss": eval_loss},
                ),
                "trainer_trace_summary": {
                    "trace_event_count": 2,
                    **trace,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            risky_path = Path(tmp) / "risky.json"
            safe_path = Path(tmp) / "safe.json"
            risky_trace = Path(tmp) / "risky-trace.jsonl"
            safe_trace = Path(tmp) / "safe-trace.jsonl"
            risky_output_dir = Path(tmp) / "risky-out"
            safe_output_dir = Path(tmp) / "safe-out"
            risky_command = [
                "python",
                "bridge",
                "--output-dir",
                str(risky_output_dir),
                "--run-card",
                str(risky_path),
                "--trainer-trace-jsonl",
                str(risky_trace),
                "--max-steps",
                "10",
                "--max-train-samples",
                "100",
                "--max-eval-samples",
                "20",
            ]
            safe_command = [
                "python",
                "bridge",
                "--output-dir",
                str(safe_output_dir),
                "--run-card",
                str(safe_path),
                "--trainer-trace-jsonl",
                str(safe_trace),
                "--max-steps",
                "10",
                "--max-train-samples",
                "100",
                "--max-eval-samples",
                "20",
            ]
            safe_checkpoint = Path(tmp) / "safe-checkpoint-512"
            safe_checkpoint.mkdir()

            risky = run_card(
                1.5,
                trace={
                    "trace_last_inference_distortion_risk_score": 1.0,
                    "trace_last_inference_distortion_api_compatibility_score": 0.0,
                    "trace_last_inference_distortion_api_request_dropped_key_count": 4,
                    "trace_last_inference_distortion_api_request_retry_dropped_key_count": 2,
                    "trace_last_inference_distortion_logits_repression_strength": 4.0,
                    "trace_last_inference_distortion_logits_ngram_repression_strength": 4.0,
                },
            )
            safe = run_card(
                1.54,
                trace={
                    "trace_last_inference_distortion_risk_score": 0.0,
                    "trace_last_inference_distortion_api_compatibility_score": 1.0,
                    "trace_last_inference_distortion_api_request_dropped_key_count": 0,
                    "trace_last_inference_distortion_api_request_retry_dropped_key_count": 0,
                    "trace_last_inference_distortion_logits_repression_strength": 0.0,
                    "trace_last_inference_distortion_logits_ngram_repression_strength": 0.0,
                },
            )
            hf_ft.write_hf_gpt2_finetune_run_card(risky, risky_path)
            hf_ft.write_hf_gpt2_finetune_run_card(safe, safe_path)

            comparison = hf_ft.compare_hf_gpt2_finetune_run_cards(
                [risky_path, safe_path],
                run_labels=["risky", "safe"],
            )
            sweep_report = {
                "row_type": "hf_gpt2_finetune_sweep_report",
                "dry_run": False,
                "run_count": 2,
                "completed_run_count": 2,
                "failed_run_count": 0,
                "comparison": comparison,
                "runs": [
                    {
                        "name": "risky",
                        "status": "completed",
                        "returncode": 0,
                        "run_card": str(risky_path),
                        "trainer_trace_jsonl": str(risky_trace),
                        "run_dir": str(Path(tmp) / "risky-run"),
                        "output_dir": str(risky_output_dir),
                        "command": risky_command,
                        "command_display": " ".join(risky_command),
                    },
                    {
                        "name": "safe",
                        "status": "completed",
                        "returncode": 0,
                        "run_card": str(safe_path),
                        "trainer_trace_jsonl": str(safe_trace),
                        "run_dir": str(Path(tmp) / "safe-run"),
                        "output_dir": str(safe_output_dir),
                        "command": safe_command,
                        "command_display": " ".join(safe_command),
                    },
                ],
            }
            sweep_summary = hf_ft.summarize_hf_gpt2_finetune_sweep_report(
                sweep_report,
                top_n=2,
            )
            lines = hf_ft.summarize_hf_gpt2_finetune_sweep_report_lines(
                sweep_report,
                top_n=2,
            )
            scale_up_command = hf_ft.hf_gpt2_finetune_scale_up_command(
                sweep_summary
            )
            explicit_scale_up_command = hf_ft.hf_gpt2_finetune_scale_up_command(
                sweep_summary,
                model_name=safe_checkpoint,
                resume_from_checkpoint=safe_checkpoint,
                max_steps=64,
                max_train_samples=4096,
                max_eval_samples=256,
                output_dir=Path(tmp) / "safe-long-run",
            )

        self.assertEqual(comparison["best_eval_after_run_label"], "risky")
        self.assertEqual(comparison["best_distortion_adjusted_run_label"], "safe")
        self.assertAlmostEqual(
            comparison["summaries"][0]["distortion_pressure_index"],
            1.0,
        )
        self.assertAlmostEqual(
            comparison["summaries"][0]["distortion_adjusted_eval_loss"],
            1.6,
        )
        self.assertAlmostEqual(
            comparison["summaries"][1]["distortion_pressure_index"],
            0.0,
        )
        self.assertAlmostEqual(
            comparison["summaries"][1]["distortion_adjusted_eval_loss"],
            1.54,
        )
        self.assertEqual(sweep_summary["selected_run_label"], "risky")
        self.assertEqual(sweep_summary["scale_up_candidate_label"], "safe")
        self.assertEqual(
            sweep_summary["scale_up_candidate_reason"],
            "lowest_distortion_adjusted_eval_loss",
        )
        self.assertEqual(sweep_summary["scale_up_candidate_status"], "completed")
        self.assertEqual(sweep_summary["scale_up_candidate_run_card"], str(safe_path))
        self.assertEqual(
            sweep_summary["scale_up_candidate_trainer_trace_jsonl"],
            str(safe_trace),
        )
        self.assertEqual(
            sweep_summary["scale_up_candidate_command"],
            safe_command,
        )
        self.assertEqual(scale_up_command["status"], "ok")
        self.assertEqual(scale_up_command["scale_up_candidate_label"], "safe")
        self.assertEqual(scale_up_command["base_command"], safe_command)
        self.assertIn("--max-steps 20", scale_up_command["command_display"])
        self.assertIn("--max-train-samples 200", scale_up_command["command_display"])
        self.assertIn(
            f"--output-dir {safe_output_dir}-scaleup",
            scale_up_command["command_display"],
        )
        self.assertIn(
            f"--run-card {safe_output_dir}-scaleup/spiraltorch-hf-gpt2-ft-run-card.json",
            scale_up_command["command_display"],
        )
        self.assertIn(
            (
                f"--trainer-trace-jsonl {safe_output_dir}-scaleup/"
                "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            ),
            scale_up_command["command_display"],
        )
        self.assertIn("--max-eval-samples 20", scale_up_command["command_display"])
        self.assertEqual(scale_up_command["applied_override_count"], 5)
        self.assertIn("--max-steps 64", explicit_scale_up_command["command_display"])
        self.assertIn(
            "--max-train-samples 4096",
            explicit_scale_up_command["command_display"],
        )
        self.assertIn(
            "--max-eval-samples 256",
            explicit_scale_up_command["command_display"],
        )
        self.assertIn(
            f"--output-dir {Path(tmp) / 'safe-long-run'}",
            explicit_scale_up_command["command_display"],
        )
        self.assertIn(
            f"--model-name {safe_checkpoint}",
            explicit_scale_up_command["command_display"],
        )
        self.assertIn(
            f"--resume-from-checkpoint {safe_checkpoint}",
            explicit_scale_up_command["command_display"],
        )
        self.assertEqual(explicit_scale_up_command["applied_override_count"], 8)
        self.assertTrue(
            any("hf_gpt2_ft_sweep_scale_up candidate=safe" in line for line in lines)
        )
        self.assertTrue(any(f"card={safe_path}" in line for line in lines))
        missing_scale_up_command = hf_ft.hf_gpt2_finetune_scale_up_command(
            {
                "row_type": "hf_gpt2_finetune_sweep_report_summary",
                "scale_up_candidate_label": "missing",
            }
        )
        self.assertEqual(
            missing_scale_up_command["status"],
            "missing_candidate_command",
        )

    def test_scale_up_accepts_command_manifest_for_future_checkpoint_handoff(
        self,
    ) -> None:
        scale_up_module = load_scale_up_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_output_dir = tmp_path / "fineweb-8192"
            base_card = base_output_dir / "spiraltorch-hf-gpt2-ft-run-card.json"
            base_trace = (
                base_output_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            )
            manifest = {
                "row_type": "hf_gpt2_finetune_wait_launch",
                "next_run": "fineweb-8192",
                "command": [
                    sys.executable,
                    "bindings/st-py/examples/hf_gpt2_finetune_bridge.py",
                    "--model-name",
                    "gpt2",
                    "--dataset-name",
                    "HuggingFaceFW/fineweb-edu",
                    "--dataset-config",
                    "CC-MAIN-2025-26",
                    "--dataset-streaming",
                    "--output-dir",
                    str(base_output_dir),
                    "--run-card",
                    str(base_card),
                    "--trainer-trace-jsonl",
                    str(base_trace),
                    "--max-steps",
                    "8192",
                    "--max-train-samples",
                    "32768",
                    "--max-eval-samples",
                    "2048",
                ],
            }
            manifest_path = tmp_path / "launch-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            future_checkpoint = base_output_dir / "checkpoint-8192"
            next_output_dir = tmp_path / "fineweb-16384"
            command_artifact = tmp_path / "fineweb-16384-command.json"

            direct_command = hf_ft.hf_gpt2_finetune_scale_up_command(
                manifest,
                model_name=future_checkpoint,
                resume_from_checkpoint=future_checkpoint,
                max_steps=16384,
                max_train_samples=131072,
                max_eval_samples=4096,
                max_eval_blocks=4096,
                streaming_validation_samples=4096,
                output_dir=next_output_dir,
                trainer_trace_run_id="fineweb-16384",
            )
            args = scale_up_module.parse_args(
                [
                    str(manifest_path),
                    "--write-command",
                    str(command_artifact),
                    "--allow-missing-resume-checkpoint",
                    "--model-name",
                    str(future_checkpoint),
                    "--resume-from-checkpoint",
                    str(future_checkpoint),
                    "--max-steps",
                    "16384",
                    "--max-train-samples",
                    "131072",
                    "--max-eval-samples",
                    "4096",
                    "--max-eval-blocks",
                    "4096",
                    "--streaming-validation-samples",
                    "4096",
                    "--output-dir",
                    str(next_output_dir),
                    "--trainer-trace-run-id",
                    "fineweb-16384",
                ]
            )
            cli_command = scale_up_module.run_scale_up(args)
            written = json.loads(command_artifact.read_text(encoding="utf-8"))

        self.assertEqual(direct_command["status"], "ok")
        self.assertEqual(
            direct_command["scale_up_candidate_label"],
            "fineweb-8192",
        )
        self.assertEqual(
            direct_command["scale_up_candidate_reason"],
            "source_command_manifest",
        )
        self.assertIn("HuggingFaceFW/fineweb-edu", direct_command["command_display"])
        self.assertIn("--max-steps 16384", direct_command["command_display"])
        self.assertIn("--max-train-samples 131072", direct_command["command_display"])
        self.assertIn("--max-eval-samples 4096", direct_command["command_display"])
        self.assertIn("--max-eval-blocks 4096", direct_command["command_display"])
        self.assertIn(
            "--streaming-validation-samples 4096",
            direct_command["command_display"],
        )
        self.assertIn(
            "--trainer-trace-run-id fineweb-16384",
            direct_command["command_display"],
        )
        self.assertIn(
            f"--model-name {future_checkpoint}",
            direct_command["command_display"],
        )
        self.assertIn(
            f"--resume-from-checkpoint {future_checkpoint}",
            direct_command["command_display"],
        )
        self.assertEqual(cli_command["status"], "ok")
        self.assertEqual(cli_command["preflight_status"], "blocked")
        self.assertEqual(cli_command["preflight_error_count"], 1)
        self.assertEqual(written["status"], "ok")
        self.assertTrue(
            any(
                issue.get("field") == "--resume-from-checkpoint"
                and issue.get("severity") == "error"
                for issue in written["preflight"]["issues"]
            )
        )

    def test_sweep_example_builds_grid_and_writes_dry_run_report(self) -> None:
        module = load_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            validation_path = Path(tmp) / "validation.txt"
            distortion_report_path = Path(tmp) / "distortion-sweep-report.json"
            train_path.write_text("alpha spiral\nbeta zspace\n", encoding="utf-8")
            validation_path.write_text("gamma eval\n", encoding="utf-8")
            distortion_report_path.write_text(
                json.dumps(
                    {
                        "row_type": "zspace_inference_distortion_sweep",
                        "status": "complete",
                        "prompt": "SpiralTorch is",
                        "runtime": {
                            "api_provider": "fake",
                            "api_model": "fake-distorted-api",
                        },
                        "runs": [
                            {
                                "name": "strong",
                                "status": "ok",
                                "probe_path": "/tmp/missing-strong-probe.json",
                                "config": {
                                    "desire_pressure": 0.8,
                                    "desire_stability": 0.45,
                                    "psi_total": 0.7,
                                    "coherence": 0.5,
                                    "distortion_strength": 1.0,
                                    "base_temperature": 0.7,
                                    "base_top_p": 0.95,
                                },
                            }
                        ],
                        "comparison": {
                            "row_type": "zspace_inference_distortion_probe_comparison",
                            "recommended_probe": "strong",
                            "recommended_reason": (
                                "highest_effect_score_lowest_risk_tiebreak"
                            ),
                            "top_probes": [
                                {
                                    "label": "strong",
                                    "effect_score": 0.91,
                                    "risk_score": 0.22,
                                    "api_provider": "fake",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
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
                    "--generation-zspace-softmax",
                    "--generation-zspace-entropy-target",
                    "3.0",
                    "--generation-repression-strength",
                    "1.25",
                    "--generation-ngram-size",
                    "3",
                    "--generation-ngram-window",
                    "96",
                    "--generation-ngram-repression-strength",
                    "0.75",
                    "--generation-ngram-decay",
                    "0.9",
                    "--trainer-telemetry",
                    "--trainer-telemetry-prefix",
                    "hf_ft",
                    "--trainer-desire-gain",
                    "1.2",
                    "--trainer-psi-gain",
                    "0.8",
                    "--inference-distortion-sweep-report",
                    str(distortion_report_path),
                    "--generation-zspace-report-limit",
                    "2",
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
            stored_plan = json.loads((out_dir / "sweep-plan.json").read_text())
            stored_report = json.loads((out_dir / "sweep-report.json").read_text())
            scale_up_command = json.loads(
                (out_dir / "scale-up-command.json").read_text()
            )
            sweep_lines = hf_ft.summarize_hf_gpt2_finetune_sweep_report_lines(
                stored_report,
            )

        self.assertEqual(len(runs), 8)
        self.assertEqual(report["run_count"], 8)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["skipped_run_count"], 8)
        self.assertEqual(report["reused_run_count"], 0)
        self.assertEqual(report["summary"]["status"], "planned")
        self.assertEqual(report["scale_up_command_status"], "missing_candidate_command")
        self.assertEqual(
            report["scale_up_command_path"],
            str(out_dir / "scale-up-command.json"),
        )
        self.assertEqual(scale_up_command["status"], "missing_candidate_command")
        self.assertEqual(
            scale_up_command["artifact_path"],
            str(out_dir / "scale-up-command.json"),
        )
        self.assertEqual(stored_report["row_type"], "hf_gpt2_finetune_sweep_report")
        self.assertEqual(stored_report["summary"]["run_count"], 8)
        self.assertEqual(
            stored_report["summary"]["scale_up_command_status"],
            "missing_candidate_command",
        )
        self.assertEqual(
            stored_report["inference_distortion_sweep_report"],
            str(distortion_report_path),
        )
        self.assertEqual(
            stored_plan["inference_distortion_handoff"]["recommended_probe"],
            "strong",
        )
        self.assertTrue(
            any(
                "probe=strong" in line
                for line in stored_plan["inference_distortion_handoff_lines"]
            )
        )
        self.assertEqual(
            stored_report["inference_distortion_handoff"]["recommended_probe"],
            "strong",
        )
        self.assertTrue(
            any(
                "probe=strong" in line
                for line in stored_report["inference_distortion_handoff_lines"]
            )
        )
        self.assertEqual(
            stored_report["summary"]["inference_distortion_recommended_probe"],
            "strong",
        )
        self.assertTrue(stored_plan["trainer_telemetry_requested"])
        self.assertTrue(stored_plan["trainer_telemetry_enabled"])
        self.assertIsNone(stored_plan["trainer_telemetry_auto_reason"])
        self.assertTrue(stored_report["summary"]["trainer_telemetry_enabled"])
        self.assertTrue(any("probe=strong" in line for line in sweep_lines))
        self.assertTrue(
            any(
                "hf_gpt2_ft_sweep_scale_up_command "
                "status=missing_candidate_command" in line
                for line in sweep_lines
            )
        )
        self.assertTrue(
            any("hf_gpt2_ft_sweep_trainer_telemetry" in line for line in sweep_lines)
        )
        first_command = runs[0]["command"]
        self.assertIn("--train", first_command)
        self.assertIn("--corpus-scan", first_command)
        self.assertIn("--generation-do-sample", first_command)
        self.assertIn("--generation-zspace-softmax", first_command)
        self.assertIn("--generation-zspace-entropy-target", first_command)
        self.assertIn("3.0", first_command)
        self.assertIn("--generation-repression-strength", first_command)
        self.assertIn("1.25", first_command)
        self.assertIn("--generation-ngram-size", first_command)
        self.assertIn("3", first_command)
        self.assertIn("--generation-ngram-window", first_command)
        self.assertIn("96", first_command)
        self.assertIn("--generation-ngram-repression-strength", first_command)
        self.assertIn("0.75", first_command)
        self.assertIn("--generation-ngram-decay", first_command)
        self.assertIn("0.9", first_command)
        self.assertIn("--trainer-telemetry", first_command)
        self.assertIn("--trainer-telemetry-prefix", first_command)
        self.assertIn("hf_ft", first_command)
        self.assertIn("--trainer-desire-gain", first_command)
        self.assertIn("1.2", first_command)
        self.assertIn("--trainer-psi-gain", first_command)
        self.assertIn("0.8", first_command)
        self.assertIn("--inference-distortion-sweep-report", first_command)
        self.assertIn(str(distortion_report_path), first_command)
        self.assertIn("--generation-zspace-report-limit", first_command)
        self.assertIn("2", first_command)
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

    def test_sweep_example_accepts_inference_distortion_probe(self) -> None:
        module = load_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            probe_path = Path(tmp) / "direct-probe.json"
            train_path.write_text("alpha spiral\nbeta zspace\n", encoding="utf-8")
            probe_path.write_text(
                json.dumps(
                    {
                        "row_type": "zspace_inference_distortion_probe",
                        "prompt": "SpiralTorch direct sweep probe",
                        "runtime": {
                            "api_provider": "fake",
                            "api_model": "fake-distorted-api",
                        },
                        "config": {
                            "desire_pressure": 0.84,
                            "desire_stability": 0.42,
                            "psi_total": 0.74,
                            "coherence": 0.5,
                            "distortion_strength": 1.0,
                            "base_temperature": 0.7,
                            "base_top_p": 0.95,
                        },
                        "adapter": {
                            "request": {"temperature": 1.04, "top_p": 0.78},
                            "logits_processor_kwargs": {
                                "entropy_target": 3.25,
                                "repression_strength": 1.5,
                                "ngram_repression_strength": 0.75,
                            },
                            "activation_hook": {"name_contains": ["attn"]},
                        },
                        "local_hf": {
                            "status": "ok",
                            "changed": True,
                            "generation_control": {
                                "status": "ok",
                                "top_token_changed_count": 3,
                            },
                            "activation_report": {
                                "status": "ok",
                                "event_count": 6,
                            },
                        },
                        "api": {
                            "provider": "fake",
                            "model": "fake-distorted-api",
                            "text": "Fake probe",
                        },
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--out-dir",
                    str(out_dir),
                    "--train-file",
                    str(train_path),
                    "--block-size-values",
                    "8",
                    "--learning-rate-values",
                    "0.001",
                    "--max-step-values",
                    "1",
                    "--seed-values",
                    "7",
                    "--generation-prompt",
                    "SpiralTorch is",
                    "--generation-from-inference-distortion",
                    "--inference-distortion-probe",
                    str(probe_path),
                ]
            )
            runs = module.build_sweep_runs(args)
            report = module.run_sweep(args)
            stored_plan = json.loads((out_dir / "sweep-plan.json").read_text())
            stored_report = json.loads((out_dir / "sweep-report.json").read_text())

        self.assertEqual(report["run_count"], 1)
        self.assertEqual(stored_plan["inference_distortion_probe"], str(probe_path))
        self.assertIsNone(stored_plan["inference_distortion_sweep_report"])
        self.assertFalse(stored_plan["trainer_telemetry_requested"])
        self.assertTrue(stored_plan["trainer_telemetry_enabled"])
        self.assertEqual(
            stored_plan["trainer_telemetry_auto_reason"],
            "inference_distortion_handoff",
        )
        self.assertEqual(
            stored_report["inference_distortion_handoff"]["source_kind"],
            "probe",
        )
        self.assertTrue(
            any(
                "probe=direct-probe" in line
                for line in stored_report["inference_distortion_handoff_lines"]
            )
        )
        self.assertTrue(
            any(
                "hf_gpt2_ft_inference_handoff_replay" in line
                for line in stored_report["inference_distortion_handoff_lines"]
            )
        )
        self.assertTrue(
            any(
                "hf_gpt2_ft_inference_handoff_generation" in line
                for line in stored_report["inference_distortion_handoff_lines"]
            )
        )
        self.assertEqual(
            stored_report["summary"]["inference_distortion_recommended_probe"],
            "direct-probe",
        )
        self.assertTrue(
            any(
                "probe=direct-probe" in line
                for line in stored_report["summary"][
                    "inference_distortion_handoff_lines"
                ]
            )
        )
        self.assertIn(
            "--generation-repression-strength",
            stored_report["summary"]["inference_distortion_replay_cli_preview"],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            stored_report["summary"][
                "inference_distortion_generation_handoff_cli_preview"
            ],
        )
        self.assertIn(
            "--generation-from-inference-distortion",
            stored_report["summary"][
                "inference_distortion_generation_handoff_cli_display"
            ],
        )
        self.assertIn(
            "--inference-distortion-probe",
            stored_report["summary"][
                "inference_distortion_generation_handoff_cli_args"
            ],
        )
        self.assertEqual(
            stored_plan["generation_from_inference_distortion_plan"]["status"],
            "ok",
        )
        self.assertEqual(
            stored_plan["generation_from_inference_distortion_plan"][
                "processor_kwargs"
            ]["repression_strength"],
            1.5,
        )
        self.assertIn(
            "--generation-repression-strength",
            stored_plan["generation_from_inference_distortion_plan"][
                "bridge_cli_args"
            ],
        )
        self.assertEqual(
            stored_report["summary"][
                "generation_from_inference_distortion_plan_status"
            ],
            "ok",
        )
        self.assertFalse(stored_report["summary"]["trainer_telemetry_requested"])
        self.assertTrue(stored_report["summary"]["trainer_telemetry_enabled"])
        self.assertEqual(
            stored_report["summary"]["trainer_telemetry_auto_reason"],
            "inference_distortion_handoff",
        )
        self.assertEqual(
            stored_report["summary"][
                "generation_from_inference_distortion_plan_probe"
            ],
            "direct-probe",
        )
        self.assertEqual(
            stored_report["summary"][
                "generation_from_inference_distortion_plan_entropy_target"
            ],
            3.25,
        )
        self.assertEqual(
            stored_report["summary"][
                "generation_from_inference_distortion_plan_repression_strength"
            ],
            1.5,
        )
        self.assertEqual(
            stored_report["summary"][
                "generation_from_inference_distortion_plan_ngram_repression_strength"
            ],
            0.75,
        )
        plan_bridge_cli_args = stored_report["summary"][
            "generation_from_inference_distortion_plan_bridge_cli_args"
        ]
        self.assertIn("--generation-zspace-softmax", plan_bridge_cli_args)
        self.assertIn("--generation-zspace-entropy-target", plan_bridge_cli_args)
        self.assertIn("3.25", plan_bridge_cli_args)
        self.assertIn("--generation-repression-strength", plan_bridge_cli_args)
        self.assertIn("1.5", plan_bridge_cli_args)
        self.assertIn("--generation-ngram-repression-strength", plan_bridge_cli_args)
        self.assertIn("0.75", plan_bridge_cli_args)
        direct_lines = hf_ft.summarize_hf_gpt2_finetune_sweep_report_lines(
            stored_report,
        )
        self.assertTrue(
            any("hf_gpt2_ft_sweep_generation_inference_plan" in line for line in direct_lines)
        )
        self.assertTrue(
            any(
                "hf_gpt2_ft_sweep_trainer_telemetry" in line
                and "auto=inference_distortion_handoff" in line
                for line in direct_lines
            )
        )
        self.assertTrue(any("repress=1.5" in line for line in direct_lines))
        first_command = runs[0]["command"]
        self.assertIn("--inference-distortion-probe", first_command)
        self.assertIn(str(probe_path), first_command)
        self.assertNotIn("--inference-distortion-sweep-report", first_command)
        self.assertIn("--generation-from-inference-distortion", first_command)

    def test_bridge_auto_enables_trainer_telemetry_for_inference_handoff(self) -> None:
        module = load_bridge_example()
        args = types.SimpleNamespace(
            trainer_telemetry=False,
            no_trainer_trace=False,
        )
        handoff = {"row_type": "hf_gpt2_finetune_inference_distortion_handoff"}

        self.assertTrue(module._trainer_telemetry_enabled(args, handoff))
        self.assertEqual(
            module._trainer_telemetry_auto_reason(args, handoff),
            "inference_distortion_handoff",
        )

        args.trainer_telemetry = True
        self.assertTrue(module._trainer_telemetry_enabled(args, handoff))
        self.assertIsNone(module._trainer_telemetry_auto_reason(args, handoff))

        args.trainer_telemetry = False
        args.no_trainer_trace = True
        self.assertFalse(module._trainer_telemetry_enabled(args, handoff))
        self.assertIsNone(module._trainer_telemetry_auto_reason(args, handoff))

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
            stored_report = json.loads((out_dir / "sweep-report.json").read_text())
            scale_up_command = json.loads(
                (out_dir / "scale-up-command.json").read_text()
            )
            sweep_lines = hf_ft.summarize_hf_gpt2_finetune_sweep_report_lines(
                stored_report,
            )

        self.assertEqual(report["attempted_run_count"], 2)
        self.assertEqual(report["completed_run_count"], 2)
        self.assertEqual(report["failed_run_count"], 0)
        self.assertEqual(report["reused_run_count"], 0)
        self.assertEqual(report["comparison"]["run_count"], 2)
        self.assertIn("seed13", report["comparison"]["best_eval_after_run_label"])
        self.assertIn("seed13", report["summary"]["selected_run_label"])
        self.assertEqual(report["scale_up_command_status"], "ok")
        self.assertEqual(
            report["scale_up_command_path"],
            str(out_dir / "scale-up-command.json"),
        )
        self.assertEqual(scale_up_command["status"], "ok")
        self.assertIn("seed13", scale_up_command["scale_up_candidate_label"])
        self.assertIn("--max-steps 2", scale_up_command["command_display"])
        self.assertIn("--max-train-samples 8192", scale_up_command["command_display"])
        self.assertIn("-scaleup", scale_up_command["command_display"])
        self.assertEqual(scale_up_command["applied_override_count"], 5)
        self.assertEqual(stored_report["summary"]["scale_up_command_status"], "ok")
        self.assertIn(
            "--max-steps 2",
            stored_report["summary"]["scale_up_command_preview"],
        )
        self.assertTrue(
            any(
                "hf_gpt2_ft_sweep_scale_up_command status=ok" in line
                for line in sweep_lines
            )
        )

    def test_scale_up_example_replays_sweep_report_and_command_artifact(self) -> None:
        sweep_module = load_sweep_example()
        scale_up_module = load_scale_up_example()
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            train_path.write_text("alpha spiral\nbeta zspace\n", encoding="utf-8")
            out_dir = Path(tmp) / "sweep"
            sweep_args = sweep_module.parse_args(
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

            def fake_sweep_run(command, *, check=False):
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

            with mock.patch.object(
                sweep_module.subprocess,
                "run",
                side_effect=fake_sweep_run,
            ):
                sweep_module.run_sweep(sweep_args)

            sweep_report_path = out_dir / "sweep-report.json"
            scale_up_artifact_path = out_dir / "scale-up-command.json"
            rewritten_artifact_path = out_dir / "scale-up-command-long.json"
            resume_checkpoint = out_dir / "checkpoint-1"
            resume_checkpoint.mkdir()
            (resume_checkpoint / "model.safetensors").write_bytes(b"x" * 1024)
            scale_up_args = scale_up_module.parse_args(
                [
                    str(sweep_report_path),
                    "--write-command",
                    str(rewritten_artifact_path),
                    "--model-name",
                    str(resume_checkpoint),
                    "--resume-from-checkpoint",
                    str(resume_checkpoint),
                    "--max-steps",
                    "64",
                    "--max-train-samples",
                    "4096",
                    "--max-eval-samples",
                    "256",
                    "--output-dir",
                    str(out_dir / "long-run"),
                ]
            )
            scale_up = scale_up_module.run_scale_up(scale_up_args)
            rewritten = json.loads(rewritten_artifact_path.read_text())
            direct_preflight = hf_ft.hf_gpt2_finetune_scale_up_preflight_report(
                rewritten_artifact_path
            )
            direct_preflight_lines = (
                hf_ft.hf_gpt2_finetune_scale_up_preflight_lines(direct_preflight)
            )
            broken_artifact = dict(rewritten)
            broken_command = [str(item) for item in rewritten["command"]]
            broken_train_path = Path(tmp) / "missing-train.txt"
            broken_command[broken_command.index("--train-file") + 1] = str(
                broken_train_path
            )
            broken_artifact["command"] = broken_command
            broken_artifact_path = out_dir / "scale-up-command-broken.json"
            broken_artifact_path.write_text(json.dumps(broken_artifact), encoding="utf-8")
            broken_args = scale_up_module.parse_args(
                [
                    str(broken_artifact_path),
                    "--require-ready",
                ]
            )
            broken = scale_up_module.run_scale_up(broken_args)
            direct_broken_preflight = st.hf_gpt2_finetune_scale_up_preflight_report(
                broken_artifact_path
            )
            from_command_artifact_args = scale_up_module.parse_args(
                [
                    str(scale_up_artifact_path),
                    "--output-suffix",
                    "longer",
                ]
            )
            from_command_artifact = scale_up_module.run_scale_up(
                from_command_artifact_args
            )
            wait_launch_artifact_path = out_dir / "scale-up-wait-launch-command.json"
            wait_launch_manifest = out_dir / "long-run-wait-launch.json"
            wait_launch_history = out_dir / "long-run-wait-launch-history.jsonl"
            wait_launch_args = scale_up_module.parse_args(
                [
                    str(rewritten_artifact_path),
                    "--write-command",
                    str(wait_launch_artifact_path),
                    "--wait-launch-manifest",
                    str(wait_launch_manifest),
                    "--wait-launch-jsonl-out",
                    str(wait_launch_history),
                    "--wait-launch-pid",
                    "123",
                    "--wait-launch-checkpoint",
                    str(resume_checkpoint),
                    "--wait-launch-launched-pid-file",
                    str(out_dir / "long-run.pid"),
                    "--wait-launch-launched-log-file",
                    str(out_dir / "long-run.log"),
                    "--wait-launch-launched-log-mode",
                    "write",
                    "--wait-launch-detach",
                ]
            )
            wait_launch_command = scale_up_module.run_scale_up(wait_launch_args)
            wait_launch_written = json.loads(wait_launch_artifact_path.read_text())

            exact_run_args = scale_up_module.parse_args(
                [
                    str(rewritten_artifact_path),
                    "--run",
                ]
            )

            def fake_exact_replay(command, *, check=False):
                del check
                self.assertIn("--max-steps", command)
                self.assertEqual(command[command.index("--max-steps") + 1], "64")
                self.assertIn("--output-dir", command)
                self.assertEqual(
                    command[command.index("--output-dir") + 1],
                    str(out_dir / "long-run"),
                )
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(
                scale_up_module.subprocess,
                "run",
                side_effect=fake_exact_replay,
            ) as exact_run_mock:
                exact_executed = scale_up_module.run_scale_up(exact_run_args)

            run_result_artifact_path = out_dir / "scale-up-command-run-result.json"
            run_args = scale_up_module.parse_args(
                [
                    str(rewritten_artifact_path),
                    "--run",
                    "--write-command",
                    str(run_result_artifact_path),
                    "--max-steps",
                    "128",
                    "--output-dir",
                    str(out_dir / "long-run-exec"),
                ]
            )

            def fake_scale_up_run(command, *, check=False):
                del check
                self.assertIn("--max-steps", command)
                self.assertEqual(command[command.index("--max-steps") + 1], "128")
                self.assertIn("--output-dir", command)
                self.assertEqual(
                    command[command.index("--output-dir") + 1],
                    str(out_dir / "long-run-exec"),
                )
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(
                scale_up_module.subprocess,
                "run",
                side_effect=fake_scale_up_run,
            ) as run_mock:
                executed = scale_up_module.run_scale_up(run_args)
            run_result = json.loads(run_result_artifact_path.read_text())

        self.assertEqual(scale_up["status"], "ok")
        self.assertEqual(scale_up["preflight_status"], "ready")
        self.assertEqual(scale_up["preflight_error_count"], 0)
        self.assertEqual(rewritten["status"], "ok")
        self.assertEqual(rewritten["preflight_status"], "ready")
        self.assertEqual(direct_preflight["status"], "ready")
        self.assertEqual(direct_preflight["error_count"], 0)
        self.assertEqual(
            direct_preflight["disk_plan"]["resume_checkpoint_bytes"],
            1024,
        )
        self.assertEqual(
            direct_preflight["disk_plan"]["estimated_peak_checkpoint_bytes"],
            2048,
        )
        self.assertEqual(direct_preflight["disk_plan"]["save_total_limit"], 1)
        self.assertTrue(
            any(
                "hf_gpt2_ft_scale_up_preflight status=ready" in line
                for line in direct_preflight_lines
            )
        )
        self.assertTrue(
            any("hf_gpt2_ft_scale_up_disk_plan" in line for line in direct_preflight_lines)
        )
        self.assertIn("seed13", scale_up["scale_up_candidate_label"])
        self.assertIn("--max-steps 64", scale_up["command_display"])
        self.assertIn("--max-train-samples 4096", scale_up["command_display"])
        self.assertIn("--max-eval-samples 256", scale_up["command_display"])
        self.assertIn(f"--output-dir {out_dir / 'long-run'}", scale_up["command_display"])
        self.assertIn(f"--model-name {resume_checkpoint}", scale_up["command_display"])
        self.assertIn(
            f"--resume-from-checkpoint {resume_checkpoint}",
            scale_up["command_display"],
        )
        self.assertEqual(scale_up["artifact_path"], str(rewritten_artifact_path))
        self.assertEqual(broken["preflight_status"], "blocked")
        self.assertEqual(broken["run_returncode"], 2)
        self.assertEqual(direct_broken_preflight["status"], "blocked")
        self.assertTrue(
            any(
                issue.get("field") == "--train-file"
                and issue.get("severity") == "error"
                for issue in broken["preflight"]["issues"]
            )
        )
        self.assertEqual(from_command_artifact["status"], "ok")
        self.assertEqual(from_command_artifact["preflight_status"], "ready")
        self.assertIn("-longer", from_command_artifact["command_display"])
        self.assertIn("--max-steps 2", from_command_artifact["command_display"])
        self.assertEqual(wait_launch_command["status"], "ok")
        self.assertEqual(wait_launch_command["preflight_status"], "ready")
        self.assertEqual(
            wait_launch_command["wait_launch_manifest"],
            str(wait_launch_manifest),
        )
        self.assertTrue(wait_launch_command["wait_launch_detach"])
        self.assertIn("--detach", wait_launch_command["wait_launch_command"])
        self.assertIn("--checkpoint", wait_launch_command["wait_launch_command"])
        self.assertIn(
            str(resume_checkpoint),
            wait_launch_command["wait_launch_command"],
        )
        self.assertIn("--", wait_launch_command["wait_launch_command"])
        self.assertEqual(
            wait_launch_written["wait_launch_command"],
            wait_launch_command["wait_launch_command"],
        )
        self.assertIn(
            "hf_gpt2_finetune_wait_launch.py",
            wait_launch_command["wait_launch_command_display"],
        )
        self.assertIn(
            "--max-steps 64",
            wait_launch_command["wait_launch_command_display"],
        )
        self.assertEqual(exact_executed["preflight_status"], "ready")
        self.assertEqual(exact_executed["run_returncode"], 0)
        exact_run_mock.assert_called_once()
        self.assertEqual(executed["preflight_status"], "ready")
        self.assertEqual(executed["run_returncode"], 0)
        self.assertEqual(run_result["preflight_status"], "ready")
        self.assertEqual(run_result["run_returncode"], 0)
        self.assertIn("--max-steps 128", run_result["command_display"])
        run_mock.assert_called_once()

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
            save_total_limit=2,
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
            save_total_limit=2,
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

    def test_example_training_arguments_include_save_total_limit(self) -> None:
        module = load_bridge_example()

        class SaveLimitTrainingArguments:
            def __init__(self, output_dir=None, save_total_limit=None):
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
            save_total_limit=1,
            seed=13,
            max_steps=10,
            eval_steps=5,
        )
        raw = module._raw_training_arguments_kwargs(
            args,
            has_eval=False,
            cls=SaveLimitTrainingArguments,
        )
        filtered = module._filter_training_arguments_kwargs(
            SaveLimitTrainingArguments,
            raw,
        )

        self.assertEqual(raw["save_total_limit"], 1)
        self.assertEqual(filtered["save_total_limit"], 1)

    def test_example_disk_report_records_free_space_and_threshold(self) -> None:
        module = load_bridge_example()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "future-run"
            checkpoint = Path(tmp) / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"x" * 2048)
            ok_report = module._disk_report(output_dir, min_free_gb=0.0)
            blocked_report = module._disk_report(output_dir, min_free_gb=10**9)
            headroom = hf_ft.hf_gpt2_finetune_disk_headroom_plan(
                output_dir,
                resume_from_checkpoint=checkpoint,
                save_total_limit=2,
            )
            headroom_lines = hf_ft.hf_gpt2_finetune_summary_lines(
                {
                    "hf_model_name": "gpt2",
                    "hf_dataset_name": "local-files",
                    "hf_dataset_config": "text",
                    "hf_train_split": "train",
                    "hf_text_column": "text",
                    "disk_headroom_plan": headroom,
                }
            )

        self.assertEqual(ok_report["row_type"], "hf_gpt2_ft_disk_report")
        self.assertEqual(ok_report["path"], str(output_dir))
        self.assertEqual(ok_report["status"], "ok")
        self.assertGreater(ok_report["free_bytes"], 0)
        self.assertTrue(ok_report["meets_min_free"])
        self.assertEqual(blocked_report["status"], "blocked")
        self.assertFalse(blocked_report["meets_min_free"])
        self.assertEqual(
            headroom["row_type"],
            "hf_gpt2_finetune_disk_headroom_plan",
        )
        self.assertEqual(headroom["resume_checkpoint_bytes"], 2048)
        self.assertEqual(headroom["save_total_limit"], 2)
        self.assertEqual(headroom["estimated_peak_checkpoint_count"], 3)
        self.assertEqual(headroom["estimated_peak_checkpoint_bytes"], 6144)
        self.assertTrue(
            any("hf_gpt2_ft_disk_headroom" in line for line in headroom_lines)
        )

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

    def test_example_model_train_dtype_auto_casts_saved_fp16_checkpoint(self) -> None:
        module = load_bridge_example()

        class FakeParam:
            def __init__(self, dtype):
                self.dtype = dtype

        class FakeModel:
            def __init__(self, dtype):
                self.param = FakeParam(dtype)
                self.float_calls = 0

            def parameters(self):
                return iter([self.param])

            def float(self):
                self.float_calls += 1
                self.param.dtype = "torch.float32"
                return self

        fp16_model = FakeModel("torch.float16")
        fp32_model = FakeModel("torch.float32")
        fp16_report = module._prepare_model_train_dtype(
            fp16_model,
            types.SimpleNamespace(train=True, model_train_dtype="auto"),
        )
        fp32_report = module._prepare_model_train_dtype(
            fp32_model,
            types.SimpleNamespace(train=True, model_train_dtype="auto"),
        )
        native_report = module._prepare_model_train_dtype(
            FakeModel("torch.float16"),
            types.SimpleNamespace(train=True, model_train_dtype="native"),
        )

        self.assertEqual(fp16_report["cast_status"], "cast_float32")
        self.assertEqual(fp16_report["dtype_before"], "torch.float16")
        self.assertEqual(fp16_report["dtype_after"], "torch.float32")
        self.assertEqual(fp16_model.float_calls, 1)
        self.assertEqual(fp32_report["cast_status"], "not_requested")
        self.assertEqual(fp32_model.float_calls, 0)
        self.assertEqual(native_report["cast_status"], "not_requested")

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

    def test_example_trainer_train_kwargs_records_resume_checkpoint(self) -> None:
        module = load_bridge_example()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-512"
            checkpoint.mkdir()
            args = module.parse_args(
                [
                    "--metadata-only",
                    "--resume-from-checkpoint",
                    str(checkpoint),
                ]
            )

        self.assertEqual(
            module._trainer_train_kwargs(args),
            {"resume_from_checkpoint": str(checkpoint)},
        )
        self.assertEqual(
            module._trainer_train_kwargs(
                types.SimpleNamespace(resume_from_checkpoint=None)
            ),
            {},
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

    def test_example_streaming_remote_dataset_materializes_bounded_splits(self) -> None:
        module = load_bridge_example()

        class FakeStream:
            def __init__(self, rows):
                self.rows = list(rows)
                self.shuffle_calls = []

            def __iter__(self):
                return iter(self.rows)

            def shuffle(self, *, buffer_size, seed):
                self.shuffle_calls.append((buffer_size, seed))
                return self

            def take(self, count):
                return FakeStream(self.rows[: int(count)])

            def skip(self, count):
                return FakeStream(self.rows[int(count) :])

        class FakeStreamingDatasets:
            class Dataset:
                @staticmethod
                def from_list(rows):
                    return FakeDataset(rows)

            def __init__(self):
                self.calls = []
                self.train_stream = FakeStream(
                    [
                        {"text": "validation zero"},
                        {"text": "validation one"},
                        {"text": "train two"},
                        {"text": "train three"},
                        {"text": "train four"},
                        {"text": "train five"},
                    ]
                )

            def load_dataset(self, name, *args, **kwargs):
                self.calls.append((name, args, kwargs))
                if kwargs.get("split") == "validation":
                    raise ValueError("validation split unavailable")
                return self.train_stream

        fake_datasets = FakeStreamingDatasets()
        args = types.SimpleNamespace(
            train_file=[],
            validation_file=[],
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config=None,
            dataset_revision="main",
            dataset_streaming=True,
            train_split="train",
            eval_split="validation",
            max_train_samples=3,
            max_eval_samples=2,
            streaming_validation_samples=2,
            streaming_shuffle_buffer_size=16,
            seed=13,
        )

        raw_train, raw_eval, report = module._load_raw_datasets(fake_datasets, args)

        self.assertIsNone(report)
        self.assertEqual(len(raw_train), 3)
        self.assertEqual(len(raw_eval), 2)
        self.assertEqual(raw_eval[0]["text"], "validation zero")
        self.assertEqual(raw_train[0]["text"], "train two")
        self.assertEqual(fake_datasets.train_stream.shuffle_calls, [(16, 13)])
        self.assertEqual(fake_datasets.calls[0][2]["streaming"], True)
        self.assertEqual(fake_datasets.calls[0][2]["revision"], "main")

    def test_wait_launch_example_dry_run_records_ready_handoff(self) -> None:
        module = load_wait_launch_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "checkpoint-4096"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("ready", encoding="utf-8")
            status_card = tmp_path / "generation-control.json"
            status_card.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            manifest = tmp_path / "wait-launch.json"
            history = tmp_path / "wait-launch-history.jsonl"
            args = module.parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--status-card",
                    str(status_card),
                    "--manifest",
                    str(manifest),
                    "--jsonl-out",
                    str(history),
                    "--dry-run",
                    "--",
                    sys.executable,
                    "-c",
                    "print('next')",
                ]
            )
            payload = module.run_wait_launch(args)
            stored = json.loads(manifest.read_text())
            history_rows = [
                json.loads(line)
                for line in history.read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["returncode"], 0)
        self.assertTrue(stored["checkpoint_ready"])
        self.assertEqual(stored["status_card_status"], "ok")
        self.assertEqual(stored["launch_disk_guard"]["status"], "unchecked")
        self.assertEqual(stored["command"][:2], [sys.executable, "-c"])
        self.assertEqual(len(history_rows), 1)
        self.assertEqual(history_rows[0]["status"], "dry_run")
        self.assertEqual(history_rows[0]["launch_disk_guard"]["status"], "unchecked")

    def test_wait_launch_example_blocks_missing_checkpoint(self) -> None:
        module = load_wait_launch_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "wait-launch.json"
            args = module.parse_args(
                [
                    "--checkpoint",
                    str(tmp_path / "checkpoint-missing"),
                    "--manifest",
                    str(manifest),
                    "--checkpoint-timeout-seconds",
                    "0",
                    "--dry-run",
                    "--",
                    sys.executable,
                    "-c",
                    "print('next')",
                ]
            )
            payload = module.run_wait_launch(args)
            stored = json.loads(manifest.read_text())

        self.assertEqual(payload["status"], "blocked_missing_checkpoint")
        self.assertEqual(payload["returncode"], 2)
        self.assertFalse(stored["checkpoint_ready"])

    def test_wait_launch_example_blocks_prelaunch_when_disk_headroom_low(
        self,
    ) -> None:
        module = load_wait_launch_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "checkpoint-4096"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("ready", encoding="utf-8")
            output_dir = tmp_path / "next-run"
            manifest = tmp_path / "wait-launch.json"
            launched_pid = tmp_path / "next.pid"
            args = module.parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest),
                    "--launched-pid-file",
                    str(launched_pid),
                    "--",
                    sys.executable,
                    "-c",
                    "print('should-not-run')",
                    "--output-dir",
                    str(output_dir),
                    "--save-total-limit",
                    "1",
                    "--min-free-disk-gb",
                    "1000000000",
                ]
            )
            payload = module.run_wait_launch(args)
            stored = json.loads(manifest.read_text())

        self.assertEqual(payload["status"], "blocked_prelaunch_disk")
        self.assertEqual(payload["returncode"], 2)
        self.assertFalse(launched_pid.exists())
        self.assertEqual(stored["launch_disk_guard"]["status"], "blocked")
        self.assertFalse(stored["launch_disk_guard"]["meets_min_free"])
        self.assertEqual(stored["launch_disk_guard"]["save_total_limit"], 1)
        self.assertEqual(stored["launch_disk_guard"]["resume_checkpoint_bytes"], 5)

    def test_wait_launch_example_records_launched_pid_and_log(self) -> None:
        module = load_wait_launch_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "checkpoint-4096"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("ready", encoding="utf-8")
            manifest = tmp_path / "wait-launch.json"
            launched_pid = tmp_path / "next.pid"
            launched_log = tmp_path / "next.log"
            args = module.parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest),
                    "--launched-pid-file",
                    str(launched_pid),
                    "--launched-log-file",
                    str(launched_log),
                    "--launched-log-mode",
                    "write",
                    "--",
                    sys.executable,
                    "-c",
                    "print('next-run-ready')",
                ]
            )
            payload = module.run_wait_launch(args)
            stored = json.loads(manifest.read_text())
            launched_pid_value = int(launched_pid.read_text().strip())
            launched_log_text = launched_log.read_text()

        self.assertEqual(payload["status"], "finished")
        self.assertEqual(payload["returncode"], 0)
        self.assertGreater(launched_pid_value, 0)
        self.assertEqual(stored["launched_pid"], launched_pid_value)
        self.assertEqual(stored["launched_log_file"], str(launched_log))
        self.assertIn("next-run-ready", launched_log_text)

    def test_wait_launch_example_detaches_after_launch(self) -> None:
        module = load_wait_launch_example()
        launched_pid_value = None
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "checkpoint-4096"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("ready", encoding="utf-8")
            manifest = tmp_path / "wait-launch.json"
            history = tmp_path / "wait-launch-history.jsonl"
            launched_pid = tmp_path / "next.pid"
            args = module.parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest),
                    "--jsonl-out",
                    str(history),
                    "--launched-pid-file",
                    str(launched_pid),
                    "--detach",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(0.2)",
                ]
            )
            payload = module.run_wait_launch(args)
            stored = json.loads(manifest.read_text())
            history_rows = [
                json.loads(line)
                for line in history.read_text().splitlines()
                if line.strip()
            ]
            launched_pid_value = int(launched_pid.read_text().strip())

        try:
            os.waitpid(launched_pid_value, 0)
        except ChildProcessError:
            pass
        self.assertEqual(payload["status"], "launched")
        self.assertEqual(payload["returncode"], 0)
        self.assertTrue(payload["detach"])
        self.assertEqual(payload["launched_pid"], launched_pid_value)
        self.assertEqual(stored["status"], "launched")
        self.assertEqual(stored["launched_pid"], launched_pid_value)
        self.assertEqual(
            [row["status"] for row in history_rows],
            ["launching", "launched"],
        )

    def test_wait_launch_summary_example_summarizes_history(self) -> None:
        module = load_wait_launch_summary_example()
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "wait-launch-history.jsonl"
            rows = [
                {
                    "row_type": "hf_gpt2_finetune_wait_launch",
                    "status": "waiting_for_process",
                    "time_unix_s": 10.0,
                    "process_alive": True,
                    "checkpoint_ready": False,
                    "status_card_status": "waiting_for_process",
                    "launched_pid": None,
                    "returncode": None,
                },
                {
                    "row_type": "hf_gpt2_finetune_wait_launch",
                    "status": "launching",
                    "time_unix_s": 70.0,
                    "process_alive": False,
                    "checkpoint_ready": True,
                    "status_card_status": "ok",
                    "launched_pid": 123,
                    "launched_pid_file": "next.pid",
                    "launched_log_file": "next.log",
                    "launch_disk_guard": {
                        "status": "ok",
                        "min_free_gb": 4.0,
                        "free_gb": 12.0,
                        "estimated_peak_checkpoint_gb": 2.0,
                        "free_after_estimated_peak_gb": 10.0,
                    },
                    "returncode": None,
                },
                {
                    "row_type": "hf_gpt2_finetune_wait_launch",
                    "status": "finished",
                    "time_unix_s": 80.0,
                    "process_alive": False,
                    "checkpoint_ready": True,
                    "status_card_status": "ok",
                    "launched_pid": 123,
                    "launch_disk_guard": {
                        "status": "ok",
                        "min_free_gb": 4.0,
                        "free_gb": 12.0,
                        "estimated_peak_checkpoint_gb": 2.0,
                        "free_after_estimated_peak_gb": 10.0,
                    },
                    "returncode": 0,
                },
            ]
            history.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            loaded = module._load_history(history)
            summary = module.summarize_history(
                loaded,
                label="fineweb",
                history_jsonl=history,
            )
            lines = module.history_lines(summary, loaded, tail=2)
            ok = module.main([str(history), "--require-launched"])

        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["duration_seconds"], 70.0)
        self.assertEqual(summary["last_status"], "finished")
        self.assertTrue(summary["launched"])
        self.assertEqual(summary["last_launched_pid"], 123)
        self.assertEqual(summary["last_returncode"], 0)
        self.assertEqual(summary["last_launch_disk_status"], "ok")
        self.assertEqual(summary["last_launch_disk_min_free_gb"], 4.0)
        self.assertEqual(summary["last_launch_disk_peak_gb"], 2.0)
        self.assertEqual(summary["last_launch_disk_free_after_gb"], 10.0)
        self.assertEqual(ok, 0)
        self.assertTrue(any("launched=true" in line for line in lines))
        self.assertTrue(any("launch_disk_status=ok" in line for line in lines))
        self.assertTrue(any("launch_disk_free_after_gb=10" in line for line in lines))

    def test_wait_launch_summary_reconstructs_legacy_disk_guard(self) -> None:
        module = load_wait_launch_summary_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_checkpoint = tmp_path / "checkpoint-6144"
            source_checkpoint.mkdir()
            (source_checkpoint / "model.safetensors").write_text(
                "ready",
                encoding="utf-8",
            )
            checkpoint = tmp_path / "checkpoint-8192"
            output_dir = tmp_path / "next-run"
            history = tmp_path / "wait-launch-history.jsonl"
            row = {
                "row_type": "hf_gpt2_finetune_wait_launch",
                "status": "waiting_for_process",
                "time_unix_s": 10.0,
                "process_alive": True,
                "checkpoint": str(checkpoint),
                "checkpoint_ready": False,
                "command": [
                    sys.executable,
                    "bindings/st-py/examples/hf_gpt2_finetune_bridge.py",
                    "--resume-from-checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                    "--save-total-limit",
                    "1",
                    "--min-free-disk-gb",
                    "0",
                ],
            }
            history.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = module._load_history(history)
            summary = module.summarize_history(
                loaded,
                label="legacy",
                history_jsonl=history,
            )
            lines = module.history_lines(summary, loaded, tail=1)
            guard = module._launch_disk_guard(loaded[0])

        self.assertEqual(summary["last_launch_disk_status"], "reconstructed_ok")
        self.assertEqual(summary["last_launch_disk_min_free_gb"], 0.0)
        self.assertGreater(summary["last_launch_disk_free_after_gb"], 0.0)
        self.assertEqual(guard["resume_checkpoint_bytes"], 5)
        self.assertEqual(
            guard["resume_checkpoint_estimate_source"],
            str(source_checkpoint),
        )
        self.assertIn("launch_disk_status=reconstructed_ok", lines[0])
        self.assertIn("launch_disk_status=reconstructed_ok", lines[1])

    def test_wait_launch_summary_example_require_launched_fails_before_launch(self) -> None:
        module = load_wait_launch_summary_example()
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "wait-launch-history.jsonl"
            history.write_text(
                json.dumps(
                    {
                        "row_type": "hf_gpt2_finetune_wait_launch",
                        "status": "waiting_for_process",
                        "time_unix_s": 10.0,
                        "process_alive": True,
                        "checkpoint_ready": False,
                        "status_card_status": "waiting_for_process",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = module.main([str(history), "--require-launched"])

        self.assertEqual(result, 2)

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

    def test_generation_from_inference_distortion_applies_processor_kwargs(self) -> None:
        module = load_bridge_example()
        args = types.SimpleNamespace(
            generation_from_inference_distortion=True,
            generation_zspace_softmax=False,
            generation_zspace_keep_non_top_k=False,
            generation_zspace_no_native=False,
        )
        handoff = {
            "source_kind": "probe",
            "recommended_probe": "direct-probe",
            "recommended_processor_kwargs": {
                "top_k": 32,
                "curvature": -0.05,
                "temperature": 1.1,
                "entropy_target": 3.2,
                "entropy_tolerance": 1.0e-5,
                "entropy_gain": 0.7,
                "min_temperature": 0.6,
                "max_temperature": 2.2,
                "repression_window": 24,
                "repression_strength": 1.7,
                "last_token_repression": 0.4,
                "ngram_size": 3,
                "ngram_window": 80,
                "ngram_repression_strength": 0.9,
                "ngram_decay": 0.85,
                "mask_non_top_k": False,
                "use_native_zspace": False,
            },
        }

        report = module._apply_inference_distortion_generation_defaults(
            args,
            handoff,
        )

        self.assertEqual(report["status"], "ok")
        self.assertTrue(args.generation_zspace_softmax)
        self.assertEqual(args.generation_zspace_top_k, 32)
        self.assertEqual(args.generation_zspace_curvature, -0.05)
        self.assertEqual(args.generation_zspace_entropy_target, 3.2)
        self.assertEqual(args.generation_repression_strength, 1.7)
        self.assertEqual(args.generation_ngram_repression_strength, 0.9)
        self.assertTrue(args.generation_zspace_keep_non_top_k)
        self.assertTrue(args.generation_zspace_no_native)
        self.assertEqual(
            args._generation_from_inference_distortion_applied["recommended_probe"],
            "direct-probe",
        )

    def test_generation_from_inference_distortion_records_missing_processor(self) -> None:
        module = load_bridge_example()
        args = types.SimpleNamespace(
            generation_from_inference_distortion=True,
            generation_zspace_softmax=False,
        )

        report = module._apply_inference_distortion_generation_defaults(
            args,
            {
                "source_kind": "probe",
                "recommended_probe": "empty-probe",
            },
        )

        self.assertEqual(report["status"], "missing_processor_kwargs")
        self.assertFalse(args.generation_zspace_softmax)
        self.assertEqual(
            args._generation_from_inference_distortion_applied["status"],
            "missing_processor_kwargs",
        )
        self.assertEqual(
            args._generation_from_inference_distortion_applied["recommended_probe"],
            "empty-probe",
        )

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

    def test_trainer_training_telemetry_frame_derives_desire_and_psi(self) -> None:
        state = types.SimpleNamespace(global_step=4, epoch=0.4, max_steps=10)

        frame = hf_ft.hf_gpt2_finetune_training_telemetry_frame(
            "log",
            logs={"loss": 2.0, "grad_norm": 4.0, "learning_rate": 5e-5},
            state=state,
            previous_loss=2.5,
            telemetry_prefix="hf_ft",
            desire_gain=1.2,
            psi_gain=0.8,
            inference_distortion_handoff={
                "recommended_probe": "distort-002",
                "recommended_effect_score": 0.88,
                "recommended_risk_score": 0.21,
                "recommended_api_compatibility_score": 0.84,
                "desire_pressure": 0.8,
                "psi_total": 0.7,
                "coherence": 0.5,
                "include_penalties": True,
                "api_request_dropped_key_count": 2,
                "api_request_retry_dropped_key_count": 1,
                "recommended_runtime_adapter": {
                    "kind": "spiraltorch.zspace_inference_distortion_adapter",
                    "request": {"temperature": 1.05, "top_p": 0.82},
                    "context_partial": {
                        "origin": "zspace:inference_distortion",
                        "weight": 1.0,
                        "metrics": {"speed": 0.7},
                    },
                },
                "recommended_runtime_adapter_request": {
                    "temperature": 1.05,
                    "top_p": 0.82,
                },
                "recommended_processor_kwargs": {
                    "repression_strength": 1.7,
                    "ngram_repression_strength": 0.9,
                },
            },
        )

        self.assertEqual(frame["row_type"], "hf_gpt2_finetune_training_telemetry")
        self.assertEqual(frame["status"], "ok")
        self.assertEqual(frame["loss_key"], "loss")
        self.assertAlmostEqual(frame["loss_delta"], -0.5)
        self.assertAlmostEqual(frame["loss_improvement"], 0.5)
        self.assertAlmostEqual(frame["progress"], 0.4)
        self.assertIn("pressure", frame["desire"])
        self.assertIn("total", frame["psi"])
        telemetry = frame["telemetry"]
        self.assertIn("hf_ft.loss", telemetry)
        self.assertIn("hf_ft.desire.pressure", telemetry)
        self.assertIn("hf_ft.psi.total", telemetry)
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.desire_pressure"],
            0.8,
        )
        self.assertEqual(telemetry["hf_ft.inference_distortion.psi_total"], 0.7)
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.effect_score"],
            0.88,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.api_compatibility_score"],
            0.84,
        )
        self.assertEqual(
            telemetry[
                "hf_ft.inference_distortion.api_request_retry_dropped_key_count"
            ],
            1,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.api_request_dropped_key_count"],
            2,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.runtime_adapter_present"],
            1.0,
        )
        self.assertEqual(
            telemetry[
                "hf_ft.inference_distortion.runtime_adapter_request_temperature"
            ],
            1.05,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.runtime_adapter_request_top_p"],
            0.82,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.logits_repression_strength"],
            1.7,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.logits_ngram_repression_strength"],
            0.9,
        )
        self.assertEqual(
            telemetry["hf_ft.inference_distortion.include_penalties"],
            1.0,
        )
        self.assertEqual(
            frame["inference_distortion_handoff"]["recommended_probe"],
            "distort-002",
        )

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

        summary = hf_ft.summarize_hf_gpt2_finetune_trainer_trace(rows, max_steps=50)

        self.assertEqual(summary["trace_duration_s"], 45.0)
        self.assertEqual(summary["trace_log_interval_count"], 1)
        self.assertEqual(summary["trace_log_steps_per_second_min"], 0.5)
        self.assertEqual(summary["trace_log_steps_per_second_mean"], 0.5)
        self.assertEqual(summary["trace_log_steps_per_second_max"], 0.5)
        self.assertEqual(summary["trace_eval_loss_series"], "10=1.8,30=1.6")
        self.assertEqual(summary["trace_eval_loss_count"], 2)
        self.assertEqual(summary["trace_first_eval_loss"], 1.8)
        self.assertAlmostEqual(summary["trace_eval_loss_improvement"], 0.2)
        self.assertAlmostEqual(summary["trace_eval_loss_last_delta"], -0.2)
        self.assertEqual(summary["trace_max_steps"], 50)
        self.assertEqual(summary["trace_eval_loss_last_step_delta"], 20.0)
        self.assertAlmostEqual(summary["trace_eval_loss_last_improvement"], 0.2)
        self.assertAlmostEqual(
            summary["trace_eval_loss_last_improvement_per_step"],
            0.01,
        )
        self.assertAlmostEqual(
            summary["trace_eval_loss_mean_improvement_per_step"],
            0.01,
        )
        self.assertEqual(summary["trace_eval_loss_projection_step"], 50)
        self.assertEqual(
            summary["trace_eval_loss_projection_remaining_steps"],
            20.0,
        )
        self.assertAlmostEqual(
            summary["trace_eval_loss_projected_remaining_improvement"],
            0.2,
        )
        self.assertAlmostEqual(
            summary["trace_eval_loss_projected_final_loss"],
            1.4,
        )
        self.assertTrue(summary["trace_eval_loss_monotonic_nonincreasing"])
        self.assertEqual(summary["trace_best_eval_loss_step"], 30)
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

    def test_trace_summary_example_renders_live_status_lines(self) -> None:
        module = load_trace_summary_example()
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
                "global_step": 20,
                "time_unix_s": 130.0,
                "metrics": {"loss": 1.7, "learning_rate": 4e-5},
                "training_loss_guard": {"status": "ok"},
            },
            {
                "event": "evaluate",
                "global_step": 20,
                "time_unix_s": 135.0,
                "metrics": {"eval_loss": 1.6, "eval_runtime": 5.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            out_path = Path(tmp) / "summary.json"
            lines_path = Path(tmp) / "summary.txt"
            for row in rows:
                hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, trace_path)
            argv = [
                str(trace_path),
                "--label",
                "demo",
                "--max-steps",
                "40",
                "--tail-evals",
                "1",
                "--out",
                str(out_path),
                "--lines-out",
                str(lines_path),
                "--require-eval-loss-monotonic",
                "--min-eval-loss-improvement",
                "0.1",
            ]
            args = module.parse_args(argv)
            summary = module.summarize_trace(args)
            lines = module.summary_lines(summary, tail_evals=1)
            self.assertEqual(module.main(argv), 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            written_lines = lines_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["trace_max_global_step"], 20)
        self.assertEqual(summary["training_loss_guard_count"], 1)
        self.assertEqual(summary["progress"], 0.5)
        self.assertEqual(summary["trace_eval_loss_improvement"], 0.19999999999999996)
        self.assertAlmostEqual(
            summary["trace_eval_loss_last_improvement_per_step"],
            0.02,
        )
        self.assertAlmostEqual(summary["trace_eval_loss_projected_final_loss"], 1.2)
        self.assertTrue(summary["trace_eval_loss_monotonic_nonincreasing"])
        self.assertEqual(written["training_loss_guard_count"], 1)
        self.assertIn("label=demo", lines[0])
        self.assertIn("latest_step=20", lines[0])
        self.assertIn("guard_count=1", lines[0])
        self.assertIn("eval_loss_projected_final=1.2", lines[0])
        self.assertIn("eval_loss_monotonic=true", lines[0])
        self.assertIn("step=20 eval_loss=1.6", lines[1])
        self.assertEqual(
            module.validate_summary_gates(
                summary,
                require_eval_loss_monotonic=True,
                min_eval_loss_improvement=0.5,
            ),
            ["eval_loss_improvement_below_min:0.19999999999999996<0.5"],
        )
        self.assertEqual(written_lines, lines)

    def test_run_status_example_summarizes_live_ft_run_directory(self) -> None:
        module = load_run_status_example()
        rows = [
            {
                "event": "log",
                "global_step": 10,
                "time_unix_s": 100.0,
                "metrics": {"loss": 2.0},
            },
            {
                "event": "evaluate",
                "global_step": 10,
                "time_unix_s": 110.0,
                "metrics": {"eval_loss": 1.8, "eval_runtime": 4.0},
            },
            {
                "event": "log",
                "global_step": 20,
                "time_unix_s": 130.0,
                "metrics": {"loss": 1.7},
            },
            {
                "event": "evaluate",
                "global_step": 20,
                "time_unix_s": 135.0,
                "metrics": {"eval_loss": 1.6, "eval_runtime": 5.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            trace_path = run_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            for row in rows:
                hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, trace_path)
            (run_dir / "spiraltorch-hf-gpt2-ft-run-card.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )
            (run_dir / "ft.log").write_text(
                " 50%|#####     | 20/40 [00:20<00:20,  1.00s/it]\n"
                " 60%|######    | 24/40 [00:24<00:16,  1.00s/it]\n"
                "  5%|5         | 51/964 [00:07<01:56,  7.83it/s]\n",
                encoding="utf-8",
            )
            pid_file = run_dir / "ft.pid"
            pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
            checkpoint = run_dir / "checkpoint-20"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("ready", encoding="utf-8")
            checkpoint_card = run_dir / "checkpoint-20-generation-control.json"
            checkpoint_card.write_text(
                json.dumps(
                    {
                        "status": "waiting_for_process",
                        "process_wait": {
                            "status": "waiting",
                            "pid": os.getpid(),
                            "waited_seconds": 12.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            out_path = run_dir / "status.json"
            lines_path = run_dir / "status.txt"
            jsonl_path = run_dir / "status.jsonl"
            argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--save-total-limit",
                "1",
                "--min-free-disk-gb",
                "1.0",
                "--final-checkpoint",
                "checkpoint-20",
                "--checkpoint-card",
                str(checkpoint_card),
                "--tail-evals",
                "1",
                "--out",
                str(out_path),
                "--lines-out",
                str(lines_path),
                "--jsonl-out",
                str(jsonl_path),
            ]
            args = module.parse_args(argv)
            status = module.summarize_run(args)
            lines = module.status_lines(status, tail_evals=1)
            self.assertEqual(module.main(argv), 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            written_lines = lines_path.read_text(encoding="utf-8").splitlines()
            written_jsonl = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            watch_path = run_dir / "watch-status.json"
            watch_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--watch-interval-seconds",
                "0.01",
                "--watch-count",
                "1",
                "--out",
                str(watch_path),
            ]
            self.assertEqual(module.main(watch_argv), 0)
            watch_written = json.loads(watch_path.read_text(encoding="utf-8"))
            quiet_path = run_dir / "quiet-status.json"
            quiet_lines_path = run_dir / "quiet-status.txt"
            quiet_jsonl_path = run_dir / "quiet-status.jsonl"
            quiet_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--checkpoint-card",
                str(checkpoint_card),
                "--tail-evals",
                "1",
                "--out",
                str(quiet_path),
                "--lines-out",
                str(quiet_lines_path),
                "--jsonl-out",
                str(quiet_jsonl_path),
                "--quiet",
            ]
            quiet_stdout = io.StringIO()
            with redirect_stdout(quiet_stdout):
                self.assertEqual(module.main(quiet_argv), 0)
            quiet_written = json.loads(quiet_path.read_text(encoding="utf-8"))
            quiet_written_lines = quiet_lines_path.read_text(
                encoding="utf-8"
            ).splitlines()
            quiet_written_jsonl = [
                json.loads(line)
                for line in quiet_jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            quiet_watch_path = run_dir / "quiet-watch-status.json"
            quiet_watch_jsonl_path = run_dir / "quiet-watch-status.jsonl"
            quiet_watch_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--watch-interval-seconds",
                "0.01",
                "--watch-count",
                "2",
                "--out",
                str(quiet_watch_path),
                "--jsonl-out",
                str(quiet_watch_jsonl_path),
                "--quiet",
            ]
            quiet_watch_stdout = io.StringIO()
            with redirect_stdout(quiet_watch_stdout):
                self.assertEqual(module.main(quiet_watch_argv), 0)
            quiet_watch_written = json.loads(
                quiet_watch_path.read_text(encoding="utf-8")
            )
            quiet_watch_jsonl = [
                json.loads(line)
                for line in quiet_watch_jsonl_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            watch_final_path = run_dir / "watch-final-status.json"
            watch_final_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--final-checkpoint",
                "checkpoint-20",
                "--watch-interval-seconds",
                "0.01",
                "--watch-count",
                "2",
                "--watch-stop-on-final",
                "--out",
                str(watch_final_path),
            ]
            watch_final_args = module.parse_args(watch_final_argv)
            self.assertTrue(
                module._should_stop_watch(
                    watch_final_args, module.summarize_run(watch_final_args)
                )
            )
            self.assertEqual(module.main(watch_final_argv), 0)
            watch_final_written = json.loads(
                watch_final_path.read_text(encoding="utf-8")
            )
            watch_eval_path = run_dir / "watch-eval-status.json"
            watch_eval_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--watch-interval-seconds",
                "0.01",
                "--watch-count",
                "2",
                "--watch-stop-on-eval-step",
                "10",
                "--out",
                str(watch_eval_path),
            ]
            watch_eval_args = module.parse_args(watch_eval_argv)
            self.assertTrue(
                module._should_stop_watch(
                    watch_eval_args, module.summarize_run(watch_eval_args)
                )
            )
            self.assertFalse(
                module._should_stop_watch(
                    watch_eval_args,
                    {
                        "trace": {
                            "trace_max_global_step": 10,
                            "trace_eval_loss_points": [{"step": 0, "eval_loss": 2.0}],
                        }
                    },
                )
            )
            self.assertEqual(module.main(watch_eval_argv), 0)
            watch_eval_written = json.loads(
                watch_eval_path.read_text(encoding="utf-8")
            )
            watch_checkpoint_path = run_dir / "watch-checkpoint-status.json"
            watch_checkpoint_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--watch-interval-seconds",
                "0.01",
                "--watch-count",
                "2",
                "--watch-stop-on-checkpoint",
                "20",
                "--out",
                str(watch_checkpoint_path),
            ]
            watch_checkpoint_args = module.parse_args(watch_checkpoint_argv)
            self.assertTrue(
                module._should_stop_watch(
                    watch_checkpoint_args, module.summarize_run(watch_checkpoint_args)
                )
            )
            self.assertEqual(module.main(watch_checkpoint_argv), 0)
            watch_checkpoint_written = json.loads(
                watch_checkpoint_path.read_text(encoding="utf-8")
            )
            watch_disk_low_path = run_dir / "watch-disk-low-status.json"
            watch_disk_low_argv = [
                str(run_dir),
                "--max-steps",
                "40",
                "--eval-steps",
                "10",
                "--save-steps",
                "20",
                "--min-free-disk-gb",
                "1000000000",
                "--watch-interval-seconds",
                "0.01",
                "--watch-count",
                "2",
                "--watch-stop-on-disk-low",
                "--out",
                str(watch_disk_low_path),
            ]
            watch_disk_low_args = module.parse_args(watch_disk_low_argv)
            watch_disk_low_status = module.summarize_run(watch_disk_low_args)
            self.assertEqual(
                module._watch_stop_reason(
                    watch_disk_low_args, watch_disk_low_status
                ),
                "disk_low",
            )
            self.assertEqual(module.main(watch_disk_low_argv), 0)
            watch_disk_low_written = json.loads(
                watch_disk_low_path.read_text(encoding="utf-8")
            )
            watch_guard_args = module.parse_args(
                [
                    str(run_dir),
                    "--watch-interval-seconds",
                    "0.01",
                    "--watch-count",
                    "1",
                    "--watch-stop-on-training-guard",
                ]
            )
            self.assertEqual(
                module._watch_stop_reason(
                    watch_guard_args,
                    {"trace": {"training_loss_guard_count": 1}},
                ),
                "training_loss_guard",
            )

        self.assertEqual(status["process_status"], "alive")
        self.assertIsInstance(status["time_unix_s"], float)
        self.assertEqual(status["trace"]["trace_max_global_step"], 20)
        self.assertEqual(status["trace"]["progress"], 0.5)
        self.assertEqual(status["trace"]["trace_last_eval_loss_step"], 20)
        self.assertAlmostEqual(
            status["trace"]["trace_eval_loss_last_improvement_per_step"],
            0.02,
        )
        self.assertAlmostEqual(
            status["trace"]["trace_eval_loss_projected_final_loss"],
            1.2,
        )
        self.assertEqual(status["log_progress"]["log_latest_step"], 24)
        self.assertEqual(status["log_progress"]["log_progress"], 0.6)
        self.assertEqual(status["log_progress"]["log_elapsed_seconds"], 24.0)
        self.assertEqual(status["log_progress"]["log_remaining_seconds"], 16.0)
        self.assertEqual(status["eval_progress"]["next_eval_step"], 30)
        self.assertEqual(status["eval_progress"]["log_steps_until_next_eval"], 6)
        self.assertEqual(status["eval_progress"]["latest_due_eval_step"], 20)
        self.assertTrue(status["eval_progress"]["latest_due_eval_ready"])
        self.assertIsNone(status["eval_progress"]["pending_eval_step"])
        self.assertIsNone(status["eval_progress"]["log_steps_since_pending_eval"])
        self.assertEqual(status["trace"]["trace_best_eval_loss_step"], 20)
        self.assertEqual(status["checkpoint_progress"]["next_checkpoint_step"], 40)
        self.assertEqual(
            status["checkpoint_progress"]["log_steps_until_next_checkpoint"], 16
        )
        self.assertEqual(status["min_free_disk_gb"], 1.0)
        self.assertAlmostEqual(
            status["disk_margin_gb"],
            status["disk_free_gb"] - 1.0,
        )
        self.assertEqual(status["disk_status"], "ok")
        self.assertEqual(status["checkpoint_count"], 1)
        self.assertEqual(status["save_total_limit"], 1)
        self.assertEqual(status["checkpoint_headroom"]["resume_checkpoint_bytes"], 5)
        self.assertEqual(
            status["checkpoint_headroom"]["estimated_peak_checkpoint_bytes"], 10
        )
        self.assertTrue(status["final_checkpoint_ready"])
        self.assertEqual(status["checkpoint_card_status"], "waiting_for_process")
        self.assertIn("process=alive", lines[0])
        self.assertIn("latest_step=20", lines[0])
        self.assertIn("log_latest_step=24", lines[0])
        self.assertIn("log_remaining_seconds=16", lines[0])
        self.assertIn("last_eval_step=20", lines[0])
        self.assertIn("eval_loss_projected_final=1.2", lines[0])
        self.assertIn("next_eval_step=30", lines[0])
        self.assertIn("log_steps_until_next_eval=6", lines[0])
        self.assertIn("latest_due_eval_step=20", lines[0])
        self.assertIn("latest_due_eval_ready=true", lines[0])
        self.assertIn("pending_eval_step=none", lines[0])
        self.assertIn("next_checkpoint_step=40", lines[0])
        self.assertIn("log_steps_until_next_checkpoint=16", lines[0])
        self.assertIn("best_eval_loss_step=20", lines[0])
        self.assertIn("min_free_disk_gb=1", lines[0])
        self.assertIn("disk_status=ok", lines[0])
        self.assertIn("checkpoint_card=waiting_for_process", lines[0])
        self.assertIn("save_total_limit=1", lines[0])
        self.assertIn("checkpoint_headroom_peak_gb=", lines[0])
        self.assertIn("checkpoint_headroom_free_after_gb=", lines[0])
        self.assertIn("final_ready=true", lines[0])
        self.assertIn("hf_gpt2_ft_run_wait status=waiting", lines[-1])
        self.assertEqual(written["process_status"], "alive")
        self.assertEqual(written["checkpoint_headroom"]["resume_checkpoint_bytes"], 5)
        self.assertEqual(len(written_jsonl), 1)
        self.assertEqual(written_jsonl[0]["process_status"], "alive")
        self.assertEqual(watch_written["process_status"], "alive")
        self.assertEqual(watch_written["log_progress"]["log_latest_step"], 24)
        self.assertEqual(watch_written["eval_progress"]["next_eval_step"], 30)
        self.assertIsNone(watch_written["eval_progress"]["pending_eval_step"])
        self.assertEqual(
            watch_written["checkpoint_progress"]["next_checkpoint_step"], 40
        )
        self.assertEqual(quiet_stdout.getvalue(), "")
        self.assertEqual(quiet_written["process_status"], "alive")
        self.assertEqual(
            quiet_written_lines,
            module.status_lines(quiet_written, tail_evals=1),
        )
        self.assertEqual(len(quiet_written_jsonl), 1)
        self.assertEqual(quiet_written_jsonl[0]["process_status"], "alive")
        self.assertEqual(quiet_watch_stdout.getvalue(), "")
        self.assertEqual(quiet_watch_written["process_status"], "alive")
        self.assertEqual(len(quiet_watch_jsonl), 2)
        self.assertTrue(watch_final_written["final_checkpoint_ready"])
        self.assertEqual(
            watch_eval_written["trace"]["trace_eval_loss_points"][-1]["step"], 20
        )
        self.assertEqual(watch_eval_written["trace"]["trace_last_eval_loss_step"], 20)
        self.assertEqual(watch_eval_written["watch_stop_eval_step"], 10)
        self.assertTrue(watch_eval_written["watch_stop_eval_ready"])
        self.assertEqual(
            watch_checkpoint_written["latest_checkpoint"]["name"], "checkpoint-20"
        )
        self.assertEqual(watch_disk_low_written["watch_stop_reason"], "disk_low")
        self.assertEqual(watch_disk_low_written["disk_status"], "low")
        self.assertEqual(written_lines, module.status_lines(written, tail_evals=1))

    def test_run_status_example_uses_live_process_training_flags(self) -> None:
        module = load_run_status_example()
        rows = [
            {
                "event": "log",
                "global_step": 20,
                "time_unix_s": 100.0,
                "metrics": {"loss": 2.0},
            },
            {
                "event": "evaluate",
                "global_step": 20,
                "time_unix_s": 110.0,
                "metrics": {"eval_loss": 1.8},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            trace_path = run_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            for row in rows:
                hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, trace_path)
            (run_dir / "ft.log").write_text(
                " 30%|###       | 24/80 [00:24<00:56,  1.00s/it]\n",
                encoding="utf-8",
            )
            pid_file = run_dir / "ft.pid"
            pid_file.write_text("12345\n", encoding="utf-8")
            checkpoint = run_dir / "checkpoint-20"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text("ready", encoding="utf-8")
            args = module.parse_args([str(run_dir)])
            with mock.patch.object(
                module,
                "_process_command_args",
                return_value=[
                    sys.executable,
                    "bindings/st-py/examples/hf_gpt2_finetune_bridge.py",
                    "--max-steps",
                    "80",
                    "--eval-steps",
                    "20",
                    "--save-steps",
                    "40",
                    "--save-total-limit",
                    "2",
                    "--min-free-disk-gb",
                    "0",
                ],
            ):
                status = module.summarize_run(args)
                lines = module.status_lines(status, tail_evals=0)

        self.assertEqual(status["trace"]["max_steps"], 80)
        self.assertEqual(status["log_progress"]["log_latest_step"], 24)
        self.assertEqual(status["log_progress"]["log_max_steps"], 80)
        self.assertEqual(status["eval_progress"]["eval_steps"], 20)
        self.assertEqual(status["eval_progress"]["next_eval_step"], 40)
        self.assertEqual(status["checkpoint_progress"]["checkpoint_steps"], 40)
        self.assertEqual(status["checkpoint_progress"]["next_checkpoint_step"], 40)
        self.assertEqual(status["save_total_limit"], 2)
        self.assertEqual(status["min_free_disk_gb"], 0.0)
        self.assertEqual(status["disk_status"], "ok")
        self.assertEqual(
            status["checkpoint_headroom"]["estimated_peak_checkpoint_bytes"], 15
        )
        self.assertTrue(status["runtime_settings"]["process_command_available"])
        self.assertIn("runtime_eval_steps=20", lines[0])
        self.assertIn("runtime_save_steps=40", lines[0])
        self.assertIn("save_total_limit=2", lines[0])

    def test_run_status_example_marks_due_eval_pending_during_eval_progress(self) -> None:
        module = load_run_status_example()
        rows = [
            {
                "event": "evaluate",
                "global_step": 5632,
                "max_steps": 8192,
                "time_unix_s": 100.0,
                "metrics": {"eval_loss": 3.28},
            },
            {
                "event": "log",
                "global_step": 6144,
                "max_steps": 8192,
                "time_unix_s": 120.0,
                "metrics": {"loss": 3.36},
            },
            {
                "event": "evaluate",
                "global_step": 6144,
                "max_steps": 8192,
                "time_unix_s": 140.0,
                "metrics": {"eval_loss": 3.26},
            },
            {
                "event": "log",
                "global_step": 6656,
                "max_steps": 8192,
                "time_unix_s": 180.0,
                "metrics": {"loss": 3.35},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            trace_path = run_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            for row in rows:
                hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, trace_path)
            (run_dir / "ft.log").write_text(
                " 78%|#######8  | 1600/2048 [05:05<01:04,  6.87it/s]\n",
                encoding="utf-8",
            )
            args = module.parse_args([str(run_dir)])
            status = module.summarize_run(args)
            lines = module.status_lines(status, tail_evals=1)

        self.assertEqual(status["trace"]["max_steps"], 8192)
        self.assertEqual(status["trace"]["trace_max_global_step"], 6656)
        self.assertEqual(status["log_progress"]["log_status"], "fallback_trace")
        self.assertEqual(status["log_progress"]["log_latest_step"], 6656)
        self.assertEqual(status["log_progress"]["log_max_steps"], 8192)
        self.assertEqual(status["eval_progress"]["eval_steps"], 512)
        self.assertEqual(status["eval_progress"]["latest_due_eval_step"], 6656)
        self.assertFalse(status["eval_progress"]["latest_due_eval_ready"])
        self.assertEqual(status["eval_progress"]["pending_eval_step"], 6656)
        self.assertEqual(status["eval_progress"]["log_steps_since_pending_eval"], 0)
        self.assertEqual(status["eval_progress"]["next_eval_step"], 7168)
        self.assertIn("log_latest_step=6656", lines[0])
        self.assertIn("latest_due_eval_step=6656", lines[0])
        self.assertIn("latest_due_eval_ready=false", lines[0])
        self.assertIn("pending_eval_step=6656", lines[0])

    def test_run_status_example_maps_resume_baseline_eval_to_resume_step(
        self,
    ) -> None:
        module = load_run_status_example()
        rows = [
            {
                "event": "log",
                "global_step": 0,
                "max_steps": 0,
                "time_unix_s": 100.0,
                "metrics": {"eval_loss": 3.25},
            },
            {
                "event": "evaluate",
                "global_step": 0,
                "max_steps": 0,
                "time_unix_s": 100.1,
                "metrics": {"eval_loss": 3.25},
            },
            {
                "event": "train_begin",
                "global_step": 8192,
                "max_steps": 16384,
                "time_unix_s": 101.0,
                "metrics": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            trace_path = run_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
            for row in rows:
                hf_ft.write_hf_gpt2_finetune_trainer_trace_event(row, trace_path)
            (run_dir / "ft.log").write_text(
                " 50%|#####     | 8253/16384 [00:01<07:31, 18.0it/s]\n",
                encoding="utf-8",
            )
            args = module.parse_args([str(run_dir), "--eval-steps", "512"])
            status = module.summarize_run(args)
            lines = module.status_lines(status, tail_evals=1)

        self.assertEqual(status["trace"]["trace_max_global_step"], 8192)
        self.assertEqual(status["trace"]["trace_last_eval_loss_step"], 0)
        self.assertEqual(
            status["trace"]["trace_effective_last_eval_loss_step"], 8192
        )
        self.assertEqual(status["trace"]["trace_resume_baseline_eval_step"], 8192)
        self.assertEqual(status["eval_progress"]["latest_due_eval_step"], 8192)
        self.assertTrue(status["eval_progress"]["latest_due_eval_ready"])
        self.assertIsNone(status["eval_progress"]["pending_eval_step"])
        self.assertIsNone(status["eval_progress"]["log_steps_since_pending_eval"])
        self.assertEqual(status["eval_progress"]["next_eval_step"], 8704)
        self.assertEqual(status["eval_progress"]["log_steps_until_next_eval"], 451)
        self.assertIn("last_eval_step=8192", lines[0])
        self.assertIn("latest_due_eval_step=8192", lines[0])
        self.assertIn("latest_due_eval_ready=true", lines[0])
        self.assertIn("pending_eval_step=none", lines[0])
        self.assertIn("hf_gpt2_ft_run_eval step=8192 raw_step=0", lines[-1])

    def test_status_history_summary_example_summarizes_jsonl_progress(self) -> None:
        module = load_status_history_summary_example()
        rows = [
            {
                "time_unix_s": 100.0,
                "process_status": "alive",
                "final_checkpoint_ready": False,
                "checkpoint_count": 1,
                "latest_checkpoint": {"name": "checkpoint-20"},
                "save_total_limit": 1,
                "runtime_settings": {
                    "max_steps": 40,
                    "eval_steps": 10,
                    "save_steps": 20,
                    "save_total_limit": 1,
                    "min_free_disk_gb": 4.0,
                    "process_command_available": True,
                },
                "checkpoint_headroom": {
                    "resume_checkpoint_gb": 0.5,
                    "estimated_peak_checkpoint_gb": 1.0,
                    "free_after_estimated_peak_gb": 11.0,
                },
                "disk_free_gb": 12.0,
                "disk_margin_gb": 8.0,
                "disk_status": "ok",
                "trace": {
                    "trace_max_global_step": 20,
                    "trace_last_loss": 2.1,
                    "trace_last_eval_loss": 1.8,
                    "trace_last_eval_loss_step": 20,
                    "trace_eval_loss_points": [
                        {"step": 10, "eval_loss": 1.9},
                        {"step": 20, "eval_loss": 1.8},
                    ],
                    "trace_best_eval_loss_step": 20,
                    "training_loss_guard_count": 0,
                },
                "log_progress": {
                    "log_latest_step": 24,
                    "log_max_steps": 40,
                    "log_remaining_seconds": 160.0,
                },
                "eval_progress": {
                    "next_eval_step": 30,
                    "log_steps_until_next_eval": 6,
                    "latest_due_eval_step": 20,
                    "latest_due_eval_ready": True,
                    "pending_eval_step": None,
                    "log_steps_since_pending_eval": None,
                },
                "checkpoint_progress": {
                    "next_checkpoint_step": 40,
                    "log_steps_until_next_checkpoint": 16,
                },
                "watch_stop_eval_step": 30,
                "watch_stop_eval_ready": False,
            },
            {
                "time_unix_s": 140.0,
                "process_status": "alive",
                "final_checkpoint_ready": True,
                "watch_stop_reason": "checkpoint_ready",
                "checkpoint_count": 1,
                "latest_checkpoint": {"name": "checkpoint-20"},
                "save_total_limit": 1,
                "runtime_settings": {
                    "max_steps": 40,
                    "eval_steps": 10,
                    "save_steps": 20,
                    "save_total_limit": 1,
                    "min_free_disk_gb": 4.0,
                    "process_command_available": True,
                },
                "checkpoint_headroom": {
                    "resume_checkpoint_gb": 0.5,
                    "estimated_peak_checkpoint_gb": 1.0,
                    "free_after_estimated_peak_gb": 10.5,
                },
                "disk_free_gb": 11.5,
                "disk_margin_gb": 7.5,
                "disk_status": "ok",
                "trace": {
                    "trace_max_global_step": 30,
                    "trace_last_loss": 1.9,
                    "trace_last_eval_loss": 1.7,
                    "trace_eval_loss_points": [
                        {"step": 20, "eval_loss": 1.8},
                        {"step": 30, "eval_loss": 1.7},
                    ],
                    "trace_best_eval_loss_step": 30,
                    "training_loss_guard_count": 0,
                },
                "log_progress": {
                    "log_latest_step": 34,
                    "log_max_steps": 40,
                    "log_remaining_seconds": 90.0,
                },
                "eval_progress": {
                    "next_eval_step": 40,
                    "log_steps_until_next_eval": 6,
                    "latest_due_eval_step": 30,
                    "latest_due_eval_ready": True,
                    "pending_eval_step": None,
                    "log_steps_since_pending_eval": None,
                },
                "checkpoint_progress": {
                    "next_checkpoint_step": 40,
                    "log_steps_until_next_checkpoint": 6,
                },
                "watch_stop_eval_step": 40,
                "watch_stop_eval_ready": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            out_path = Path(tmp) / "history-summary.json"
            lines_path = Path(tmp) / "history-summary.txt"
            history_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            argv = [
                str(history_path),
                "--label",
                "demo",
                "--tail",
                "1",
                "--tail-evals",
                "2",
                "--out",
                str(out_path),
                "--lines-out",
                str(lines_path),
            ]
            args = module.parse_args(argv)
            loaded_rows = module._load_history(args.history_jsonl)
            package_rows = st.load_hf_gpt2_finetune_status_history(history_path)
            summary = module.summarize_history(
                loaded_rows, label=args.label, history_jsonl=args.history_jsonl
            )
            package_summary = st.summarize_hf_gpt2_finetune_status_history(
                package_rows, label=args.label, history_jsonl=args.history_jsonl
            )
            lines = module.history_lines(
                summary,
                loaded_rows,
                tail=args.tail,
                tail_evals=args.tail_evals,
            )
            package_lines = st.hf_gpt2_finetune_status_history_lines(
                package_summary,
                package_rows,
                tail=args.tail,
                tail_evals=args.tail_evals,
            )
            self.assertEqual(module.main(argv), 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            written_lines = lines_path.read_text(encoding="utf-8").splitlines()

        self.assertIs(
            st.hf_ft_status.load_hf_gpt2_finetune_status_history,
            st.load_hf_gpt2_finetune_status_history,
        )
        self.assertEqual(package_rows, loaded_rows)
        self.assertEqual(package_summary, summary)
        self.assertEqual(package_lines, lines)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["duration_seconds"], 40.0)
        self.assertEqual(summary["log_duration_seconds"], 40.0)
        self.assertEqual(summary["delta_log_step"], 10)
        self.assertEqual(summary["log_steps_per_second"], 0.25)
        self.assertEqual(summary["delta_trace_step"], 10)
        self.assertEqual(summary["last_runtime_max_steps"], 40)
        self.assertEqual(summary["last_runtime_eval_steps"], 10)
        self.assertEqual(summary["last_runtime_save_steps"], 20)
        self.assertEqual(summary["last_runtime_save_total_limit"], 1)
        self.assertEqual(summary["last_runtime_min_free_disk_gb"], 4.0)
        self.assertTrue(summary["last_runtime_process_command_available"])
        self.assertEqual(summary["last_log_max_steps"], 40)
        self.assertEqual(summary["last_log_remaining_seconds"], 90.0)
        self.assertEqual(summary["last_log_steps_until_final"], 6)
        self.assertEqual(summary["estimated_seconds_until_final"], 24.0)
        self.assertEqual(summary["last_next_eval_step"], 40)
        self.assertEqual(summary["last_log_steps_until_next_eval"], 6)
        self.assertEqual(summary["estimated_seconds_until_next_eval"], 24.0)
        self.assertEqual(summary["last_latest_due_eval_step"], 30)
        self.assertTrue(summary["last_latest_due_eval_ready"])
        self.assertIsNone(summary["last_pending_eval_step"])
        self.assertIsNone(summary["last_log_steps_since_pending_eval"])
        self.assertEqual(summary["last_next_checkpoint_step"], 40)
        self.assertEqual(summary["last_log_steps_until_next_checkpoint"], 6)
        self.assertEqual(summary["estimated_seconds_until_next_checkpoint"], 24.0)
        self.assertEqual(summary["last_best_eval_loss_step"], 30)
        self.assertEqual(summary["last_eval_loss_step"], 30)
        self.assertEqual(summary["first_loss"], 2.1)
        self.assertEqual(summary["last_loss"], 1.9)
        self.assertEqual(summary["min_loss"], 1.9)
        self.assertAlmostEqual(summary["loss_delta"], -0.2)
        self.assertEqual(summary["min_eval_loss"], 1.7)
        self.assertEqual(summary["last_save_total_limit"], 1)
        self.assertEqual(summary["last_checkpoint_headroom_checkpoint_gb"], 0.5)
        self.assertEqual(summary["last_checkpoint_headroom_peak_gb"], 1.0)
        self.assertEqual(summary["last_checkpoint_headroom_free_after_gb"], 10.5)
        self.assertEqual(summary["min_disk_free_gb"], 11.5)
        self.assertEqual(summary["last_disk_margin_gb"], 7.5)
        self.assertEqual(summary["min_disk_margin_gb"], 7.5)
        self.assertEqual(summary["last_disk_status"], "ok")
        self.assertEqual(summary["last_watch_stop_eval_step"], 40)
        self.assertFalse(summary["last_watch_stop_eval_ready"])
        self.assertEqual(summary["last_watch_stop_reason"], "checkpoint_ready")
        self.assertEqual(written["delta_log_step"], 10)
        self.assertIn("label=demo", lines[0])
        self.assertIn("last_log_step=34", lines[0])
        self.assertIn("log_duration_seconds=40", lines[0])
        self.assertIn("log_steps_per_second=0.25", lines[0])
        self.assertIn("runtime_max_steps=40", lines[0])
        self.assertIn("runtime_eval_steps=10", lines[0])
        self.assertIn("runtime_save_steps=20", lines[0])
        self.assertIn("runtime_save_total_limit=1", lines[0])
        self.assertIn("runtime_min_free_disk_gb=4", lines[0])
        self.assertIn("runtime_process_command=true", lines[0])
        self.assertIn("last_log_remaining_seconds=90", lines[0])
        self.assertIn("last_steps_until_final=6", lines[0])
        self.assertIn("estimated_seconds_until_final=24", lines[0])
        self.assertIn("estimated_seconds_until_next_eval=24", lines[0])
        self.assertIn("last_latest_due_eval_step=30", lines[0])
        self.assertIn("last_latest_due_eval_ready=true", lines[0])
        self.assertIn("last_pending_eval_step=none", lines[0])
        self.assertIn("last_next_checkpoint_step=40", lines[0])
        self.assertIn("last_steps_until_next_checkpoint=6", lines[0])
        self.assertIn("estimated_seconds_until_next_checkpoint=24", lines[0])
        self.assertIn("last_eval_step=30", lines[0])
        self.assertIn("best_eval_loss_step=30", lines[0])
        self.assertIn("last_loss=1.9", lines[0])
        self.assertIn("min_loss=1.9", lines[0])
        self.assertIn("loss_delta=-0.2", lines[0])
        self.assertIn("min_eval_loss=1.7", lines[0])
        self.assertIn("last_save_total_limit=1", lines[0])
        self.assertIn("last_checkpoint_headroom_peak_gb=1", lines[0])
        self.assertIn("last_checkpoint_headroom_free_after_gb=10.5", lines[0])
        self.assertIn("last_disk_free_gb=11.5", lines[0])
        self.assertIn("min_disk_free_gb=11.5", lines[0])
        self.assertIn("last_disk_margin_gb=7.5", lines[0])
        self.assertIn("min_disk_margin_gb=7.5", lines[0])
        self.assertIn("disk_status=ok", lines[0])
        self.assertIn("watch_stop_eval_step=40", lines[0])
        self.assertIn("watch_stop_eval_ready=false", lines[0])
        self.assertIn("watch_stop_reason=checkpoint_ready", lines[0])
        self.assertIn("index=1", lines[1])
        self.assertIn("runtime_eval_steps=10", lines[1])
        self.assertIn("runtime_save_steps=20", lines[1])
        self.assertIn("runtime_process_command=true", lines[1])
        self.assertIn("log_remaining_seconds=90", lines[1])
        self.assertIn("next_checkpoint_step=40", lines[1])
        self.assertIn("steps_until_next_checkpoint=6", lines[1])
        self.assertIn("pending_eval_step=none", lines[1])
        self.assertIn("last_eval_step=30", lines[1])
        self.assertIn("best_eval_loss_step=30", lines[1])
        self.assertIn("last_loss=1.9", lines[1])
        self.assertIn("save_total_limit=1", lines[1])
        self.assertIn("checkpoint_headroom_peak_gb=1", lines[1])
        self.assertIn("checkpoint_headroom_free_after_gb=10.5", lines[1])
        self.assertIn("disk_free_gb=11.5", lines[1])
        self.assertIn("disk_margin_gb=7.5", lines[1])
        self.assertIn("disk_status=ok", lines[1])
        self.assertIn("watch_stop_eval_step=40", lines[1])
        self.assertIn("watch_stop_eval_ready=false", lines[1])
        self.assertIn("watch_stop_reason=checkpoint_ready", lines[1])
        self.assertIn("hf_gpt2_ft_status_history_eval", lines[2])
        self.assertIn("index=0", lines[2])
        self.assertIn("step=20", lines[2])
        self.assertIn("eval_loss=1.8", lines[2])
        self.assertIn("hf_gpt2_ft_status_history_eval", lines[3])
        self.assertIn("index=1", lines[3])
        self.assertIn("step=30", lines[3])
        self.assertIn("eval_loss=1.7", lines[3])
        self.assertEqual(written_lines, lines)

    def test_status_history_summary_uses_first_available_log_step_for_rate(
        self,
    ) -> None:
        module = load_status_history_summary_example()
        rows = [
            {"time_unix_s": 100.0, "process_status": "alive"},
            {
                "time_unix_s": 110.0,
                "process_status": "alive",
                "log_progress": {"log_latest_step": 100, "log_max_steps": 200},
            },
            {
                "time_unix_s": 130.0,
                "process_status": "alive",
                "log_progress": {"log_latest_step": 150, "log_max_steps": 200},
                "eval_progress": {"log_steps_until_next_eval": 10},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.jsonl"
            history_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = module.summarize_history(
                module._load_history(history_path),
                label="resume",
                history_jsonl=history_path,
            )
            lines = module.history_lines(summary, rows, tail=0)

        self.assertEqual(summary["duration_seconds"], 30.0)
        self.assertEqual(summary["log_duration_seconds"], 20.0)
        self.assertEqual(summary["first_log_step"], 100)
        self.assertEqual(summary["last_log_step"], 150)
        self.assertEqual(summary["delta_log_step"], 50)
        self.assertEqual(summary["log_steps_per_second"], 2.5)
        self.assertEqual(summary["estimated_seconds_until_next_eval"], 4.0)
        self.assertIn("first_log_step=100", lines[0])
        self.assertIn("log_duration_seconds=20", lines[0])
        self.assertIn("log_steps_per_second=2.5", lines[0])

    def test_monitor_snapshot_example_summarizes_long_ft_watchers(self) -> None:
        module = load_monitor_snapshot_example()
        eval_rows = [
            {
                "time_unix_s": 100.0,
                "process_status": "alive",
                "final_checkpoint_ready": False,
                "checkpoint_count": 0,
                "save_total_limit": 1,
                "runtime_settings": {
                    "max_steps": 8192,
                    "eval_steps": 512,
                    "save_steps": 2048,
                    "save_total_limit": 1,
                    "min_free_disk_gb": 4.0,
                    "process_command_available": True,
                },
                "checkpoint_headroom": {
                    "resume_checkpoint_gb": 0.75,
                    "estimated_peak_checkpoint_gb": 1.5,
                    "free_after_estimated_peak_gb": 8.5,
                },
                "disk_free_gb": 10.0,
                "disk_margin_gb": 6.0,
                "disk_status": "ok",
                "trace": {
                    "trace_last_loss": 3.6,
                    "trace_last_eval_loss": 3.290631,
                    "trace_last_eval_loss_step": 5120,
                    "trace_best_eval_loss_step": 5120,
                    "training_loss_guard_count": 0,
                },
                "log_progress": {
                    "log_latest_step": 5120,
                    "log_max_steps": 8192,
                    "log_remaining_seconds": 900.0,
                },
                "eval_progress": {
                    "next_eval_step": 5632,
                    "log_steps_until_next_eval": 512,
                    "latest_due_eval_step": 5120,
                    "latest_due_eval_ready": True,
                    "pending_eval_step": None,
                    "log_steps_since_pending_eval": None,
                },
                "checkpoint_progress": {
                    "next_checkpoint_step": 6144,
                    "log_steps_until_next_checkpoint": 1024,
                },
                "watch_stop_eval_step": 6144,
                "watch_stop_eval_ready": False,
            },
            {
                "time_unix_s": 180.0,
                "process_status": "alive",
                "final_checkpoint_ready": False,
                "checkpoint_count": 0,
                "save_total_limit": 1,
                "runtime_settings": {
                    "max_steps": 8192,
                    "eval_steps": 512,
                    "save_steps": 2048,
                    "save_total_limit": 1,
                    "min_free_disk_gb": 4.0,
                    "process_command_available": True,
                },
                "checkpoint_headroom": {
                    "resume_checkpoint_gb": 0.75,
                    "estimated_peak_checkpoint_gb": 1.5,
                    "free_after_estimated_peak_gb": 8.0,
                },
                "disk_free_gb": 9.5,
                "disk_margin_gb": 5.5,
                "disk_status": "ok",
                "trace": {
                    "trace_last_loss": 3.4,
                    "trace_last_eval_loss": 3.27533,
                    "trace_last_eval_loss_step": 5632,
                    "trace_best_eval_loss_step": 5632,
                    "trace_eval_loss_improvement": 0.015301,
                    "trace_eval_loss_last_delta": -0.015301,
                    "trace_eval_loss_last_improvement": 0.015301,
                    "trace_eval_loss_last_improvement_per_step": 0.0000298848,
                    "trace_eval_loss_mean_improvement_per_step": 0.0000298848,
                    "trace_eval_loss_last_improvement_ratio_to_previous": 0.5,
                    "trace_eval_loss_projection_step": 8192,
                    "trace_eval_loss_projection_remaining_steps": 2560.0,
                    "trace_eval_loss_projected_remaining_improvement": 0.076505,
                    "trace_eval_loss_projected_final_loss": 3.198825,
                    "trace_eval_loss_monotonic_nonincreasing": True,
                    "training_loss_guard_count": 0,
                },
                "log_progress": {
                    "log_latest_step": 5716,
                    "log_max_steps": 8192,
                    "log_remaining_seconds": 420.0,
                },
                "eval_progress": {
                    "next_eval_step": 6144,
                    "log_steps_until_next_eval": 428,
                    "latest_due_eval_step": 5632,
                    "latest_due_eval_ready": True,
                    "pending_eval_step": None,
                    "log_steps_since_pending_eval": None,
                },
                "checkpoint_progress": {
                    "next_checkpoint_step": 6144,
                    "log_steps_until_next_checkpoint": 428,
                },
                "watch_stop_eval_step": 6144,
                "watch_stop_eval_ready": False,
            },
        ]
        checkpoint_rows = [
            {
                **eval_rows[-1],
                "time_unix_s": 170.0,
                "watch_stop_reason": None,
            }
        ]
        final_rows = [
            {
                **eval_rows[-1],
                "time_unix_s": 175.0,
                "watch_stop_reason": None,
            }
        ]
        wait_rows = [
            {
                "time_unix_s": 190.0,
                "status": "waiting_for_process",
                "process_alive": True,
                "checkpoint_ready": False,
                "status_card_status": None,
                "launched_pid": None,
                "returncode": None,
                "launch_disk_guard": {
                    "status": "ok",
                    "min_free_gb": 4.0,
                    "estimated_peak_checkpoint_gb": 2.8,
                    "free_after_estimated_peak_gb": 5.6,
                },
            }
        ]
        direct_status_first = {
            **eval_rows[-1],
            "time_unix_s": 100.0,
            "log_progress": {
                "log_latest_step": 5700,
                "log_max_steps": 8192,
                "log_remaining_seconds": 500.0,
            },
            "eval_progress": {
                "next_eval_step": 6144,
                "log_steps_until_next_eval": 444,
                "latest_due_eval_step": 5632,
                "latest_due_eval_ready": True,
                "pending_eval_step": None,
                "log_steps_since_pending_eval": None,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 6144,
                "log_steps_until_next_checkpoint": 444,
            },
        }
        direct_status_last = {
            **eval_rows[-1],
            "time_unix_s": 200.0,
            "log_progress": {
                "log_latest_step": 5800,
                "log_max_steps": 8192,
                "log_remaining_seconds": 390.0,
            },
            "eval_progress": {
                "next_eval_step": 6144,
                "log_steps_until_next_eval": 344,
                "latest_due_eval_step": 5632,
                "latest_due_eval_ready": True,
                "pending_eval_step": None,
                "log_steps_since_pending_eval": None,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 6144,
                "log_steps_until_next_checkpoint": 344,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            next_run_dir = Path(tmp) / "next"
            run_dir.mkdir()
            next_run_dir.mkdir()
            run_status_history = run_dir / "direct-run-status-history.jsonl"
            eval_history = run_dir / "watch-6144-eval-confirm-history.jsonl"
            checkpoint_history = (
                run_dir / "watch-6144-checkpoint-confirm-history.jsonl"
            )
            final_history = run_dir / "watch-8192-final-history.jsonl"
            wait_history = next_run_dir / "finewebedu-16384-wait-launch-history.jsonl"
            run_status_history.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [direct_status_first, direct_status_last]
                )
                + "\n",
                encoding="utf-8",
            )
            for path, rows in [
                (eval_history, eval_rows),
                (checkpoint_history, checkpoint_rows),
                (final_history, final_rows),
                (wait_history, wait_rows),
            ]:
                path.write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
            out_path = run_dir / "monitor-snapshot.json"
            lines_path = run_dir / "monitor-snapshot.txt"
            argv = [
                str(run_dir),
                "--next-run-dir",
                str(next_run_dir),
                "--label",
                "long-ft",
                "--run-status-history-jsonl",
                str(run_status_history),
                "--milestone-step",
                "6144",
                "--out",
                str(out_path),
                "--lines-out",
                str(lines_path),
            ]
            args = module.parse_args(argv)
            snapshot = module.build_monitor_snapshot(args)
            lines = module.snapshot_lines(snapshot)
            self.assertEqual(module.main(argv), 0)
            quiet_stderr = io.StringIO()
            with mock.patch("sys.stderr", quiet_stderr):
                self.assertEqual(module.main(argv + ["--require-milestone-ready"]), 4)
            self.assertIn("milestone is not ready yet", quiet_stderr.getvalue())
            written = json.loads(out_path.read_text(encoding="utf-8"))
            written_lines = lines_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(snapshot["row_type"], "hf_gpt2_ft_monitor_snapshot")
        self.assertEqual(snapshot["primary_watch"], "direct")
        self.assertEqual(snapshot["process_status"], "alive")
        self.assertEqual(snapshot["log_latest_step"], 5800)
        self.assertEqual(snapshot["log_max_steps"], 8192)
        self.assertEqual(snapshot["runtime_max_steps"], 8192)
        self.assertEqual(snapshot["runtime_eval_steps"], 512)
        self.assertEqual(snapshot["runtime_save_steps"], 2048)
        self.assertEqual(snapshot["runtime_save_total_limit"], 1)
        self.assertEqual(snapshot["runtime_min_free_disk_gb"], 4.0)
        self.assertTrue(snapshot["runtime_process_command_available"])
        self.assertEqual(snapshot["log_remaining_seconds"], 390.0)
        self.assertEqual(snapshot["estimated_seconds_until_final"], 2392.0)
        self.assertEqual(snapshot["last_eval_loss_step"], 5632)
        self.assertEqual(snapshot["last_eval_loss"], 3.27533)
        self.assertEqual(snapshot["min_eval_loss"], 3.27533)
        self.assertEqual(snapshot["best_eval_loss_step"], 5632)
        self.assertAlmostEqual(snapshot["eval_loss_improvement"], 0.015301)
        self.assertAlmostEqual(snapshot["eval_loss_last_delta"], -0.015301)
        self.assertAlmostEqual(
            snapshot["eval_loss_last_improvement_per_step"],
            0.0000298848,
        )
        self.assertEqual(snapshot["eval_loss_projection_step"], 8192)
        self.assertAlmostEqual(
            snapshot["eval_loss_projected_final_loss"],
            3.198825,
        )
        self.assertTrue(snapshot["eval_loss_monotonic_nonincreasing"])
        self.assertEqual(snapshot["next_eval_step"], 6144)
        self.assertEqual(snapshot["steps_until_next_eval"], 344)
        self.assertEqual(snapshot["latest_due_eval_step"], 5632)
        self.assertTrue(snapshot["latest_due_eval_ready"])
        self.assertIsNone(snapshot["pending_eval_step"])
        self.assertIsNone(snapshot["log_steps_since_pending_eval"])
        self.assertAlmostEqual(
            snapshot["estimated_seconds_until_next_eval"],
            344.0,
        )
        self.assertEqual(snapshot["next_checkpoint_step"], 6144)
        self.assertEqual(snapshot["steps_until_next_checkpoint"], 344)
        self.assertEqual(snapshot["save_total_limit"], 1)
        self.assertEqual(snapshot["checkpoint_headroom_checkpoint_gb"], 0.75)
        self.assertEqual(snapshot["checkpoint_headroom_peak_gb"], 1.5)
        self.assertEqual(snapshot["checkpoint_headroom_free_after_gb"], 8.0)
        self.assertEqual(snapshot["disk_status"], "ok")
        self.assertEqual(snapshot["disk_margin_gb"], 5.5)
        self.assertTrue(snapshot["direct_status_available"])
        self.assertEqual(snapshot["milestone_step"], 6144)
        self.assertEqual(snapshot["milestone_status"], "waiting_for_step")
        self.assertFalse(snapshot["milestone_ready"])
        self.assertFalse(snapshot["milestone_step_reached"])
        self.assertEqual(snapshot["milestone_steps_until"], 344)
        self.assertFalse(snapshot["milestone_eval_ready"])
        self.assertIsNone(snapshot["milestone_eval_loss"])
        self.assertFalse(snapshot["milestone_checkpoint_ready"])
        self.assertEqual(snapshot["milestone_checkpoint"], "checkpoint-6144")
        self.assertFalse(snapshot["eval_watch_ready"])
        self.assertEqual(snapshot["eval_watch_step"], 6144)
        self.assertFalse(snapshot["wait_launch_checkpoint_ready"])
        self.assertFalse(snapshot["wait_launch_launched"])
        self.assertEqual(snapshot["wait_launch_status"], "waiting_for_process")
        self.assertEqual(snapshot["wait_launch_disk_status"], "ok")
        self.assertEqual(snapshot["wait_launch_disk_free_after_gb"], 5.6)
        self.assertEqual(written["log_latest_step"], 5800)
        self.assertIn("label=long-ft", lines[0])
        self.assertIn("primary=direct", lines[0])
        self.assertIn("log_step=5800", lines[0])
        self.assertIn("runtime_max_steps=8192", lines[0])
        self.assertIn("runtime_eval_steps=512", lines[0])
        self.assertIn("runtime_save_steps=2048", lines[0])
        self.assertIn("runtime_save_total_limit=1", lines[0])
        self.assertIn("runtime_min_free_disk_gb=4", lines[0])
        self.assertIn("runtime_process_command=true", lines[0])
        self.assertIn("log_remaining_seconds=390", lines[0])
        self.assertIn("estimated_seconds_until_final=2392", lines[0])
        self.assertIn("last_eval_step=5632", lines[0])
        self.assertIn("last_eval_loss=3.27533", lines[0])
        self.assertIn("eval_loss_improvement=0.015301", lines[0])
        self.assertIn("eval_loss_last_delta=-0.015301", lines[0])
        self.assertIn("eval_loss_projected_final=3.19882", lines[0])
        self.assertIn("eval_loss_monotonic=true", lines[0])
        self.assertIn("next_eval_step=6144", lines[0])
        self.assertIn("latest_due_eval_step=5632", lines[0])
        self.assertIn("latest_due_eval_ready=true", lines[0])
        self.assertIn("pending_eval_step=none", lines[0])
        self.assertIn("steps_until_next_checkpoint=344", lines[0])
        self.assertIn("save_total_limit=1", lines[0])
        self.assertIn("checkpoint_headroom_peak_gb=1.5", lines[0])
        self.assertIn("checkpoint_headroom_free_after_gb=8", lines[0])
        self.assertIn("disk_margin_gb=5.5", lines[0])
        self.assertIn("direct_status_available=true", lines[0])
        self.assertIn("milestone_step=6144", lines[0])
        self.assertIn("milestone_status=waiting_for_step", lines[0])
        self.assertIn("milestone_ready=false", lines[0])
        self.assertIn("milestone_eval_ready=false", lines[0])
        self.assertIn("milestone_checkpoint_ready=false", lines[0])
        self.assertIn("eval_watch_ready=false", lines[0])
        self.assertIn("wait_status=waiting_for_process", lines[0])
        self.assertIn("wait_disk_status=ok", lines[0])
        self.assertIn("wait_disk_free_after_gb=5.6", lines[0])
        self.assertIn("name=direct", lines[1])
        self.assertIn("rows=2", lines[1])
        self.assertIn("runtime_eval_steps=512", lines[1])
        self.assertIn("runtime_save_steps=2048", lines[1])
        self.assertIn("runtime_process_command=true", lines[1])
        self.assertIn("eval_loss_projected_final=3.19882", lines[1])
        self.assertIn("pending_eval_step=none", lines[1])
        self.assertIn("checkpoint_headroom_peak_gb=1.5", lines[1])
        self.assertIn("name=eval", lines[2])
        self.assertIn("rows=2", lines[2])
        self.assertIn("name=checkpoint", lines[3])
        self.assertIn("name=final", lines[4])
        self.assertIn("status=waiting_for_process", lines[5])
        self.assertIn("launch_disk_status=ok", lines[5])
        self.assertIn("launch_disk_free_after_gb=5.6", lines[5])
        self.assertEqual(written_lines, lines)
        ready_watches = {
            "direct": {
                "log_latest_step": 6144,
                "eval_loss_points": [{"step": 6144, "eval_loss": 3.2}],
                "checkpoint_names": ["checkpoint-6144"],
            },
            "eval": {},
            "checkpoint": {},
            "final": {},
        }
        ready = module._milestone_summary(
            milestone_step=6144,
            primary=ready_watches["direct"],
            watches=ready_watches,
        )
        self.assertEqual(ready["milestone_status"], "ready")
        self.assertTrue(ready["milestone_ready"])
        self.assertTrue(ready["milestone_step_reached"])
        self.assertEqual(ready["milestone_eval_loss"], 3.2)
        self.assertTrue(ready["milestone_checkpoint_ready"])

    def test_package_monitor_report_summarizes_watchers_and_milestone(self) -> None:
        base_status = {
            "row_type": "hf_gpt2_finetune_run_status",
            "process_status": "alive",
            "final_checkpoint_ready": False,
            "checkpoint_count": 1,
            "save_total_limit": 1,
            "runtime_settings": {
                "max_steps": 8192,
                "eval_steps": 512,
                "save_steps": 2048,
                "save_total_limit": 1,
                "min_free_disk_gb": 4.0,
                "process_command_available": True,
            },
            "checkpoint_headroom": {
                "resume_checkpoint_gb": 0.75,
                "estimated_peak_checkpoint_gb": 1.5,
                "free_after_estimated_peak_gb": 8.0,
            },
            "disk_free_gb": 9.5,
            "disk_margin_gb": 5.5,
            "disk_status": "ok",
            "trace": {
                "trace_last_loss": 3.4,
                "trace_last_eval_loss": 3.27533,
                "trace_last_eval_loss_step": 5632,
                "trace_best_eval_loss_step": 5632,
                "trace_eval_loss_projected_final_loss": 3.198825,
                "trace_eval_loss_points": [
                    {"step": 5632, "eval_loss": 3.27533}
                ],
                "training_loss_guard_count": 0,
            },
            "log_progress": {
                "log_latest_step": 5800,
                "log_max_steps": 8192,
                "log_remaining_seconds": 390.0,
            },
            "eval_progress": {
                "next_eval_step": 6144,
                "log_steps_until_next_eval": 344,
                "latest_due_eval_step": 5632,
                "latest_due_eval_ready": True,
                "pending_eval_step": None,
                "log_steps_since_pending_eval": None,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 6144,
                "log_steps_until_next_checkpoint": 344,
            },
            "latest_checkpoint": {"name": "checkpoint-4096"},
        }
        direct_rows = [
            {
                **base_status,
                "time_unix_s": 100.0,
                "log_progress": {
                    **base_status["log_progress"],
                    "log_latest_step": 5700,
                },
            },
            {**base_status, "time_unix_s": 200.0},
        ]
        eval_rows = [
            {
                **base_status,
                "time_unix_s": 180.0,
                "watch_stop_eval_step": 6144,
                "watch_stop_eval_ready": False,
            }
        ]
        wait_rows = [
            {
                "time_unix_s": 205.0,
                "status": "waiting_for_process",
                "process_alive": True,
                "checkpoint_ready": False,
                "launch_disk_guard": {
                    "status": "ok",
                    "min_free_gb": 4.0,
                    "estimated_peak_checkpoint_gb": 2.8,
                    "free_after_estimated_peak_gb": 5.6,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "direct-run-status-history.jsonl"
            history_path.write_text(
                "\n".join(json.dumps(row) for row in direct_rows) + "\n",
                encoding="utf-8",
            )
            snapshot = st.hf_gpt2_finetune_monitor_report(
                direct=history_path,
                eval_watch=eval_rows,
                checkpoint_watch=[],
                final_watch=[],
                wait_launch=wait_rows,
                milestone_step=6144,
                label="long-ft",
                run_dir=Path(tmp) / "run",
            )

        lines = st.hf_gpt2_finetune_monitor_lines(snapshot)

        self.assertEqual(snapshot["row_type"], "hf_gpt2_finetune_monitor_report")
        self.assertEqual(snapshot["primary_watch"], "direct")
        self.assertEqual(snapshot["process_status"], "alive")
        self.assertEqual(snapshot["log_latest_step"], 5800)
        self.assertEqual(snapshot["runtime_eval_steps"], 512)
        self.assertTrue(snapshot["direct_status_available"])
        self.assertEqual(snapshot["last_eval_loss_step"], 5632)
        self.assertEqual(snapshot["last_eval_loss"], 3.27533)
        self.assertEqual(snapshot["next_eval_step"], 6144)
        self.assertEqual(snapshot["next_checkpoint_step"], 6144)
        self.assertEqual(snapshot["wait_launch_status"], "waiting_for_process")
        self.assertFalse(snapshot["wait_launch_launched"])
        self.assertEqual(snapshot["wait_launch_disk_status"], "ok")
        self.assertEqual(snapshot["wait_launch_disk_free_after_gb"], 5.6)
        self.assertEqual(snapshot["milestone_status"], "waiting_for_step")
        self.assertFalse(snapshot["milestone_ready"])
        self.assertFalse(snapshot["milestone_eval_ready"])
        self.assertFalse(snapshot["milestone_checkpoint_ready"])
        self.assertIn("hf_gpt2_ft_monitor ", lines[0])
        self.assertIn("label=long-ft", lines[0])
        self.assertIn("primary=direct", lines[0])
        self.assertIn("milestone_status=waiting_for_step", lines[0])
        self.assertIn("wait_status=waiting_for_process", lines[0])
        self.assertIn("name=direct", lines[1])
        self.assertIn("rows=2", lines[1])
        self.assertIn("name=eval", lines[2])
        self.assertIn("rows=1", lines[2])
        self.assertIn("hf_gpt2_ft_monitor_wait_launch", lines[-1])
        capture = st.hf_gpt2_finetune_milestone_capture_report(
            snapshot,
            iteration=1,
            commands=[{"returncode": 0}],
        )
        capture_lines = st.hf_gpt2_finetune_milestone_capture_lines(capture)

        self.assertEqual(capture["row_type"], "hf_gpt2_finetune_milestone_capture")
        self.assertEqual(capture["status"], "waiting_for_step")
        self.assertFalse(capture["milestone_ready"])
        self.assertEqual(capture["milestone_step"], 6144)
        self.assertEqual(capture["iteration"], 1)
        self.assertEqual(capture["commands"][0]["returncode"], 0)
        self.assertEqual(capture["next_action"], "keep_watching")
        self.assertTrue(capture["should_continue_watch"])
        self.assertIn("next_action=keep_watching", capture_lines[0])

        ready_direct = {
            **base_status,
            "time_unix_s": 240.0,
            "trace": {
                **base_status["trace"],
                "trace_last_eval_loss": 3.2,
                "trace_last_eval_loss_step": 6144,
                "trace_eval_loss_points": [
                    *base_status["trace"]["trace_eval_loss_points"],
                    {"step": 6144, "eval_loss": 3.2},
                ],
            },
            "log_progress": {
                **base_status["log_progress"],
                "log_latest_step": 6144,
            },
        }
        ready_snapshot = st.hf_gpt2_finetune_monitor_report(
            direct=[ready_direct],
            eval_watch=[ready_direct],
            checkpoint_watch=[
                {
                    **ready_direct,
                    "checkpoint_names": ["checkpoint-6144"],
                    "latest_checkpoint": {"name": "checkpoint-6144"},
                }
            ],
            milestone_step=6144,
            label="ready-ft",
        )

        self.assertEqual(ready_snapshot["milestone_status"], "ready")
        self.assertTrue(ready_snapshot["milestone_ready"])
        self.assertTrue(ready_snapshot["milestone_eval_ready"])
        self.assertTrue(ready_snapshot["milestone_checkpoint_ready"])
        self.assertEqual(ready_snapshot["milestone_eval_loss"], 3.2)
        ready_capture = st.hf_gpt2_finetune_milestone_capture_report(ready_snapshot)
        self.assertEqual(ready_capture["next_action"], "handoff")
        self.assertFalse(ready_capture["should_continue_watch"])
        handoff = st.hf_gpt2_finetune_milestone_handoff_report(
            ready_capture,
            run_dir="/tmp/spiraltorch-ft",
            compare_with_sweep="previous-sweep.json",
            compare_with_label="previous",
            dry_run=True,
        )
        handoff_lines = st.hf_gpt2_finetune_milestone_handoff_lines(handoff)

        self.assertEqual(handoff["row_type"], "hf_gpt2_finetune_milestone_handoff")
        self.assertEqual(handoff["status"], "ready")
        self.assertTrue(handoff["ready"])
        self.assertEqual(handoff["action"], "checkpoint_generation_control")
        self.assertEqual(handoff["checkpoint"], "checkpoint-6144")
        self.assertEqual(handoff["checkpoint_path"], "/tmp/spiraltorch-ft/checkpoint-6144")
        self.assertIn("--checkpoint", handoff["command"])
        self.assertIn("checkpoint-6144", handoff["command"])
        self.assertIn("--compare-with-sweep", handoff["command"])
        self.assertIn("previous-sweep.json", handoff["command"])
        self.assertIn("--dry-run", handoff["command"])
        self.assertEqual(
            handoff["package_function"],
            "spiraltorch.zspace_checkpoint_generation_control_report",
        )
        self.assertEqual(handoff["package_kwargs"]["run_dir"], "/tmp/spiraltorch-ft")
        self.assertEqual(handoff["package_kwargs"]["checkpoint"], "checkpoint-6144")
        self.assertEqual(handoff["package_kwargs"]["compare_with_sweep"], ["previous-sweep.json"])
        self.assertEqual(handoff["package_kwargs"]["compare_with_label"], ["previous"])
        self.assertTrue(handoff["package_kwargs"]["dry_run"])
        self.assertIn("checkpoint=checkpoint-6144", handoff_lines[0])
        execution_plan = st.hf_gpt2_finetune_milestone_handoff_execution_report(
            handoff
        )
        execution_plan_lines = (
            st.hf_gpt2_finetune_milestone_handoff_execution_lines(execution_plan)
        )

        self.assertEqual(
            execution_plan["row_type"],
            "hf_gpt2_finetune_milestone_handoff_execution",
        )
        self.assertEqual(execution_plan["status"], "planned")
        self.assertFalse(execution_plan["run"])
        self.assertEqual(execution_plan["checkpoint"], "checkpoint-6144")
        self.assertEqual(execution_plan["command"], handoff["command"])
        self.assertEqual(execution_plan["execution_backend"], "command")
        self.assertIn("status=planned", execution_plan_lines[0])
        self.assertIn("run=false", execution_plan_lines[0])
        self.assertIn("backend=command", execution_plan_lines[0])

        runner_calls = []

        def fake_runner(command, cwd, env, text, capture_output, check, timeout):
            runner_calls.append(
                {
                    "command": list(command),
                    "cwd": cwd,
                    "env": env,
                    "text": text,
                    "capture_output": capture_output,
                    "check": check,
                    "timeout": timeout,
                }
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"status": "planned", "sweep_count": 3}),
                stderr="",
            )

        execution = st.hf_gpt2_finetune_milestone_handoff_execution_report(
            handoff,
            run=True,
            cwd="/tmp/spiraltorch-ft",
            env={"SPIRALTORCH_TEST_FLAG": "1"},
            timeout=12.5,
            runner=fake_runner,
        )
        execution_lines = st.hf_gpt2_finetune_milestone_handoff_execution_lines(
            execution
        )

        self.assertEqual(execution["status"], "complete")
        self.assertTrue(execution["run"])
        self.assertEqual(execution["returncode"], 0)
        self.assertEqual(execution["command_report"]["sweep_count"], 3)
        self.assertEqual(runner_calls[0]["command"], handoff["command"])
        self.assertEqual(runner_calls[0]["cwd"], "/tmp/spiraltorch-ft")
        self.assertEqual(runner_calls[0]["env"]["SPIRALTORCH_TEST_FLAG"], "1")
        self.assertTrue(runner_calls[0]["capture_output"])
        self.assertEqual(runner_calls[0]["timeout"], 12.5)
        self.assertIn("status=complete", execution_lines[0])
        self.assertIn("command_report_status=planned", execution_lines[0])
        package_calls = []

        def fake_package_runner(**kwargs):
            package_calls.append(dict(kwargs))
            return {"status": "planned", "sweep_count": 3}

        package_execution = st.hf_gpt2_finetune_milestone_handoff_execution_report(
            handoff,
            run=True,
            use_package_api=True,
            package_runner=fake_package_runner,
        )
        package_execution_lines = (
            st.hf_gpt2_finetune_milestone_handoff_execution_lines(package_execution)
        )

        self.assertEqual(package_execution["status"], "complete")
        self.assertEqual(package_execution["execution_backend"], "package_api")
        self.assertEqual(package_execution["returncode"], 0)
        self.assertEqual(package_execution["command_report"]["sweep_count"], 3)
        self.assertEqual(package_calls[0]["checkpoint"], "checkpoint-6144")
        self.assertEqual(package_calls[0]["run_dir"], "/tmp/spiraltorch-ft")
        self.assertTrue(package_calls[0]["dry_run"])
        self.assertIn("backend=package_api", package_execution_lines[0])
        self.assertIn("command_report_status=planned", package_execution_lines[0])
        runtime_calls = []

        def fake_runtime_package_runner(**kwargs):
            runtime_calls.append(dict(kwargs))
            return {"status": "planned", "sweep_count": 3}

        runtime = st.hf_gpt2_finetune_milestone_runtime_report(
            direct=[ready_direct],
            eval_watch=[ready_direct],
            checkpoint_watch=[
                {
                    **ready_direct,
                    "checkpoint_names": ["checkpoint-6144"],
                    "latest_checkpoint": {"name": "checkpoint-6144"},
                }
            ],
            milestone_step=6144,
            label="ready-ft",
            run_dir="/tmp/spiraltorch-ft",
            compare_with_sweep="previous-sweep.json",
            compare_with_label="previous",
            execute=True,
            use_package_api=True,
            package_runner=fake_runtime_package_runner,
        )
        runtime_lines = st.hf_gpt2_finetune_milestone_runtime_lines(runtime)

        self.assertEqual(runtime["row_type"], "hf_gpt2_finetune_milestone_runtime")
        self.assertEqual(runtime["status"], "executed")
        self.assertEqual(runtime["handoff_status"], "ready")
        self.assertEqual(runtime["execution_status"], "complete")
        self.assertEqual(runtime["execution_backend"], "package_api")
        self.assertEqual(runtime["checkpoint"], "checkpoint-6144")
        self.assertEqual(runtime_calls[0]["checkpoint"], "checkpoint-6144")
        self.assertEqual(runtime_calls[0]["run_dir"], "/tmp/spiraltorch-ft")
        self.assertTrue(runtime_calls[0]["dry_run"])
        self.assertIn("hf_gpt2_ft_milestone_runtime", runtime_lines[0])
        self.assertIn("status=executed", runtime_lines[0])
        self.assertTrue(
            any("hf_gpt2_ft_milestone_handoff_execution" in line for line in runtime_lines)
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            direct_history = run_dir / "direct-run-status-history.jsonl"
            eval_history = run_dir / "watch-6144-eval-confirm-history.jsonl"
            checkpoint_history = run_dir / "watch-6144-checkpoint-confirm-history.jsonl"
            capture_json = run_dir / "milestone-6144-capture.json"
            for path, row in (
                (direct_history, ready_direct),
                (eval_history, ready_direct),
                (
                    checkpoint_history,
                    {
                        **ready_direct,
                        "checkpoint_names": ["checkpoint-6144"],
                        "latest_checkpoint": {"name": "checkpoint-6144"},
                    },
                ),
            ):
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            capture_json.write_text(
                json.dumps({"milestone_step": 6144, "milestone_ready": True}) + "\n",
                encoding="utf-8",
            )
            source_calls = []

            def fake_source_package_runner(**kwargs):
                source_calls.append(dict(kwargs))
                return {"status": "planned", "sweep_count": 3}

            sources = st.hf_gpt2_finetune_milestone_runtime_sources(run_dir)
            source_runtime = (
                st.hf_gpt2_finetune_milestone_runtime_from_run_dir_report(
                    run_dir,
                    label="source-ft",
                    execute=True,
                    use_package_api=True,
                    package_runner=fake_source_package_runner,
                )
            )

        self.assertEqual(sources["direct"], str(direct_history))
        self.assertEqual(sources["eval"], str(eval_history))
        self.assertEqual(sources["checkpoint"], str(checkpoint_history))
        self.assertEqual(source_runtime["status"], "executed")
        self.assertEqual(source_runtime["source_count"], 3)
        self.assertEqual(source_runtime["milestone_step"], 6144)
        self.assertEqual(source_runtime["milestone_step_source"], str(capture_json))
        self.assertEqual(source_runtime["sources"]["direct"], str(direct_history))
        self.assertEqual(source_runtime["checkpoint"], "checkpoint-6144")
        self.assertEqual(source_calls[0]["run_dir"], str(run_dir))
        self.assertEqual(source_calls[0]["checkpoint"], "checkpoint-6144")
        self.assertTrue(source_calls[0]["dry_run"])
        waiting_handoff = st.hf_gpt2_finetune_milestone_handoff_report(
            capture,
            run_dir="/tmp/spiraltorch-ft",
        )
        self.assertEqual(waiting_handoff["status"], "waiting_for_milestone")

    def test_milestone_runtime_example_writes_inferred_artifacts(self) -> None:
        module = load_milestone_runtime_example()
        ready_direct = {
            "row_type": "hf_gpt2_finetune_run_status",
            "process_status": "alive",
            "final_checkpoint_ready": False,
            "checkpoint_count": 2,
            "runtime_settings": {
                "max_steps": 8192,
                "eval_steps": 512,
                "save_steps": 2048,
                "save_total_limit": 2,
                "min_free_disk_gb": 4.0,
                "process_command_available": True,
            },
            "checkpoint_headroom": {
                "resume_checkpoint_gb": 0.75,
                "estimated_peak_checkpoint_gb": 1.5,
                "free_after_estimated_peak_gb": 8.0,
            },
            "disk_free_gb": 9.5,
            "disk_margin_gb": 5.5,
            "disk_status": "ok",
            "trace": {
                "trace_last_loss": 3.35,
                "trace_last_eval_loss": 3.2,
                "trace_last_eval_loss_step": 6144,
                "trace_best_eval_loss_step": 6144,
                "trace_eval_loss_projected_final_loss": 3.12,
                "trace_eval_loss_points": [{"step": 6144, "eval_loss": 3.2}],
                "training_loss_guard_count": 0,
            },
            "log_progress": {
                "log_latest_step": 6144,
                "log_max_steps": 8192,
                "log_remaining_seconds": 120.0,
            },
            "eval_progress": {
                "next_eval_step": 6656,
                "log_steps_until_next_eval": 512,
                "latest_due_eval_step": 6144,
                "latest_due_eval_ready": True,
                "pending_eval_step": None,
                "log_steps_since_pending_eval": None,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 8192,
                "log_steps_until_next_checkpoint": 2048,
            },
            "latest_checkpoint": {"name": "checkpoint-6144"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "checkpoint-6144").mkdir()
            direct_history = run_dir / "direct-run-status-history.jsonl"
            eval_history = run_dir / "watch-6144-eval-confirm-history.jsonl"
            checkpoint_history = (
                run_dir / "watch-6144-checkpoint-confirm-history.jsonl"
            )
            capture_json = run_dir / "milestone-6144-capture.json"
            checkpoint_row = {
                **ready_direct,
                "checkpoint_names": ["checkpoint-6144"],
                "latest_checkpoint": {"name": "checkpoint-6144"},
            }
            for path, row in (
                (direct_history, ready_direct),
                (eval_history, ready_direct),
                (checkpoint_history, checkpoint_row),
            ):
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            capture_json.write_text(
                json.dumps({"milestone_step": 6144, "milestone_ready": True}) + "\n",
                encoding="utf-8",
            )
            package_calls = []

            def fake_package_runner(**kwargs):
                package_calls.append(dict(kwargs))
                return {"status": "planned", "sweep_count": 3}

            args = module.parse_args(
                ["--run-dir", str(run_dir), "--label", "runtime-ft", "--execute"]
            )
            report = module.build_report(args, package_runner=fake_package_runner)
            out_path, lines_path = module.write_report(report, args)

            self.assertEqual(out_path, run_dir / "milestone-6144-runtime.json")
            self.assertEqual(lines_path, run_dir / "milestone-6144-runtime.txt")
            self.assertTrue(out_path.is_file())
            self.assertTrue(lines_path.is_file())
            written = json.loads(out_path.read_text(encoding="utf-8"))
            written_lines = lines_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(written["row_type"], "hf_gpt2_finetune_milestone_runtime")
        self.assertEqual(written["status"], "executed")
        self.assertEqual(written["label"], "runtime-ft")
        self.assertEqual(written["milestone_step"], 6144)
        self.assertEqual(written["milestone_step_source"], str(capture_json))
        self.assertEqual(written["execution_backend"], "package_api")
        self.assertEqual(written["checkpoint"], "checkpoint-6144")
        self.assertEqual(written["out"], str(out_path))
        self.assertEqual(written["lines_out"], str(lines_path))
        self.assertEqual(package_calls[0]["run_dir"], str(run_dir))
        self.assertEqual(package_calls[0]["checkpoint"], "checkpoint-6144")
        self.assertTrue(package_calls[0]["dry_run"])
        self.assertIn("hf_gpt2_ft_milestone_runtime ", written_lines[0])
        self.assertIn("status=executed", written_lines[0])
        self.assertTrue(
            any(
                "hf_gpt2_ft_milestone_handoff_execution" in line
                for line in written_lines
            )
        )

    def test_package_milestone_report_tracks_run_status_readiness(self) -> None:
        waiting_status = {
            "row_type": "hf_gpt2_finetune_run_status",
            "time_unix_s": 123.0,
            "run_dir": "/tmp/spiraltorch-ft",
            "process_status": "alive",
            "trace": {
                "trace_max_global_step": 10075,
                "trace_last_eval_loss": 3.236955165863037,
                "trace_effective_last_eval_loss_step": 9728,
                "trace_min_eval_loss": 3.236955165863037,
                "trace_best_eval_loss_step": 9728,
                "trace_eval_loss_points": [
                    {"step": 9216, "eval_loss": 3.242262601852417},
                    {"step": 9728, "eval_loss": 3.236955165863037},
                ],
                "trace_eval_loss_last_delta": -0.005307435989379883,
                "trace_eval_loss_projected_final_loss": 3.1679584980010986,
                "trace_eval_loss_monotonic_nonincreasing": True,
            },
            "log_progress": {
                "log_latest_step": 10075,
                "log_max_steps": 16384,
                "log_progress": 0.61492919921875,
                "log_remaining_seconds": 1234.0,
            },
            "runtime_settings": {
                "max_steps": 16384,
                "eval_steps": 512,
                "save_steps": 2048,
                "save_total_limit": 1,
                "min_free_disk_gb": 4.0,
            },
            "eval_progress": {
                "next_eval_step": 10240,
                "log_steps_until_next_eval": 165,
                "latest_due_eval_step": 9728,
                "latest_due_eval_ready": True,
                "pending_eval_step": None,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 10240,
                "log_steps_until_next_checkpoint": 165,
            },
            "checkpoints": [{"name": "checkpoint-8192"}],
            "latest_checkpoint": {"name": "checkpoint-8192"},
            "checkpoint_count": 1,
            "save_total_limit": 1,
            "checkpoint_headroom": {
                "resume_checkpoint_gb": 0.35,
                "estimated_peak_checkpoint_gb": 0.7,
                "free_after_estimated_peak_gb": 13.9,
            },
            "final_checkpoint_ready": False,
            "disk_free_gb": 14.6,
            "disk_margin_gb": 10.6,
            "disk_status": "ok",
        }

        waiting = hf_ft.hf_gpt2_finetune_milestone_report(
            waiting_status,
            milestone_step=10240,
            label="finewebedu-10240-live",
        )
        waiting_lines = hf_ft.hf_gpt2_finetune_milestone_lines(waiting)

        self.assertEqual(waiting["row_type"], "hf_gpt2_finetune_milestone_report")
        self.assertEqual(waiting["milestone_status"], "waiting_for_step")
        self.assertFalse(waiting["milestone_ready"])
        self.assertFalse(waiting["milestone_step_reached"])
        self.assertEqual(waiting["milestone_steps_until"], 165)
        self.assertFalse(waiting["milestone_eval_ready"])
        self.assertFalse(waiting["milestone_checkpoint_ready"])
        self.assertEqual(waiting["runtime_eval_steps"], 512)
        self.assertEqual(waiting["next_eval_step"], 10240)
        self.assertEqual(waiting["latest_due_eval_step"], 9728)
        self.assertTrue(waiting["latest_due_eval_ready"])
        self.assertEqual(waiting["latest_checkpoint"], "checkpoint-8192")
        self.assertIn("status=waiting_for_step", waiting_lines[0])
        self.assertIn("steps_until=165", waiting_lines[0])
        self.assertIn("eval_ready=false", waiting_lines[0])
        self.assertIn("projected_final=3.16796", waiting_lines[1])

        ready_status = dict(waiting_status)
        ready_status["log_progress"] = {
            **waiting_status["log_progress"],
            "log_latest_step": 10240,
            "log_progress": 0.625,
        }
        ready_status["trace"] = {
            **waiting_status["trace"],
            "trace_max_global_step": 10240,
            "trace_last_eval_loss": 3.2301,
            "trace_effective_last_eval_loss_step": 10240,
            "trace_eval_loss_points": [
                *waiting_status["trace"]["trace_eval_loss_points"],
                {"step": 10240, "eval_loss": 3.2301},
            ],
        }
        ready_status["checkpoints"] = [
            {"name": "checkpoint-8192"},
            {"name": "checkpoint-10240"},
        ]
        ready_status["latest_checkpoint"] = {"name": "checkpoint-10240"}

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "direct-run-status.json"
            status_path.write_text(json.dumps(ready_status), encoding="utf-8")
            ready = st.hf_gpt2_finetune_milestone_report(
                status_path,
                milestone_step=10240,
            )

        self.assertEqual(ready["source_row_type"], "hf_gpt2_finetune_run_status")
        self.assertEqual(ready["milestone_status"], "ready")
        self.assertTrue(ready["milestone_ready"])
        self.assertTrue(ready["milestone_step_reached"])
        self.assertEqual(ready["milestone_steps_until"], 0)
        self.assertTrue(ready["milestone_eval_ready"])
        self.assertEqual(ready["milestone_eval_loss"], 3.2301)
        self.assertTrue(ready["milestone_checkpoint_ready"])
        self.assertEqual(ready["latest_checkpoint"], "checkpoint-10240")

    def test_monitor_snapshot_keeps_resume_eval_ready_when_direct_is_stale(
        self,
    ) -> None:
        module = load_monitor_snapshot_example()
        stale_direct = {
            "time_unix_s": 100.0,
            "process_status": "alive",
            "final_checkpoint_ready": False,
            "trace": {
                "trace_last_eval_loss": 3.2568,
                "trace_last_eval_loss_step": 0,
                "trace_eval_loss_points": [{"step": 0, "eval_loss": 3.2568}],
            },
            "log_progress": {
                "log_latest_step": 8238,
                "log_max_steps": 16384,
                "log_remaining_seconds": 30000.0,
            },
            "eval_progress": {
                "next_eval_step": 8704,
                "log_steps_until_next_eval": 466,
                "latest_due_eval_step": 8192,
                "latest_due_eval_ready": False,
                "pending_eval_step": 8192,
                "log_steps_since_pending_eval": 46,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 10240,
                "log_steps_until_next_checkpoint": 2002,
            },
        }
        fixed_final = {
            "time_unix_s": 120.0,
            "process_status": "alive",
            "final_checkpoint_ready": False,
            "trace": {
                "trace_last_eval_loss": 3.2568,
                "trace_last_eval_loss_step": 0,
                "trace_effective_last_eval_loss_step": 8192,
                "trace_eval_loss_points": [{"step": 0, "eval_loss": 3.2568}],
            },
            "log_progress": {
                "log_latest_step": 8302,
                "log_max_steps": 16384,
                "log_remaining_seconds": 31000.0,
            },
            "eval_progress": {
                "next_eval_step": 8704,
                "log_steps_until_next_eval": 402,
                "latest_due_eval_step": 8192,
                "latest_due_eval_ready": True,
                "pending_eval_step": None,
                "log_steps_since_pending_eval": None,
            },
            "checkpoint_progress": {
                "next_checkpoint_step": 10240,
                "log_steps_until_next_checkpoint": 1938,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            direct_history = run_dir / "direct-run-status-history.jsonl"
            final_history = run_dir / "watch-16384-final-v2-history.jsonl"
            direct_history.write_text(json.dumps(stale_direct) + "\n", encoding="utf-8")
            final_history.write_text(json.dumps(fixed_final) + "\n", encoding="utf-8")
            args = module.parse_args(
                [
                    str(run_dir),
                    "--run-status-history-jsonl",
                    str(direct_history),
                    "--final-history-jsonl",
                    str(final_history),
                    "--milestone-step",
                    "16384",
                ]
            )
            snapshot = module.build_monitor_snapshot(args)
            lines = module.snapshot_lines(snapshot)

        self.assertEqual(snapshot["primary_watch"], "final")
        self.assertEqual(snapshot["last_eval_loss_step"], 8192)
        self.assertEqual(snapshot["latest_due_eval_step"], 8192)
        self.assertTrue(snapshot["latest_due_eval_ready"])
        self.assertIsNone(snapshot["pending_eval_step"])
        self.assertIsNone(snapshot["log_steps_since_pending_eval"])
        self.assertIn("last_eval_step=8192", lines[0])
        self.assertIn("latest_due_eval_ready=true", lines[0])
        self.assertIn("pending_eval_step=none", lines[0])

    def test_monitor_snapshot_reconstructs_legacy_wait_launch_disk_guard(
        self,
    ) -> None:
        module = load_monitor_snapshot_example()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_checkpoint = tmp_path / "checkpoint-6144"
            source_checkpoint.mkdir()
            (source_checkpoint / "model.safetensors").write_text(
                "ready",
                encoding="utf-8",
            )
            checkpoint = tmp_path / "checkpoint-8192"
            output_dir = tmp_path / "next-run"
            rows = [
                {
                    "row_type": "hf_gpt2_finetune_wait_launch",
                    "status": "waiting_for_process",
                    "time_unix_s": 10.0,
                    "process_alive": True,
                    "checkpoint": str(checkpoint),
                    "checkpoint_ready": False,
                    "command": [
                        sys.executable,
                        "bindings/st-py/examples/hf_gpt2_finetune_bridge.py",
                        "--resume-from-checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(output_dir),
                        "--save-total-limit",
                        "1",
                        "--min-free-disk-gb",
                        "0",
                    ],
                }
            ]
            summary = module._wait_launch_summary(rows, history_jsonl=None)
            guard = module._launch_disk_guard(rows[0])

        self.assertEqual(summary["launch_disk_status"], "reconstructed_ok")
        self.assertGreater(summary["launch_disk_free_after_gb"], 0.0)
        self.assertEqual(guard["resume_checkpoint_bytes"], 5)
        self.assertEqual(
            guard["resume_checkpoint_estimate_source"],
            str(source_checkpoint),
        )

    def test_milestone_capture_example_refreshes_status_and_monitor(self) -> None:
        module = load_milestone_capture_example()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_dir = repo / "run"
            next_run_dir = repo / "next"
            run_dir.mkdir()
            next_run_dir.mkdir()
            argv = [
                str(run_dir),
                "--next-run-dir",
                str(next_run_dir),
                "--label",
                "demo-6144",
                "--milestone-step",
                "6144",
                "--max-steps",
                "8192",
                "--eval-steps",
                "512",
                "--save-steps",
                "2048",
                "--min-free-disk-gb",
                "4",
                "--final-checkpoint",
                "checkpoint-8192",
                "--watch-count",
                "1",
                "--watch-interval-seconds",
                "0",
                "--quiet",
            ]
            args = module.parse_args(argv)
            snapshot = {
                "row_type": "hf_gpt2_ft_monitor_snapshot",
                "milestone_status": "waiting_for_step",
                "milestone_ready": False,
                "milestone_step": 6144,
                "milestone_steps_until": 128,
                "milestone_eval_loss": None,
                "milestone_checkpoint_ready": False,
                "process_status": "alive",
                "log_latest_step": 6016,
            }

            def fake_run(command, cwd, env, text, capture_output, check):
                self.assertEqual(cwd, repo)
                self.assertTrue(text)
                self.assertTrue(capture_output)
                self.assertFalse(check)
                if "hf_gpt2_finetune_run_status.py" in command[1]:
                    out_path = repo / command[command.index("--out") + 1]
                    lines_path = repo / command[command.index("--lines-out") + 1]
                    jsonl_path = repo / command[command.index("--jsonl-out") + 1]
                    out_path.write_text(
                        json.dumps({"process_status": "alive"}),
                        encoding="utf-8",
                    )
                    lines_path.write_text("status\n", encoding="utf-8")
                    jsonl_path.write_text(
                        json.dumps({"process_status": "alive"}) + "\n",
                        encoding="utf-8",
                    )
                if "hf_gpt2_finetune_monitor_snapshot.py" in command[1]:
                    out_path = repo / command[command.index("--out") + 1]
                    lines_path = repo / command[command.index("--lines-out") + 1]
                    out_path.write_text(json.dumps(snapshot), encoding="utf-8")
                    lines_path.write_text("monitor\n", encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                state = module.capture_once(args, repo=repo, env={})
                with mock.patch.object(module, "_repo_root", return_value=repo):
                    exit_code = module.main(argv)
            state_written = json.loads(args.state_out.read_text(encoding="utf-8"))
            history_rows = [
                json.loads(line)
                for line in args.history_jsonl_out.read_text(encoding="utf-8").splitlines()
            ]
            run_status_command = module.build_run_status_command(args, repo=repo)
            monitor_command = module.build_monitor_command(args, repo=repo)

        self.assertEqual(state["status"], "waiting_for_step")
        self.assertFalse(state["milestone_ready"])
        self.assertEqual(state["milestone_step"], 6144)
        self.assertEqual(state["milestone_steps_until"], 128)
        self.assertEqual(state["process_status"], "alive")
        self.assertEqual(state["log_latest_step"], 6016)
        self.assertEqual(exit_code, 0)
        self.assertEqual(state["row_type"], "hf_gpt2_finetune_milestone_capture")
        self.assertEqual(state_written["status"], "waiting_for_step")
        self.assertEqual(history_rows[-1]["iteration"], 1)
        self.assertIn("--max-steps", run_status_command)
        self.assertIn("--run-status-history-jsonl", monitor_command)
        self.assertIn("--milestone-step", monitor_command)
        self.assertIn("milestone_ready=false", module.state_line(state))
        self.assertEqual(state["next_action"], "keep_watching")

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
        previous_loss = None
        for row in rows:
            if row["event"] != "log":
                continue
            frame = hf_ft.hf_gpt2_finetune_training_telemetry_frame(
                row["event"],
                logs=row["metrics"],
                state=types.SimpleNamespace(
                    global_step=row["global_step"],
                    max_steps=5,
                    epoch=None,
                ),
                previous_loss=previous_loss,
            )
            previous_loss = frame["loss"]
            row["training_telemetry"] = frame
            row["telemetry"] = frame["telemetry"]
            row["desire"] = frame["desire"]
            row["psi"] = frame["psi"]
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
        self.assertEqual(summary["trace_training_telemetry_count"], 2)
        self.assertIsNotNone(summary["trace_last_desire_pressure"])
        self.assertIsNotNone(summary["trace_last_psi_total"])
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
                    inference_distortion_handoff={
                        "status": "ok",
                        "recommended_probe": "distort-002",
                        "recommended_effect_score": 0.88,
                        "recommended_risk_score": 0.21,
                        "recommended_api_compatibility_score": 0.84,
                        "desire_pressure": 0.8,
                        "psi_total": 0.7,
                        "include_penalties": True,
                        "api_request_dropped_key_count": 2,
                        "api_request_retry_dropped_key_count": 1,
                        "recommended_processor_kwargs": {
                            "repression_strength": 1.7,
                            "ngram_repression_strength": 0.9,
                        },
                    },
                    training_telemetry=True,
                    desire_gain=1.2,
                    psi_gain=0.8,
                )
            callback.on_train_begin(args, state, control)
            callback.on_log(args, state, control, logs={"loss": 2.0})
            self.assertFalse(getattr(control, "should_training_stop", False))
            callback.on_log(
                args,
                state,
                control,
                logs={"loss": 2.0e6, "grad_norm": float("nan")},
            )
            rows = hf_ft.load_hf_gpt2_finetune_trainer_trace(path)
            summary = hf_ft.summarize_hf_gpt2_finetune_trainer_trace(rows)

        self.assertEqual([row["event"] for row in rows], ["train_begin", "log", "log"])
        self.assertEqual(rows[0]["run_id"], "fake-run")
        self.assertEqual(
            rows[0]["inference_distortion_handoff"]["recommended_probe"],
            "distort-002",
        )
        self.assertEqual(rows[1]["metrics"], {"loss": 2.0})
        self.assertIn("training_telemetry", rows[1])
        self.assertIn("telemetry", rows[1])
        self.assertIn("desire", rows[1])
        self.assertIn("psi", rows[1])
        self.assertIn("hf_ft.psi.total", rows[1]["telemetry"])
        self.assertTrue(getattr(control, "should_training_stop", False))
        self.assertEqual(
            rows[2]["training_loss_guard"]["status"],
            "stop_requested",
        )
        guard_kinds = {
            issue["kind"] for issue in rows[2]["training_loss_guard"]["issues"]
        }
        self.assertIn("loss_exceeds_threshold", guard_kinds)
        self.assertIn("nonfinite_metric", guard_kinds)
        self.assertEqual(
            rows[1]["telemetry"]["hf_ft.inference_distortion.desire_pressure"],
            0.8,
        )
        self.assertEqual(
            rows[1]["training_telemetry"]["inference_distortion_handoff"][
                "recommended_probe"
            ],
            "distort-002",
        )
        self.assertEqual(summary["trace_training_telemetry_count"], 3)
        self.assertEqual(summary["trace_inference_distortion_telemetry_count"], 3)
        self.assertEqual(
            summary["trace_last_inference_distortion_desire_pressure"],
            0.8,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_effect_score"],
            0.88,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_risk_score"],
            0.21,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_api_compatibility_score"],
            0.84,
        )
        self.assertEqual(
            summary[
                "trace_last_inference_distortion_api_request_dropped_key_count"
            ],
            2,
        )
        self.assertEqual(
            summary[
                "trace_last_inference_distortion_api_request_retry_dropped_key_count"
            ],
            1,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_logits_repression_strength"],
            1.7,
        )
        self.assertEqual(
            summary[
                "trace_last_inference_distortion_logits_ngram_repression_strength"
            ],
            0.9,
        )
        self.assertEqual(
            summary["trace_last_inference_distortion_include_penalties"],
            1.0,
        )
        self.assertIsNotNone(summary["trace_last_desire_pressure"])
        self.assertIsNotNone(summary["trace_last_psi_total"])
        self.assertEqual(callback.event_count, 3)

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
        self.assertIs(
            st.hf_gpt2_finetune_disk_headroom_plan,
            hf_ft.hf_gpt2_finetune_disk_headroom_plan,
        )
        self.assertIn("hf_ft", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_file_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_scan_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_dataset_fit_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_disk_headroom_plan", st.__all__)
        self.assertIn("hf_gpt2_finetune_eval_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_generation_curve_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_generation_curve_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_generation_report", st.__all__)
        self.assertIn(
            "hf_gpt2_finetune_inference_distortion_handoff_report",
            st.__all__,
        )
        self.assertIn(
            "hf_gpt2_finetune_inference_distortion_handoff_lines",
            st.__all__,
        )
        self.assertIn(
            "hf_gpt2_finetune_inference_distortion_runtime_plan",
            st.__all__,
        )
        self.assertIn(
            "hf_gpt2_finetune_inference_distortion_runtime_adapter",
            st.__all__,
        )
        self.assertIn(
            "hf_gpt2_finetune_inference_distortion_request_kwargs",
            st.__all__,
        )
        self.assertIn("hf_gpt2_finetune_milestone_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_milestone_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_monitor_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_monitor_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_milestone_capture_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_milestone_capture_report", st.__all__)
        self.assertIn(
            "hf_gpt2_finetune_milestone_handoff_execution_lines",
            st.__all__,
        )
        self.assertIn(
            "hf_gpt2_finetune_milestone_handoff_execution_report",
            st.__all__,
        )
        self.assertIn("hf_gpt2_finetune_milestone_handoff_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_milestone_handoff_report", st.__all__)
        self.assertIn(
            "hf_gpt2_finetune_milestone_runtime_from_run_dir_report",
            st.__all__,
        )
        self.assertIn("hf_gpt2_finetune_milestone_runtime_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_milestone_runtime_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_milestone_runtime_sources", st.__all__)
        self.assertIn("hf_gpt2_finetune_preflight_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_scale_up_command", st.__all__)
        self.assertIn("hf_gpt2_finetune_scale_up_preflight_lines", st.__all__)
        self.assertIn("hf_gpt2_finetune_scale_up_preflight_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_training_telemetry_frame", st.__all__)
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
            st.hf_gpt2_finetune_generation_curve_report,
            hf_ft.hf_gpt2_finetune_generation_curve_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_generation_curve_lines,
            hf_ft.hf_gpt2_finetune_generation_curve_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_inference_distortion_handoff_report,
            hf_ft.hf_gpt2_finetune_inference_distortion_handoff_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_inference_distortion_handoff_lines,
            hf_ft.hf_gpt2_finetune_inference_distortion_handoff_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_inference_distortion_runtime_plan,
            hf_ft.hf_gpt2_finetune_inference_distortion_runtime_plan,
        )
        self.assertIs(
            st.hf_gpt2_finetune_inference_distortion_runtime_adapter,
            hf_ft.hf_gpt2_finetune_inference_distortion_runtime_adapter,
        )
        self.assertIs(
            st.hf_gpt2_finetune_inference_distortion_request_kwargs,
            hf_ft.hf_gpt2_finetune_inference_distortion_request_kwargs,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_lines,
            hf_ft.hf_gpt2_finetune_milestone_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_report,
            hf_ft.hf_gpt2_finetune_milestone_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_monitor_lines,
            st.hf_ft_status.hf_gpt2_finetune_monitor_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_monitor_report,
            st.hf_ft_status.hf_gpt2_finetune_monitor_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_capture_lines,
            st.hf_ft_status.hf_gpt2_finetune_milestone_capture_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_capture_report,
            st.hf_ft_status.hf_gpt2_finetune_milestone_capture_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_handoff_lines,
            st.hf_ft_status.hf_gpt2_finetune_milestone_handoff_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_handoff_report,
            st.hf_ft_status.hf_gpt2_finetune_milestone_handoff_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_handoff_execution_lines,
            st.hf_ft_status.hf_gpt2_finetune_milestone_handoff_execution_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_handoff_execution_report,
            st.hf_ft_status.hf_gpt2_finetune_milestone_handoff_execution_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_runtime_lines,
            st.hf_ft_status.hf_gpt2_finetune_milestone_runtime_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_runtime_report,
            st.hf_ft_status.hf_gpt2_finetune_milestone_runtime_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_runtime_from_run_dir_report,
            st.hf_ft_status.hf_gpt2_finetune_milestone_runtime_from_run_dir_report,
        )
        self.assertIs(
            st.hf_gpt2_finetune_milestone_runtime_sources,
            st.hf_ft_status.hf_gpt2_finetune_milestone_runtime_sources,
        )
        self.assertIs(
            st.hf_gpt2_finetune_training_telemetry_frame,
            hf_ft.hf_gpt2_finetune_training_telemetry_frame,
        )
        self.assertIs(
            st.hf_gpt2_finetune_scale_up_command,
            hf_ft.hf_gpt2_finetune_scale_up_command,
        )
        self.assertIs(
            st.hf_gpt2_finetune_scale_up_preflight_lines,
            hf_ft.hf_gpt2_finetune_scale_up_preflight_lines,
        )
        self.assertIs(
            st.hf_gpt2_finetune_scale_up_preflight_report,
            hf_ft.hf_gpt2_finetune_scale_up_preflight_report,
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
