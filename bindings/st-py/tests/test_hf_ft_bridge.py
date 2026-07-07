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
SCALE_UP_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "hf_gpt2_finetune_scale_up.py"
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
        self.assertTrue(
            any(
                "hf_gpt2_ft_scale_up_preflight status=ready" in line
                for line in direct_preflight_lines
            )
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
            ok_report = module._disk_report(output_dir, min_free_gb=0.0)
            blocked_report = module._disk_report(output_dir, min_free_gb=10**9)

        self.assertEqual(ok_report["row_type"], "hf_gpt2_ft_disk_report")
        self.assertEqual(ok_report["path"], str(output_dir))
        self.assertEqual(ok_report["status"], "ok")
        self.assertGreater(ok_report["free_bytes"], 0)
        self.assertTrue(ok_report["meets_min_free"])
        self.assertEqual(blocked_report["status"], "blocked")
        self.assertFalse(blocked_report["meets_min_free"])

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
        self.assertIn("hf_ft", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_file_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_scan_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_dataset_fit_report", st.__all__)
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
