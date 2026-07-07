from __future__ import annotations

import importlib.util
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


def load_bridge_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_finetune_bridge_test",
        EXAMPLE_PATH,
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
        self.assertIn("hf_ft", st.__all__)
        self.assertIn("hf_gpt2_finetune_corpus_file_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_preflight_report", st.__all__)
        self.assertIn("hf_gpt2_finetune_trainer_trace_callback", st.__all__)
        self.assertIs(
            st.summarize_hf_gpt2_finetune_trainer_trace,
            hf_ft.summarize_hf_gpt2_finetune_trainer_trace,
        )


if __name__ == "__main__":
    unittest.main()
