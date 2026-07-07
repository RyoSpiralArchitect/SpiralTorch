from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import spiraltorch as st
from spiraltorch import hf_ft
from spiraltorch.hf_generation import (
    ZSpaceActivationProbeHook,
    ZSpaceRepressionLogitsProcessor,
    build_zspace_activation_probe_hook,
    compare_zspace_inference_distortion_probes,
    load_zspace_inference_distortion_probe,
    load_zspace_inference_distortion_sweep,
    load_zspace_generation_control_sweep,
    summarize_zspace_inference_distortion_probe,
    summarize_zspace_inference_distortion_probe_comparison_lines,
    summarize_zspace_inference_distortion_probe_lines,
    summarize_zspace_inference_distortion_sweep,
    summarize_zspace_inference_distortion_sweep_lines,
    summarize_zspace_generation_control_sweep,
    summarize_zspace_generation_control_sweep_lines,
    zspace_inference_distortion_probe_cli_args,
    zspace_inference_distortion_processor_kwargs,
    zspace_inference_distortion_sweep_cli_args,
    zspace_generation_control_bridge_cli_args,
    zspace_generation_control_processor_kwargs,
    zspace_generation_control_sweep_cli_args,
    zspace_inference_distortion_sweep_report_from_probes,
)

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional test env
    torch = None


SWEEP_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_zspace_generation_control_sweep.py"
)
DISTORTION_PROBE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "zspace_inference_distortion_probe.py"
)
DISTORTION_SWEEP_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "zspace_inference_distortion_sweep.py"
)
DISTORTION_OPENAI_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "zspace_inference_distortion_local_gpt2_openai_sample.json"
)
DISTORTION_GPT5NANO_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "zspace_inference_distortion_local_gpt2_gpt5nano_sample.json"
)


def load_generation_control_sweep_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_zspace_generation_control_sweep_test",
        SWEEP_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_distortion_probe_example():
    spec = importlib.util.spec_from_file_location(
        "zspace_inference_distortion_probe_test",
        DISTORTION_PROBE_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_distortion_sweep_example():
    spec = importlib.util.spec_from_file_location(
        "zspace_inference_distortion_sweep_test",
        DISTORTION_SWEEP_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is not installed")
class ZSpaceRepressionLogitsProcessorTests(unittest.TestCase):
    def test_repression_can_change_greedy_top_token(self) -> None:
        processor = ZSpaceRepressionLogitsProcessor(
            top_k=3,
            curvature=-1.0,
            temperature=1.0,
            entropy_target=1.0,
            min_temperature=0.5,
            max_temperature=2.0,
            repression_window=4,
            repression_strength=2.0,
            last_token_repression=1.0,
            use_native_zspace=False,
        )
        input_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
        scores = torch.tensor([[4.0, 3.5, 1.0]], dtype=torch.float32)

        processed = processor(input_ids, scores)
        report = processor.report()
        aggregate_only = processor.report(limit=0)

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 1)
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["reported_rows"], 1)
        self.assertEqual(report["top_token_changed_count"], 1)
        self.assertEqual(report["reported_top_token_changed_count"], 1)
        self.assertEqual(aggregate_only["calls"], 1)
        self.assertEqual(aggregate_only["reported_rows"], 0)
        self.assertEqual(aggregate_only["rows"], [])
        self.assertEqual(aggregate_only["top_token_changed_count"], 1)
        self.assertEqual(report["rows"][0]["backend"], "math_zspace_softmax")
        self.assertEqual(report["backend"], "math_zspace_softmax")
        self.assertGreater(report["rows"][0]["max_repression"], 0.0)

    def test_softmax_only_records_entropy_without_reordering_greedy(self) -> None:
        processor = ZSpaceRepressionLogitsProcessor(
            top_k=3,
            curvature=-1.0,
            temperature=1.0,
            entropy_target=1.0,
            min_temperature=0.5,
            max_temperature=2.0,
            repression_window=4,
            repression_strength=0.0,
            last_token_repression=0.0,
            use_native_zspace=False,
        )
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        scores = torch.tensor([[4.0, 3.5, 1.0]], dtype=torch.float32)

        processed = processor(input_ids, scores)
        report = processor.report()

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 0)
        self.assertEqual(report["top_token_changed_count"], 0)
        self.assertIsNotNone(report["entropy_min"])
        self.assertIsNotNone(report["temperature_max"])

    def test_ngram_repression_penalizes_phrase_completion(self) -> None:
        processor = ZSpaceRepressionLogitsProcessor(
            top_k=3,
            curvature=-1.0,
            temperature=1.0,
            min_temperature=0.5,
            max_temperature=2.0,
            repression_window=8,
            repression_strength=0.0,
            last_token_repression=0.0,
            ngram_size=3,
            ngram_repression_strength=2.0,
            use_native_zspace=False,
        )
        input_ids = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)
        scores = torch.tensor([[0.0, 0.0, 0.0, 4.0, 3.9]], dtype=torch.float32)

        processed = processor(input_ids, scores)
        report = processor.report()

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 4)
        self.assertEqual(report["top_token_changed_count"], 1)
        self.assertEqual(report["ngram_repressed_token_total"], 1)
        self.assertGreater(report["max_ngram_repression"], 0.0)
        self.assertEqual(report["rows"][0]["ngram_repressed_token_count"], 1)

    def test_generation_report_embeds_zspace_control_payload(self) -> None:
        control = {
            "row_type": "zspace_repression_generation_control",
            "status": "ok",
            "calls": 2,
        }

        report = hf_ft.hf_gpt2_finetune_generation_report(
            stage="after_train",
            prompt="SpiralTorch is",
            generated_text="SpiralTorch is a runtime.",
            generated_continuation_text=" a runtime.",
            generation_method="model.generate+zspace_repression_softmax",
            generation_control=control,
        )

        self.assertEqual(report["generation_control"], control)

    def test_distortion_adapter_feeds_logits_processor_kwargs(self) -> None:
        adapter = st.api_llm_zspace_inference_distortion_adapter(
            desire_pressure=0.9,
            desire_stability=0.35,
            psi_total=0.8,
            distortion_strength=1.0,
            use_native_zspace=False,
        )
        kwargs = zspace_inference_distortion_processor_kwargs(
            adapter,
            top_k=3,
        )
        processor = ZSpaceRepressionLogitsProcessor(**kwargs)
        input_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
        scores = torch.tensor([[4.0, 3.5, 1.0]], dtype=torch.float32)

        processed = processor(input_ids, scores)
        report = processor.report()

        self.assertEqual(int(torch.argmax(processed, dim=-1).item()), 1)
        self.assertGreater(kwargs["repression_strength"], 0.75)
        self.assertEqual(report["top_token_changed_count"], 1)

    def test_activation_probe_hook_records_and_intervenes(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
        hook = ZSpaceActivationProbeHook(
            module_names=["0"],
            intervention_scale=0.5,
        ).attach(model)

        try:
            output = model(torch.tensor([[2.0, 4.0]], dtype=torch.float32))
            report = hook.report()
            aggregate_only = hook.report(limit=0)
        finally:
            hook.close()

        self.assertTrue(torch.allclose(output, torch.tensor([[1.0, 2.0]])))
        self.assertEqual(report["event_count"], 1)
        self.assertEqual(report["reported_event_count"], 1)
        self.assertEqual(aggregate_only["reported_event_count"], 0)
        self.assertEqual(report["events"][0]["module"], "0")
        self.assertTrue(report["events"][0]["intervened"])
        self.assertGreater(report["events"][0]["output_l2"], 0.0)


class ZSpaceGenerationExportTests(unittest.TestCase):
    def test_top_level_exports_generation_processor(self) -> None:
        self.assertIn("ZSpaceRepressionLogitsProcessor", st.__all__)
        self.assertIn("ZSpaceActivationProbeHook", st.__all__)
        self.assertIn("build_zspace_activation_probe_hook", st.__all__)
        self.assertIn("compare_zspace_inference_distortion_probes", st.__all__)
        self.assertIn("build_zspace_repression_logits_processor", st.__all__)
        self.assertIn("build_zspace_softmax_logits_processor", st.__all__)
        self.assertIn(
            "zspace_inference_distortion_sweep_report_from_probes",
            st.__all__,
        )
        self.assertIn("zspace_generation_control_bridge_cli_args", st.__all__)
        self.assertIn("zspace_generation_control_processor_kwargs", st.__all__)
        self.assertIn("zspace_generation_control_sweep_cli_args", st.__all__)
        self.assertIn("load_zspace_generation_control_sweep", st.__all__)
        self.assertIn("summarize_zspace_generation_control_sweep", st.__all__)
        self.assertIn("zspace_inference_distortion_processor_kwargs", st.__all__)
        self.assertIn("load_zspace_inference_distortion_probe", st.__all__)
        self.assertIn("summarize_zspace_inference_distortion_probe", st.__all__)
        self.assertIn(
            "summarize_zspace_inference_distortion_probe_comparison_lines",
            st.__all__,
        )
        self.assertIn("summarize_zspace_inference_distortion_probe_lines", st.__all__)
        self.assertIs(st.ZSpaceRepressionLogitsProcessor, ZSpaceRepressionLogitsProcessor)
        self.assertIs(st.ZSpaceActivationProbeHook, ZSpaceActivationProbeHook)
        self.assertIs(
            st.build_zspace_activation_probe_hook,
            build_zspace_activation_probe_hook,
        )
        self.assertIs(
            st.zspace_generation_control_bridge_cli_args,
            zspace_generation_control_bridge_cli_args,
        )
        self.assertIs(
            st.zspace_inference_distortion_processor_kwargs,
            zspace_inference_distortion_processor_kwargs,
        )
        self.assertIs(
            st.load_zspace_inference_distortion_probe,
            load_zspace_inference_distortion_probe,
        )
        self.assertIs(
            st.zspace_inference_distortion_sweep_report_from_probes,
            zspace_inference_distortion_sweep_report_from_probes,
        )
        self.assertIs(
            st.compare_zspace_inference_distortion_probes,
            compare_zspace_inference_distortion_probes,
        )
        self.assertIs(
            st.summarize_zspace_inference_distortion_probe,
            summarize_zspace_inference_distortion_probe,
        )
        self.assertIs(
            st.summarize_zspace_inference_distortion_probe_comparison_lines,
            summarize_zspace_inference_distortion_probe_comparison_lines,
        )
        self.assertIs(
            st.summarize_zspace_generation_control_sweep,
            summarize_zspace_generation_control_sweep,
        )

    def test_local_gpt2_openai_distortion_sample_is_sanitized(self) -> None:
        sample_text = DISTORTION_OPENAI_SAMPLE_PATH.read_text(encoding="utf-8")
        sample = json.loads(sample_text)

        self.assertEqual(
            sample["row_type"],
            "zspace_inference_distortion_local_hf_api_sample",
        )
        self.assertNotIn("OPENAI_API_KEY", sample_text)
        self.assertNotIn("resp_", sample_text)
        self.assertNotIn("/Users/", sample_text)
        summary = sample["summary"]
        self.assertTrue(summary["local_changed"])
        self.assertEqual(summary["api_request_dropped_key_count"], 2)
        self.assertEqual(
            summary["api_request_dropped_keys"],
            ["frequency_penalty", "presence_penalty"],
        )

    def test_local_gpt2_gpt5nano_distortion_sample_is_sanitized(self) -> None:
        sample_text = DISTORTION_GPT5NANO_SAMPLE_PATH.read_text(encoding="utf-8")
        sample = json.loads(sample_text)

        self.assertEqual(
            sample["row_type"],
            "zspace_inference_distortion_local_hf_gpt5nano_sample",
        )
        self.assertNotIn("OPENAI_API_KEY", sample_text)
        self.assertNotIn("ANTHROPIC_API_KEY", sample_text)
        self.assertNotIn("sk-", sample_text)
        self.assertNotIn("resp_", sample_text)
        self.assertNotIn("/Users/", sample_text)
        summary = sample["summary"]
        self.assertTrue(summary["local_changed"])
        self.assertEqual(summary["api_model"], "gpt-5-nano")
        self.assertEqual(summary["api_empty_text"], 0.0)
        self.assertEqual(summary["api_request_dropped_key_count"], 4)
        self.assertEqual(
            summary["api_request_retry_dropped_keys"],
            ["temperature", "top_p"],
        )
        self.assertEqual(
            sample["runtime"]["api_reasoning_effort"],
            "minimal",
        )
        self.assertEqual(sample["runtime"]["api_text_verbosity"], "low")
        self.assertIn("api_retry_dropped=2", sample["compact_lines"][0])
        self.assertIn("visible output", " ".join(sample["notes"]))

    def test_inference_distortion_probe_summary_flattens_local_and_api(self) -> None:
        report = {
            "row_type": "zspace_inference_distortion_probe",
            "prompt": "SpiralTorch is",
            "adapter": {
                "distortion_energy": 0.62,
                "request": {"temperature": 0.98, "top_p": 0.77},
                "logits_processor_kwargs": {
                    "repression_strength": 1.5,
                    "ngram_repression_strength": 1.1,
                },
                "context_partial": {
                    "telemetry": {
                        "zspace.desire.pressure": 0.8,
                        "zspace.desire.stability": 0.45,
                        "zspace.psi.total": 0.7,
                        "zspace.coherence": 0.45,
                    }
                },
            },
            "local_hf": {
                "status": "ok",
                "changed": True,
                "model": "local-model",
                "baseline_method": "manual_forward_fallback",
                "distorted_method": "manual_forward_fallback+zspace_repression_softmax",
                "baseline_text": "SpiralTorch is baseline language.",
                "distorted_text": "SpiralTorch is distorted geometry.",
                "generation_control": {
                    "status": "ok",
                    "backend": "spiraltorch_zspace_softmax",
                    "calls": 24,
                    "top_token_changed_count": 5,
                    "ngram_repressed_token_total": 1,
                },
                "activation_report": {
                    "status": "ok",
                    "event_count": 64,
                    "reported_event_count": 16,
                    "output_l2_min": 15.0,
                    "output_l2_max": 366.0,
                },
            },
            "api": {
                "provider": "fake",
                "model": "fake-distorted-api",
                "text": "Fake API distortion route.",
                "request_filter": {
                    "dropped_key_count": 2,
                    "dropped_keys": ["frequency_penalty", "presence_penalty"],
                    "retry_dropped_key_count": 1,
                    "retry_dropped_keys": ["temperature"],
                    "sent_keys": ["input", "model", "temperature", "top_p"],
                },
                "telemetry": {
                    "api_llm.total_tokens": 56.0,
                    "api_llm.response_entropy_norm": 0.9,
                    "api_llm.empty_text": 0.0,
                    "zspace.request.temperature": 0.98,
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            loaded = load_zspace_inference_distortion_probe(path)
            summary = summarize_zspace_inference_distortion_probe(path)
            lines = summarize_zspace_inference_distortion_probe_lines(loaded)

        self.assertEqual(loaded["row_type"], "zspace_inference_distortion_probe")
        self.assertEqual(summary["local_status"], "ok")
        self.assertTrue(summary["local_changed"])
        self.assertEqual(summary["generation_control_backend"], "spiraltorch_zspace_softmax")
        self.assertEqual(summary["generation_control_top_token_changed_count"], 5)
        self.assertEqual(summary["activation_event_count"], 64)
        self.assertEqual(summary["api_provider"], "fake")
        self.assertEqual(summary["api_total_tokens"], 56.0)
        self.assertEqual(summary["api_request_dropped_key_count"], 2)
        self.assertEqual(
            summary["api_request_dropped_keys"],
            ["frequency_penalty", "presence_penalty"],
        )
        self.assertEqual(summary["api_request_retry_dropped_key_count"], 1)
        self.assertEqual(summary["api_request_retry_dropped_keys"], ["temperature"])
        self.assertEqual(summary["distortion_energy"], 0.62)
        self.assertGreater(summary["effect_score"], 0.0)
        self.assertGreater(summary["risk_score"], 0.0)
        self.assertTrue(str(summary["probe_path"]).endswith("probe.json"))
        self.assertIn("zspace_inference_distortion_probe", lines[0])
        self.assertIn("top_changes=5", lines[0])
        self.assertIn("api_dropped=2", lines[0])
        self.assertIn("effect=", lines[0])
        self.assertIn("risk=", lines[0])

        report["probe_path"] = "memory-probe.json"
        memory_summary = summarize_zspace_inference_distortion_probe(report)
        self.assertEqual(memory_summary["probe_path"], "memory-probe.json")

    def test_inference_distortion_probe_comparison_ranks_effect(self) -> None:
        weak = {
            "row_type": "zspace_inference_distortion_probe",
            "prompt": "weak",
            "adapter": {
                "distortion_energy": 0.2,
                "request": {"temperature": 0.8, "top_p": 0.9},
                "logits_processor_kwargs": {"repression_strength": 0.8},
            },
            "local_hf": {
                "status": "ok",
                "changed": False,
                "generation_control": {
                    "status": "ok",
                    "backend": "spiraltorch_zspace_softmax",
                    "top_token_changed_count": 0,
                },
                "activation_report": {"status": "unused", "event_count": 0},
            },
            "api": {
                "provider": "fake",
                "text": "",
                "request_filter": {
                    "dropped_key_count": 0,
                    "dropped_keys": [],
                    "sent_keys": ["input", "model"],
                },
                "telemetry": {"api_llm.empty_text": 1.0},
            },
        }
        strong = {
            "row_type": "zspace_inference_distortion_probe",
            "prompt": "strong",
            "adapter": {
                "distortion_energy": 0.62,
                "request": {"temperature": 0.98, "top_p": 0.77},
                "logits_processor_kwargs": {"repression_strength": 1.5},
            },
            "local_hf": {
                "status": "ok",
                "changed": True,
                "generation_control": {
                    "status": "ok",
                    "backend": "spiraltorch_zspace_softmax",
                    "top_token_changed_count": 5,
                },
                "activation_report": {"status": "ok", "event_count": 64},
            },
            "api": {
                "provider": "fake",
                "text": "strong route",
                "request_filter": {
                    "dropped_key_count": 2,
                    "dropped_keys": ["temperature", "top_p"],
                    "retry_dropped_key_count": 2,
                    "retry_dropped_keys": ["temperature", "top_p"],
                    "sent_keys": ["input", "model", "reasoning", "text"],
                },
                "telemetry": {"api_llm.empty_text": 0.0},
            },
        }

        comparison = compare_zspace_inference_distortion_probes(
            {"weak": weak, "strong": strong},
            top_n=2,
        )
        lines = summarize_zspace_inference_distortion_probe_comparison_lines(
            comparison
        )

        self.assertEqual(
            comparison["row_type"],
            "zspace_inference_distortion_probe_comparison",
        )
        self.assertEqual(comparison["probe_count"], 2)
        self.assertEqual(comparison["recommended_probe"], "strong")
        self.assertEqual(comparison["local_changed_count"], 1)
        self.assertEqual(comparison["activation_observed_count"], 1)
        self.assertEqual(comparison["api_visible_text_count"], 1)
        self.assertEqual(comparison["api_empty_text_count"], 1)
        self.assertEqual(comparison["api_retry_dropped_probe_count"], 1)
        self.assertEqual(comparison["api_retry_dropped_key_total"], 2.0)
        self.assertEqual(
            comparison["api_retry_dropped_keys"],
            ["temperature", "top_p"],
        )
        self.assertEqual(comparison["top_probes"][0]["label"], "strong")
        self.assertGreater(
            comparison["top_probes"][0]["effect_score"],
            comparison["top_probes"][1]["effect_score"],
        )
        self.assertIn("recommended=strong", lines[0])
        self.assertIn("api_visible=1", lines[0])
        self.assertIn("api_empty=1", lines[0])
        self.assertIn("api_retry_dropped=1", lines[0])
        self.assertIn("label=strong", lines[1])
        self.assertIn("api_retry_dropped=2", lines[1])

    def test_inference_distortion_sweep_dry_run_writes_plan(self) -> None:
        module = load_distortion_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--out-dir",
                    str(out_dir),
                    "--prompt",
                    "SpiralTorch sweep",
                    "--desire-pressure-values",
                    "0.4,0.8",
                    "--psi-total-values",
                    "0.5",
                ]
            )
            runs = module.build_sweep_runs(args)
            report = module.run_sweep(args)
            plan = json.loads((out_dir / "sweep-plan.json").read_text())

        self.assertEqual(len(runs), 2)
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["run_count"], 2)
        self.assertEqual(plan["run_count"], 2)
        self.assertIn("dp0p4", runs[0]["name"])
        self.assertIn("dp0p8", runs[1]["name"])
        self.assertEqual(plan["runtime"]["api_provider"], "fake")
        self.assertEqual(plan["execution"]["resume_existing"], False)
        self.assertEqual(report["attempted_run_count"], 0)
        self.assertEqual(report["completed_run_count"], 0)
        self.assertEqual(report["skipped_run_count"], 2)

    def test_inference_distortion_probe_generate_compat_drops_batch_size(self) -> None:
        module = load_distortion_probe_example()

        class DummyModel:
            def _prepare_special_tokens(self, generation_config, device=None):
                return {"generation_config": generation_config, "device": device}

        model = DummyModel()
        with self.assertRaises(TypeError):
            model._prepare_special_tokens("cfg", batch_size=1)

        with module._prepare_special_tokens_batch_size_compat(model) as installed:
            self.assertTrue(installed)
            self.assertEqual(
                model._prepare_special_tokens("cfg", device="cpu", batch_size=1),
                {"generation_config": "cfg", "device": "cpu"},
            )

        with self.assertRaises(TypeError):
            model._prepare_special_tokens("cfg", batch_size=1)

    def test_inference_distortion_sweep_runs_fake_api_and_compares(self) -> None:
        module = load_distortion_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--prompt",
                    "SpiralTorch sweep",
                    "--desire-pressure-values",
                    "0.4,0.8",
                    "--psi-total-values",
                    "0.5",
                    "--api-reasoning-effort",
                    "minimal",
                    "--api-text-verbosity",
                    "low",
                    "--top-n",
                    "2",
                ]
            )
            report = module.run_sweep(args)
            stored_report = json.loads((out_dir / "sweep-report.json").read_text())
            first_probe_exists = Path(stored_report["runs"][0]["probe_path"]).exists()
            markdown = (out_dir / "sweep-report.md").read_text()
            loaded_sweep = load_zspace_inference_distortion_sweep(
                out_dir / "sweep-report.json"
            )
            sweep_summary = summarize_zspace_inference_distortion_sweep(
                out_dir / "sweep-report.json",
                top_n=2,
            )
            sweep_lines = summarize_zspace_inference_distortion_sweep_lines(
                loaded_sweep,
                top_n=1,
            )
            probe_module = load_distortion_probe_example()
            replay_args = probe_module.parse_args(
                [
                    "--from-sweep-report",
                    str(out_dir / "sweep-report.json"),
                ]
            )
            override_args = probe_module.parse_args(
                [
                    "--from-sweep-report",
                    str(out_dir / "sweep-report.json"),
                    "--desire-pressure",
                    "0.2",
                ]
            )
            replay_path = out_dir / "replay-probe.json"
            with contextlib.redirect_stdout(io.StringIO()):
                replay_exit = probe_module.main(
                    [
                        "--from-sweep-report",
                        str(out_dir / "sweep-report.json"),
                        "--out",
                        str(replay_path),
                    ]
                )
            replay_report = json.loads(replay_path.read_text())

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_run_count"], 2)
        self.assertEqual(stored_report["comparison"]["probe_count"], 2)
        self.assertEqual(len(stored_report["comparison"]["top_probes"]), 2)
        self.assertIn("summary", stored_report)
        self.assertIn("recommendation", stored_report)
        self.assertIn("recommended_commands", stored_report)
        self.assertIn("markdown_path", stored_report)
        self.assertIn("summary_lines", stored_report)
        self.assertTrue(first_probe_exists)
        self.assertTrue(stored_report["runs"][0]["summary"]["probe_path"])
        self.assertEqual(
            stored_report["comparison"]["top_probes"][0]["api_provider"],
            "fake",
        )
        self.assertEqual(sweep_summary["row_type"], "zspace_inference_distortion_sweep_summary")
        self.assertEqual(sweep_summary["completed_run_count"], 2)
        self.assertIn("--desire-pressure", sweep_summary["recommended_probe_cli_args"])
        self.assertIn("--desire-pressure-values", sweep_summary["recommended_sweep_cli_args"])
        self.assertEqual(
            zspace_inference_distortion_probe_cli_args(
                sweep_summary["recommended_config"]
            ),
            sweep_summary["recommended_probe_cli_args"],
        )
        self.assertEqual(
            zspace_inference_distortion_sweep_cli_args(
                sweep_summary["recommended_config"]
            ),
            sweep_summary["recommended_sweep_cli_args"],
        )
        self.assertIn("zspace_inference_distortion_sweep", sweep_lines[0])
        self.assertIn("Z-Space Inference Distortion Sweep", markdown)
        self.assertIn("Single-probe replay", markdown)
        self.assertIn("Focused sweep replay", markdown)
        self.assertIn("--api-reasoning-effort minimal", stored_report["recommended_commands"]["probe"])
        self.assertIn("--api-text-verbosity low", stored_report["recommended_commands"]["sweep"])
        self.assertEqual(replay_args.prompt, "SpiralTorch sweep")
        self.assertEqual(replay_args.api_reasoning_effort, "minimal")
        self.assertEqual(replay_args.api_text_verbosity, "low")
        self.assertEqual(
            replay_args.desire_pressure,
            sweep_summary["recommended_config"]["desire_pressure"],
        )
        self.assertEqual(override_args.desire_pressure, 0.2)
        self.assertEqual(replay_exit, 0)
        self.assertEqual(
            replay_report["handoff"]["recommended_probe"],
            sweep_summary["recommended_probe"],
        )
        self.assertEqual(
            replay_report["config"]["desire_pressure"],
            sweep_summary["recommended_config"]["desire_pressure"],
        )
        self.assertEqual(replay_report["runtime"]["api_provider"], "fake")
        self.assertEqual(replay_report["runtime"]["api_reasoning_effort"], "minimal")
        self.assertEqual(replay_report["runtime"]["api_text_verbosity"], "low")

    def test_inference_distortion_sweep_reuses_existing_probe(self) -> None:
        module = load_distortion_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sweep"
            base_args = module.parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--prompt",
                    "SpiralTorch resume sweep",
                    "--desire-pressure-values",
                    "0.4",
                    "--psi-total-values",
                    "0.5",
                    "--top-n",
                    "1",
                ]
            )
            initial_report = module.run_sweep(base_args)
            original_run_probe = module._run_probe

            def _fail_if_called(*_args, **_kwargs):
                raise AssertionError("resume-existing should not rerun probes")

            try:
                module._run_probe = _fail_if_called
                resume_args = module.parse_args(
                    [
                        "--resume-existing",
                        "--out-dir",
                        str(out_dir),
                        "--prompt",
                        "SpiralTorch resume sweep",
                        "--desire-pressure-values",
                        "0.4",
                        "--psi-total-values",
                        "0.5",
                        "--top-n",
                        "1",
                    ]
                )
                resumed_report = module.run_sweep(resume_args)
                report_only_args = module.parse_args(
                    [
                        "--report-only",
                        "--out-dir",
                        str(out_dir),
                        "--prompt",
                        "SpiralTorch resume sweep",
                        "--desire-pressure-values",
                        "0.4",
                        "--psi-total-values",
                        "0.5",
                        "--top-n",
                        "1",
                    ]
                )
                report_only = module.run_sweep(report_only_args)
                stale_args = module.parse_args(
                    [
                        "--report-only",
                        "--out-dir",
                        str(out_dir),
                        "--prompt",
                        "Changed prompt",
                        "--desire-pressure-values",
                        "0.4",
                        "--psi-total-values",
                        "0.5",
                        "--top-n",
                        "1",
                    ]
                )
                stale_report = module.run_sweep(stale_args)
            finally:
                module._run_probe = original_run_probe

        self.assertEqual(initial_report["status"], "complete")
        self.assertEqual(resumed_report["status"], "complete")
        self.assertEqual(resumed_report["attempted_run_count"], 0)
        self.assertEqual(resumed_report["reused_run_count"], 1)
        self.assertEqual(resumed_report["completed_run_count"], 1)
        self.assertEqual(resumed_report["runs"][0]["status"], "reused")
        self.assertEqual(report_only["status"], "reported")
        self.assertEqual(report_only["reported_run_count"], 1)
        self.assertEqual(report_only["attempted_run_count"], 0)
        self.assertEqual(report_only["comparison"]["probe_count"], 1)
        self.assertEqual(stale_report["status"], "partial")
        self.assertEqual(stale_report["stale_run_count"], 1)
        self.assertEqual(stale_report["comparison"]["probe_count"], 0)

    def test_inference_distortion_sweep_promotes_saved_probe(self) -> None:
        module = load_distortion_sweep_example()
        probe_module = load_distortion_probe_example()
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_args = module.parse_args(
                [
                    "--out-dir",
                    str(source_dir),
                    "--prompt",
                    "SpiralTorch probe promotion",
                    "--desire-pressure-values",
                    "0.8",
                    "--psi-total-values",
                    "0.7",
                    "--api-provider",
                    "fake",
                    "--top-n",
                    "1",
                ]
            )
            source_report = module.run_sweep(source_args)
            probe_path = Path(source_report["runs"][0]["probe_path"])
            probe_payload = json.loads(probe_path.read_text())
            probe_payload["api"]["request_filter"] = {
                "dropped_key_count": 1,
                "dropped_keys": ["frequency_penalty"],
                "retry_dropped_key_count": 1,
                "retry_dropped_keys": ["temperature"],
                "sent_keys": ["input", "model", "temperature", "top_p"],
            }
            probe_path.write_text(json.dumps(probe_payload, indent=2) + "\n")

            import_dir = Path(tmp) / "imported"
            import_args = module.parse_args(
                [
                    "--from-probe",
                    str(probe_path),
                    "--from-probe-label",
                    "saved-live-probe",
                    "--out-dir",
                    str(import_dir),
                    "--top-n",
                    "1",
                ]
            )
            imported_report = module.run_sweep(import_args)
            imported_summary = summarize_zspace_inference_distortion_sweep(
                import_dir / "sweep-report.json",
                top_n=1,
            )
            replay_args = probe_module.parse_args(
                [
                    "--from-sweep-report",
                    str(import_dir / "sweep-report.json"),
                ]
            )

        self.assertEqual(imported_report["status"], "reported")
        self.assertEqual(imported_report["completed_run_count"], 1)
        self.assertEqual(imported_report["reported_run_count"], 1)
        self.assertEqual(imported_report["attempted_run_count"], 0)
        self.assertEqual(imported_report["prompt"], "SpiralTorch probe promotion")
        self.assertEqual(imported_report["runtime"]["api_provider"], "fake")
        self.assertEqual(
            imported_summary["recommended_probe"],
            "saved-live-probe",
        )
        self.assertEqual(
            imported_summary["recommended_probe_path"],
            str(probe_path),
        )
        self.assertEqual(
            imported_summary["recommended_api_request_dropped_keys"],
            ["frequency_penalty"],
        )
        self.assertEqual(
            imported_summary["recommended_api_request_dropped_key_count"],
            1,
        )
        self.assertEqual(
            imported_summary["recommended_api_request_retry_dropped_keys"],
            ["temperature"],
        )
        self.assertEqual(
            imported_summary["recommended_api_request_retry_dropped_key_count"],
            1,
        )
        self.assertEqual(replay_args.prompt, "SpiralTorch probe promotion")
        self.assertEqual(replay_args.api_provider, "fake")

    def test_inference_distortion_probe_helper_promotes_inline_probe(self) -> None:
        probe = {
            "row_type": "zspace_inference_distortion_probe",
            "name": "inline-probe",
            "prompt": "SpiralTorch inline probe",
            "runtime": {
                "api_provider": "fake",
                "api_model": "fake-distorted-api",
                "local_model": "runs/gpt2-small-zspace-ft",
            },
            "config": {
                "desire_pressure": 0.82,
                "desire_stability": 0.4,
                "psi_total": 0.72,
                "coherence": 0.5,
                "distortion_strength": 1.1,
                "base_temperature": 0.7,
                "base_top_p": 0.95,
                "include_penalties": True,
            },
            "adapter": {
                "request": {"temperature": 1.05, "top_p": 0.75},
                "activation_hook": {
                    "name_contains": ["attn"],
                    "intervention_scale": 0.9,
                },
                "logits_processor_kwargs": {
                    "repression_strength": 1.6,
                    "ngram_repression_strength": 0.8,
                },
            },
            "local_hf": {
                "status": "ok",
                "changed": True,
                "generation_control": {
                    "status": "ok",
                    "top_token_changed_count": 4,
                },
                "activation_report": {"status": "ok", "event_count": 8},
            },
            "api": {
                "provider": "fake",
                "model": "fake-distorted-api",
                "text": "fake distorted response",
                "request_filter": {
                    "dropped_key_count": 1,
                    "dropped_keys": ["presence_penalty"],
                    "retry_dropped_key_count": 1,
                    "retry_dropped_keys": ["top_p"],
                    "sent_keys": ["temperature", "top_p"],
                },
            },
        }

        report = zspace_inference_distortion_sweep_report_from_probes(
            probe,
            labels=["inline-live-probe"],
            top_n=1,
        )
        summary = summarize_zspace_inference_distortion_sweep(report, top_n=1)

        self.assertEqual(report["status"], "reported")
        self.assertEqual(report["completed_run_count"], 1)
        self.assertEqual(summary["recommended_probe"], "inline-live-probe")
        self.assertEqual(summary["recommended_config"]["desire_pressure"], 0.82)
        self.assertEqual(summary["recommended_request"]["temperature"], 1.05)
        self.assertEqual(
            summary["recommended_processor_kwargs"]["repression_strength"],
            1.6,
        )
        self.assertEqual(
            summary["recommended_activation_hook"]["name_contains"],
            ["attn"],
        )
        self.assertEqual(
            summary["recommended_api_request_dropped_keys"],
            ["presence_penalty"],
        )
        self.assertEqual(
            summary["recommended_api_request_retry_dropped_keys"],
            ["top_p"],
        )


class ZSpaceGenerationControlSweepExampleTests(unittest.TestCase):
    def test_summarize_generation_control_sweep_ranks_loop_scores(self) -> None:
        baseline = {
            "name": "baseline-greedy",
            "kind": "baseline",
            "status": "ok",
            "config": {},
            "generation": hf_ft.hf_gpt2_finetune_generation_report(
                stage="baseline",
                prompt="SpiralTorch is",
                generated_text="SpiralTorch is wrapper wrapper wrapper",
                generated_continuation_text=" wrapper wrapper wrapper",
                input_token_count=3,
                output_token_count=6,
            ),
            "repetition": {
                "loop_score": 3.0,
                "unique_word_ratio": 0.33,
                "repeated_ngram_total": 2,
                "max_ngram_repetition": 2,
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
                prompt="SpiralTorch is",
                generated_text="SpiralTorch is geometry runtime",
                generated_continuation_text=" geometry runtime",
                input_token_count=3,
                output_token_count=5,
                generation_control={
                    "status": "ok",
                    "calls": 2,
                    "backend": "spiraltorch_zspace_softmax",
                    "top_token_changed_count": 1,
                    "temperature_min": 0.7,
                    "temperature_max": 1.1,
                },
            ),
            "repetition": {
                "loop_score": 0.0,
                "unique_word_ratio": 1.0,
                "repeated_ngram_total": 0,
                "max_ngram_repetition": 1,
            },
        }
        report = {
            "row_type": "hf_gpt2_zspace_generation_control_sweep",
            "status": "complete",
            "dry_run": False,
            "model_name": "gpt2",
            "prompt": "SpiralTorch is",
            "run_count": 2,
            "runs": [baseline, controlled],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            loaded = load_zspace_generation_control_sweep(path)
            summary = summarize_zspace_generation_control_sweep(path, top_n=1)
            lines = summarize_zspace_generation_control_sweep_lines(loaded, top_n=1)

        self.assertEqual(loaded["run_count"], 2)
        self.assertEqual(summary["completed_run_count"], 2)
        self.assertEqual(summary["changed_from_baseline_count"], 1)
        self.assertEqual(summary["baseline_loop_score"], 3.0)
        self.assertEqual(summary["best_loop_score_run"], "zt3-rs1p25-lr0-k64")
        self.assertEqual(summary["recommended_run"], "zt3-rs1p25-lr0-k64")
        self.assertEqual(
            summary["recommendation_reason"],
            "lowest_loop_score_with_baseline_reduction",
        )
        self.assertEqual(summary["recommended_config"]["repression_strength"], 1.25)
        self.assertEqual(summary["recommended_config"]["entropy_target"], 3.0)
        self.assertEqual(summary["recommended_config"]["ngram_size"], 3)
        self.assertEqual(summary["recommended_config"]["ngram_window"], 96)
        self.assertEqual(
            summary["recommended_config"]["ngram_repression_strength"],
            0.75,
        )
        self.assertEqual(summary["recommended_processor_kwargs"]["top_k"], 64)
        self.assertEqual(
            summary["recommended_processor_kwargs"]["min_temperature"],
            0.7,
        )
        self.assertEqual(summary["recommended_processor_kwargs"]["ngram_decay"], 0.9)
        self.assertEqual(summary["recommended_processor_kwargs"]["ngram_window"], 96)
        self.assertIn("--repression-strength-values", summary["recommended_sweep_cli_args"])
        self.assertIn("1.25", summary["recommended_sweep_cli_args"])
        self.assertIn("--ngram-size-values", summary["recommended_sweep_cli_args"])
        self.assertIn("--ngram-window-values", summary["recommended_sweep_cli_args"])
        self.assertIn("--ngram-repression-strength-values", summary["recommended_sweep_cli_args"])
        self.assertEqual(summary["recommended_cli_args"], summary["recommended_sweep_cli_args"])
        self.assertIn("--generation-zspace-softmax", summary["recommended_bridge_cli_args"])
        self.assertIn("--generation-repression-strength", summary["recommended_bridge_cli_args"])
        self.assertIn("--generation-ngram-size", summary["recommended_bridge_cli_args"])
        self.assertIn("--generation-ngram-window", summary["recommended_bridge_cli_args"])
        self.assertIn(
            "--generation-ngram-repression-strength",
            summary["recommended_bridge_cli_args"],
        )
        self.assertIn("1.25", summary["recommended_bridge_cli_args"])
        self.assertEqual(
            zspace_generation_control_processor_kwargs(summary["recommended_config"]),
            summary["recommended_processor_kwargs"],
        )
        self.assertEqual(
            zspace_generation_control_sweep_cli_args(summary["recommended_config"]),
            summary["recommended_sweep_cli_args"],
        )
        self.assertEqual(
            zspace_generation_control_bridge_cli_args(summary["recommended_config"]),
            summary["recommended_bridge_cli_args"],
        )
        self.assertEqual(summary["best_loop_score_delta_from_baseline"], -3.0)
        self.assertEqual(summary["best_loop_score_reduction_ratio"], 1.0)
        self.assertEqual(summary["top_runs"][0]["loop_score"], 0.0)
        self.assertEqual(
            summary["top_runs"][0]["loop_score_delta_from_baseline"],
            -3.0,
        )
        self.assertEqual(
            summary["top_runs"][0]["control_backend"],
            "spiraltorch_zspace_softmax",
        )
        self.assertIn("best=zt3-rs1p25-lr0-k64", lines[0])
        self.assertIn("recommend=zt3-rs1p25-lr0-k64", lines[0])
        self.assertIn("loop_delta=-3.0", lines[0])
        self.assertIn("top_changes=1", lines[1])
        self.assertIn("ngram=3/96/0.75", lines[1])

    def test_dry_run_builds_control_grid_without_loading_model(self) -> None:
        module = load_generation_control_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "control-sweep.json"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--prompt",
                    "SpiralTorch is",
                    "--out",
                    str(out_path),
                    "--zspace-entropy-target-values",
                    "none,3.0",
                    "--repression-strength-values",
                    "0.0,1.25",
                    "--last-token-repression-values",
                    "0.0",
                    "--report-limit",
                    "2",
                ]
            )
            runs = module.build_control_runs(args)
            report = module.run_sweep(args)

        self.assertEqual(len(runs), 5)
        self.assertEqual(runs[0]["kind"], "baseline")
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["run_count"], 5)
        self.assertEqual(report["summary"]["completed_run_count"], 0)
        self.assertTrue(any(str(row["name"]).startswith("zt3") for row in runs))

    def test_repetition_report_scores_repeated_ngrams(self) -> None:
        module = load_generation_control_sweep_example()
        loop = module.text_repetition_report("a b c a b c a b c")
        clean = module.text_repetition_report("a b c d e f")

        self.assertGreater(loop["loop_score"], clean["loop_score"])
        self.assertEqual(loop["max_ngram_repetition"], 3)


if __name__ == "__main__":
    unittest.main()
