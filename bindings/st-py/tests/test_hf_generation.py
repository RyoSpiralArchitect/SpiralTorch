from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import spiraltorch as st
from spiraltorch import hf_ft
from spiraltorch import hf_cli
from spiraltorch import hf_peft
from spiraltorch.hf_generation import (
    ZSpaceActivationProbeHook,
    ZSpaceCheckpointPromptSpec,
    ZSpaceCheckpointSweepJob,
    ZSpaceRepressionLogitsProcessor,
    build_zspace_activation_probe_hook,
    compare_zspace_inference_distortion_probes,
    compare_zspace_generation_control_sweeps,
    default_zspace_checkpoint_generation_prompts,
    load_zspace_inference_distortion_probe,
    load_zspace_inference_distortion_sweep,
    load_zspace_generation_control_sweep,
    summarize_zspace_inference_distortion_probe,
    summarize_zspace_inference_distortion_probe_comparison_lines,
    summarize_zspace_inference_distortion_probe_lines,
    summarize_zspace_inference_distortion_sweep,
    summarize_zspace_inference_distortion_sweep_lines,
    summarize_zspace_generation_control_sweep,
    summarize_zspace_generation_control_sweep_comparison_lines,
    summarize_zspace_generation_control_sweep_lines,
    zspace_inference_distortion_geometry_probe,
    zspace_inference_distortion_probe_cli_args,
    zspace_inference_distortion_probe_report,
    zspace_inference_distortion_processor_kwargs,
    zspace_inference_distortion_runtime_cli_args,
    zspace_inference_distortion_runtime_plan,
    zspace_inference_distortion_runtime_preflight,
    zspace_inference_distortion_sweep_cli_args,
    zspace_generation_control_bridge_cli_args,
    zspace_generation_control_profile_config,
    zspace_generation_control_processor_kwargs,
    zspace_generation_control_sweep_cli_args,
    zspace_checkpoint_generation_control_compare_command,
    zspace_checkpoint_generation_control_jobs,
    zspace_checkpoint_generation_control_report,
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
GENERIC_SWEEP_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_zspace_generation_control_sweep.py"
)
SWEEP_COMPARE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_zspace_generation_control_compare.py"
)
GENERIC_SWEEP_COMPARE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_zspace_generation_control_compare.py"
)
CHECKPOINT_GENERATION_CONTROL_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_ft_checkpoint_generation_control.py"
)
GENERIC_CHECKPOINT_GENERATION_CONTROL_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_checkpoint_generation_control.py"
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
MODEL_CONFIGS_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_finetune_model_configs.example.json"
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


def load_generic_generation_control_sweep_example():
    spec = importlib.util.spec_from_file_location(
        "hf_zspace_generation_control_sweep_test",
        GENERIC_SWEEP_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_generation_control_compare_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_zspace_generation_control_compare_test",
        SWEEP_COMPARE_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_generic_generation_control_compare_example():
    spec = importlib.util.spec_from_file_location(
        "hf_zspace_generation_control_compare_test",
        GENERIC_SWEEP_COMPARE_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_checkpoint_generation_control_example():
    spec = importlib.util.spec_from_file_location(
        "hf_gpt2_ft_checkpoint_generation_control_test",
        CHECKPOINT_GENERATION_CONTROL_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def load_generic_checkpoint_generation_control_example():
    spec = importlib.util.spec_from_file_location(
        "hf_checkpoint_generation_control_test",
        GENERIC_CHECKPOINT_GENERATION_CONTROL_EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
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
        self.assertIn("ZSpaceCheckpointPromptSpec", st.__all__)
        self.assertIn("ZSpaceCheckpointSweepJob", st.__all__)
        self.assertIn("build_zspace_activation_probe_hook", st.__all__)
        self.assertIn("compare_zspace_inference_distortion_probes", st.__all__)
        self.assertIn("build_zspace_repression_logits_processor", st.__all__)
        self.assertIn("build_zspace_softmax_logits_processor", st.__all__)
        self.assertIn("default_zspace_checkpoint_generation_prompts", st.__all__)
        self.assertIn("hf_causal_lm_artifact_probe_lines", st.__all__)
        self.assertIn("hf_causal_lm_artifact_probe_report", st.__all__)
        self.assertIn(
            "zspace_inference_distortion_sweep_report_from_probes",
            st.__all__,
        )
        self.assertIn("zspace_generation_control_bridge_cli_args", st.__all__)
        self.assertIn("zspace_generation_control_profile_config", st.__all__)
        self.assertIn("zspace_generation_control_processor_kwargs", st.__all__)
        self.assertIn("zspace_generation_control_sweep_cli_args", st.__all__)
        self.assertIn("zspace_checkpoint_generation_control_jobs", st.__all__)
        self.assertIn("zspace_checkpoint_generation_control_report", st.__all__)
        self.assertIn(
            "zspace_checkpoint_generation_control_compare_command",
            st.__all__,
        )
        self.assertIn("load_zspace_generation_control_sweep", st.__all__)
        self.assertIn("summarize_zspace_generation_control_sweep", st.__all__)
        self.assertIn("compare_zspace_generation_control_sweeps", st.__all__)
        self.assertIn(
            "summarize_zspace_generation_control_sweep_comparison_lines",
            st.__all__,
        )
        self.assertIn("zspace_inference_distortion_processor_kwargs", st.__all__)
        self.assertIn("load_zspace_inference_distortion_probe", st.__all__)
        self.assertIn("summarize_zspace_inference_distortion_probe", st.__all__)
        self.assertIn(
            "summarize_zspace_inference_distortion_probe_comparison_lines",
            st.__all__,
        )
        self.assertIn("summarize_zspace_inference_distortion_probe_lines", st.__all__)
        self.assertIn("zspace_inference_distortion_geometry_probe", st.__all__)
        self.assertIn("zspace_inference_distortion_probe_report", st.__all__)
        self.assertIn("zspace_inference_distortion_runtime_plan", st.__all__)
        self.assertIn("zspace_inference_distortion_runtime_cli_args", st.__all__)
        self.assertIn("zspace_inference_distortion_runtime_preflight", st.__all__)
        self.assertIs(st.ZSpaceRepressionLogitsProcessor, ZSpaceRepressionLogitsProcessor)
        self.assertIs(st.ZSpaceActivationProbeHook, ZSpaceActivationProbeHook)
        self.assertIs(st.ZSpaceCheckpointPromptSpec, ZSpaceCheckpointPromptSpec)
        self.assertIs(st.ZSpaceCheckpointSweepJob, ZSpaceCheckpointSweepJob)
        self.assertIs(
            st.build_zspace_activation_probe_hook,
            build_zspace_activation_probe_hook,
        )
        self.assertIs(
            st.default_zspace_checkpoint_generation_prompts,
            default_zspace_checkpoint_generation_prompts,
        )
        self.assertIs(
            st.zspace_generation_control_bridge_cli_args,
            zspace_generation_control_bridge_cli_args,
        )
        self.assertIs(
            st.zspace_generation_control_profile_config,
            zspace_generation_control_profile_config,
        )
        self.assertIs(
            st.zspace_checkpoint_generation_control_jobs,
            zspace_checkpoint_generation_control_jobs,
        )
        self.assertIs(
            st.zspace_checkpoint_generation_control_report,
            zspace_checkpoint_generation_control_report,
        )
        self.assertIs(
            st.zspace_checkpoint_generation_control_compare_command,
            zspace_checkpoint_generation_control_compare_command,
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
            st.zspace_inference_distortion_geometry_probe,
            zspace_inference_distortion_geometry_probe,
        )
        self.assertIs(
            st.zspace_inference_distortion_probe_report,
            zspace_inference_distortion_probe_report,
        )
        self.assertIs(
            st.summarize_zspace_generation_control_sweep,
            summarize_zspace_generation_control_sweep,
        )
        self.assertIs(
            st.compare_zspace_generation_control_sweeps,
            compare_zspace_generation_control_sweeps,
        )
        self.assertIs(
            st.summarize_zspace_generation_control_sweep_comparison_lines,
            summarize_zspace_generation_control_sweep_comparison_lines,
        )
        self.assertIs(
            st.zspace_inference_distortion_runtime_plan,
            zspace_inference_distortion_runtime_plan,
        )
        self.assertIs(
            st.zspace_inference_distortion_runtime_cli_args,
            zspace_inference_distortion_runtime_cli_args,
        )
        self.assertIs(
            st.zspace_inference_distortion_runtime_preflight,
            zspace_inference_distortion_runtime_preflight,
        )

    def test_causal_lm_artifact_probe_loads_and_generates_with_compat(self) -> None:
        class FakeParameter:
            device = "cpu"

        class LegacyModel:
            def __init__(self) -> None:
                self.generation_kwargs: dict[str, object] = {}

            def parameters(self):
                return iter([FakeParameter()])

            def to(self, _device: str):
                return self

            def eval(self) -> None:
                return None

            def _prepare_special_tokens(self, generation_config, device=None):
                return {"generation_config": generation_config, "device": device}

            def generate(self, **kwargs):
                self._prepare_special_tokens("cfg", device="cpu", batch_size=1)
                self.generation_kwargs = dict(kwargs)
                return [[11, 12, 13, 14]]

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, _prompt: str, *, return_tensors: str):
                self.return_tensors = return_tensors
                return {"input_ids": [[11, 12]]}

            def decode(self, values, *, skip_special_tokens: bool):
                self.skip_special_tokens = skip_special_tokens
                return " ".join(str(value) for value in values)

        model = LegacyModel()
        tokenizer = FakeTokenizer()
        config = types.SimpleNamespace(model_type="gpt_neox")
        load_report = {
            "status": "loaded",
            "artifact_kind": "peft_adapter",
            "loaded_artifact_kind": "peft_adapter",
            "artifact_source": "/tmp/adapter",
            "artifact_is_local": True,
            "resolved_base_model_name_or_path": "org/pythia",
            "resolved_tokenizer_source": "/tmp/adapter",
            "resolved_tokenizer_source_kind": "adapter_artifact",
            "model_loaded": True,
            "model_class": "PeftModelForCausalLM",
            "config_class": "GPTNeoXConfig",
            "tokenizer_class": "GPTNeoXTokenizer",
            "adapter_loaded": True,
            "adapter_trainable": False,
            "adapter_merged": False,
            "peft_version": "0.test",
            "loaded_parameter_report": {
                "parameter_count": 100,
                "trainable_parameter_count": 0,
                "trainable_parameter_ratio": 0.0,
            },
        }
        torch_runtime = types.SimpleNamespace(
            inference_mode=contextlib.nullcontext,
        )

        with mock.patch.object(
            hf_peft,
            "load_hf_causal_lm_artifact",
            return_value=(model, tokenizer, config, load_report),
        ) as load:
            report = st.hf_causal_lm_artifact_probe_report(
                "/tmp/adapter",
                artifact_kind="peft_adapter",
                prompt="SpiralTorch is",
                max_new_tokens=2,
                do_sample=True,
                temperature=0.8,
                top_k=12,
                device="cpu",
                local_files_only=True,
                torch_module=torch_runtime,
            )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["model_family"], "gpt_neox")
        self.assertEqual(report["generated_text"], "11 12 13 14")
        self.assertEqual(report["continuation_text"], "13 14")
        self.assertEqual(report["input_token_count"], 2)
        self.assertEqual(report["new_token_count"], 2)
        self.assertTrue(report["batch_size_compat_installed"])
        self.assertTrue(report["artifact"]["adapter_loaded"])
        self.assertEqual(model.generation_kwargs["temperature"], 0.8)
        self.assertEqual(model.generation_kwargs["top_k"], 12)
        self.assertEqual(model.generation_kwargs["pad_token_id"], 0)
        self.assertTrue(load.call_args.kwargs["loader_kwargs"]["local_files_only"])
        lines = st.hf_causal_lm_artifact_probe_lines(report)
        self.assertTrue(lines[0].startswith("hf_causal_lm_artifact_probe status=ready"))
        self.assertIn('continuation="13 14"', lines[1])

    def test_installed_artifact_probe_cli_writes_json(self) -> None:
        report = {
            "row_type": "hf_causal_lm_artifact_probe",
            "status": "ready",
            "artifact": {
                "artifact_kind": "peft_adapter",
                "artifact_source": "/tmp/adapter",
                "base_model_name_or_path": "org/pythia",
                "adapter_loaded": True,
            },
            "model_family": "gpt_neox",
            "model_class": "PeftModelForCausalLM",
            "tokenizer_class": "GPTNeoXTokenizer",
            "device": "cpu",
            "prompt": "SpiralTorch is",
            "continuation_text": " geometry",
            "input_token_count": 2,
            "new_token_count": 1,
            "batch_size_compat_installed": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.json"
            stdout = io.StringIO()
            with mock.patch.object(
                hf_cli,
                "hf_causal_lm_artifact_probe_report",
                return_value=report,
            ) as probe:
                with contextlib.redirect_stdout(stdout):
                    code = hf_cli.artifact_probe_main(
                        [
                            "/tmp/adapter",
                            "--artifact-kind",
                            "peft-adapter",
                            "--prompt",
                            "SpiralTorch is",
                            "--max-new-tokens",
                            "1",
                            "--allow-remote",
                            "--out",
                            str(out),
                            "--json",
                        ]
                    )

            written = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), report)
        self.assertEqual(written, report)
        self.assertEqual(probe.call_args.args, ("/tmp/adapter",))
        self.assertEqual(probe.call_args.kwargs["artifact_kind"], "peft_adapter")
        self.assertFalse(probe.call_args.kwargs["local_files_only"])

    def test_pythia_artifact_probe_sample_records_promotion_qualification(
        self,
    ) -> None:
        sample_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "hf_pythia70m_lora_artifact_probe_sample.json"
        )
        sample = json.loads(sample_path.read_text(encoding="utf-8"))

        self.assertEqual(
            sample["schema"],
            "spiraltorch.hf_causal_lm_artifact_probe.sample.v2",
        )
        self.assertEqual(sample["model_family"], "gpt_neox")
        self.assertTrue(sample["artifact_probe"]["adapter_loaded"])
        self.assertTrue(sample["artifact_probe"]["local_files_only"])
        self.assertFalse(sample["artifact_probe"]["do_sample"])
        self.assertEqual(
            sample["artifact_probe"]["tokenizer_source_kind"],
            "adapter_artifact",
        )
        self.assertGreater(sample["artifact_probe"]["new_token_count"], 0)
        self.assertTrue(
            sample["training"]["adapter_promotion_artifact_probe_required"]
        )
        self.assertTrue(sample["promotion_qualification"]["promotion_ready"])
        self.assertTrue(
            sample["promotion_qualification"][
                "artifact_probe_candidate_matches"
            ]
        )
        self.assertTrue(
            sample["promotion_qualification"]["artifact_probe_local_files_only"]
        )
        self.assertFalse(
            sample["promotion_qualification"]["artifact_probe_do_sample"]
        )
        self.assertTrue(sample["continuation"]["promotion_revalidated_ready"])
        self.assertTrue(sample["continuation"]["continuation_ready"])
        self.assertTrue(sample["continuation"]["probe_provenance_preserved"])

    def test_inference_distortion_runtime_plan_and_cli_args_are_importable(self) -> None:
        runtime = zspace_inference_distortion_runtime_plan(
            local_model=Path("models/gpt2-zspace"),
            tokenizer_name="models/gpt2-tokenizer",
            allow_remote=True,
            trust_remote_code=True,
            max_new_tokens=24,
            activation_module_name=["transformer.h.0.attn"],
            activation_name_contains="mlp",
            api_provider="openai-responses",
            api_model="gpt-5-nano",
            api_max_tokens=72,
            api_reasoning_effort="minimal",
            api_text_verbosity="low",
        )

        self.assertEqual(runtime["local_model"], "models/gpt2-zspace")
        self.assertEqual(runtime["tokenizer_name"], "models/gpt2-tokenizer")
        self.assertEqual(runtime["max_new_tokens"], 24)
        self.assertEqual(runtime["activation_module_name"], ["transformer.h.0.attn"])
        self.assertEqual(runtime["activation_name_contains"], ["mlp"])
        self.assertEqual(runtime["api_provider"], "openai-responses")
        self.assertEqual(runtime["api_model"], "gpt-5-nano")
        self.assertEqual(runtime["api_reasoning_effort"], "minimal")
        self.assertEqual(runtime["api_text_verbosity"], "low")
        self.assertIsNone(runtime["model_profile_runtime_contract"])
        self.assertEqual(runtime["generation_control_profile_config"], {})

        probe_args = zspace_inference_distortion_runtime_cli_args(runtime)
        sweep_args = zspace_inference_distortion_runtime_cli_args(runtime, sweep=True)

        self.assertIn("--local-model", probe_args)
        self.assertIn("models/gpt2-zspace", probe_args)
        self.assertIn("--tokenizer-name", probe_args)
        self.assertIn("models/gpt2-tokenizer", probe_args)
        self.assertIn("--allow-remote", probe_args)
        self.assertIn("--trust-remote-code", probe_args)
        self.assertIn("--activation-module-name", probe_args)
        self.assertIn("transformer.h.0.attn", probe_args)
        self.assertIn("--activation-name-contains", probe_args)
        self.assertIn("mlp", probe_args)
        self.assertIn("--api-provider", probe_args)
        self.assertIn("openai-responses", probe_args)
        self.assertIn("--api-model", probe_args)
        self.assertIn("gpt-5-nano", probe_args)
        self.assertIn("--api-reasoning-effort", probe_args)
        self.assertIn("minimal", probe_args)
        self.assertIn("--api-text-verbosity", probe_args)
        self.assertIn("low", probe_args)
        self.assertNotIn("--resume-existing", probe_args)
        self.assertIn("--resume-existing", sweep_args)
        self.assertEqual(
            zspace_inference_distortion_runtime_cli_args(None, sweep=True),
            ["--resume-existing"],
        )
        profile_runtime = zspace_inference_distortion_runtime_plan(
            model_configs=MODEL_CONFIGS_PATH,
            model_profile="qwen2-0.5b-local-smoke",
            api_provider="fake",
        )
        profile_args = zspace_inference_distortion_runtime_cli_args(profile_runtime)
        overridden_profile_runtime = zspace_inference_distortion_runtime_plan(
            model_configs=MODEL_CONFIGS_PATH,
            model_profile="pythia-70m-local-smoke",
            local_model=Path("models/local-causal-lm"),
            tokenizer_name="models/local-tokenizer",
            max_new_tokens=7,
            activation_name_contains=["custom.block"],
            api_provider="fake",
        )

        self.assertEqual(profile_runtime["local_model"], "Qwen/Qwen2-0.5B")
        self.assertEqual(profile_runtime["tokenizer_name"], "Qwen/Qwen2-0.5B")
        self.assertEqual(profile_runtime["max_new_tokens"], 128)
        self.assertEqual(
            profile_runtime["activation_name_contains"],
            ["model.layers.0"],
        )
        self.assertEqual(
            profile_runtime["model_profile"]["profile_id"],
            "qwen2-0.5b-local-smoke",
        )
        self.assertIn(
            "profile=qwen2-0.5b-local-smoke",
            profile_runtime["model_profile_lines"][0],
        )
        self.assertEqual(
            profile_runtime["model_profile_runtime_contract"]["profile_id"],
            "qwen2-0.5b-local-smoke",
        )
        self.assertEqual(
            profile_runtime["model_profile_runtime_contract"][
                "generation_control_processor_kwargs"
            ]["top_k"],
            96,
        )
        self.assertIn(
            "--local-model",
            profile_runtime["model_profile_runtime_contract"][
                "explicit_inference_runtime_cli_args"
            ],
        )
        self.assertIn(
            "Qwen/Qwen2-0.5B",
            profile_runtime["model_profile_runtime_contract"][
                "explicit_inference_runtime_cli_args"
            ],
        )
        self.assertEqual(profile_runtime["runtime_import_preset"], "hf-runtime")
        self.assertTrue(
            profile_runtime["model_profile_runtime_contract_lines"][0].startswith(
                "hf_ft_model_profile_runtime_contract "
            )
        )
        self.assertEqual(
            profile_runtime["generation_control_profile_config"]["top_k"],
            96,
        )
        self.assertIn(
            "--generation-zspace-top-k",
            profile_runtime["generation_control_bridge_cli_args"],
        )
        self.assertIn("--model-configs", profile_args)
        self.assertIn(str(MODEL_CONFIGS_PATH), profile_args)
        self.assertIn("--model-profile", profile_args)
        self.assertIn("qwen2-0.5b-local-smoke", profile_args)
        self.assertNotIn("Qwen/Qwen2-0.5B", profile_args)
        self.assertNotIn("model.layers.0", profile_args)
        with tempfile.TemporaryDirectory() as tmp:
            contract_path = Path(tmp) / "runtime-contract.json"
            contract_payload = st.hf_finetune_model_profile_runtime_contract(
                MODEL_CONFIGS_PATH,
                profile="pythia-70m-local-smoke",
                mode="inference",
            )
            contract_path.write_text(
                json.dumps(contract_payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            artifact_runtime = zspace_inference_distortion_runtime_plan(
                runtime_contract_artifact=contract_path,
                api_provider="fake",
            )
            artifact_args = zspace_inference_distortion_runtime_cli_args(
                artifact_runtime,
            )

        self.assertEqual(
            artifact_runtime["runtime_contract_artifact"],
            str(contract_path),
        )
        self.assertEqual(
            artifact_runtime["model_profile_runtime_contract"]["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertEqual(
            artifact_runtime["local_model"],
            "EleutherAI/pythia-70m-deduped",
        )
        self.assertEqual(artifact_runtime["max_new_tokens"], 96)
        self.assertEqual(
            artifact_runtime["activation_name_contains"],
            ["gpt_neox.layers.0"],
        )
        self.assertIn("--runtime-contract-artifact", artifact_args)
        self.assertIn(str(contract_path), artifact_args)
        self.assertNotIn("--model-configs", artifact_args)
        self.assertNotIn("--model-profile", artifact_args)
        self.assertNotIn("EleutherAI/pythia-70m-deduped", artifact_args)
        self.assertEqual(
            overridden_profile_runtime["local_model"],
            "models/local-causal-lm",
        )
        self.assertEqual(
            overridden_profile_runtime["tokenizer_name"],
            "models/local-tokenizer",
        )
        self.assertEqual(overridden_profile_runtime["max_new_tokens"], 7)
        self.assertEqual(
            overridden_profile_runtime["activation_name_contains"],
            ["custom.block"],
        )
        overridden_profile_args = zspace_inference_distortion_runtime_cli_args(
            overridden_profile_runtime,
        )
        self.assertIn("--model-profile", overridden_profile_args)
        self.assertIn("pythia-70m-local-smoke", overridden_profile_args)
        self.assertIn("--local-model", overridden_profile_args)
        self.assertIn("models/local-causal-lm", overridden_profile_args)
        self.assertIn("--tokenizer-name", overridden_profile_args)
        self.assertIn("models/local-tokenizer", overridden_profile_args)
        self.assertIn("--max-new-tokens", overridden_profile_args)
        self.assertIn("7", overridden_profile_args)
        self.assertIn("--activation-name-contains", overridden_profile_args)
        self.assertIn("custom.block", overridden_profile_args)

    def test_inference_distortion_probe_accepts_model_profile_defaults(self) -> None:
        module = load_distortion_probe_example()
        args = module.parse_args(
            [
                "--model-configs",
                str(MODEL_CONFIGS_PATH),
                "--model-profile",
                "pythia-70m-local-smoke",
                "--prompt",
                "SpiralTorch profile probe",
            ]
        )
        overridden = module.parse_args(
            [
                "--model-configs",
                str(MODEL_CONFIGS_PATH),
                "--model-profile",
                "pythia-70m-local-smoke",
                "--local-model",
                "models/local-causal-lm",
                "--tokenizer-name",
                "models/local-tokenizer",
                "--max-new-tokens",
                "7",
            ]
        )

        self.assertEqual(args.local_model, Path("EleutherAI/pythia-70m-deduped"))
        self.assertEqual(args.tokenizer_name, "EleutherAI/pythia-70m-deduped")
        self.assertEqual(args.max_new_tokens, 96)
        self.assertEqual(args.activation_name_contains, ["gpt_neox.layers.0"])
        self.assertEqual(
            args._hf_finetune_model_profile["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertIn(
            "profile=pythia-70m-local-smoke",
            args._hf_finetune_model_profile_lines[0],
        )
        self.assertEqual(overridden.local_model, Path("models/local-causal-lm"))
        self.assertEqual(overridden.tokenizer_name, "models/local-tokenizer")
        self.assertEqual(overridden.max_new_tokens, 7)
        with tempfile.TemporaryDirectory() as tmp:
            contract_path = Path(tmp) / "runtime-contract.json"
            contract_payload = st.hf_finetune_model_profile_runtime_contract(
                MODEL_CONFIGS_PATH,
                profile="pythia-70m-local-smoke",
                mode="inference",
            )
            contract_path.write_text(
                json.dumps(contract_payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            artifact_args = module.parse_args(
                [
                    "--runtime-contract-artifact",
                    str(contract_path),
                    "--prompt",
                    "SpiralTorch artifact probe",
                ]
            )
            sweep_module = load_distortion_sweep_example()
            sweep_args = sweep_module.parse_args(
                [
                    "--runtime-contract-artifact",
                    str(contract_path),
                    "--dry-run",
                ]
            )

        self.assertEqual(
            artifact_args.local_model,
            Path("EleutherAI/pythia-70m-deduped"),
        )
        self.assertEqual(
            artifact_args.tokenizer_name,
            "EleutherAI/pythia-70m-deduped",
        )
        self.assertEqual(artifact_args.max_new_tokens, 96)
        self.assertEqual(
            artifact_args._hf_finetune_model_profile_runtime_contract[
                "source_artifact_path"
            ],
            str(contract_path),
        )
        self.assertEqual(
            sweep_args.local_model,
            Path("EleutherAI/pythia-70m-deduped"),
        )
        self.assertEqual(sweep_args.max_new_tokens, 96)

    def test_inference_distortion_geometry_probe_uses_zspace_evaluator(self) -> None:
        adapter = st.api_llm_zspace_inference_distortion_adapter(
            desire_pressure=0.8,
            desire_stability=0.45,
            psi_total=0.7,
            coherence=0.5,
            distortion_strength=1.1,
        )
        calls = []

        def fake_eval(real, imag, z_re, z_im):
            calls.append((list(real), list(imag), list(z_re), list(z_im)))
            return (
                [(1.0, 2.0), (0.5, 0.25), (0.0, 1.0)],
                [(2.0, 0.0), (0.0, 3.0), (1.0, 1.0)],
            )

        probe = zspace_inference_distortion_geometry_probe(
            adapter,
            zspace_eval_with_derivative_stable=fake_eval,
        )

        self.assertEqual(probe["status"], "ok")
        self.assertEqual(probe["backend"], "native_zspace_eval_with_derivative_stable")
        self.assertEqual(probe["sample_count"], 6)
        self.assertEqual(probe["eval_point_count"], 3)
        self.assertGreater(probe["value_l2"], 0.0)
        self.assertAlmostEqual(probe["derivative_l2"], (15.0**0.5))
        self.assertEqual(len(calls), 1)

    def test_inference_distortion_probe_report_builds_package_artifact(self) -> None:
        runtime = zspace_inference_distortion_runtime_plan(api_provider="fake")
        report = zspace_inference_distortion_probe_report(
            name="package-probe",
            prompt="SpiralTorch package probe",
            probe_path=Path("runs/package-probe.json"),
            config={
                "desire_pressure": 0.8,
                "desire_stability": 0.45,
                "psi_total": 0.7,
                "coherence": 0.5,
                "distortion_strength": 1.1,
                "base_temperature": 0.7,
                "base_top_p": 0.95,
            },
            runtime=runtime,
            runtime_preflight={
                "row_type": "zspace_inference_distortion_runtime_preflight",
                "status": "ok",
                "runtime_ready": True,
                "ready_backends": ["wgpu"],
                "missing_ready_backends": [],
            },
            local_hf={"status": "skipped"},
            api={"provider": "fake", "text": "fake"},
        )

        self.assertEqual(report["row_type"], "zspace_inference_distortion_probe")
        self.assertEqual(report["name"], "package-probe")
        self.assertEqual(report["runtime"], runtime)
        self.assertEqual(
            report["adapter"]["kind"],
            "spiraltorch.zspace_inference_distortion_adapter",
        )
        self.assertEqual(report["geometry_probe"]["status"], "ok")
        self.assertEqual(report["summary"]["runtime_preflight_status"], "ok")
        self.assertEqual(report["summary"]["geometry_status"], "ok")
        self.assertIn("summary_lines", report)

    def test_inference_distortion_runtime_preflight_uses_device_reporter(self) -> None:
        runtime = zspace_inference_distortion_runtime_plan(
            local_model="models/gpt2-zspace",
            api_provider="fake",
        )
        calls = []

        def fake_describe(backends, *, continue_on_error=True, **kwargs):
            calls.append((list(backends), continue_on_error, dict(kwargs)))
            return {
                "backends": list(backends),
                "ready_backends": ["wgpu"],
                "not_ready_backends": ["cpu", "mps"],
                "status_by_backend": {
                    "wgpu": "kernel_wired",
                    "cpu": "cpu",
                    "mps": "placeholder",
                },
                "all_ready": False,
                "has_errors": False,
            }

        preflight = zspace_inference_distortion_runtime_preflight(
            runtime,
            backends=["wgpu", "cpu", "mps", "wgpu"],
            required_ready_backends=["wgpu", "mps"],
            describe_runtime_devices=fake_describe,
            device_kwargs={"workgroup": 128},
        )

        self.assertEqual(preflight["status"], "ok")
        self.assertEqual(preflight["runtime"], runtime)
        self.assertEqual(preflight["backends"], ["wgpu", "cpu", "mps"])
        self.assertEqual(preflight["ready_backends"], ["wgpu"])
        self.assertEqual(preflight["missing_ready_backends"], ["mps"])
        self.assertFalse(preflight["runtime_ready"])
        self.assertEqual(
            preflight["status_by_backend"]["mps"],
            "placeholder",
        )
        self.assertEqual(calls, [(["wgpu", "cpu", "mps"], True, {"workgroup": 128})])

    def test_inference_distortion_runtime_preflight_handles_missing_reporter(self) -> None:
        preflight = zspace_inference_distortion_runtime_preflight(
            {"api_provider": "fake"},
            describe_runtime_devices=False,
        )

        self.assertEqual(preflight["status"], "unavailable")
        self.assertEqual(preflight["row_type"], "zspace_inference_distortion_runtime_preflight")
        self.assertFalse(preflight["runtime_ready"])

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
            "runtime_preflight": {
                "status": "ok",
                "runtime_ready": True,
                "ready_backends": ["wgpu"],
                "missing_ready_backends": [],
            },
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
            "geometry_probe": {
                "status": "ok",
                "backend": "native_zspace_eval_with_derivative_stable",
                "value_l2": 2.5,
                "derivative_l2": 3.5,
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
        self.assertEqual(summary["runtime_preflight_status"], "ok")
        self.assertTrue(summary["runtime_ready"])
        self.assertEqual(summary["runtime_ready_backends"], ["wgpu"])
        self.assertEqual(summary["geometry_status"], "ok")
        self.assertEqual(
            summary["geometry_backend"],
            "native_zspace_eval_with_derivative_stable",
        )
        self.assertEqual(summary["geometry_derivative_l2"], 3.5)
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
        self.assertIn("runtime=ok", lines[0])
        self.assertIn("geom=3.5", lines[0])
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
        self.assertAlmostEqual(
            comparison["best_api_compatibility_score"],
            0.8477329113244411,
        )
        self.assertAlmostEqual(
            comparison["api_compatibility_score_min"],
            0.5430435064265411,
        )
        self.assertEqual(comparison["top_probes"][0]["label"], "strong")
        self.assertGreater(
            comparison["top_probes"][0]["effect_score"],
            comparison["top_probes"][1]["effect_score"],
        )
        self.assertIn("recommended=strong", lines[0])
        self.assertIn("api_compat=0.8477329113244411", lines[0])
        self.assertIn("api_visible=1", lines[0])
        self.assertIn("api_empty=1", lines[0])
        self.assertIn("api_retry_dropped=1", lines[0])
        self.assertIn("label=strong", lines[1])
        self.assertIn("api_retry_dropped=2", lines[1])

    def test_inference_distortion_comparison_prefers_api_compatibility_tiebreak(self) -> None:
        base = {
            "row_type": "zspace_inference_distortion_probe",
            "prompt": "tie",
            "adapter": {
                "distortion_energy": 0.6,
                "request": {"temperature": 0.95, "top_p": 0.8},
            },
            "local_hf": {
                "status": "ok",
                "changed": True,
                "generation_control": {"top_token_changed_count": 4},
                "activation_report": {"event_count": 64},
            },
            "api": {
                "provider": "fake",
                "text": "visible route",
                "telemetry": {"api_llm.empty_text": 0.0},
            },
        }
        retry = json.loads(json.dumps(base))
        retry["api"]["request_filter"] = {
            "dropped_key_count": 2,
            "dropped_keys": ["temperature", "top_p"],
            "retry_dropped_key_count": 2,
            "retry_dropped_keys": ["temperature", "top_p"],
            "sent_keys": ["input", "model", "reasoning", "text"],
        }
        clean = json.loads(json.dumps(base))
        clean["api"]["request_filter"] = {
            "dropped_key_count": 0,
            "dropped_keys": [],
            "sent_keys": ["input", "model", "temperature", "top_p"],
        }

        comparison = compare_zspace_inference_distortion_probes(
            {"a-retry": retry, "z-clean": clean},
            top_n=2,
        )
        lines = summarize_zspace_inference_distortion_probe_comparison_lines(
            comparison
        )

        self.assertEqual(comparison["recommended_probe"], "z-clean")
        self.assertEqual(
            comparison["recommended_reason"],
            "highest_effect_score_lowest_risk_api_compatibility_tiebreak",
        )
        self.assertEqual(comparison["top_probes"][0]["api_compatibility_score"], 1.0)
        self.assertLess(
            comparison["top_probes"][1]["api_compatibility_score"],
            comparison["top_probes"][0]["api_compatibility_score"],
        )
        self.assertIn("recommended=z-clean", lines[0])

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
        self.assertEqual(
            plan["runtime_preflight"]["row_type"],
            "zspace_inference_distortion_runtime_preflight",
        )
        self.assertEqual(plan["execution"]["resume_existing"], False)
        self.assertEqual(
            report["runtime_preflight"]["row_type"],
            "zspace_inference_distortion_runtime_preflight",
        )
        self.assertEqual(report["attempted_run_count"], 0)
        self.assertEqual(report["completed_run_count"], 0)
        self.assertEqual(report["skipped_run_count"], 2)

    def test_installed_inference_distortion_clis_plan_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep_dir = root / "installed-sweep"
            probe_out = root / "installed-probe.json"
            sweep_stdout = io.StringIO()
            with contextlib.redirect_stdout(sweep_stdout):
                sweep_code = hf_cli.zspace_inference_distortion_sweep_main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(sweep_dir),
                        "--prompt",
                        "SpiralTorch installed sweep",
                        "--model-configs",
                        str(MODEL_CONFIGS_PATH),
                        "--model-profile",
                        "qwen2-0.5b-local-smoke",
                        "--desire-pressure-values",
                        "0.4",
                        "--psi-total-values",
                        "0.5",
                    ]
                )
            probe_stdout = io.StringIO()
            with contextlib.redirect_stdout(probe_stdout):
                probe_code = hf_cli.zspace_inference_distortion_probe_main(
                    [
                        "--prompt",
                        "SpiralTorch installed probe",
                        "--api-provider",
                        "fake",
                        "--out",
                        str(probe_out),
                    ]
                )
            sweep_report = json.loads((sweep_dir / "sweep-report.json").read_text())
            probe_report = json.loads(probe_out.read_text())

        self.assertEqual(sweep_code, 0)
        self.assertEqual(probe_code, 0)
        self.assertEqual(
            sweep_report["row_type"],
            "zspace_inference_distortion_sweep",
        )
        self.assertEqual(
            sweep_report["runtime"]["model_profile"]["profile_id"],
            "qwen2-0.5b-local-smoke",
        )
        self.assertEqual(sweep_report["runtime"]["local_model"], "Qwen/Qwen2-0.5B")
        self.assertIn("zspace_inference_distortion_sweep", sweep_stdout.getvalue())
        self.assertEqual(
            probe_report["row_type"],
            "zspace_inference_distortion_probe",
        )
        self.assertEqual(probe_report["api"]["provider"], "fake")
        self.assertIn("zspace_inference_distortion_probe", probe_stdout.getvalue())

    def test_inference_distortion_sweep_accepts_model_profile_defaults(self) -> None:
        module = load_distortion_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sweep"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--model-configs",
                    str(MODEL_CONFIGS_PATH),
                    "--model-profile",
                    "qwen2-0.5b-local-smoke",
                    "--out-dir",
                    str(out_dir),
                    "--prompt",
                    "SpiralTorch profile sweep",
                ]
            )
            runtime = module._runtime_plan(args)
            report = module.run_sweep(args)

        self.assertEqual(args.local_model, Path("Qwen/Qwen2-0.5B"))
        self.assertEqual(args.tokenizer_name, "Qwen/Qwen2-0.5B")
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.activation_name_contains, ["model.layers.0"])
        self.assertEqual(runtime["local_model"], "Qwen/Qwen2-0.5B")
        self.assertEqual(runtime["tokenizer_name"], "Qwen/Qwen2-0.5B")
        self.assertEqual(runtime["activation_name_contains"], ["model.layers.0"])
        self.assertEqual(report["runtime"]["local_model"], "Qwen/Qwen2-0.5B")
        self.assertEqual(report["runtime"]["tokenizer_name"], "Qwen/Qwen2-0.5B")
        self.assertEqual(
            report["runtime"]["activation_name_contains"],
            ["model.layers.0"],
        )

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
            first_probe_payload = json.loads(
                Path(stored_report["runs"][0]["probe_path"]).read_text()
            )
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
        self.assertEqual(
            stored_report["runtime_preflight"]["row_type"],
            "zspace_inference_distortion_runtime_preflight",
        )
        self.assertEqual(len(stored_report["comparison"]["top_probes"]), 2)
        self.assertIn("summary", stored_report)
        self.assertIn("recommendation", stored_report)
        self.assertIn("recommended_commands", stored_report)
        self.assertIn("markdown_path", stored_report)
        self.assertIn("summary_lines", stored_report)
        self.assertTrue(first_probe_exists)
        self.assertEqual(
            first_probe_payload["geometry_probe"]["row_type"],
            "zspace_inference_distortion_geometry_probe",
        )
        self.assertEqual(first_probe_payload["geometry_probe"]["status"], "ok")
        self.assertEqual(
            first_probe_payload["runtime_preflight"]["row_type"],
            "zspace_inference_distortion_runtime_preflight",
        )
        self.assertTrue(stored_report["runs"][0]["summary"]["probe_path"])
        self.assertEqual(
            stored_report["comparison"]["top_probes"][0]["api_provider"],
            "fake",
        )
        self.assertEqual(sweep_summary["row_type"], "zspace_inference_distortion_sweep_summary")
        self.assertEqual(sweep_summary["completed_run_count"], 2)
        self.assertIn(
            sweep_summary["runtime_preflight_status"],
            {"ok", "unavailable", "error"},
        )
        self.assertIsInstance(sweep_summary["runtime_ready_backends"], list)
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
        self.assertIn(
            "spiral-zspace-inference-distortion-probe",
            stored_report["recommended_commands"]["installed_probe"],
        )
        self.assertIn(
            "spiral-zspace-inference-distortion-sweep",
            stored_report["recommended_commands"]["installed_sweep"],
        )
        self.assertIn("zspace_inference_distortion_sweep", sweep_lines[0])
        self.assertIn("Z-Space Inference Distortion Sweep", markdown)
        self.assertIn("Single-probe replay", markdown)
        self.assertIn("Focused sweep replay", markdown)
        self.assertIn("Installed single-probe replay", markdown)
        self.assertIn("Installed focused sweep replay", markdown)
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
        self.assertEqual(
            replay_report["runtime_preflight"]["row_type"],
            "zspace_inference_distortion_runtime_preflight",
        )
        self.assertEqual(
            replay_report["geometry_probe"]["row_type"],
            "zspace_inference_distortion_geometry_probe",
        )
        self.assertEqual(replay_report["geometry_probe"]["status"], "ok")

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
                "telemetry": {"api_llm.empty_text": 0.0},
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
        self.assertGreater(summary["recommended_api_compatibility_score"], 0.0)
        self.assertEqual(summary["api_visible_text_count"], 1)
        self.assertEqual(summary["api_retry_dropped_probe_count"], 1)
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
                    "reported_rows": 2,
                    "temperature_min": 0.7,
                    "temperature_max": 1.1,
                    "entropy_min": 2.5,
                    "entropy_max": 3.25,
                    "ngram_repressed_token_total": 4,
                    "max_ngram_repression": 0.75,
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
        pythia_profile = st.resolve_hf_finetune_model_profile(
            MODEL_CONFIGS_PATH,
            profile="pythia-70m-local-smoke",
        )
        pythia_profile_config = zspace_generation_control_profile_config(
            MODEL_CONFIGS_PATH,
            model_profile="pythia-70m-local-smoke",
        )
        pythia_kwargs = zspace_generation_control_processor_kwargs(pythia_profile)
        pythia_bridge_args = zspace_generation_control_bridge_cli_args(pythia_profile)
        qwen_runtime = zspace_inference_distortion_runtime_plan(
            model_configs=MODEL_CONFIGS_PATH,
            model_profile="qwen2-0.5b-local-smoke",
        )
        qwen_sweep_args = zspace_generation_control_sweep_cli_args(qwen_runtime)

        self.assertEqual(pythia_kwargs["top_k"], 64)
        self.assertEqual(
            pythia_profile_config["row_type"],
            "zspace_generation_control_profile_config",
        )
        self.assertEqual(pythia_profile_config["status"], "ready")
        self.assertEqual(
            pythia_profile_config["model_profile"]["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertEqual(
            pythia_profile_config["model_profile_runtime_contract"]["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertEqual(
            pythia_profile_config["runtime_import_preset"],
            "hf-runtime",
        )
        self.assertTrue(
            pythia_profile_config["model_profile_runtime_contract_lines"][
                0
            ].startswith("hf_ft_model_profile_runtime_contract ")
        )
        self.assertEqual(pythia_profile_config["recommended_config"], pythia_kwargs)
        self.assertEqual(pythia_profile_config["processor_kwargs"], pythia_kwargs)
        self.assertEqual(
            zspace_generation_control_processor_kwargs(
                pythia_profile_config["model_profile_runtime_contract"]
            ),
            pythia_kwargs,
        )
        self.assertEqual(pythia_kwargs["curvature"], -0.04)
        self.assertEqual(pythia_kwargs["entropy_target"], 3.0)
        self.assertEqual(pythia_kwargs["repression_strength"], 0.8)
        self.assertEqual(pythia_kwargs["ngram_decay"], 0.85)
        self.assertIn("--generation-zspace-top-k", pythia_bridge_args)
        self.assertIn("64", pythia_bridge_args)
        self.assertIn("--generation-repression-strength", pythia_bridge_args)
        self.assertIn("0.8", pythia_bridge_args)
        self.assertEqual(
            zspace_generation_control_processor_kwargs(pythia_profile_config),
            pythia_kwargs,
        )
        self.assertEqual(
            zspace_generation_control_bridge_cli_args(pythia_profile_config),
            pythia_profile_config["bridge_cli_args"],
        )
        self.assertIn("--zspace-top-k-values", qwen_sweep_args)
        self.assertIn("96", qwen_sweep_args)
        self.assertIn("--repression-strength-values", qwen_sweep_args)
        self.assertIn("0.65", qwen_sweep_args)
        self.assertEqual(summary["best_loop_score_delta_from_baseline"], -3.0)
        self.assertEqual(summary["best_loop_score_reduction_ratio"], 1.0)
        self.assertEqual(summary["max_control_calls"], 2.0)
        self.assertEqual(summary["control_entropy_min"], 2.5)
        self.assertEqual(summary["control_entropy_max"], 3.25)
        self.assertEqual(summary["control_temperature_min"], 0.7)
        self.assertEqual(summary["control_temperature_max"], 1.1)
        self.assertEqual(summary["max_control_ngram_repressed_token_total"], 4.0)
        self.assertEqual(summary["max_control_ngram_repression"], 0.75)
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

    def test_generation_control_sweep_comparison_groups_model_runs(self) -> None:
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
                "repetition": {"loop_score": baseline_loop, "unique_word_ratio": 0.25},
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

        comparison = compare_zspace_generation_control_sweeps(
            {
                "base-prompt": sweep_report(
                    model_name="base-model",
                    prompt="SpiralTorch is",
                    baseline_loop=10.0,
                    controlled_loop=4.0,
                    top_changes=3,
                ),
                "ft-prompt": sweep_report(
                    model_name="ft-model",
                    prompt="SpiralTorch is",
                    baseline_loop=50.0,
                    controlled_loop=5.0,
                    top_changes=9,
                ),
            },
            top_n=2,
        )
        lines = summarize_zspace_generation_control_sweep_comparison_lines(
            comparison,
            top_n=2,
        )

        self.assertEqual(
            comparison["row_type"],
            "zspace_generation_control_sweep_comparison",
        )
        self.assertEqual(comparison["sweep_count"], 2)
        self.assertEqual(comparison["completed_sweep_count"], 2)
        self.assertEqual(comparison["changed_from_baseline_total"], 2.0)
        self.assertEqual(comparison["zspace_helped_count"], 2)
        self.assertEqual(comparison["recommended_sweep_label"], "ft-prompt")
        self.assertAlmostEqual(comparison["mean_baseline_loop_score"], 30.0)
        self.assertAlmostEqual(comparison["mean_best_loop_score"], 4.5)
        self.assertAlmostEqual(
            comparison["mean_loop_score_delta_from_baseline"],
            -25.5,
        )
        self.assertEqual(comparison["top_sweeps"][0]["label"], "ft-prompt")
        model_rows = {
            str(row["model_name"]): row for row in comparison["model_rows"]
        }
        self.assertEqual(model_rows["ft-model"]["mean_best_loop_score"], 5.0)
        self.assertEqual(model_rows["ft-model"]["max_top_token_changed_count"], 9.0)
        self.assertIn("recommend=ft-prompt", lines[0])
        self.assertIn("model=ft-model", " ".join(lines))
        self.assertIn("label=ft-prompt", lines[-2])

    def test_generation_control_compare_example_writes_json_and_lines(self) -> None:
        module = load_generation_control_compare_example()

        def sweep_report(path: Path, *, model_name: str, loop_score: float) -> None:
            report = {
                "row_type": "hf_gpt2_zspace_generation_control_sweep",
                "status": "complete",
                "dry_run": False,
                "model_name": model_name,
                "prompt": "SpiralTorch is",
                "run_count": 2,
                "runs": [
                    {
                        "name": "baseline-greedy",
                        "kind": "baseline",
                        "status": "ok",
                        "config": {},
                        "generation": hf_ft.hf_gpt2_finetune_generation_report(
                            stage="baseline",
                            prompt="SpiralTorch is",
                            generated_text="SpiralTorch is loop loop loop",
                            generated_continuation_text=" loop loop loop",
                        ),
                        "repetition": {
                            "loop_score": 9.0,
                            "unique_word_ratio": 0.25,
                        },
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
                            "ngram_size": 3,
                            "ngram_window": 96,
                            "ngram_repression_strength": 1.0,
                            "ngram_decay": 0.95,
                        },
                        "generation": hf_ft.hf_gpt2_finetune_generation_report(
                            stage="controlled",
                            prompt="SpiralTorch is",
                            generated_text=f"SpiralTorch is {model_name}",
                            generated_continuation_text=f" {model_name}",
                            generation_control={
                                "status": "ok",
                                "calls": 4,
                                "top_token_changed_count": 2,
                            },
                        ),
                        "repetition": {
                            "loop_score": loop_score,
                            "unique_word_ratio": 1.0,
                        },
                    },
                ],
            }
            path.write_text(json.dumps(report), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            out = root / "comparison.json"
            lines_out = root / "comparison.txt"
            sweep_report(first, model_name="warmstart", loop_score=5.0)
            sweep_report(second, model_name="checkpoint-512", loop_score=2.0)
            args = module.parse_args(
                [
                    str(first),
                    str(second),
                    "--label",
                    "warmstart",
                    "--label",
                    "checkpoint-512",
                    "--out",
                    str(out),
                    "--lines-out",
                    str(lines_out),
                    "--top-n",
                    "2",
                ]
            )
            comparison = module.compare_sweeps(args)
            rc = module.main(
                [
                    str(first),
                    str(second),
                    "--label",
                    "warmstart",
                    "--label",
                    "checkpoint-512",
                    "--out",
                    str(out),
                    "--lines-out",
                    str(lines_out),
                    "--top-n",
                    "2",
                ]
            )
            stored = json.loads(out.read_text(encoding="utf-8"))
            lines = lines_out.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rc, 0)
        self.assertEqual(comparison["recommended_sweep_label"], "checkpoint-512")
        self.assertEqual(stored["recommended_sweep_label"], "checkpoint-512")
        self.assertTrue(any("recommend=checkpoint-512" in line for line in lines))

    def test_checkpoint_generation_control_dry_run_plans_default_prompts(self) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--label-prefix",
                    "new",
                    "--dry-run",
                ]
            )
            jobs = module.build_sweep_jobs(args)
            compare_command = module.build_compare_command(args, jobs)
            report = module.run_checkpoint_generation_control(args)

        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].out.name, "checkpoint-2048-generation-control-sweep.json")
        self.assertEqual(
            jobs[1].out.name,
            "prompt-desire-coherence-checkpoint-2048-generation-control-sweep.json",
        )
        self.assertEqual(
            jobs[2].out.name,
            "prompt-tokenless-ft-checkpoint-2048-generation-control-sweep.json",
        )
        self.assertIn("new-spiral-checkpoint-2048", compare_command)
        self.assertTrue(
            any(
                item.endswith("generation-control-compare-3prompt-2048.json")
                for item in compare_command
            )
        )
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["sweep_count"], 3)
        self.assertEqual(report["compare"]["status"], "planned")

    def test_package_checkpoint_generation_control_plans_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoint-2048"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.safetensors").write_text(
                "ready",
                encoding="utf-8",
            )
            run_card = root / "checkpoint-generation-control.json"
            prompts = default_zspace_checkpoint_generation_prompts()
            jobs = zspace_checkpoint_generation_control_jobs(
                run_dir=run_dir,
                checkpoint="checkpoint-2048",
                prompt=prompts,
                label_prefix="pkg",
            )
            compare_command = zspace_checkpoint_generation_control_compare_command(
                jobs,
                run_dir=run_dir,
                checkpoint="checkpoint-2048",
                top_n=2,
            )
            planned = zspace_checkpoint_generation_control_report(
                run_dir=run_dir,
                checkpoint="checkpoint-2048",
                prompt=prompts,
                label_prefix="pkg",
                dry_run=True,
                curve_out=run_dir / "curve.json",
                curve_lines_out=run_dir / "curve.txt",
                top_n=2,
            )
            executed: list[list[str]] = []

            def fake_runner(command):
                executed.append(list(command))
                command_script = Path(command[1]).name
                if command_script in {
                    "hf_gpt2_zspace_generation_control_sweep.py",
                    "hf_zspace_generation_control_sweep.py",
                }:
                    out = Path(command[command.index("--out") + 1])
                    out.write_text(
                        json.dumps(
                            {
                                "row_type": "hf_gpt2_zspace_generation_control_sweep",
                                "status": "complete",
                            }
                        ),
                        encoding="utf-8",
                    )
                if command_script in {
                    "hf_gpt2_zspace_generation_control_compare.py",
                    "hf_zspace_generation_control_compare.py",
                }:
                    out = Path(command[command.index("--out") + 1])
                    lines_out = Path(command[command.index("--lines-out") + 1])
                    out.write_text('{"status":"complete"}\n', encoding="utf-8")
                    lines_out.write_text("ok\n", encoding="utf-8")
                return None

            report = zspace_checkpoint_generation_control_report(
                run_dir=run_dir,
                checkpoint="checkpoint-2048",
                prompt=prompts,
                label_prefix="pkg",
                dry_run=False,
                run_card=run_card,
                top_n=2,
                runner=fake_runner,
            )
            stored = json.loads(run_card.read_text(encoding="utf-8"))

        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].out.name, "checkpoint-2048-generation-control-sweep.json")
        self.assertIn("pkg-spiral-checkpoint-2048", compare_command)
        self.assertEqual(planned["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(planned["curve"]["status"], "planned")
        self.assertEqual(report["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(stored["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(stored["sweep_count"], 3)
        self.assertEqual(len(executed), 4)

    def test_package_checkpoint_generation_control_wait_card_is_generic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            (run_dir / "checkpoint-4096").mkdir(parents=True)
            run_card = root / "checkpoint-wait-card.json"

            with self.assertRaises(TimeoutError):
                zspace_checkpoint_generation_control_report(
                    run_dir=run_dir,
                    checkpoint="checkpoint-4096",
                    run_card=run_card,
                    wait=True,
                    poll_seconds=0.01,
                    timeout_seconds=0.01,
                    dry_run=False,
                    no_compare=True,
                    runner=lambda _: None,
                )
            stored = json.loads(run_card.read_text(encoding="utf-8"))

        self.assertEqual(stored["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(stored["status"], "waiting_for_checkpoint")
        self.assertIn("model.safetensors", stored["checkpoint_wait"]["missing"][0])

    def test_package_checkpoint_generation_control_uses_model_profile_defaults(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            planned = zspace_checkpoint_generation_control_report(
                run_dir=run_dir,
                checkpoint="checkpoint-2048",
                model_configs=MODEL_CONFIGS_PATH,
                model_profile="tiny-gpt2-ci",
                dry_run=True,
                no_compare=True,
                curve_out=run_dir / "curve.json",
                curve_lines_out=run_dir / "curve.lines",
            )
            pythia_planned = zspace_checkpoint_generation_control_report(
                run_dir=run_dir,
                checkpoint="checkpoint-4096",
                model_configs=MODEL_CONFIGS_PATH,
                model_profile="pythia-70m-local-smoke",
                dry_run=True,
                no_compare=True,
            )

        command = planned["sweeps"][0]["command"]
        curve_command = planned["curve"]["command"]
        pythia_command = pythia_planned["sweeps"][0]["command"]
        self.assertEqual(planned["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(planned["tokenizer_name"], "sshleifer/tiny-gpt2")
        self.assertEqual(planned["model_profile"]["profile_id"], "tiny-gpt2-ci")
        self.assertEqual(
            planned["model_profile_runtime_contract"]["profile_id"],
            "tiny-gpt2-ci",
        )
        self.assertTrue(
            planned["model_profile_runtime_contract_lines"][0].startswith(
                "hf_ft_model_profile_runtime_contract "
            )
        )
        self.assertIn("profile=tiny-gpt2-ci", planned["model_profile_lines"][0])
        self.assertEqual(planned["sweeps"][0]["tokenizer_name"], "sshleifer/tiny-gpt2")
        self.assertEqual(planned["generation_control_profile_config"], {})
        self.assertEqual(planned["generation_control_sweep_cli_args"], [])
        self.assertEqual(planned["generation_control_bridge_cli_args"], [])
        self.assertEqual(
            command[command.index("--model-configs") + 1],
            str(MODEL_CONFIGS_PATH),
        )
        self.assertEqual(command[command.index("--model-profile") + 1], "tiny-gpt2-ci")
        self.assertEqual(
            command[command.index("--model-name") + 1],
            str(run_dir / "checkpoint-2048"),
        )
        self.assertEqual(
            command[command.index("--tokenizer-name") + 1],
            "sshleifer/tiny-gpt2",
        )
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "32")
        self.assertEqual(
            curve_command[curve_command.index("--model-configs") + 1],
            str(MODEL_CONFIGS_PATH),
        )
        self.assertEqual(
            curve_command[curve_command.index("--model-profile") + 1],
            "tiny-gpt2-ci",
        )
        self.assertEqual(
            pythia_planned["model_profile"]["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertEqual(
            pythia_planned["model_profile_runtime_contract"]["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertEqual(
            pythia_planned["generation_control_profile_config"]["top_k"],
            64,
        )
        self.assertEqual(
            pythia_planned["generation_control_profile_config"][
                "repression_strength"
            ],
            0.8,
        )
        self.assertIn(
            "--zspace-top-k-values",
            pythia_planned["generation_control_sweep_cli_args"],
        )
        self.assertIn(
            "--generation-zspace-top-k",
            pythia_planned["generation_control_bridge_cli_args"],
        )
        self.assertEqual(
            pythia_command[pythia_command.index("--model-profile") + 1],
            "pythia-70m-local-smoke",
        )

    def test_package_checkpoint_generation_control_uses_profile_runtime_defaults(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            config_path = root / "profiles.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema": "spiraltorch.hf_finetune_model_configs.v1",
                        "default_profile": "remote-checkpoint",
                        "profiles": [
                            {
                                "id": "remote-checkpoint",
                                "model_name": "org/remote-base",
                                "tokenizer_name": "org/remote-tokenizer",
                                "architecture": "causal_lm",
                                "generation": {
                                    "max_new_tokens": 19,
                                    "do_sample": True,
                                    "temperature": 0.7,
                                    "top_k": 30,
                                },
                                "runtime": {
                                    "allow_remote": True,
                                    "trust_remote_code": True,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            planned = zspace_checkpoint_generation_control_report(
                run_dir=run_dir,
                checkpoint="checkpoint-2048",
                model_configs=config_path,
                dry_run=True,
                no_compare=True,
            )

        command = planned["sweeps"][0]["command"]
        self.assertTrue(planned["allow_remote"])
        self.assertTrue(planned["trust_remote_code"])
        self.assertIn("--allow-remote", command)
        self.assertIn("--trust-remote-code", command)
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "19")
        self.assertIn("--do-sample", command)
        self.assertEqual(command[command.index("--sample-temperature") + 1], "0.7")
        self.assertEqual(command[command.index("--sample-top-k") + 1], "30")

    def test_checkpoint_generation_control_runs_sweeps_and_compare_with_runner(
        self,
    ) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            (run_dir / "checkpoint-2048").mkdir(parents=True)
            (run_dir / "checkpoint-2048" / "model.safetensors").write_text(
                "ready",
                encoding="utf-8",
            )
            run_card = root / "checkpoint-generation-control.json"
            executed: list[list[str]] = []

            def fake_runner(command):
                executed.append(list(command))
                command_script = Path(command[1]).name
                if command_script in {
                    "hf_gpt2_zspace_generation_control_sweep.py",
                    "hf_zspace_generation_control_sweep.py",
                }:
                    out = Path(command[command.index("--out") + 1])
                    out.write_text(
                        json.dumps(
                            {
                                "row_type": "hf_gpt2_zspace_generation_control_sweep",
                                "status": "complete",
                            }
                        ),
                        encoding="utf-8",
                    )
                if command_script in {
                    "hf_gpt2_zspace_generation_control_compare.py",
                    "hf_zspace_generation_control_compare.py",
                }:
                    out = Path(command[command.index("--out") + 1])
                    lines_out = Path(command[command.index("--lines-out") + 1])
                    out.write_text('{"status":"complete"}\n', encoding="utf-8")
                    lines_out.write_text("ok\n", encoding="utf-8")
                return None

            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--run-card",
                    str(run_card),
                    "--top-n",
                    "2",
                ]
            )
            report = module.run_checkpoint_generation_control(args, runner=fake_runner)
            stored = json.loads(run_card.read_text(encoding="utf-8"))
            lines_exists = (
                run_dir / "generation-control-compare-3prompt-2048.txt"
            ).is_file()

        self.assertEqual(report["status"], "complete")
        self.assertEqual(stored["sweep_count"], 3)
        self.assertEqual(len(executed), 4)
        self.assertTrue(lines_exists)

    def test_checkpoint_generation_control_model_profile_flows_to_sweep_commands(
        self,
    ) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--model-configs",
                    str(MODEL_CONFIGS_PATH),
                    "--model-profile",
                    "tiny-gpt2-ci",
                    "--dry-run",
                    "--no-compare",
                ]
            )
            jobs = module.build_sweep_jobs(args)
            command = module.build_sweep_command(args, jobs[0])
            report = module.run_checkpoint_generation_control(args)

        self.assertEqual(args.tokenizer_name, "sshleifer/tiny-gpt2")
        self.assertEqual(args.max_new_tokens, 32)
        self.assertEqual(args.curve_model_name, "sshleifer/tiny-gpt2")
        self.assertEqual(
            args._hf_finetune_model_profile["profile_id"],
            "tiny-gpt2-ci",
        )
        self.assertEqual(
            command[command.index("--tokenizer-name") + 1],
            "sshleifer/tiny-gpt2",
        )
        self.assertEqual(
            command[command.index("--model-configs") + 1],
            str(MODEL_CONFIGS_PATH),
        )
        self.assertEqual(command[command.index("--model-profile") + 1], "tiny-gpt2-ci")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "32")
        self.assertEqual(report["model_profile"]["profile_id"], "tiny-gpt2-ci")
        self.assertEqual(report["tokenizer_name"], "sshleifer/tiny-gpt2")

    def test_generic_checkpoint_generation_control_defaults_to_generic_scripts(
        self,
    ) -> None:
        module = load_generic_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            default_run_card = (
                Path(tmp) / "generic-checkpoint-generation-control-default.json"
            )
            default_args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--run-card",
                    str(default_run_card),
                    "--dry-run",
                    "--no-compare",
                ]
            )
            default_jobs = module.build_sweep_jobs(default_args)
            default_command = module.build_sweep_command(
                default_args,
                default_jobs[0],
            )
            default_report = module.run_checkpoint_generation_control(default_args)
            explicit_tokenizer_args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--tokenizer-name",
                    "custom/tokenizer",
                    "--dry-run",
                    "--no-compare",
                ]
            )
            run_card = Path(tmp) / "generic-checkpoint-generation-control.json"
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--model-configs",
                    str(MODEL_CONFIGS_PATH),
                    "--model-profile",
                    "tiny-gpt2-ci",
                    "--run-card",
                    str(run_card),
                    "--dry-run",
                    "--no-compare",
                ]
            )
            jobs = module.build_sweep_jobs(args)
            command = module.build_sweep_command(args, jobs[0])
            report = module.run_checkpoint_generation_control(args)
            stored = json.loads(run_card.read_text(encoding="utf-8"))

        self.assertEqual(
            default_args.model_profile,
            st.HF_FINETUNE_DEFAULT_MODEL_PROFILE,
        )
        self.assertEqual(default_args.tokenizer_name, "EleutherAI/pythia-70m-deduped")
        self.assertEqual(default_args.max_new_tokens, 96)
        self.assertEqual(
            default_command[default_command.index("--model-profile") + 1],
            st.HF_FINETUNE_DEFAULT_MODEL_PROFILE,
        )
        self.assertEqual(
            default_report["model_profile"]["profile_id"],
            st.HF_FINETUNE_DEFAULT_MODEL_PROFILE,
        )
        self.assertIsNone(explicit_tokenizer_args.model_profile)
        self.assertEqual(explicit_tokenizer_args.tokenizer_name, "custom/tokenizer")
        self.assertEqual(
            args.sweep_script.name,
            "hf_zspace_generation_control_sweep.py",
        )
        self.assertEqual(
            args.compare_script.name,
            "hf_zspace_generation_control_compare.py",
        )
        self.assertEqual(args.curve_script.name, "hf_finetune_generation_curve.py")
        self.assertEqual(
            Path(command[1]).name,
            "hf_zspace_generation_control_sweep.py",
        )
        self.assertEqual(
            command[command.index("--tokenizer-name") + 1],
            "sshleifer/tiny-gpt2",
        )
        self.assertEqual(report["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(stored["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(report["tokenizer_name"], "sshleifer/tiny-gpt2")
        self.assertEqual(stored["tokenizer_name"], "sshleifer/tiny-gpt2")

    def test_checkpoint_generation_control_requires_ready_file(self) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "checkpoint-2048").mkdir(parents=True)
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-2048",
                    "--no-compare",
                ]
            )

            with self.assertRaises(FileNotFoundError) as cm:
                module.run_checkpoint_generation_control(args, runner=lambda _: None)

        self.assertIn("model.safetensors", str(cm.exception))

    def test_checkpoint_generation_control_writes_checkpoint_wait_card(self) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            (run_dir / "checkpoint-4096").mkdir(parents=True)
            run_card = root / "wait-card.json"
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-4096",
                    "--run-card",
                    str(run_card),
                    "--wait",
                    "--poll-seconds",
                    "0.01",
                    "--timeout-seconds",
                    "0.01",
                    "--no-compare",
                ]
            )

            with self.assertRaises(TimeoutError):
                module.run_checkpoint_generation_control(args, runner=lambda _: None)
            stored = json.loads(run_card.read_text(encoding="utf-8"))

        self.assertEqual(stored["status"], "waiting_for_checkpoint")
        self.assertIn("model.safetensors", stored["checkpoint_wait"]["missing"][0])

    def test_checkpoint_generation_control_records_process_wait_plan(self) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            pid_file = Path(tmp) / "ft.pid"
            pid_file.write_text("99999999\n", encoding="utf-8")
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-4096",
                    "--wait-for-process-pid-file",
                    str(pid_file),
                    "--dry-run",
                ]
            )
            report = module.run_checkpoint_generation_control(args)

        self.assertEqual(report["process_wait"]["status"], "planned")
        self.assertEqual(report["process_wait"]["pid_file"], str(pid_file))

    def test_checkpoint_generation_control_continues_after_exited_process(
        self,
    ) -> None:
        module = load_checkpoint_generation_control_example()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoint-4096"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.safetensors").write_text("ready", encoding="utf-8")
            pid_file = root / "ft.pid"
            pid_file.write_text("99999999\n", encoding="utf-8")
            executed: list[list[str]] = []
            args = module.parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    "checkpoint-4096",
                    "--wait-for-process-pid-file",
                    str(pid_file),
                    "--no-compare",
                ]
            )
            report = module.run_checkpoint_generation_control(
                args,
                runner=lambda command: executed.append(list(command)),
            )

        self.assertEqual(report["process_wait"]["status"], "already_exited")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(executed), 3)

    def test_dry_run_builds_control_grid_without_loading_model(self) -> None:
        module = load_generation_control_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "control-sweep.json"
            args = module.parse_args(
                [
                    "--dry-run",
                    "--model-name",
                    str(Path(tmp) / "checkpoint-local"),
                    "--tokenizer-name",
                    "sshleifer/tiny-gpt2",
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
        self.assertEqual(report["model_name"], str(Path(tmp) / "checkpoint-local"))
        self.assertEqual(report["tokenizer_name"], "sshleifer/tiny-gpt2")
        self.assertEqual(report["run_count"], 5)
        self.assertEqual(report["summary"]["completed_run_count"], 0)
        self.assertTrue(any(str(row["name"]).startswith("zt3") for row in runs))

    def test_generation_sweep_compat_reaches_peft_base_model(self) -> None:
        module = load_generation_control_sweep_example()
        self.assertIs(
            module._prepare_special_tokens_batch_size_compat,
            st.hf_generation_batch_size_compat,
        )

        class BaseModel:
            def _prepare_special_tokens(self, generation_config, device=None):
                return {"generation_config": generation_config, "device": device}

        class PeftWrapper:
            def __init__(self):
                self.base_model = types.SimpleNamespace(model=BaseModel())

            def get_base_model(self):
                return self.base_model.model

        model = PeftWrapper()
        base = model.get_base_model()
        with self.assertRaises(TypeError):
            base._prepare_special_tokens("cfg", batch_size=1)
        self.assertNotIn("_prepare_special_tokens", model.__dict__)
        self.assertNotIn("_prepare_special_tokens", base.__dict__)

        with module._prepare_special_tokens_batch_size_compat(model) as installed:
            self.assertTrue(installed)
            self.assertEqual(
                base._prepare_special_tokens(
                    "cfg",
                    device="cpu",
                    batch_size=1,
                ),
                {"generation_config": "cfg", "device": "cpu"},
            )

        with self.assertRaises(TypeError):
            base._prepare_special_tokens("cfg", batch_size=1)
        self.assertNotIn("_prepare_special_tokens", model.__dict__)
        self.assertNotIn("_prepare_special_tokens", base.__dict__)

    def test_generic_generation_control_wrappers_parse_and_dry_run(self) -> None:
        sweep_module = load_generic_generation_control_sweep_example()
        compare_module = load_generic_generation_control_compare_example()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_args = sweep_module.parse_args(
                [
                    "--dry-run",
                    "--prompt",
                    "SpiralTorch is",
                ]
            )
            default_report = sweep_module.run_sweep(default_args)
            explicit_model_args = sweep_module.parse_args(
                [
                    "--dry-run",
                    "--model-name",
                    "gpt2",
                    "--prompt",
                    "SpiralTorch is",
                ]
            )
            args = sweep_module.parse_args(
                [
                    "--dry-run",
                    "--model-configs",
                    str(MODEL_CONFIGS_PATH),
                    "--model-profile",
                    "pythia-70m-local-smoke",
                    "--prompt",
                    "SpiralTorch is",
                ]
            )
            report = sweep_module.run_sweep(args)
            out_path = root / "control-sweep.json"
            out_path.write_text(json.dumps(report), encoding="utf-8")
            compare_args = compare_module.parse_args(
                [
                    str(out_path),
                    "--label",
                    "generic",
                ]
            )
            comparison = compare_module.compare_sweeps(compare_args)
            comparison_lines = summarize_zspace_generation_control_sweep_comparison_lines(
                comparison,
            )
            legacy_nested_report = {
                "row_type": "hf_gpt2_zspace_generation_control_sweep",
                "status": "complete",
                "summary": {
                    "row_type": "hf_gpt2_zspace_generation_control_sweep_summary",
                },
                "runs": [
                    {
                        "name": "baseline",
                        "generation": {
                            "row_type": "hf_gpt2_finetune_generation_report",
                        },
                    }
                ],
            }
            with mock.patch.object(
                sweep_module._legacy,
                "run_sweep",
                return_value=legacy_nested_report,
            ):
                nested_report = sweep_module.run_sweep(args)

        self.assertEqual(default_args.model_profile, "causal-lm-local-smoke")
        self.assertEqual(default_report["model_name"], "EleutherAI/pythia-70m-deduped")
        self.assertEqual(
            default_report["model_profile"]["profile_id"],
            "causal-lm-local-smoke",
        )
        self.assertIsNone(explicit_model_args.model_profile)
        self.assertEqual(explicit_model_args.model_name, "gpt2")
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["row_type"], "hf_zspace_generation_control_sweep")
        self.assertEqual(
            report["summary"]["row_type"],
            "hf_zspace_generation_control_sweep_summary",
        )
        self.assertEqual(args.out, Path("runs/hf-zspace-generation-control-sweep.json"))
        self.assertEqual(report["model_name"], "EleutherAI/pythia-70m-deduped")
        self.assertEqual(report["tokenizer_name"], "EleutherAI/pythia-70m-deduped")
        self.assertEqual(
            report["model_profile"]["profile_id"],
            "pythia-70m-local-smoke",
        )
        self.assertIn(
            "profile=pythia-70m-local-smoke",
            report["model_profile_lines"][0],
        )
        self.assertEqual(report["max_new_tokens"], 96)
        self.assertTrue(report["do_sample"])
        self.assertEqual(report["run_count"], 2)
        zspace_run = report["runs"][1]
        self.assertEqual(zspace_run["config"]["top_k"], 64)
        self.assertEqual(zspace_run["config"]["entropy_target"], 3.0)
        self.assertEqual(zspace_run["config"]["repression_strength"], 0.8)
        self.assertEqual(zspace_run["config"]["ngram_window"], 32)
        self.assertEqual(report["generation_control_profile_config"]["top_k"], 64)
        self.assertEqual(
            report["generation_control_resolved_config"]["repression_strength"],
            0.8,
        )
        self.assertEqual(report["generation_control_grid"]["top_k_values"], [64])
        self.assertIn(
            "--generation-zspace-top-k",
            report["generation_control_bridge_cli_args"],
        )
        self.assertIn(
            "--zspace-top-k-values",
            report["generation_control_sweep_cli_args"],
        )
        self.assertEqual(compare_args.label, ["generic"])
        self.assertEqual(compare_args.sweeps, [out_path])
        self.assertEqual(
            comparison["row_type"],
            "zspace_generation_control_sweep_comparison",
        )
        self.assertEqual(comparison["labels"], "generic")
        self.assertIn("zspace_generation_control_compare ", comparison_lines[0])
        self.assertEqual(
            nested_report["runs"][0]["generation"]["row_type"],
            "hf_finetune_generation_report",
        )

    def test_generation_control_profile_runtime_defaults_flow_to_generic_wrapper(
        self,
    ) -> None:
        sweep_module = load_generic_generation_control_sweep_example()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "profiles.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema": "spiraltorch.hf_finetune_model_configs.v1",
                        "default_profile": "remote-causal",
                        "profiles": [
                            {
                                "id": "remote-causal",
                                "model_name": "org/remote-causal",
                                "tokenizer_name": "org/remote-tokenizer",
                                "architecture": "causal_lm",
                                "generation": {
                                    "max_new_tokens": 17,
                                    "do_sample": True,
                                    "temperature": 0.75,
                                    "top_k": 24,
                                    "zspace_top_k": 48,
                                    "zspace_curvature": -0.03,
                                    "zspace_temperature": 1.1,
                                    "zspace_entropy_target": 2.5,
                                    "repression_window": 12,
                                    "repression_strength": 0.6,
                                    "last_token_repression": 0.4,
                                },
                                "runtime": {
                                    "allow_remote": True,
                                    "trust_remote_code": True,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = sweep_module.parse_args(
                [
                    "--dry-run",
                    "--model-configs",
                    str(config_path),
                    "--prompt",
                    "SpiralTorch is",
                ]
            )
            report = sweep_module.run_sweep(args)

        self.assertTrue(args.allow_remote)
        self.assertTrue(args.trust_remote_code)
        self.assertEqual(report["row_type"], "hf_zspace_generation_control_sweep")
        self.assertTrue(report["allow_remote"])
        self.assertTrue(report["trust_remote_code"])
        self.assertEqual(report["max_new_tokens"], 17)
        self.assertTrue(report["do_sample"])
        self.assertEqual(report["sample_top_k"], 24)
        self.assertEqual(report["runs"][1]["config"]["top_k"], 48)
        self.assertEqual(report["generation_control_profile_config"]["top_k"], 48)
        self.assertEqual(report["generation_control_grid"]["top_k_values"], [48])
        self.assertIn(
            "--generation-zspace-top-k",
            report["generation_control_bridge_cli_args"],
        )

    def test_installed_hf_generation_control_cli_dry_run_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_path = root / "control-sweep.json"
            compare_path = root / "compare.json"
            code = hf_cli.zspace_generation_control_sweep_main(
                [
                    "--dry-run",
                    "--model-configs",
                    str(MODEL_CONFIGS_PATH),
                    "--model-profile",
                    "qwen2-0.5b-local-smoke",
                    "--prompt",
                    "SpiralTorch is",
                    "--out",
                    str(out_path),
                ]
            )
            with mock.patch.object(hf_cli, "_run_example") as run_example:
                compare_code = hf_cli.zspace_generation_control_compare_main(
                    [
                        str(out_path),
                        "--label",
                        "installed",
                        "--out",
                        str(compare_path),
                    ]
                )
                run_example.assert_not_called()
            report = json.loads(out_path.read_text())
            comparison = json.loads(compare_path.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(compare_code, 0)
        self.assertEqual(report["model_name"], "Qwen/Qwen2-0.5B")
        self.assertEqual(report["row_type"], "hf_zspace_generation_control_sweep")
        self.assertEqual(report["tokenizer_name"], "Qwen/Qwen2-0.5B")
        self.assertEqual(
            report["model_profile"]["profile_id"],
            "qwen2-0.5b-local-smoke",
        )
        self.assertEqual(report["max_new_tokens"], 128)
        self.assertEqual(report["runs"][1]["config"]["top_k"], 96)
        self.assertEqual(report["generation_control_profile_config"]["top_k"], 96)
        self.assertEqual(
            report["generation_control_resolved_config"]["top_k"],
            96,
        )
        self.assertIn(
            "--zspace-top-k-values",
            report["generation_control_sweep_cli_args"],
        )
        self.assertEqual(
            comparison["row_type"],
            "zspace_generation_control_sweep_comparison",
        )
        self.assertEqual(comparison["model_count"], 1)

    def test_installed_hf_checkpoint_control_cli_uses_generic_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(hf_cli, "_run_example") as run_example:
                default_code = hf_cli.checkpoint_generation_control_main(
                    [
                        "--dry-run",
                        "--no-compare",
                        "--run-dir",
                        str(root / "default-run"),
                        "--checkpoint",
                        "checkpoint-2048",
                        "--run-card",
                        str(root / "checkpoint-control-default.json"),
                    ]
                )
                code = hf_cli.checkpoint_generation_control_main(
                    [
                        "--dry-run",
                        "--no-compare",
                        "--run-dir",
                        str(root / "run"),
                        "--checkpoint",
                        "checkpoint-2048",
                        "--model-configs",
                        str(MODEL_CONFIGS_PATH),
                        "--model-profile",
                        "tiny-gpt2-ci",
                        "--run-card",
                        str(root / "checkpoint-control.json"),
                    ]
                )
                run_example.assert_not_called()
            default_report = json.loads(
                (root / "checkpoint-control-default.json").read_text()
            )
            report = json.loads((root / "checkpoint-control.json").read_text())

        default_command = default_report["sweeps"][0]["command"]
        command = report["sweeps"][0]["command"]
        self.assertEqual(default_code, 0)
        self.assertEqual(
            default_report["model_profile"]["profile_id"],
            st.HF_FINETUNE_DEFAULT_MODEL_PROFILE,
        )
        self.assertEqual(
            default_report["tokenizer_name"],
            "EleutherAI/pythia-70m-deduped",
        )
        self.assertEqual(
            default_command[default_command.index("--model-profile") + 1],
            st.HF_FINETUNE_DEFAULT_MODEL_PROFILE,
        )
        self.assertEqual(code, 0)
        self.assertEqual(Path(command[1]).name, "hf_zspace_generation_control_sweep.py")
        self.assertEqual(report["row_type"], "hf_checkpoint_generation_control")
        self.assertEqual(report["tokenizer_name"], "sshleifer/tiny-gpt2")
        self.assertEqual(
            command[command.index("--model-configs") + 1],
            str(MODEL_CONFIGS_PATH),
        )
        self.assertEqual(command[command.index("--model-profile") + 1], "tiny-gpt2-ci")

    def test_installed_hf_checkpoint_control_cli_accepts_script_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            custom_sweep = root / "custom_sweep.py"
            custom_compare = root / "custom_compare.py"
            custom_curve = root / "custom_curve.py"
            with mock.patch.object(hf_cli, "_run_example") as run_example:
                code = hf_cli.checkpoint_generation_control_main(
                    [
                        "--dry-run",
                        "--run-dir",
                        str(root / "run"),
                        "--checkpoint",
                        "checkpoint-2048",
                        "--sweep-script",
                        str(custom_sweep),
                        "--compare-script",
                        str(custom_compare),
                        "--curve-script",
                        str(custom_curve),
                        "--curve-out",
                        str(root / "curve.json"),
                        "--run-card",
                        str(root / "checkpoint-control.json"),
                    ]
                )
                run_example.assert_not_called()
            report = json.loads((root / "checkpoint-control.json").read_text())

        self.assertEqual(code, 0)
        self.assertEqual(report["sweeps"][0]["command"][1], str(custom_sweep))
        self.assertEqual(report["compare"]["command"][1], str(custom_compare))
        self.assertEqual(report["curve"]["command"][1], str(custom_curve))

    def test_repetition_report_scores_repeated_ngrams(self) -> None:
        module = load_generation_control_sweep_example()
        loop = module.text_repetition_report("a b c a b c a b c")
        clean = module.text_repetition_report("a b c d e f")

        self.assertGreater(loop["loop_score"], clean["loop_score"])
        self.assertEqual(loop["max_ngram_repetition"], 3)


if __name__ == "__main__":
    unittest.main()
