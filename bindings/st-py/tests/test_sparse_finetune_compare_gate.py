import argparse
import contextlib
import io
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def install_spiraltorch_stub():
    spiraltorch = types.ModuleType("spiraltorch")
    spiraltorch.__path__ = [str(ROOT / "spiraltorch")]
    spiraltorch.dataset = types.SimpleNamespace(BYTE_LM_VOCAB=256)

    class FakeTensor:
        def __init__(self, rows, cols, data=None):
            self.rows = rows
            self.cols = cols
            self._data = [0.0] * (rows * cols) if data is None else list(data)

        def data(self):
            return list(self._data)

        def shape(self):
            return (self.rows, self.cols)

        def transpose(self):
            transposed = []
            for col in range(self.cols):
                for row in range(self.rows):
                    transposed.append(self._data[row * self.cols + col])
            return FakeTensor(self.cols, self.rows, transposed)

    spiraltorch.Tensor = FakeTensor

    nn = types.ModuleType("spiraltorch.nn")
    for name in [
        "Linear",
        "LoraLinear",
        "Relu",
        "Sequential",
        "SoftmaxCrossEntropy",
        "ZSpaceProjector",
    ]:
        setattr(nn, name, object)
    nn.compare_sparse_finetune_summaries = lambda *_args, **_kwargs: {
        "accepted_changed": False,
        "passed": True,
    }
    nn.sparse_classification_delta = lambda *_args, **_kwargs: {}
    spiraltorch.nn = nn
    sys.modules["spiraltorch"] = spiraltorch
    sys.modules["spiraltorch.nn"] = nn


def load_example(name):
    install_spiraltorch_stub()
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_compare_helper():
    install_spiraltorch_stub()
    path = EXAMPLES / "sparse_finetune_compare.py"
    spec = importlib.util.spec_from_file_location("sparse_finetune_compare", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sparse_finetune_compare"] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint_helper():
    install_spiraltorch_stub()
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    path = EXAMPLES / "checkpoint_preflight.py"
    spec = importlib.util.spec_from_file_location("checkpoint_preflight", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["checkpoint_preflight"] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def fake_transformers_module(*, tokenizer_error=False, model_error=False):
    previous = sys.modules.get("transformers")
    calls = {"config": [], "tokenizer": [], "model": []}
    fake = types.ModuleType("transformers")
    fake.__version__ = "9.9.9"

    class FakeConfig:
        model_type = "llama"
        architectures = ["LlamaForCausalLM"]
        vocab_size = 320
        hidden_size = 32
        num_hidden_layers = 2
        num_attention_heads = 4
        max_position_embeddings = 2048

    class FakeTokenizer:
        vocab_size = 319
        model_max_length = 4096

        def __len__(self):
            return 320

        def __call__(self, prompt, return_tensors=None):
            del return_tensors
            return {"input_ids": [[1, 2, min(319, len(prompt))]]}

        def decode(self, token_id):
            if isinstance(token_id, (list, tuple)):
                token_id = token_id[0]
            return f"<tok:{int(token_id)}>"

    class FakeParam:
        def __init__(self, count):
            self.count = count

        def numel(self):
            return self.count

    class FakeModel:
        def eval(self):
            return self

        def parameters(self):
            return [FakeParam(3), FakeParam(4)]

        def __call__(self, **kwargs):
            del kwargs
            return types.SimpleNamespace(
                logits=[[[0.1, 0.4, -0.2, 1.1]]],
                hidden_states=[
                    [[[0.0, 0.0, 0.0]]],
                    [[[0.5, -0.25, 0.75]]],
                ],
            )

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["config"].append((path, dict(kwargs)))
            return FakeConfig()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"].append((path, dict(kwargs)))
            if tokenizer_error:
                raise RuntimeError("tokenizer fixture missing")
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"].append((path, dict(kwargs)))
            if model_error:
                raise RuntimeError("model fixture missing")
            return FakeModel()

    fake.AutoConfig = FakeAutoConfig
    fake.AutoTokenizer = FakeAutoTokenizer
    fake.AutoModelForCausalLM = FakeAutoModelForCausalLM
    sys.modules["transformers"] = fake
    try:
        yield fake, calls
    finally:
        if previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = previous


def old_compare_signature(current, baseline, **kwargs):
    unsupported = [
        "require_accepted_match",
        "min_target_loss_margin",
        "min_retention_loss_margin",
        "min_retention_accuracy_margin",
        "min_retention_perplexity_margin",
        "max_target_retention_gap_regression",
        "max_target_retention_ratio_regression",
        "min_target_retention_ratio",
    ]
    for key in unsupported:
        if key in kwargs:
            break
    else:
        key = None
    if key is not None:
        raise TypeError(
            "compare_sparse_finetune_summaries() got an unexpected keyword "
            f"argument '{key}'"
        )
    return {
        "accepted_changed": current["accepted"] != baseline["accepted"],
        "passed": True,
    }


def new_compare_signature(seen_kwargs):
    def compare(current, baseline, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return {
            "accepted_changed": current["accepted"] != baseline["accepted"],
            "passed": not (
                kwargs["require_accepted_match"]
                and current["accepted"] != baseline["accepted"]
            ),
        }

    return compare


class SparseFineTuneCompareGateTests(unittest.TestCase):
    def test_helper_passes_accepted_gate_to_new_extension_signature(self):
        baseline = {"accepted": True}
        current = {"accepted": False}
        helper = load_compare_helper()
        seen_kwargs = []
        helper.compare_sparse_finetune_summaries = new_compare_signature(seen_kwargs)
        comparison = helper.compare_summaries(
            current,
            baseline,
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            min_target_loss_margin=None,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=True,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
        )
        self.assertTrue(seen_kwargs[0]["require_accepted_match"])
        self.assertTrue(comparison["accepted_changed"])
        self.assertFalse(comparison["passed"])

    def test_helper_gates_accepted_change_with_old_extension_signature(self):
        baseline = {"accepted": True}
        current = {"accepted": False}
        helper = load_compare_helper()
        helper.compare_sparse_finetune_summaries = old_compare_signature
        comparison = helper.compare_summaries(
            current,
            baseline,
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            min_target_loss_margin=None,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=True,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
        )
        self.assertTrue(comparison["accepted_changed"])
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["current_target_loss_margin"], 0.0)
        self.assertEqual(comparison["baseline_retention_loss_margin"], 0.0)

    def test_helper_backfills_target_retention_gates_with_old_extension_signature(self):
        baseline = {
            "accepted": True,
            "target_loss_delta": 1.2,
            "retention_loss_delta": 0.4,
            "target_min_loss_delta": 0.0,
            "best_retention_loss_increase": 0.0,
            "best_retention_accuracy_drop": 0.0,
            "best_retention_perplexity_increase": 0.0,
            "retention_max_loss_increase": 1.0,
            "retention_max_accuracy_drop": 1.0,
        }
        current = dict(baseline, target_loss_delta=1.0)
        helper = load_compare_helper()
        helper.compare_sparse_finetune_summaries = old_compare_signature
        comparison = helper.compare_summaries(
            current,
            baseline,
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            max_target_retention_gap_regression=0.1,
            max_target_retention_ratio_regression=0.25,
            min_target_loss_margin=None,
            min_target_retention_ratio=2.75,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=False,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
        )
        self.assertFalse(comparison["passed"])
        self.assertAlmostEqual(comparison["target_retention_gap_regression"], 0.2)
        self.assertAlmostEqual(comparison["target_retention_ratio_regression"], 0.5)
        self.assertAlmostEqual(comparison["target_retention_ratio_shortfall"], 0.25)
        self.assertAlmostEqual(comparison["current_target_retention_ratio"], 2.5)

    def test_examples_share_sparse_finetune_compare_helper(self):
        helper = load_compare_helper()
        for name in [
            "byte_lm_replay_sweep",
            "byte_lm_mlp_lora_sweep",
            "byte_lm_zspace_compare",
            "byte_lm_handoff_strategy_compare",
        ]:
            with self.subTest(name=name):
                module = load_example(name)
                self.assertIs(module.compare_summaries, helper.compare_summaries)
        for name in [
            "byte_lm_lora_adapter",
            "byte_lm_mlp_lora_adapter",
        ]:
            with self.subTest(name=name):
                module = load_example(name)
                self.assertIs(module.compare_single_summary, helper.compare_single_summary)
                self.assertIs(module.write_summary_jsonl, helper.write_summary_jsonl)
                self.assertIs(module.load_single_summary_jsonl, helper.load_single_summary_jsonl)

    def test_examples_share_checkpoint_preflight_helper(self):
        helper = load_checkpoint_helper()
        for name in [
            "byte_lm_lora_adapter",
            "byte_lm_mlp_lora_adapter",
            "byte_lm_mlp_lora_sweep",
        ]:
            with self.subTest(name=name):
                module = load_example(name)
                self.assertIs(module.preflight_and_load, helper.preflight_and_load)
                if name in {"byte_lm_mlp_lora_adapter", "byte_lm_mlp_lora_sweep"}:
                    self.assertIs(
                        module.apply_checkpoint_projection,
                        helper.apply_checkpoint_projection,
                    )
                if name == "byte_lm_mlp_lora_adapter":
                    self.assertIs(
                        module.project_checkpoint_tensors,
                        helper.project_checkpoint_tensors,
                    )

    def test_handoff_strategy_compare_selects_exact_baseline(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        self.assertEqual(
            [spec["label"] for spec in module.selected_strategies(["hf_overlap_resize"])],
            ["hf_exact", "hf_overlap_resize"],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_strategies(["hf_exact"])],
            ["hf_exact"],
        )

    def test_handoff_strategy_compare_selects_target_cases(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        self.assertEqual(
            [case["label"] for case in module.selected_cases(None)],
            [module.DEFAULT_CASE_LABEL],
        )
        self.assertEqual(
            [case["label"] for case in module.selected_cases(["route_cats"])],
            ["route_cats"],
        )

    def test_handoff_strategy_compare_builds_overlap_rules(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        source_state = {
            "embed::weight": module.st.Tensor(module.VOCAB, module.HIDDEN),
            "embed::bias": module.st.Tensor(1, module.HIDDEN),
            "head::weight": module.st.Tensor(module.HIDDEN, module.VOCAB),
            "head::bias": module.st.Tensor(1, module.VOCAB),
        }
        strategy = [
            spec
            for spec in module.STRATEGY_SPECS
            if spec["label"] == "hf_overlap_resize"
        ][0]
        checkpoint, rules, include_extra_keys, shapes = module.external_state_for_strategy(
            source_state,
            strategy,
            key_preset="llama",
        )
        self.assertEqual(
            shapes,
            (
                module.VOCAB + strategy["extra_vocab_rows"],
                module.HIDDEN + strategy["extra_hidden_cols"],
                module.VOCAB + strategy["extra_head_rows"],
            ),
        )
        self.assertEqual(len(include_extra_keys), 1)
        self.assertIn("model.layers.0.input_layernorm.weight", checkpoint)
        self.assertEqual(
            rules["model.embed_tokens.weight"],
            {"target": "embed::weight", "transform": "copy_overlap_zeros"},
        )
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose_copy_overlap_zeros"},
        )

    def test_handoff_strategy_compare_exposes_projection_audit_fields(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        exact = [
            spec for spec in module.STRATEGY_SPECS if spec["label"] == "hf_exact"
        ][0]
        zspace = [
            spec
            for spec in module.STRATEGY_SPECS
            if spec["label"] == "hf_zspace_projected"
        ][0]
        self.assertEqual(
            module.projection_audit_fields(exact),
            {
                "checkpoint_projection": "none",
                "checkpoint_projection_strength": None,
                "checkpoint_projection_curvature": None,
                "checkpoint_projection_frequency": None,
            },
        )
        self.assertEqual(
            module.projection_audit_fields(zspace),
            {
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": module.ZSPACE_STRENGTH,
                "checkpoint_projection_curvature": module.ZSPACE_CURVATURE,
                "checkpoint_projection_frequency": module.ZSPACE_FREQUENCY,
            },
        )

    def test_handoff_strategy_compare_configures_zspace_projection_args(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        args = types.SimpleNamespace(
            strategies=["hf_zspace_projected"],
            zspace_strength=0.25,
            zspace_curvature=-0.5,
            zspace_frequency=0.9,
        )
        strategies = module.configured_strategies(args)
        self.assertEqual(
            [strategy["label"] for strategy in strategies],
            ["hf_exact", "hf_zspace_projected"],
        )
        zspace = strategies[1]
        self.assertEqual(zspace["zspace_strength"], 0.25)
        self.assertEqual(zspace["zspace_curvature"], -0.5)
        self.assertEqual(zspace["zspace_frequency"], 0.9)
        self.assertEqual(
            module.projection_audit_fields(zspace),
            {
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 0.25,
                "checkpoint_projection_curvature": -0.5,
                "checkpoint_projection_frequency": 0.9,
            },
        )

    def test_handoff_strategy_compare_configures_healthy_zspace_preset(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        args = types.SimpleNamespace(
            strategies=["hf_zspace_projected"],
            zspace_strength=module.ZSPACE_STRENGTH,
            zspace_strengths=None,
            zspace_curvature=module.ZSPACE_CURVATURE,
            zspace_curvatures=None,
            zspace_frequency=module.ZSPACE_FREQUENCY,
            zspace_frequencies=None,
            zspace_preset="healthy",
        )
        strategies = module.configured_strategies(args)
        self.assertEqual(
            [strategy["label"] for strategy in strategies],
            ["hf_exact", "hf_zspace_s1_cm0p04_f0p65"],
        )
        zspace = strategies[1]
        self.assertEqual(zspace["strategy_family"], "hf_zspace_projected")
        self.assertEqual(zspace["zspace_strength"], 1.0)
        self.assertEqual(zspace["zspace_curvature"], -0.04)
        self.assertEqual(zspace["zspace_frequency"], module.ZSPACE_FREQUENCY)

    def test_handoff_strategy_compare_expands_zspace_projection_grid(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        args = types.SimpleNamespace(
            strategies=["hf_zspace_projected"],
            zspace_strength=0.5,
            zspace_strengths="0.25,0.5",
            zspace_curvature=-1.0,
            zspace_curvatures="-0.5",
            zspace_frequency=0.65,
            zspace_frequencies="0.65,0.9",
        )
        strategies = module.configured_strategies(args)
        self.assertEqual(
            [strategy["label"] for strategy in strategies],
            [
                "hf_exact",
                "hf_zspace_s0p25_cm0p5_f0p65",
                "hf_zspace_s0p25_cm0p5_f0p9",
                "hf_zspace_s0p5_cm0p5_f0p65",
                "hf_zspace_s0p5_cm0p5_f0p9",
            ],
        )
        self.assertEqual(strategies[1]["strategy_family"], "hf_zspace_projected")
        self.assertEqual(strategies[1]["zspace_strength"], 0.25)
        self.assertEqual(strategies[1]["zspace_curvature"], -0.5)
        self.assertEqual(strategies[1]["zspace_frequency"], 0.65)

    def test_handoff_strategy_compare_keys_rows_by_case_and_strategy(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        rows = [
            {"case": "case_a", "strategy": "hf_exact"},
            {"case": "case_b", "strategy": "hf_exact"},
            {"case": "case_a", "strategy": "hf_zspace"},
        ]
        self.assertEqual(
            sorted(module.rows_by_case_strategy(rows, "current")),
            ["case_a::hf_exact", "case_a::hf_zspace", "case_b::hf_exact"],
        )
        self.assertEqual(
            module.row_key({"strategy": "hf_exact"}),
            f"{module.DEFAULT_CASE_LABEL}::hf_exact",
        )
        with self.assertRaisesRegex(ValueError, "duplicate case/strategy"):
            module.rows_by_case_strategy(rows + [rows[0]], "current")

    def test_handoff_strategy_compare_aggregates_strategy_rows(self):
        module = load_example("byte_lm_handoff_strategy_compare")

        def row(strategy, case, target_delta, retention_delta, target_margin):
            return {
                "case": case,
                "strategy": strategy,
                "strategy_family": strategy,
                "accepted": True,
                "movement_ok": True,
                "target_loss_delta": target_delta,
                "retention_loss_delta": retention_delta,
                "retention_accuracy_delta": 0.0,
                "target_loss_margin": target_margin,
                "retention_loss_margin": 10.0 + retention_delta,
                "retention_accuracy_margin": 0.9 + retention_delta,
                "retention_perplexity_margin": 100.0 + retention_delta,
                "checkpoint_key_preset": "llama",
                "checkpoint_source_origin": "synthetic_hf_strategy",
                "checkpoint_vocab": module.VOCAB,
                "checkpoint_hidden": module.HIDDEN,
                "checkpoint_target_classes": module.VOCAB,
                "checkpoint_overlap_resize": False,
                "checkpoint_projection": "zspace" if "zspace" in strategy else "none",
                "checkpoint_projection_strength": 0.5 if "zspace" in strategy else None,
                "checkpoint_projection_curvature": -0.5 if "zspace" in strategy else None,
                "checkpoint_projection_frequency": 0.65 if "zspace" in strategy else None,
            }

        rows = [
            row("hf_exact", "case_a", 0.01, 0.02, 0.01),
            row("hf_exact", "case_b", 0.03, 0.04, 0.03),
            row("hf_zspace", "case_a", 0.05, 0.06, 0.05),
            row("hf_zspace", "case_b", 0.07, 0.08, 0.07),
        ]
        aggregates = module.aggregate_strategy_rows(rows)
        zspace = next(row for row in aggregates if row["strategy"] == "hf_zspace")
        self.assertEqual(zspace["row_type"], "strategy_aggregate")
        self.assertEqual(zspace["cases"], 2)
        self.assertEqual(zspace["case_labels"], "case_a,case_b")
        self.assertEqual(zspace["accepted_cases"], 2)
        self.assertEqual(zspace["rejected_cases"], 0)
        self.assertEqual(zspace["accepted_rate"], 1.0)
        self.assertEqual(zspace["movement_ok_cases"], 2)
        self.assertEqual(zspace["movement_not_ok_cases"], 0)
        self.assertEqual(zspace["movement_ok_rate"], 1.0)
        self.assertAlmostEqual(zspace["target_loss_delta_mean"], 0.06)
        self.assertAlmostEqual(zspace["retention_loss_delta_mean"], 0.07)
        self.assertAlmostEqual(zspace["target_loss_margin_min"], 0.05)
        self.assertEqual(module.aggregate_winner(aggregates)[0], "hf_zspace")

    def test_handoff_strategy_compare_rejects_nonboolean_flat_acceptance(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        row = {
            "case": "case_a",
            "strategy": "hf_zspace",
            "strategy_family": "hf_zspace",
            "accepted": "false",
            "movement_ok": True,
            "target_loss_delta": 0.05,
            "retention_loss_delta": 0.06,
            "retention_accuracy_delta": 0.0,
            "target_loss_margin": 0.05,
            "retention_loss_margin": 10.0,
        }
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            module.aggregate_strategy_rows([row])
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            module.strategy_winner([row])
        aggregate = {
            "strategy": "hf_zspace",
            "accepted_all": "false",
            "movement_ok_all": True,
            "target_loss_delta_mean": 0.05,
            "retention_loss_delta_mean": 0.06,
            "retention_accuracy_delta_mean": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "accepted_all is not boolean"):
            module.aggregate_winner([aggregate])

    def test_handoff_strategy_compare_rejects_duplicate_aggregate_cases(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        row = {
            "case": "case_a",
            "strategy": "hf_zspace",
            "strategy_family": "hf_zspace",
            "accepted": True,
            "movement_ok": True,
            "target_loss_delta": 0.05,
            "retention_loss_delta": 0.06,
            "retention_accuracy_delta": 0.0,
            "target_loss_margin": 0.05,
            "retention_loss_margin": 10.0,
            "checkpoint_key_preset": "llama",
            "checkpoint_source_origin": "synthetic_hf_strategy",
            "checkpoint_vocab": module.VOCAB,
            "checkpoint_hidden": module.HIDDEN,
            "checkpoint_target_classes": module.VOCAB,
            "checkpoint_overlap_resize": False,
            "checkpoint_projection": "zspace",
            "checkpoint_projection_strength": 0.5,
            "checkpoint_projection_curvature": -0.5,
            "checkpoint_projection_frequency": 0.65,
        }
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            module.aggregate_strategy_rows([dict(row), dict(row)])

    def test_handoff_strategy_compare_rejects_inconsistent_aggregate_rows(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        row = {
            "row_type": "strategy_aggregate",
            "strategy": "hf_zspace",
            "strategy_family": "hf_zspace",
            "cases": 2,
            "case_labels": "case_a",
            "accepted_all": True,
            "movement_ok_all": True,
            "target_loss_delta_mean": 0.06,
            "retention_loss_delta_mean": 0.07,
            "retention_accuracy_delta_mean": 0.0,
            "target_loss_margin_min": 0.05,
            "retention_loss_margin_min": 10.0,
        }
        with self.assertRaisesRegex(ValueError, "case_labels count 1 != cases 2"):
            module.aggregate_rows_by_strategy([row], "baseline")

    def test_handoff_strategy_compare_rejects_inconsistent_aggregate_rates(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        row = {
            "row_type": "strategy_aggregate",
            "strategy": "hf_zspace",
            "strategy_family": "hf_zspace",
            "cases": 2,
            "case_labels": "case_a,case_b",
            "accepted_cases": 1,
            "rejected_cases": 1,
            "accepted_rate": 1.0,
            "accepted_all": False,
            "movement_ok_cases": 2,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
            "target_loss_delta_mean": 0.06,
            "retention_loss_delta_mean": 0.07,
            "retention_accuracy_delta_mean": 0.0,
            "target_loss_margin_min": 0.05,
            "retention_loss_margin_min": 10.0,
        }
        with self.assertRaisesRegex(ValueError, "accepted_rate 1.000000000"):
            module.aggregate_rows_by_strategy([row], "baseline")
        row["accepted_rate"] = True
        with self.assertRaisesRegex(ValueError, "accepted_rate is not numeric"):
            module.aggregate_rows_by_strategy([row], "baseline")

    def test_handoff_strategy_compare_aggregate_coverage_gate_detects_rate_floors(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        rows = [
            {
                "row_type": "strategy_aggregate",
                "strategy": "hf_zspace",
                "strategy_family": "hf_zspace",
                "cases": 2,
                "case_labels": "case_a,case_b",
                "accepted_cases": 1,
                "rejected_cases": 1,
                "accepted_rate": 0.5,
                "accepted_all": False,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 1,
                "movement_ok_rate": 0.5,
                "movement_ok_all": False,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
            module.check_aggregate_coverage(rows, min_accepted_rate=0.75)
        with self.assertRaisesRegex(RuntimeError, "movement_ok_rate"):
            module.check_aggregate_coverage(rows, min_movement_ok_rate=0.75)

    def test_handoff_strategy_compare_aggregate_coverage_requires_acceptance_fields(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        row = {
            "row_type": "strategy_aggregate",
            "strategy": "hf_zspace",
            "strategy_family": "hf_zspace",
            "cases": 1,
            "case_labels": "adapter_ja",
            "rejected_cases": 0,
            "accepted_rate": 1.0,
            "accepted_all": True,
            "movement_ok_cases": 1,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
        }
        with self.assertRaisesRegex(ValueError, "missing integer accepted_cases"):
            module.check_aggregate_coverage([row])
        row["accepted_cases"] = 1
        row.pop("accepted_all")
        with self.assertRaisesRegex(ValueError, "missing boolean accepted_all"):
            module.check_aggregate_coverage([row])

    def test_handoff_strategy_compare_aggregate_coverage_gate_detects_case_scope(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        rows = [
            {
                "row_type": "strategy_aggregate",
                "strategy": "hf_zspace",
                "strategy_family": "hf_zspace",
                "cases": 1,
                "case_labels": "adapter_ja",
                "accepted_cases": 1,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "aggregate cases 1 below floor 2"):
            module.check_aggregate_coverage(rows, min_cases=2)
        with self.assertRaisesRegex(RuntimeError, "missing aggregate cases route_cats"):
            module.check_aggregate_coverage(
                rows,
                required_cases=["adapter_ja", "route_cats"],
            )

    def test_handoff_strategy_compare_aggregate_gate_detects_regressions(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        baseline = [
            {
                "row_type": "strategy_aggregate",
                "strategy": "hf_zspace",
                "cases": 2,
                "case_labels": "case_a,case_b",
                "accepted_all": True,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.06,
                "retention_loss_delta_mean": 0.07,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.05,
                "retention_loss_margin_min": 10.0,
            }
        ]
        current = [
            dict(
                baseline[0],
                target_loss_delta_mean=0.04,
                retention_loss_delta_mean=0.05,
                target_loss_margin_min=0.03,
            )
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=True,
            max_aggregate_target_loss_regression=0.01,
            max_aggregate_retention_loss_regression=0.01,
            max_aggregate_accepted_rate_regression=None,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=0.04,
            min_aggregate_retention_loss_margin=None,
            require_checkpoint_match=False,
        )
        with self.assertRaisesRegex(RuntimeError, "aggregate target_loss_delta_mean"):
            module.compare_aggregate_rows(current, baseline, args)

    def test_handoff_strategy_compare_aggregate_gate_detects_coverage_regressions(self):
        module = load_example("byte_lm_handoff_strategy_compare")
        baseline = [
            {
                "row_type": "strategy_aggregate",
                "strategy": "hf_zspace",
                "strategy_family": "hf_zspace",
                "cases": 2,
                "case_labels": "case_a,case_b",
                "accepted_cases": 2,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 2,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.02,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.0,
                "retention_loss_margin_min": 10.0,
            }
        ]
        current = [
            dict(
                baseline[0],
                accepted_cases=1,
                rejected_cases=1,
                accepted_rate=0.5,
                accepted_all=False,
            )
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=False,
            max_aggregate_target_loss_regression=None,
            max_aggregate_retention_loss_regression=None,
            max_aggregate_accepted_rate_regression=0.25,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=None,
            min_aggregate_retention_loss_margin=None,
            require_checkpoint_match=False,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
                module.compare_aggregate_rows(current, baseline, args)
        self.assertIn("passed=False", output.getvalue())

    def test_mlp_lora_adapter_imports_external_hf_state_dict_source(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        helper = load_checkpoint_helper()
        module = load_example("byte_lm_mlp_lora_adapter")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.zeros(
                            module.VOCAB,
                            module.HIDDEN,
                        ),
                        "lm_head.weight": torch.zeros(
                            module.VOCAB,
                            module.HIDDEN,
                        ),
                        "model.layers.0.input_layernorm.weight": torch.ones(1, 2),
                        "unused.deep.weight": torch.ones(1, 1),
                    }
                },
                path,
            )
            args = types.SimpleNamespace(
                hf_state_dict=path,
                key_preset="llama",
                include_extra_keys=["model.layers.0.input_layernorm.weight"],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=False,
            )
            checkpoint, rules, loaded_files, shapes = (
                module.externalize_mlp_state_from_hf_state_dict(args)
            )

        self.assertEqual(loaded_files, [str(path)])
        self.assertEqual(shapes, (module.VOCAB, module.HIDDEN, module.VOCAB))
        self.assertEqual(checkpoint["model.embed_tokens.bias"].shape(), (1, module.HIDDEN))
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, module.VOCAB))
        self.assertIn("model.layers.0.input_layernorm.weight", checkpoint)
        self.assertNotIn("unused.deep.weight", checkpoint)
        self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

    def test_byte_hf_state_dict_writer_builds_preflight_compatible_fixture(self):
        writer = load_example("write_byte_lm_hf_state_dict")
        helper = load_checkpoint_helper()
        state = writer.build_byte_hf_state_dict(
            key_preset="llama",
            vocab=4,
            hidden=2,
            target_classes=3,
            include_biases=False,
            include_extra_key=True,
        )
        extra_key = writer.HF_UNUSED_KEYS["llama"]

        self.assertEqual(
            sorted(state),
            [
                "lm_head.weight",
                "model.embed_tokens.weight",
                extra_key,
            ],
        )
        self.assertEqual(
            helper.infer_hf_lm_module_shapes(state, key_preset="llama"),
            (4, 2, 3),
        )
        checkpoint, rules = helper.hf_lm_handoff_from_external_state(
            state,
            key_preset="llama",
            include_extra_keys=[extra_key],
        )
        self.assertEqual(checkpoint["model.embed_tokens.bias"].shape(), (1, 2))
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, 3))
        self.assertIn(extra_key, checkpoint)
        self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

    def test_mlp_lora_adapter_allows_explicit_external_overlap_resize(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        module = load_example("byte_lm_mlp_lora_adapter")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.zeros(
                            module.VOCAB + 2,
                            module.HIDDEN + 1,
                        ),
                        "lm_head.weight": torch.zeros(
                            module.VOCAB + 3,
                            module.HIDDEN + 1,
                        ),
                    }
                },
                path,
            )
            args = types.SimpleNamespace(
                hf_state_dict=path,
                key_preset="llama",
                include_extra_keys=[],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=True,
            )
            checkpoint, rules, loaded_files, shapes = (
                module.externalize_mlp_state_from_hf_state_dict(args)
            )

        self.assertEqual(loaded_files, [str(path)])
        self.assertEqual(shapes, (module.VOCAB + 2, module.HIDDEN + 1, module.VOCAB + 3))
        self.assertEqual(
            checkpoint["model.embed_tokens.weight"].shape(),
            (module.VOCAB, module.HIDDEN),
        )
        self.assertEqual(checkpoint["lm_head.weight"].shape(), (module.VOCAB, module.HIDDEN))
        self.assertEqual(checkpoint["model.embed_tokens.bias"].shape(), (1, module.HIDDEN))
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, module.VOCAB))
        self.assertEqual(
            rules["model.embed_tokens.weight"],
            {"target": "embed::weight", "transform": "copy_overlap_zeros"},
        )
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose_copy_overlap_zeros"},
        )

    def test_mlp_lora_adapter_auto_detects_external_hf_state_dict_preset(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        module = load_example("byte_lm_mlp_lora_adapter")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.zeros(
                            module.VOCAB + 1,
                            module.HIDDEN + 1,
                        ),
                        "lm_head.weight": torch.zeros(
                            module.VOCAB + 2,
                            module.HIDDEN + 1,
                        ),
                    }
                },
                path,
            )
            args = types.SimpleNamespace(
                hf_state_dict=path,
                key_preset="auto",
                include_extra_keys=[],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=True,
            )
            checkpoint, rules, loaded_files, shapes = (
                module.externalize_mlp_state_from_hf_state_dict(args)
            )

        self.assertEqual(args.key_preset, "llama")
        self.assertEqual(loaded_files, [str(path)])
        self.assertEqual(shapes, (module.VOCAB + 1, module.HIDDEN + 1, module.VOCAB + 2))
        self.assertEqual(
            checkpoint["model.embed_tokens.weight"].shape(),
            (module.VOCAB, module.HIDDEN),
        )
        self.assertEqual(checkpoint["lm_head.weight"].shape(), (module.VOCAB, module.HIDDEN))
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose_copy_overlap_zeros"},
        )

    def test_mlp_lora_sweep_imports_external_hf_state_dict_source(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.zeros(
                            module.VOCAB,
                            module.HIDDEN,
                        ),
                        "lm_head.weight": torch.zeros(
                            module.VOCAB,
                            module.HIDDEN,
                        ),
                        "model.layers.0.input_layernorm.weight": torch.ones(1, 2),
                        "unused.deep.weight": torch.ones(1, 1),
                    }
                },
                path,
            )
            args = types.SimpleNamespace(
                hf_state_dict=path,
                key_preset="llama",
                include_extra_keys=["model.layers.0.input_layernorm.weight"],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=False,
            )
            checkpoint, rules, loaded_files, shapes = (
                module.externalize_mlp_state_from_hf_state_dict(args)
            )

        self.assertEqual(loaded_files, [str(path)])
        self.assertEqual(shapes, (module.VOCAB, module.HIDDEN, module.VOCAB))
        self.assertEqual(checkpoint["model.embed_tokens.bias"].shape(), (1, module.HIDDEN))
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, module.VOCAB))
        self.assertIn("model.layers.0.input_layernorm.weight", checkpoint)
        self.assertNotIn("unused.deep.weight", checkpoint)
        self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

    def test_mlp_lora_sweep_auto_detects_external_hf_state_dict_preset(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.zeros(
                            module.VOCAB,
                            module.HIDDEN,
                        ),
                        "lm_head.weight": torch.zeros(
                            module.VOCAB,
                            module.HIDDEN,
                        ),
                    }
                },
                path,
            )
            args = types.SimpleNamespace(
                hf_state_dict=path,
                key_preset="auto",
                include_extra_keys=[],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=False,
            )
            checkpoint, rules, loaded_files, shapes = (
                module.externalize_mlp_state_from_hf_state_dict(args)
            )

        self.assertEqual(args.key_preset, "llama")
        self.assertEqual(loaded_files, [str(path)])
        self.assertEqual(shapes, (module.VOCAB, module.HIDDEN, module.VOCAB))
        self.assertEqual(checkpoint["model.embed_tokens.bias"].shape(), (1, module.HIDDEN))
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, module.VOCAB))
        self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")

    def test_mlp_lora_sweep_loads_external_lora_config_jsonl(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora-configs.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "row_type": "byte_lm_lora_config",
                        "label": "r18_a96_lr4p5",
                        "rank": 18,
                        "alpha": 96.0,
                        "adapter_lr_scale": 4.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            configs = module.lora_configs_with_external([path])
            selected = module.selected_configs(["r18_a96_lr4p5"], configs)

            self.assertEqual(selected[0]["label"], "r18_a96_lr4p5")
            self.assertEqual(selected[0]["rank"], 18)
            self.assertEqual(selected[0]["alpha"], 96.0)
            self.assertEqual(selected[0]["adapter_lr_scale"], 4.5)
            self.assertEqual(
                [config["label"] for config in module.selected_configs(None, configs)],
                ["r6_a32_lr3", "r12_a64_lr4", "r18_a96_lr4p5"],
            )
            with self.assertRaisesRegex(ValueError, "unknown LoRA config label"):
                module.selected_configs(["missing"], configs)

            duplicate = Path(tmp) / "duplicate-lora-configs.jsonl"
            duplicate.write_text(
                json.dumps(
                    {
                        "row_type": "byte_lm_lora_config",
                        "label": "r6_a32_lr3",
                        "rank": 18,
                        "alpha": 96.0,
                        "adapter_lr_scale": 4.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate LoRA config label"):
                module.lora_configs_with_external([duplicate])

        example_path = EXAMPLES / "byte_lm_mlp_lora_capacity_lanes.jsonl"
        example_configs = module.load_lora_config_jsonl(example_path)
        self.assertEqual(
            [config["label"] for config in example_configs],
            ["r18_a96_lr4p5", "r18_a128_lr5", "r24_a96_lr4p5", "r24_a128_lr5"],
        )

    def test_mlp_lora_sweep_expands_checkpoint_projection_grid(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(
            checkpoint_projection="zspace",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_strengths="0.25,0.5",
            checkpoint_projection_curvature=-1.0,
            checkpoint_projection_curvatures="-0.5",
            checkpoint_projection_frequency=0.65,
            checkpoint_projection_frequencies="0.65,0.9",
        )
        variants = module.checkpoint_projection_variants(args)
        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "zspace_s0p25_cm0p5_f0p65",
                "zspace_s0p25_cm0p5_f0p9",
                "zspace_s0p5_cm0p5_f0p65",
                "zspace_s0p5_cm0p5_f0p9",
            ],
        )
        config = {"label": "r12_a64_lr4", "rank": 12, "alpha": 64.0}
        projected = module.config_for_projection_variant(config, variants[0])
        self.assertEqual(
            projected["label"],
            "r12_a64_lr4::zspace_s0p25_cm0p5_f0p65",
        )
        self.assertEqual(projected["base_label"], "r12_a64_lr4")
        self.assertEqual(projected["projection_label"], variants[0]["label"])

    def test_mlp_lora_sweep_preserves_single_projection_config_label(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(
            checkpoint_projection="zspace",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_strengths=None,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_curvatures=None,
            checkpoint_projection_frequency=0.65,
            checkpoint_projection_frequencies=None,
        )
        variants = module.checkpoint_projection_variants(args)
        self.assertEqual(len(variants), 1)
        self.assertIsNone(variants[0]["label"])
        config = {"label": "r12_a64_lr4", "rank": 12, "alpha": 64.0}
        self.assertEqual(
            module.config_for_projection_variant(config, variants[0])["label"],
            "r12_a64_lr4",
        )

    def test_mlp_lora_sweep_configures_healthy_projection_preset(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(
            checkpoint_projection="none",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_strengths=None,
            checkpoint_projection_curvature=-1.0,
            checkpoint_projection_curvatures=None,
            checkpoint_projection_frequency=0.65,
            checkpoint_projection_frequencies=None,
            checkpoint_projection_preset="healthy",
        )
        variants = module.checkpoint_projection_variants(args)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["label"], "zspace_s1_cm0p04_f0p65")
        self.assertEqual(
            variants[0]["fields"]["checkpoint_projection_strength"],
            1.0,
        )
        self.assertEqual(
            variants[0]["fields"]["checkpoint_projection_curvature"],
            -0.04,
        )
        config = {"label": "r12_a64_lr4", "rank": 12, "alpha": 64.0}
        projected = module.config_for_projection_variant(config, variants[0])
        self.assertEqual(
            projected["label"],
            "r12_a64_lr4::zspace_s1_cm0p04_f0p65",
        )

    def test_mlp_lora_sweep_expands_checkpoint_source_gain_grid(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(
            checkpoint_source_gain=1.0,
            checkpoint_source_gains="1.0,2.5",
        )
        variants = module.checkpoint_source_gain_variants(args)
        self.assertEqual(
            [variant["label"] for variant in variants],
            ["gain_g1", "gain_g2p5"],
        )
        self.assertEqual(
            [variant["fields"]["checkpoint_source_gain"] for variant in variants],
            [1.0, 2.5],
        )

        projection_variant = {
            "label": "zspace_s1_cm0p04_f0p65",
            "fields": {},
        }
        config = {"label": "r12_a64_lr4", "rank": 12, "alpha": 64.0}
        projected = module.config_for_checkpoint_variant(
            config,
            projection_variant,
            variants[1],
        )
        self.assertEqual(
            projected["label"],
            "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g2p5",
        )
        self.assertEqual(projected["source_gain_label"], "gain_g2p5")

    def test_mlp_lora_sweep_expands_adapter_weight_decay_grid(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(adapter_weight_decays="0,0.01")
        variants = module.adapter_weight_decay_variants(args)
        self.assertEqual(
            [variant["label"] for variant in variants],
            ["wd0", "wd0p01"],
        )
        self.assertEqual(
            [variant["fields"]["adapter_weight_decay"] for variant in variants],
            [0.0, 0.01],
        )

        projection_variant = {"label": None, "fields": {}}
        source_gain_variant = {"label": None, "fields": {"checkpoint_source_gain": 1.0}}
        config = {
            "label": "r12_a64_lr4",
            "rank": 12,
            "alpha": 64.0,
            "adapter_lr_scale": 4.0,
        }
        projected = module.config_for_checkpoint_variant(
            config,
            projection_variant,
            source_gain_variant,
            variants[1],
        )
        self.assertEqual(projected["label"], "r12_a64_lr4::wd0p01")
        self.assertEqual(projected["adapter_weight_decay"], 0.01)
        self.assertEqual(projected["adapter_weight_decay_label"], "wd0p01")

        default_args = types.SimpleNamespace(adapter_weight_decays=None)
        default_variant = module.adapter_weight_decay_variants(default_args)[0]
        self.assertIsNone(default_variant["label"])
        self.assertEqual(default_variant["fields"]["adapter_weight_decay"], 0.0)

    def test_mlp_lora_sweep_expands_training_policy_grid(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(
            max_grad_norms="1,2",
            gradient_accumulation_steps_list="1,4",
        )
        variants = module.training_policy_variants(args)
        self.assertEqual(
            [variant["label"] for variant in variants],
            ["gn1::accum1", "gn1::accum4", "gn2::accum1", "gn2::accum4"],
        )
        self.assertEqual(
            [variant["fields"]["max_grad_norm"] for variant in variants],
            [1.0, 1.0, 2.0, 2.0],
        )
        self.assertEqual(
            [variant["fields"]["gradient_accumulation_steps"] for variant in variants],
            [1, 4, 1, 4],
        )

        projection_variant = {"label": None, "fields": {}}
        source_gain_variant = {"label": None, "fields": {"checkpoint_source_gain": 1.0}}
        weight_decay_variant = {"label": None, "fields": {"adapter_weight_decay": 0.0}}
        config = {
            "label": "r12_a64_lr4",
            "rank": 12,
            "alpha": 64.0,
            "adapter_lr_scale": 4.0,
        }
        projected = module.config_for_checkpoint_variant(
            config,
            projection_variant,
            source_gain_variant,
            weight_decay_variant,
            variants[1],
        )
        self.assertEqual(projected["label"], "r12_a64_lr4::gn1::accum4")
        self.assertEqual(projected["max_grad_norm"], 1.0)
        self.assertEqual(projected["gradient_accumulation_steps"], 4)
        self.assertEqual(projected["max_grad_norm_label"], "gn1")
        self.assertEqual(projected["gradient_accumulation_steps_label"], "accum4")

        default_args = types.SimpleNamespace(
            max_grad_norms=None,
            gradient_accumulation_steps_list=None,
        )
        default_variant = module.training_policy_variants(default_args)[0]
        self.assertIsNone(default_variant["label"])
        self.assertEqual(default_variant["fields"]["max_grad_norm"], 2.0)
        self.assertEqual(default_variant["fields"]["gradient_accumulation_steps"], 2)

    def test_mlp_lora_sweep_expands_ft_control_grid(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        args = types.SimpleNamespace(
            ft_epochs_list="6",
            target_min_loss_deltas="0,0.001",
            patiences="3",
            min_deltas="0.0005",
            lr_decay_patiences="2",
            lr_decay_factors="0.8",
            lr_decay_min_deltas="0.00025",
        )
        variants = module.ft_control_variants(args)
        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "ep6::tmin0::pat3::md0p0005::ldp2::ldf0p8::ldmd0p00025",
                "ep6::tmin0p001::pat3::md0p0005::ldp2::ldf0p8::ldmd0p00025",
            ],
        )
        self.assertEqual(variants[1]["fields"]["ft_epochs"], 6)
        self.assertEqual(
            variants[1]["fields"]["target_min_loss_delta_policy"],
            0.001,
        )
        self.assertEqual(variants[1]["fields"]["early_stopping_patience"], 3)
        self.assertEqual(variants[1]["fields"]["lr_decay_patience"], 2)
        self.assertEqual(variants[1]["fields"]["lr_decay_factor"], 0.8)

        projection_variant = {"label": None, "fields": {}}
        source_gain_variant = {"label": None, "fields": {"checkpoint_source_gain": 1.0}}
        weight_decay_variant = {"label": None, "fields": {"adapter_weight_decay": 0.0}}
        training_policy_variant = {
            "label": None,
            "fields": {
                "max_grad_norm": 2.0,
                "gradient_accumulation_steps": 2,
            },
            "max_grad_norm_label": None,
            "gradient_accumulation_steps_label": None,
        }
        config = {
            "label": "r12_a64_lr4",
            "rank": 12,
            "alpha": 64.0,
            "adapter_lr_scale": 4.0,
        }
        projected = module.config_for_checkpoint_variant(
            config,
            projection_variant,
            source_gain_variant,
            weight_decay_variant,
            training_policy_variant,
            variants[1],
        )
        self.assertEqual(
            projected["label"],
            "r12_a64_lr4::ep6::tmin0p001::pat3::md0p0005::ldp2::ldf0p8::ldmd0p00025",
        )
        self.assertEqual(projected["ft_epochs"], 6)
        self.assertEqual(projected["target_min_loss_delta_policy"], 0.001)
        self.assertEqual(projected["early_stopping_patience"], 3)

        default_args = types.SimpleNamespace(
            ft_epochs_list=None,
            target_min_loss_deltas=None,
            patiences=None,
            min_deltas=None,
            lr_decay_patiences=None,
            lr_decay_factors=None,
            lr_decay_min_deltas=None,
        )
        default_variant = module.ft_control_variants(default_args)[0]
        self.assertIsNone(default_variant["label"])
        self.assertEqual(default_variant["fields"]["ft_epochs"], 10)
        self.assertIsNone(default_variant["fields"]["early_stopping_patience"])

    def test_mlp_lora_sweep_loads_ft_control_jsonl_lanes(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ft-controls.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "row_type": "byte_lm_ft_control",
                                "label": "selective_ep2",
                                "ft_epochs": 2,
                                "early_stopping_patience": None,
                                "lr_decay_patience": None,
                            }
                        ),
                        json.dumps(
                            {
                                "row_type": "byte_lm_ft_control",
                                "ft_epochs": 6,
                                "target_min_loss_delta_policy": 0.0,
                                "early_stopping_patience": 3,
                                "early_stopping_min_delta": 0.0,
                                "lr_decay_patience": 2,
                                "lr_decay_factor": 0.8,
                                "lr_decay_min_delta": 0.0,
                            }
                        ),
                        json.dumps(
                            {
                                "row_type": "ignored_row",
                                "label": "not_a_lane",
                                "ft_epochs": 99,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = types.SimpleNamespace(
                ft_control_jsonls=[path],
                ft_epochs_list=None,
                target_min_loss_deltas=None,
                patiences=None,
                min_deltas=None,
                lr_decay_patiences=None,
                lr_decay_factors=None,
                lr_decay_min_deltas=None,
            )
            variants = module.ft_control_variants(args)

        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "selective_ep2",
                "ep6::tmin0::pat3::md0::ldp2::ldf0p8::ldmd0",
            ],
        )
        self.assertEqual(variants[0]["fields"]["ft_epochs"], 2)
        self.assertIsNone(variants[0]["fields"]["early_stopping_patience"])
        self.assertEqual(variants[0]["fields"]["lr_decay_factor"], 0.5)
        self.assertEqual(variants[1]["fields"]["ft_epochs"], 6)
        self.assertEqual(variants[1]["fields"]["early_stopping_patience"], 3)
        self.assertEqual(variants[1]["fields"]["lr_decay_patience"], 2)
        self.assertEqual(variants[1]["fields"]["lr_decay_factor"], 0.8)

    def test_mlp_lora_sweep_rejects_duplicate_ft_control_jsonl_label(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ft-controls.jsonl"
            rows = [
                {"row_type": "byte_lm_ft_control", "label": "same", "ft_epochs": 2},
                {"row_type": "byte_lm_ft_control", "label": "same", "ft_epochs": 6},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = types.SimpleNamespace(ft_control_jsonls=[path])
            with self.assertRaisesRegex(ValueError, "duplicate FT-control lane label"):
                module.ft_control_variants(args)

    def test_mlp_lora_sweep_selects_target_cases(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        self.assertEqual(
            [case["label"] for case in module.selected_cases(None)],
            [module.DEFAULT_CASE_LABEL],
        )
        self.assertEqual(
            [case["label"] for case in module.selected_cases(["route_cats"])],
            ["route_cats"],
        )

    def test_mlp_lora_sweep_loads_external_case_jsonl(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "row_type": "byte_lm_case",
                                "label": "long_ft_probe",
                                "source_docs": ["source alpha", "source beta"],
                                "target_docs": ["target 猫", "target 螺旋"],
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "row_type": "ignored_row",
                                "label": "not_a_case",
                                "source_text": "ignored",
                                "target_text": "ignored",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            specs = module.case_specs_with_external([path])

        selected = module.selected_cases(["long_ft_probe"], specs)
        self.assertEqual([case["label"] for case in selected], ["long_ft_probe"])
        self.assertEqual(selected[0]["source_text"], "source alpha\nsource beta")
        self.assertEqual(selected[0]["target_text"], "target 猫\ntarget 螺旋")

    def test_mlp_lora_sweep_rejects_duplicate_external_case_label(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "label": module.DEFAULT_CASE_LABEL,
                        "source_text": "source",
                        "target_text": "target",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate byte-LM case label"):
                module.case_specs_with_external([path])

    def test_mlp_lora_sweep_keys_rows_by_case_and_config(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        self.assertEqual(
            module.row_compare_key(
                {"case": module.DEFAULT_CASE_LABEL, "config": "r12_a64_lr4"}
            ),
            "r12_a64_lr4",
        )
        self.assertEqual(
            module.row_compare_key({"case": "route_cats", "config": "r12_a64_lr4"}),
            "route_cats::r12_a64_lr4",
        )
        with self.assertRaisesRegex(ValueError, "duplicate config"):
            module.rows_by_config(
                [
                    {"case": "route_cats", "config": "r12_a64_lr4"},
                    {"case": "route_cats", "config": "r12_a64_lr4"},
                ],
                "current",
            )

    def test_mlp_lora_sweep_aggregates_config_rows(self):
        module = load_example("byte_lm_mlp_lora_sweep")

        def row(
            config,
            case,
            target_delta,
            retention_delta,
            target_margin,
            guard_accepted_epochs,
            guard_retention_rejected_epochs,
            guard_target_stale_epochs,
        ):
            return {
                "case": case,
                "config": config,
                "base_config": "r12_a64_lr4",
                "checkpoint_projection_variant": "zspace_s0p5_cm0p5_f0p65",
                "adapter_weight_decay_variant": "wd0p01",
                "adapter_weight_decay": 0.01,
                "max_grad_norm_variant": "gn1p5",
                "max_grad_norm": 1.5,
                "gradient_accumulation_steps_variant": "accum4",
                "gradient_accumulation_steps": 4,
                "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
                "ft_epochs": 6,
                "target_min_loss_delta_policy": 0.001,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.0,
                "lr_decay_patience": 2,
                "lr_decay_factor": 0.8,
                "lr_decay_min_delta": 0.0,
                "ft_early_stopped": False,
                "ft_stop_epoch": None,
                "ft_lr_decay_steps": 1,
                "ft_final_hyper_learning_rate": 0.4,
                "ft_final_fallback_learning_rate": 0.08,
                "guard_epoch_counts_available": True,
                "guard_accepted_epochs": guard_accepted_epochs,
                "guard_retention_rejected_epochs": guard_retention_rejected_epochs,
                "guard_target_stale_epochs": guard_target_stale_epochs,
                "accepted": True,
                "movement_ok": True,
                "target_loss_delta": target_delta,
                "retention_loss_delta": retention_delta,
                "retention_accuracy_delta": 0.0,
                "target_loss_margin": target_margin,
                "retention_loss_margin": 10.0 + retention_delta,
                "retention_accuracy_margin": 0.9 + retention_delta,
                "retention_perplexity_margin": 100.0 + retention_delta,
                "checkpoint_key_preset": "llama",
                "checkpoint_source_origin": "hf_state_dict",
                "checkpoint_source_label": "llama32",
                "checkpoint_loaded_files": 1,
                "checkpoint_vocab": module.VOCAB,
                "checkpoint_hidden": module.HIDDEN,
                "checkpoint_target_classes": module.VOCAB,
                "checkpoint_overlap_resize": False,
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 0.5,
                "checkpoint_projection_curvature": -0.5,
                "checkpoint_projection_frequency": 0.65,
            }

        rows = [
            row(
                "r12_a64_lr4::zspace_s0p5_cm0p5_f0p65",
                "adapter_ja",
                0.01,
                0.02,
                0.01,
                6,
                0,
                0,
            ),
            row(
                "r12_a64_lr4::zspace_s0p5_cm0p5_f0p65",
                "route_cats",
                0.03,
                0.04,
                0.03,
                4,
                0,
                2,
            ),
        ]
        aggregates = module.aggregate_config_rows(rows)
        self.assertEqual(len(aggregates), 1)
        aggregate = aggregates[0]
        self.assertEqual(aggregate["row_type"], "config_aggregate")
        self.assertEqual(aggregate["cases"], 2)
        self.assertEqual(aggregate["case_labels"], "adapter_ja,route_cats")
        self.assertEqual(aggregate["accepted_cases"], 2)
        self.assertEqual(aggregate["rejected_cases"], 0)
        self.assertEqual(aggregate["accepted_rate"], 1.0)
        self.assertEqual(aggregate["movement_ok_cases"], 2)
        self.assertEqual(aggregate["movement_not_ok_cases"], 0)
        self.assertEqual(aggregate["movement_ok_rate"], 1.0)
        self.assertEqual(aggregate["guard_epoch_counts_available_cases"], 2)
        self.assertTrue(aggregate["guard_epoch_counts_available_all"])
        self.assertAlmostEqual(aggregate["guard_accepted_epochs_total"], 10.0)
        self.assertAlmostEqual(aggregate["guard_accepted_epochs_mean"], 5.0)
        self.assertAlmostEqual(aggregate["guard_accepted_epochs_max"], 6.0)
        self.assertAlmostEqual(aggregate["guard_retention_rejected_epochs_total"], 0.0)
        self.assertAlmostEqual(aggregate["guard_target_stale_epochs_total"], 2.0)
        self.assertAlmostEqual(aggregate["guard_target_stale_epochs_mean"], 1.0)
        self.assertAlmostEqual(aggregate["guard_target_stale_epochs_max"], 2.0)
        self.assertAlmostEqual(aggregate["guard_acceptance_rate_mean"], 5.0 / 6.0)
        self.assertAlmostEqual(aggregate["guard_acceptance_rate_min"], 4.0 / 6.0)
        self.assertAlmostEqual(aggregate["guard_retention_rejected_rate_mean"], 0.0)
        self.assertAlmostEqual(aggregate["guard_retention_rejected_rate_max"], 0.0)
        self.assertAlmostEqual(aggregate["guard_target_stale_rate_mean"], 1.0 / 6.0)
        self.assertAlmostEqual(aggregate["guard_target_stale_rate_max"], 2.0 / 6.0)
        self.assertAlmostEqual(aggregate["target_loss_delta_mean"], 0.02)
        self.assertAlmostEqual(aggregate["retention_loss_delta_mean"], 0.03)
        self.assertAlmostEqual(aggregate["target_loss_margin_min"], 0.01)
        self.assertAlmostEqual(aggregate["retention_accuracy_margin_mean"], 0.93)
        self.assertAlmostEqual(aggregate["retention_accuracy_margin_min"], 0.92)
        self.assertAlmostEqual(aggregate["retention_perplexity_margin_mean"], 100.03)
        self.assertAlmostEqual(aggregate["retention_perplexity_margin_min"], 100.02)
        self.assertEqual(aggregate["adapter_weight_decay_variant"], "wd0p01")
        self.assertEqual(aggregate["adapter_weight_decay"], 0.01)
        self.assertEqual(aggregate["max_grad_norm_variant"], "gn1p5")
        self.assertEqual(aggregate["max_grad_norm"], 1.5)
        self.assertEqual(aggregate["gradient_accumulation_steps_variant"], "accum4")
        self.assertEqual(aggregate["gradient_accumulation_steps"], 4)
        self.assertEqual(
            aggregate["ft_control_variant"],
            "ep6::tmin0p001::pat3::ldp2::ldf0p8",
        )
        self.assertEqual(aggregate["ft_epochs"], 6)
        self.assertEqual(aggregate["target_min_loss_delta_policy"], 0.001)
        self.assertEqual(aggregate["early_stopping_patience"], 3)
        self.assertEqual(aggregate["lr_decay_patience"], 2)
        self.assertIn("adapter_weight_decay=0.010000000", aggregate["training_policy_key"])
        self.assertIn(
            "ft_control_variant=ep6::tmin0p001::pat3::ldp2::ldf0p8",
            aggregate["training_policy_key"],
        )
        self.assertIn("ft_epochs=6.000000000", aggregate["training_policy_key"])
        self.assertEqual(aggregate["ft_early_stopped_cases"], 0)
        self.assertFalse(aggregate["ft_early_stopped_any"])
        self.assertEqual(aggregate["ft_lr_decay_steps_max"], 1)
        self.assertAlmostEqual(aggregate["ft_lr_decay_steps_mean"], 1.0)
        self.assertAlmostEqual(aggregate["ft_final_hyper_learning_rate_min"], 0.4)
        self.assertEqual(aggregate["checkpoint_source_label"], "llama32")
        self.assertEqual(aggregate["checkpoint_loaded_files"], 1)
        self.assertEqual(
            module.aggregate_winner(aggregates)[0],
            "r12_a64_lr4::zspace_s0p5_cm0p5_f0p65",
        )

    def test_mlp_lora_sweep_rejects_nonboolean_flat_acceptance(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "case": "adapter_ja",
            "config": "r12_a64_lr4",
            "base_config": "r12_a64_lr4",
            "checkpoint_projection_variant": None,
            "accepted": "false",
            "movement_ok": True,
            "target_loss_delta": 0.05,
            "retention_loss_delta": 0.06,
            "retention_accuracy_delta": 0.0,
            "target_loss_margin": 0.05,
            "retention_loss_margin": 10.0,
        }
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            module.aggregate_config_rows([row])
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            module.sweep_winner([row])
        aggregate = {
            "config": "r12_a64_lr4",
            "accepted_all": "false",
            "movement_ok_all": True,
            "target_loss_delta_mean": 0.05,
            "retention_loss_delta_mean": 0.06,
            "retention_accuracy_delta_mean": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "accepted_all is not boolean"):
            module.aggregate_winner([aggregate])

    def test_mlp_lora_sweep_aggregate_rejects_duplicate_case_rows(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "case": "adapter_ja",
            "config": "r12_a64_lr4",
            "base_config": "r12_a64_lr4",
            "checkpoint_projection_variant": None,
            "accepted": True,
            "movement_ok": True,
            "target_loss_delta": 0.01,
            "retention_loss_delta": 0.02,
            "retention_accuracy_delta": 0.0,
            "target_loss_margin": 0.01,
            "retention_loss_margin": 10.0,
            "checkpoint_key_preset": "llama",
            "checkpoint_source_origin": "hf_state_dict",
            "checkpoint_source_label": "llama32",
            "checkpoint_loaded_files": 1,
            "checkpoint_vocab": module.VOCAB,
            "checkpoint_hidden": module.HIDDEN,
            "checkpoint_target_classes": module.VOCAB,
            "checkpoint_overlap_resize": False,
            "checkpoint_projection": "none",
            "checkpoint_projection_strength": None,
            "checkpoint_projection_curvature": None,
            "checkpoint_projection_frequency": None,
        }
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            module.aggregate_config_rows([dict(row), dict(row)])

    def test_mlp_lora_sweep_rejects_stale_training_policy_key(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "case": "adapter_ja",
            "config": "r12_a64_lr4",
            "base_config": "r12_a64_lr4",
            "checkpoint_projection_variant": None,
            "checkpoint_source_gain_variant": None,
            "adapter_weight_decay_variant": None,
            "adapter_weight_decay": 0.0,
            "max_grad_norm_variant": None,
            "max_grad_norm": 2.0,
            "gradient_accumulation_steps_variant": None,
            "gradient_accumulation_steps": 2,
            "ft_control_variant": None,
            "ft_epochs": 10,
            "target_min_loss_delta_policy": 0.0,
            "early_stopping_patience": None,
            "early_stopping_min_delta": 0.0,
            "lr_decay_patience": None,
            "lr_decay_factor": 0.5,
            "lr_decay_min_delta": 0.0,
            "training_policy_key": "stale",
            "ft_early_stopped": False,
            "ft_stop_epoch": None,
            "ft_lr_decay_steps": 0,
            "ft_final_hyper_learning_rate": 0.5,
            "ft_final_fallback_learning_rate": 0.1,
            "accepted": True,
            "movement_ok": True,
            "target_loss_delta": 0.01,
            "retention_loss_delta": 0.02,
            "retention_accuracy_delta": 0.0,
            "target_loss_margin": 0.01,
            "retention_loss_margin": 10.0,
            "retention_accuracy_margin": 1.0,
            "retention_perplexity_margin": 100.0,
            "checkpoint_key_preset": "gpt2",
            "checkpoint_source_origin": "trained_dense",
            "checkpoint_source_label": "trained_dense",
            "checkpoint_loaded_files": 0,
            "checkpoint_vocab": module.VOCAB,
            "checkpoint_hidden": module.HIDDEN,
            "checkpoint_target_classes": module.VOCAB,
            "checkpoint_overlap_resize": False,
            "checkpoint_projection": "none",
            "checkpoint_projection_strength": None,
            "checkpoint_projection_curvature": None,
            "checkpoint_projection_frequency": None,
            "checkpoint_source_gain": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "stale aggregate field training_policy_key"):
            module.aggregate_config_rows([row])

    def test_mlp_lora_sweep_aggregate_acceptance_gate_detects_partial_acceptance(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        rows = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_cases": 1,
                "rejected_cases": 1,
                "accepted_rate": 0.5,
                "accepted_all": False,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 1,
                "movement_ok_rate": 0.5,
                "movement_ok_all": False,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "accepted 1/2"):
            module.check_aggregate_accepted_all(rows)

    def test_mlp_lora_sweep_aggregate_coverage_gate_detects_rate_floors(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        rows = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_cases": 1,
                "rejected_cases": 1,
                "accepted_rate": 0.5,
                "accepted_all": False,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 1,
                "movement_ok_rate": 0.5,
                "movement_ok_all": False,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
            module.check_aggregate_coverage(rows, min_accepted_rate=0.75)
        with self.assertRaisesRegex(RuntimeError, "movement_ok_rate"):
            module.check_aggregate_coverage(rows, min_movement_ok_rate=0.75)

    def test_mlp_lora_sweep_aggregate_coverage_requires_acceptance_fields(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "row_type": "config_aggregate",
            "config": "r12_a64_lr4",
            "cases": 1,
            "case_labels": "adapter_ja",
            "accepted_cases": 1,
            "rejected_cases": 0,
            "accepted_rate": 1.0,
            "accepted_all": True,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
        }
        with self.assertRaisesRegex(ValueError, "missing integer movement_ok_cases"):
            module.check_aggregate_coverage([row])
        row["movement_ok_cases"] = 1
        row.pop("accepted_all")
        with self.assertRaisesRegex(ValueError, "missing boolean accepted_all"):
            module.check_aggregate_coverage([row])

    def test_mlp_lora_sweep_aggregate_coverage_gate_detects_case_scope(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        rows = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 1,
                "case_labels": "adapter_ja",
                "accepted_cases": 1,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "aggregate cases 1 below floor 2"):
            module.check_aggregate_coverage(rows, min_cases=2)
        with self.assertRaisesRegex(RuntimeError, "missing aggregate cases route_cats"):
            module.check_aggregate_coverage(
                rows,
                required_cases=["adapter_ja", "route_cats"],
            )

    def test_mlp_lora_sweep_rejects_inconsistent_aggregate_rows(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "row_type": "config_aggregate",
            "config": "r12_a64_lr4",
            "cases": 2,
            "case_labels": "adapter_ja",
            "accepted_cases": 1,
            "rejected_cases": 1,
            "accepted_rate": 0.5,
            "accepted_all": False,
            "movement_ok_cases": 2,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
        }
        with self.assertRaisesRegex(ValueError, "case_labels count 1 != cases 2"):
            module.aggregate_rows_by_config([row], "baseline")

    def test_mlp_lora_sweep_rejects_inconsistent_aggregate_rates(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "row_type": "config_aggregate",
            "config": "r12_a64_lr4",
            "cases": 2,
            "case_labels": "adapter_ja,route_cats",
            "accepted_cases": 1,
            "rejected_cases": 1,
            "accepted_rate": 1.0,
            "accepted_all": False,
            "movement_ok_cases": 2,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
        }
        with self.assertRaisesRegex(ValueError, "accepted_rate 1.000000000"):
            module.aggregate_rows_by_config([row], "baseline")
        row["accepted_rate"] = True
        with self.assertRaisesRegex(ValueError, "accepted_rate is not numeric"):
            module.aggregate_rows_by_config([row], "baseline")

    def test_mlp_lora_sweep_aggregate_gate_detects_regressions(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        baseline = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_all": True,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.06,
                "retention_loss_delta_mean": 0.07,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.05,
                "retention_loss_margin_min": 10.0,
            }
        ]
        current = [
            dict(
                baseline[0],
                target_loss_delta_mean=0.04,
                retention_loss_delta_mean=0.05,
                target_loss_margin_min=0.03,
            )
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=True,
            max_aggregate_target_loss_regression=0.01,
            max_aggregate_retention_loss_regression=0.01,
            max_aggregate_accepted_rate_regression=None,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=0.04,
            min_aggregate_retention_loss_margin=None,
            require_checkpoint_match=False,
        )
        with self.assertRaisesRegex(RuntimeError, "aggregate target_loss_delta_mean"):
            module.compare_aggregate_rows(current, baseline, args)

    def test_mlp_lora_sweep_aggregate_gate_detects_retention_margin_floors(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        baseline = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_all": True,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.06,
                "retention_loss_delta_mean": 0.07,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.05,
                "retention_loss_margin_min": 10.0,
                "retention_accuracy_margin_min": 0.04,
                "retention_perplexity_margin_min": 0.2,
            }
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=False,
            max_aggregate_target_loss_regression=None,
            max_aggregate_retention_loss_regression=None,
            max_aggregate_accepted_rate_regression=None,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=None,
            min_aggregate_retention_loss_margin=None,
            min_aggregate_retention_accuracy_margin=0.05,
            min_aggregate_retention_perplexity_margin=0.25,
            require_checkpoint_match=False,
        )
        with self.assertRaisesRegex(RuntimeError, "retention_accuracy_margin_min"):
            module.compare_aggregate_rows(list(baseline), baseline, args)

    def test_mlp_lora_sweep_aggregate_gate_detects_coverage_regressions(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        baseline = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_all": True,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.02,
                "retention_accuracy_delta_mean": 0.0,
                "accepted_rate": 1.0,
                "movement_ok_rate": 1.0,
                "target_loss_margin_min": 0.0,
                "retention_loss_margin_min": 10.0,
            }
        ]
        current = [
            dict(
                baseline[0],
                accepted_all=False,
                movement_ok_all=False,
                accepted_rate=0.5,
                movement_ok_rate=0.5,
            )
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=False,
            max_aggregate_target_loss_regression=None,
            max_aggregate_retention_loss_regression=None,
            max_aggregate_accepted_rate_regression=0.25,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=None,
            min_aggregate_retention_loss_margin=None,
            require_checkpoint_match=False,
        )
        with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
            module.compare_aggregate_rows(current, baseline, args)

    def test_mlp_lora_sweep_aggregate_compare_allows_no_winner_without_winner_gate(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        baseline = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_all": False,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.02,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.0,
                "retention_loss_margin_min": 10.0,
            }
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=False,
            max_aggregate_target_loss_regression=0.0,
            max_aggregate_retention_loss_regression=0.0,
            max_aggregate_accepted_rate_regression=None,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=None,
            min_aggregate_retention_loss_margin=None,
            require_checkpoint_match=False,
        )
        self.assertEqual(
            module.compare_aggregate_rows(list(baseline), baseline, args),
            1,
        )

    def test_mlp_lora_sweep_aggregate_gate_detects_case_scope_drift(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        baseline = [
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_all": False,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.02,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.0,
                "retention_loss_margin_min": 10.0,
            }
        ]
        current = [
            dict(
                baseline[0],
                cases=1,
                case_labels="adapter_ja",
            )
        ]
        args = types.SimpleNamespace(
            require_aggregate_winner_match=False,
            max_aggregate_target_loss_regression=0.0,
            max_aggregate_retention_loss_regression=0.0,
            max_aggregate_accepted_rate_regression=None,
            max_aggregate_movement_ok_rate_regression=None,
            min_aggregate_target_loss_margin=None,
            min_aggregate_retention_loss_margin=None,
            require_checkpoint_match=False,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "aggregate case scope changed"):
                module.compare_aggregate_rows(current, baseline, args)
        self.assertIn("passed=False", output.getvalue())

    def test_mlp_lora_sweep_summary_compare_allows_no_winner_without_winner_gate(self):
        module = load_example("byte_lm_mlp_lora_sweep")
        row = {
            "case": "route_cats",
            "config": "r12_a64_lr4",
            "accepted": False,
            "movement_ok": False,
            "status": "guard_rejected",
            "target_loss_delta": 0.0,
            "retention_loss_delta": 0.0,
            "retention_accuracy_delta": 0.0,
            "target_loss_margin": 0.0,
            "retention_loss_margin": 10.0,
            "retention_accuracy_margin": 1.0,
            "movement_tolerance": 1e-6,
            "resume_hash": "same",
        }
        comparison = {
            "target_loss_delta_change": 0.0,
            "retention_loss_delta_change": 0.0,
            "target_loss_regression": 0.0,
            "retention_loss_regression": 0.0,
            "current_target_loss_margin": 0.0,
            "current_retention_loss_margin": 10.0,
            "current_retention_accuracy_margin": 1.0,
            "baseline_status": "guard_rejected",
            "current_status": "guard_rejected",
            "status_changed": False,
            "baseline_accepted": False,
            "current_accepted": False,
            "accepted_changed": False,
            "guard_changed": False,
            "baseline_movement_tolerance": 1e-6,
            "current_movement_tolerance": 1e-6,
            "movement_tolerance_changed": False,
            "baseline_resume_hash": "same",
            "current_resume_hash": "same",
            "resume_changed": False,
            "passed": True,
        }
        module.compare_summaries = lambda *_args, **_kwargs: dict(comparison)
        self.assertEqual(
            module.compare_summary_rows(
                [dict(row)],
                [dict(row)],
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            1,
        )

    def test_mlp_lora_source_compare_ranks_absolute_and_selective_winners(self):
        module = load_example("byte_lm_mlp_lora_source_compare")

        def row(source, target_delta, retention_delta):
            return {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "base_config": "r12_a64_lr4",
                "checkpoint_projection_variant": None,
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_cases": 2,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 2,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
                "target_loss_delta_mean": target_delta,
                "retention_loss_delta_mean": retention_delta,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": target_delta,
                "retention_loss_margin_min": 10.0,
                "checkpoint_key_preset": source,
                "checkpoint_source_label": source,
                "checkpoint_source_origin": "hf_state_dict",
                "checkpoint_overlap_resize": True,
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 1.0,
                "checkpoint_projection_curvature": -0.04,
                "checkpoint_projection_frequency": 0.65,
                "checkpoint_source_gain": 2.0 if source == "llama-3.2-3b" else 1.0,
                "adapter_weight_decay_variant": "wd0p01"
                if source == "llama-3.2-3b"
                else None,
                "adapter_weight_decay": 0.01 if source == "llama-3.2-3b" else 0.0,
                "max_grad_norm_variant": "gn1p5"
                if source == "llama-3.2-3b"
                else None,
                "max_grad_norm": 1.5 if source == "llama-3.2-3b" else 2.0,
                "gradient_accumulation_steps_variant": "accum4"
                if source == "llama-3.2-3b"
                else None,
                "gradient_accumulation_steps": 4
                if source == "llama-3.2-3b"
                else 2,
                "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8"
                if source == "llama-3.2-3b"
                else None,
                "ft_epochs": 6 if source == "llama-3.2-3b" else 10,
                "target_min_loss_delta_policy": 0.001
                if source == "llama-3.2-3b"
                else 0.0,
                "early_stopping_patience": 3
                if source == "llama-3.2-3b"
                else None,
                "early_stopping_min_delta": 0.0,
                "lr_decay_patience": 2
                if source == "llama-3.2-3b"
                else None,
                "lr_decay_factor": 0.8 if source == "llama-3.2-3b" else 0.5,
                "lr_decay_min_delta": 0.0,
            }

        candidates = module.source_candidates_from_rows(
            [
                row("gpt2-bare", 0.03, 0.02),
                row("llama-3.2-3b", 0.02, 0.004),
            ],
            Path("aggregate.jsonl"),
        )
        self.assertEqual(
            module.ranked_candidates(candidates, "target_loss_delta_mean")[0][
                "checkpoint_source_label"
            ],
            "gpt2-bare",
        )
        selective_winner, value = module.source_winner(
            candidates,
            "target_retention_gap_mean",
        )
        self.assertEqual(selective_winner["checkpoint_source_label"], "llama-3.2-3b")
        self.assertAlmostEqual(value, 0.016)
        self.assertAlmostEqual(
            selective_winner["target_retention_ratio"],
            5.0,
        )
        self.assertIn("ft_epochs=6.000000000", selective_winner["training_policy_key"])
        self.assertIn(
            "ft_control_variant=ep6::tmin0p001::pat3::ldp2::ldf0p8",
            selective_winner["training_policy_key"],
        )
        profiles = {
            row["source_profile"]: row
            for row in module.source_profile_rows(candidates)
        }
        self.assertEqual(
            profiles["strong_effect"]["checkpoint_source_label"],
            "gpt2-bare",
        )
        self.assertEqual(
            profiles["selective_gap"]["checkpoint_source_label"],
            "llama-3.2-3b",
        )
        self.assertEqual(
            profiles["selective_ratio"]["checkpoint_source_label"],
            "llama-3.2-3b",
        )
        self.assertEqual(
            profiles["selective_ratio"]["training_policy_key"],
            selective_winner["training_policy_key"],
        )
        self.assertEqual(
            module.check_source_profile_gates(
                [profiles["selective_ratio"]],
                min_profile_target_retention_ratio=4.0,
            ),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint source profile gate failed"):
            module.check_source_profile_gates(
                [profiles["selective_ratio"]],
                min_profile_target_retention_ratio=5.1,
            )
        self.assertEqual(
            profiles["selective_ratio"]["checkpoint_source_flag_fragment"],
            [
                "--checkpoint-source-label",
                "llama-3.2-3b",
                "--key-preset",
                "llama-3.2-3b",
                "--allow-overlap-resize",
                "--checkpoint-projection",
                "zspace",
                "--checkpoint-projection-strength",
                "1",
                "--checkpoint-projection-curvature",
                "-0.04",
                "--checkpoint-projection-frequency",
                "0.65",
                "--checkpoint-source-gain",
                "2",
                "--adapter-weight-decays",
                "0.01",
                "--max-grad-norms",
                "1.5",
                "--gradient-accumulation-steps-list",
                "4",
                "--ft-epochs-list",
                "6",
                "--target-min-loss-deltas",
                "0.001",
                "--patiences",
                "3",
                "--min-deltas",
                "0",
                "--lr-decay-patiences",
                "2",
                "--lr-decay-factors",
                "0.8",
                "--lr-decay-min-deltas",
                "0",
            ],
        )

    def test_mlp_lora_source_compare_coverage_gate_detects_missing_scope(self):
        module = load_example("byte_lm_mlp_lora_source_compare")
        candidate = module.source_candidate_from_aggregate(
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 1,
                "case_labels": "adapter_ja",
                "accepted_cases": 1,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.005,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.01,
                "retention_loss_margin_min": 10.0,
                "checkpoint_key_preset": "llama",
                "checkpoint_source_label": "llama-3.2-3b",
            },
            Path("aggregate.jsonl"),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "missing sources gemma"):
                module.check_source_coverage(
                    [candidate],
                    min_sources=2,
                    required_sources=["llama-3.2-3b", "gemma"],
                    min_cases=2,
                    required_cases=["adapter_ja", "route_cats"],
                    require_accepted_all=True,
                    require_movement_ok_all=True,
                    min_target_retention_ratio=1.5,
                )
        self.assertIn("source_coverage candidate=llama-3.2-3b::r12_a64_lr4", output.getvalue())

    def test_mlp_lora_source_profile_runner_roundtrip_smoke(self):
        source_compare = load_example("byte_lm_mlp_lora_source_compare")
        profile_runner = load_example("byte_lm_mlp_lora_profile_runner")

        def aggregate(source, target_delta, retention_delta):
            row = {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "base_config": "r12_a64_lr4",
                "checkpoint_projection_variant": None,
                "checkpoint_source_gain_variant": "gain_g2",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_cases": 2,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 2,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
                "target_loss_delta_mean": target_delta,
                "retention_loss_delta_mean": retention_delta,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": target_delta,
                "retention_loss_margin_min": 10.0,
                "retention_accuracy_margin_min": 1.0,
                "retention_perplexity_margin_min": 100.0,
                "checkpoint_key_preset": source,
                "checkpoint_source_label": source,
                "checkpoint_source_origin": "hf_state_dict",
                "checkpoint_loaded_files": 1,
                "checkpoint_vocab": 256,
                "checkpoint_hidden": 8,
                "checkpoint_target_classes": 256,
                "checkpoint_overlap_resize": True,
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 1.0,
                "checkpoint_projection_curvature": -0.04,
                "checkpoint_projection_frequency": 0.65,
                "checkpoint_source_gain": 2.0,
                "adapter_weight_decay_variant": "wd0p01",
                "adapter_weight_decay": 0.01,
                "max_grad_norm_variant": "gn1p5",
                "max_grad_norm": 1.5,
                "gradient_accumulation_steps_variant": "accum4",
                "gradient_accumulation_steps": 4,
                "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
                "ft_epochs": 6,
                "target_min_loss_delta_policy": 0.001,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.0,
                "lr_decay_patience": 2,
                "lr_decay_factor": 0.8,
                "lr_decay_min_delta": 0.0,
            }
            row["target_retention_gap_mean"] = target_delta - retention_delta
            row["target_retention_ratio"] = target_delta / retention_delta
            return row

        aggregates = [
            aggregate("gemma-4-e4b-it", 0.05, 0.025),
            aggregate("llama-3.2-3b", 0.04, 0.008),
        ]
        candidates = source_compare.source_candidates_from_rows(
            aggregates,
            Path("aggregate.jsonl"),
        )
        profiles = source_compare.source_profile_rows(candidates, ["selective_ratio"])
        profile = profiles[0]
        self.assertEqual(profile["selected_source"], "llama-3.2-3b")
        self.assertIn("training_policy_key", profile)

        with tempfile.TemporaryDirectory() as tmpdir:
            command_rows = profile_runner.command_rows_for_profiles(
                profiles,
                source_paths={"llama-3.2-3b": Path("/models/llama")},
                profiles=["selective_ratio"],
                output_dir=Path(tmpdir),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
            self.assertEqual(len(command_rows), 1)
            command_row = command_rows[0]
            expected_policy = profile_runner.training_policy_key(
                profile_runner.normalized_training_policy_row(profile)
            )
            self.assertEqual(command_row["training_policy_key"], expected_policy)
            self.assertIn("--hf-state-dict", command_row["command"])
            self.assertIn("--require-aggregate-case", command_row["command"])

            output_aggregate = dict(aggregates[1], training_policy_key=expected_policy)
            profile_runner.write_jsonl(Path(command_row["aggregate_jsonl"]), [output_aggregate])
            summaries = profile_runner.profile_run_summary_rows(command_rows)

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["row_type"], "checkpoint_source_profile_run")
        self.assertEqual(summary["source_profile"], "selective_ratio")
        self.assertEqual(summary["selected_source"], "llama-3.2-3b")
        self.assertEqual(summary["training_policy_key"], expected_policy)
        self.assertAlmostEqual(summary["target_retention_ratio"], 5.0)
        legacy_summary = dict(summary)
        legacy_summary.pop("target_retention_gap_mean")
        legacy_summary.pop("target_retention_ratio")
        legacy_summaries = [legacy_summary]
        self.assertEqual(
            profile_runner.check_profile_run_gates(
                legacy_summaries,
                min_target_retention_ratio=4.0,
            ),
            1,
        )
        self.assertEqual(
            profile_runner.compare_profile_run_summaries(
                legacy_summaries,
                summaries,
                max_target_retention_gap_regression=0.0,
                max_target_retention_ratio_regression=0.0,
                min_target_retention_ratio=4.0,
            ),
            1,
        )
        promotions = profile_runner.profile_run_promotion_rows(
            legacy_summaries,
            ready_min_target_retention_ratio=4.0,
        )
        self.assertAlmostEqual(promotions[0]["target_retention_ratio"], 5.0)
        self.assertTrue(promotions[0]["promotion_ready"])
        self.assertEqual(
            profile_runner.check_profile_run_gates(
                summaries,
                min_target_retention_ratio=4.0,
                min_accepted_rate=1.0,
                min_movement_ok_rate=1.0,
                min_retention_accuracy_margin=0.5,
                min_retention_perplexity_margin=50.0,
            ),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "profile run summary gate failed"):
            profile_runner.check_profile_run_gates(
                summaries,
                min_target_retention_ratio=5.1,
            )
        with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
            profile_runner.check_profile_run_gates(
                [dict(summary, accepted_rate=0.5)],
                min_accepted_rate=0.75,
            )
        with self.assertRaisesRegex(RuntimeError, "movement_ok_rate"):
            profile_runner.check_profile_run_gates(
                [dict(summary, movement_ok_rate=0.5)],
                min_movement_ok_rate=0.75,
            )
        self.assertEqual(
            profile_runner.compare_profile_run_summaries(
                summaries,
                summaries,
                max_target_loss_regression=0.0,
                max_target_retention_gap_regression=0.0,
                max_target_retention_ratio_regression=0.0,
                min_target_retention_ratio=4.0,
                min_accepted_rate=1.0,
                min_movement_ok_rate=1.0,
                require_source_match=True,
                require_config_match=True,
                require_training_policy_match=True,
            ),
            1,
        )

    def test_mlp_lora_source_compare_coverage_gate_detects_training_policy_scope_drift(self):
        module = load_example("byte_lm_mlp_lora_source_compare")

        def aggregate(source, ft_epochs):
            return {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 1,
                "case_labels": "adapter_ja",
                "accepted_cases": 1,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.005,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.01,
                "retention_loss_margin_min": 10.0,
                "checkpoint_key_preset": source,
                "checkpoint_source_label": source,
                "adapter_weight_decay_variant": "wd0p01",
                "adapter_weight_decay": 0.01,
                "max_grad_norm_variant": "gn1p5",
                "max_grad_norm": 1.5,
                "gradient_accumulation_steps_variant": "accum4",
                "gradient_accumulation_steps": 4,
                "ft_control_variant": f"ep{ft_epochs}::tmin0p001::pat3::ldp2::ldf0p8",
                "ft_epochs": ft_epochs,
                "target_min_loss_delta_policy": 0.001,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.0,
                "lr_decay_patience": 2,
                "lr_decay_factor": 0.8,
                "lr_decay_min_delta": 0.0,
            }

        matching = module.source_candidates_from_rows(
            [
                aggregate("gpt2-bare", 6),
                aggregate("llama-3.2-3b", 6),
            ],
            Path("aggregate.jsonl"),
        )
        self.assertEqual(
            module.check_source_coverage(
                matching,
                require_training_policy_scope_match=True,
            ),
            2,
        )
        drifted = module.source_candidates_from_rows(
            [
                aggregate("gpt2-bare", 6),
                aggregate("llama-3.2-3b", 10),
            ],
            Path("aggregate.jsonl"),
        )
        with self.assertRaisesRegex(RuntimeError, "training policy scope mismatch"):
            module.check_source_coverage(
                drifted,
                require_training_policy_scope_match=True,
            )

    def test_mlp_lora_source_compare_coverage_gate_detects_retention_margin_floors(self):
        module = load_example("byte_lm_mlp_lora_source_compare")
        candidate = module.source_candidate_from_aggregate(
            {
                "row_type": "config_aggregate",
                "config": "r12_a64_lr4",
                "cases": 1,
                "case_labels": "adapter_ja",
                "accepted_cases": 1,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
                "target_loss_delta_mean": 0.01,
                "retention_loss_delta_mean": 0.005,
                "retention_accuracy_delta_mean": 0.0,
                "target_loss_margin_min": 0.01,
                "retention_loss_margin_min": 10.0,
                "retention_accuracy_margin_min": 0.04,
                "retention_perplexity_margin_min": 0.2,
                "checkpoint_key_preset": "llama",
                "checkpoint_source_label": "llama-3.2-3b",
            },
            Path("aggregate.jsonl"),
        )
        with self.assertRaisesRegex(RuntimeError, "retention_accuracy_margin_min"):
            module.check_source_coverage(
                [candidate],
                min_retention_accuracy_margin=0.05,
                min_retention_perplexity_margin=0.25,
            )

    def test_mlp_lora_profile_runner_materializes_profile_commands(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        rows = [
            {
                "row_type": "checkpoint_source_profile",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "selected_config": "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g4",
                "base_config": "r12_a64_lr4",
                "case_labels": "adapter_ja,route_cats",
                "checkpoint_source_gain": 4.0,
                "winner_metric": "target_loss_delta_mean",
                "winner_value": 1.2,
                "checkpoint_source_flag_fragment": [
                    "--checkpoint-source-label",
                    "gemma-4-e4b-it",
                    "--key-preset",
                    "gemma",
                    "--allow-overlap-resize",
                    "--checkpoint-projection",
                    "zspace",
                    "--checkpoint-source-gain",
                    "4",
                ],
            },
            {
                "row_type": "checkpoint_source_profile",
                "source_profile": "selective_ratio",
                "selected_source": "llama-3.2-3b",
                "selected_config": "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep6::tmin0p001::pat3::ldp2::ldf0p8",
                "base_config": "r12_a64_lr4",
                "case_labels": "adapter_ja,route_cats",
                "checkpoint_source_gain": 2.0,
                "adapter_weight_decay_variant": "wd0p01",
                "adapter_weight_decay": 0.01,
                "max_grad_norm_variant": "gn1p5",
                "max_grad_norm": 1.5,
                "gradient_accumulation_steps_variant": "accum4",
                "gradient_accumulation_steps": 4,
                "training_policy_key": "policy:llama-ft6",
                "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
                "ft_epochs": 6,
                "target_min_loss_delta_policy": 0.001,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.0,
                "lr_decay_patience": 2,
                "lr_decay_factor": 0.8,
                "lr_decay_min_delta": 0.0,
                "winner_metric": "target_retention_ratio",
                "winner_value": 2.6,
                "checkpoint_source_flag_fragment": [
                    "--checkpoint-source-label",
                    "llama-3.2-3b",
                    "--key-preset",
                    "llama",
                    "--allow-overlap-resize",
                    "--checkpoint-source-gain",
                    "2",
                    "--adapter-weight-decays",
                    "0.01",
                    "--max-grad-norms",
                    "1.5",
                    "--gradient-accumulation-steps-list",
                    "4",
                    "--ft-epochs-list",
                    "6",
                    "--target-min-loss-deltas",
                    "0.001",
                    "--patiences",
                    "3",
                    "--min-deltas",
                    "0",
                    "--lr-decay-patiences",
                    "2",
                    "--lr-decay-factors",
                    "0.8",
                    "--lr-decay-min-deltas",
                    "0",
                ],
            },
        ]
        command_rows = module.command_rows_for_profiles(
            rows,
            source_paths={
                "gemma-4-e4b-it": Path("/models/gemma"),
                "llama-3.2-3b": Path("/models/llama"),
            },
            profiles=["selective_ratio"],
            output_dir=Path("/tmp/profile-runs"),
            output_prefix="profile",
            python_executable="python",
            sweep_script=Path("sweep.py"),
            case_jsonls=[Path("/tmp/external-cases.jsonl")],
            lora_config_jsonls=[Path("/tmp/lora-configs.jsonl")],
            extra_args=["--min-aggregate-retention-accuracy-margin", "0.5"],
        )
        self.assertEqual(len(command_rows), 1)
        row = command_rows[0]
        command = row["command"]
        self.assertEqual(row["source_profile"], "selective_ratio")
        self.assertEqual(row["selected_source"], "llama-3.2-3b")
        self.assertIn("--hf-state-dict", command)
        self.assertIn("/models/llama", command)
        self.assertIn("--checkpoint-source-gain", command)
        self.assertIn("2", command)
        self.assertIn("--adapter-weight-decays", command)
        self.assertIn("0.01", command)
        self.assertEqual(row["adapter_weight_decay_variant"], "wd0p01")
        self.assertEqual(row["adapter_weight_decay"], 0.01)
        self.assertIn("--max-grad-norms", command)
        self.assertIn("1.5", command)
        self.assertIn("--gradient-accumulation-steps-list", command)
        self.assertIn("4", command)
        self.assertIn("--case-jsonl", command)
        self.assertIn("/tmp/external-cases.jsonl", command)
        self.assertLess(command.index("--case-jsonl"), command.index("--case"))
        self.assertEqual(row["case_jsonls"], "/tmp/external-cases.jsonl")
        self.assertIn("--lora-config-jsonl", command)
        self.assertIn("/tmp/lora-configs.jsonl", command)
        self.assertLess(command.index("--lora-config-jsonl"), command.index("--config"))
        self.assertEqual(row["lora_config_jsonls"], "/tmp/lora-configs.jsonl")
        self.assertEqual(row["max_grad_norm_variant"], "gn1p5")
        self.assertEqual(row["max_grad_norm"], 1.5)
        self.assertEqual(row["gradient_accumulation_steps_variant"], "accum4")
        self.assertEqual(row["gradient_accumulation_steps"], 4)
        expected_policy = module.training_policy_key(
            module.normalized_training_policy_row(rows[1])
        )
        self.assertEqual(row["training_policy_key"], expected_policy)
        self.assertIn("--ft-epochs-list", command)
        self.assertIn("6", command)
        self.assertIn("--target-min-loss-deltas", command)
        self.assertIn("0.001", command)
        self.assertEqual(
            row["ft_control_variant"],
            "ep6::tmin0p001::pat3::md0::ldp2::ldf0p8::ldmd0",
        )
        self.assertEqual(row["ft_epochs"], 6)
        self.assertEqual(row["target_min_loss_delta_policy"], 0.001)
        self.assertEqual(row["early_stopping_patience"], 3)
        self.assertEqual(row["lr_decay_patience"], 2)
        self.assertEqual(row["lr_decay_factor"], 0.8)
        self.assertEqual(command.count("--case"), 2)
        self.assertIn("--min-aggregate-cases", command)
        self.assertIn("--require-aggregate-case", command)
        self.assertIn("--aggregate-jsonl", command)
        self.assertIn("--min-aggregate-retention-accuracy-margin", command)
        self.assertEqual(row["configs"], "r12_a64_lr4")
        self.assertIn(
            "profile-selective-ratio-llama-3-2-3b-r12-a64-lr4-zspace-s1-cm0p04-f0p65-gain-g2-wd0p01-gn1p5-accum4-ep6-tmin0p001-pat3-md0-ldp2-ldf0p8-ldmd0",
            row["jsonl"],
        )

        override_rows = module.command_rows_for_profiles(
            rows,
            source_paths={"llama-3.2-3b": Path("/models/llama")},
            profiles=["selective_ratio"],
            configs=["r6_a32_lr3"],
            output_dir=Path("/tmp/profile-runs"),
            output_prefix="profile",
            python_executable="python",
            sweep_script=Path("sweep.py"),
        )
        override = override_rows[0]
        self.assertEqual(override["configs"], "r6_a32_lr3")
        self.assertEqual(
            override["run_config_key"],
            "r6_a32_lr3::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep6::tmin0p001::pat3::md0::ldp2::ldf0p8::ldmd0",
        )
        self.assertIn("--config", override["command"])
        self.assertIn("r6_a32_lr3", override["command"])
        self.assertIn(
            "profile-selective-ratio-llama-3-2-3b-r6-a32-lr3-zspace-s1-cm0p04-f0p65-gain-g2-wd0p01-gn1p5-accum4-ep6-tmin0p001-pat3-md0-ldp2-ldf0p8-ldmd0",
            override["jsonl"],
        )
        heavier_rows = module.command_rows_for_profiles(
            rows,
            source_paths={"llama-3.2-3b": Path("/models/llama")},
            profiles=["selective_ratio"],
            output_dir=Path("/tmp/profile-runs"),
            output_prefix="profile",
            python_executable="python",
            sweep_script=Path("sweep.py"),
            ft_control_override={
                "ft_epochs": 8,
                "target_min_loss_delta_policy": 0.002,
                "early_stopping_patience": None,
                "early_stopping_min_delta": 0.0001,
                "lr_decay_patience": None,
                "lr_decay_factor": 0.7,
                "lr_decay_min_delta": 0.0002,
            },
        )
        heavier = heavier_rows[0]
        heavier_command = heavier["command"]
        effective_heavier_policy = module.apply_ft_control_override(
            rows[1],
            {
                "ft_epochs": 8,
                "target_min_loss_delta_policy": 0.002,
                "early_stopping_patience": None,
                "early_stopping_min_delta": 0.0001,
                "lr_decay_patience": None,
                "lr_decay_factor": 0.7,
                "lr_decay_min_delta": 0.0002,
            },
        )
        self.assertEqual(
            heavier["training_policy_key"],
            module.training_policy_key(effective_heavier_policy),
        )
        self.assertEqual(heavier["ft_epochs"], 8)
        self.assertEqual(heavier["target_min_loss_delta_policy"], 0.002)
        self.assertIsNone(heavier["early_stopping_patience"])
        self.assertEqual(heavier["early_stopping_min_delta"], 0.0001)
        self.assertIsNone(heavier["lr_decay_patience"])
        self.assertEqual(heavier["lr_decay_factor"], 0.7)
        self.assertEqual(heavier["lr_decay_min_delta"], 0.0002)
        self.assertEqual(
            heavier["run_config_key"],
            "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep8::tmin0p002::patnone::md0p0001::ldpnone::ldf0p7::ldmd0p0002",
        )
        ft_epoch_indices = [
            index for index, value in enumerate(heavier_command) if value == "--ft-epochs-list"
        ]
        self.assertGreaterEqual(len(ft_epoch_indices), 2)
        self.assertEqual(heavier_command[ft_epoch_indices[-1] + 1], "8")
        patience_indices = [
            index for index, value in enumerate(heavier_command) if value == "--patiences"
        ]
        self.assertEqual(heavier_command[patience_indices[-1] + 1], "none")
        lr_patience_indices = [
            index
            for index, value in enumerate(heavier_command)
            if value == "--lr-decay-patiences"
        ]
        self.assertEqual(heavier_command[lr_patience_indices[-1] + 1], "none")
        self.assertIn(
            "profile-selective-ratio-llama-3-2-3b-r12-a64-lr4-zspace-s1-cm0p04-f0p65-gain-g2-wd0p01-gn1p5-accum4-ep8-tmin0p002-patnone-md0p0001-ldpnone-ldf0p7-ldmd0p0002",
            heavier["jsonl"],
        )
        prefix_overlap_row = dict(rows[1])
        prefix_overlap_row["ft_control_variant"] = "ep6"
        prefix_overlap_row["selected_config"] = (
            "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::"
            "ep6::tmin0p001::pat3::md0::ldp2::ldf0p8::ldmd0"
        )
        prefix_overlap_row["config"] = prefix_overlap_row["selected_config"]
        prefix_overlap_effective = module.apply_ft_control_override(
            prefix_overlap_row,
            {
                "ft_epochs": 8,
                "target_min_loss_delta_policy": 0.002,
                "early_stopping_patience": None,
                "early_stopping_min_delta": 0.0001,
                "lr_decay_patience": None,
                "lr_decay_factor": 0.7,
                "lr_decay_min_delta": 0.0002,
            },
        )
        self.assertEqual(
            prefix_overlap_effective["selected_config"],
            "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep8::tmin0p002::patnone::md0p0001::ldpnone::ldf0p7::ldmd0p0002",
        )
        promotion_rows = [
            {
                "row_type": "checkpoint_source_profile_promotion",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "config": "r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g4",
                "training_policy_key": "policy:gemma-ft6",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "promotion_rank": 2,
                "promotion_metric": "target_retention_ratio",
                "promotion_value": 1.5,
                "promotion_ready": False,
                "run_key": "strong_effect::r12_a64_lr4::zspace_s1_cm0p04_f0p65::gain_g4",
            },
            {
                "row_type": "checkpoint_source_profile_promotion",
                "source_profile": "selective_ratio",
                "selected_source": "llama-3.2-3b",
                "config": "r6_a32_lr3::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep6::tmin0p001::pat3::ldp2::ldf0p8",
                "training_policy_key": "policy:llama-ft6",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "promotion_rank": 1,
                "promotion_metric": "target_retention_ratio",
                "promotion_value": 2.8,
                "promotion_ready": True,
                "promotion_ready_top_k": 2,
                "promotion_ready_within": 0.05,
                "promotion_ready_floor_passed": True,
                "promotion_ready_floor_failures": [],
                "promotion_ready_require_guard_counts_available": True,
                "promotion_ready_min_guard_acceptance_rate_mean": 0.75,
                "promotion_ready_max_guard_retention_rejected_epochs_mean": 0.0,
                "promotion_ready_max_guard_target_stale_epochs_mean": 1.0,
                "promotion_ready_max_guard_retention_rejected_rate_mean": 0.10,
                "promotion_ready_max_guard_target_stale_rate_mean": 0.40,
                "run_key": "selective_ratio::r6_a32_lr3::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep6::tmin0p001::pat3::ldp2::ldf0p8",
            },
        ]
        legacy_promotion = dict(
            promotion_rows[1],
            target_loss_delta_mean=0.056,
            retention_loss_delta_mean=0.02,
        )
        legacy_promotion.pop("promotion_value")
        selected_legacy_promotions = module.selected_promotion_rows([legacy_promotion])
        self.assertAlmostEqual(
            selected_legacy_promotions[0]["target_retention_gap_mean"],
            0.036,
        )
        self.assertAlmostEqual(
            selected_legacy_promotions[0]["target_retention_ratio"],
            2.8,
        )
        self.assertAlmostEqual(selected_legacy_promotions[0]["promotion_value"], 2.8)
        legacy_promoted_rows = module.command_rows_for_profiles(
            rows,
            source_paths={"llama-3.2-3b": Path("/models/llama")},
            promotion_rows=[legacy_promotion],
            output_dir=Path("/tmp/profile-runs"),
            output_prefix="profile",
            python_executable="python",
            sweep_script=Path("sweep.py"),
        )
        self.assertAlmostEqual(legacy_promoted_rows[0]["promotion_value"], 2.8)
        promoted_rows = module.command_rows_for_profiles(
            rows,
            source_paths={
                "gemma-4-e4b-it": Path("/models/gemma"),
                "llama-3.2-3b": Path("/models/llama"),
            },
            promotion_rows=promotion_rows,
            output_dir=Path("/tmp/profile-runs"),
            output_prefix="profile",
            python_executable="python",
            sweep_script=Path("sweep.py"),
        )
        self.assertEqual(len(promoted_rows), 1)
        promoted = promoted_rows[0]
        self.assertEqual(promoted["source_profile"], "selective_ratio")
        self.assertEqual(promoted["configs"], "r6_a32_lr3")
        self.assertEqual(promoted["promotion_rank"], 1)
        self.assertEqual(promoted["promotion_metric"], "target_retention_ratio")
        self.assertEqual(promoted["promotion_value"], 2.8)
        self.assertTrue(promoted["promotion_ready"])
        self.assertEqual(promoted["promotion_ready_top_k"], 2)
        self.assertEqual(promoted["promotion_ready_within"], 0.05)
        self.assertTrue(promoted["promotion_ready_floor_passed"])
        self.assertEqual(promoted["promotion_ready_floor_failures"], [])
        self.assertTrue(promoted["promotion_ready_require_guard_counts_available"])
        self.assertEqual(promoted["promotion_ready_min_guard_acceptance_rate_mean"], 0.75)
        self.assertEqual(
            promoted["promotion_ready_max_guard_retention_rejected_epochs_mean"],
            0.0,
        )
        self.assertEqual(promoted["promotion_ready_max_guard_target_stale_epochs_mean"], 1.0)
        self.assertEqual(
            promoted["promotion_ready_max_guard_retention_rejected_rate_mean"],
            0.10,
        )
        self.assertEqual(promoted["promotion_ready_max_guard_target_stale_rate_mean"], 0.40)
        self.assertEqual(
            promoted["run_config_key"],
            "r6_a32_lr3::zspace_s1_cm0p04_f0p65::gain_g2::wd0p01::gn1p5::accum4::ep6::tmin0p001::pat3::md0::ldp2::ldf0p8::ldmd0",
        )
        self.assertIn("r6_a32_lr3", promoted["command"])
        self.assertIn(
            "profile-selective-ratio-llama-3-2-3b-r6-a32-lr3-zspace-s1-cm0p04-f0p65-gain-g2-wd0p01-gn1p5-accum4-ep6-tmin0p001-pat3-md0-ldp2-ldf0p8-ldmd0",
            promoted["jsonl"],
        )
        already_normalized_promoted_rows = module.command_rows_for_profiles(
            rows,
            source_paths={"llama-3.2-3b": Path("/models/llama")},
            promotion_rows=[dict(promotion_rows[1], config=promoted["run_config_key"])],
            output_dir=Path("/tmp/profile-runs"),
            output_prefix="profile",
            python_executable="python",
            sweep_script=Path("sweep.py"),
        )
        already_normalized = already_normalized_promoted_rows[0]
        self.assertEqual(already_normalized["run_config_key"], promoted["run_config_key"])
        self.assertEqual(
            already_normalized["run_config_key"].count("::md0::"),
            promoted["run_config_key"].count("::md0::"),
        )
        selection = module.promotion_selection_summary(promotion_rows, ready_only=True)
        self.assertEqual(selection["row_type"], "checkpoint_source_profile_promotion_selection")
        self.assertEqual(selection["selected_promotions"], 2)
        self.assertEqual(selection["ready_promotions"], 1)
        self.assertEqual(selection["non_ready_promotions"], 1)
        self.assertEqual(selection["materialized_promotions"], 1)
        self.assertEqual(selection["guard_policy_promotions"], 1)
        self.assertEqual(selection["ready_guard_policy_promotions"], 1)
        self.assertEqual(selection["materialized_guard_policy_promotions"], 1)
        self.assertEqual(selection["non_ready_guard_failure_promotions"], 0)
        self.assertTrue(selection["promotion_ready_only"])
        self.assertIn("strong_effect::r12_a64_lr4", selection["non_ready_details"][0])
        include_non_ready_selection = module.promotion_selection_summary(
            promotion_rows,
            ready_only=False,
        )
        self.assertEqual(include_non_ready_selection["materialized_promotions"], 2)
        self.assertEqual(
            include_non_ready_selection["materialized_guard_policy_promotions"],
            1,
        )
        guard_failed_selection = module.promotion_selection_summary(
            [
                dict(
                    promotion_rows[1],
                    promotion_ready=False,
                    promotion_ready_floor_passed=False,
                    promotion_ready_floor_failures=[
                        "guard_epoch_counts_available_all=false",
                    ],
                )
            ],
            ready_only=True,
        )
        self.assertEqual(guard_failed_selection["ready_promotions"], 0)
        self.assertEqual(guard_failed_selection["guard_policy_promotions"], 1)
        self.assertEqual(guard_failed_selection["ready_guard_policy_promotions"], 0)
        self.assertEqual(guard_failed_selection["non_ready_guard_failure_promotions"], 1)
        self.assertIn(
            "guard_epoch_counts_available_all=false",
            guard_failed_selection["non_ready_guard_failure_details"][0],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.jsonl"
            promotion_path = Path(tmpdir) / "promotions.jsonl"
            commands_path = Path(tmpdir) / "commands.jsonl"
            selection_path = Path(tmpdir) / "selection.jsonl"
            module.write_jsonl(profile_path, rows)
            module.write_jsonl(promotion_path, promotion_rows)
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_mlp_lora_profile_runner.py",
                "--profile-jsonl",
                str(profile_path),
                "--promotion-input-jsonl",
                str(promotion_path),
                "--source-path",
                "gemma-4-e4b-it=/models/gemma",
                "--source-path",
                "llama-3.2-3b=/models/llama",
                "--commands-jsonl",
                str(commands_path),
                "--promotion-selection-jsonl",
                str(selection_path),
                "--case-jsonl",
                "/tmp/external-cases.jsonl",
                "--min-promotion-ready-count",
                "1",
                "--min-promotion-ready-rate",
                "0.5",
                "--min-promotion-ready-guard-policy-count",
                "1",
                "--require-promotion-ready-guard-policy",
            ]
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    module.main()
            finally:
                sys.argv = old_argv
            cli_selection = module.load_jsonl(selection_path)
            cli_commands = module.load_jsonl(commands_path)
        self.assertIn("profile_promotion_selection selected=2 ready=1", output.getvalue())
        self.assertIn("guard_policy=1 ready_guard_policy=1", output.getvalue())
        self.assertIn("profile_promotion_selection_jsonl=", output.getvalue())
        self.assertIn("profile_promotion_gate rows=2 ready_count=1", output.getvalue())
        self.assertIn("ready_guard_policy_count=1", output.getvalue())
        self.assertEqual(len(cli_selection), 1)
        self.assertEqual(cli_selection[0]["ready_promotions"], 1)
        self.assertEqual(cli_selection[0]["materialized_promotions"], 1)
        self.assertEqual(cli_selection[0]["guard_policy_promotions"], 1)
        self.assertEqual(cli_selection[0]["materialized_guard_policy_promotions"], 1)
        self.assertEqual(len(cli_commands), 1)
        self.assertEqual(cli_commands[0]["case_jsonls"], "/tmp/external-cases.jsonl")
        self.assertTrue(cli_commands[0]["promotion_ready_require_guard_counts_available"])
        self.assertEqual(
            cli_commands[0]["promotion_ready_min_guard_acceptance_rate_mean"],
            0.75,
        )
        self.assertEqual(
            cli_commands[0]["promotion_ready_max_guard_target_stale_epochs_mean"],
            1.0,
        )
        self.assertEqual(
            cli_commands[0]["promotion_ready_max_guard_retention_rejected_rate_mean"],
            0.10,
        )
        self.assertEqual(
            cli_commands[0]["promotion_ready_max_guard_target_stale_rate_mean"],
            0.40,
        )
        self.assertIn("--case-jsonl", cli_commands[0]["command"])
        all_non_ready_promotions = [
            dict(
                promotion_rows[1],
                promotion_ready=False,
                promotion_ready_floor_passed=False,
                promotion_ready_floor_failures=[
                    "target_retention_ratio<3.000000000",
                ],
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "no ready checkpoint_source_profile_promotion rows.*target_retention_ratio<3.000000000",
        ):
            module.command_rows_for_profiles(
                rows,
                source_paths={"llama-3.2-3b": Path("/models/llama")},
                promotion_rows=all_non_ready_promotions,
                output_dir=Path("/tmp/profile-runs"),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.jsonl"
            promotion_path = Path(tmpdir) / "promotions-non-ready.jsonl"
            commands_path = Path(tmpdir) / "commands.jsonl"
            selection_path = Path(tmpdir) / "selection.jsonl"
            module.write_jsonl(profile_path, rows)
            module.write_jsonl(promotion_path, all_non_ready_promotions)
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_mlp_lora_profile_runner.py",
                "--profile-jsonl",
                str(profile_path),
                "--promotion-input-jsonl",
                str(promotion_path),
                "--source-path",
                "llama-3.2-3b=/models/llama",
                "--commands-jsonl",
                str(commands_path),
                "--promotion-selection-jsonl",
                str(selection_path),
                "--min-promotion-ready-count",
                "1",
            ]
            try:
                with self.assertRaisesRegex(RuntimeError, "ready_count"):
                    module.main()
            finally:
                sys.argv = old_argv
            failed_selection = module.load_jsonl(selection_path)
        self.assertEqual(failed_selection[0]["ready_promotions"], 0)
        self.assertEqual(failed_selection[0]["materialized_promotions"], 0)
        stale_promotions = [
            dict(promotion_rows[1], cases=1, case_labels="adapter_ja"),
        ]
        with self.assertRaisesRegex(ValueError, "promotion case scope mismatch"):
            module.command_rows_for_profiles(
                rows,
                source_paths={"llama-3.2-3b": Path("/models/llama")},
                promotion_rows=stale_promotions,
                output_dir=Path("/tmp/profile-runs"),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
        source_mismatch_promotions = [
            dict(promotion_rows[1], selected_source="gemma-4-e4b-it"),
        ]
        with self.assertRaisesRegex(ValueError, "promotion source mismatch"):
            module.command_rows_for_profiles(
                rows,
                source_paths={
                    "gemma-4-e4b-it": Path("/models/gemma"),
                    "llama-3.2-3b": Path("/models/llama"),
                },
                promotion_rows=source_mismatch_promotions,
                output_dir=Path("/tmp/profile-runs"),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
        missing_config = dict(promotion_rows[1])
        missing_config.pop("config")
        with self.assertRaisesRegex(ValueError, "promotion config missing"):
            module.command_rows_for_profiles(
                rows,
                source_paths={"llama-3.2-3b": Path("/models/llama")},
                promotion_rows=[missing_config],
                output_dir=Path("/tmp/profile-runs"),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
        missing_policy = dict(promotion_rows[1])
        missing_policy.pop("training_policy_key")
        with self.assertRaisesRegex(ValueError, "promotion training_policy_key missing"):
            module.command_rows_for_profiles(
                rows,
                source_paths={"llama-3.2-3b": Path("/models/llama")},
                promotion_rows=[missing_policy],
                output_dir=Path("/tmp/profile-runs"),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
        policy_mismatch = [
            dict(promotion_rows[1], training_policy_key="policy:other"),
        ]
        with self.assertRaisesRegex(ValueError, "promotion training_policy_key mismatch"):
            module.command_rows_for_profiles(
                rows,
                source_paths={"llama-3.2-3b": Path("/models/llama")},
                promotion_rows=policy_mismatch,
                output_dir=Path("/tmp/profile-runs"),
                output_prefix="profile",
                python_executable="python",
                sweep_script=Path("sweep.py"),
            )
        generated_policy_profile = dict(
            rows[1],
            training_policy_key=module.training_policy_key(
                module.normalized_training_policy_row(rows[1])
            ),
        )
        generated_policy_promotion = dict(
            promotion_rows[1],
            training_policy_key=module.training_policy_key(
                module.apply_ft_control_override(
                    rows[1],
                    {
                        "ft_epochs": 8,
                        "target_min_loss_delta_policy": 0.002,
                        "early_stopping_patience": None,
                        "early_stopping_min_delta": 0.0001,
                        "lr_decay_patience": None,
                        "lr_decay_factor": 0.7,
                        "lr_decay_min_delta": 0.0002,
                    },
                )
            ),
        )
        module.validate_promotion_training_policy(
            generated_policy_profile,
            generated_policy_promotion,
        )
        non_ft_policy_promotion = dict(
            generated_policy_promotion,
            training_policy_key=module.training_policy_key(
                module.normalized_training_policy_row(
                    dict(rows[1], adapter_weight_decay=0.02)
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "promotion training_policy_key mismatch"):
            module.validate_promotion_training_policy(
                generated_policy_profile,
                non_ft_policy_promotion,
            )

    def test_byte_lm_profile_smoke_names_promoted_rungs(self):
        module = load_example("byte_lm_profile_smoke")
        out_dir = Path("/tmp/profile-smoke")
        first = module.promoted_rung_artifacts(out_dir, "profile-smoke-promoted", 1)
        second = module.promoted_rung_artifacts(out_dir, "profile-smoke-promoted", 2)

        self.assertEqual(first["run_dir"], out_dir / "promoted-profile-runs")
        self.assertEqual(first["commands_jsonl"], out_dir / "promoted-commands.jsonl")
        self.assertEqual(first["selection_jsonl"], out_dir / "promotion-selection.jsonl")
        self.assertEqual(
            first["run_summary_jsonl"],
            out_dir / "promoted-profile-run-summary.jsonl",
        )
        self.assertEqual(second["run_dir"], out_dir / "promoted-rung2-profile-runs")
        self.assertEqual(
            second["commands_jsonl"],
            out_dir / "promoted-rung2-commands.jsonl",
        )
        self.assertEqual(
            second["selection_jsonl"],
            out_dir / "promoted-rung2-promotion-selection.jsonl",
        )
        self.assertEqual(second["output_prefix"], "profile-smoke-promoted-rung2")
        manifest_row = module.promoted_rung_manifest_row(
            second,
            ft_epochs=3,
            input_promotion_jsonl=out_dir / "promoted-promotion.jsonl",
        )
        self.assertEqual(manifest_row["row_type"], "profile_smoke_promoted_rung")
        self.assertEqual(manifest_row["rung"], 2)
        self.assertEqual(manifest_row["ft_epochs"], 3)
        self.assertEqual(
            manifest_row["input_promotion_jsonl"],
            str(out_dir / "promoted-promotion.jsonl"),
        )
        self.assertEqual(
            manifest_row["run_summary_jsonl"],
            str(out_dir / "promoted-rung2-profile-run-summary.jsonl"),
        )
        self.assertEqual(
            manifest_row["promotion_jsonl"],
            str(out_dir / "promoted-rung2-promotion.jsonl"),
        )
        smoke_args = argparse.Namespace(
            source_label="llama-3.2-3b",
            key_preset="auto",
            ft_epochs=1,
            promoted_ft_epochs=None,
            promoted_ft_epochs_step=1,
            promotion_metric="target_loss_delta_mean",
            promoted_output_prefix="profile-smoke-promoted",
            strict_aggregate_gates=False,
            skip_checkpoint_shape_audit=False,
            skip_checkpoint_preflight=False,
            compare_checkpoint_preflight_jsonl=out_dir / "checkpoint-preflight-baseline.jsonl",
            require_checkpoint_preflight_match=True,
        )
        smoke_manifest_row = module.profile_smoke_manifest_row(
            args=smoke_args,
            out_dir=out_dir,
            checkpoint_path=out_dir / "pytorch_model.bin",
            checkpoint_source_kind="external",
            cases=["adapter_ja", "route_cats"],
            configs=["r12_a64_lr4"],
            profiles=["strong_effect"],
            checkpoint_shape_audit_jsonl=out_dir / "checkpoint-shape-audit.jsonl",
            checkpoint_preflight_jsonl=out_dir / "checkpoint-preflight.jsonl",
            sweep_jsonl=out_dir / "sweep.jsonl",
            sweep_aggregate_jsonl=out_dir / "sweep-aggregate.jsonl",
            source_compare_jsonl=out_dir / "source-compare.jsonl",
            profile_jsonl=out_dir / "profiles.jsonl",
            run_dir=out_dir / "profile-runs",
            run_events_jsonl=out_dir / "profile-run-events.jsonl",
            run_summary_jsonl=out_dir / "profile-run-summary.jsonl",
            promotion_jsonl=out_dir / "promotion.jsonl",
            promotion_compare_jsonl=out_dir / "promotion-compare.jsonl",
            promoted_rungs=2,
            promoted_rungs_jsonl=out_dir / "promoted-rungs.jsonl",
            promoted_artifacts=[first, second],
        )
        self.assertEqual(smoke_manifest_row["row_type"], "profile_smoke_manifest")
        self.assertEqual(smoke_manifest_row["checkpoint_source"], "external")
        self.assertEqual(smoke_manifest_row["source_label"], "llama-3.2-3b")
        self.assertEqual(smoke_manifest_row["cases"], ["adapter_ja", "route_cats"])
        self.assertEqual(smoke_manifest_row["promoted_ft_epochs"], [2, 3])
        self.assertTrue(smoke_manifest_row["require_checkpoint_preflight_match"])
        self.assertEqual(
            smoke_manifest_row["compare_checkpoint_preflight_jsonl"],
            str(out_dir / "checkpoint-preflight-baseline.jsonl"),
        )
        self.assertEqual(
            smoke_manifest_row["promoted_final_run_summary_jsonl"],
            str(out_dir / "promoted-rung2-profile-run-summary.jsonl"),
        )
        fields = module.required_manifest_artifact_fields(smoke_manifest_row)
        self.assertIn("checkpoint", fields)
        self.assertIn("checkpoint_preflight_jsonl", fields)
        self.assertIn("compare_checkpoint_preflight_jsonl", fields)
        self.assertIn("promoted_final_run_summary_jsonl", fields)
        with self.assertRaisesRegex(FileNotFoundError, "profile smoke manifest"):
            module.validate_profile_smoke_manifest_artifacts(smoke_manifest_row)

        default_args = argparse.Namespace(
            ft_epochs=1,
            promoted_ft_epochs=None,
            promoted_ft_epochs_step=1,
        )
        self.assertEqual(module.promoted_ft_epochs_for_rung(default_args, 1), 2)
        self.assertEqual(module.promoted_ft_epochs_for_rung(default_args, 2), 3)

        explicit_args = argparse.Namespace(
            ft_epochs=1,
            promoted_ft_epochs=4,
            promoted_ft_epochs_step=2,
        )
        self.assertEqual(module.promoted_ft_epochs_for_rung(explicit_args, 1), 4)
        self.assertEqual(module.promoted_ft_epochs_for_rung(explicit_args, 3), 8)
        with self.assertRaisesRegex(ValueError, "rung must be positive"):
            module.promoted_rung_artifacts(out_dir, "profile-smoke-promoted", 0)

    def test_byte_lm_profile_smoke_validates_promoted_rung_manifest_chain(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            first = module.promoted_rung_artifacts(out_dir, "profile-smoke-promoted", 1)
            second = module.promoted_rung_artifacts(out_dir, "profile-smoke-promoted", 2)
            smoke_args = argparse.Namespace(
                source_label="local-smoke",
                key_preset="llama",
                ft_epochs=1,
                promoted_ft_epochs=None,
                promoted_ft_epochs_step=1,
                promotion_metric="target_loss_delta_mean",
                promoted_output_prefix="profile-smoke-promoted",
                strict_aggregate_gates=False,
                skip_checkpoint_shape_audit=False,
                skip_checkpoint_preflight=False,
                compare_checkpoint_preflight_jsonl=None,
                require_checkpoint_preflight_match=False,
            )
            top_row = module.profile_smoke_manifest_row(
                args=smoke_args,
                out_dir=out_dir,
                checkpoint_path=out_dir / "pytorch_model.bin",
                checkpoint_source_kind="generated",
                cases=["adapter_ja"],
                configs=["r12_a64_lr4"],
                profiles=["strong_effect"],
                checkpoint_shape_audit_jsonl=out_dir / "checkpoint-shape-audit.jsonl",
                checkpoint_preflight_jsonl=out_dir / "checkpoint-preflight.jsonl",
                sweep_jsonl=out_dir / "sweep.jsonl",
                sweep_aggregate_jsonl=out_dir / "sweep-aggregate.jsonl",
                source_compare_jsonl=out_dir / "source-compare.jsonl",
                profile_jsonl=out_dir / "profiles.jsonl",
                run_dir=out_dir / "profile-runs",
                run_events_jsonl=out_dir / "profile-run-events.jsonl",
                run_summary_jsonl=out_dir / "profile-run-summary.jsonl",
                promotion_jsonl=out_dir / "promotion.jsonl",
                promotion_compare_jsonl=out_dir / "promotion-compare.jsonl",
                promoted_rungs=2,
                promoted_rungs_jsonl=out_dir / "promoted-rungs.jsonl",
                promoted_artifacts=[first, second],
            )
            first_row = module.promoted_rung_manifest_row(
                first,
                ft_epochs=2,
                input_promotion_jsonl=out_dir / "promotion.jsonl",
            )
            second_row = module.promoted_rung_manifest_row(
                second,
                ft_epochs=3,
                input_promotion_jsonl=first["promotion_jsonl"],
            )
            for field in module.required_manifest_artifact_fields(top_row):
                path = Path(top_row[field])
                if field in {"out_dir", "profile_run_dir"}:
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            for rung_row in [first_row, second_row]:
                for field in [
                    "input_promotion_jsonl",
                    *module.PROMOTED_RUNG_ARTIFACT_FIELDS,
                ]:
                    path = Path(rung_row[field])
                    if field == "output_dir":
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
            module.write_jsonl(
                Path(top_row["promoted_rungs_jsonl"]),
                [first_row, second_row],
            )

            self.assertTrue(
                module.validate_promoted_rung_manifest_consistency(
                    top_row,
                    [first_row, second_row],
                )
            )
            with self.assertRaisesRegex(ValueError, "promoted rung manifest mismatch"):
                module.validate_promoted_rung_manifest_consistency(
                    top_row,
                    [first_row],
                )
            broken_chain = [
                first_row,
                dict(second_row, input_promotion_jsonl=str(out_dir / "promotion.jsonl")),
            ]
            with self.assertRaisesRegex(
                ValueError,
                "promoted rung manifest chain mismatch",
            ):
                module.validate_promoted_rung_manifest_consistency(
                    top_row,
                    broken_chain,
                )
            wrong_final = dict(
                top_row,
                promoted_final_promotion_jsonl=str(
                    out_dir / "wrong-promotion.jsonl"
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "promoted rung manifest final promotion mismatch",
            ):
                module.validate_promoted_rung_manifest_consistency(
                    wrong_final,
                    [first_row, second_row],
                )

    def test_byte_lm_profile_smoke_continues_from_manifest_dry_run(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            first = module.promoted_rung_artifacts(out_dir, "profile-smoke-promoted", 1)
            smoke_args = argparse.Namespace(
                source_label="local-smoke",
                key_preset="llama",
                ft_epochs=1,
                promoted_ft_epochs=None,
                promoted_ft_epochs_step=1,
                promotion_metric="target_loss_delta_mean",
                promoted_output_prefix="profile-smoke-promoted",
                strict_aggregate_gates=False,
                skip_checkpoint_shape_audit=False,
                skip_checkpoint_preflight=False,
                compare_checkpoint_preflight_jsonl=None,
                require_checkpoint_preflight_match=False,
                transformers_audit=True,
                transformers_model_path=out_dir,
                transformers_revision=None,
                allow_transformers_remote=False,
                transformers_trust_remote_code=False,
                skip_transformers_tokenizer=False,
                transformers_load_model=False,
                require_transformers_audit=True,
                checkpoint_transformers_runtime_import_presets=["transformers"],
                checkpoint_transformers_runtime_imports=["math"],
                require_checkpoint_transformers_runtime_imports=True,
                require_checkpoint_transformers_runtime_import=["math"],
                require_checkpoint_transformers_runtime_import_preset=["transformers"],
                transformers_trace=True,
                compare_transformers_trace_jsonl=out_dir / "baseline-trace.jsonl",
                transformers_trace_prompts=["spiral"],
                transformers_trace_prompt_file=None,
                transformers_trace_top_k=3,
                transformers_trace_zspace_project=True,
                transformers_trace_zspace_source="hidden",
                transformers_trace_runtime_import_presets=["torch-transformers"],
                require_transformers_trace_runtime_import=["torch"],
                require_transformers_trace_runtime_import_preset=[
                    "torch-transformers"
                ],
                require_transformers_trace_match=True,
                require_transformers_trace_runtime_metadata_match=True,
                require_transformers_trace_top_token_match=True,
                transformers_trace_max_top_logit_regression=0.0,
                transformers_trace_max_top_probability_regression=0.1,
                transformers_trace_max_logit_l2_change=None,
                transformers_trace_max_hidden_state_l2_change=None,
                transformers_trace_require_zspace_status="ok",
            )
            row = module.profile_smoke_manifest_row(
                args=smoke_args,
                out_dir=out_dir,
                checkpoint_path=out_dir / "pytorch_model.bin",
                checkpoint_source_kind="generated",
                cases=["adapter_ja"],
                configs=["r12_a64_lr4"],
                profiles=["strong_effect"],
                checkpoint_shape_audit_jsonl=out_dir / "checkpoint-shape-audit.jsonl",
                checkpoint_preflight_jsonl=out_dir / "checkpoint-preflight.jsonl",
                sweep_jsonl=out_dir / "sweep.jsonl",
                sweep_aggregate_jsonl=out_dir / "sweep-aggregate.jsonl",
                source_compare_jsonl=out_dir / "source-compare.jsonl",
                profile_jsonl=out_dir / "profiles.jsonl",
                run_dir=out_dir / "profile-runs",
                run_events_jsonl=out_dir / "profile-run-events.jsonl",
                run_summary_jsonl=out_dir / "profile-run-summary.jsonl",
                promotion_jsonl=out_dir / "promotion.jsonl",
                promotion_compare_jsonl=out_dir / "promotion-compare.jsonl",
                promoted_rungs=1,
                promoted_rungs_jsonl=out_dir / "promoted-rungs.jsonl",
                promoted_artifacts=[first],
                transformers_trace_jsonl=out_dir / "transformers-trace.jsonl",
                transformers_trace_compare_jsonl=out_dir / "transformers-trace-compare.jsonl",
            )
            for field in module.required_manifest_artifact_fields(row):
                path = Path(row[field])
                if field in {"out_dir", "profile_run_dir"}:
                    path.mkdir(parents=True, exist_ok=True)
                elif field == "transformers_trace_jsonl":
                    module.write_jsonl(
                        path,
                        [
                            {
                                "row_type": "transformers_trace_manifest",
                                "model_path": str(out_dir),
                                "prompt_count": 1,
                                "top_k": 3,
                                "spiraltorch_imported": True,
                                "spiraltorch_version": "0.1.0",
                                "spiraltorch_module_name": "spiraltorch",
                                "transformers_imported": True,
                                "transformers_version": "9.9.9",
                                "transformers_module_name": "transformers",
                                "transformers_spiraltorch_coimport_status": "ok",
                                "runtime_import_presets": "torch-transformers",
                                "runtime_import_preset_modules": (
                                    "torch-transformers=transformers|torch"
                                ),
                                "runtime_imports_requested": "transformers,torch",
                                "runtime_import_probe_count": 2,
                                "runtime_imports_imported": "transformers,torch",
                                "runtime_imports_failed": "none",
                                "runtime_imports_all_ok": True,
                                "runtime_import_coimport_status": "ok",
                                "runtime_imports_coimported": True,
                                "runtime_import_coimport_modules": (
                                    "transformers,torch"
                                ),
                                "runtime_import_coimport_missing_modules": "none",
                                "runtime_import_versions": (
                                    "transformers=9.9.9,torch=2.0.0"
                                ),
                                "runtime_import_module_names": (
                                    "transformers=transformers,torch=torch"
                                ),
                            }
                        ],
                    )
                elif field == "checkpoint_preflight_jsonl":
                    module.write_jsonl(
                        path,
                        [
                            {
                                "row_type": "report",
                                "label": "profile-smoke",
                                "compatible": True,
                                "matched": 1,
                                "missing": 0,
                                "shape_mismatched": 0,
                                "extra": 0,
                                "source_hash": "source",
                                "matched_subset_hash": "matched",
                                "transformers_audit_requested": True,
                                "transformers_audit_status": "ok",
                                "transformers_audit_error": None,
                                "transformers_model_path": str(out_dir),
                                "transformers_available": True,
                                "transformers_version": "9.9.9",
                                "transformers_config_loaded": True,
                                "transformers_tokenizer_loaded": True,
                                "transformers_model_loaded": False,
                                "runtime_import_presets": "transformers",
                                "runtime_import_preset_modules": (
                                    "transformers=transformers"
                                ),
                                "runtime_import_presets_satisfied": "transformers",
                                "runtime_import_presets_failed": "none",
                                "runtime_import_preset_missing_modules": "none",
                                "runtime_imports_requested": "transformers,math",
                                "runtime_import_probe_count": 2,
                                "runtime_imports_imported": "transformers,math",
                                "runtime_imports_failed": "none",
                                "runtime_imports_all_ok": True,
                                "runtime_import_coimport_status": "ok",
                                "runtime_imports_coimported": True,
                                "runtime_import_coimport_modules": (
                                    "transformers,math"
                                ),
                                "runtime_import_coimport_missing_modules": "none",
                                "runtime_import_versions": (
                                    "transformers=9.9.9,math=none"
                                ),
                                "runtime_import_module_names": (
                                    "transformers=transformers,math=math"
                                ),
                                "required_runtime_imports": "math",
                                "required_runtime_imports_imported": "math",
                                "required_runtime_imports_missing": "none",
                                "required_runtime_imports_passed": True,
                                "required_runtime_import_presets": "transformers",
                                "required_runtime_import_presets_observed": (
                                    "transformers"
                                ),
                                "required_runtime_import_presets_satisfied": (
                                    "transformers"
                                ),
                                "required_runtime_import_presets_missing": "none",
                                "required_runtime_import_presets_unsatisfied": "none",
                                "required_runtime_import_presets_passed": True,
                            }
                        ],
                    )
                elif field == "transformers_trace_compare_jsonl":
                    module.write_jsonl(
                        path,
                        [
                            {
                                "row_type": "transformers_trace_compare_summary",
                                "passed": True,
                                "failures": 0,
                                "compared_prompt_rows": 1,
                                "runtime_metadata_available": True,
                                "runtime_metadata_changed_count": 0,
                                "runtime_metadata_changed_fields": "none",
                                "runtime_metadata_failures": "none",
                                "missing_prompt_rows": 0,
                                "extra_prompt_rows": 0,
                                "prompt_changed_rows": 0,
                                "top_token_changed_rows": 0,
                                "zspace_status_changed_rows": 0,
                            }
                        ],
                    )
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            first_row = module.promoted_rung_manifest_row(
                first,
                ft_epochs=2,
                input_promotion_jsonl=out_dir / "promotion.jsonl",
            )
            for field in [
                "input_promotion_jsonl",
                *module.PROMOTED_RUNG_ARTIFACT_FIELDS,
            ]:
                path = Path(first_row[field])
                if field == "output_dir":
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            module.write_jsonl(Path(row["promoted_rungs_jsonl"]), [first_row])
            manifest_path = out_dir / "profile-smoke-manifest.jsonl"
            validation_path = out_dir / "profile-smoke-manifest-validation.jsonl"
            continue_plan_path = out_dir / "continue-plan.jsonl"
            module.write_jsonl(manifest_path, [row])

            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--manifest-validation-jsonl",
                str(validation_path),
                "--key-preset",
                "auto",
            ]
            validate_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(validate_output):
                    module.main()
            finally:
                sys.argv = old_argv
            validate_text = validate_output.getvalue()
            self.assertIn("profile_smoke_manifest_validate", validate_text)
            self.assertIn("promoted_rungs=1", validate_text)
            self.assertIn("promoted_ft_epochs=2", validate_text)
            self.assertIn("promoted_rung_rows=1", validate_text)
            self.assertIn("next_promoted_rung=2", validate_text)
            self.assertIn("next_ft_epochs=3", validate_text)
            validation_rows = module.load_jsonl(validation_path)
            self.assertEqual(len(validation_rows), 1)
            validation_row = validation_rows[0]
            self.assertEqual(
                validation_row["row_type"],
                "profile_smoke_manifest_validation",
            )
            self.assertTrue(validation_row["valid"])
            self.assertEqual(validation_row["manifest_jsonl"], str(manifest_path))
            self.assertEqual(validation_row["promoted_rungs"], 1)
            self.assertEqual(validation_row["promoted_ft_epochs"], [2])
            self.assertEqual(validation_row["promoted_rung_rows"], 1)
            self.assertEqual(validation_row["next_promoted_rung"], 2)
            self.assertEqual(validation_row["next_ft_epochs"], 3)
            self.assertEqual(validation_row["promoted_rung_artifacts_checked"], 7)

            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--continue-manifest-jsonl",
                str(manifest_path),
                "--continue-rungs",
                "2",
                "--continue-plan-jsonl",
                str(continue_plan_path),
                "--dry-run",
            ]
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    module.main()
            finally:
                sys.argv = old_argv
            text = output.getvalue()
            continue_plan_rows = module.load_jsonl(continue_plan_path)

            continued_manifest_path = out_dir / "continued-manifest.jsonl"
            continued_validation_path = (
                out_dir / "continued-manifest-validation.jsonl"
            )

            def fake_run_command(cmd, *, dry_run=False):
                self.assertFalse(dry_run)
                for flag in [
                    "--commands-jsonl",
                    "--promotion-selection-jsonl",
                    "--run-events-jsonl",
                    "--run-summary-jsonl",
                    "--promotion-jsonl",
                ]:
                    if flag in cmd:
                        module.write_jsonl(
                            Path(cmd[cmd.index(flag) + 1]),
                            [{"row_type": f"fake_{flag.removeprefix('--')}"}],
                        )
                if "--output-dir" in cmd:
                    Path(cmd[cmd.index("--output-dir") + 1]).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

            old_run_command = module.run_command
            old_argv = sys.argv
            module.run_command = fake_run_command
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--continue-manifest-jsonl",
                str(manifest_path),
                "--continue-manifest-output-jsonl",
                str(continued_manifest_path),
                "--continue-rungs",
                "1",
                "--validate-produced-manifest",
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-coimport",
            ]
            continued_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(continued_output):
                    module.main()
            finally:
                module.run_command = old_run_command
                sys.argv = old_argv
            continued_text = continued_output.getvalue()
            continued_manifest_rows = module.load_jsonl(continued_manifest_path)
            continued_validation_rows = module.load_jsonl(continued_validation_path)
        self.assertIn("--promotion-input-jsonl", text)
        self.assertIn(str(out_dir / "promoted-promotion.jsonl"), text)
        self.assertIn("--output-prefix profile-smoke-promoted-rung2", text)
        self.assertIn("--output-prefix profile-smoke-promoted-rung3", text)
        self.assertIn("--override-ft-epochs 3", text)
        self.assertIn("--override-ft-epochs 4", text)
        self.assertIn("profile_smoke_manifest_continue", text)
        self.assertIn("promoted_rungs=3", text)
        self.assertIn(f"continue_plan_jsonl={continue_plan_path}", text)
        self.assertEqual(len(continue_plan_rows), 2)
        self.assertEqual(
            [plan["row_type"] for plan in continue_plan_rows],
            ["profile_smoke_continue_plan", "profile_smoke_continue_plan"],
        )
        self.assertEqual([plan["rung"] for plan in continue_plan_rows], [2, 3])
        self.assertEqual([plan["ft_epochs"] for plan in continue_plan_rows], [3, 4])
        self.assertEqual(
            [plan["input_promotion_jsonl"] for plan in continue_plan_rows],
            [
                str(out_dir / "promoted-promotion.jsonl"),
                str(out_dir / "promoted-rung2-promotion.jsonl"),
            ],
        )
        self.assertEqual(
            [plan["promotion_jsonl"] for plan in continue_plan_rows],
            [
                str(out_dir / "promoted-rung2-promotion.jsonl"),
                str(out_dir / "promoted-rung3-promotion.jsonl"),
            ],
        )
        self.assertEqual(
            [plan["output_prefix"] for plan in continue_plan_rows],
            ["profile-smoke-promoted-rung2", "profile-smoke-promoted-rung3"],
        )
        self.assertEqual(
            [plan["transformers_trace"] for plan in continue_plan_rows],
            [True, True],
        )
        self.assertEqual(
            [
                plan["require_transformers_trace_runtime_metadata_match"]
                for plan in continue_plan_rows
            ],
            [True, True],
        )
        self.assertEqual(
            [
                plan["transformers_trace_runtime_import_presets"]
                for plan in continue_plan_rows
            ],
            [["torch-transformers"], ["torch-transformers"]],
        )
        self.assertEqual(
            [
                plan["declared_transformers_trace_runtime_import_preset_modules"]
                for plan in continue_plan_rows
            ],
            [
                ["torch-transformers=transformers|torch"],
                ["torch-transformers=transformers|torch"],
            ],
        )
        self.assertEqual(
            [
                plan["require_transformers_trace_runtime_import"]
                for plan in continue_plan_rows
            ],
            [["torch"], ["torch"]],
        )
        self.assertEqual(
            [
                plan["require_transformers_trace_runtime_import_preset"]
                for plan in continue_plan_rows
            ],
            [["torch-transformers"], ["torch-transformers"]],
        )
        self.assertEqual(
            [
                plan["checkpoint_transformers_runtime_import_presets"]
                for plan in continue_plan_rows
            ],
            [["transformers"], ["transformers"]],
        )
        self.assertEqual(
            [
                plan["checkpoint_transformers_runtime_imports"]
                for plan in continue_plan_rows
            ],
            [["math"], ["math"]],
        )
        self.assertEqual(
            [
                plan["require_checkpoint_transformers_runtime_imports"]
                for plan in continue_plan_rows
            ],
            [True, True],
        )
        self.assertEqual(
            [
                plan["require_checkpoint_transformers_runtime_import"]
                for plan in continue_plan_rows
            ],
            [["math"], ["math"]],
        )
        self.assertEqual(
            [
                plan["require_checkpoint_transformers_runtime_import_preset"]
                for plan in continue_plan_rows
            ],
            [["transformers"], ["transformers"]],
        )
        self.assertEqual(
            [
                plan["transformers_trace_compare_jsonl"]
                for plan in continue_plan_rows
            ],
            [
                str(out_dir / "transformers-trace-compare.jsonl"),
                str(out_dir / "transformers-trace-compare.jsonl"),
            ],
        )
        self.assertEqual(
            [
                plan["transformers_trace_manifest_available"]
                for plan in continue_plan_rows
            ],
            [True, True],
        )
        self.assertEqual(
            [
                plan["transformers_trace_runtime_imports_imported"]
                for plan in continue_plan_rows
            ],
            ["transformers,torch", "transformers,torch"],
        )
        self.assertEqual(
            [
                plan["transformers_trace_runtime_import_coimport_status"]
                for plan in continue_plan_rows
            ],
            ["ok", "ok"],
        )
        self.assertEqual(
            [
                plan["transformers_trace_runtime_import_coimport_modules"]
                for plan in continue_plan_rows
            ],
            ["transformers,torch", "transformers,torch"],
        )
        self.assertEqual(
            [
                plan["transformers_trace_runtime_import_coimport_missing_modules"]
                for plan in continue_plan_rows
            ],
            ["none", "none"],
        )
        self.assertEqual(
            [
                plan["checkpoint_transformers_audit_available"]
                for plan in continue_plan_rows
            ],
            [True, True],
        )
        self.assertEqual(
            [
                plan["checkpoint_transformers_runtime_imports_imported"]
                for plan in continue_plan_rows
            ],
            ["transformers,math", "transformers,math"],
        )
        self.assertEqual(
            [
                plan["checkpoint_transformers_runtime_import_coimport_status"]
                for plan in continue_plan_rows
            ],
            ["ok", "ok"],
        )
        self.assertEqual(
            [
                plan["checkpoint_transformers_runtime_import_coimport_modules"]
                for plan in continue_plan_rows
            ],
            ["transformers,math", "transformers,math"],
        )
        self.assertEqual(
            [
                plan[
                    "checkpoint_transformers_runtime_import_coimport_missing_modules"
                ]
                for plan in continue_plan_rows
            ],
            ["none", "none"],
        )
        self.assertIn("profile_smoke_manifest_validate", continued_text)
        self.assertIn("profile_smoke_manifest_continue", continued_text)
        self.assertIn("validated_produced_manifest=True", continued_text)
        self.assertIn("gate=checkpoint_transformers", continued_text)
        self.assertIn(
            "checkpoint_transformers_runtime_imports_imported=transformers,math",
            continued_text,
        )
        self.assertIn(
            f"manifest_validation_jsonl={continued_validation_path}",
            continued_text,
        )
        self.assertEqual(continued_manifest_rows[0]["promoted_rungs"], 2)
        self.assertEqual(continued_manifest_rows[0]["promoted_ft_epochs"], [2, 3])
        self.assertEqual(
            continued_manifest_rows[0][
                "declared_transformers_trace_runtime_import_preset_modules"
            ],
            ["torch-transformers=transformers|torch"],
        )
        self.assertEqual(
            continued_manifest_rows[0][
                "checkpoint_transformers_runtime_import_presets"
            ],
            ["transformers"],
        )
        self.assertEqual(
            continued_manifest_rows[0]["checkpoint_transformers_runtime_imports"],
            ["math"],
        )
        self.assertEqual(continued_validation_rows[0]["promoted_rungs"], 2)
        self.assertEqual(continued_validation_rows[0]["next_promoted_rung"], 3)
        self.assertEqual(
            continued_validation_rows[0]["transformers_trace_coimport_status"],
            "ok",
        )
        self.assertEqual(
            continued_validation_rows[0][
                "declared_checkpoint_transformers_runtime_import_presets"
            ],
            "transformers",
        )
        self.assertEqual(
            continued_validation_rows[0][
                "checkpoint_transformers_runtime_imports_imported"
            ],
            "transformers,math",
        )
        self.assertTrue(
            continued_validation_rows[0][
                "checkpoint_transformers_required_runtime_import_presets_passed"
            ]
        )

    def write_profile_smoke_manifest_with_transformers_trace_compare(
        self,
        module,
        out_dir,
        *,
        passed=True,
        failures=0,
        top_token_changed_rows=1,
        top_probability_regression=0.15,
        coimport_status="ok",
        spiraltorch_imported=True,
        transformers_imported=True,
        runtime_import_probe_count=1,
        runtime_imports_all_ok=True,
        runtime_imports_failed="none",
        runtime_import_coimport_status=None,
        runtime_imports_coimported=None,
        runtime_import_coimport_modules=None,
        runtime_import_coimport_missing_modules=None,
        runtime_import_presets="torch-transformers",
        runtime_import_presets_satisfied=None,
        runtime_import_presets_failed=None,
        runtime_import_preset_missing_modules=None,
        declared_runtime_import_presets=None,
        declared_runtime_import_preset_modules=None,
        trace_runtime_import_preset_modules=None,
        direct_required_runtime_imports="none",
        direct_required_runtime_imports_imported="none",
        direct_required_runtime_imports_missing="none",
        direct_required_runtime_imports_passed=None,
        direct_required_runtime_import_presets="none",
        direct_required_runtime_import_presets_observed="none",
        direct_required_runtime_import_presets_satisfied="none",
        direct_required_runtime_import_presets_missing="none",
        direct_required_runtime_import_presets_unsatisfied="none",
        direct_required_runtime_import_presets_passed=None,
        checkpoint_runtime_import_probe_count=2,
        checkpoint_runtime_imports_all_ok=True,
        checkpoint_runtime_imports_failed="none",
        checkpoint_runtime_import_coimport_status=None,
        checkpoint_runtime_imports_coimported=None,
        checkpoint_runtime_import_coimport_modules=None,
        checkpoint_runtime_import_coimport_missing_modules=None,
        checkpoint_runtime_import_presets="transformers",
        checkpoint_runtime_import_presets_satisfied=None,
        checkpoint_runtime_import_presets_failed=None,
        checkpoint_runtime_import_preset_missing_modules=None,
        checkpoint_required_runtime_imports="math",
        checkpoint_required_runtime_imports_imported="math",
        checkpoint_required_runtime_imports_missing="none",
        checkpoint_required_runtime_imports_passed=True,
        checkpoint_required_runtime_import_presets="transformers",
        checkpoint_required_runtime_import_presets_observed="transformers",
        checkpoint_required_runtime_import_presets_satisfied="transformers",
        checkpoint_required_runtime_import_presets_missing="none",
        checkpoint_required_runtime_import_presets_unsatisfied="none",
        checkpoint_required_runtime_import_presets_passed=True,
    ):
        from spiraltorch.runtime_imports import (
            runtime_import_preset_missing_modules_label,
            runtime_import_preset_modules_label,
            runtime_import_preset_status_rows,
        )

        trace_jsonl = out_dir / "transformers-trace.jsonl"
        baseline_trace_jsonl = out_dir / "transformers-trace-baseline.jsonl"
        trace_compare_jsonl = out_dir / "transformers-trace-compare.jsonl"
        preset_module_map = module.TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
        requested_presets = [
            preset
            for preset in str(runtime_import_presets).split(",")
            if preset and preset != "none"
        ]
        if runtime_import_presets_satisfied is None:
            runtime_import_presets_satisfied = (
                runtime_import_presets if runtime_imports_all_ok else "none"
            )
        if runtime_import_presets_failed is None:
            runtime_import_presets_failed = (
                "none" if runtime_imports_all_ok else runtime_import_presets
            )
        if runtime_import_coimport_status is None:
            runtime_import_coimport_status = (
                "ok" if runtime_imports_all_ok else "missing"
            )
        if runtime_imports_coimported is None:
            runtime_imports_coimported = runtime_import_coimport_status == "ok"
        if runtime_import_coimport_modules is None:
            runtime_import_coimport_modules = (
                "torch" if runtime_imports_all_ok else "none"
            )
        if runtime_import_coimport_missing_modules is None:
            runtime_import_coimport_missing_modules = (
                "none" if runtime_imports_all_ok else runtime_imports_failed
            )
        failed_set = {
            preset
            for preset in str(runtime_import_presets_failed).split(",")
            if preset and preset != "none"
        }
        missing_modules = set()
        for preset in requested_presets:
            modules = preset_module_map.get(preset, [])
            if preset in failed_set and "torch" in modules:
                missing_modules.add("torch")
        preset_probe_rows = []
        seen_modules = set()
        for preset in requested_presets:
            modules = preset_module_map.get(preset, [])
            for module_name in modules:
                if module_name not in seen_modules:
                    seen_modules.add(module_name)
                    preset_probe_rows.append(
                        {
                            "module": module_name,
                            "imported": module_name not in missing_modules,
                        }
                    )
        runtime_import_preset_status = runtime_import_preset_status_rows(
            requested_presets,
            preset_probe_rows,
            preset_modules=preset_module_map,
        )
        runtime_import_preset_modules = runtime_import_preset_modules_label(
            runtime_import_preset_status
        )
        if runtime_import_preset_missing_modules is None:
            runtime_import_preset_missing_modules = (
                runtime_import_preset_missing_modules_label(
                    runtime_import_preset_status
                )
            )
        checkpoint_requested_presets = [
            preset
            for preset in str(checkpoint_runtime_import_presets).split(",")
            if preset and preset != "none"
        ]
        if checkpoint_runtime_import_presets_satisfied is None:
            checkpoint_runtime_import_presets_satisfied = (
                checkpoint_runtime_import_presets
                if checkpoint_runtime_imports_all_ok
                else "none"
            )
        if checkpoint_runtime_import_presets_failed is None:
            checkpoint_runtime_import_presets_failed = (
                "none"
                if checkpoint_runtime_imports_all_ok
                else checkpoint_runtime_import_presets
            )
        if checkpoint_runtime_import_coimport_status is None:
            checkpoint_runtime_import_coimport_status = (
                "ok" if checkpoint_runtime_imports_all_ok else "missing"
            )
        if checkpoint_runtime_imports_coimported is None:
            checkpoint_runtime_imports_coimported = (
                checkpoint_runtime_import_coimport_status == "ok"
            )
        if checkpoint_runtime_import_coimport_modules is None:
            checkpoint_runtime_import_coimport_modules = (
                "transformers,math"
                if checkpoint_runtime_imports_all_ok
                else "none"
            )
        if checkpoint_runtime_import_coimport_missing_modules is None:
            checkpoint_runtime_import_coimport_missing_modules = (
                "none"
                if checkpoint_runtime_imports_all_ok
                else checkpoint_runtime_imports_failed
            )
        checkpoint_failed_set = {
            preset
            for preset in str(checkpoint_runtime_import_presets_failed).split(",")
            if preset and preset != "none"
        }
        checkpoint_missing_modules = set()
        for preset in checkpoint_requested_presets:
            modules = preset_module_map.get(preset, [])
            if preset in checkpoint_failed_set:
                checkpoint_missing_modules.update(modules)
        checkpoint_preset_probe_rows = []
        checkpoint_seen_modules = set()
        for preset in checkpoint_requested_presets:
            modules = preset_module_map.get(preset, [])
            for module_name in modules:
                if module_name not in checkpoint_seen_modules:
                    checkpoint_seen_modules.add(module_name)
                    checkpoint_preset_probe_rows.append(
                        {
                            "module": module_name,
                            "imported": module_name not in checkpoint_missing_modules,
                        }
                    )
        checkpoint_runtime_import_preset_status = runtime_import_preset_status_rows(
            checkpoint_requested_presets,
            checkpoint_preset_probe_rows,
            preset_modules=preset_module_map,
        )
        checkpoint_runtime_import_preset_modules = runtime_import_preset_modules_label(
            checkpoint_runtime_import_preset_status
        )
        if checkpoint_runtime_import_preset_missing_modules is None:
            checkpoint_runtime_import_preset_missing_modules = (
                runtime_import_preset_missing_modules_label(
                    checkpoint_runtime_import_preset_status
                )
            )
        if declared_runtime_import_presets is None:
            declared_runtime_import_presets = ["torch-transformers"]
        smoke_args = argparse.Namespace(
            source_label="llama-3.2-3b",
            key_preset="auto",
            ft_epochs=1,
            promoted_ft_epochs=None,
            promoted_ft_epochs_step=1,
            promotion_metric="target_loss_delta_mean",
            promoted_output_prefix="profile-smoke-promoted",
            strict_aggregate_gates=False,
            skip_checkpoint_shape_audit=False,
            skip_checkpoint_preflight=False,
            compare_checkpoint_preflight_jsonl=None,
            require_checkpoint_preflight_match=False,
            transformers_trace=True,
            compare_transformers_trace_jsonl=baseline_trace_jsonl,
            transformers_trace_prompts=["spiral"],
            transformers_trace_prompt_file=None,
            transformers_trace_top_k=3,
            transformers_trace_zspace_project=True,
            transformers_trace_zspace_source="hidden",
            require_transformers_trace_match=True,
            require_transformers_trace_top_token_match=True,
            transformers_trace_max_top_logit_regression=0.0,
            transformers_trace_max_top_probability_regression=0.1,
            transformers_trace_max_logit_l2_change=None,
            transformers_trace_max_hidden_state_l2_change=None,
            transformers_trace_require_zspace_status="ok",
            transformers_trace_runtime_import_presets=declared_runtime_import_presets,
            transformers_trace_runtime_imports=["torch"],
            require_transformers_trace_runtime_imports=True,
        )
        row = module.profile_smoke_manifest_row(
            args=smoke_args,
            out_dir=out_dir,
            checkpoint_path=out_dir / "pytorch_model.bin",
            checkpoint_source_kind="external",
            cases=["adapter_ja"],
            configs=["r12_a64_lr4"],
            profiles=["strong_effect"],
            checkpoint_shape_audit_jsonl=out_dir / "checkpoint-shape-audit.jsonl",
            checkpoint_preflight_jsonl=out_dir / "checkpoint-preflight.jsonl",
            sweep_jsonl=out_dir / "sweep.jsonl",
            sweep_aggregate_jsonl=out_dir / "sweep-aggregate.jsonl",
            source_compare_jsonl=out_dir / "source-compare.jsonl",
            profile_jsonl=out_dir / "profiles.jsonl",
            run_dir=out_dir / "profile-runs",
            run_events_jsonl=out_dir / "profile-run-events.jsonl",
            run_summary_jsonl=out_dir / "profile-run-summary.jsonl",
            promotion_jsonl=out_dir / "promotion.jsonl",
            promotion_compare_jsonl=out_dir / "promotion-compare.jsonl",
            promoted_rungs=0,
            promoted_rungs_jsonl=out_dir / "promoted-rungs.jsonl",
            promoted_artifacts=[],
            transformers_trace_jsonl=trace_jsonl,
            transformers_trace_compare_jsonl=trace_compare_jsonl,
        )
        if declared_runtime_import_preset_modules is not None:
            row["declared_transformers_trace_runtime_import_preset_modules"] = (
                declared_runtime_import_preset_modules
            )
        for field in module.required_manifest_artifact_fields(row):
            path = Path(row[field])
            if field in {"out_dir", "profile_run_dir"}:
                path.mkdir(parents=True, exist_ok=True)
            elif field == "transformers_trace_jsonl":
                module.write_jsonl(
                    path,
                    [
                        {
                            "row_type": "transformers_trace_manifest",
                            "model_path": str(out_dir),
                            "prompt_count": 1,
                            "top_k": 3,
                            "spiraltorch_imported": spiraltorch_imported,
                            "spiraltorch_version": "0.1.0",
                            "spiraltorch_module_name": "spiraltorch",
                            "transformers_imported": transformers_imported,
                            "transformers_version": "9.9.9",
                            "transformers_module_name": "transformers",
                            "transformers_spiraltorch_coimport_status": (
                                coimport_status
                            ),
                            "runtime_import_presets": runtime_import_presets,
                            "runtime_import_preset_modules": (
                                runtime_import_preset_modules
                                if trace_runtime_import_preset_modules is None
                                else trace_runtime_import_preset_modules
                            ),
                            "runtime_import_presets_satisfied": (
                                runtime_import_presets_satisfied
                            ),
                            "runtime_import_presets_failed": (
                                runtime_import_presets_failed
                            ),
                            "runtime_import_preset_missing_modules": (
                                runtime_import_preset_missing_modules
                            ),
                            "runtime_imports_requested": "torch",
                            "runtime_import_probe_count": runtime_import_probe_count,
                            "runtime_imports_imported": (
                                "torch" if runtime_imports_all_ok else "none"
                            ),
                            "runtime_imports_failed": runtime_imports_failed,
                            "runtime_imports_all_ok": runtime_imports_all_ok,
                            "runtime_import_coimport_status": (
                                runtime_import_coimport_status
                            ),
                            "runtime_imports_coimported": (
                                runtime_imports_coimported
                            ),
                            "runtime_import_coimport_modules": (
                                runtime_import_coimport_modules
                            ),
                            "runtime_import_coimport_missing_modules": (
                                runtime_import_coimport_missing_modules
                            ),
                            "runtime_import_versions": (
                                "torch=2.0.0"
                                if runtime_imports_all_ok
                                else "torch=none"
                            ),
                            "runtime_import_module_names": (
                                "torch=torch"
                                if runtime_imports_all_ok
                                else "torch=none"
                            ),
                            "runtime_imports_json": json.dumps(
                                [
                                    {
                                        "module": "torch",
                                        "imported": runtime_imports_all_ok,
                                        "version": (
                                            "2.0.0" if runtime_imports_all_ok else None
                                        ),
                                        "module_name": (
                                            "torch" if runtime_imports_all_ok else None
                                        ),
                                        "module_file": (
                                            "/env/torch.py"
                                            if runtime_imports_all_ok
                                            else None
                                        ),
                                        "error": (
                                            None
                                            if runtime_imports_all_ok
                                            else "ImportError: missing"
                                        ),
                                    }
                                ],
                                sort_keys=True,
                            ),
                            "runtime_import_preset_status_json": json.dumps(
                                runtime_import_preset_status,
                                sort_keys=True,
                            ),
                            "required_runtime_imports": direct_required_runtime_imports,
                            "required_runtime_imports_imported": (
                                direct_required_runtime_imports_imported
                            ),
                            "required_runtime_imports_missing": (
                                direct_required_runtime_imports_missing
                            ),
                            "required_runtime_imports_passed": (
                                direct_required_runtime_imports_passed
                            ),
                            "required_runtime_import_presets": (
                                direct_required_runtime_import_presets
                            ),
                            "required_runtime_import_presets_observed": (
                                direct_required_runtime_import_presets_observed
                            ),
                            "required_runtime_import_presets_satisfied": (
                                direct_required_runtime_import_presets_satisfied
                            ),
                            "required_runtime_import_presets_missing": (
                                direct_required_runtime_import_presets_missing
                            ),
                            "required_runtime_import_presets_unsatisfied": (
                                direct_required_runtime_import_presets_unsatisfied
                            ),
                            "required_runtime_import_presets_passed": (
                                direct_required_runtime_import_presets_passed
                            ),
                        },
                        {
                            "row_type": "transformers_prompt_trace",
                            "prompt_index": 0,
                            "prompt": "spiral",
                            "input_ids_tensor_available": True,
                            "input_ids_tensor_backend": "python_sequence",
                            "input_ids_tensor_shape": "1x3",
                            "logits_tensor_available": True,
                            "logits_tensor_backend": "torch",
                            "logits_tensor_device": "mps:0",
                            "logits_tensor_device_kind": "mps",
                            "logits_tensor_dtype": "torch.float16",
                            "logits_tensor_shape": "1x3x8",
                            "hidden_state_tensor_available": True,
                            "hidden_state_tensor_backend": "torch",
                            "hidden_state_tensor_device": "mps:0",
                            "hidden_state_tensor_device_kind": "mps",
                            "hidden_state_tensor_dtype": "torch.float16",
                            "hidden_state_tensor_shape": "1x3x4",
                        },
                    ],
                )
            elif field == "run_events_jsonl":
                module.write_jsonl(
                    path,
                    [
                        {
                            "event": {
                                "kind": "Custom",
                                "data": {
                                    "event_type": "TensorOpMeta",
                                    "data": {
                                        "op_name": "matmul",
                                        "data": {
                                            "backend": "naive",
                                            "requested_backend": "wgpu",
                                            "fallback": {
                                                "from": "wgpu",
                                                "reason": "runtime_unavailable",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        {
                            "event": {
                                "kind": "Custom",
                                "data": {
                                    "event_type": "TensorOpMeta",
                                    "data": {
                                        "op_name": "matmul_prepacked_bias",
                                        "data": {
                                            "backend": "wgpu",
                                            "requested_backend": "wgpu",
                                        },
                                    },
                                },
                            },
                        },
                        {
                            "event": {
                                "kind": "Custom",
                                "data": {
                                    "event_type": "TrainerStep",
                                    "data": {
                                        "step": 1,
                                        "metrics": {"step_time_ms": 0.5},
                                    },
                                },
                            },
                        },
                    ],
                )
            elif field == "checkpoint_preflight_jsonl":
                module.write_jsonl(
                    path,
                    [
                        {
                            "row_type": "report",
                            "label": "profile-smoke",
                            "compatible": True,
                            "matched": 1,
                            "missing": 0,
                            "shape_mismatched": 0,
                            "extra": 0,
                            "source_hash": "source",
                            "matched_subset_hash": "matched",
                            "transformers_audit_requested": True,
                            "transformers_audit_status": "ok",
                            "transformers_audit_error": None,
                            "transformers_model_path": str(out_dir),
                            "transformers_available": True,
                            "transformers_version": "9.9.9",
                            "transformers_config_loaded": True,
                            "transformers_tokenizer_loaded": True,
                            "transformers_model_loaded": False,
                            "runtime_import_presets": checkpoint_runtime_import_presets,
                            "runtime_import_preset_modules": (
                                checkpoint_runtime_import_preset_modules
                            ),
                            "runtime_import_presets_satisfied": (
                                checkpoint_runtime_import_presets_satisfied
                            ),
                            "runtime_import_presets_failed": (
                                checkpoint_runtime_import_presets_failed
                            ),
                            "runtime_import_preset_missing_modules": (
                                checkpoint_runtime_import_preset_missing_modules
                            ),
                            "runtime_imports_requested": "transformers,math",
                            "runtime_import_probe_count": (
                                checkpoint_runtime_import_probe_count
                            ),
                            "runtime_imports_imported": (
                                "transformers,math"
                                if checkpoint_runtime_imports_all_ok
                                else "none"
                            ),
                            "runtime_imports_failed": (
                                checkpoint_runtime_imports_failed
                            ),
                            "runtime_imports_all_ok": (
                                checkpoint_runtime_imports_all_ok
                            ),
                            "runtime_import_coimport_status": (
                                checkpoint_runtime_import_coimport_status
                            ),
                            "runtime_imports_coimported": (
                                checkpoint_runtime_imports_coimported
                            ),
                            "runtime_import_coimport_modules": (
                                checkpoint_runtime_import_coimport_modules
                            ),
                            "runtime_import_coimport_missing_modules": (
                                checkpoint_runtime_import_coimport_missing_modules
                            ),
                            "runtime_import_versions": (
                                "transformers=9.9.9,math=none"
                                if checkpoint_runtime_imports_all_ok
                                else "transformers=none,math=none"
                            ),
                            "runtime_import_module_names": (
                                "transformers=transformers,math=math"
                                if checkpoint_runtime_imports_all_ok
                                else "transformers=none,math=none"
                            ),
                            "runtime_imports_json": json.dumps(
                                [
                                    {
                                        "module": "transformers",
                                        "imported": (
                                            checkpoint_runtime_imports_all_ok
                                        ),
                                        "version": (
                                            "9.9.9"
                                            if checkpoint_runtime_imports_all_ok
                                            else None
                                        ),
                                        "module_name": (
                                            "transformers"
                                            if checkpoint_runtime_imports_all_ok
                                            else None
                                        ),
                                        "module_file": (
                                            "/env/transformers.py"
                                            if checkpoint_runtime_imports_all_ok
                                            else None
                                        ),
                                        "error": (
                                            None
                                            if checkpoint_runtime_imports_all_ok
                                            else "ImportError: missing"
                                        ),
                                    },
                                    {
                                        "module": "math",
                                        "imported": (
                                            checkpoint_runtime_imports_all_ok
                                        ),
                                        "version": None,
                                        "module_name": (
                                            "math"
                                            if checkpoint_runtime_imports_all_ok
                                            else None
                                        ),
                                        "module_file": None,
                                        "error": (
                                            None
                                            if checkpoint_runtime_imports_all_ok
                                            else "ImportError: missing"
                                        ),
                                    },
                                ],
                                sort_keys=True,
                            ),
                            "runtime_import_preset_status_json": json.dumps(
                                checkpoint_runtime_import_preset_status,
                                sort_keys=True,
                            ),
                            "required_runtime_imports": (
                                checkpoint_required_runtime_imports
                            ),
                            "required_runtime_imports_imported": (
                                checkpoint_required_runtime_imports_imported
                            ),
                            "required_runtime_imports_missing": (
                                checkpoint_required_runtime_imports_missing
                            ),
                            "required_runtime_imports_passed": (
                                checkpoint_required_runtime_imports_passed
                            ),
                            "required_runtime_import_presets": (
                                checkpoint_required_runtime_import_presets
                            ),
                            "required_runtime_import_presets_observed": (
                                checkpoint_required_runtime_import_presets_observed
                            ),
                            "required_runtime_import_presets_satisfied": (
                                checkpoint_required_runtime_import_presets_satisfied
                            ),
                            "required_runtime_import_presets_missing": (
                                checkpoint_required_runtime_import_presets_missing
                            ),
                            "required_runtime_import_presets_unsatisfied": (
                                checkpoint_required_runtime_import_presets_unsatisfied
                            ),
                            "required_runtime_import_presets_passed": (
                                checkpoint_required_runtime_import_presets_passed
                            ),
                        }
                    ],
                )
            elif field == "transformers_trace_compare_jsonl":
                module.write_jsonl(
                    path,
                    [
                        {
                            "row_type": "transformers_trace_compare_summary",
                            "passed": passed,
                            "failures": failures,
                            "compared_prompt_rows": 2,
                            "runtime_metadata_available": True,
                            "runtime_metadata_changed_count": 0,
                            "runtime_metadata_changed_fields": "none",
                            "runtime_metadata_failures": "none",
                            "missing_prompt_rows": 0,
                            "extra_prompt_rows": 0,
                            "prompt_changed_rows": 0,
                            "top_token_changed_rows": top_token_changed_rows,
                            "zspace_status_changed_rows": 1,
                            "observed_max_top_logit_regression": 0.2,
                            "observed_max_top_probability_regression": (
                                top_probability_regression
                            ),
                            "observed_max_logit_l2_change": 0.6,
                            "observed_max_hidden_state_l2_change": 0.3,
                        }
                    ],
                )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        manifest_path = out_dir / "profile-smoke-manifest.jsonl"
        validation_path = out_dir / "profile-smoke-manifest-validation.jsonl"
        module.write_jsonl(manifest_path, [row])
        return manifest_path, validation_path

    def test_byte_lm_profile_smoke_validation_reads_transformers_trace_compare(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_path, validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir,
                )
            )
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--manifest-validation-jsonl",
                str(validation_path),
            ]
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    module.main()
            finally:
                sys.argv = old_argv
            text = output.getvalue()
            validation_row = module.load_jsonl(validation_path)[0]

        self.assertIn("transformers_trace_compare_passed=True", text)
        self.assertIn("transformers_trace_coimport_status=ok", text)
        self.assertIn("transformers_trace_runtime_import_coimport_status=ok", text)
        self.assertIn("transformers_trace_runtime_imports_coimported=True", text)
        self.assertIn("transformers_trace_top_token_changed_rows=1", text)
        self.assertIn(
            "transformers_trainer_runtime_status="
            "external_gpu_with_trainer_wgpu_fallback",
            text,
        )
        self.assertIn(
            "transformers_trainer_runtime_requested_wgpu_hit_rate=0.5",
            text,
        )
        self.assertIn("checkpoint_transformers_audit_status=ok", text)
        self.assertIn("checkpoint_transformers_runtime_imports_all_ok=True", text)
        self.assertIn(
            "checkpoint_transformers_runtime_import_coimport_status=ok",
            text,
        )
        self.assertIn(
            "declared_transformers_trace_runtime_import_preset_modules="
            "torch-transformers=transformers|torch",
            text,
        )
        self.assertIn(
            "declared_transformers_trace_runtime_import_preset_modules_match=True",
            text,
        )
        self.assertIn(
            "transformers_trace_declared_runtime_import_preset_modules_match=True",
            text,
        )
        self.assertTrue(validation_row["transformers_trace"])
        self.assertTrue(validation_row["transformers_trace_manifest_available"])
        self.assertTrue(validation_row["transformers_trace_spiraltorch_imported"])
        self.assertTrue(validation_row["transformers_trace_transformers_imported"])
        self.assertEqual(validation_row["transformers_trace_coimport_status"], "ok")
        self.assertEqual(
            validation_row["transformers_trace_transformers_version"],
            "9.9.9",
        )
        self.assertTrue(validation_row["transformers_trace_compare_passed"])
        self.assertEqual(validation_row["transformers_trace_compare_failures"], 0)
        self.assertEqual(validation_row["transformers_trace_compared_prompt_rows"], 2)
        self.assertTrue(validation_row["transformers_trace_runtime_metadata_available"])
        self.assertEqual(
            validation_row["transformers_trace_runtime_metadata_changed_count"],
            0,
        )
        self.assertEqual(validation_row["transformers_trace_top_token_changed_rows"], 1)
        self.assertTrue(
            validation_row["transformers_trainer_runtime_bridge_available"]
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_status"],
            "external_gpu_with_trainer_wgpu_fallback",
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_prompt_rows"],
            1,
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_tensor_fields"],
            3,
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_tensor_backends"],
            '{"python_sequence":1,"torch":2}',
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_tensor_device_kinds"],
            '{"mps":2}',
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_tensor_dtypes"],
            '{"torch.float16":2}',
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_gpu_tensor_fields"],
            2,
        )
        self.assertEqual(
            validation_row[
                "transformers_trainer_runtime_python_sequence_tensor_fields"
            ],
            1,
        )
        self.assertEqual(
            validation_row["transformers_trainer_runtime_trainer_steps"],
            1,
        )
        self.assertAlmostEqual(
            validation_row["transformers_trainer_runtime_requested_wgpu_hits"],
            1.0,
        )
        self.assertAlmostEqual(
            validation_row[
                "transformers_trainer_runtime_requested_wgpu_runtime_fallbacks"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            validation_row["transformers_trainer_runtime_requested_wgpu_hit_rate"],
            0.5,
        )
        self.assertAlmostEqual(
            validation_row[
                "transformers_trainer_runtime_requested_wgpu_runtime_fallback_rate"
            ],
            0.5,
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_probe_count"],
            1,
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_imports_requested"],
            "torch",
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_presets"],
            "torch-transformers",
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_preset_modules"],
            "torch-transformers=transformers|torch",
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_presets_satisfied"],
            "torch-transformers",
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_presets_failed"],
            "none",
        )
        self.assertEqual(
            validation_row["declared_transformers_trace_runtime_import_presets"],
            "torch-transformers",
        )
        self.assertEqual(
            validation_row[
                "declared_transformers_trace_runtime_import_preset_modules"
            ],
            "torch-transformers=transformers|torch",
        )
        self.assertEqual(
            validation_row[
                "declared_transformers_trace_runtime_import_preset_modules_expected"
            ],
            "torch-transformers=transformers|torch",
        )
        self.assertTrue(
            validation_row[
                "declared_transformers_trace_runtime_import_preset_modules_match"
            ]
        )
        self.assertEqual(
            validation_row[
                "transformers_trace_declared_runtime_import_preset_modules"
            ],
            "torch-transformers=transformers|torch",
        )
        self.assertTrue(
            validation_row[
                "transformers_trace_declared_runtime_import_preset_modules_match"
            ]
        )
        self.assertTrue(validation_row["transformers_trace_runtime_imports_all_ok"])
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_coimport_status"],
            "ok",
        )
        self.assertTrue(
            validation_row["transformers_trace_runtime_imports_coimported"]
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_coimport_modules"],
            "torch",
        )
        self.assertEqual(
            validation_row[
                "transformers_trace_runtime_import_coimport_missing_modules"
            ],
            "none",
        )
        self.assertEqual(
            validation_row["transformers_trace_runtime_import_versions"],
            "torch=2.0.0",
        )
        self.assertEqual(
            validation_row["transformers_trace_required_runtime_imports"],
            "none",
        )
        self.assertIsNone(
            validation_row["transformers_trace_required_runtime_imports_passed"]
        )
        self.assertEqual(
            validation_row["transformers_trace_required_runtime_import_presets"],
            "none",
        )
        self.assertIsNone(
            validation_row[
                "transformers_trace_required_runtime_import_presets_passed"
            ]
        )
        self.assertTrue(validation_row["checkpoint_transformers_audit_available"])
        self.assertEqual(
            validation_row["checkpoint_transformers_audit_status"],
            "ok",
        )
        self.assertEqual(
            validation_row["checkpoint_transformers_runtime_import_probe_count"],
            2,
        )
        self.assertEqual(
            validation_row["checkpoint_transformers_runtime_imports_requested"],
            "transformers,math",
        )
        self.assertEqual(
            validation_row["checkpoint_transformers_runtime_import_presets"],
            "transformers",
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_runtime_import_preset_modules"
            ],
            "transformers=transformers",
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_runtime_import_presets_satisfied"
            ],
            "transformers",
        )
        self.assertEqual(
            validation_row["checkpoint_transformers_runtime_imports_imported"],
            "transformers,math",
        )
        self.assertTrue(
            validation_row["checkpoint_transformers_runtime_imports_all_ok"]
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_runtime_import_coimport_status"
            ],
            "ok",
        )
        self.assertTrue(
            validation_row["checkpoint_transformers_runtime_imports_coimported"]
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_runtime_import_coimport_modules"
            ],
            "transformers,math",
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_runtime_import_coimport_missing_modules"
            ],
            "none",
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_direct_required_runtime_imports"
            ],
            "math",
        )
        self.assertTrue(
            validation_row[
                "checkpoint_transformers_direct_required_runtime_imports_passed"
            ]
        )
        self.assertEqual(
            validation_row[
                "checkpoint_transformers_direct_required_runtime_import_presets"
            ],
            "transformers",
        )
        self.assertTrue(
            validation_row[
                "checkpoint_transformers_direct_required_runtime_import_presets_passed"
            ]
        )
        self.assertEqual(
            validation_row["transformers_trace_zspace_status_changed_rows"],
            1,
        )
        self.assertAlmostEqual(
            validation_row[
                "transformers_trace_observed_max_top_probability_regression"
            ],
            0.15,
        )

    def test_byte_lm_profile_smoke_validation_gates_transformers_trace_compare(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_path, validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir,
                )
            )
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--manifest-validation-jsonl",
                str(validation_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-compare-pass",
                "--require-manifest-transformers-trace-coimport",
                "--require-manifest-transformers-trace-runtime-imports",
                "--require-manifest-transformers-trace-runtime-metadata-match",
                "--max-manifest-transformers-trace-top-token-changed-rows",
                "1",
                "--max-manifest-transformers-trace-top-probability-regression",
                "0.2",
            ]
            passing_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(passing_output):
                    module.main()
            finally:
                sys.argv = old_argv
            text = passing_output.getvalue()
            self.assertIn("profile_smoke_manifest_gate", text)
            self.assertIn("passed=True", text)

            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-compare-pass",
                "--max-manifest-transformers-trace-top-token-changed-rows",
                "0",
            ]
            failing_output = io.StringIO()
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "profile smoke manifest trace validation gate failed",
                ):
                    with contextlib.redirect_stdout(failing_output):
                        module.main()
            finally:
                sys.argv = old_argv
            text = failing_output.getvalue()
            self.assertIn("transformers_trace_top_token_changed_rows", text)
            self.assertIn("passed=False", text)

    def test_byte_lm_profile_smoke_validation_gates_transformers_trace_coimport(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_path, _validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir,
                    coimport_status="transformers_missing",
                    transformers_imported=False,
                )
            )
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-coimport",
            ]
            output = io.StringIO()
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "profile smoke manifest trace validation gate failed",
                ):
                    with contextlib.redirect_stdout(output):
                        module.main()
            finally:
                sys.argv = old_argv
            text = output.getvalue()

        self.assertIn("transformers_trace_coimport_failed", text)
        self.assertIn("passed=False", text)

    def test_byte_lm_profile_smoke_validation_gates_runtime_imports(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_path, _validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir,
                    runtime_imports_all_ok=False,
                    runtime_imports_failed="torch",
                )
            )
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-runtime-imports",
            ]
            output = io.StringIO()
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "profile smoke manifest trace validation gate failed",
                ):
                    with contextlib.redirect_stdout(output):
                        module.main()
            finally:
                sys.argv = old_argv
            text = output.getvalue()

        self.assertIn("transformers_trace_runtime_imports_failed", text)
        self.assertIn("passed=False", text)

    def test_byte_lm_profile_smoke_validation_gates_required_runtime_import_module(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_path, _validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir,
                )
            )
            validation_path = out_dir / "runtime-import-validation.jsonl"
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--manifest-validation-jsonl",
                str(validation_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-runtime-import",
                "torch",
                "--require-manifest-transformers-trace-runtime-import-preset",
                "torch-transformers",
            ]
            passing_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(passing_output):
                    module.main()
            finally:
                sys.argv = old_argv
            passing_text = passing_output.getvalue()
            passing_validation = module.load_jsonl(validation_path)[0]

            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-runtime-import",
                "tokenizers",
            ]
            failing_output = io.StringIO()
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "profile smoke manifest trace validation gate failed",
                ):
                    with contextlib.redirect_stdout(failing_output):
                        module.main()
            finally:
                sys.argv = old_argv
            failing_text = failing_output.getvalue()

            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--require-manifest-transformers-trace",
                "--require-manifest-transformers-trace-runtime-import-preset",
                "hf-runtime",
            ]
            preset_failing_output = io.StringIO()
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "profile smoke manifest trace validation gate failed",
                ):
                    with contextlib.redirect_stdout(preset_failing_output):
                        module.main()
            finally:
                sys.argv = old_argv
            preset_failing_text = preset_failing_output.getvalue()

        self.assertIn(
            "transformers_trace_runtime_imports_imported=torch",
            passing_text,
        )
        self.assertIn("passed=True", passing_text)
        self.assertEqual(
            passing_validation["transformers_trace_required_runtime_imports"],
            "torch",
        )
        self.assertEqual(
            passing_validation["transformers_trace_required_runtime_imports_imported"],
            "torch",
        )
        self.assertEqual(
            passing_validation["transformers_trace_required_runtime_imports_missing"],
            "none",
        )
        self.assertTrue(
            passing_validation["transformers_trace_required_runtime_imports_passed"]
        )
        self.assertEqual(
            passing_validation["transformers_trace_required_runtime_import_presets"],
            "torch-transformers",
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_observed"
            ],
            "torch-transformers",
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_satisfied"
            ],
            "torch-transformers",
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_unsatisfied"
            ],
            "none",
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_missing"
            ],
            "none",
        )
        self.assertTrue(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_passed"
            ]
        )
        self.assertIn(
            "transformers_trace_runtime_import_missing:tokenizers",
            failing_text,
        )
        self.assertIn("passed=False", failing_text)
        self.assertIn(
            "transformers_trace_runtime_import_preset_missing:hf-runtime",
            preset_failing_text,
        )
        self.assertIn("passed=False", preset_failing_text)

    def test_byte_lm_profile_smoke_validation_gates_checkpoint_runtime_import_module(self):
        module = load_example("byte_lm_profile_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifest_path, _validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir,
                )
            )
            validation_path = out_dir / "checkpoint-runtime-import-validation.jsonl"
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--manifest-validation-jsonl",
                str(validation_path),
                "--require-manifest-checkpoint-transformers-runtime-imports",
                "--require-manifest-checkpoint-transformers-runtime-import",
                "math",
                "--require-manifest-checkpoint-transformers-runtime-import-preset",
                "transformers",
            ]
            passing_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(passing_output):
                    module.main()
            finally:
                sys.argv = old_argv
            passing_text = passing_output.getvalue()
            passing_validation = module.load_jsonl(validation_path)[0]

            sys.argv = [
                "byte_lm_profile_smoke.py",
                "--validate-manifest-jsonl",
                str(manifest_path),
                "--require-manifest-checkpoint-transformers-runtime-import",
                "tokenizers",
            ]
            failing_output = io.StringIO()
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "profile smoke manifest checkpoint Transformers validation gate failed",
                ):
                    with contextlib.redirect_stdout(failing_output):
                        module.main()
            finally:
                sys.argv = old_argv
            failing_text = failing_output.getvalue()

        self.assertIn("gate=checkpoint_transformers", passing_text)
        self.assertIn(
            "checkpoint_transformers_runtime_imports_imported=transformers,math",
            passing_text,
        )
        self.assertIn("passed=True", passing_text)
        self.assertEqual(
            passing_validation["checkpoint_transformers_required_runtime_imports"],
            "math",
        )
        self.assertEqual(
            passing_validation[
                "checkpoint_transformers_required_runtime_imports_imported"
            ],
            "math,transformers",
        )
        self.assertTrue(
            passing_validation[
                "checkpoint_transformers_required_runtime_imports_passed"
            ]
        )
        self.assertEqual(
            passing_validation[
                "checkpoint_transformers_required_runtime_import_presets"
            ],
            "transformers",
        )
        self.assertTrue(
            passing_validation[
                "checkpoint_transformers_required_runtime_import_presets_passed"
            ]
        )
        self.assertIn(
            "checkpoint_transformers_runtime_import_missing:tokenizers",
            failing_text,
        )
        self.assertIn("passed=False", failing_text)

    def test_byte_lm_profile_smoke_produced_validation_gates_declared_runtime_import_preset(self):
        module = load_example("byte_lm_profile_smoke")

        def produced_validation_args():
            return argparse.Namespace(
                validate_produced_manifest=True,
                require_manifest_transformers_trace=False,
                require_manifest_transformers_trace_compare_pass=False,
                require_manifest_transformers_trace_runtime_metadata_match=False,
                require_manifest_transformers_trace_coimport=False,
                require_manifest_transformers_trace_runtime_imports=False,
                require_manifest_transformers_trace_runtime_import=[],
                require_manifest_transformers_trace_runtime_import_preset=[],
                max_manifest_transformers_trace_top_token_changed_rows=None,
                max_manifest_transformers_trace_top_probability_regression=None,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            passing_manifest, passing_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "passing",
                )
            )
            passing_output = io.StringIO()
            with contextlib.redirect_stdout(passing_output):
                module.validate_profile_smoke_manifest_file(
                    passing_manifest,
                    validation_jsonl=passing_validation_path,
                    args=produced_validation_args(),
                )
            passing_validation = module.load_jsonl(passing_validation_path)[0]

            failing_manifest, _failing_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "failing",
                    runtime_import_presets="hf-runtime",
                )
            )
            unsatisfied_manifest, _unsatisfied_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "unsatisfied",
                    runtime_import_presets="torch-transformers",
                    runtime_import_presets_satisfied="none",
                    runtime_import_presets_failed="torch-transformers",
                    runtime_import_preset_missing_modules=(
                        "torch-transformers=torch"
                    ),
                )
            )
            failing_output = io.StringIO()
            with self.assertRaisesRegex(
                RuntimeError,
                "profile smoke manifest trace validation gate failed",
            ):
                with contextlib.redirect_stdout(failing_output):
                    module.validate_profile_smoke_manifest_file(
                        failing_manifest,
                        args=produced_validation_args(),
                    )
            unsatisfied_output = io.StringIO()
            with self.assertRaisesRegex(
                RuntimeError,
                "profile smoke manifest trace validation gate failed",
            ):
                with contextlib.redirect_stdout(unsatisfied_output):
                    module.validate_profile_smoke_manifest_file(
                        unsatisfied_manifest,
                        args=produced_validation_args(),
                    )

        self.assertIn("profile_smoke_manifest_gate", passing_output.getvalue())
        self.assertIn("passed=True", passing_output.getvalue())
        self.assertEqual(
            passing_validation["declared_transformers_trace_runtime_import_presets"],
            "torch-transformers",
        )
        self.assertEqual(
            passing_validation["transformers_trace_required_runtime_import_presets"],
            "torch-transformers",
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_satisfied"
            ],
            "torch-transformers",
        )
        self.assertTrue(
            passing_validation[
                "transformers_trace_required_runtime_import_presets_passed"
            ]
        )
        failing_text = failing_output.getvalue()
        self.assertIn(
            "transformers_trace_runtime_import_preset_missing:torch-transformers",
            failing_text,
        )
        self.assertIn("passed=False", failing_text)
        unsatisfied_text = unsatisfied_output.getvalue()
        self.assertIn(
            "transformers_trace_runtime_import_preset_unsatisfied:torch-transformers",
            unsatisfied_text,
        )
        self.assertIn("passed=False", unsatisfied_text)

    def test_byte_lm_profile_smoke_produced_validation_gates_trace_direct_runtime_requirements(self):
        module = load_example("byte_lm_profile_smoke")

        def produced_validation_args():
            return argparse.Namespace(
                validate_produced_manifest=True,
                require_manifest_transformers_trace=False,
                require_manifest_transformers_trace_compare_pass=False,
                require_manifest_transformers_trace_runtime_metadata_match=False,
                require_manifest_transformers_trace_coimport=False,
                require_manifest_transformers_trace_runtime_imports=False,
                require_manifest_transformers_trace_runtime_import=[],
                require_manifest_transformers_trace_runtime_import_preset=[],
                max_manifest_transformers_trace_top_token_changed_rows=None,
                max_manifest_transformers_trace_top_probability_regression=None,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            passing_manifest, passing_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "passing",
                    direct_required_runtime_imports="torch",
                    direct_required_runtime_imports_imported="torch",
                    direct_required_runtime_imports_missing="none",
                    direct_required_runtime_imports_passed=True,
                    direct_required_runtime_import_presets="torch-transformers",
                    direct_required_runtime_import_presets_observed="torch-transformers",
                    direct_required_runtime_import_presets_satisfied="torch-transformers",
                    direct_required_runtime_import_presets_missing="none",
                    direct_required_runtime_import_presets_unsatisfied="none",
                    direct_required_runtime_import_presets_passed=True,
                )
            )
            passing_output = io.StringIO()
            with contextlib.redirect_stdout(passing_output):
                module.validate_profile_smoke_manifest_file(
                    passing_manifest,
                    validation_jsonl=passing_validation_path,
                    args=produced_validation_args(),
                )
            passing_validation = module.load_jsonl(passing_validation_path)[0]

            missing_manifest, _missing_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "missing",
                    direct_required_runtime_imports="tokenizers",
                    direct_required_runtime_imports_imported="none",
                    direct_required_runtime_imports_missing="tokenizers",
                    direct_required_runtime_imports_passed=False,
                )
            )
            missing_output = io.StringIO()
            with self.assertRaisesRegex(
                RuntimeError,
                "profile smoke manifest trace validation gate failed",
            ):
                with contextlib.redirect_stdout(missing_output):
                    module.validate_profile_smoke_manifest_file(
                        missing_manifest,
                        args=produced_validation_args(),
                    )

            unsatisfied_manifest, _unsatisfied_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "unsatisfied",
                    direct_required_runtime_import_presets="torch-transformers",
                    direct_required_runtime_import_presets_observed="torch-transformers",
                    direct_required_runtime_import_presets_satisfied="none",
                    direct_required_runtime_import_presets_missing="none",
                    direct_required_runtime_import_presets_unsatisfied=(
                        "torch-transformers"
                    ),
                    direct_required_runtime_import_presets_passed=False,
                )
            )
            unsatisfied_output = io.StringIO()
            with self.assertRaisesRegex(
                RuntimeError,
                "profile smoke manifest trace validation gate failed",
            ):
                with contextlib.redirect_stdout(unsatisfied_output):
                    module.validate_profile_smoke_manifest_file(
                        unsatisfied_manifest,
                        args=produced_validation_args(),
                    )

        self.assertIn("profile_smoke_manifest_gate", passing_output.getvalue())
        self.assertIn("passed=True", passing_output.getvalue())
        self.assertEqual(
            passing_validation[
                "transformers_trace_direct_required_runtime_imports"
            ],
            "torch",
        )
        self.assertTrue(
            passing_validation[
                "transformers_trace_direct_required_runtime_imports_passed"
            ]
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_direct_required_runtime_import_presets"
            ],
            "torch-transformers",
        )
        self.assertTrue(
            passing_validation[
                "transformers_trace_direct_required_runtime_import_presets_passed"
            ]
        )
        missing_text = missing_output.getvalue()
        self.assertIn(
            "transformers_trace_direct_runtime_import_missing:tokenizers",
            missing_text,
        )
        self.assertIn("passed=False", missing_text)
        unsatisfied_text = unsatisfied_output.getvalue()
        self.assertIn(
            "transformers_trace_direct_runtime_import_preset_unsatisfied:"
            "torch-transformers",
            unsatisfied_text,
        )
        self.assertIn("passed=False", unsatisfied_text)

    def test_byte_lm_profile_smoke_produced_validation_gates_runtime_preset_module_contracts(self):
        module = load_example("byte_lm_profile_smoke")

        def produced_validation_args():
            return argparse.Namespace(
                validate_produced_manifest=True,
                require_manifest_transformers_trace=False,
                require_manifest_transformers_trace_compare_pass=False,
                require_manifest_transformers_trace_runtime_metadata_match=False,
                require_manifest_transformers_trace_coimport=False,
                require_manifest_transformers_trace_runtime_imports=False,
                require_manifest_transformers_trace_runtime_import=[],
                require_manifest_transformers_trace_runtime_import_preset=[],
                max_manifest_transformers_trace_top_token_changed_rows=None,
                max_manifest_transformers_trace_top_probability_regression=None,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            passing_manifest, passing_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "passing",
                    runtime_import_presets="hf-runtime",
                    declared_runtime_import_presets=["hf-runtime"],
                )
            )
            passing_output = io.StringIO()
            with contextlib.redirect_stdout(passing_output):
                module.validate_profile_smoke_manifest_file(
                    passing_manifest,
                    validation_jsonl=passing_validation_path,
                    args=produced_validation_args(),
                )
            passing_validation = module.load_jsonl(passing_validation_path)[0]

            stale_declared_manifest, _stale_declared_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "stale-declared",
                    runtime_import_presets="hf-runtime",
                    declared_runtime_import_presets=["hf-runtime"],
                    declared_runtime_import_preset_modules=(
                        "hf-runtime=transformers|torch"
                    ),
                )
            )
            stale_declared_output = io.StringIO()
            with self.assertRaisesRegex(
                RuntimeError,
                "profile smoke manifest trace validation gate failed",
            ):
                with contextlib.redirect_stdout(stale_declared_output):
                    module.validate_profile_smoke_manifest_file(
                        stale_declared_manifest,
                        args=produced_validation_args(),
                    )

            stale_trace_manifest, _stale_trace_validation_path = (
                self.write_profile_smoke_manifest_with_transformers_trace_compare(
                    module,
                    out_dir / "stale-trace",
                    runtime_import_presets="hf-runtime",
                    declared_runtime_import_presets=["hf-runtime"],
                    trace_runtime_import_preset_modules=(
                        "hf-runtime=transformers|torch"
                    ),
                )
            )
            stale_trace_output = io.StringIO()
            with self.assertRaisesRegex(
                RuntimeError,
                "profile smoke manifest trace validation gate failed",
            ):
                with contextlib.redirect_stdout(stale_trace_output):
                    module.validate_profile_smoke_manifest_file(
                        stale_trace_manifest,
                        args=produced_validation_args(),
                    )

        self.assertIn("profile_smoke_manifest_gate", passing_output.getvalue())
        self.assertIn("passed=True", passing_output.getvalue())
        self.assertTrue(
            passing_validation[
                "declared_transformers_trace_runtime_import_preset_modules_match"
            ]
        )
        self.assertEqual(
            passing_validation[
                "declared_transformers_trace_runtime_import_preset_modules_expected"
            ],
            "hf-runtime=transformers|torch|tokenizers",
        )
        self.assertEqual(
            passing_validation[
                "transformers_trace_declared_runtime_import_preset_modules"
            ],
            "hf-runtime=transformers|torch|tokenizers",
        )
        self.assertTrue(
            passing_validation[
                "transformers_trace_declared_runtime_import_preset_modules_match"
            ]
        )
        stale_declared_text = stale_declared_output.getvalue()
        self.assertIn(
            "declared_transformers_trace_runtime_import_preset_modules_mismatch",
            stale_declared_text,
        )
        self.assertIn("passed=False", stale_declared_text)
        stale_trace_text = stale_trace_output.getvalue()
        self.assertIn(
            "transformers_trace_declared_runtime_import_preset_modules_mismatch",
            stale_trace_text,
        )
        self.assertIn("passed=False", stale_trace_text)

    def test_byte_lm_profile_smoke_accepts_produced_manifest_trace_gates(self):
        module = load_example("byte_lm_profile_smoke")
        old_argv = sys.argv
        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--out-dir",
            "/tmp/profile-smoke-real-hf",
            "--hf-state-dict",
            "/models/llama",
            "--transformers-trace",
            "--transformers-trace-runtime-import-preset",
            "torch-transformers",
            "--transformers-trace-runtime-import",
            "torch",
            "--require-transformers-trace-runtime-imports",
            "--require-transformers-trace-runtime-import",
            "torch",
            "--require-transformers-trace-runtime-import-preset",
            "torch-transformers",
            "--validate-produced-manifest",
            "--require-manifest-transformers-trace",
            "--require-manifest-transformers-trace-coimport",
            "--require-manifest-transformers-trace-runtime-imports",
            "--require-manifest-transformers-trace-runtime-import",
            "torch",
            "--require-manifest-transformers-trace-runtime-import-preset",
            "torch-transformers",
        ]
        try:
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.validate_produced_manifest)
        self.assertEqual(
            args.transformers_trace_runtime_import_presets,
            ["torch-transformers"],
        )
        self.assertEqual(args.transformers_trace_runtime_imports, ["torch"])
        self.assertTrue(args.require_transformers_trace_runtime_imports)
        self.assertEqual(args.require_transformers_trace_runtime_import, ["torch"])
        self.assertEqual(
            args.require_transformers_trace_runtime_import_preset,
            ["torch-transformers"],
        )
        self.assertTrue(args.require_manifest_transformers_trace)
        self.assertTrue(args.require_manifest_transformers_trace_coimport)
        self.assertTrue(args.require_manifest_transformers_trace_runtime_imports)
        self.assertEqual(
            args.require_manifest_transformers_trace_runtime_import,
            ["torch"],
        )
        self.assertEqual(
            args.require_manifest_transformers_trace_runtime_import_preset,
            ["torch-transformers"],
        )
        self.assertEqual(
            module.transformers_trace_runtime_import_preset_modules(["hf-runtime"]),
            ["hf-runtime=transformers|torch|tokenizers"],
        )
        self.assertIsNone(args.manifest_validation_jsonl)
        self.assertEqual(
            module.produced_manifest_validation_jsonl(
                args,
                Path("/tmp/profile-smoke-real-hf/profile-smoke-manifest.jsonl"),
            ),
            Path("/tmp/profile-smoke-real-hf/profile-smoke-manifest-validation.jsonl"),
        )

    def test_byte_lm_profile_smoke_runtime_contract_preset_expands_execution_gates(self):
        module = load_example("byte_lm_profile_smoke")
        old_argv = sys.argv
        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--out-dir",
            "/tmp/profile-smoke-real-hf",
            "--hf-state-dict",
            "/models/llama",
            "--runtime-contract-preset",
            "hf-runtime",
        ]
        try:
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.transformers_audit)
        self.assertTrue(args.transformers_trace)
        self.assertTrue(args.validate_produced_manifest)
        self.assertEqual(
            args.checkpoint_transformers_runtime_import_presets,
            ["hf-runtime"],
        )
        self.assertTrue(args.require_checkpoint_transformers_runtime_imports)
        self.assertEqual(
            args.require_checkpoint_transformers_runtime_import_preset,
            ["hf-runtime"],
        )
        self.assertEqual(
            args.transformers_trace_runtime_import_presets,
            ["hf-runtime"],
        )
        self.assertTrue(args.require_transformers_trace_runtime_imports)
        self.assertEqual(
            args.require_transformers_trace_runtime_import_preset,
            ["hf-runtime"],
        )
        self.assertTrue(args.require_manifest_checkpoint_transformers_runtime_imports)
        self.assertEqual(
            args.require_manifest_checkpoint_transformers_runtime_import_preset,
            ["hf-runtime"],
        )
        self.assertTrue(args.require_manifest_transformers_trace)
        self.assertTrue(args.require_manifest_transformers_trace_coimport)
        self.assertTrue(args.require_manifest_transformers_trace_runtime_imports)
        self.assertEqual(
            args.require_manifest_transformers_trace_runtime_import_preset,
            ["hf-runtime"],
        )
        checkpoint_args = module.checkpoint_transformers_args(args)
        self.assertIn("--transformers-audit", checkpoint_args)
        self.assertIn("--transformers-runtime-import-preset", checkpoint_args)
        self.assertIn("hf-runtime", checkpoint_args)
        self.assertIn("--require-transformers-runtime-imports", checkpoint_args)
        self.assertIn("--require-transformers-runtime-import-preset", checkpoint_args)
        trace_args = module.transformers_trace_args(
            args,
            Path("/models/llama"),
            Path("/tmp/trace.jsonl"),
            None,
        )
        self.assertIn("--runtime-import-preset", trace_args)
        self.assertIn("hf-runtime", trace_args)
        self.assertIn("--require-runtime-imports", trace_args)
        self.assertIn("--require-runtime-import-preset", trace_args)

        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--out-dir",
            "/tmp/profile-smoke-real-hf",
            "--hf-state-dict",
            "/models/llama",
            "--runtime-contract-preset",
            "hf-runtime",
            "--dry-run",
        ]
        try:
            dry_run_args = module.parse_args()
        finally:
            sys.argv = old_argv
        self.assertTrue(dry_run_args.transformers_audit)
        self.assertTrue(dry_run_args.transformers_trace)
        self.assertTrue(dry_run_args.require_checkpoint_transformers_runtime_imports)
        self.assertTrue(dry_run_args.require_transformers_trace_runtime_imports)
        self.assertFalse(dry_run_args.validate_produced_manifest)
        self.assertFalse(
            dry_run_args.require_manifest_checkpoint_transformers_runtime_imports
        )
        self.assertFalse(dry_run_args.require_manifest_transformers_trace)

        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--validate-manifest-jsonl",
            "/tmp/profile-smoke-real-hf/profile-smoke-manifest.jsonl",
            "--runtime-contract-preset",
            "hf-runtime",
        ]
        try:
            validation_args = module.parse_args()
        finally:
            sys.argv = old_argv
        self.assertFalse(validation_args.transformers_audit)
        self.assertFalse(validation_args.transformers_trace)
        self.assertFalse(validation_args.validate_produced_manifest)
        self.assertTrue(
            validation_args.require_manifest_checkpoint_transformers_runtime_imports
        )
        self.assertEqual(
            validation_args.require_manifest_checkpoint_transformers_runtime_import_preset,
            ["hf-runtime"],
        )
        self.assertTrue(validation_args.require_manifest_transformers_trace)
        self.assertTrue(validation_args.require_manifest_transformers_trace_coimport)
        self.assertTrue(
            validation_args.require_manifest_transformers_trace_runtime_imports
        )
        self.assertEqual(
            validation_args.require_manifest_transformers_trace_runtime_import_preset,
            ["hf-runtime"],
        )

        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--continue-manifest-jsonl",
            "/tmp/profile-smoke-real-hf/profile-smoke-manifest.jsonl",
            "--runtime-contract-preset",
            "hf-runtime",
        ]
        try:
            continue_args = module.parse_args()
        finally:
            sys.argv = old_argv
        self.assertFalse(continue_args.transformers_audit)
        self.assertFalse(continue_args.transformers_trace)
        self.assertTrue(continue_args.validate_produced_manifest)
        self.assertTrue(
            continue_args.require_manifest_checkpoint_transformers_runtime_imports
        )
        self.assertEqual(
            continue_args.require_manifest_checkpoint_transformers_runtime_import_preset,
            ["hf-runtime"],
        )
        self.assertTrue(continue_args.require_manifest_transformers_trace)
        self.assertTrue(continue_args.require_manifest_transformers_trace_coimport)
        self.assertTrue(continue_args.require_manifest_transformers_trace_runtime_imports)
        self.assertEqual(
            continue_args.require_manifest_transformers_trace_runtime_import_preset,
            ["hf-runtime"],
        )

        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--continue-manifest-jsonl",
            "/tmp/profile-smoke-real-hf/profile-smoke-manifest.jsonl",
            "--runtime-contract-preset",
            "hf-runtime",
            "--dry-run",
        ]
        try:
            continue_dry_run_args = module.parse_args()
        finally:
            sys.argv = old_argv
        self.assertFalse(continue_dry_run_args.transformers_audit)
        self.assertFalse(continue_dry_run_args.transformers_trace)
        self.assertFalse(continue_dry_run_args.validate_produced_manifest)
        self.assertFalse(
            continue_dry_run_args.require_manifest_checkpoint_transformers_runtime_imports
        )
        self.assertFalse(continue_dry_run_args.require_manifest_transformers_trace)

    def test_transformers_trace_runtime_import_presets_are_shared(self):
        profile_module = load_example("byte_lm_profile_smoke")
        trace_module = load_example("byte_lm_transformers_trace")
        from spiraltorch.runtime_imports import (
            runtime_import_preset_module_map,
            runtime_import_preset_module_rows,
            runtime_import_preset_missing_modules_label,
            runtime_import_preset_modules_label,
            runtime_import_preset_status_rows,
            runtime_import_coimport_status,
            runtime_import_names_from_source,
            runtime_import_names_from_args,
            runtime_import_presets_from_args,
            runtime_import_presets_from_source,
            runtime_import_required_gate_fields,
            runtime_imports_from_args,
            runtime_imports_from_source,
            runtime_import_requirement_failures,
            required_runtime_import_presets_from_args,
            required_runtime_import_presets_from_source,
            required_runtime_imports_from_args,
            required_runtime_imports_from_source,
        )

        self.assertEqual(
            profile_module.TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS,
            trace_module.RUNTIME_IMPORT_PRESETS,
        )
        self.assertEqual(
            profile_module.transformers_trace_runtime_import_preset_modules(
                ["hf-runtime"]
            ),
            ["hf-runtime=transformers|torch|tokenizers"],
        )
        self.assertEqual(
            trace_module.runtime_import_presets_from_args(
                argparse.Namespace(
                    runtime_import_presets=[],
                    required_runtime_import_presets=["hf-runtime"],
                )
            ),
            ["hf-runtime"],
        )
        args = argparse.Namespace(
            runtime_import_presets=["torch-transformers"],
            runtime_imports=[" tokenizers "],
            required_runtime_imports=["torch"],
            required_runtime_import_presets=["hf-runtime"],
        )
        self.assertEqual(
            runtime_import_presets_from_args(args),
            ["torch-transformers", "hf-runtime"],
        )
        self.assertEqual(required_runtime_imports_from_args(args), ["torch"])
        self.assertEqual(
            required_runtime_import_presets_from_args(args),
            ["hf-runtime"],
        )
        self.assertEqual(
            runtime_import_names_from_args(
                args,
                preset_modules=trace_module.RUNTIME_IMPORT_PRESETS,
            ),
            ["transformers", "torch", "tokenizers"],
        )
        self.assertEqual(runtime_imports_from_args(args), ["tokenizers"])
        prefixed_args = argparse.Namespace(
            transformers_trace_runtime_import_presets=[" torch-transformers "],
            transformers_trace_runtime_imports=[" tokenizers "],
            require_transformers_trace_runtime_import=["torch"],
            require_transformers_trace_runtime_import_preset=[" hf-runtime "],
        )
        self.assertEqual(
            runtime_import_presets_from_source(
                prefixed_args,
                runtime_import_presets_key="transformers_trace_runtime_import_presets",
                required_runtime_import_presets_key=(
                    "require_transformers_trace_runtime_import_preset"
                ),
            ),
            ["torch-transformers", "hf-runtime"],
        )
        self.assertEqual(
            runtime_imports_from_source(
                prefixed_args,
                runtime_imports_key="transformers_trace_runtime_imports",
            ),
            ["tokenizers"],
        )
        self.assertEqual(
            required_runtime_imports_from_source(
                prefixed_args,
                required_runtime_imports_key=(
                    "require_transformers_trace_runtime_import"
                ),
            ),
            ["torch"],
        )
        self.assertEqual(
            required_runtime_import_presets_from_source(
                prefixed_args,
                required_runtime_import_presets_key=(
                    "require_transformers_trace_runtime_import_preset"
                ),
            ),
            ["hf-runtime"],
        )
        self.assertEqual(
            runtime_import_names_from_source(
                prefixed_args,
                preset_modules=trace_module.RUNTIME_IMPORT_PRESETS,
                runtime_imports_key="transformers_trace_runtime_imports",
                runtime_import_presets_key="transformers_trace_runtime_import_presets",
                required_runtime_imports_key=(
                    "require_transformers_trace_runtime_import"
                ),
                required_runtime_import_presets_key=(
                    "require_transformers_trace_runtime_import_preset"
                ),
            ),
            ["transformers", "torch", "tokenizers"],
        )
        self.assertEqual(
            profile_module.transformers_trace_runtime_import_presets(prefixed_args),
            ["torch-transformers", "hf-runtime"],
        )
        self.assertEqual(
            profile_module.transformers_trace_runtime_import_names(prefixed_args),
            ["transformers", "torch", "tokenizers"],
        )
        required_only_policy = profile_module.trace_policy_fields(
            {
                "transformers_trace": True,
                "transformers_trace_runtime_import_presets": [],
                "transformers_trace_runtime_imports": [],
                "require_transformers_trace_runtime_imports": True,
                "require_transformers_trace_runtime_import": [" torch "],
                "require_transformers_trace_runtime_import_preset": ["hf-runtime"],
            }
        )
        self.assertEqual(
            required_only_policy["transformers_trace_runtime_import_presets"],
            ["hf-runtime"],
        )
        self.assertEqual(
            required_only_policy[
                "declared_transformers_trace_runtime_import_preset_modules"
            ],
            ["hf-runtime=transformers|torch|tokenizers"],
        )
        self.assertEqual(
            required_only_policy["require_transformers_trace_runtime_import"],
            ["torch"],
        )
        self.assertEqual(
            required_only_policy[
                "require_transformers_trace_runtime_import_preset"
            ],
            ["hf-runtime"],
        )
        status_rows = runtime_import_preset_status_rows(
            ["hf-runtime"],
            [
                {"module": "transformers", "imported": True},
                {"module": "torch", "imported": True},
                {"module": "tokenizers", "imported": False},
            ],
        )
        self.assertEqual(
            runtime_import_preset_modules_label(status_rows),
            "hf-runtime=transformers|torch|tokenizers",
        )
        self.assertEqual(
            runtime_import_preset_missing_modules_label(status_rows),
            "hf-runtime=tokenizers",
        )
        self.assertEqual(runtime_import_coimport_status([]), "not_requested")
        self.assertEqual(
            runtime_import_coimport_status(
                [
                    {"module": "transformers", "imported": True},
                    {"module": "torch", "imported": True},
                ]
            ),
            "ok",
        )
        self.assertEqual(
            runtime_import_coimport_status(
                [
                    {"module": "transformers", "imported": True},
                    {"module": "torch", "imported": True},
                    {"module": "tokenizers", "imported": False},
                ]
            ),
            "missing",
        )
        self.assertEqual(
            runtime_import_preset_module_map(
                "torch-transformers=transformers|torch,"
                "hf-runtime=transformers|torch|tokenizers"
            ),
            {
                "torch-transformers": "torch-transformers=transformers|torch",
                "hf-runtime": "hf-runtime=transformers|torch|tokenizers",
            },
        )
        self.assertEqual(
            runtime_import_preset_module_rows(
                "torch-transformers=transformers|torch,"
                "hf-runtime=transformers|torch|tokenizers",
                ["hf-runtime"],
            ),
            ["hf-runtime=transformers|torch|tokenizers"],
        )
        self.assertFalse(status_rows[0]["passed"])
        gate_fields = runtime_import_required_gate_fields(
            ["tokenizers"],
            ["hf-runtime"],
            probes=[
                {"module": "transformers", "imported": True},
                {"module": "torch", "imported": True},
                {"module": "tokenizers", "imported": False},
            ],
            preset_status=status_rows,
        )
        self.assertEqual(gate_fields["required_runtime_imports"], "tokenizers")
        self.assertEqual(
            gate_fields["required_runtime_imports_missing"],
            "tokenizers",
        )
        self.assertEqual(
            gate_fields["required_runtime_import_presets_unsatisfied"],
            "hf-runtime",
        )
        self.assertEqual(
            runtime_import_requirement_failures(gate_fields),
            [
                "runtime_import_missing:tokenizers",
                "runtime_import_preset_unsatisfied:hf-runtime",
            ],
        )
        prefixed_gate_fields = runtime_import_required_gate_fields(
            ["tokenizers"],
            ["hf-runtime"],
            imported_modules=["transformers", "torch"],
            observed_presets=["hf-runtime"],
            satisfied_presets=[],
            failed_presets=["hf-runtime"],
            field_prefix="transformers_trace_",
            include_failed_presets=True,
        )
        self.assertEqual(
            prefixed_gate_fields[
                "transformers_trace_required_runtime_import_presets_failed"
            ],
            "hf-runtime",
        )
        self.assertEqual(
            runtime_import_requirement_failures(
                prefixed_gate_fields,
                field_prefix="transformers_trace_",
                failure_prefix="transformers_trace_runtime_import",
            ),
            [
                "transformers_trace_runtime_import_missing:tokenizers",
                "transformers_trace_runtime_import_preset_unsatisfied:hf-runtime",
            ],
        )

    def test_byte_lm_profile_smoke_rejects_produced_manifest_validation_dry_run(self):
        module = load_example("byte_lm_profile_smoke")
        old_argv = sys.argv
        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--out-dir",
            "/tmp/profile-smoke-real-hf",
            "--hf-state-dict",
            "/models/llama",
            "--validate-produced-manifest",
            "--dry-run",
        ]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args()
        finally:
            sys.argv = old_argv

    def test_byte_lm_profile_smoke_builds_checkpoint_policy_args(self):
        module = load_example("byte_lm_profile_smoke")
        args = argparse.Namespace(
            key_preset="auto",
            include_extra_keys=["model.layers.0.input_layernorm.weight"],
            no_synthesize_missing_biases=True,
            allow_overlap_resize=True,
            checkpoint_projection="none",
            checkpoint_projection_preset="healthy",
            checkpoint_projection_strength=None,
            checkpoint_projection_curvature=-0.04,
            checkpoint_projection_frequency=0.65,
            checkpoint_source_gain=2.0,
        )
        flags = module.checkpoint_policy_args(args)
        self.assertEqual(
            flags,
            [
                "--key-preset",
                "auto",
                "--include-extra-key",
                "model.layers.0.input_layernorm.weight",
                "--no-synthesize-missing-biases",
                "--allow-overlap-resize",
                "--checkpoint-projection-preset",
                "healthy",
                "--checkpoint-projection-curvature",
                "-0.04",
                "--checkpoint-projection-frequency",
                "0.65",
                "--checkpoint-source-gain",
                "2",
            ],
        )
        self.assertEqual(
            module.checkpoint_shape_args(),
            ["--vocab", "256", "--hidden", "24", "--target-classes", "256"],
        )
        args.transformers_audit = True
        args.transformers_model_path = Path("/models/llama")
        args.transformers_revision = "main"
        args.allow_transformers_remote = True
        args.transformers_trust_remote_code = True
        args.skip_transformers_tokenizer = True
        args.transformers_load_model = True
        args.require_transformers_audit = True
        args.checkpoint_transformers_runtime_import_presets = ["transformers"]
        args.checkpoint_transformers_runtime_imports = ["math"]
        args.require_checkpoint_transformers_runtime_imports = True
        args.require_checkpoint_transformers_runtime_import = ["math"]
        args.require_checkpoint_transformers_runtime_import_preset = ["transformers"]
        self.assertEqual(
            module.checkpoint_transformers_args(args),
            [
                "--transformers-audit",
                "--transformers-model-path",
                Path("/models/llama"),
                "--transformers-revision",
                "main",
                "--allow-transformers-remote",
                "--transformers-trust-remote-code",
                "--skip-transformers-tokenizer",
                "--transformers-load-model",
                "--require-transformers-audit",
                "--transformers-runtime-import-preset",
                "transformers",
                "--transformers-runtime-import",
                "math",
                "--require-transformers-runtime-imports",
                "--require-transformers-runtime-import",
                "math",
                "--require-transformers-runtime-import-preset",
                "transformers",
            ],
        )
        args.transformers_trace = True
        args.transformers_trace_prompts = ["spiral"]
        args.transformers_trace_prompt_file = Path("/tmp/prompts.txt")
        args.transformers_trace_top_k = 3
        args.transformers_trace_zspace_project = True
        args.transformers_trace_zspace_source = "top_logits"
        args.transformers_trace_runtime_import_presets = ["torch-transformers"]
        args.transformers_trace_runtime_imports = ["torch", "tokenizers"]
        args.require_transformers_trace_runtime_imports = True
        args.require_transformers_trace_runtime_import = ["torch"]
        args.require_transformers_trace_runtime_import_preset = ["torch-transformers"]
        args.compare_transformers_trace_jsonl = Path("/tmp/baseline-trace.jsonl")
        args.require_transformers_trace_match = True
        args.require_transformers_trace_runtime_metadata_match = True
        args.require_transformers_trace_top_token_match = True
        args.transformers_trace_max_top_logit_regression = 0.0
        args.transformers_trace_max_top_probability_regression = 0.1
        args.transformers_trace_max_logit_l2_change = 0.2
        args.transformers_trace_max_hidden_state_l2_change = 0.3
        args.transformers_trace_require_zspace_status = "ok"
        self.assertEqual(
            module.transformers_trace_args(
                args,
                Path("/models/llama"),
                Path("/tmp/current-trace.jsonl"),
                Path("/tmp/trace-compare.jsonl"),
            ),
            [
                "--model-path",
                Path("/models/llama"),
                "--jsonl",
                Path("/tmp/current-trace.jsonl"),
                "--top-k",
                "3",
                "--prompt",
                "spiral",
                "--prompt-file",
                Path("/tmp/prompts.txt"),
                "--revision",
                "main",
                "--allow-remote",
                "--trust-remote-code",
                "--zspace-project",
                "--zspace-source",
                "top_logits",
                "--runtime-import-preset",
                "torch-transformers",
                "--runtime-import",
                "torch",
                "--runtime-import",
                "tokenizers",
                "--require-runtime-imports",
                "--require-runtime-import",
                "torch",
                "--require-runtime-import-preset",
                "torch-transformers",
                "--compare-jsonl",
                Path("/tmp/baseline-trace.jsonl"),
                "--compare-output-jsonl",
                Path("/tmp/trace-compare.jsonl"),
                "--require-trace-match",
                "--require-runtime-metadata-match",
                "--require-top-token-match",
                "--max-top-logit-regression",
                "0",
                "--max-top-probability-regression",
                "0.1",
                "--max-logit-l2-change",
                "0.2",
                "--max-hidden-state-l2-change",
                "0.3",
                "--require-zspace-status",
                "ok",
            ],
        )
        args.transformers_revision = None
        args.allow_transformers_remote = False
        args.transformers_trust_remote_code = False
        args.transformers_trace_prompts = []
        args.transformers_trace_prompt_file = None
        args.transformers_trace_zspace_project = False
        args.transformers_trace_runtime_import_presets = []
        args.transformers_trace_runtime_imports = []
        args.require_transformers_trace_runtime_import = []
        args.require_transformers_trace_runtime_import_preset = ["hf-runtime"]
        args.compare_transformers_trace_jsonl = None
        args.require_transformers_trace_match = False
        args.require_transformers_trace_runtime_metadata_match = False
        args.require_transformers_trace_top_token_match = False
        args.transformers_trace_max_top_logit_regression = None
        args.transformers_trace_max_top_probability_regression = None
        args.transformers_trace_max_logit_l2_change = None
        args.transformers_trace_max_hidden_state_l2_change = None
        args.transformers_trace_require_zspace_status = None
        required_only_flags = module.transformers_trace_args(
            args,
            Path("/models/llama"),
            Path("/tmp/current-trace.jsonl"),
            None,
        )
        runtime_preset_index = required_only_flags.index("--runtime-import-preset")
        require_preset_index = required_only_flags.index(
            "--require-runtime-import-preset"
        )
        self.assertEqual(required_only_flags[runtime_preset_index + 1], "hf-runtime")
        self.assertEqual(required_only_flags[require_preset_index + 1], "hf-runtime")

    def test_byte_lm_profile_smoke_dry_run_external_checkpoint_preflights(self):
        module = load_example("byte_lm_profile_smoke")
        old_argv = sys.argv
        sys.argv = [
            "byte_lm_profile_smoke.py",
            "--out-dir",
            "/tmp/profile-smoke-real-hf",
            "--hf-state-dict",
            "/models/llama",
            "--source-label",
            "llama-3.2-3b",
            "--key-preset",
            "auto",
            "--allow-overlap-resize",
            "--checkpoint-projection-preset",
            "healthy",
            "--checkpoint-source-gain",
            "2.0",
            "--transformers-audit",
            "--transformers-model-path",
            "/models/llama",
            "--transformers-revision",
            "main",
            "--transformers-trust-remote-code",
            "--skip-transformers-tokenizer",
            "--require-transformers-audit",
            "--checkpoint-transformers-runtime-import-preset",
            "transformers",
            "--checkpoint-transformers-runtime-import",
            "math",
            "--require-checkpoint-transformers-runtime-imports",
            "--require-checkpoint-transformers-runtime-import",
            "math",
            "--require-checkpoint-transformers-runtime-import-preset",
            "transformers",
            "--transformers-trace",
            "--transformers-trace-prompt",
            "spiral",
            "--transformers-trace-top-k",
            "3",
            "--transformers-trace-runtime-import-preset",
            "torch-transformers",
            "--transformers-trace-runtime-import",
            "torch",
            "--require-transformers-trace-runtime-imports",
            "--require-transformers-trace-runtime-import",
            "torch",
            "--require-transformers-trace-runtime-import-preset",
            "torch-transformers",
            "--compare-transformers-trace-jsonl",
            "/tmp/profile-smoke-real-hf/transformers-trace-baseline.jsonl",
            "--require-transformers-trace-match",
            "--require-transformers-trace-runtime-metadata-match",
            "--require-transformers-trace-top-token-match",
            "--transformers-trace-max-top-logit-regression",
            "0.0",
            "--transformers-trace-max-top-probability-regression",
            "0.1",
            "--transformers-trace-require-zspace-status",
            "not_requested",
            "--compare-checkpoint-preflight-jsonl",
            "/tmp/profile-smoke-real-hf/checkpoint-preflight-baseline.jsonl",
            "--require-checkpoint-preflight-match",
            "--shape-audit-require-detected-key-preset",
            "llama",
            "--skip-promoted-follow-up",
            "--dry-run",
        ]
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                module.main()
        finally:
            sys.argv = old_argv

        text = output.getvalue()
        self.assertNotIn("write_byte_lm_hf_state_dict.py", text)
        self.assertIn("checkpoint_preflight.py", text)
        self.assertIn("--shape-only", text)
        self.assertIn("--require-shape-materializable", text)
        self.assertIn("--require-detected-key-preset llama", text)
        self.assertIn("byte_lm_mlp_lora_sweep.py", text)
        self.assertIn("--hf-state-dict /models/llama", text)
        self.assertIn("--key-preset auto", text)
        self.assertIn("--allow-overlap-resize", text)
        self.assertIn("--checkpoint-projection-preset healthy", text)
        self.assertIn("--checkpoint-source-gain 2", text)
        self.assertIn("--transformers-audit", text)
        self.assertIn("--transformers-model-path /models/llama", text)
        self.assertIn("--transformers-revision main", text)
        self.assertIn("--transformers-trust-remote-code", text)
        self.assertIn("--skip-transformers-tokenizer", text)
        self.assertIn("--require-transformers-audit", text)
        self.assertIn("--transformers-runtime-import-preset transformers", text)
        self.assertIn("--transformers-runtime-import math", text)
        self.assertIn("--require-transformers-runtime-imports", text)
        self.assertIn("--require-transformers-runtime-import math", text)
        self.assertIn("--require-transformers-runtime-import-preset transformers", text)
        self.assertIn("byte_lm_transformers_trace.py", text)
        self.assertIn("--model-path /models/llama", text)
        self.assertIn("--jsonl /tmp/profile-smoke-real-hf/transformers-trace.jsonl", text)
        self.assertIn("--top-k 3", text)
        self.assertIn("--prompt spiral", text)
        self.assertIn("--runtime-import-preset torch-transformers", text)
        self.assertIn("--runtime-import torch", text)
        self.assertIn("--require-runtime-imports", text)
        self.assertIn("--require-runtime-import torch", text)
        self.assertIn("--require-runtime-import-preset torch-transformers", text)
        self.assertIn(
            "--compare-jsonl /tmp/profile-smoke-real-hf/transformers-trace-baseline.jsonl",
            text,
        )
        self.assertIn(
            "--compare-output-jsonl /tmp/profile-smoke-real-hf/transformers-trace-compare.jsonl",
            text,
        )
        self.assertIn("--require-trace-match", text)
        self.assertIn("--require-runtime-metadata-match", text)
        self.assertIn("--require-top-token-match", text)
        self.assertIn("--max-top-logit-regression 0", text)
        self.assertIn("--max-top-probability-regression 0.1", text)
        self.assertIn("--require-zspace-status not_requested", text)
        self.assertIn(
            "--compare-jsonl /tmp/profile-smoke-real-hf/checkpoint-preflight-baseline.jsonl",
            text,
        )
        self.assertIn("--require-preflight-match", text)
        self.assertIn("checkpoint_source=external", text)
        self.assertIn("profile_smoke_manifest_jsonl=", text)
        self.assertIn("checkpoint_shape_audit_jsonl=", text)
        self.assertIn("checkpoint_preflight_jsonl=", text)
        self.assertIn(
            "compare_checkpoint_preflight_jsonl="
            "/tmp/profile-smoke-real-hf/checkpoint-preflight-baseline.jsonl",
            text,
        )
        self.assertIn(
            "transformers_trace_jsonl=/tmp/profile-smoke-real-hf/transformers-trace.jsonl",
            text,
        )
        self.assertIn(
            "compare_transformers_trace_jsonl="
            "/tmp/profile-smoke-real-hf/transformers-trace-baseline.jsonl",
            text,
        )
        self.assertIn(
            "transformers_trace_compare_jsonl="
            "/tmp/profile-smoke-real-hf/transformers-trace-compare.jsonl",
            text,
        )
        self.assertIn("promoted_follow_up=skipped", text)

    def test_mlp_lora_profile_runner_rejects_missing_source_path(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        rows = [
            {
                "row_type": "checkpoint_source_profile",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "base_config": "r12_a64_lr4",
                "case_labels": "adapter_ja",
                "checkpoint_source_flag_fragment": [
                    "--checkpoint-source-label",
                    "gemma-4-e4b-it",
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "missing source path"):
            module.command_rows_for_profiles(rows, source_paths={})

    def test_mlp_lora_profile_runner_writes_run_events_on_success_and_failure(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")

        def command_row(returncode):
            code = f"import sys; sys.exit({returncode})"
            return {
                "row_type": "checkpoint_source_profile_command",
                "source_profile": "strong_effect",
                "selected_source": "fixture",
                "selected_config": "r6_a32_lr3",
                "run_config_key": "strong_effect::r6_a32_lr3",
                "jsonl": "/tmp/profile.jsonl",
                "aggregate_jsonl": "/tmp/profile-aggregate.jsonl",
                "command": [sys.executable, "-c", code],
                "shell": f"{sys.executable} -c {code!r}",
                "promotion_rank": 1,
                "promotion_metric": "target_retention_ratio",
                "promotion_ready": True,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = Path(tmpdir) / "events.jsonl"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                events = module.run_command_rows([command_row(0)], events_jsonl=events_path)
            persisted = module.load_jsonl(events_path)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            persisted[0]["row_type"],
            "checkpoint_source_profile_command_event",
        )
        self.assertEqual(persisted[0]["status"], "passed")
        self.assertEqual(persisted[0]["returncode"], 0)
        self.assertEqual(persisted[0]["promotion_rank"], 1)
        self.assertIn(
            "profile_command_event profile=strong_effect status=passed",
            output.getvalue(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = Path(tmpdir) / "events-fail.jsonl"
            with self.assertRaises(subprocess.CalledProcessError):
                module.run_command_rows([command_row(7)], events_jsonl=events_path)
            failed = module.load_jsonl(events_path)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["status"], "failed")
        self.assertEqual(failed[0]["returncode"], 7)

    def test_mlp_lora_profile_runner_summarizes_profile_outputs(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        with tempfile.TemporaryDirectory() as tmpdir:
            aggregate_path = Path(tmpdir) / "profile-aggregate.jsonl"
            module.write_jsonl(
                aggregate_path,
                [
                    {
                        "row_type": "config_aggregate",
                        "config": "r12_a64_lr4",
                        "checkpoint_source_label": "gemma-4-e4b-it",
                        "checkpoint_source_gain": 4.0,
                        "adapter_weight_decay": 0.01,
                        "max_grad_norm_variant": "gn1p5",
                        "max_grad_norm": 1.5,
                        "gradient_accumulation_steps_variant": "accum4",
                        "gradient_accumulation_steps": 4,
                        "training_policy_key": "policy:aggregate-ft6",
                        "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
                        "ft_epochs": 6,
                        "target_min_loss_delta_policy": 0.001,
                        "early_stopping_patience": 3,
                        "early_stopping_min_delta": 0.0,
                        "lr_decay_patience": 2,
                        "lr_decay_factor": 0.8,
                        "lr_decay_min_delta": 0.0,
                        "cases": 2,
                        "case_labels": "adapter_ja,route_cats",
                        "accepted_rate": 1.0,
                        "movement_ok_rate": 1.0,
                        "guard_epoch_counts_available_cases": 2,
                        "guard_epoch_counts_available_all": True,
                        "guard_accepted_epochs_total": 10.0,
                        "guard_accepted_epochs_mean": 5.0,
                        "guard_accepted_epochs_max": 6.0,
                        "guard_retention_rejected_epochs_total": 1.0,
                        "guard_retention_rejected_epochs_mean": 0.5,
                        "guard_retention_rejected_epochs_max": 1.0,
                        "guard_target_stale_epochs_total": 3.0,
                        "guard_target_stale_epochs_mean": 1.5,
                        "guard_target_stale_epochs_max": 2.0,
                        "guard_acceptance_rate_mean": 0.75,
                        "guard_acceptance_rate_min": 0.5,
                        "guard_retention_rejected_rate_mean": 0.10,
                        "guard_retention_rejected_rate_max": 0.20,
                        "guard_target_stale_rate_mean": 0.30,
                        "guard_target_stale_rate_max": 0.40,
                        "target_loss_delta_mean": 1.2,
                        "retention_loss_delta_mean": 0.4,
                    }
                ],
            )
            rows = module.profile_run_summary_rows(
                [
                    {
                        "source_profile": "strong_effect",
                        "selected_source": "gemma-4-e4b-it",
                        "selected_config": "r12_a64_lr4::gain_g4",
                        "run_config_key": "r12_a64_lr4::gain_g4",
                        "winner_metric": "target_loss_delta_mean",
                        "winner_value": 1.2,
                        "promotion_run_key": "strong_effect::r12_a64_lr4::gain_g4",
                        "promotion_rank": 1,
                        "promotion_metric": "target_retention_ratio",
                        "promotion_value": 2.8,
                        "promotion_ready": True,
                        "promotion_ready_top_k": 2,
                        "promotion_ready_within": 0.25,
                        "promotion_ready_floor_passed": True,
                        "promotion_ready_floor_failures": [],
                        "promotion_ready_min_target_retention_ratio": 2.5,
                        "promotion_ready_min_accepted_rate": 1.0,
                        "promotion_ready_min_movement_ok_rate": 1.0,
                        "promotion_ready_require_guard_counts_available": True,
                        "promotion_ready_min_guard_acceptance_rate_mean": 0.70,
                        "promotion_ready_max_guard_retention_rejected_epochs_mean": 0.0,
                        "promotion_ready_max_guard_target_stale_epochs_mean": 2.0,
                        "promotion_ready_max_guard_retention_rejected_rate_mean": 0.10,
                        "promotion_ready_max_guard_target_stale_rate_mean": 0.50,
                        "jsonl": str(Path(tmpdir) / "profile.jsonl"),
                        "aggregate_jsonl": str(aggregate_path),
                        "shell": "python sweep.py",
                    }
                ]
            )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["row_type"], "checkpoint_source_profile_run")
        self.assertEqual(row["source_profile"], "strong_effect")
        self.assertEqual(row["selected_source"], "gemma-4-e4b-it")
        self.assertEqual(row["run_config_key"], "r12_a64_lr4::gain_g4")
        self.assertEqual(row["adapter_weight_decay"], 0.01)
        self.assertEqual(row["max_grad_norm_variant"], "gn1p5")
        self.assertEqual(row["max_grad_norm"], 1.5)
        self.assertEqual(row["gradient_accumulation_steps_variant"], "accum4")
        self.assertEqual(row["gradient_accumulation_steps"], 4)
        self.assertEqual(row["training_policy_key"], "policy:aggregate-ft6")
        self.assertEqual(row["ft_control_variant"], "ep6::tmin0p001::pat3::ldp2::ldf0p8")
        self.assertEqual(row["ft_epochs"], 6)
        self.assertEqual(row["target_min_loss_delta_policy"], 0.001)
        self.assertEqual(row["early_stopping_patience"], 3)
        self.assertEqual(row["lr_decay_patience"], 2)
        self.assertEqual(row["lr_decay_factor"], 0.8)
        self.assertAlmostEqual(row["target_retention_gap_mean"], 0.8)
        self.assertAlmostEqual(row["target_retention_ratio"], 3.0)
        self.assertEqual(row["guard_epoch_counts_available_cases"], 2)
        self.assertTrue(row["guard_epoch_counts_available_all"])
        self.assertAlmostEqual(row["guard_accepted_epochs_mean"], 5.0)
        self.assertAlmostEqual(row["guard_retention_rejected_epochs_mean"], 0.5)
        self.assertAlmostEqual(row["guard_target_stale_epochs_mean"], 1.5)
        self.assertAlmostEqual(row["guard_acceptance_rate_mean"], 0.75)
        self.assertAlmostEqual(row["guard_retention_rejected_rate_mean"], 0.10)
        self.assertAlmostEqual(row["guard_target_stale_rate_mean"], 0.30)
        self.assertEqual(row["input_promotion_run_key"], "strong_effect::r12_a64_lr4::gain_g4")
        self.assertEqual(row["input_promotion_rank"], 1)
        self.assertEqual(row["input_promotion_metric"], "target_retention_ratio")
        self.assertEqual(row["input_promotion_value"], 2.8)
        self.assertAlmostEqual(row["input_promotion_metric_current"], 3.0)
        self.assertAlmostEqual(row["input_promotion_metric_delta"], 0.2)
        self.assertAlmostEqual(row["input_promotion_metric_regression"], 0.0)
        self.assertTrue(row["input_promotion_ready"])
        self.assertEqual(row["input_promotion_ready_top_k"], 2)
        self.assertEqual(row["input_promotion_ready_within"], 0.25)
        self.assertTrue(row["input_promotion_ready_floor_passed"])
        self.assertEqual(row["input_promotion_ready_floor_failures"], [])
        self.assertEqual(row["input_promotion_ready_min_target_retention_ratio"], 2.5)
        self.assertEqual(row["input_promotion_ready_min_accepted_rate"], 1.0)
        self.assertEqual(row["input_promotion_ready_min_movement_ok_rate"], 1.0)
        self.assertTrue(row["input_promotion_ready_require_guard_counts_available"])
        self.assertEqual(row["input_promotion_ready_min_guard_acceptance_rate_mean"], 0.70)
        self.assertEqual(
            row["input_promotion_ready_max_guard_retention_rejected_epochs_mean"],
            0.0,
        )
        self.assertEqual(row["input_promotion_ready_max_guard_target_stale_epochs_mean"], 2.0)
        self.assertEqual(
            row["input_promotion_ready_max_guard_retention_rejected_rate_mean"],
            0.10,
        )
        self.assertEqual(row["input_promotion_ready_max_guard_target_stale_rate_mean"], 0.50)
        self.assertEqual(row["aggregate_row_type"], "config_aggregate")
        promotions = module.profile_run_promotion_rows(rows)
        self.assertAlmostEqual(promotions[0]["input_promotion_metric_current"], 3.0)
        self.assertAlmostEqual(promotions[0]["input_promotion_metric_delta"], 0.2)
        self.assertAlmostEqual(promotions[0]["input_promotion_metric_regression"], 0.0)
        self.assertEqual(promotions[0]["guard_epoch_counts_available_cases"], 2)
        self.assertTrue(promotions[0]["guard_epoch_counts_available_all"])
        self.assertAlmostEqual(promotions[0]["guard_target_stale_epochs_mean"], 1.5)
        self.assertAlmostEqual(promotions[0]["guard_acceptance_rate_mean"], 0.75)
        self.assertAlmostEqual(promotions[0]["guard_retention_rejected_rate_mean"], 0.10)
        self.assertAlmostEqual(promotions[0]["guard_target_stale_rate_mean"], 0.30)
        with tempfile.TemporaryDirectory() as tmpdir:
            no_ratio_aggregate_path = Path(tmpdir) / "profile-aggregate.jsonl"
            module.write_jsonl(
                no_ratio_aggregate_path,
                [
                    {
                        "row_type": "config_aggregate",
                        "config": "r12_a64_lr4",
                        "checkpoint_source_label": "gemma-4-e4b-it",
                        "cases": 1,
                        "case_labels": "adapter_ja",
                        "accepted_rate": 1.0,
                        "movement_ok_rate": 1.0,
                        "retention_accuracy_margin_min": 1.0,
                        "retention_perplexity_margin_min": 100.0,
                        "target_loss_delta_mean": 1.2,
                        "retention_loss_delta_mean": 0.0,
                    }
                ],
            )
            no_ratio_rows = module.profile_run_summary_rows(
                [
                    {
                        "source_profile": "strong_effect",
                        "selected_source": "gemma-4-e4b-it",
                        "promotion_metric": "target_retention_ratio",
                        "promotion_value": 2.8,
                        "jsonl": str(Path(tmpdir) / "profile.jsonl"),
                        "aggregate_jsonl": str(no_ratio_aggregate_path),
                        "shell": "python sweep.py",
                    }
                ]
            )
        self.assertIsNone(no_ratio_rows[0]["target_retention_ratio"])
        self.assertIsNone(no_ratio_rows[0]["input_promotion_metric_current"])
        self.assertIsNone(no_ratio_rows[0]["input_promotion_metric_delta"])
        self.assertIsNone(no_ratio_rows[0]["input_promotion_metric_regression"])
        with self.assertRaisesRegex(RuntimeError, "input_promotion_metric_regression unavailable"):
            module.check_profile_run_gates(
                no_ratio_rows,
                max_input_promotion_metric_regression=0.0,
            )

    def test_mlp_lora_profile_runner_compares_multi_config_profile_runs(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        rows = [
            {
                "row_type": "checkpoint_source_profile_run",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "config": "r6_a32_lr3",
                "accepted_rate": 1.0,
                "movement_ok_rate": 1.0,
                "target_loss_delta_mean": 1.0,
                "retention_loss_delta_mean": 0.5,
                "target_retention_gap_mean": 0.5,
                "target_retention_ratio": 2.0,
                "retention_accuracy_margin_min": 1.0,
                "retention_perplexity_margin_min": 100.0,
            },
            {
                "row_type": "checkpoint_source_profile_run",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "config": "r12_a64_lr4",
                "accepted_rate": 1.0,
                "movement_ok_rate": 1.0,
                "target_loss_delta_mean": 1.2,
                "retention_loss_delta_mean": 0.4,
                "target_retention_gap_mean": 0.8,
                "target_retention_ratio": 3.0,
                "retention_accuracy_margin_min": 1.0,
                "retention_perplexity_margin_min": 100.0,
            },
        ]
        self.assertEqual(
            module.check_profile_run_gates(
                rows,
                min_target_retention_ratio=1.5,
                min_accepted_rate=1.0,
                min_movement_ok_rate=1.0,
            ),
            2,
        )
        guarded_rows = [
            dict(
                rows[0],
                guard_epoch_counts_available_cases=2,
                guard_epoch_counts_available_all=True,
                guard_retention_rejected_epochs_mean=0.0,
                guard_target_stale_epochs_mean=0.25,
                guard_acceptance_rate_mean=0.90,
                guard_retention_rejected_rate_mean=0.0,
                guard_target_stale_rate_mean=0.05,
            ),
            dict(
                rows[1],
                guard_epoch_counts_available_cases=2,
                guard_epoch_counts_available_all=True,
                guard_retention_rejected_epochs_mean=0.0,
                guard_target_stale_epochs_mean=0.5,
                guard_acceptance_rate_mean=0.85,
                guard_retention_rejected_rate_mean=0.0,
                guard_target_stale_rate_mean=0.10,
            ),
        ]
        self.assertEqual(
            module.check_profile_run_gates(
                guarded_rows,
                require_guard_counts_available=True,
                min_guard_acceptance_rate_mean=0.85,
                max_guard_retention_rejected_epochs_mean=0.0,
                max_guard_target_stale_epochs_mean=0.5,
                max_guard_retention_rejected_rate_mean=0.0,
                max_guard_target_stale_rate_mean=0.10,
            ),
            2,
        )
        with self.assertRaisesRegex(RuntimeError, "guard_epoch_counts_available_all"):
            module.check_profile_run_gates(
                [dict(guarded_rows[0], guard_epoch_counts_available_all=False)],
                require_guard_counts_available=True,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_retention_rejected_epochs_mean"):
            module.check_profile_run_gates(
                [dict(guarded_rows[0], guard_retention_rejected_epochs_mean=1.0)],
                max_guard_retention_rejected_epochs_mean=0.0,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_target_stale_epochs_mean"):
            module.check_profile_run_gates(
                [dict(guarded_rows[0], guard_target_stale_epochs_mean=1.5)],
                max_guard_target_stale_epochs_mean=1.0,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_acceptance_rate_mean"):
            module.check_profile_run_gates(
                [dict(guarded_rows[0], guard_acceptance_rate_mean=0.5)],
                min_guard_acceptance_rate_mean=0.8,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_retention_rejected_rate_mean"):
            module.check_profile_run_gates(
                [dict(guarded_rows[0], guard_retention_rejected_rate_mean=0.25)],
                max_guard_retention_rejected_rate_mean=0.0,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_target_stale_rate_mean"):
            module.check_profile_run_gates(
                [dict(guarded_rows[0], guard_target_stale_rate_mean=0.5)],
                max_guard_target_stale_rate_mean=0.25,
            )
        self.assertEqual(
            module.compare_profile_run_summaries(
                rows,
                rows,
                max_target_loss_regression=0.0,
                max_target_retention_ratio_regression=0.0,
                min_target_retention_ratio=1.5,
                min_accepted_rate=1.0,
                min_movement_ok_rate=1.0,
            ),
            2,
        )
        promotions = module.profile_run_promotion_rows(
            rows,
            promotion_metric="target_retention_ratio",
        )
        self.assertEqual(len(promotions), 2)
        self.assertEqual(promotions[0]["row_type"], "checkpoint_source_profile_promotion")
        self.assertEqual(promotions[0]["run_key"], "strong_effect::r12_a64_lr4")
        self.assertEqual(promotions[0]["promotion_rank"], 1)
        self.assertEqual(promotions[0]["promotion_value"], 3.0)
        self.assertTrue(promotions[0]["promotion_ready"])
        self.assertEqual(promotions[0]["promotion_ready_top_k"], 1)
        self.assertIsNone(promotions[0]["promotion_ready_within"])
        self.assertEqual(promotions[1]["run_key"], "strong_effect::r6_a32_lr3")
        self.assertFalse(promotions[1]["promotion_ready"])
        top_k_promotions = module.profile_run_promotion_rows(
            rows,
            promotion_metric="target_retention_ratio",
            ready_top_k=2,
        )
        self.assertTrue(top_k_promotions[1]["promotion_ready"])
        self.assertEqual(top_k_promotions[1]["promotion_ready_top_k"], 2)
        near_tie_promotions = module.profile_run_promotion_rows(
            rows,
            promotion_metric="target_retention_ratio",
            ready_within=1.0,
        )
        self.assertTrue(near_tie_promotions[1]["promotion_ready"])
        self.assertEqual(near_tie_promotions[1]["promotion_ready_within"], 1.0)
        strict_gap_promotions = module.profile_run_promotion_rows(
            rows,
            promotion_metric="target_retention_ratio",
            ready_within=0.5,
        )
        self.assertFalse(strict_gap_promotions[1]["promotion_ready"])
        floored_promotions = module.profile_run_promotion_rows(
            rows,
            promotion_metric="target_retention_ratio",
            ready_top_k=2,
            ready_min_target_retention_ratio=2.5,
            ready_min_accepted_rate=1.0,
        )
        self.assertTrue(floored_promotions[0]["promotion_ready"])
        self.assertTrue(floored_promotions[0]["promotion_ready_floor_passed"])
        self.assertFalse(floored_promotions[1]["promotion_ready"])
        self.assertFalse(floored_promotions[1]["promotion_ready_floor_passed"])
        self.assertEqual(
            floored_promotions[1]["promotion_ready_floor_failures"],
            ["target_retention_ratio<2.500000000"],
        )
        self.assertEqual(
            floored_promotions[1]["promotion_ready_min_target_retention_ratio"],
            2.5,
        )
        unavailable_metric_rows = rows + [
            {
                "row_type": "checkpoint_source_profile_run",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "config": "r24_a128_lr5",
                "accepted_rate": 1.0,
                "movement_ok_rate": 1.0,
                "target_loss_delta_mean": 1.4,
                "retention_loss_delta_mean": 0.0,
                "target_retention_gap_mean": 1.4,
                "target_retention_ratio": None,
                "retention_accuracy_margin_min": 1.0,
                "retention_perplexity_margin_min": 100.0,
            }
        ]
        unavailable_promotions = module.profile_run_promotion_rows(
            unavailable_metric_rows,
            promotion_metric="target_retention_ratio",
            ready_top_k=3,
        )
        self.assertEqual(len(unavailable_promotions), 3)
        self.assertEqual(
            unavailable_promotions[-1]["run_key"],
            "strong_effect::r24_a128_lr5",
        )
        self.assertIsNone(unavailable_promotions[-1]["promotion_value"])
        self.assertFalse(unavailable_promotions[-1]["promotion_ready"])
        self.assertFalse(unavailable_promotions[-1]["promotion_ready_floor_passed"])
        self.assertEqual(
            unavailable_promotions[-1]["promotion_ready_floor_failures"],
            ["target_retention_ratio=unavailable"],
        )
        regression_limited_rows = [
            dict(rows[0], input_promotion_metric_regression=0.3),
            dict(rows[1], input_promotion_metric_regression=0.02),
        ]
        regression_limited_promotions = module.profile_run_promotion_rows(
            regression_limited_rows,
            promotion_metric="target_retention_ratio",
            ready_top_k=2,
            ready_max_input_promotion_metric_regression=0.05,
        )
        self.assertTrue(regression_limited_promotions[0]["promotion_ready"])
        self.assertEqual(
            regression_limited_promotions[0][
                "promotion_ready_max_input_promotion_metric_regression"
            ],
            0.05,
        )
        self.assertFalse(regression_limited_promotions[1]["promotion_ready"])
        self.assertEqual(
            regression_limited_promotions[1]["promotion_ready_floor_failures"],
            ["input_promotion_metric_regression>0.050000000"],
        )
        guard_limited_rows = [
            dict(
                rows[0],
                guard_epoch_counts_available_cases=2,
                guard_epoch_counts_available_all=False,
                guard_retention_rejected_epochs_mean=0.0,
                guard_target_stale_epochs_mean=0.25,
                guard_acceptance_rate_mean=1.0,
                guard_retention_rejected_rate_mean=0.0,
                guard_target_stale_rate_mean=0.0,
            ),
            dict(
                rows[1],
                guard_epoch_counts_available_cases=2,
                guard_epoch_counts_available_all=True,
                guard_retention_rejected_epochs_mean=0.0,
                guard_target_stale_epochs_mean=1.5,
                guard_acceptance_rate_mean=0.4,
                guard_retention_rejected_rate_mean=0.0,
                guard_target_stale_rate_mean=0.75,
            ),
        ]
        guard_limited_promotions = module.profile_run_promotion_rows(
            guard_limited_rows,
            promotion_metric="target_retention_ratio",
            ready_top_k=2,
            ready_require_guard_counts_available=True,
            ready_min_guard_acceptance_rate_mean=0.8,
            ready_max_guard_retention_rejected_epochs_mean=0.0,
            ready_max_guard_target_stale_epochs_mean=1.0,
            ready_max_guard_retention_rejected_rate_mean=0.0,
            ready_max_guard_target_stale_rate_mean=0.5,
        )
        self.assertFalse(guard_limited_promotions[0]["promotion_ready"])
        self.assertTrue(
            guard_limited_promotions[0]["promotion_ready_require_guard_counts_available"]
        )
        self.assertEqual(
            guard_limited_promotions[0][
                "promotion_ready_max_guard_retention_rejected_epochs_mean"
            ],
            0.0,
        )
        self.assertEqual(
            guard_limited_promotions[0]["promotion_ready_max_guard_target_stale_epochs_mean"],
            1.0,
        )
        self.assertEqual(
            guard_limited_promotions[0]["promotion_ready_min_guard_acceptance_rate_mean"],
            0.8,
        )
        self.assertEqual(
            guard_limited_promotions[0][
                "promotion_ready_max_guard_retention_rejected_rate_mean"
            ],
            0.0,
        )
        self.assertEqual(
            guard_limited_promotions[0]["promotion_ready_max_guard_target_stale_rate_mean"],
            0.5,
        )
        self.assertEqual(
            guard_limited_promotions[0]["promotion_ready_floor_failures"],
            [
                "guard_acceptance_rate_mean<0.800000000",
                "guard_target_stale_epochs_mean>1.000000000",
                "guard_target_stale_rate_mean>0.500000000",
            ],
        )
        self.assertFalse(guard_limited_promotions[1]["promotion_ready"])
        self.assertEqual(
            guard_limited_promotions[1]["promotion_ready_floor_failures"],
            ["guard_epoch_counts_available_all=false"],
        )
        guard_ready_promotions = module.profile_run_promotion_rows(
            guarded_rows,
            promotion_metric="target_retention_ratio",
            ready_top_k=2,
            ready_require_guard_counts_available=True,
            ready_min_guard_acceptance_rate_mean=0.85,
            ready_max_guard_retention_rejected_epochs_mean=0.0,
            ready_max_guard_target_stale_epochs_mean=0.5,
            ready_max_guard_retention_rejected_rate_mean=0.0,
            ready_max_guard_target_stale_rate_mean=0.10,
        )
        self.assertEqual(
            module.check_promotion_ready_gates(
                guard_ready_promotions,
                min_ready_count=2,
                min_ready_guard_policy_count=2,
                require_ready_guard_policy=True,
            ),
            2,
        )
        self.assertEqual(
            module.check_promotion_ready_gates(
                floored_promotions,
                min_ready_count=1,
                min_ready_rate=0.5,
            ),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "ready_count"):
            module.check_promotion_ready_gates(
                floored_promotions,
                min_ready_count=2,
            )
        with self.assertRaisesRegex(RuntimeError, "ready_rate"):
            module.check_promotion_ready_gates(
                floored_promotions,
                min_ready_rate=0.75,
            )
        with self.assertRaisesRegex(RuntimeError, "ready_guard_policy_count"):
            module.check_promotion_ready_gates(
                floored_promotions,
                min_ready_guard_policy_count=1,
            )
        with self.assertRaisesRegex(RuntimeError, "ready_guard_policy_count"):
            module.check_promotion_ready_gates(
                floored_promotions,
                require_ready_guard_policy=True,
            )
        duplicate = [dict(rows[0]), dict(rows[0])]
        with self.assertRaisesRegex(ValueError, "duplicate profile run key"):
            module.compare_profile_run_summaries(duplicate, duplicate)

    def test_mlp_lora_profile_runner_gates_input_promotion_metric_regression(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        row = {
            "row_type": "checkpoint_source_profile_run",
            "source_profile": "selective_ratio",
            "selected_source": "gemma-4-e4b-it",
            "config": "r12_a64_lr4",
            "accepted_rate": 1.0,
            "movement_ok_rate": 1.0,
            "target_loss_delta_mean": 1.0,
            "retention_loss_delta_mean": 0.75,
            "target_retention_gap_mean": 0.25,
            "target_retention_ratio": 1.35,
            "input_promotion_metric": "target_retention_ratio",
            "input_promotion_value": 1.68,
        }
        self.assertEqual(
            module.check_profile_run_gates(
                [row],
                max_input_promotion_metric_regression=0.34,
            ),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "input_promotion_metric_regression"):
            module.check_profile_run_gates(
                [row],
                max_input_promotion_metric_regression=0.1,
            )
        unsupported = dict(row, input_promotion_metric="unknown_metric")
        with self.assertRaisesRegex(RuntimeError, "unsupported input_promotion_metric"):
            module.check_profile_run_gates(
                [unsupported],
                max_input_promotion_metric_regression=0.34,
            )
        missing = dict(row)
        del missing["input_promotion_metric"]
        with self.assertRaisesRegex(RuntimeError, "input_promotion_metric unavailable"):
            module.check_profile_run_gates(
                [missing],
                max_input_promotion_metric_regression=0.34,
            )

    def test_mlp_lora_profile_runner_preserves_ft_control_from_command_row(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        with tempfile.TemporaryDirectory() as tmpdir:
            aggregate_path = Path(tmpdir) / "profile-aggregate.jsonl"
            module.write_jsonl(
                aggregate_path,
                [
                    {
                        "row_type": "config_aggregate",
                        "config": "r12_a64_lr4",
                        "checkpoint_source_label": "gemma-4-e4b-it",
                        "checkpoint_source_gain": 4.0,
                        "cases": 1,
                        "case_labels": "adapter_ja",
                        "accepted_rate": 1.0,
                        "movement_ok_rate": 1.0,
                        "target_loss_delta_mean": 0.9,
                        "retention_loss_delta_mean": 0.3,
                    }
                ],
            )
            rows = module.profile_run_summary_rows(
                [
                    {
                        "source_profile": "strong_effect",
                        "selected_source": "gemma-4-e4b-it",
                        "selected_config": "r12_a64_lr4::ep6::tmin0p001::pat3",
                        "run_config_key": "r12_a64_lr4::ep6::tmin0p001::pat3",
                        "training_policy_key": "policy:command-ft6",
                        "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
                        "ft_epochs": 6,
                        "target_min_loss_delta_policy": 0.001,
                        "early_stopping_patience": 3,
                        "early_stopping_min_delta": 0.0005,
                        "lr_decay_patience": 2,
                        "lr_decay_factor": 0.8,
                        "lr_decay_min_delta": 0.00025,
                        "winner_metric": "target_loss_delta_mean",
                        "winner_value": 0.9,
                        "jsonl": str(Path(tmpdir) / "profile.jsonl"),
                        "aggregate_jsonl": str(aggregate_path),
                        "shell": "python sweep.py",
                    }
                ]
            )
        row = rows[0]
        self.assertEqual(row["training_policy_key"], "policy:command-ft6")
        self.assertEqual(row["ft_control_variant"], "ep6::tmin0p001::pat3::ldp2::ldf0p8")
        self.assertEqual(row["ft_epochs"], 6)
        self.assertEqual(row["target_min_loss_delta_policy"], 0.001)
        self.assertEqual(row["early_stopping_patience"], 3)
        self.assertEqual(row["early_stopping_min_delta"], 0.0005)
        self.assertEqual(row["lr_decay_patience"], 2)
        self.assertEqual(row["lr_decay_factor"], 0.8)
        self.assertEqual(row["lr_decay_min_delta"], 0.00025)

    def test_mlp_lora_profile_runner_normalizes_default_ft_control_key(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        command_policy = {
            "adapter_weight_decay_variant": None,
            "adapter_weight_decay": 0.0,
            "max_grad_norm_variant": None,
            "max_grad_norm": 2.0,
            "gradient_accumulation_steps_variant": None,
            "gradient_accumulation_steps": 2,
            "ft_control_variant": "ep10::tmin0::patnone::md0::ldpnone::ldf0p5::ldmd0",
            "ft_epochs": 10,
            "target_min_loss_delta_policy": 0.0,
            "early_stopping_patience": None,
            "early_stopping_min_delta": 0.0,
            "lr_decay_patience": None,
            "lr_decay_factor": 0.5,
            "lr_decay_min_delta": 0.0,
        }
        aggregate_policy = dict(command_policy, ft_control_variant=None)
        aggregate_key = module.training_policy_key(aggregate_policy)
        command_key = module.training_policy_key(command_policy)
        self.assertNotEqual(aggregate_key, command_key)
        self.assertEqual(
            module.canonical_training_policy_key(aggregate_policy),
            module.canonical_training_policy_key(command_policy),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            aggregate_path = Path(tmpdir) / "profile-aggregate.jsonl"
            module.write_jsonl(
                aggregate_path,
                [
                    {
                        "row_type": "config_aggregate",
                        "config": "r18_a128_lr5",
                        "checkpoint_source_label": "byte_mlp_trained_hf",
                        "training_policy_key": aggregate_key,
                        "cases": 3,
                        "case_labels": "adapter_ja,route_cats,geometry_tokens",
                        "accepted_rate": 1.0,
                        "movement_ok_rate": 1.0,
                        "target_loss_delta_mean": 1.9,
                        "retention_loss_delta_mean": 0.8,
                        **aggregate_policy,
                    }
                ],
            )
            rows = module.profile_run_summary_rows(
                [
                    {
                        "source_profile": "selective_ratio",
                        "selected_source": "byte_mlp_trained_hf",
                        "training_policy_key": command_key,
                        "jsonl": str(Path(tmpdir) / "profile.jsonl"),
                        "aggregate_jsonl": str(aggregate_path),
                        "shell": "python sweep.py",
                        **command_policy,
                    }
                ]
            )

        row = rows[0]
        self.assertEqual(row["training_policy_key"], command_key)
        self.assertEqual(
            row["ft_control_variant"],
            "ep10::tmin0::patnone::md0::ldpnone::ldf0p5::ldmd0",
        )

    def test_mlp_lora_profile_runner_rejects_training_policy_key_mismatch(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        with tempfile.TemporaryDirectory() as tmpdir:
            aggregate_path = Path(tmpdir) / "profile-aggregate.jsonl"
            module.write_jsonl(
                aggregate_path,
                [
                    {
                        "row_type": "config_aggregate",
                        "config": "r12_a64_lr4",
                        "training_policy_key": "policy:aggregate-ft6",
                        "target_loss_delta_mean": 0.9,
                        "retention_loss_delta_mean": 0.3,
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "training_policy_key mismatch"):
                module.profile_run_summary_rows(
                    [
                        {
                            "source_profile": "strong_effect",
                            "selected_source": "gemma-4-e4b-it",
                            "training_policy_key": "policy:command-ft6",
                            "jsonl": str(Path(tmpdir) / "profile.jsonl"),
                            "aggregate_jsonl": str(aggregate_path),
                            "shell": "python sweep.py",
                        }
                    ]
                )

    def test_mlp_lora_profile_runner_compares_profile_run_summaries(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        baseline = [
            {
                "row_type": "checkpoint_source_profile_run",
                "source_profile": "strong_effect",
                "selected_source": "gemma-4-e4b-it",
                "config": "r12_a64_lr4",
                "adapter_weight_decay_variant": "wd0p01",
                "adapter_weight_decay": 0.01,
                "max_grad_norm_variant": "gn1p5",
                "max_grad_norm": 1.5,
                "gradient_accumulation_steps_variant": "accum4",
                "gradient_accumulation_steps": 4,
                "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
                "ft_epochs": 6,
                "target_min_loss_delta_policy": 0.001,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.0,
                "lr_decay_patience": 2,
                "lr_decay_factor": 0.8,
                "lr_decay_min_delta": 0.0,
                "training_policy_key": "policy:llama-ft6",
                "cases": 2,
                "case_labels": "adapter_ja,route_cats",
                "accepted_rate": 1.0,
                "movement_ok_rate": 1.0,
                "target_loss_delta_mean": 1.2,
                "retention_loss_delta_mean": 0.4,
                "target_retention_gap_mean": 0.8,
                "target_retention_ratio": 3.0,
                "retention_accuracy_margin_min": 1.0,
                "retention_perplexity_margin_min": 100.0,
                "guard_acceptance_rate_mean": 1.0,
                "guard_retention_rejected_rate_mean": 0.0,
                "guard_target_stale_rate_mean": 0.0,
                "input_promotion_run_key": "strong_effect::r12_a64_lr4",
                "input_promotion_rank": 1,
                "input_promotion_metric": "target_retention_ratio",
                "input_promotion_value": 3.0,
                "input_promotion_ready": True,
                "input_promotion_ready_top_k": 2,
                "input_promotion_ready_within": 0.25,
                "input_promotion_ready_floor_passed": True,
                "input_promotion_ready_floor_failures": [],
                "input_promotion_ready_min_target_retention_ratio": 2.5,
                "input_promotion_ready_min_accepted_rate": 1.0,
                "input_promotion_ready_require_guard_counts_available": True,
                "input_promotion_ready_min_guard_acceptance_rate_mean": 0.75,
                "input_promotion_ready_max_guard_retention_rejected_epochs_mean": 0.0,
                "input_promotion_ready_max_guard_target_stale_epochs_mean": 1.0,
                "input_promotion_ready_max_guard_retention_rejected_rate_mean": 0.0,
                "input_promotion_ready_max_guard_target_stale_rate_mean": 0.25,
            }
        ]
        self.assertEqual(
            module.compare_profile_run_summaries(
                list(baseline),
                baseline,
                max_target_loss_regression=0.0,
                max_retention_loss_regression=0.0,
                max_target_retention_gap_regression=0.0,
                max_target_retention_ratio_regression=0.0,
                max_accepted_rate_regression=0.0,
                max_movement_ok_rate_regression=0.0,
                min_target_retention_ratio=2.5,
                min_accepted_rate=1.0,
                min_movement_ok_rate=1.0,
                max_guard_acceptance_rate_regression=0.0,
                max_guard_retention_rejected_rate_regression=0.0,
                max_guard_target_stale_rate_regression=0.0,
                min_retention_accuracy_margin=0.5,
                min_retention_perplexity_margin=50.0,
                require_source_match=True,
                require_config_match=True,
                require_case_scope_match=True,
                require_training_policy_match=True,
                require_input_promotion_match=True,
            ),
            1,
        )
        no_ratio = [
            dict(
                baseline[0],
                retention_loss_delta_mean=0.0,
                target_retention_gap_mean=1.2,
                target_retention_ratio=None,
            )
        ]
        self.assertEqual(
            module.compare_profile_run_summaries(
                no_ratio,
                no_ratio,
                max_target_loss_regression=0.0,
                max_target_retention_gap_regression=0.0,
                min_accepted_rate=1.0,
            ),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "target_retention_ratio regression is unavailable"):
            module.compare_profile_run_summaries(
                no_ratio,
                no_ratio,
                max_target_retention_ratio_regression=0.0,
            )
        with self.assertRaisesRegex(RuntimeError, "target_retention_ratio is unavailable"):
            module.compare_profile_run_summaries(
                no_ratio,
                no_ratio,
                min_target_retention_ratio=1.0,
            )
        legacy_baseline = [dict(baseline[0])]
        legacy_baseline[0].pop("input_promotion_ready_top_k")
        legacy_baseline[0].pop("input_promotion_ready_within")
        legacy_baseline[0].pop("input_promotion_ready_floor_passed")
        legacy_baseline[0].pop("input_promotion_ready_floor_failures")
        legacy_baseline[0].pop("input_promotion_ready_min_target_retention_ratio")
        legacy_baseline[0].pop("input_promotion_ready_min_accepted_rate")
        legacy_baseline[0].pop("input_promotion_ready_require_guard_counts_available")
        legacy_baseline[0].pop("input_promotion_ready_min_guard_acceptance_rate_mean")
        legacy_baseline[0].pop("input_promotion_ready_max_guard_retention_rejected_epochs_mean")
        legacy_baseline[0].pop("input_promotion_ready_max_guard_target_stale_epochs_mean")
        legacy_baseline[0].pop("input_promotion_ready_max_guard_retention_rejected_rate_mean")
        legacy_baseline[0].pop("input_promotion_ready_max_guard_target_stale_rate_mean")
        default_policy_current = [
            dict(
                baseline[0],
                input_promotion_ready_top_k=1,
                input_promotion_ready_within=None,
                input_promotion_ready_floor_passed=None,
                input_promotion_ready_floor_failures=None,
                input_promotion_ready_min_target_retention_ratio=None,
                input_promotion_ready_min_accepted_rate=None,
                input_promotion_ready_require_guard_counts_available=None,
                input_promotion_ready_min_guard_acceptance_rate_mean=None,
                input_promotion_ready_max_guard_retention_rejected_epochs_mean=None,
                input_promotion_ready_max_guard_target_stale_epochs_mean=None,
                input_promotion_ready_max_guard_retention_rejected_rate_mean=None,
                input_promotion_ready_max_guard_target_stale_rate_mean=None,
            )
        ]
        self.assertEqual(
            module.compare_profile_run_summaries(
                default_policy_current,
                legacy_baseline,
                require_input_promotion_match=True,
            ),
            1,
        )
        current = [
            dict(
                baseline[0],
                target_loss_delta_mean=1.0,
                target_retention_gap_mean=0.6,
                target_retention_ratio=2.0,
                accepted_rate=0.5,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "target_loss_delta_mean"):
            module.compare_profile_run_summaries(
                current,
                baseline,
                max_target_loss_regression=0.1,
                max_target_retention_gap_regression=0.1,
                max_target_retention_ratio_regression=0.1,
                min_target_retention_ratio=2.5,
                max_accepted_rate_regression=0.1,
                min_accepted_rate=0.75,
            )
        with self.assertRaisesRegex(RuntimeError, "target_retention_ratio"):
            module.compare_profile_run_summaries(
                baseline,
                baseline,
                min_target_retention_ratio=3.1,
            )
        with self.assertRaisesRegex(RuntimeError, "movement_ok_rate"):
            module.compare_profile_run_summaries(
                [dict(baseline[0], movement_ok_rate=0.5)],
                baseline,
                min_movement_ok_rate=0.75,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_acceptance_rate_mean"):
            module.compare_profile_run_summaries(
                [dict(baseline[0], guard_acceptance_rate_mean=0.75)],
                baseline,
                max_guard_acceptance_rate_regression=0.1,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_retention_rejected_rate_mean"):
            module.compare_profile_run_summaries(
                [dict(baseline[0], guard_retention_rejected_rate_mean=0.25)],
                baseline,
                max_guard_retention_rejected_rate_regression=0.0,
            )
        with self.assertRaisesRegex(RuntimeError, "guard_target_stale_rate_mean"):
            module.compare_profile_run_summaries(
                [dict(baseline[0], guard_target_stale_rate_mean=0.25)],
                baseline,
                max_guard_target_stale_rate_regression=0.0,
            )
        with self.assertRaisesRegex(RuntimeError, "case scope changed"):
            module.compare_profile_run_summaries(
                [dict(baseline[0], cases=1, case_labels="adapter_ja")],
                baseline,
                require_case_scope_match=True,
            )
        policy_changed = [
            dict(
                baseline[0],
                ft_control_variant="ep10::tmin0p001::pat3::ldp2::ldf0p8",
                ft_epochs=10,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "training policy changed"):
            module.compare_profile_run_summaries(
                policy_changed,
                baseline,
                require_training_policy_match=True,
            )
        policy_key_changed = [
            dict(
                baseline[0],
                training_policy_key="policy:llama-ft6-shadow",
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "training policy changed"):
            module.compare_profile_run_summaries(
                policy_key_changed,
                baseline,
                require_training_policy_match=True,
            )
        promotion_changed = [
            dict(
                baseline[0],
                input_promotion_rank=2,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "input promotion changed"):
            module.compare_profile_run_summaries(
                promotion_changed,
                baseline,
                require_input_promotion_match=True,
            )
        promotion_policy_changed = [
            dict(
                baseline[0],
                input_promotion_ready_top_k=1,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "input promotion changed"):
            module.compare_profile_run_summaries(
                promotion_policy_changed,
                baseline,
                require_input_promotion_match=True,
            )
        promotion_floor_changed = [
            dict(
                baseline[0],
                input_promotion_ready_min_target_retention_ratio=2.0,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "input promotion changed"):
            module.compare_profile_run_summaries(
                promotion_floor_changed,
                baseline,
                require_input_promotion_match=True,
            )
        promotion_guard_policy_changed = [
            dict(
                baseline[0],
                input_promotion_ready_max_guard_target_stale_epochs_mean=2.0,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "input promotion changed"):
            module.compare_profile_run_summaries(
                promotion_guard_policy_changed,
                baseline,
                require_input_promotion_match=True,
            )
        promotion_guard_rate_policy_changed = [
            dict(
                baseline[0],
                input_promotion_ready_max_guard_target_stale_rate_mean=0.5,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "input promotion changed"):
            module.compare_profile_run_summaries(
                promotion_guard_rate_policy_changed,
                baseline,
                require_input_promotion_match=True,
            )

    def test_mlp_lora_profile_runner_cli_compares_saved_run_summaries(self):
        module = load_example("byte_lm_mlp_lora_profile_runner")
        row = {
            "row_type": "checkpoint_source_profile_run",
            "source_profile": "strong_effect",
            "selected_source": "gemma-4-e4b-it",
            "config": "r12_a64_lr4",
            "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
            "ft_epochs": 6,
            "target_min_loss_delta_policy": 0.001,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0,
            "lr_decay_patience": 2,
            "lr_decay_factor": 0.8,
            "lr_decay_min_delta": 0.0,
            "accepted_rate": 1.0,
            "movement_ok_rate": 1.0,
            "guard_epoch_counts_available_cases": 2,
            "guard_epoch_counts_available_all": True,
            "guard_accepted_epochs_total": 12.0,
            "guard_accepted_epochs_mean": 6.0,
            "guard_accepted_epochs_max": 6.0,
            "guard_retention_rejected_epochs_total": 0.0,
            "guard_retention_rejected_epochs_mean": 0.0,
            "guard_retention_rejected_epochs_max": 0.0,
            "guard_target_stale_epochs_total": 0.0,
            "guard_target_stale_epochs_mean": 0.0,
            "guard_target_stale_epochs_max": 0.0,
            "guard_acceptance_rate_mean": 1.0,
            "guard_acceptance_rate_min": 1.0,
            "guard_retention_rejected_rate_mean": 0.0,
            "guard_retention_rejected_rate_max": 0.0,
            "guard_target_stale_rate_mean": 0.0,
            "guard_target_stale_rate_max": 0.0,
            "cases": 2,
            "case_labels": "adapter_ja,route_cats",
            "target_loss_delta_mean": 1.2,
            "retention_loss_delta_mean": 0.4,
            "target_retention_gap_mean": 0.8,
            "target_retention_ratio": 3.0,
            "retention_accuracy_margin_min": 1.0,
            "retention_perplexity_margin_min": 100.0,
            "input_promotion_run_key": "strong_effect::r12_a64_lr4",
            "input_promotion_rank": 1,
            "input_promotion_metric": "target_retention_ratio",
            "input_promotion_value": 3.0,
            "input_promotion_metric_current": 3.0,
            "input_promotion_metric_delta": 0.0,
            "input_promotion_metric_regression": 0.0,
            "input_promotion_ready": True,
            "input_promotion_ready_top_k": 2,
            "input_promotion_ready_within": 0.25,
            "input_promotion_ready_floor_passed": True,
            "input_promotion_ready_floor_failures": [],
            "input_promotion_ready_min_target_retention_ratio": 2.5,
            "input_promotion_ready_min_accepted_rate": 1.0,
            "input_promotion_ready_max_input_promotion_metric_regression": 0.0,
            "input_promotion_ready_require_guard_counts_available": True,
            "input_promotion_ready_min_guard_acceptance_rate_mean": 1.0,
            "input_promotion_ready_max_guard_retention_rejected_epochs_mean": 0.0,
            "input_promotion_ready_max_guard_target_stale_epochs_mean": 1.0,
            "input_promotion_ready_max_guard_retention_rejected_rate_mean": 0.0,
            "input_promotion_ready_max_guard_target_stale_rate_mean": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir) / "current.jsonl"
            baseline_path = Path(tmpdir) / "baseline.jsonl"
            promotion_path = Path(tmpdir) / "promotion.jsonl"
            module.write_jsonl(current_path, [row])
            module.write_jsonl(baseline_path, [row])
            old_argv = sys.argv
            sys.argv = [
                "byte_lm_mlp_lora_profile_runner.py",
                "--current-run-summary-jsonl",
                str(current_path),
                "--compare-run-summary-jsonl",
                str(baseline_path),
                "--max-run-target-loss-regression",
                "0.0",
                "--max-run-retention-loss-regression",
                "0.0",
                "--max-run-target-retention-gap-regression",
                "0.0",
                "--max-run-target-retention-ratio-regression",
                "0.0",
                "--min-run-target-retention-ratio",
                "2.5",
                "--max-run-accepted-rate-regression",
                "0.0",
                "--min-run-accepted-rate",
                "1.0",
                "--max-run-movement-ok-rate-regression",
                "0.0",
                "--max-run-guard-acceptance-rate-regression",
                "0.0",
                "--max-run-guard-retention-rejected-rate-regression",
                "0.0",
                "--max-run-guard-target-stale-rate-regression",
                "0.0",
                "--min-run-movement-ok-rate",
                "1.0",
                "--min-run-retention-accuracy-margin",
                "0.5",
                "--min-run-retention-perplexity-margin",
                "50.0",
                "--max-run-input-promotion-metric-regression",
                "0.0",
                "--require-run-guard-counts-available",
                "--min-run-guard-acceptance-rate-mean",
                "1.0",
                "--max-run-guard-retention-rejected-epochs-mean",
                "0.0",
                "--max-run-guard-target-stale-epochs-mean",
                "1.0",
                "--max-run-guard-retention-rejected-rate-mean",
                "0.0",
                "--max-run-guard-target-stale-rate-mean",
                "0.25",
                "--require-run-source-match",
                "--require-run-config-match",
                "--require-run-case-scope-match",
                "--require-run-training-policy-match",
                "--require-run-input-promotion-match",
                "--promotion-jsonl",
                str(promotion_path),
                "--promotion-metric",
                "target_retention_ratio",
                "--promotion-ready-top-k",
                "2",
                "--promotion-ready-within",
                "0.25",
                "--promotion-ready-min-target-retention-ratio",
                "2.5",
                "--promotion-ready-min-accepted-rate",
                "1.0",
                "--promotion-ready-max-input-promotion-metric-regression",
                "0.0",
                "--promotion-ready-require-guard-counts-available",
                "--promotion-ready-min-guard-acceptance-rate-mean",
                "1.0",
                "--promotion-ready-max-guard-retention-rejected-epochs-mean",
                "0.0",
                "--promotion-ready-max-guard-target-stale-epochs-mean",
                "1.0",
                "--promotion-ready-max-guard-retention-rejected-rate-mean",
                "0.0",
                "--promotion-ready-max-guard-target-stale-rate-mean",
                "0.25",
                "--min-promotion-ready-count",
                "1",
                "--min-promotion-ready-rate",
                "1.0",
                "--min-promotion-ready-guard-policy-count",
                "1",
                "--require-promotion-ready-guard-policy",
            ]
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    module.main()
            finally:
                sys.argv = old_argv
            promotions = module.load_jsonl(promotion_path)
        self.assertIn("profile_run_compare profile=strong_effect", output.getvalue())
        self.assertIn("profile_run_compare_rows=1", output.getvalue())
        self.assertIn("input_promotion_metric_regression=0.000000000", output.getvalue())
        self.assertIn("guard_acceptance_rate_regression=0.000000000", output.getvalue())
        self.assertIn(
            "guard_retention_rejected_rate_regression=0.000000000",
            output.getvalue(),
        )
        self.assertIn("guard_target_stale_rate_regression=0.000000000", output.getvalue())
        self.assertIn("profile_promotion_jsonl=", output.getvalue())
        self.assertIn("profile_promotion_gate rows=1 ready_count=1", output.getvalue())
        self.assertIn("ready_guard_policy_count=1", output.getvalue())
        self.assertEqual(len(promotions), 1)
        self.assertEqual(promotions[0]["row_type"], "checkpoint_source_profile_promotion")
        self.assertEqual(promotions[0]["promotion_metric"], "target_retention_ratio")
        self.assertEqual(promotions[0]["promotion_ready_top_k"], 2)
        self.assertEqual(promotions[0]["promotion_ready_within"], 0.25)
        self.assertTrue(promotions[0]["promotion_ready_floor_passed"])
        self.assertEqual(promotions[0]["promotion_ready_floor_failures"], [])
        self.assertEqual(promotions[0]["promotion_ready_min_target_retention_ratio"], 2.5)
        self.assertEqual(promotions[0]["promotion_ready_min_accepted_rate"], 1.0)
        self.assertEqual(
            promotions[0]["promotion_ready_max_input_promotion_metric_regression"],
            0.0,
        )
        self.assertTrue(promotions[0]["promotion_ready_require_guard_counts_available"])
        self.assertEqual(promotions[0]["promotion_ready_min_guard_acceptance_rate_mean"], 1.0)
        self.assertEqual(
            promotions[0]["promotion_ready_max_guard_retention_rejected_epochs_mean"],
            0.0,
        )
        self.assertEqual(
            promotions[0]["promotion_ready_max_guard_target_stale_epochs_mean"],
            1.0,
        )
        self.assertEqual(
            promotions[0]["promotion_ready_max_guard_retention_rejected_rate_mean"],
            0.0,
        )
        self.assertEqual(promotions[0]["promotion_ready_max_guard_target_stale_rate_mean"], 0.25)
        self.assertTrue(promotions[0]["promotion_ready"])

    def test_mlp_lora_adapter_exposes_checkpoint_projection_fields(self):
        module = load_example("byte_lm_mlp_lora_adapter")
        args = types.SimpleNamespace(
            checkpoint_projection="zspace",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
            checkpoint_source_gain=1.0,
        )
        self.assertEqual(
            module.checkpoint_projection_fields(args),
            {
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 0.5,
                "checkpoint_projection_curvature": -0.5,
                "checkpoint_projection_frequency": 0.65,
            },
        )
        self.assertEqual(
            module.checkpoint_source_gain_fields(args),
            {"checkpoint_source_gain": 1.0},
        )
        args.checkpoint_projection = "none"
        self.assertEqual(
            module.checkpoint_projection_fields(args),
            {
                "checkpoint_projection": "none",
                "checkpoint_projection_strength": None,
                "checkpoint_projection_curvature": None,
                "checkpoint_projection_frequency": None,
            },
        )

    def test_mlp_lora_adapter_labels_nonzero_weight_decay(self):
        module = load_example("byte_lm_mlp_lora_adapter")
        self.assertEqual(module.adapter_weight_decay_variant_label(0.01), "wd0p01")
        self.assertEqual(module.max_grad_norm_variant_label(1.5), "gn1p5")
        self.assertEqual(module.gradient_accumulation_variant_label(4), "accum4")
        self.assertIsNone(module.ft_control_variant_label())
        self.assertEqual(
            module.ft_control_variant_label(6, 0.001, 3, 0.0, 2, 0.8, 0.0),
            "ep6::tmin0p001::pat3::ldp2::ldf0p8",
        )
        row = {
            "adapter_weight_decay_variant": "wd0p01",
            "adapter_weight_decay": 0.01,
            "max_grad_norm_variant": "gn1p5",
            "max_grad_norm": 1.5,
            "gradient_accumulation_steps_variant": "accum4",
            "gradient_accumulation_steps": 4,
            "ft_control_variant": "ep6::tmin0p001::pat3::ldp2::ldf0p8",
            "ft_epochs": 6,
            "target_min_loss_delta_policy": 0.001,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0,
            "lr_decay_patience": 2,
            "lr_decay_factor": 0.8,
            "lr_decay_min_delta": 0.0,
        }
        module.attach_training_policy_key(row)
        self.assertIn("adapter_weight_decay=0.010000000", row["training_policy_key"])
        self.assertIn(
            "ft_control_variant=ep6::tmin0p001::pat3::ldp2::ldf0p8",
            row["training_policy_key"],
        )
        self.assertEqual(module.adapter_config_label(0.0), "r12_a64")
        self.assertEqual(module.adapter_config_label(0.01), "r12_a64::wd0p01")
        self.assertEqual(
            module.adapter_config_label(0.01, 1.5, 4),
            "r12_a64::wd0p01::gn1p5::accum4",
        )
        self.assertEqual(
            module.adapter_config_label(0.01, 1.5, 4, 6, 0.001, 3, 0.0, 2, 0.8, 0.0),
            "r12_a64::wd0p01::gn1p5::accum4::ep6::tmin0p001::pat3::ldp2::ldf0p8",
        )

    def test_mlp_lora_adapter_configures_healthy_projection_preset(self):
        module = load_example("byte_lm_mlp_lora_adapter")
        args = types.SimpleNamespace(
            checkpoint_projection="none",
            checkpoint_projection_preset="healthy",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
            checkpoint_source_gain=1.0,
        )
        self.assertEqual(
            module.checkpoint_projection_fields(args),
            {
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 1.0,
                "checkpoint_projection_curvature": -0.04,
                "checkpoint_projection_frequency": 0.65,
            },
        )

    def test_mlp_lora_adapter_projects_transposed_checkpoint_in_target_orientation(self):
        module = load_example("byte_lm_mlp_lora_adapter")

        class RowIndexProjector:
            def forward(self, tensor):
                return module.st.Tensor(
                    tensor.rows,
                    tensor.cols,
                    [
                        float(row + 1)
                        for row in range(tensor.rows)
                        for _ in range(tensor.cols)
                    ],
                )

        checkpoint = {
            "lm_head.weight": module.st.Tensor(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "model.embed_tokens.weight": module.st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0]),
            "extra.weight": module.st.Tensor(1, 1, [9.0]),
        }
        rules = {
            "lm_head.weight": {"target": "head::weight", "transform": "transpose"},
            "model.embed_tokens.weight": "embed::weight",
        }
        projected = module.project_checkpoint_tensors(
            checkpoint,
            rules,
            RowIndexProjector(),
        )
        self.assertEqual(
            projected["lm_head.weight"].data(),
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(
            projected["model.embed_tokens.weight"].data(),
            [1.0, 1.0, 2.0, 2.0],
        )
        self.assertEqual(projected["extra.weight"].data(), [9.0])

    def test_checkpoint_source_gain_scales_only_mapped_tensors(self):
        helper = load_checkpoint_helper()
        checkpoint = {
            "mapped.weight": helper.st.Tensor(1, 2, [1.0, -2.0]),
            "extra.weight": helper.st.Tensor(1, 1, [9.0]),
        }
        rules = {"mapped.weight": "target::weight"}
        args = types.SimpleNamespace(checkpoint_source_gain=2.5)
        scaled = helper.apply_checkpoint_source_gain(checkpoint, rules, args)
        self.assertEqual(scaled["mapped.weight"].data(), [2.5, -5.0])
        self.assertEqual(scaled["extra.weight"].data(), [9.0])
        self.assertEqual(checkpoint["mapped.weight"].data(), [1.0, -2.0])

    def test_checkpoint_preflight_helper_selects_module_or_lora_base_load(self):
        helper = load_checkpoint_helper()
        report = {
            "compatible": True,
            "matched": 1,
            "missing": 0,
            "shape_mismatched": 0,
            "extra": 0,
        }
        load = {"matched": True}

        class FakeModule:
            def __init__(self):
                self.calls = []

            def state_dict_compatibility_with_key_map(self, checkpoint, rules):
                self.calls.append(("module_report", checkpoint, rules))
                return report

            def load_state_dict_subset_mapped_checked(self, checkpoint, rules):
                self.calls.append(("module_load", checkpoint, rules))
                return load

            def base_state_dict_compatibility_with_key_map(self, checkpoint, rules):
                self.calls.append(("base_report", checkpoint, rules))
                return report

            def load_base_from_state_dict_mapped(self, checkpoint, rules):
                self.calls.append(("base_load", checkpoint, rules))
                return load

        checkpoint = {"external.weight": object()}
        rules = {"external.weight": "layer::weight"}
        module = FakeModule()
        self.assertEqual(
            helper.preflight_and_load(
                "module",
                module,
                checkpoint,
                rules,
                emit=False,
            ),
            (report, load),
        )
        self.assertEqual([call[0] for call in module.calls], ["module_report", "module_load"])

        lora = FakeModule()
        helper.preflight_and_load(
            "lora",
            lora,
            checkpoint,
            rules,
            lora_base=True,
            emit=False,
        )
        self.assertEqual([call[0] for call in lora.calls], ["base_report", "base_load"])

    def test_checkpoint_preflight_converts_external_tensor_likes(self):
        helper = load_checkpoint_helper()
        matrix = helper.tensor_from_external([[1, 2, 3], [4, 5, 6]], name="weight")
        self.assertEqual(matrix.shape(), (2, 3))
        self.assertEqual(matrix.data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        bias = helper.tensor_from_external([0.1, -0.2], name="bias")
        self.assertEqual(bias.shape(), (1, 2))
        self.assertEqual(bias.data(), [0.1, -0.2])

    def test_checkpoint_preflight_reuses_ecosystem_external_tensor_bridge(self):
        helper = load_checkpoint_helper()
        ecosystem = sys.modules["spiraltorch.ecosystem"]

        self.assertIs(helper.tensor_from_external, ecosystem.tensor_from_external)
        self.assertIs(helper.slice_external_tensor, ecosystem.slice_external_tensor)
        self.assertIs(
            helper.checkpoint_from_external_state,
            ecosystem.checkpoint_from_external_state,
        )

    def test_checkpoint_preflight_bounds_external_tensor_likes_before_flattening(self):
        helper = load_checkpoint_helper()
        state = helper.bound_external_state_tensors(
            {
                "weight": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "bias": [0.1, 0.2, 0.3, 0.4],
                "extra": [[9.0]],
            },
            {
                "weight": (2, 2),
                "bias": (1, 3),
            },
        )

        weight = helper.tensor_from_external(state["weight"], name="weight")
        bias = helper.tensor_from_external(state["bias"], name="bias")
        self.assertEqual(weight.shape(), (2, 2))
        self.assertEqual(weight.data(), [1.0, 2.0, 4.0, 5.0])
        self.assertEqual(bias.shape(), (1, 3))
        self.assertEqual(bias.data(), [0.1, 0.2, 0.3])
        self.assertEqual(state["extra"], [[9.0]])

    def test_checkpoint_preflight_rejects_non_numeric_external_tensor_data(self):
        helper = load_checkpoint_helper()
        with self.assertRaisesRegex(TypeError, "boolean checkpoint value"):
            helper.tensor_from_external([[1.0, True]], name="weight")
        with self.assertRaisesRegex(TypeError, "non-numeric checkpoint value"):
            helper.tensor_from_external(["1.0", 2.0], name="bias")

    def test_checkpoint_preflight_converts_torch_like_state_dict(self):
        helper = load_checkpoint_helper()

        class TorchLike:
            def __init__(self, shape, data):
                self.shape = shape
                self._data = list(data)

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def reshape(self, size):
                if size != -1:
                    raise TypeError("only flat reshape is supported in this test")
                return TorchLike((len(self._data),), self._data)

            def tolist(self):
                return list(self._data)

        checkpoint = helper.checkpoint_from_external_state(
            {
                "model.embed.weight": TorchLike((2, 2), [1, 2, 3, 4]),
                "model.embed.bias": TorchLike((2,), [0.5, -0.5]),
                "model.unused": TorchLike((1,), [9.0]),
            },
            include=["model.embed.weight", "model.embed.bias"],
        )
        self.assertEqual(sorted(checkpoint), ["model.embed.bias", "model.embed.weight"])
        self.assertEqual(checkpoint["model.embed.weight"].shape(), (2, 2))
        self.assertEqual(checkpoint["model.embed.bias"].shape(), (1, 2))

    def test_checkpoint_preflight_builds_hf_lm_key_rules(self):
        helper = load_checkpoint_helper()
        rules = helper.hf_lm_key_rules()
        self.assertEqual(rules["transformer.wte.weight"], "embed::weight")
        self.assertEqual(rules["transformer.wte.bias"], "embed::bias")
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )
        self.assertEqual(
            rules["lm_head.bias"],
            {"target": "head::bias", "transform": "copy_overlap_zeros"},
        )

        no_bias_rules = helper.hf_lm_key_rules(
            embed_bias_key=None,
            lm_head_bias_key=None,
            lm_head_weight_transform="identity",
        )
        self.assertNotIn("transformer.wte.bias", no_bias_rules)
        self.assertNotIn("lm_head.bias", no_bias_rules)
        self.assertEqual(
            no_bias_rules["lm_head.weight"],
            {"target": "head::weight", "transform": "identity"},
        )

        resize_rules = helper.hf_lm_key_rules(**helper.hf_lm_overlap_resize_kwargs())
        self.assertEqual(
            resize_rules["transformer.wte.weight"],
            {"target": "embed::weight", "transform": "copy_overlap_zeros"},
        )
        self.assertEqual(
            resize_rules["transformer.wte.bias"],
            {"target": "embed::bias", "transform": "copy_overlap_zeros"},
        )
        self.assertEqual(
            resize_rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose_copy_overlap_zeros"},
        )

    def test_checkpoint_preflight_builds_hf_lm_key_presets(self):
        helper = load_checkpoint_helper()
        llama = helper.hf_lm_key_preset("llama")
        self.assertEqual(llama["embed_weight_key"], "model.embed_tokens.weight")
        self.assertEqual(llama["lm_head_weight_key"], "lm_head.weight")

        gpt2_bare = helper.hf_lm_key_preset("gpt2_bare")
        self.assertEqual(gpt2_bare["embed_weight_key"], "wte.weight")

        gemma = helper.hf_lm_key_preset("gemma")
        self.assertEqual(
            gemma["embed_weight_key"],
            "model.language_model.embed_tokens.weight",
        )

        gpt_neox = helper.hf_lm_key_preset("gpt_neox")
        self.assertEqual(gpt_neox["embed_weight_key"], "gpt_neox.embed_in.weight")
        self.assertEqual(gpt_neox["lm_head_weight_key"], "embed_out.weight")

        with self.assertRaisesRegex(ValueError, "unsupported HF key preset"):
            helper.hf_lm_key_preset("mystery")

    def test_checkpoint_preflight_detects_tied_lm_head_from_embed_only_layout(self):
        helper = load_checkpoint_helper()
        shape_state = {
            "model.embed_tokens.weight": (320, 32),
        }

        self.assertEqual(helper.detect_hf_lm_key_preset(shape_state), "llama")
        self.assertTrue(
            helper.hf_lm_uses_tied_head_weight(shape_state, key_preset="llama")
        )
        self.assertEqual(
            helper.infer_hf_lm_module_shapes(shape_state, key_preset="llama"),
            (320, 32, 320),
        )

    def test_checkpoint_preflight_ties_missing_external_lm_head_weight_to_embed(self):
        helper = load_checkpoint_helper()
        external_state = {
            "model.embed_tokens.weight": [[1, 2], [3, 4], [5, 6], [7, 8]],
        }

        checkpoint, rules = helper.hf_lm_handoff_from_external_state(
            external_state,
            key_preset="llama",
        )

        self.assertEqual(checkpoint["model.embed_tokens.weight"].shape(), (4, 2))
        self.assertEqual(checkpoint["lm_head.weight"].shape(), (4, 2))
        self.assertEqual(
            checkpoint["lm_head.weight"].data(),
            checkpoint["model.embed_tokens.weight"].data(),
        )
        self.assertEqual(checkpoint["model.embed_tokens.bias"].shape(), (1, 2))
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, 4))
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

    def test_checkpoint_preflight_externalizes_hf_lm_handoff(self):
        helper = load_checkpoint_helper()
        state = {
            "embed::weight": helper.st.Tensor(4, 2, range(8)),
            "embed::bias": helper.st.Tensor(1, 2, [0.1, -0.1]),
            "head::weight": helper.st.Tensor(2, 3, range(6)),
            "head::bias": helper.st.Tensor(1, 3, [0.2, -0.2, 0.4]),
        }
        checkpoint, rules = helper.hf_lm_handoff_from_spiraltorch_state(state)
        self.assertEqual(
            sorted(checkpoint),
            [
                "lm_head.bias",
                "lm_head.weight",
                "transformer.wte.bias",
                "transformer.wte.weight",
            ],
        )
        self.assertEqual(checkpoint["transformer.wte.weight"].shape(), (4, 2))
        self.assertEqual(checkpoint["lm_head.weight"].shape(), (3, 2))
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

        raw_layout_checkpoint, raw_layout_rules = (
            helper.hf_lm_handoff_from_spiraltorch_state(
                state,
                transpose_lm_head_weight=False,
            )
        )
        self.assertEqual(raw_layout_checkpoint["lm_head.weight"].shape(), (2, 3))
        self.assertEqual(
            raw_layout_rules["lm_head.weight"],
            {"target": "head::weight", "transform": "identity"},
        )

        identity_checkpoint, identity_rules = (
            helper.hf_lm_handoff_from_spiraltorch_state(
                state,
                lm_head_weight_transform="identity",
            )
        )
        self.assertEqual(identity_checkpoint["lm_head.weight"].shape(), (2, 3))
        self.assertEqual(
            identity_rules["lm_head.weight"],
            {"target": "head::weight", "transform": "identity"},
        )

        with self.assertRaisesRegex(ValueError, "transpose_lm_head_weight"):
            helper.hf_lm_handoff_from_spiraltorch_state(
                state,
                transpose_lm_head_weight=False,
                lm_head_weight_transform="transpose",
            )

        with self.assertRaisesRegex(TypeError, "unsupported HF LM handoff option"):
            helper.hf_lm_handoff_from_spiraltorch_state(state, surprise=True)

    def test_checkpoint_preflight_externalizes_hf_lm_handoff_with_preset(self):
        helper = load_checkpoint_helper()
        state = {
            "embed::weight": helper.st.Tensor(4, 2, range(8)),
            "head::weight": helper.st.Tensor(2, 3, range(6)),
        }
        checkpoint, rules = helper.hf_lm_handoff_from_spiraltorch_state(
            state,
            key_preset="llama",
            embed_bias_source=None,
            lm_head_bias_source=None,
        )
        self.assertEqual(
            sorted(checkpoint),
            [
                "lm_head.bias",
                "lm_head.weight",
                "model.embed_tokens.bias",
                "model.embed_tokens.weight",
            ],
        )
        self.assertEqual(checkpoint["model.embed_tokens.bias"].data(), [0.0, 0.0])
        self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

    def test_checkpoint_preflight_imports_external_hf_state_with_preset(self):
        helper = load_checkpoint_helper()
        external_state = {
            "model.embed_tokens.weight": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "lm_head.weight": [[1, 2], [3, 4], [5, 6]],
            "model.layers.0.input_layernorm.weight": [[1.0, 1.0]],
            "unused.deep.weight": [[9.0]],
        }
        checkpoint, rules = helper.hf_lm_handoff_from_external_state(
            external_state,
            key_preset="llama",
            include_extra_keys=["model.layers.0.input_layernorm.weight"],
        )
        self.assertEqual(
            sorted(checkpoint),
            [
                "lm_head.bias",
                "lm_head.weight",
                "model.embed_tokens.bias",
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
            ],
        )
        self.assertEqual(checkpoint["model.embed_tokens.bias"].data(), [0.0, 0.0])
        self.assertEqual(checkpoint["lm_head.bias"].data(), [0.0, 0.0, 0.0])
        self.assertNotIn("unused.deep.weight", checkpoint)
        self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "transpose"},
        )

    def test_checkpoint_preflight_shape_audit_flags_resize_without_modules(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            key_preset="llama",
            include_extra_keys=["model.layers.0.input_layernorm.weight"],
            no_synthesize_missing_biases=False,
            allow_overlap_resize=False,
            vocab=256,
            hidden=24,
            target_classes=256,
            checkpoint_projection="none",
            checkpoint_projection_preset="healthy",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        row = helper.hf_lm_shape_audit_row(
            args,
            "hf-state-dict:llama",
            ["pytorch_model.bin"],
            {
                "model.embed_tokens.weight": (320, 32),
                "lm_head.weight": (300, 32),
                "model.layers.0.input_layernorm.weight": (1, 2),
            },
        )
        self.assertEqual(row["row_type"], "shape_audit")
        self.assertEqual(row["checkpoint_vocab"], 320)
        self.assertEqual(row["checkpoint_hidden"], 32)
        self.assertEqual(row["checkpoint_target_classes"], 300)
        self.assertFalse(row["exact_shape_match"])
        self.assertTrue(row["overlap_resize_required"])
        self.assertFalse(row["can_materialize_requested"])
        self.assertEqual(row["missing_required_keys"], "none")
        self.assertEqual(row["present_extra_keys"], "model.layers.0.input_layernorm.weight")
        self.assertTrue(row["embed_bias_synthesized"])
        self.assertTrue(row["lm_head_bias_synthesized"])
        self.assertEqual(row["checkpoint_projection"], "zspace")
        self.assertEqual(row["checkpoint_projection_strength"], 1.0)
        self.assertEqual(row["checkpoint_projection_curvature"], -0.04)

    def test_checkpoint_preflight_transformers_audit_imports_runtime_metadata(self):
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                transformers_audit=True,
                transformers_model_path=Path(tmp),
                hf_state_dict=None,
                allow_transformers_remote=False,
                transformers_trust_remote_code=True,
                transformers_revision="main",
                skip_transformers_tokenizer=False,
                transformers_load_model=True,
                runtime_import_presets=["transformers"],
                runtime_imports=["math"],
                require_runtime_imports=True,
                required_runtime_imports=["math"],
                required_runtime_import_presets=["transformers"],
            )
            with fake_transformers_module() as (_fake, calls):
                fields = helper.transformers_runtime_audit_fields(args, (320, 32, 320))

        self.assertEqual(fields["transformers_audit_status"], "ok")
        self.assertTrue(fields["transformers_available"])
        self.assertEqual(fields["transformers_version"], "9.9.9")
        self.assertEqual(fields["transformers_model_type"], "llama")
        self.assertEqual(fields["transformers_architectures"], "LlamaForCausalLM")
        self.assertEqual(fields["transformers_config_vocab_size"], 320)
        self.assertEqual(fields["transformers_config_hidden_size"], 32)
        self.assertTrue(fields["transformers_config_vocab_matches_checkpoint"])
        self.assertTrue(fields["transformers_config_hidden_matches_checkpoint"])
        self.assertTrue(fields["transformers_config_lm_head_matches_checkpoint"])
        self.assertEqual(fields["transformers_tokenizer_class"], "FakeTokenizer")
        self.assertEqual(fields["transformers_tokenizer_vocab_size"], 320)
        self.assertEqual(fields["transformers_model_class"], "FakeModel")
        self.assertEqual(fields["transformers_model_parameter_count"], 7)
        self.assertEqual(calls["config"][0][1]["local_files_only"], True)
        self.assertEqual(calls["config"][0][1]["trust_remote_code"], True)
        self.assertEqual(calls["config"][0][1]["revision"], "main")
        self.assertEqual(calls["model"][0][1]["local_files_only"], True)
        self.assertEqual(fields["runtime_import_presets"], "transformers")
        self.assertEqual(
            fields["runtime_import_preset_modules"],
            "transformers=transformers",
        )
        self.assertEqual(fields["runtime_imports_requested"], "transformers,math")
        self.assertEqual(fields["runtime_import_probe_count"], 2)
        self.assertEqual(fields["runtime_imports_imported"], "transformers,math")
        self.assertEqual(fields["runtime_imports_failed"], "none")
        self.assertTrue(fields["runtime_imports_all_ok"])
        self.assertEqual(fields["runtime_import_coimport_status"], "ok")
        self.assertTrue(fields["runtime_imports_coimported"])
        self.assertEqual(
            fields["runtime_import_coimport_modules"],
            "transformers,math",
        )
        self.assertEqual(fields["runtime_import_coimport_missing_modules"], "none")
        self.assertEqual(fields["required_runtime_imports"], "math")
        self.assertTrue(fields["required_runtime_imports_passed"])
        self.assertEqual(fields["required_runtime_import_presets"], "transformers")
        self.assertTrue(fields["required_runtime_import_presets_passed"])

    def test_checkpoint_preflight_transformers_runtime_contract_preset_expands_audit(self):
        helper = load_checkpoint_helper()
        old_argv = sys.argv
        sys.argv = [
            "checkpoint_preflight.py",
            "--hf-state-dict",
            "/models/llama",
            "--transformers-runtime-contract-preset",
            "hf-runtime",
        ]
        try:
            args = helper.parse_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.transformers_audit)
        self.assertEqual(args.runtime_import_presets, ["hf-runtime"])
        self.assertEqual(args.required_runtime_import_presets, ["hf-runtime"])
        self.assertTrue(args.require_runtime_imports)

    def test_checkpoint_preflight_transformers_audit_gate_rejects_partial_runtime(self):
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                key_preset="auto",
                include_extra_keys=[],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=False,
                vocab=None,
                hidden=None,
                target_classes=None,
                checkpoint_projection="none",
                checkpoint_projection_preset=None,
                checkpoint_projection_strength=0.5,
                checkpoint_projection_curvature=-0.5,
                checkpoint_projection_frequency=0.65,
                transformers_audit=True,
                transformers_model_path=Path(tmp),
                hf_state_dict=None,
                allow_transformers_remote=False,
                transformers_trust_remote_code=False,
                transformers_revision=None,
                skip_transformers_tokenizer=False,
                transformers_load_model=False,
                require_transformers_audit=True,
            )
            with fake_transformers_module(tokenizer_error=True):
                row = helper.hf_lm_shape_audit_row(
                    args,
                    "hf-state-dict:auto",
                    ["model.safetensors"],
                    {
                        "model.embed_tokens.weight": (320, 32),
                        "lm_head.weight": (320, 32),
                    },
                )

        self.assertEqual(row["transformers_audit_status"], "tokenizer_error")
        self.assertTrue(row["transformers_config_loaded"])
        self.assertFalse(row["transformers_tokenizer_loaded"])
        self.assertIn("tokenizer fixture missing", row["transformers_tokenizer_error"])
        with self.assertRaisesRegex(RuntimeError, "Transformers audit gate failed"):
            helper.check_transformers_audit_gate(row, args)

    def test_checkpoint_preflight_transformers_runtime_import_gate_rejects_missing_module(self):
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                transformers_audit=True,
                transformers_model_path=Path(tmp),
                hf_state_dict=None,
                allow_transformers_remote=False,
                transformers_trust_remote_code=False,
                transformers_revision=None,
                skip_transformers_tokenizer=True,
                transformers_load_model=False,
                runtime_import_presets=[],
                runtime_imports=[],
                require_runtime_imports=False,
                required_runtime_imports=["spiraltorch_missing_runtime_fixture"],
                required_runtime_import_presets=[],
            )
            with fake_transformers_module():
                row = helper.transformers_runtime_audit_fields(args, (320, 32, 320))

        self.assertEqual(row["transformers_audit_status"], "ok")
        self.assertEqual(
            row["runtime_imports_requested"],
            "spiraltorch_missing_runtime_fixture",
        )
        self.assertEqual(
            row["required_runtime_imports_missing"],
            "spiraltorch_missing_runtime_fixture",
        )
        self.assertFalse(row["required_runtime_imports_passed"])
        with self.assertRaisesRegex(
            RuntimeError,
            "Transformers runtime import gate failed",
        ):
            helper.check_transformers_runtime_import_gate(row, args)

    def test_transformers_trace_records_prompt_logits_and_hidden_state(self):
        module = load_example("byte_lm_transformers_trace")
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                model_path=Path(tmp),
                top_k=2,
                allow_remote=False,
                trust_remote_code=False,
                revision=None,
                metadata_only=False,
                capture_hidden_states=True,
                zspace_project=False,
                zspace_source="hidden",
                zspace_curvature=-0.04,
                zspace_frequency=0.65,
                zspace_strength=1.0,
                runtime_import_presets=["transformers"],
                runtime_imports=["math"],
                require_runtime_imports=True,
                require_hidden_states=False,
                require_zspace_projection=False,
            )
            with fake_transformers_module() as (fake, _calls):
                config = fake.AutoConfig.from_pretrained(str(args.model_path))
                tokenizer = fake.AutoTokenizer.from_pretrained(str(args.model_path))
                model = fake.AutoModelForCausalLM.from_pretrained(str(args.model_path))
                manifest = module.manifest_row(
                    args,
                    ["spiral"],
                    fake,
                    config,
                    tokenizer,
                    model_loaded=True,
                    model=model,
                )
                row = module.trace_prompt(args, tokenizer, model, "spiral", 0)

        self.assertEqual(manifest["row_type"], "transformers_trace_manifest")
        self.assertTrue(manifest["spiraltorch_imported"])
        self.assertEqual(manifest["spiraltorch_module_name"], "spiraltorch")
        self.assertTrue(manifest["transformers_imported"])
        self.assertEqual(manifest["transformers_module_name"], "transformers")
        self.assertEqual(
            manifest["transformers_spiraltorch_coimport_status"],
            "ok",
        )
        self.assertEqual(manifest["runtime_import_presets"], "transformers")
        self.assertEqual(
            manifest["runtime_import_preset_modules"],
            "transformers=transformers",
        )
        self.assertEqual(manifest["runtime_import_presets_satisfied"], "transformers")
        self.assertEqual(manifest["runtime_import_presets_failed"], "none")
        self.assertEqual(manifest["runtime_import_preset_missing_modules"], "none")
        self.assertEqual(manifest["runtime_import_probe_count"], 2)
        self.assertEqual(manifest["runtime_imports_requested"], "transformers,math")
        self.assertEqual(manifest["runtime_imports_imported"], "transformers,math")
        self.assertEqual(manifest["runtime_imports_failed"], "none")
        self.assertTrue(manifest["runtime_imports_all_ok"])
        self.assertEqual(manifest["runtime_import_coimport_status"], "ok")
        self.assertTrue(manifest["runtime_imports_coimported"])
        self.assertEqual(
            manifest["runtime_import_coimport_modules"],
            "transformers,math",
        )
        self.assertEqual(manifest["runtime_import_coimport_missing_modules"], "none")
        self.assertEqual(
            manifest["runtime_import_versions"],
            "transformers=9.9.9,math=none",
        )
        self.assertEqual(
            manifest["runtime_import_module_names"],
            "transformers=transformers,math=math",
        )
        probe_rows = json.loads(manifest["runtime_imports_json"])
        self.assertEqual(
            [row["module"] for row in probe_rows],
            ["transformers", "math"],
        )
        self.assertTrue(all(row["imported"] for row in probe_rows))
        preset_status_rows = json.loads(manifest["runtime_import_preset_status_json"])
        self.assertEqual(
            preset_status_rows,
            [
                {
                    "preset": "transformers",
                    "modules": ["transformers"],
                    "imported": ["transformers"],
                    "missing": [],
                    "passed": True,
                }
            ],
        )
        self.assertEqual(manifest["transformers_model_type"], "llama")
        self.assertEqual(manifest["transformers_config_num_hidden_layers"], 2)
        self.assertEqual(manifest["transformers_config_num_attention_heads"], 4)
        self.assertEqual(
            manifest["transformers_config_max_position_embeddings"],
            2048,
        )
        self.assertEqual(manifest["transformers_tokenizer_class"], "FakeTokenizer")
        self.assertEqual(manifest["transformers_tokenizer_vocab_size"], 320)
        self.assertEqual(manifest["transformers_tokenizer_len"], 320)
        self.assertEqual(manifest["transformers_model_class"], "FakeModel")
        self.assertEqual(manifest["transformers_model_parameter_count"], 7)
        self.assertEqual(row["row_type"], "transformers_prompt_trace")
        self.assertEqual(row["input_token_count"], 3)
        self.assertEqual(row["logit_vocab_size"], 4)
        self.assertEqual(row["top_token_ids"], "3,1")
        self.assertEqual(json.loads(row["top_token_texts"]), ["<tok:3>", "<tok:1>"])
        self.assertTrue(row["top_probability_sum"] > 0.5)
        self.assertTrue(row["hidden_state_available"])
        self.assertEqual(row["hidden_state_shape"], "1x1x3")
        self.assertEqual(row["hidden_state_dims"], 3)
        self.assertTrue(row["input_ids_tensor_available"])
        self.assertEqual(row["input_ids_tensor_backend"], "python_sequence")
        self.assertEqual(row["input_ids_tensor_shape"], "1x3")
        self.assertEqual(row["input_ids_tensor_shape_rank"], 2)
        self.assertEqual(row["logits_tensor_backend"], "python_sequence")
        self.assertEqual(row["logits_tensor_shape"], "1x1x4")
        self.assertEqual(row["logits_tensor_shape_rank"], 3)
        self.assertTrue(row["hidden_state_tensor_available"])
        self.assertEqual(row["hidden_state_tensor_backend"], "python_sequence")
        self.assertEqual(row["hidden_state_tensor_shape"], "1x1x3")
        self.assertEqual(row["hidden_state_tensor_shape_rank"], 3)
        self.assertIsNone(row["hidden_state_tensor_device"])
        self.assertFalse(row["zspace_projection_requested"])

    def test_transformers_trace_cli_writes_jsonl_without_real_transformers(self):
        module = load_example("byte_lm_transformers_trace")
        old_argv = sys.argv
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            sys.argv = [
                "byte_lm_transformers_trace.py",
                "--model-path",
                tmp,
                "--prompt",
                "spiral",
                "--top-k",
                "2",
                "--jsonl",
                str(out),
                "--runtime-contract-preset",
                "transformers",
            ]
            output = io.StringIO()
            try:
                with fake_transformers_module(), contextlib.redirect_stdout(output):
                    module.main()
            finally:
                sys.argv = old_argv
            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["row_type"] for row in rows], [
            "transformers_trace_manifest",
            "transformers_prompt_trace",
        ])
        self.assertTrue(rows[0]["model_loaded"])
        self.assertEqual(rows[0]["runtime_import_presets"], "transformers")
        self.assertEqual(rows[0]["runtime_import_presets_satisfied"], "transformers")
        self.assertEqual(rows[0]["runtime_import_presets_failed"], "none")
        self.assertEqual(rows[0]["runtime_imports_requested"], "transformers")
        self.assertTrue(rows[0]["runtime_imports_all_ok"])
        self.assertEqual(rows[0]["runtime_import_coimport_status"], "ok")
        self.assertTrue(rows[0]["runtime_imports_coimported"])
        self.assertEqual(
            rows[0]["required_runtime_import_presets"],
            "transformers",
        )
        self.assertTrue(rows[0]["required_runtime_import_presets_passed"])
        self.assertEqual(rows[1]["top_token_ids"], "3,1")
        self.assertIn("transformers_prompt_trace", output.getvalue())

    def test_transformers_trace_reuses_ecosystem_external_tensor_bridge(self):
        module = load_example("byte_lm_transformers_trace")
        ecosystem = sys.modules["spiraltorch.ecosystem"]

        self.assertIs(module.external_tensor_shape, ecosystem.external_tensor_shape)
        self.assertIs(
            module.external_tensor_last_token,
            ecosystem.external_tensor_last_token,
        )
        self.assertIs(
            module.external_tensor_to_list,
            ecosystem.external_tensor_to_list,
        )
        self.assertIs(
            module.external_tensor_metadata,
            ecosystem.external_tensor_metadata,
        )

    def test_transformers_trace_runtime_contract_preset_expands_direct_gates(self):
        module = load_example("byte_lm_transformers_trace")
        old_argv = sys.argv
        sys.argv = [
            "byte_lm_transformers_trace.py",
            "--model-path",
            "/models/llama",
            "--runtime-contract-preset",
            "hf-runtime",
        ]
        try:
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.runtime_import_presets, ["hf-runtime"])
        self.assertEqual(args.required_runtime_import_presets, ["hf-runtime"])
        self.assertTrue(args.require_runtime_imports)
        self.assertFalse(args.require_runtime_metadata_match)

        sys.argv = [
            "byte_lm_transformers_trace.py",
            "--model-path",
            "/models/llama",
            "--compare-jsonl",
            "/tmp/baseline-transformers-trace.jsonl",
            "--runtime-contract-preset",
            "hf-runtime",
        ]
        try:
            compare_args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(compare_args.runtime_import_presets, ["hf-runtime"])
        self.assertEqual(compare_args.required_runtime_import_presets, ["hf-runtime"])
        self.assertTrue(compare_args.require_runtime_imports)
        self.assertTrue(compare_args.require_runtime_metadata_match)

    def test_transformers_trace_runtime_import_preset_status_tracks_missing_modules(self):
        module = load_example("byte_lm_transformers_trace")
        original = dict(module.RUNTIME_IMPORT_PRESETS)
        try:
            module.RUNTIME_IMPORT_PRESETS["fixture-runtime"] = [
                "transformers",
                "spiraltorch_missing_runtime_fixture",
            ]
            args = types.SimpleNamespace(
                runtime_import_presets=["fixture-runtime"],
                runtime_imports=[],
            )
            with fake_transformers_module():
                fields = module.runtime_import_fields(args)
        finally:
            module.RUNTIME_IMPORT_PRESETS.clear()
            module.RUNTIME_IMPORT_PRESETS.update(original)

        self.assertEqual(fields["runtime_import_presets"], "fixture-runtime")
        self.assertEqual(
            fields["runtime_import_preset_modules"],
            "fixture-runtime=transformers|spiraltorch_missing_runtime_fixture",
        )
        self.assertEqual(fields["runtime_import_presets_satisfied"], "none")
        self.assertEqual(fields["runtime_import_presets_failed"], "fixture-runtime")
        self.assertEqual(
            fields["runtime_import_preset_missing_modules"],
            "fixture-runtime=spiraltorch_missing_runtime_fixture",
        )
        self.assertFalse(fields["runtime_imports_all_ok"])
        self.assertEqual(fields["runtime_import_coimport_status"], "missing")
        self.assertFalse(fields["runtime_imports_coimported"])
        self.assertEqual(fields["runtime_import_coimport_modules"], "transformers")
        self.assertEqual(
            fields["runtime_import_coimport_missing_modules"],
            "spiraltorch_missing_runtime_fixture",
        )
        status_rows = json.loads(fields["runtime_import_preset_status_json"])
        self.assertEqual(status_rows[0]["preset"], "fixture-runtime")
        self.assertEqual(status_rows[0]["imported"], ["transformers"])
        self.assertEqual(
            status_rows[0]["missing"],
            ["spiraltorch_missing_runtime_fixture"],
        )
        self.assertFalse(status_rows[0]["passed"])

    def test_transformers_trace_runtime_import_gate_rejects_missing_module(self):
        module = load_example("byte_lm_transformers_trace")
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                model_path=Path(tmp),
                top_k=2,
                allow_remote=False,
                trust_remote_code=False,
                revision=None,
                metadata_only=True,
                capture_hidden_states=True,
                zspace_project=False,
                zspace_source="hidden",
                zspace_curvature=-0.04,
                zspace_frequency=0.65,
                zspace_strength=1.0,
                runtime_imports=["spiraltorch_missing_runtime_fixture"],
                require_runtime_imports=True,
            )
            with fake_transformers_module() as (fake, _calls):
                config = fake.AutoConfig.from_pretrained(str(args.model_path))
                tokenizer = fake.AutoTokenizer.from_pretrained(str(args.model_path))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "runtime import probe failed",
                ):
                    module.manifest_row(
                        args,
                        ["spiral"],
                        fake,
                        config,
                        tokenizer,
                        model_loaded=False,
                    )

    def test_transformers_trace_required_runtime_imports_probe_and_record(self):
        module = load_example("byte_lm_transformers_trace")
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                model_path=Path(tmp),
                top_k=2,
                allow_remote=False,
                trust_remote_code=False,
                revision=None,
                metadata_only=True,
                capture_hidden_states=True,
                zspace_project=False,
                zspace_source="hidden",
                zspace_curvature=-0.04,
                zspace_frequency=0.65,
                zspace_strength=1.0,
                runtime_import_presets=[],
                runtime_imports=[],
                required_runtime_imports=["math"],
                required_runtime_import_presets=["transformers"],
                require_runtime_imports=False,
            )
            with fake_transformers_module() as (fake, _calls):
                config = fake.AutoConfig.from_pretrained(str(args.model_path))
                tokenizer = fake.AutoTokenizer.from_pretrained(str(args.model_path))
                manifest = module.manifest_row(
                    args,
                    ["spiral"],
                    fake,
                    config,
                    tokenizer,
                    model_loaded=False,
                )

        self.assertEqual(manifest["runtime_import_presets"], "transformers")
        self.assertEqual(manifest["runtime_imports_requested"], "transformers,math")
        self.assertEqual(manifest["required_runtime_imports"], "math")
        self.assertEqual(
            manifest["required_runtime_imports_imported"],
            "transformers,math",
        )
        self.assertEqual(manifest["required_runtime_imports_missing"], "none")
        self.assertTrue(manifest["required_runtime_imports_passed"])
        self.assertEqual(
            manifest["required_runtime_import_presets"],
            "transformers",
        )
        self.assertEqual(
            manifest["required_runtime_import_presets_observed"],
            "transformers",
        )
        self.assertEqual(
            manifest["required_runtime_import_presets_satisfied"],
            "transformers",
        )
        self.assertEqual(
            manifest["required_runtime_import_presets_unsatisfied"],
            "none",
        )
        self.assertTrue(manifest["required_runtime_import_presets_passed"])

    def test_transformers_trace_required_runtime_import_preset_rejects_unsatisfied(self):
        module = load_example("byte_lm_transformers_trace")
        original = dict(module.RUNTIME_IMPORT_PRESETS)
        try:
            module.RUNTIME_IMPORT_PRESETS["fixture-runtime"] = [
                "transformers",
                "spiraltorch_missing_runtime_fixture",
            ]
            with tempfile.TemporaryDirectory() as tmp:
                args = types.SimpleNamespace(
                    model_path=Path(tmp),
                    top_k=2,
                    allow_remote=False,
                    trust_remote_code=False,
                    revision=None,
                    metadata_only=True,
                    capture_hidden_states=True,
                    zspace_project=False,
                    zspace_source="hidden",
                    zspace_curvature=-0.04,
                    zspace_frequency=0.65,
                    zspace_strength=1.0,
                    runtime_import_presets=[],
                    runtime_imports=[],
                    required_runtime_imports=[],
                    required_runtime_import_presets=["fixture-runtime"],
                    require_runtime_imports=False,
                )
                with fake_transformers_module() as (fake, _calls):
                    config = fake.AutoConfig.from_pretrained(str(args.model_path))
                    tokenizer = fake.AutoTokenizer.from_pretrained(
                        str(args.model_path)
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "runtime_import_preset_unsatisfied:fixture-runtime",
                    ):
                        module.manifest_row(
                            args,
                            ["spiral"],
                            fake,
                            config,
                            tokenizer,
                            model_loaded=False,
                        )
        finally:
            module.RUNTIME_IMPORT_PRESETS.clear()
            module.RUNTIME_IMPORT_PRESETS.update(original)

    def test_transformers_trace_compares_prompt_rows_with_gates(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=True,
            require_top_token_match=True,
            max_top_logit_regression=0.0,
            max_top_probability_regression=0.0,
            max_logit_l2_change=0.0,
            max_hidden_state_l2_change=0.0,
        )
        baseline = [
            {
                "row_type": "transformers_prompt_trace",
                "prompt_index": 0,
                "prompt": "spiral",
                "top_token_ids": "3,1",
                "top_logits": "1.1,0.4",
                "top_probabilities": "0.5,0.2",
                "logit_l2": 1.2,
                "hidden_state_l2": 0.9,
                "zspace_projection_status": "not_requested",
            }
        ]
        rows = module.compare_trace_rows(list(baseline), baseline, args)

        self.assertEqual(rows[0]["row_type"], "transformers_trace_compare_summary")
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[0]["failures"], 0)
        self.assertEqual(rows[0]["top_token_changed_rows"], 0)
        self.assertEqual(rows[0]["zspace_status_changed_rows"], 0)
        self.assertEqual(rows[0]["observed_max_top_logit_regression"], 0.0)
        self.assertEqual(rows[0]["observed_max_top_probability_regression"], 0.0)
        self.assertEqual(rows[0]["observed_max_logit_l2_change"], 0.0)
        self.assertEqual(rows[0]["observed_max_hidden_state_l2_change"], 0.0)
        self.assertTrue(rows[1]["passed"])

    def test_transformers_trace_compare_gate_detects_top_token_drift(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=True,
            require_runtime_metadata_match=False,
            require_top_token_match=True,
            max_top_logit_regression=None,
            max_top_probability_regression=None,
            max_logit_l2_change=None,
            max_hidden_state_l2_change=None,
        )
        baseline = [
            {
                "row_type": "transformers_prompt_trace",
                "prompt_index": 0,
                "prompt": "spiral",
                "top_token_ids": "3,1",
                "top_logits": "1.1,0.4",
                "top_probabilities": "0.5,0.2",
                "logit_l2": 1.2,
                "hidden_state_l2": 0.9,
            }
        ]
        current = [dict(baseline[0], top_token_ids="1,3")]
        rows = module.compare_trace_rows(current, baseline, args)

        self.assertFalse(rows[0]["passed"])
        self.assertFalse(rows[1]["passed"])
        self.assertIn("top_token_changed", rows[1]["failures"])

    def test_transformers_trace_compare_gate_detects_runtime_metadata_drift(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=False,
            require_runtime_metadata_match=True,
            require_top_token_match=False,
            max_top_logit_regression=None,
            max_top_probability_regression=None,
            max_logit_l2_change=None,
            max_hidden_state_l2_change=None,
        )
        manifest = {
            "row_type": "transformers_trace_manifest",
            "model_path": "/models/llama",
            "top_k": 2,
            "transformers_model_type": "llama",
            "transformers_tokenizer_class": "FakeTokenizer",
            "transformers_tokenizer_vocab_size": 320,
            "transformers_config_hidden_size": 32,
        }
        prompt = {
            "row_type": "transformers_prompt_trace",
            "prompt_index": 0,
            "prompt": "spiral",
            "top_token_ids": "3,1",
        }
        current = [
            dict(
                manifest,
                transformers_tokenizer_class="OtherTokenizer",
            ),
            prompt,
        ]
        rows = module.compare_trace_rows(current, [manifest, prompt], args)

        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["runtime_metadata_changed_count"], 1)
        self.assertEqual(
            rows[0]["runtime_metadata_changed_fields"],
            "transformers_tokenizer_class",
        )
        self.assertEqual(
            rows[0]["runtime_metadata_failures"],
            "runtime_metadata_changed",
        )
        self.assertTrue(rows[1]["passed"])

    def test_transformers_trace_compare_gate_detects_import_context_drift(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=False,
            require_runtime_metadata_match=True,
            require_top_token_match=False,
            max_top_logit_regression=None,
            max_top_probability_regression=None,
            max_logit_l2_change=None,
            max_hidden_state_l2_change=None,
        )
        manifest = {
            "row_type": "transformers_trace_manifest",
            "model_path": "/models/llama",
            "top_k": 2,
            "spiraltorch_imported": True,
            "spiraltorch_version": "0.1.0",
            "spiraltorch_module_name": "spiraltorch",
            "transformers_imported": True,
            "transformers_module_name": "transformers",
            "transformers_spiraltorch_coimport_status": "ok",
        }
        prompt = {
            "row_type": "transformers_prompt_trace",
            "prompt_index": 0,
            "prompt": "spiral",
            "top_token_ids": "3,1",
        }
        current = [
            dict(
                manifest,
                transformers_spiraltorch_coimport_status="transformers_missing",
            ),
            prompt,
        ]
        rows = module.compare_trace_rows(current, [manifest, prompt], args)

        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["runtime_metadata_changed_count"], 1)
        self.assertEqual(
            rows[0]["runtime_metadata_changed_fields"],
            "transformers_spiraltorch_coimport_status",
        )
        self.assertEqual(
            rows[0]["runtime_metadata_failures"],
            "runtime_metadata_changed",
        )
        self.assertTrue(rows[1]["passed"])

    def test_transformers_trace_compare_gate_detects_runtime_import_drift(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=False,
            require_runtime_metadata_match=True,
            require_top_token_match=False,
            max_top_logit_regression=None,
            max_top_probability_regression=None,
            max_logit_l2_change=None,
            max_hidden_state_l2_change=None,
        )
        manifest = {
            "row_type": "transformers_trace_manifest",
            "model_path": "/models/llama",
            "top_k": 2,
            "runtime_imports_requested": "torch",
            "runtime_import_probe_count": 1,
            "runtime_imports_imported": "torch",
            "runtime_imports_failed": "none",
            "runtime_imports_all_ok": True,
            "runtime_import_coimport_status": "ok",
            "runtime_imports_coimported": True,
            "runtime_import_coimport_modules": "torch",
            "runtime_import_coimport_missing_modules": "none",
            "runtime_import_versions": "torch=2.0.0",
            "runtime_import_module_names": "torch=torch",
            "runtime_imports_json": json.dumps(
                [
                    {
                        "module": "torch",
                        "imported": True,
                        "version": "2.0.0",
                        "module_name": "torch",
                        "module_file": "/env/torch.py",
                        "error": None,
                    }
                ],
                sort_keys=True,
            ),
        }
        prompt = {
            "row_type": "transformers_prompt_trace",
            "prompt_index": 0,
            "prompt": "spiral",
            "top_token_ids": "3,1",
        }
        current = [
            dict(
                manifest,
                runtime_imports_imported="none",
                runtime_imports_failed="torch",
                runtime_imports_all_ok=False,
                runtime_import_coimport_status="missing",
                runtime_imports_coimported=False,
                runtime_import_coimport_modules="none",
                runtime_import_coimport_missing_modules="torch",
            ),
            prompt,
        ]
        rows = module.compare_trace_rows(current, [manifest, prompt], args)

        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["runtime_metadata_changed_count"], 7)
        self.assertEqual(
            rows[0]["runtime_metadata_failures"],
            "runtime_metadata_changed",
        )
        self.assertIn(
            "runtime_imports_all_ok",
            rows[0]["runtime_metadata_changed_fields"],
        )
        self.assertIn(
            "runtime_import_coimport_status",
            rows[0]["runtime_metadata_changed_fields"],
        )
        self.assertIn(
            "runtime_import_coimport_missing_modules",
            rows[0]["runtime_metadata_changed_fields"],
        )

    def test_transformers_trace_compare_summary_reports_drift_metrics(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=False,
            require_runtime_metadata_match=False,
            require_top_token_match=False,
            max_top_logit_regression=None,
            max_top_probability_regression=None,
            max_logit_l2_change=None,
            max_hidden_state_l2_change=None,
        )
        baseline = [
            {
                "row_type": "transformers_prompt_trace",
                "prompt_index": 0,
                "prompt": "spiral",
                "top_token_ids": "3,1",
                "top_logits": "1.1,0.4",
                "top_probabilities": "0.5,0.2",
                "logit_l2": 1.2,
                "hidden_state_l2": 0.9,
                "zspace_projection_status": "ok",
            },
            {
                "row_type": "transformers_prompt_trace",
                "prompt_index": 1,
                "prompt": "torch",
                "top_token_ids": "2,1",
                "top_logits": "0.8,0.4",
                "top_probabilities": "0.4,0.2",
                "logit_l2": 0.7,
                "hidden_state_l2": 0.5,
                "zspace_projection_status": "ok",
            },
        ]
        current = [
            dict(
                baseline[0],
                top_logits="0.9,0.4",
                top_probabilities="0.35,0.2",
                logit_l2=1.8,
                hidden_state_l2=0.6,
            ),
            dict(
                baseline[1],
                top_token_ids="1,2",
                zspace_projection_status="error",
            ),
        ]

        rows = module.compare_trace_rows(current, baseline, args)
        summary = rows[0]

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["top_token_changed_rows"], 1)
        self.assertEqual(summary["zspace_status_changed_rows"], 1)
        self.assertAlmostEqual(summary["observed_max_top_logit_regression"], 0.2)
        self.assertAlmostEqual(
            summary["observed_max_top_probability_regression"],
            0.15,
        )
        self.assertAlmostEqual(summary["observed_max_logit_l2_change"], 0.6)
        self.assertAlmostEqual(summary["observed_max_hidden_state_l2_change"], 0.3)
        self.assertEqual(summary["tensor_runtime_changed_rows"], 0)

    def test_transformers_trace_compare_reports_tensor_runtime_drift(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(
            require_trace_match=False,
            require_runtime_metadata_match=False,
            require_top_token_match=False,
            max_top_logit_regression=None,
            max_top_probability_regression=None,
            max_logit_l2_change=None,
            max_hidden_state_l2_change=None,
        )
        baseline = [
            {
                "row_type": "transformers_prompt_trace",
                "prompt_index": 0,
                "prompt": "spiral",
                "top_token_ids": "3",
                "top_logits": "1.1",
                "top_probabilities": "0.5",
                "logit_l2": 1.2,
                "hidden_state_l2": 0.9,
                "logits_tensor_backend": "torch",
                "logits_tensor_device": "mps:0",
                "logits_tensor_device_kind": "mps",
                "hidden_state_tensor_backend": "torch",
                "hidden_state_tensor_device": "mps:0",
                "hidden_state_tensor_device_kind": "mps",
            }
        ]
        current = [
            dict(
                baseline[0],
                logits_tensor_device="cuda:0",
                logits_tensor_device_kind="cuda",
                hidden_state_tensor_backend="python_sequence",
                hidden_state_tensor_device=None,
                hidden_state_tensor_device_kind=None,
            )
        ]

        rows = module.compare_trace_rows(current, baseline, args)
        summary, detail = rows

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["tensor_runtime_changed_rows"], 1)
        self.assertTrue(detail["tensor_runtime_changed"])
        self.assertIn(
            "logits_tensor_device",
            detail["tensor_runtime_changed_fields"],
        )
        self.assertIn(
            "hidden_state_tensor_backend",
            detail["tensor_runtime_changed_fields"],
        )

    def test_transformers_trace_zspace_status_gate_checks_current_rows(self):
        module = load_example("byte_lm_transformers_trace")
        args = argparse.Namespace(require_zspace_status="ok")
        rows = module.current_trace_gate_rows(
            [
                {
                    "row_type": "transformers_prompt_trace",
                    "prompt_index": 0,
                    "prompt": "spiral",
                    "zspace_projection_status": "not_requested",
                }
            ],
            args,
        )

        self.assertEqual(rows[0]["row_type"], "transformers_trace_gate_summary")
        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["failures"], 1)
        self.assertFalse(rows[1]["passed"])

    def test_transformers_trace_cli_writes_compare_jsonl(self):
        module = load_example("byte_lm_transformers_trace")
        old_argv = sys.argv
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.jsonl"
            current = Path(tmp) / "current.jsonl"
            compare = Path(tmp) / "compare.jsonl"
            try:
                sys.argv = [
                    "byte_lm_transformers_trace.py",
                    "--model-path",
                    tmp,
                    "--prompt",
                    "spiral",
                    "--top-k",
                    "2",
                    "--jsonl",
                    str(baseline),
                ]
                with fake_transformers_module(), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                sys.argv = [
                    "byte_lm_transformers_trace.py",
                    "--model-path",
                    tmp,
                    "--prompt",
                    "spiral",
                    "--top-k",
                    "2",
                    "--jsonl",
                    str(current),
                    "--compare-jsonl",
                    str(baseline),
                    "--compare-output-jsonl",
                    str(compare),
                    "--require-trace-match",
                    "--require-top-token-match",
                ]
                output = io.StringIO()
                with fake_transformers_module(), contextlib.redirect_stdout(output):
                    module.main()
            finally:
                sys.argv = old_argv
            rows = [
                json.loads(line)
                for line in compare.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(rows[0]["row_type"], "transformers_trace_compare_summary")
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[0]["compared_prompt_rows"], 1)
        self.assertTrue(rows[0]["runtime_metadata_available"])
        self.assertEqual(rows[0]["runtime_metadata_changed_count"], 0)
        self.assertEqual(rows[0]["runtime_metadata_changed_fields"], "none")
        self.assertIn("transformers_trace_compare", output.getvalue())

    def test_checkpoint_preflight_shape_audit_accepts_tied_lm_head_weight(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            key_preset="auto",
            include_extra_keys=[],
            no_synthesize_missing_biases=False,
            allow_overlap_resize=True,
            vocab=256,
            hidden=24,
            target_classes=256,
            checkpoint_projection="none",
            checkpoint_projection_preset=None,
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        row = helper.hf_lm_shape_audit_row(
            args,
            "hf-state-dict:auto",
            ["model.safetensors"],
            {
                "model.embed_tokens.weight": (320, 32),
            },
        )

        self.assertEqual(row["checkpoint_key_preset"], "llama")
        self.assertEqual(row["checkpoint_target_classes"], 320)
        self.assertEqual(row["missing_required_keys"], "none")
        self.assertEqual(row["lm_head_weight_shape"], "320x32")
        self.assertTrue(row["lm_head_weight_synthesized_from_embed"])
        self.assertTrue(row["can_materialize_requested"])

    def test_checkpoint_preflight_shape_audit_gates_fail_unsafe_resize(self):
        helper = load_checkpoint_helper()
        row = {
            "can_materialize_requested": False,
            "exact_shape_match": False,
            "checkpoint_key_preset": "llama",
        }
        args = types.SimpleNamespace(
            require_shape_materializable=True,
            require_exact_shape_match=False,
            require_detected_key_preset=None,
        )
        with self.assertRaisesRegex(RuntimeError, "not materializable"):
            helper.check_shape_audit_gates(row, args)

    def test_checkpoint_preflight_shape_audit_gates_accept_expected_preset(self):
        helper = load_checkpoint_helper()
        row = {
            "can_materialize_requested": True,
            "exact_shape_match": True,
            "checkpoint_key_preset": "llama",
        }
        args = types.SimpleNamespace(
            require_shape_materializable=True,
            require_exact_shape_match=True,
            require_detected_key_preset="llama",
        )
        self.assertTrue(helper.check_shape_audit_gates(row, args))

    def test_checkpoint_preflight_shape_audit_gates_fail_unexpected_preset(self):
        helper = load_checkpoint_helper()
        row = {
            "can_materialize_requested": True,
            "exact_shape_match": True,
            "checkpoint_key_preset": "gpt2",
        }
        args = types.SimpleNamespace(
            require_shape_materializable=False,
            require_exact_shape_match=False,
            require_detected_key_preset="llama",
        )
        with self.assertRaisesRegex(RuntimeError, "detected key preset"):
            helper.check_shape_audit_gates(row, args)

    def test_checkpoint_preflight_shape_audit_auto_detects_llama_layout(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            key_preset="auto",
            include_extra_keys=[],
            no_synthesize_missing_biases=False,
            allow_overlap_resize=False,
            vocab=None,
            hidden=None,
            target_classes=None,
            checkpoint_projection="none",
            checkpoint_projection_preset=None,
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        row = helper.hf_lm_shape_audit_row(
            args,
            "hf-state-dict:auto",
            ["model.safetensors"],
            {
                "model.embed_tokens.weight": (4, 2),
                "lm_head.weight": (3, 2),
            },
        )
        self.assertEqual(row["requested_key_preset"], "auto")
        self.assertEqual(row["checkpoint_key_preset"], "llama")
        self.assertEqual(row["checkpoint_vocab"], 4)
        self.assertEqual(row["checkpoint_hidden"], 2)
        self.assertEqual(row["checkpoint_target_classes"], 3)

    def test_checkpoint_preflight_shape_audit_auto_detects_gpt_neox_layout(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            key_preset="auto",
            include_extra_keys=[],
            no_synthesize_missing_biases=False,
            allow_overlap_resize=False,
            vocab=None,
            hidden=None,
            target_classes=None,
            checkpoint_projection="none",
            checkpoint_projection_preset=None,
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        row = helper.hf_lm_shape_audit_row(
            args,
            "hf-state-dict:auto",
            ["model.safetensors"],
            {
                "gpt_neox.embed_in.weight": (4, 2),
                "embed_out.weight": (3, 2),
            },
        )
        self.assertEqual(row["checkpoint_key_preset"], "gpt_neox")
        self.assertEqual(row["checkpoint_vocab"], 4)
        self.assertEqual(row["checkpoint_target_classes"], 3)

    def test_checkpoint_preflight_shape_audit_auto_detects_gpt2_bare_layout(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            key_preset="auto",
            include_extra_keys=[],
            no_synthesize_missing_biases=False,
            allow_overlap_resize=True,
            vocab=256,
            hidden=24,
            target_classes=256,
            checkpoint_projection="none",
            checkpoint_projection_preset=None,
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        row = helper.hf_lm_shape_audit_row(
            args,
            "hf-state-dict:auto",
            ["model.safetensors"],
            {
                "wte.weight": (50257, 768),
            },
        )
        self.assertEqual(row["checkpoint_key_preset"], "gpt2_bare")
        self.assertEqual(row["checkpoint_vocab"], 50257)
        self.assertEqual(row["checkpoint_hidden"], 768)
        self.assertEqual(row["checkpoint_target_classes"], 50257)
        self.assertTrue(row["lm_head_weight_synthesized_from_embed"])
        self.assertTrue(row["can_materialize_requested"])

    def test_checkpoint_preflight_shape_audit_auto_detects_gemma_wrapper_layout(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            key_preset="auto",
            include_extra_keys=[],
            no_synthesize_missing_biases=False,
            allow_overlap_resize=True,
            vocab=256,
            hidden=24,
            target_classes=256,
            checkpoint_projection="none",
            checkpoint_projection_preset=None,
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        row = helper.hf_lm_shape_audit_row(
            args,
            "hf-state-dict:auto",
            ["model.safetensors"],
            {
                "model.language_model.embed_tokens.weight": (262144, 2560),
            },
        )
        self.assertEqual(row["checkpoint_key_preset"], "gemma")
        self.assertEqual(row["checkpoint_vocab"], 262144)
        self.assertEqual(row["checkpoint_hidden"], 2560)
        self.assertEqual(row["checkpoint_target_classes"], 262144)
        self.assertTrue(row["lm_head_weight_synthesized_from_embed"])
        self.assertTrue(row["can_materialize_requested"])

    def test_checkpoint_preflight_shape_audit_auto_rejects_ambiguous_layout(self):
        helper = load_checkpoint_helper()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            helper.detect_hf_lm_key_preset(
                {
                    "transformer.wte.weight": (4, 2),
                    "model.embed_tokens.weight": (4, 2),
                    "lm_head.weight": (3, 2),
                }
            )

    def test_checkpoint_preflight_loads_local_hf_torch_state_dict_subset(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.tensor(
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                        ),
                        "lm_head.weight": torch.tensor(
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                        ),
                        "model.layers.0.input_layernorm.weight": torch.tensor([[1.0, 1.0]]),
                        "unused.deep.weight": torch.tensor([[9.0]]),
                    }
                },
                path,
            )

            include_keys = helper.hf_lm_state_keys(
                "llama",
                include_extra_keys=["model.layers.0.input_layernorm.weight"],
            )
            external_state, loaded_files = helper.load_hf_state_dict(
                path,
                include_keys=include_keys,
            )
            self.assertEqual(loaded_files, [str(path)])
            self.assertNotIn("unused.deep.weight", external_state)
            self.assertEqual(
                helper.infer_hf_lm_module_shapes(
                    external_state,
                    key_preset="llama",
                ),
                (4, 2, 3),
            )

            checkpoint, rules = helper.hf_lm_handoff_from_external_state(
                external_state,
                key_preset="llama",
                include_extra_keys=["model.layers.0.input_layernorm.weight"],
            )
            self.assertEqual(checkpoint["model.embed_tokens.bias"].data(), [0.0, 0.0])
            self.assertEqual(checkpoint["lm_head.bias"].data(), [0.0, 0.0, 0.0])
            self.assertIn("model.layers.0.input_layernorm.weight", checkpoint)
            self.assertEqual(rules["model.embed_tokens.weight"], "embed::weight")

    def test_checkpoint_preflight_loads_local_hf_torch_state_dict_shapes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            torch.save(
                {
                    "model": {
                        "model.embed_tokens.weight": torch.zeros(4, 2),
                        "lm_head.weight": torch.zeros(3, 2),
                        "model.layers.0.input_layernorm.weight": torch.ones(1, 2),
                        "unused.deep.weight": torch.ones(1, 1),
                    }
                },
                path,
            )

            shape_state, loaded_files = helper.load_hf_state_dict_shapes(
                path,
                include_keys=helper.hf_lm_state_keys(
                    "llama",
                    include_extra_keys=["model.layers.0.input_layernorm.weight"],
                ),
            )
            self.assertEqual(loaded_files, [str(path)])
            self.assertEqual(
                shape_state,
                {
                    "model.embed_tokens.weight": (4, 2),
                    "lm_head.weight": (3, 2),
                    "model.layers.0.input_layernorm.weight": (1, 2),
                },
            )

    def test_checkpoint_preflight_loads_safetensors_state_dict_shapes(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("PyTorch or safetensors is not installed")
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(
                {
                    "model.embed_tokens.weight": torch.zeros(4, 2),
                    "lm_head.weight": torch.zeros(3, 2),
                    "model.layers.0.input_layernorm.weight": torch.ones(1, 2),
                    "unused.deep.weight": torch.ones(1, 1),
                },
                path,
            )
            shape_state, loaded_files = helper.load_hf_state_dict_shapes(
                path,
                include_keys=helper.hf_lm_state_keys(
                    "llama",
                    include_extra_keys=["model.layers.0.input_layernorm.weight"],
                ),
            )
            self.assertEqual(loaded_files, [str(path)])
            self.assertEqual(
                shape_state,
                {
                    "model.embed_tokens.weight": (4, 2),
                    "lm_head.weight": (3, 2),
                    "model.layers.0.input_layernorm.weight": (1, 2),
                },
            )

    def test_checkpoint_preflight_loads_safetensors_state_dict_with_bounds(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("PyTorch or safetensors is not installed")
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(
                {
                    "model.embed_tokens.weight": torch.arange(
                        20,
                        dtype=torch.float32,
                    ).reshape(4, 5),
                    "model.embed_tokens.bias": torch.arange(5, dtype=torch.float32),
                    "lm_head.weight": torch.arange(30, dtype=torch.float32).reshape(6, 5),
                    "lm_head.bias": torch.arange(6, dtype=torch.float32),
                },
                path,
            )
            bounds = helper.hf_lm_tensor_bounds_for_module_shapes(
                (2, 3, 4),
                key_preset="llama",
                lm_head_weight_transform="transpose_copy_overlap_zeros",
            )
            state, loaded_files = helper.load_hf_state_dict(
                path,
                include_keys=helper.hf_lm_state_keys("llama"),
                tensor_bounds=bounds,
            )

            self.assertEqual(loaded_files, [str(path)])
            embed = helper.tensor_from_external(
                state["model.embed_tokens.weight"],
                name="model.embed_tokens.weight",
            )
            embed_bias = helper.tensor_from_external(
                state["model.embed_tokens.bias"],
                name="model.embed_tokens.bias",
            )
            head = helper.tensor_from_external(
                state["lm_head.weight"],
                name="lm_head.weight",
            )
            head_bias = helper.tensor_from_external(
                state["lm_head.bias"],
                name="lm_head.bias",
            )
            self.assertEqual(embed.shape(), (2, 3))
            self.assertEqual(embed.data(), [0.0, 1.0, 2.0, 5.0, 6.0, 7.0])
            self.assertEqual(embed_bias.shape(), (1, 3))
            self.assertEqual(embed_bias.data(), [0.0, 1.0, 2.0])
            self.assertEqual(head.shape(), (4, 3))
            self.assertEqual(
                head.data(),
                [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
            )
            self.assertEqual(head_bias.shape(), (1, 4))
            self.assertEqual(head_bias.data(), [0.0, 1.0, 2.0, 3.0])

    def test_checkpoint_preflight_loads_indexed_hf_state_dict_shards(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shard_a = directory / "pytorch_model-00001-of-00002.bin"
            shard_b = directory / "pytorch_model-00002-of-00002.bin"
            torch.save(
                {
                    "transformer.wte.weight": torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]]
                    ),
                    "transformer.wte.bias": torch.tensor([0.1, -0.1]),
                    "unused.shard_a.weight": torch.tensor([[8.0]]),
                },
                shard_a,
            )
            torch.save(
                {
                    "lm_head.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "lm_head.bias": torch.tensor([0.2, -0.2]),
                    "unused.shard_b.weight": torch.tensor([[9.0]]),
                },
                shard_b,
            )
            (directory / "pytorch_model.bin.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "transformer.wte.weight": shard_a.name,
                            "transformer.wte.bias": shard_a.name,
                            "lm_head.weight": shard_b.name,
                            "lm_head.bias": shard_b.name,
                            "unused.shard_b.weight": shard_b.name,
                        },
                    }
                ),
                encoding="utf-8",
            )

            external_state, loaded_files = helper.load_hf_state_dict(
                directory,
                include_keys=helper.hf_lm_state_keys("gpt2"),
            )
            self.assertEqual(set(map(Path, loaded_files)), {shard_a, shard_b})
            self.assertEqual(
                sorted(external_state),
                [
                    "lm_head.bias",
                    "lm_head.weight",
                    "transformer.wte.bias",
                    "transformer.wte.weight",
                ],
            )

    def test_checkpoint_preflight_shape_audit_auto_loads_indexed_safetensors(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("PyTorch or safetensors is not installed")
        helper = load_checkpoint_helper()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shard_a = directory / "model-00001-of-00002.safetensors"
            shard_b = directory / "model-00002-of-00002.safetensors"
            save_file(
                {
                    "model.embed_tokens.weight": torch.zeros(4, 2),
                    "model.layers.0.input_layernorm.weight": torch.ones(1, 2),
                },
                shard_a,
            )
            save_file(
                {
                    "lm_head.weight": torch.zeros(3, 2),
                    "unused.deep.weight": torch.ones(1, 1),
                },
                shard_b,
            )
            (directory / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "model.embed_tokens.weight": shard_a.name,
                            "model.layers.0.input_layernorm.weight": shard_a.name,
                            "lm_head.weight": shard_b.name,
                            "unused.deep.weight": shard_b.name,
                        },
                    }
                ),
                encoding="utf-8",
            )

            shape_state, loaded_files = helper.load_hf_state_dict_shapes(
                directory,
                include_keys=helper.all_hf_lm_state_keys(
                    include_extra_keys=["model.layers.0.input_layernorm.weight"],
                ),
            )

        self.assertEqual(set(map(Path, loaded_files)), {shard_a, shard_b})
        row = helper.hf_lm_shape_audit_row(
            types.SimpleNamespace(
                key_preset="auto",
                include_extra_keys=["model.layers.0.input_layernorm.weight"],
                no_synthesize_missing_biases=False,
                allow_overlap_resize=False,
                vocab=None,
                hidden=None,
                target_classes=None,
                checkpoint_projection="none",
                checkpoint_projection_preset=None,
                checkpoint_projection_strength=0.5,
                checkpoint_projection_curvature=-0.5,
                checkpoint_projection_frequency=0.65,
            ),
            "hf-state-dict:auto",
            loaded_files,
            shape_state,
        )
        self.assertEqual(row["checkpoint_key_preset"], "llama")
        self.assertEqual(row["checkpoint_vocab"], 4)
        self.assertEqual(row["checkpoint_hidden"], 2)
        self.assertEqual(row["checkpoint_target_classes"], 3)
        self.assertEqual(row["present_extra_keys"], "model.layers.0.input_layernorm.weight")
        self.assertNotIn("unused.deep.weight", shape_state)

    def test_checkpoint_preflight_external_hf_state_can_require_biases(self):
        helper = load_checkpoint_helper()
        external_state = {
            "transformer.wte.weight": [[1, 2], [3, 4]],
            "lm_head.weight": [[1, 2], [3, 4]],
        }
        with self.assertRaisesRegex(KeyError, "transformer.wte.bias"):
            helper.hf_lm_handoff_from_external_state(
                external_state,
                synthesize_missing_biases=False,
            )

    def test_checkpoint_preflight_external_hf_state_infers_identity_head_bias_width(self):
        helper = load_checkpoint_helper()
        external_state = {
            "transformer.wte.weight": [[1, 2], [3, 4]],
            "lm_head.weight": [[1, 2, 3], [4, 5, 6]],
        }
        checkpoint, rules = helper.hf_lm_handoff_from_external_state(
            external_state,
            lm_head_weight_transform="identity",
        )
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, 3))
        self.assertEqual(checkpoint["lm_head.bias"].data(), [0.0, 0.0, 0.0])
        self.assertEqual(
            rules["lm_head.weight"],
            {"target": "head::weight", "transform": "identity"},
        )

    def test_checkpoint_preflight_synthesizes_hf_lm_biases(self):
        helper = load_checkpoint_helper()
        state = {
            "embed::weight": helper.st.Tensor(4, 2, range(8)),
            "head::weight": helper.st.Tensor(2, 3, range(6)),
        }
        checkpoint, rules = helper.hf_lm_handoff_from_spiraltorch_state(
            state,
            embed_bias_source=None,
            lm_head_bias_source=None,
        )
        self.assertEqual(checkpoint["transformer.wte.bias"].shape(), (1, 2))
        self.assertEqual(checkpoint["transformer.wte.bias"].data(), [0.0, 0.0])
        self.assertEqual(checkpoint["lm_head.bias"].shape(), (1, 3))
        self.assertEqual(checkpoint["lm_head.bias"].data(), [0.0, 0.0, 0.0])
        self.assertEqual(rules["transformer.wte.bias"], "embed::bias")
        self.assertEqual(
            rules["lm_head.bias"],
            {"target": "head::bias", "transform": "copy_overlap_zeros"},
        )

    def test_checkpoint_preflight_externalizes_head_only_hf_lm_handoff(self):
        helper = load_checkpoint_helper()
        state = {
            "adapter_head::weight": helper.st.Tensor(2, 3, range(6)),
            "adapter_head::bias": helper.st.Tensor(1, 3, [0.2, -0.2, 0.4]),
        }
        checkpoint, rules = helper.hf_lm_handoff_from_spiraltorch_state(
            state,
            embed_weight_key=None,
            embed_bias_key=None,
            lm_head_weight_source="adapter_head::weight",
            lm_head_bias_source="adapter_head::bias",
            lm_head_weight_target="adapter_head::weight",
            lm_head_bias_target="adapter_head::bias",
            lm_head_bias_transform="identity",
        )
        self.assertEqual(sorted(checkpoint), ["lm_head.bias", "lm_head.weight"])
        self.assertEqual(checkpoint["lm_head.weight"].shape(), (3, 2))
        self.assertEqual(
            rules,
            {
                "lm_head.weight": {
                    "target": "adapter_head::weight",
                    "transform": "transpose",
                },
                "lm_head.bias": {
                    "target": "adapter_head::bias",
                    "transform": "identity",
                },
            },
        )

    def test_checkpoint_preflight_rejects_unknown_source(self):
        helper = load_checkpoint_helper()
        with self.assertRaisesRegex(ValueError, "unknown checkpoint source"):
            helper.checkpoint_source("mystery")

    def test_checkpoint_preflight_builds_checkpoint_audit_fields(self):
        helper = load_checkpoint_helper()
        report = {
            "matched": 2,
            "extra": 1,
            "source": {"hash": "source-hash"},
            "matched_subset": {"hash": "matched-hash"},
            "entries": [
                {
                    "name": "head::weight",
                    "status": "matched",
                    "source_name": "lm_head.weight",
                    "transform": "transpose",
                    "expected_shape": (2, 3),
                    "source_shape": (2, 3),
                    "original_source_shape": (3, 2),
                },
                {
                    "name": "embed::weight",
                    "status": "extra",
                    "source_name": "transformer.wte.weight",
                    "transform": "identity",
                    "expected_shape": None,
                    "source_shape": (4, 2),
                    "original_source_shape": (4, 2),
                },
            ],
        }
        load = {
            "matched": True,
            "source": {"hash": "load-source-hash"},
            "loaded": {"hash": "loaded-hash"},
        }
        fields = helper.checkpoint_audit_fields("head", report, load)
        self.assertEqual(fields["head_preflight_matched"], 2)
        self.assertEqual(fields["head_preflight_source_hash"], "source-hash")
        self.assertEqual(
            fields["head_preflight_matched_subset_hash"],
            "matched-hash",
        )
        self.assertIn("lm_head.weight:transpose", fields["head_preflight_signature"])
        self.assertEqual(fields["head_load_source_hash"], "load-source-hash")
        self.assertEqual(fields["head_load_loaded_hash"], "loaded-hash")

    def test_checkpoint_preflight_flattens_projection_context(self):
        helper = load_checkpoint_helper()
        args = types.SimpleNamespace(
            allow_overlap_resize=True,
            checkpoint_projection="zspace",
            checkpoint_projection_strength=0.5,
            checkpoint_projection_curvature=-0.5,
            checkpoint_projection_frequency=0.65,
        )
        context = helper.preflight_context_fields(
            args,
            "hf-state-dict:llama",
            ["shard-a", "shard-b"],
            (256, 24, 256),
        )
        self.assertEqual(
            context,
            {
                "checkpoint_source": "hf-state-dict:llama",
                "checkpoint_loaded_files": 2,
                "checkpoint_vocab": 256,
                "checkpoint_hidden": 24,
                "checkpoint_target_classes": 256,
                "checkpoint_overlap_resize": True,
                "checkpoint_projection": "zspace",
                "checkpoint_projection_strength": 0.5,
                "checkpoint_projection_curvature": -0.5,
                "checkpoint_projection_frequency": 0.65,
                "checkpoint_source_gain": 1.0,
            },
        )
        report = {
            "compatible": True,
            "matched": 1,
            "missing": 0,
            "shape_mismatched": 0,
            "extra": 0,
            "source": {"hash": "source-hash"},
            "matched_subset": {"hash": "matched-hash"},
            "entries": [],
        }
        rows = helper.flatten_report("embed", report, context)
        self.assertEqual(rows[0]["checkpoint_projection"], "zspace")
        self.assertEqual(rows[0]["checkpoint_source"], "hf-state-dict:llama")

    def test_checkpoint_preflight_compares_flat_jsonl_rows(self):
        helper = load_checkpoint_helper()
        baseline = [
            {
                "row_type": "report",
                "label": "embed",
                "matched": 2,
                "source_hash": "old-source",
            },
            {
                "row_type": "entry",
                "label": "embed",
                "name": "embed::weight",
                "source_name": "transformer.wte.weight",
                "transform": "identity",
            },
            {
                "row_type": "entry",
                "label": "lora_head_base",
                "name": "head::weight",
                "source_name": "lm_head.weight",
                "transform": "transpose",
            },
        ]
        current = [
            {
                "row_type": "report",
                "label": "embed",
                "matched": 2,
                "source_hash": "new-source",
            },
            {
                "row_type": "entry",
                "label": "embed",
                "name": "embed::weight",
                "source_name": "transformer.wte.weight",
                "transform": "identity",
            },
            {
                "row_type": "entry",
                "label": "embed",
                "name": "embed::bias",
                "source_name": "transformer.wte.bias",
                "transform": "identity",
            },
        ]
        differences = helper.compare_preflight_rows(current, baseline)
        self.assertEqual(
            [(diff["kind"], helper.preflight_row_key_label(diff["key"])) for diff in differences],
            [
                ("missing", "entry::lora_head_base::head::weight"),
                ("extra", "entry::embed::embed::bias"),
                ("changed", "report::embed"),
            ],
        )
        changed = differences[-1]
        self.assertEqual(changed["field"], "source_hash")
        self.assertEqual(changed["before"], "old-source")
        self.assertEqual(changed["after"], "new-source")

        baseline = [{"row_type": "report", "label": "embed"}]
        current = [
            {
                "row_type": "report",
                "label": "embed",
                "checkpoint_projection_strength": None,
            }
        ]
        differences = helper.compare_preflight_rows(current, baseline)
        self.assertEqual(differences[0]["field"], "checkpoint_projection_strength")
        self.assertEqual(differences[0]["before"], "<missing>")
        self.assertIsNone(differences[0]["after"])

    def test_checkpoint_preflight_rejects_duplicate_compare_rows(self):
        helper = load_checkpoint_helper()
        rows = [
            {"row_type": "report", "label": "embed"},
            {"row_type": "report", "label": "embed"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate preflight row key"):
            helper.compare_preflight_rows(rows, [])

    def test_helper_adds_and_validates_summary_compare_cli_args(self):
        helper = load_compare_helper()
        parser = argparse.ArgumentParser()
        helper.add_summary_compare_args(parser, subject="route")
        args = parser.parse_args(
            [
                "--require-accepted-match",
                "--require-checkpoint-match",
                "--max-target-loss-regression",
                "0.0",
                "--min-target-loss-margin",
                "0.01",
            ]
        )
        self.assertTrue(helper.validate_summary_compare_args(parser, args))
        self.assertTrue(args.require_accepted_match)
        self.assertTrue(args.require_checkpoint_match)
        self.assertEqual(args.max_target_loss_regression, 0.0)
        self.assertEqual(args.min_target_loss_margin, 0.01)

    def test_helper_rejects_negative_summary_compare_tolerance(self):
        helper = load_compare_helper()
        parser = argparse.ArgumentParser()
        helper.add_summary_compare_args(parser, subject="ratio")
        args = parser.parse_args(["--max-retention-loss-regression", "-0.1"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            helper.validate_summary_compare_args(parser, args)

    def test_helper_builds_summary_compare_failure_messages(self):
        helper = load_compare_helper()
        comparison = {
            "baseline_status": "ok",
            "current_status": "guard_rejected",
            "status_changed": True,
            "accepted_changed": True,
            "guard_changed": True,
            "movement_tolerance_changed": True,
            "resume_changed": True,
            "target_loss_regression": 0.25,
            "retention_loss_regression": 0.125,
        }
        failures = helper.summary_compare_failures(
            "case::route",
            comparison,
            max_target_loss_regression=0.0,
            max_retention_loss_regression=0.0,
            min_target_loss_margin=None,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=True,
            require_accepted_match=True,
            require_guard_match=True,
            require_movement_tolerance_match=True,
            require_resume_match=True,
        )
        self.assertEqual(
            failures,
            [
                "case::route: status changed from ok to guard_rejected",
                "case::route: sparse FT guard acceptance changed",
                "case::route: sparse retention guard settings changed",
                "case::route: movement audit tolerance changed",
                "case::route: FT-ready resume fingerprint changed",
                "case::route: target_loss_delta regressed by 0.250000000",
                "case::route: retention_loss_delta regressed by 0.125000000",
            ],
        )

    def test_helper_backfills_summary_guard_margins(self):
        helper = load_compare_helper()
        row = {
            "target_loss_delta": 0.25,
            "target_min_loss_delta": 0.1,
            "best_retention_loss_increase": 0.125,
            "best_retention_accuracy_drop": 0.2,
            "best_retention_perplexity_increase": 0.75,
            "retention_max_loss_increase": 0.5,
            "retention_max_accuracy_drop": 0.25,
            "retention_max_perplexity_increase": 1.0,
        }
        margins = helper.summary_guard_margins(row)
        self.assertAlmostEqual(margins["target_loss_margin"], 0.15)
        self.assertAlmostEqual(margins["retention_loss_margin"], 0.375)
        self.assertAlmostEqual(margins["retention_accuracy_margin"], 0.05)
        self.assertAlmostEqual(margins["retention_perplexity_margin"], 0.25)
        helper.attach_summary_guard_margins(row)
        self.assertEqual(row["target_loss_margin"], margins["target_loss_margin"])
        self.assertEqual(
            row["retention_perplexity_margin"],
            margins["retention_perplexity_margin"],
        )

    def test_helper_backfills_summary_guard_counts(self):
        helper = load_compare_helper()
        legacy = {
            "accepted": True,
            "guarded_best_epoch": 2,
            "epochs_run": 3,
        }
        counts = helper.summary_guard_counts(legacy)
        self.assertEqual(counts["guard_epochs_run"], 3)
        self.assertEqual(counts["guard_accepted_epochs"], 1)
        self.assertEqual(counts["guard_retention_rejected_epochs"], 0)
        self.assertEqual(counts["guard_target_stale_epochs"], 0)
        self.assertAlmostEqual(counts["guard_acceptance_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(counts["guard_retention_rejected_rate"], 0.0)
        self.assertAlmostEqual(counts["guard_target_stale_rate"], 0.0)
        self.assertFalse(counts["guard_epoch_counts_available"])
        helper.attach_summary_guard_counts(legacy)
        self.assertEqual(legacy["guard_epochs_run"], 3)
        self.assertEqual(legacy["guard_accepted_epochs"], 1)
        self.assertAlmostEqual(legacy["guard_acceptance_rate"], 1.0 / 3.0)
        self.assertFalse(legacy["guard_epoch_counts_available"])

        captured = types.SimpleNamespace(
            guarded_best_epoch=3,
            guard_accepted_epochs=2,
            guard_retention_rejected_epochs=1,
            guard_target_stale_epochs=4,
        )
        current = {"accepted": True, "guarded_best_epoch": 3, "epochs_run": 7}
        counts = helper.summary_guard_counts(current, captured)
        self.assertEqual(counts["guard_epochs_run"], 7)
        self.assertEqual(counts["guard_accepted_epochs"], 2)
        self.assertEqual(counts["guard_retention_rejected_epochs"], 1)
        self.assertEqual(counts["guard_target_stale_epochs"], 4)
        self.assertAlmostEqual(counts["guard_acceptance_rate"], 2.0 / 7.0)
        self.assertAlmostEqual(counts["guard_retention_rejected_rate"], 1.0 / 7.0)
        self.assertAlmostEqual(counts["guard_target_stale_rate"], 4.0 / 7.0)
        self.assertTrue(counts["guard_epoch_counts_available"])

        explicit_rates = helper.summary_guard_counts(
            {
                "accepted": True,
                "guarded_best_epoch": 1,
                "epochs_run": 4,
                "guard_epochs_run": 4,
                "guard_accepted_epochs": 2,
                "guard_retention_rejected_epochs": 1,
                "guard_target_stale_epochs": 1,
                "guard_acceptance_rate": 0.75,
                "guard_retention_rejected_rate": 0.125,
                "guard_target_stale_rate": 0.125,
            }
        )
        self.assertAlmostEqual(explicit_rates["guard_acceptance_rate"], 0.75)
        self.assertAlmostEqual(explicit_rates["guard_retention_rejected_rate"], 0.125)
        self.assertAlmostEqual(explicit_rates["guard_target_stale_rate"], 0.125)

    def test_helper_rejects_boolean_summary_margin_inputs(self):
        helper = load_compare_helper()
        with self.assertRaisesRegex(ValueError, "target_loss_delta is not numeric"):
            helper.summary_guard_margins({"target_loss_delta": True})
        with self.assertRaisesRegex(
            ValueError,
            "retention_perplexity_margin is not numeric",
        ):
            helper.summary_guard_margins({"retention_perplexity_margin": False})
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            helper.summary_bool_value({"accepted": "false"}, "accepted")

    def test_replay_sweep_rejects_nonboolean_winner_acceptance(self):
        module = load_example("byte_lm_replay_sweep")
        row = {
            "ratio": "target-per-replay-1",
            "target_per_replay": 1,
            "accepted": "false",
            "target_loss_delta": 0.1,
            "retention_loss_delta": 0.2,
            "retention_accuracy_delta": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            module.replay_winner([row])
        row["accepted"] = True
        row["target_loss_delta"] = True
        with self.assertRaisesRegex(ValueError, "target_loss_delta is not numeric"):
            module.replay_winner([row])

    def test_helper_writes_and_loads_single_summary_jsonl(self):
        helper = load_compare_helper()
        row = {
            "config": "r4_a16",
            "accepted": True,
            "status": "ok",
            "target_loss_delta": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.jsonl"
            helper.write_summary_jsonl(path, [row])
            self.assertEqual(helper.load_single_summary_jsonl(path), row)
            helper.write_summary_jsonl(path, [row, row])
            with self.assertRaisesRegex(ValueError, "exactly one summary row"):
                helper.load_single_summary_jsonl(path)

    def test_helper_compares_single_summary_rows(self):
        helper = load_compare_helper()
        helper.compare_sparse_finetune_summaries = old_compare_signature
        args = types.SimpleNamespace(
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            min_target_loss_margin=None,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=True,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
            require_checkpoint_match=False,
        )
        current = {"config": "r4_a16", "accepted": True}
        baseline = {"config": "r4_a16", "accepted": True}
        comparison = helper.compare_single_summary(current, baseline, args)
        self.assertTrue(comparison["passed"])

        current = {"config": "r4_a16", "accepted": False}
        with self.assertRaisesRegex(RuntimeError, "summary regression gate failed"):
            helper.compare_single_summary(current, baseline, args)

    def test_helper_gates_checkpoint_audit_changes(self):
        helper = load_compare_helper()
        helper.compare_sparse_finetune_summaries = old_compare_signature
        args = types.SimpleNamespace(
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            min_target_loss_margin=None,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=False,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
            require_checkpoint_match=True,
        )
        baseline = {
            "config": "r12_a64",
            "accepted": True,
            "head_preflight_extra": 2,
            "head_load_matched": True,
            "checkpoint_source_origin": "hf_state_dict",
            "checkpoint_vocab": 256,
            "checkpoint_hidden": 24,
            "checkpoint_target_classes": 256,
            "checkpoint_overlap_resize": False,
            "checkpoint_projection": "none",
            "checkpoint_projection_strength": None,
            "checkpoint_projection_curvature": None,
            "checkpoint_projection_frequency": None,
        }
        current = {
            "config": "r12_a64",
            "accepted": True,
            "head_preflight_extra": 3,
            "head_load_matched": True,
            "checkpoint_source_origin": "hf_state_dict",
            "checkpoint_vocab": 260,
            "checkpoint_hidden": 25,
            "checkpoint_target_classes": 259,
            "checkpoint_overlap_resize": True,
            "checkpoint_projection": "zspace",
            "checkpoint_projection_strength": 0.5,
            "checkpoint_projection_curvature": -1.0,
            "checkpoint_projection_frequency": 0.65,
        }
        self.assertEqual(
            helper.checkpoint_audit_differences(current, baseline),
            [
                ("head_preflight_extra", 2, 3),
                ("checkpoint_vocab", 256, 260),
                ("checkpoint_hidden", 24, 25),
                ("checkpoint_target_classes", 256, 259),
                ("checkpoint_overlap_resize", False, True),
                ("checkpoint_projection", "none", "zspace"),
                ("checkpoint_projection_strength", None, 0.5),
                ("checkpoint_projection_curvature", None, -1.0),
                ("checkpoint_projection_frequency", None, 0.65),
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint audit head_preflight_extra"):
            helper.compare_single_summary(current, baseline, args)

        current = {
            "config": "r12_a64",
            "accepted": True,
            "head_preflight_signature": "lm_head.weight:matched:transpose",
        }
        baseline = {
            "config": "r12_a64",
            "accepted": True,
            "head_preflight_signature": "external.weight:matched:transpose",
        }
        self.assertEqual(
            helper.checkpoint_audit_differences(current, baseline),
            [
                (
                    "head_preflight_signature",
                    "external.weight:matched:transpose",
                    "lm_head.weight:matched:transpose",
                )
            ],
        )

        current = {
            "config": "r12_a64",
            "accepted": True,
            "checkpoint_projection_strength": None,
        }
        baseline = {"config": "r12_a64", "accepted": True}
        self.assertEqual(
            helper.checkpoint_audit_differences(current, baseline),
            [("checkpoint_projection_strength", "<missing>", None)],
        )

    def test_helper_gates_current_summary_margins_with_old_extension_signature(self):
        baseline = {"accepted": True}
        current = {
            "accepted": True,
            "target_loss_delta": 0.25,
            "target_min_loss_delta": 0.1,
            "best_retention_loss_increase": 0.125,
            "best_retention_accuracy_drop": 0.2,
            "retention_max_loss_increase": 0.5,
            "retention_max_accuracy_drop": 0.25,
        }
        helper = load_compare_helper()
        helper.compare_sparse_finetune_summaries = old_compare_signature
        comparison = helper.compare_summaries(
            current,
            baseline,
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            min_target_loss_margin=0.2,
            min_retention_loss_margin=0.4,
            min_retention_accuracy_margin=0.1,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=False,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
        )
        self.assertFalse(comparison["passed"])
        self.assertAlmostEqual(comparison["target_loss_margin_shortfall"], 0.05)
        self.assertAlmostEqual(comparison["retention_loss_margin_shortfall"], 0.025)
        self.assertAlmostEqual(comparison["retention_accuracy_margin_shortfall"], 0.05)

    def test_zspace_compare_selects_geometry_tokens_case(self):
        module = load_example("byte_lm_zspace_compare")
        self.assertEqual(
            [case["label"] for case in module.selected_cases(["geometry_tokens"])],
            ["geometry_tokens"],
        )
        self.assertIn(
            "geometry_tokens",
            [case["label"] for case in module.selected_cases(None)],
        )

    def test_zspace_compare_selects_fine_route_preset(self):
        module = load_example("byte_lm_zspace_compare")
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None)],
            [
                "baseline",
                "zspace_s025",
                "zspace_s050",
                "zspace_post_s050_c025",
            ],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None, "fine")],
            [
                "baseline",
                "zspace_s035_c025",
                "zspace_post_s050_c025",
                "zspace_s075_c025",
                "zspace_s050_c050",
            ],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None, "ridge")],
            [
                "baseline",
                "zspace_post_s050_c025",
                "zspace_s075_c025",
                "zspace_s100_c025",
                "zspace_s075_c010",
                "zspace_s075_c050",
            ],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None, "crest")],
            [
                "baseline",
                "zspace_s050_c010",
                "zspace_s075_c010",
                "zspace_s100_c010",
                "zspace_s075_c005",
                "zspace_s075_c025",
            ],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None, "summit")],
            [
                "baseline",
                "zspace_s075_c005",
                "zspace_s100_c010",
                "zspace_s100_c005",
                "zspace_s100_c0025",
                "zspace_s100_c025",
            ],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None, "horizon")],
            [
                "baseline",
                "zspace_s100_c010",
                "zspace_s100_c005",
                "zspace_s100_c0025",
                "zspace_s100_c001",
                "zspace_s100_c0005",
            ],
        )
        self.assertEqual(
            [spec["label"] for spec in module.selected_routes(None, "health")],
            [
                "baseline",
                "zspace_s100_c010",
                "zspace_s100_c0075",
                "zspace_s100_c005",
                "zspace_s100_c004",
                "zspace_s100_c0035",
                "zspace_s090_c0025",
            ],
        )
        self.assertEqual(
            [
                spec["label"]
                for spec in module.selected_routes(["zspace_s035_c025"])
            ],
            ["baseline", "zspace_s035_c025"],
        )

    def test_zspace_aggregate_rows_include_guard_margins(self):
        module = load_example("byte_lm_zspace_compare")

        def report(route, case, target_margin, retention_loss_margin):
            return {
                "route": route,
                "case": case,
                "source": {"loss_delta": 1.0, "accuracy_delta": 0.1, "perplexity_delta": 2.0},
                "ft": {"loss_delta": 0.5, "accuracy_delta": 0.2, "perplexity_delta": 1.5},
                "retention": {
                    "loss_delta": 0.25,
                    "accuracy_delta": 0.05,
                    "perplexity_delta": 0.75,
                },
                "ft_summary": {
                    "target_loss_margin": target_margin,
                    "retention_loss_margin": retention_loss_margin,
                    "retention_accuracy_margin": 0.2,
                },
                "projection": {
                    "projection_probe_samples": 2,
                    "projection_probe_rows": 16,
                    "projection_delta_input_l2_ratio": 0.25,
                    "projection_output_input_l2_ratio": 0.75,
                    "projection_output_input_col_variance_ratio": 0.5,
                },
            }

        route_specs = [
            {"label": "baseline", "strength": None, "curvature": -1.0},
            {"label": "zspace_s025", "strength": 0.25, "curvature": -1.0},
        ]
        reports = [
            report("baseline", "case_a", 0.01, 0.5),
            report("baseline", "case_b", 0.03, 0.7),
            report("zspace_s025", "case_a", 0.02, 0.6),
            report("zspace_s025", "case_b", 0.04, 0.8),
        ]
        aggregates = module.aggregate_reports(reports, route_specs)
        zspace = next(row for row in aggregates if row["route"] == "zspace_s025")
        self.assertEqual(zspace["accepted_cases"], 2)
        self.assertEqual(zspace["rejected_cases"], 0)
        self.assertEqual(zspace["accepted_rate"], 1.0)
        self.assertEqual(zspace["movement_ok_cases"], 2)
        self.assertEqual(zspace["movement_not_ok_cases"], 0)
        self.assertEqual(zspace["movement_ok_rate"], 1.0)
        self.assertEqual(zspace["case_labels"], "case_a,case_b")
        self.assertAlmostEqual(zspace["target_loss_margin_mean"], 0.03)
        self.assertAlmostEqual(zspace["target_loss_margin_min"], 0.02)
        self.assertAlmostEqual(zspace["retention_loss_margin_mean"], 0.7)
        self.assertAlmostEqual(zspace["retention_loss_margin_min"], 0.6)
        self.assertEqual(zspace["projection_probe_samples"], 4)
        self.assertEqual(zspace["projection_probe_rows"], 32)
        self.assertAlmostEqual(zspace["projection_delta_input_l2_ratio_mean"], 0.25)
        self.assertAlmostEqual(zspace["projection_output_input_l2_ratio_mean"], 0.75)
        self.assertAlmostEqual(
            zspace["projection_output_input_col_variance_ratio_mean"], 0.5
        )
        self.assertFalse(zspace["projection_variance_collapse_risk"])
        self.assertFalse(zspace["projection_norm_expansion_risk"])
        self.assertTrue(zspace["projection_healthy"])
        row = module.aggregate_row(
            zspace,
            next(row for row in aggregates if row["route"] == "baseline"),
            [{"label": "case_a"}, {"label": "case_b"}],
            route_specs,
        )
        self.assertEqual(row["row_type"], "route_aggregate")
        self.assertEqual(row["case_labels"], "case_a,case_b")
        self.assertAlmostEqual(row["loss_delta_advantage_sum"], 0.0)
        self.assertAlmostEqual(row["target_loss_margin_min"], 0.02)
        self.assertAlmostEqual(row["retention_loss_margin_min"], 0.6)
        self.assertAlmostEqual(row["projection_delta_input_l2_ratio_mean"], 0.25)
        self.assertTrue(row["projection_healthy"])

        ranked = module.ranked_route_rows(
            [
                dict(row, route="baseline"),
                dict(
                    row,
                    route="zspace_s050",
                    source_loss_delta_advantage=0.01,
                    ft_loss_delta_advantage=0.02,
                    retention_loss_delta_advantage=0.03,
                    loss_delta_advantage_sum=0.06,
                ),
            ]
        )
        self.assertEqual(ranked[0]["route"], "zspace_s050")

    def test_zspace_healthy_rank_skips_projection_risk_routes(self):
        module = load_example("byte_lm_zspace_compare")
        rows = [
            {
                "route": "baseline",
                "zspace_strength": None,
                "loss_delta_advantage_sum": 0.0,
                "ft_loss_delta_advantage": 0.0,
                "source_loss_delta_advantage": 0.0,
                "projection_healthy": False,
                "accepted_all": True,
                "movement_ok_all": True,
            },
            {
                "route": "zspace_edge",
                "zspace_strength": 1.0,
                "loss_delta_advantage_sum": 0.4,
                "ft_loss_delta_advantage": 0.2,
                "source_loss_delta_advantage": 0.1,
                "projection_healthy": False,
                "accepted_all": True,
                "movement_ok_all": True,
            },
            {
                "route": "zspace_safe",
                "zspace_strength": 0.75,
                "loss_delta_advantage_sum": 0.3,
                "ft_loss_delta_advantage": 0.12,
                "source_loss_delta_advantage": 0.09,
                "projection_healthy": True,
                "accepted_all": True,
                "movement_ok_all": True,
            },
        ]
        self.assertEqual(
            [row["route"] for row in module.healthy_ranked_route_rows(rows)],
            ["zspace_safe"],
        )

    def test_zspace_route_edge_diagnostics_flags_boundary_winner(self):
        module = load_example("byte_lm_zspace_compare")
        rows = [
            {
                "route": "baseline",
                "zspace_strength": None,
                "zspace_curvature": -1.0,
                "loss_delta_advantage_sum": 0.0,
                "ft_loss_delta_advantage": 0.0,
                "source_loss_delta_advantage": 0.0,
                "projection_delta_input_l2_ratio_mean": None,
                "projection_output_input_l2_ratio_mean": None,
                "projection_output_input_col_variance_ratio_mean": None,
            },
            {
                "route": "zspace_s100_c0025",
                "zspace_strength": 1.0,
                "zspace_curvature": -0.025,
                "loss_delta_advantage_sum": 0.2,
                "ft_loss_delta_advantage": 0.08,
                "source_loss_delta_advantage": 0.06,
                "projection_delta_input_l2_ratio_mean": 0.4,
                "projection_output_input_l2_ratio_mean": 0.8,
                "projection_output_input_col_variance_ratio_mean": 0.6,
            },
            {
                "route": "zspace_s100_c0005",
                "zspace_strength": 1.0,
                "zspace_curvature": -0.005,
                "loss_delta_advantage_sum": 0.3,
                "ft_loss_delta_advantage": 0.11,
                "source_loss_delta_advantage": 0.09,
                "projection_delta_input_l2_ratio_mean": 2.4,
                "projection_output_input_l2_ratio_mean": 3.4,
                "projection_output_input_col_variance_ratio_mean": 0.1,
            },
            {
                "route": "zspace_s075_c0005",
                "zspace_strength": 0.75,
                "zspace_curvature": -0.005,
                "loss_delta_advantage_sum": 0.25,
                "ft_loss_delta_advantage": 0.1,
                "source_loss_delta_advantage": 0.07,
                "projection_delta_input_l2_ratio_mean": 0.45,
                "projection_output_input_l2_ratio_mean": 0.85,
                "projection_output_input_col_variance_ratio_mean": 0.65,
            },
        ]
        edge = module.route_edge_diagnostics(rows)
        self.assertEqual(edge["route"], "zspace_s100_c0005")
        self.assertTrue(edge["strength_edge"])
        self.assertTrue(edge["near_zero_curvature_edge"])
        self.assertFalse(edge["steep_curvature_edge"])
        self.assertAlmostEqual(edge["max_strength"], 1.0)
        self.assertAlmostEqual(edge["min_abs_curvature"], 0.005)
        self.assertAlmostEqual(edge["projection_delta_input_l2_ratio_mean"], 2.4)
        self.assertAlmostEqual(
            edge["projection_output_input_col_variance_ratio_mean"], 0.1
        )
        self.assertTrue(edge["projection_variance_collapse_risk"])
        self.assertTrue(edge["projection_norm_expansion_risk"])

    def test_zspace_aggregate_rejects_inconsistent_case_labels(self):
        module = load_example("byte_lm_zspace_compare")
        row = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 2,
            "routes": 2,
            "case_labels": "case_a",
            "source_loss_delta_mean": 1.0,
            "ft_loss_delta_mean": 0.5,
            "retention_loss_delta_mean": 0.25,
            "source_loss_delta_advantage": 0.1,
            "ft_loss_delta_advantage": 0.05,
            "retention_loss_delta_advantage": 0.025,
        }
        with self.assertRaisesRegex(ValueError, "case_labels count 1 != cases 2"):
            module.aggregate_rows_by_route([row], "baseline")

    def test_zspace_aggregate_rejects_invalid_route_count(self):
        module = load_example("byte_lm_zspace_compare")
        row = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 2,
            "routes": 0,
            "case_labels": "case_a,case_b",
            "source_loss_delta_mean": 1.0,
            "ft_loss_delta_mean": 0.5,
            "retention_loss_delta_mean": 0.25,
            "source_loss_delta_advantage": 0.1,
            "ft_loss_delta_advantage": 0.05,
            "retention_loss_delta_advantage": 0.025,
        }
        with self.assertRaisesRegex(ValueError, "routes must be a positive integer"):
            module.aggregate_rows_by_route([row], "baseline")

    def test_zspace_aggregate_rejects_inconsistent_coverage_rates(self):
        module = load_example("byte_lm_zspace_compare")
        row = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 2,
            "routes": 2,
            "case_labels": "case_a,case_b",
            "accepted_cases": 1,
            "rejected_cases": 1,
            "accepted_rate": 1.0,
            "accepted_all": False,
            "movement_ok_cases": 2,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
            "source_loss_delta_mean": 1.0,
            "ft_loss_delta_mean": 0.5,
            "retention_loss_delta_mean": 0.25,
            "source_loss_delta_advantage": 0.1,
            "ft_loss_delta_advantage": 0.05,
            "retention_loss_delta_advantage": 0.025,
        }
        with self.assertRaisesRegex(ValueError, "accepted_rate 1.000000000"):
            module.aggregate_rows_by_route([row], "baseline")
        row["accepted_rate"] = True
        with self.assertRaisesRegex(ValueError, "accepted_rate is not numeric"):
            module.aggregate_rows_by_route([row], "baseline")

    def test_zspace_aggregate_rejects_nonboolean_summary_acceptance(self):
        module = load_example("byte_lm_zspace_compare")
        reports = [
            {
                "route": "zspace_s025",
                "ft_summary": {"accepted": "false", "movement_ok": True},
            }
        ]
        route_specs = [{"label": "zspace_s025", "strength": 0.25, "curvature": -1.0}]
        with self.assertRaisesRegex(ValueError, "accepted is not boolean"):
            module.aggregate_reports(reports, route_specs)

    def test_zspace_aggregate_coverage_gate_detects_rate_floors(self):
        module = load_example("byte_lm_zspace_compare")
        rows = [
            {
                "row_type": "route_aggregate",
                "route": "zspace_s025",
                "cases": 2,
                "routes": 2,
                "case_labels": "case_a,case_b",
                "accepted_cases": 1,
                "rejected_cases": 1,
                "accepted_rate": 0.5,
                "accepted_all": False,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 1,
                "movement_ok_rate": 0.5,
                "movement_ok_all": False,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
            module.check_aggregate_coverage(rows, min_accepted_rate=0.75)
        with self.assertRaisesRegex(RuntimeError, "movement_ok_rate"):
            module.check_aggregate_coverage(rows, min_movement_ok_rate=0.75)

    def test_zspace_aggregate_coverage_requires_acceptance_fields(self):
        module = load_example("byte_lm_zspace_compare")
        row = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 1,
            "routes": 2,
            "case_labels": "byte_patterns_to_jp",
            "accepted_cases": 1,
            "rejected_cases": 0,
            "accepted_rate": 1.0,
            "accepted_all": True,
            "movement_ok_cases": 1,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "missing boolean movement_ok_all"):
            module.check_aggregate_coverage([row])
        row["movement_ok_all"] = True
        row.pop("accepted_cases")
        with self.assertRaisesRegex(ValueError, "missing integer accepted_cases"):
            module.check_aggregate_coverage([row])

    def test_zspace_aggregate_coverage_gate_detects_case_scope(self):
        module = load_example("byte_lm_zspace_compare")
        rows = [
            {
                "row_type": "route_aggregate",
                "route": "zspace_s025",
                "cases": 1,
                "routes": 2,
                "case_labels": "byte_patterns_to_jp",
                "accepted_cases": 1,
                "rejected_cases": 0,
                "accepted_rate": 1.0,
                "accepted_all": True,
                "movement_ok_cases": 1,
                "movement_not_ok_cases": 0,
                "movement_ok_rate": 1.0,
                "movement_ok_all": True,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "aggregate cases 1 below floor 2"):
            module.check_aggregate_coverage(rows, min_cases=2)
        with self.assertRaisesRegex(
            RuntimeError,
            "missing aggregate cases byte_patterns_to_cats",
        ):
            module.check_aggregate_coverage(
                rows,
                required_cases=["byte_patterns_to_jp", "byte_patterns_to_cats"],
            )

    def test_zspace_aggregate_compare_detects_coverage_regressions(self):
        module = load_example("byte_lm_zspace_compare")
        baseline = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 2,
            "routes": 2,
            "case_labels": "case_a,case_b",
            "accepted_cases": 2,
            "rejected_cases": 0,
            "accepted_rate": 1.0,
            "accepted_all": True,
            "movement_ok_cases": 2,
            "movement_not_ok_cases": 0,
            "movement_ok_rate": 1.0,
            "movement_ok_all": True,
            "source_loss_delta_mean": 1.0,
            "ft_loss_delta_mean": 0.5,
            "retention_loss_delta_mean": 0.25,
            "source_loss_delta_advantage": 0.1,
            "ft_loss_delta_advantage": 0.05,
            "retention_loss_delta_advantage": 0.025,
        }
        current = dict(
            baseline,
            accepted_cases=1,
            rejected_cases=1,
            accepted_rate=0.5,
            accepted_all=False,
        )
        with self.assertRaisesRegex(RuntimeError, "accepted_rate"):
            module.compare_aggregate_rows(
                [current],
                [baseline],
                max_source_loss_regression=None,
                max_ft_loss_regression=None,
                max_retention_loss_regression=None,
                min_target_loss_margin=None,
                min_retention_loss_margin=None,
                min_retention_accuracy_margin=None,
                min_retention_perplexity_margin=None,
                require_winner_match=False,
                max_aggregate_accepted_rate_regression=0.25,
                max_aggregate_movement_ok_rate_regression=None,
            )

    def test_zspace_aggregate_compare_detects_case_scope_drift(self):
        module = load_example("byte_lm_zspace_compare")
        baseline = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 2,
            "routes": 2,
            "case_labels": "case_a,case_b",
            "source_loss_delta_mean": 1.0,
            "ft_loss_delta_mean": 0.5,
            "retention_loss_delta_mean": 0.25,
            "source_loss_delta_advantage": 0.1,
            "ft_loss_delta_advantage": 0.05,
            "retention_loss_delta_advantage": 0.025,
        }
        current = dict(
            baseline,
            cases=1,
            case_labels="case_a",
        )
        with self.assertRaisesRegex(RuntimeError, "aggregate case scope changed"):
            module.compare_aggregate_rows(
                [current],
                [baseline],
                max_source_loss_regression=None,
                max_ft_loss_regression=None,
                max_retention_loss_regression=None,
                min_target_loss_margin=None,
                min_retention_loss_margin=None,
                min_retention_accuracy_margin=None,
                min_retention_perplexity_margin=None,
                require_winner_match=False,
                max_aggregate_accepted_rate_regression=None,
                max_aggregate_movement_ok_rate_regression=None,
            )

    def test_zspace_require_advantage_can_allow_exploratory_nonadvantage(self):
        module = load_example("byte_lm_zspace_compare")
        with self.assertRaisesRegex(RuntimeError, "did not beat baseline"):
            module.require_advantage("source", "zspace_s025", -0.01)
        self.assertFalse(
            module.require_advantage(
                "source",
                "zspace_s025",
                -0.01,
                allow_nonadvantage=True,
            )
        )
        self.assertTrue(module.require_advantage("source", "zspace_s025", 0.01))

    def test_zspace_aggregate_compare_gates_guard_margin_floors(self):
        module = load_example("byte_lm_zspace_compare")
        baseline = {
            "row_type": "route_aggregate",
            "route": "zspace_s025",
            "cases": 2,
            "routes": 2,
            "case_labels": "case_a,case_b",
            "source_loss_delta_mean": 1.0,
            "ft_loss_delta_mean": 0.5,
            "retention_loss_delta_mean": 0.25,
            "source_loss_delta_advantage": 0.1,
            "ft_loss_delta_advantage": 0.05,
            "retention_loss_delta_advantage": 0.025,
            "target_loss_margin_min": 0.02,
            "retention_loss_margin_min": 0.6,
            "retention_accuracy_margin_min": 0.2,
        }
        current = dict(baseline)
        current["target_loss_margin_min"] = 0.005
        current["retention_loss_margin_min"] = 0.4
        with self.assertRaisesRegex(RuntimeError, "target_loss_margin_min"):
            module.compare_aggregate_rows(
                [current],
                [baseline],
                max_source_loss_regression=None,
                max_ft_loss_regression=None,
                max_retention_loss_regression=None,
                min_target_loss_margin=0.01,
                min_retention_loss_margin=0.5,
                min_retention_accuracy_margin=None,
                min_retention_perplexity_margin=None,
                require_winner_match=False,
            )

    def test_examples_allow_accepted_change_when_gate_is_disabled(self):
        baseline = {"accepted": True}
        current = {"accepted": False}
        helper = load_compare_helper()
        helper.compare_sparse_finetune_summaries = old_compare_signature
        comparison = helper.compare_summaries(
            current,
            baseline,
            max_target_loss_regression=None,
            max_retention_loss_regression=None,
            min_target_loss_margin=None,
            min_retention_loss_margin=None,
            min_retention_accuracy_margin=None,
            min_retention_perplexity_margin=None,
            require_status_match=False,
            require_accepted_match=False,
            require_guard_match=False,
            require_movement_tolerance_match=False,
            require_resume_match=False,
        )
        self.assertTrue(comparison["accepted_changed"])
        self.assertTrue(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
