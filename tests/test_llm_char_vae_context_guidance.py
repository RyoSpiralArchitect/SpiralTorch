from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "models" / "python" / "llm_char_vae_context.py"
_MISSING = object()


def _load_module():
    module_name = "_spiraltorch_llm_char_vae_context_under_test"
    cached = sys.modules.get(module_name)
    if cached is not None and hasattr(cached, "_follow_up_guidance_record"):
        return cached
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_spiraltorch = sys.modules.get("spiraltorch", _MISSING)
    sys.modules["spiraltorch"] = types.ModuleType("spiraltorch")
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_spiraltorch is _MISSING:
            sys.modules.pop("spiraltorch", None)
        else:
            sys.modules["spiraltorch"] = previous_spiraltorch
    return module


def _ancestor(*, feature: str, nll: float, raw_delta: float | None = None) -> dict:
    best_config = {
        "best_feature": feature,
        "feature_normalize": "blocks",
        "hybrid_latent_scale": 1.0,
        "mean_best_nll": nll,
    }
    if raw_delta is not None:
        best_config["mean_best_nll_delta_vs_raw"] = raw_delta
    return {
        "schema": "st.llm_char_vae_context.follow_up_ancestor.v1",
        "summary_path": "/tmp/source/summary.json",
        "run_dir": "/tmp/source",
        "generation": 0,
        "status": "improved",
        "best_feature": feature,
        "best_config": best_config,
        "mean_best_nll": nll,
        "verdict": None,
        "guidance_action": None,
        "guided_enabled": None,
        "gate_failed": None,
    }


def _summary(
    *,
    best_feature: str,
    nll: float,
    verdict: str,
    config_verdict: str,
    source_feature_verdict: str,
    source_retained: bool,
    gate_failed: bool,
    source_feature_raw_verdict: str = "unknown",
    source_feature_delta_vs_raw: float | None = None,
) -> dict:
    return {
        "status": "improved",
        "best_feature": best_feature,
        "best_config": {
            "best_feature": best_feature,
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 1.0,
            "mean_best_nll": nll,
        },
        "follow_up_result": {
            "verdict": verdict,
            "config_verdict": config_verdict,
            "source_feature_verdict": source_feature_verdict,
            "source_feature_raw_verdict": source_feature_raw_verdict,
            "source_feature_mean_best_nll_delta_vs_raw": source_feature_delta_vs_raw,
            "source_best_feature_retained": source_retained,
        },
        "follow_up_chain": {
            "generation": 1,
            "latest_verdict": verdict,
            "verdict_history": [verdict],
            "improved_streak": 1 if verdict == "improved" else 0,
            "regressed_streak": 1 if verdict == "regressed" else 0,
        },
        "follow_up_gate": {
            "failed": gate_failed,
            "verdict": verdict,
            "fail_on_verdicts": ["regressed", "unknown"],
        },
    }


def _next_follow_up() -> dict:
    return {
        "script_path": "/tmp/current/next_follow_up_command.sh",
        "script_usage": "NEW_SEEDS=17 bash /tmp/current/next_follow_up_command.sh",
        "default_follow_up_from": "/tmp/current/summary.json",
        "default_new_seeds": "17",
        "default_run_dir": "/tmp/current/follow_up_best_config",
        "default_follow_up_fail_on_verdict": "regressed,unknown",
        "shell_command": "PYTHONNOUSERSITE=1 python3 ...",
        "script_command": ["python3", "models/python/llm_char_vae_context.py"],
    }


class CharVaeContextGuidanceTests(unittest.TestCase):
    def test_next_follow_up_avoids_source_and_current_seed_history(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["models/samples/spiral_corpus_en"])
        best_config = {
            "best_feature": "raw_latent",
            "feature_normalize": "blocks",
            "hybrid_latent_scale": 4.0,
            "mean_best_nll": 4.2,
        }
        follow_up = {
            "source_seeds": [101, 103, 107],
            "source_chain": {"ancestors": []},
            "resolved": {"seeds": [109, 113, 127]},
        }

        record = mod._next_follow_up_command_record(
            args,
            ["raw", "latent", "raw_latent"],
            best_config,
            Path("/tmp/current"),
            [109, 113, 127],
            follow_up,
        )

        self.assertEqual(record["default_new_seeds"], "131,137,139")
        self.assertEqual(
            record["used_seed_history"],
            [101, 103, 107, 109, 113, 127],
        )

    def test_default_follow_up_seeds_refreshes_stale_seed_history(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_summary = root / "parent_summary.json"
            first_follow_up_summary = root / "first_follow_up_summary.json"
            parent_summary.write_text(
                json.dumps({"run": {"seeds": [7, 13, 17]}}),
                encoding="utf-8",
            )
            first_follow_up_summary.write_text(
                json.dumps({"run": {"seeds": [101, 103, 107]}}),
                encoding="utf-8",
            )
            summary = {
                "run": {"seeds": [109, 113, 127]},
                "follow_up_chain": {
                    "generation": 2,
                    "ancestors": [str(parent_summary), str(first_follow_up_summary)],
                },
                "next_follow_up_command": {"default_new_seeds": "101,103,107"},
            }

            self.assertEqual(mod._default_follow_up_seeds(summary), "131,137,139")

    def test_safe_trajectory_promotes_guided_confirmation(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="reconstruction_latent",
            nll=4.2,
            verdict="improved",
            config_verdict="improved",
            source_feature_verdict="improved",
            source_retained=True,
            gate_failed=False,
        )
        next_follow_up = _next_follow_up()
        preliminary = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            next_follow_up,
        )
        summary["follow_up_guidance"] = preliminary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {"ancestors": [_ancestor(feature="reconstruction_latent", nll=4.3)]},
                min_nll_delta=0.0,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                next_follow_up,
            )

        self.assertEqual(
            trajectory["trajectory_action"],
            "confirm_trajectory_with_fresh_seeds",
        )
        self.assertIs(trajectory["unsafe_promotion"], False)
        self.assertEqual(guidance["local_action"], "continue_fresh_seed_confirmation")
        self.assertEqual(guidance["action"], "confirm_trajectory_with_fresh_seeds")
        self.assertIs(guidance["promote_current_best"], True)
        self.assertIs(guidance["use_next_follow_up_command"], True)
        self.assertIs(guided["enabled"], True)
        self.assertEqual(
            guided["trajectory_action"],
            "confirm_trajectory_with_fresh_seeds",
        )
        self.assertIs(guided["unsafe_promotion"], False)

    def test_unsafe_trajectory_blocks_guided_promotion(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="latent",
            nll=4.1,
            verdict="regressed",
            config_verdict="improved",
            source_feature_verdict="regressed",
            source_retained=False,
            gate_failed=True,
        )
        next_follow_up = _next_follow_up()
        preliminary = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            next_follow_up,
        )
        summary["follow_up_guidance"] = preliminary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {"ancestors": [_ancestor(feature="reconstruction_latent", nll=4.3)]},
                min_nll_delta=0.0,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                next_follow_up,
            )

        self.assertEqual(trajectory["trajectory_verdict"], "improved")
        self.assertEqual(
            trajectory["trajectory_action"],
            "audit_feature_swap_before_promotion",
        )
        self.assertIs(trajectory["source_feature_tradeoff"], True)
        self.assertIs(trajectory["unsafe_promotion"], True)
        self.assertEqual(guidance["local_action"], "stop_on_follow_up_gate")
        self.assertEqual(guidance["action"], "audit_feature_swap_before_promotion")
        self.assertIs(guidance["promote_current_best"], False)
        self.assertIs(guidance["use_next_follow_up_command"], False)
        self.assertIn("trajectory marked promotion unsafe", guidance["reasons"])
        self.assertIs(guided["enabled"], False)
        self.assertIsNone(guided["script_path"])
        self.assertEqual(
            guided["trajectory_action"],
            "audit_feature_swap_before_promotion",
        )
        self.assertIs(guided["unsafe_promotion"], True)

    def test_retained_source_feature_regression_has_precise_reason(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="latent",
            nll=4.23,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_retained=True,
            gate_failed=True,
        )

        guidance = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            _next_follow_up(),
        )

        self.assertEqual(guidance["action"], "stop_on_follow_up_gate")
        self.assertIn(
            "source best feature regressed on fresh seeds",
            guidance["reasons"],
        )
        self.assertNotIn(
            "source best feature did not retain its role",
            guidance["reasons"],
        )

    def test_raw_positive_gate_stop_widens_seed_confirmation(self) -> None:
        mod = _load_module()
        summary = _summary(
            best_feature="raw_latent",
            nll=4.223,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_feature_raw_verdict="improved",
            source_feature_delta_vs_raw=-0.008,
            source_retained=True,
            gate_failed=True,
        )
        next_follow_up = _next_follow_up()
        preliminary = mod._follow_up_guidance_record(
            summary["follow_up_result"],
            summary["follow_up_chain"],
            summary["follow_up_gate"],
            next_follow_up,
        )
        summary["follow_up_guidance"] = preliminary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {"ancestors": [_ancestor(feature="raw_latent", nll=4.216)]},
                min_nll_delta=0.0,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                next_follow_up,
            )

        self.assertEqual(
            preliminary["action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )
        self.assertEqual(
            trajectory["trajectory_action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )
        self.assertIs(trajectory["current_raw_positive"], True)
        self.assertIs(trajectory["unsafe_promotion"], False)
        self.assertEqual(trajectory["raw_evidence_count"], 1)
        self.assertEqual(trajectory["raw_positive_count"], 1)
        self.assertEqual(trajectory["raw_positive_rate"], 1.0)
        self.assertEqual(trajectory["raw_positive_streak"], 1)
        self.assertEqual(trajectory["raw_negative_streak"], 0)
        self.assertAlmostEqual(trajectory["current_raw_delta_vs_raw"], -0.008)
        self.assertAlmostEqual(trajectory["mean_raw_delta_vs_raw"], -0.008)
        self.assertIs(trajectory["points"][-1]["raw_positive"], True)
        self.assertEqual(
            guidance["action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )
        self.assertIs(guidance["promote_current_best"], False)
        self.assertIs(guidance["use_next_follow_up_command"], True)
        self.assertIn("source best feature remained raw-positive", guidance["reasons"])
        self.assertIs(guided["enabled"], True)
        self.assertEqual(
            guided["guidance_action"],
            "widen_seed_confirmation_on_raw_positive_regression",
        )

        summary["follow_up_trajectory"] = trajectory
        report = mod._aggregate_report(summary)
        self.assertIn("- raw_positive_points: 1/1 (100.0%)", report)
        self.assertIn("raw_delta_vs_raw", report)

    def test_raw_positive_streak_reconfirms_best_generation(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["models/samples/spiral_corpus_en"])
        summary = _summary(
            best_feature="raw_latent",
            nll=4.214,
            verdict="regressed",
            config_verdict="regressed",
            source_feature_verdict="regressed",
            source_feature_raw_verdict="improved",
            source_feature_delta_vs_raw=-0.017,
            source_retained=True,
            gate_failed=True,
        )
        summary["follow_up_chain"]["verdict_history"] = [
            "regressed",
            "improved",
            "regressed",
            "regressed",
        ]
        summary["follow_up_chain"]["latest_verdict"] = "regressed"
        next_follow_up = _next_follow_up()
        next_follow_up["used_seed_history"] = [
            7,
            13,
            17,
            19,
            23,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            1001,
            1003,
            1005,
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = mod._follow_up_trajectory_record(
                root,
                summary,
                {
                    "ancestors": [
                        _ancestor(feature="raw_latent", nll=4.216, raw_delta=-0.014),
                        _ancestor(feature="raw_latent", nll=4.207, raw_delta=-0.024),
                        _ancestor(feature="raw_latent", nll=4.211, raw_delta=-0.019),
                    ]
                },
                min_nll_delta=0.0,
            )
            best_generation = mod._best_generation_follow_up_command_record(
                args,
                ["raw", "latent", "raw_latent"],
                root,
                [1001, 1003, 1005],
                trajectory,
                next_follow_up,
            )
            guidance = mod._follow_up_guidance_record(
                summary["follow_up_result"],
                summary["follow_up_chain"],
                summary["follow_up_gate"],
                next_follow_up,
                trajectory,
                best_generation,
            )
            guided = mod._guided_next_follow_up_command_record(
                root,
                guidance,
                best_generation,
            )

        self.assertEqual(
            trajectory["trajectory_action"],
            "reconfirm_best_raw_positive_generation",
        )
        self.assertEqual(trajectory["raw_positive_streak"], 4)
        self.assertEqual(trajectory["best_mean_best_nll"], 4.207)
        self.assertIsNotNone(best_generation)
        self.assertEqual(
            best_generation["action"],
            "reconfirm_best_raw_positive_generation",
        )
        self.assertEqual(best_generation["default_new_seeds"], "1007,1009,1011")
        self.assertEqual(
            best_generation["best_summary_path"],
            "/tmp/source/summary.json",
        )
        self.assertIn("--follow-up-used-seeds", best_generation["script_command"])
        self.assertIn(
            (
                "7,13,17,19,23,101,103,107,109,113,127,131,137,139,"
                "149,151,157,1001,1003,1005"
            ),
            best_generation["script_command"],
        )
        self.assertEqual(guidance["action"], "reconfirm_best_raw_positive_generation")
        self.assertIs(guidance["use_next_follow_up_command"], False)
        self.assertIs(guidance["use_best_generation_follow_up_command"], True)
        self.assertEqual(guidance["command_usage"], best_generation["script_usage"])
        self.assertIs(guided["enabled"], True)
        self.assertEqual(guided["default_follow_up_from"], "/tmp/source/summary.json")
        self.assertEqual(guided["default_new_seeds"], "1007,1009,1011")


if __name__ == "__main__":
    unittest.main()
