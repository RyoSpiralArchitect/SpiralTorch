#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_char_vae_context_chain.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_char_vae_context_chain", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharVaeContextChainTests(unittest.TestCase):
    def test_help_documents_tie_aware_seed_group_override(self) -> None:
        mod = _load_module()
        help_text = " ".join(mod._build_parser().format_help().split())

        self.assertIn("--follow-up-seed-groups", help_text)
        self.assertIn("supplied groups override matching follow-ups", help_text)
        self.assertIn("unspecified follow-ups still use generated", help_text)
        self.assertIn("tie-aware default_new_seeds", help_text)
        self.assertIn("extra groups beyond --follow-ups and planned", help_text)
        self.assertIn("reported separately", help_text)

    def test_follow_up_seed_policy_records_precedence(self) -> None:
        mod = _load_module()

        implicit = mod._follow_up_seed_policy_record(explicit_seed_groups=False)
        explicit = mod._follow_up_seed_policy_record(explicit_seed_groups=True)

        self.assertEqual(
            implicit["precedence"],
            ["command_default", "preset_seed_group", "script_default"],
        )
        self.assertIn("default_new_seeds wins", implicit["reason"])
        self.assertEqual(
            explicit["precedence"],
            ["explicit_seed_group", "command_default", "script_default"],
        )
        self.assertIn("overrides matching follow-ups", explicit["reason"])
        self.assertIn("backfills unspecified follow-ups", explicit["reason"])

    def test_extra_explicit_seed_groups_reports_unmatched_groups(self) -> None:
        mod = _load_module()

        self.assertEqual(
            mod._extra_explicit_seed_groups(
                ["17", "19", "23"],
                explicit_seed_groups=True,
                follow_up_count=1,
            ),
            ["19", "23"],
        )
        self.assertEqual(
            mod._extra_explicit_seed_groups(
                ["17"],
                explicit_seed_groups=True,
                follow_up_count=2,
            ),
            [],
        )
        self.assertEqual(
            mod._extra_explicit_seed_groups(
                ["17", "19"],
                explicit_seed_groups=False,
                follow_up_count=1,
            ),
            [],
        )

    def test_unused_explicit_seed_groups_reports_unattempted_groups(self) -> None:
        mod = _load_module()
        manifest = {
            "schema": mod.SCHEMA,
            "preset": "smoke",
            "run_root": "/tmp/chain",
            "allowed_gate_stop": False,
            "steps": [
                {"index": 0, "role": "parent"},
                {"index": 1, "role": "follow_up"},
            ],
            "follow_up_seed_group_source": "explicit",
            "planned_follow_up_seed_groups": ["17", "19", "23"],
            "extra_explicit_seed_groups": [],
            "unused_explicit_seed_groups": ["19", "23"],
            "follow_up_seed_group_plan": [
                {
                    "group_index": 1,
                    "follow_up_index": 1,
                    "seed_group": "17",
                    "source": "explicit",
                    "planned": True,
                    "attempted": True,
                    "status": "attempted_slot",
                },
                {
                    "group_index": 2,
                    "follow_up_index": 2,
                    "seed_group": "19",
                    "source": "explicit",
                    "planned": True,
                    "attempted": False,
                    "status": "unused_after_stop",
                },
            ],
            "follow_up_seed_policy": mod._follow_up_seed_policy_record(
                explicit_seed_groups=True
            ),
        }
        report = mod._render_report(manifest)

        self.assertEqual(mod._attempted_follow_up_count(manifest), 1)
        self.assertIn("- unused_explicit_seed_groups: 19, 23", report)
        self.assertIn(
            "- follow_up_seed_group_plan: #1 source=explicit -> follow_up=1 "
            "status=attempted_slot seeds=17; #2 source=explicit -> follow_up=2 "
            "status=unused_after_stop seeds=19",
            report,
        )
        self.assertEqual(
            mod._unused_explicit_seed_groups(
                ["17", "19", "23"],
                explicit_seed_groups=True,
                attempted_follow_ups=1,
                planned_follow_ups=3,
            ),
            ["19", "23"],
        )
        self.assertEqual(
            mod._unused_explicit_seed_groups(
                ["17", "19", "23"],
                explicit_seed_groups=True,
                attempted_follow_ups=1,
                planned_follow_ups=1,
            ),
            [],
        )
        self.assertEqual(
            mod._unused_explicit_seed_groups(
                ["17", "19"],
                explicit_seed_groups=True,
                attempted_follow_ups=2,
                planned_follow_ups=2,
            ),
            [],
        )
        self.assertEqual(
            mod._unused_explicit_seed_groups(
                ["17", "19"],
                explicit_seed_groups=False,
                attempted_follow_ups=1,
                planned_follow_ups=2,
            ),
            [],
        )

    def test_follow_up_seed_group_plan_maps_attempted_unused_and_extra(self) -> None:
        mod = _load_module()

        plan = mod._follow_up_seed_group_plan(
            ["17", "19", "23"],
            explicit_seed_groups=True,
            planned_follow_ups=2,
            attempted_follow_ups=1,
            dry_run=False,
        )

        self.assertEqual(
            plan,
            [
                {
                    "group_index": 1,
                    "follow_up_index": 1,
                    "seed_group": "17",
                    "source": "explicit",
                    "planned": True,
                    "attempted": True,
                    "status": "attempted_slot",
                },
                {
                    "group_index": 2,
                    "follow_up_index": 2,
                    "seed_group": "19",
                    "source": "explicit",
                    "planned": True,
                    "attempted": False,
                    "status": "unused_after_stop",
                },
                {
                    "group_index": 3,
                    "follow_up_index": None,
                    "seed_group": "23",
                    "source": "explicit",
                    "planned": False,
                    "attempted": False,
                    "status": "extra",
                },
            ],
        )

        dry_run_plan = mod._follow_up_seed_group_plan(
            ["17", "19"],
            explicit_seed_groups=True,
            planned_follow_ups=1,
            attempted_follow_ups=0,
            dry_run=True,
        )
        self.assertEqual(dry_run_plan[0]["status"], "dry_run_planned")
        self.assertEqual(dry_run_plan[1]["status"], "extra")

    def test_preset_latent_scale_defaults_keep_smoke_light_and_scout_small(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()

        smoke = parser.parse_args(["models/samples/spiral_corpus_en", "--preset", "smoke"])
        smoke_command = mod._parent_command(smoke, Path("/tmp/smoke"))
        self.assertEqual(
            smoke_command[smoke_command.index("--hybrid-latent-scales") + 1],
            "0.5,1.0",
        )
        self.assertEqual(
            smoke_command[smoke_command.index("--feature-normalize-modes") + 1],
            "blocks,vector",
        )
        self.assertEqual(
            smoke_command[smoke_command.index("--head-init") + 1],
            "legacy",
        )

        small = parser.parse_args(["models/samples/spiral_corpus_en", "--preset", "small"])
        small_command = mod._parent_command(small, Path("/tmp/small"))
        self.assertEqual(
            small_command[small_command.index("--hybrid-latent-scales") + 1],
            "0.5,1.0,2.0,4.0",
        )

        hybrid4 = parser.parse_args(
            ["models/samples/spiral_corpus_en", "--preset", "hybrid4"]
        )
        hybrid4_command = mod._parent_command(hybrid4, Path("/tmp/hybrid4"))
        self.assertEqual(
            hybrid4_command[hybrid4_command.index("--features") + 1],
            "raw,latent,raw_latent,reconstruction_latent",
        )
        self.assertEqual(
            hybrid4_command[hybrid4_command.index("--feature-normalize-modes") + 1],
            "blocks",
        )
        self.assertEqual(
            hybrid4_command[hybrid4_command.index("--hybrid-latent-scales") + 1],
            "2.0,4.0",
        )
        self.assertEqual(
            hybrid4_command[hybrid4_command.index("--head-init") + 1],
            "xavier",
        )
        self.assertEqual(hybrid4_command[hybrid4_command.index("--epochs") + 1], "8")
        self.assertEqual(hybrid4_command[hybrid4_command.index("--batches") + 1], "16")
        self.assertEqual(
            hybrid4_command[hybrid4_command.index("--eval-samples") + 1],
            "128",
        )

        hybrid4_deep = parser.parse_args(
            ["models/samples/spiral_corpus_en", "--preset", "hybrid4_deep"]
        )
        hybrid4_deep_command = mod._parent_command(
            hybrid4_deep,
            Path("/tmp/hybrid4_deep"),
        )
        self.assertEqual(
            hybrid4_deep_command[
                hybrid4_deep_command.index("--hybrid-latent-scales") + 1
            ],
            "4.0",
        )
        self.assertEqual(
            hybrid4_deep_command[hybrid4_deep_command.index("--head-init") + 1],
            "xavier",
        )
        self.assertEqual(
            hybrid4_deep_command[hybrid4_deep_command.index("--epochs") + 1],
            "10",
        )
        self.assertEqual(
            hybrid4_deep_command[hybrid4_deep_command.index("--batches") + 1],
            "24",
        )
        self.assertEqual(
            hybrid4_deep_command[hybrid4_deep_command.index("--eval-samples") + 1],
            "256",
        )

        explicit = parser.parse_args(
            [
                "models/samples/spiral_corpus_en",
                "--preset",
                "small",
                "--hybrid-latent-scales",
                "3.0",
            ]
        )
        explicit_command = mod._parent_command(explicit, Path("/tmp/explicit"))
        self.assertEqual(
            explicit_command[explicit_command.index("--hybrid-latent-scales") + 1],
            "3.0",
        )

    def test_step_record_and_report_surface_follow_up_deltas(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "follow_up_01"
            run_dir.mkdir()
            summary = {
                "status": "improved",
                "best_feature": "latent",
                "best_config": {
                    "best_feature": "latent",
                    "feature_normalize": "blocks",
                    "hybrid_latent_scale": 0.5,
                    "mean_best_nll": 4.2258,
                    "mean_best_accuracy": 0.104,
                    "mean_best_nll_delta_vs_raw": -0.0023,
                    "runner_up_feature": "raw_latent",
                    "runner_up_mean_best_nll": 4.2261,
                    "margin_to_runner_up": 0.0003,
                    "combined_runner_up_margin_stderr": 0.0005,
                    "runner_up_within_uncertainty": True,
                },
                "follow_up_result": {
                    "verdict": "regressed",
                    "mean_best_nll_delta_vs_source": 0.001,
                    "source_feature_mean_best_nll_delta_vs_source": 0.001,
                    "source_best_feature_retained": True,
                },
                "follow_up_gate": {"failed": True},
                "follow_up_trajectory": {
                    "trajectory_action": "stop_on_follow_up_gate",
                    "trajectory_verdict": "regressed",
                },
                "follow_up_guidance": {
                    "action": "stop_on_follow_up_gate",
                    "unsafe_promotion": True,
                },
                "guided_next_follow_up_command": {"enabled": False},
                "next_follow_up_command": {
                    "default_new_seed_count": 5,
                    "default_new_seeds": "131,137,139,149,151",
                    "seed_confirmation_policy": {
                        "reason": "best runner-up within combined seed uncertainty",
                        "uncertainty_tie_seed_boost": True,
                    },
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            step = mod._step_record(
                index=1,
                role="follow_up",
                run_dir=run_dir,
                command=["bash", "next_follow_up_command.sh"],
                exit_code=1,
                dry_run=False,
            )
            step["new_seeds"] = "131,137,139,149,151"
            step["new_seed_source"] = "command_default"
            report = mod._render_report(
                {
                    "schema": mod.SCHEMA,
                    "preset": "base",
                    "run_root": str(run_dir.parent),
                    "steps": [step],
                    "allowed_gate_stop": True,
                    "follow_up_seed_group_source": "preset_fallback",
                    "planned_follow_up_seed_groups": ["17", "19"],
                    "extra_explicit_seed_groups": [],
                    "unused_explicit_seed_groups": [],
                    "follow_up_seed_group_plan": [],
                    "follow_up_seed_policy": mod._follow_up_seed_policy_record(
                        explicit_seed_groups=False
                    ),
                }
            )

        self.assertEqual(step["mean_best_nll_delta_vs_raw"], -0.0023)
        self.assertEqual(step["runner_up_feature"], "raw_latent")
        self.assertAlmostEqual(step["margin_to_runner_up"], 0.0003)
        self.assertAlmostEqual(step["combined_runner_up_margin_stderr"], 0.0005)
        self.assertIs(step["runner_up_within_uncertainty"], True)
        self.assertEqual(step["mean_best_nll_delta_vs_source"], 0.001)
        self.assertEqual(step["source_feature_mean_best_nll_delta_vs_source"], 0.001)
        self.assertIs(step["source_best_feature_retained"], True)
        self.assertIs(step["follow_up_gate_failed"], True)
        self.assertEqual(step["next_default_new_seed_count"], 5)
        self.assertEqual(step["next_default_new_seeds"], "131,137,139,149,151")
        self.assertEqual(
            step["seed_policy_reason"],
            "best runner-up within combined seed uncertainty",
        )
        self.assertIs(step["uncertainty_tie_seed_boost"], True)
        self.assertEqual(step["best_config_label"], "latent@normalize=blocks,scale=0.5")
        self.assertIn("delta_vs_raw", report)
        self.assertIn("runner_up", report)
        self.assertIn("raw_latent", report)
        self.assertIn("0.000300", report)
        self.assertIn("0.000500", report)
        self.assertIn("within_uncertainty", report)
        self.assertIn("next_seed_count", report)
        self.assertIn("tie_seed_boost", report)
        self.assertIn("best runner-up within combined seed uncertainty", report)
        self.assertIn("run_seed_source", report)
        self.assertIn("command_default", report)
        self.assertIn("- follow_up_seed_groups: preset_fallback (17, 19)", report)
        self.assertIn("- extra_explicit_seed_groups: -", report)
        self.assertIn("- unused_explicit_seed_groups: -", report)
        self.assertIn(
            "- follow_up_seed_policy: command_default -> preset_seed_group -> "
            "script_default",
            report,
        )
        self.assertIn("source_feature_delta_vs_source", report)
        self.assertIn("latent@normalize=blocks,scale=0.5", report)
        self.assertIn("0.001000", report)

    def test_follow_up_new_seeds_prefers_command_defaults_until_explicit(self) -> None:
        mod = _load_module()
        command_record = {
            "default_new_seeds": "131,137,139,149,151",
            "script_path": "/tmp/next_follow_up_command.sh",
        }

        seeds, source = mod._follow_up_new_seeds(
            command_record,
            ["17"],
            index=1,
            explicit_seed_groups=False,
        )
        self.assertEqual(seeds, "131,137,139,149,151")
        self.assertEqual(source, "command_default")

        seeds, source = mod._follow_up_new_seeds(
            command_record,
            ["17"],
            index=1,
            explicit_seed_groups=True,
        )
        self.assertEqual(seeds, "17")
        self.assertEqual(source, "explicit_seed_group")

        seeds, source = mod._follow_up_new_seeds(
            command_record,
            ["17"],
            index=2,
            explicit_seed_groups=True,
        )
        self.assertEqual(seeds, "131,137,139,149,151")
        self.assertEqual(source, "command_default")

        seeds, source = mod._follow_up_new_seeds(
            {},
            ["19,23"],
            index=1,
            explicit_seed_groups=False,
        )
        self.assertEqual(seeds, "19,23")
        self.assertEqual(source, "preset_seed_group")

    def test_follow_up_command_record_prefers_enabled_guided_script(self) -> None:
        mod = _load_module()
        summary = {
            "next_follow_up_command": {"script_path": "/tmp/next.sh"},
            "guided_next_follow_up_command": {
                "enabled": True,
                "script_path": "/tmp/guided.sh",
            },
        }

        record, source = mod._follow_up_command_record(summary, index=2)

        self.assertEqual(record["script_path"], "/tmp/guided.sh")
        self.assertEqual(source, "guided_next_follow_up_command")

    def test_chain_selection_retains_parent_when_follow_up_gate_stops(self) -> None:
        mod = _load_module()
        parent = {
            "index": 0,
            "role": "parent",
            "run_dir": "/tmp/chain/parent",
            "summary_path": "/tmp/chain/parent/summary.json",
            "exit_code": 0,
            "status": "improved",
            "best_feature": "raw_latent",
            "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
            "best_config": {
                "best_feature": "raw_latent",
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "mean_best_nll": 4.155,
            },
            "mean_best_nll": 4.155,
            "mean_best_nll_delta_vs_raw": -0.038,
        }
        follow_up = {
            "index": 1,
            "role": "follow_up",
            "run_dir": "/tmp/chain/follow_up_01",
            "summary_path": "/tmp/chain/follow_up_01/summary.json",
            "exit_code": 1,
            "status": "improved",
            "best_feature": "raw_latent",
            "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
            "best_config": {
                "best_feature": "raw_latent",
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "mean_best_nll": 4.168,
            },
            "mean_best_nll": 4.168,
            "mean_best_nll_delta_vs_raw": -0.030,
            "mean_best_nll_delta_vs_source": 0.013,
            "follow_up_verdict": "regressed",
            "follow_up_gate_failed": True,
        }
        manifest = {
            "schema": mod.SCHEMA,
            "preset": "hybrid4",
            "run_root": "/tmp/chain",
            "steps": [parent, follow_up],
            "stopped_reason": "follow-up 1 exited 1",
            "allowed_gate_stop": True,
        }

        mod._refresh_chain_selection(manifest)
        report = mod._render_report(manifest)

        self.assertEqual(manifest["accepted_step"]["index"], 0)
        self.assertEqual(
            manifest["accepted_summary_path"],
            "/tmp/chain/parent/summary.json",
        )
        self.assertEqual(manifest["best_step"]["index"], 0)
        self.assertEqual(
            manifest["best_summary_path"],
            "/tmp/chain/parent/summary.json",
        )
        self.assertIn("accepted: raw_latent@normalize=blocks,scale=4.0", report)
        self.assertIn("best: raw_latent@normalize=blocks,scale=4.0", report)

    def test_allow_gate_stop_continues_when_guided_confirmation_remains(self) -> None:
        mod = _load_module()
        summary = {
            "follow_up_gate": {"failed": True},
            "guided_next_follow_up_command": {"enabled": True},
        }

        self.assertTrue(
            mod._can_continue_after_gate_stop(
                summary,
                allow_gate_stop=True,
                index=1,
                follow_up_count=2,
            )
        )
        self.assertFalse(
            mod._can_continue_after_gate_stop(
                summary,
                allow_gate_stop=False,
                index=1,
                follow_up_count=2,
            )
        )
        self.assertFalse(
            mod._can_continue_after_gate_stop(
                summary,
                allow_gate_stop=True,
                index=2,
                follow_up_count=2,
            )
        )
        self.assertFalse(
            mod._can_continue_after_gate_stop(
                {
                    "follow_up_gate": {"failed": True},
                    "guided_next_follow_up_command": {"enabled": False},
                },
                allow_gate_stop=True,
                index=1,
                follow_up_count=2,
            )
        )

    def test_chain_selection_accepts_later_confirmed_follow_up_after_gate_stop(
        self,
    ) -> None:
        mod = _load_module()
        parent = {
            "index": 0,
            "role": "parent",
            "run_dir": "/tmp/chain/parent",
            "summary_path": "/tmp/chain/parent/summary.json",
            "exit_code": 0,
            "status": "improved",
            "best_feature": "raw_latent",
            "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
            "best_config": {
                "best_feature": "raw_latent",
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "mean_best_nll": 4.118,
            },
            "mean_best_nll": 4.118,
            "mean_best_nll_delta_vs_raw": -0.072,
        }
        gate_stopped = {
            "index": 1,
            "role": "follow_up",
            "run_dir": "/tmp/chain/follow_up_01",
            "summary_path": "/tmp/chain/follow_up_01/summary.json",
            "exit_code": 1,
            "status": "improved",
            "best_feature": "raw_latent",
            "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
            "best_config": {
                "best_feature": "raw_latent",
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "mean_best_nll": 4.128,
            },
            "mean_best_nll": 4.128,
            "mean_best_nll_delta_vs_raw": -0.065,
            "mean_best_nll_delta_vs_source": 0.010,
            "follow_up_verdict": "regressed",
            "follow_up_gate_failed": True,
        }
        confirmed = {
            "index": 2,
            "role": "follow_up",
            "run_dir": "/tmp/chain/follow_up_02",
            "summary_path": "/tmp/chain/follow_up_02/summary.json",
            "exit_code": 0,
            "status": "improved",
            "best_feature": "raw_latent",
            "best_config_label": "raw_latent@normalize=blocks,scale=4.0",
            "best_config": {
                "best_feature": "raw_latent",
                "feature_normalize": "blocks",
                "hybrid_latent_scale": 4.0,
                "mean_best_nll": 4.128,
            },
            "mean_best_nll": 4.128,
            "mean_best_nll_delta_vs_raw": -0.063,
            "mean_best_nll_delta_vs_source": -0.0001,
            "follow_up_verdict": "confirmed",
            "follow_up_gate_failed": False,
            "follow_up_command_source": "guided_next_follow_up_command",
            "new_seed_source": "command_default",
            "new_seeds": "101,103,107,109,113",
        }
        manifest = {
            "schema": mod.SCHEMA,
            "preset": "hybrid4_deep",
            "run_root": "/tmp/chain",
            "steps": [parent, gate_stopped, confirmed],
            "allowed_gate_stop": True,
        }

        mod._refresh_chain_selection(manifest)

        self.assertEqual(manifest["accepted_step"]["index"], 2)
        self.assertEqual(manifest["best_step"]["index"], 0)
        self.assertEqual(
            manifest["accepted_summary_path"],
            "/tmp/chain/follow_up_02/summary.json",
        )
        self.assertEqual(
            manifest["accepted_step"]["follow_up_command_source"],
            "guided_next_follow_up_command",
        )
        self.assertEqual(manifest["accepted_step"]["new_seed_source"], "command_default")
        self.assertEqual(
            manifest["accepted_step"]["new_seeds"],
            "101,103,107,109,113",
        )


if __name__ == "__main__":
    unittest.main()
