from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pytest
import spiraltorch as st
from spiraltorch import hf_cli


def _write_adapter(path: Path, weights: bytes) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "org/base",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def _flag(command: Sequence[str], name: str) -> str:
    index = list(command).index(name)
    return str(command[index + 1])


def _run_card(
    parent: Path,
    *,
    before: float,
    after: float,
    launch_command: Sequence[str],
) -> dict[str, object]:
    return {
        "row_type": "hf_finetune_run_card",
        "load_status": "ok",
        "failure_stage": None,
        "model_saved": True,
        "adapter_saved": True,
        "model_name": str(parent),
        "model_artifact_kind": "peft_adapter",
        "finetune_mode": "lora",
        "finetune_start_report": {
            "mode": "adapter_warm_start",
            "adapter_weights_source": str(parent),
            "weights_only_warm_start": True,
            "trainer_checkpoint_resume": False,
        },
        "model_prepare_report": {
            "mode": "lora",
            "adapter_attached": True,
            "parameter_report_after": {
                "parameter_count": 100,
                "trainable_parameter_count": 8,
                "frozen_parameter_count": 92,
                "trainable_parameter_ratio": 0.08,
            },
        },
        "trainer_metrics": {"train_loss": after},
        "eval_before_train": {"status": "ok", "eval_loss": before},
        "eval_after_train": {"status": "ok", "eval_loss": after},
        "launch_command": list(launch_command),
        "launch_command_display": " ".join(launch_command),
        "launch_command_source": "test",
    }


def _launch_command(parent: Path, output: Path) -> list[str]:
    bridge = Path(__file__).resolve().parents[1] / "examples" / "hf_finetune_bridge.py"
    return [
        sys.executable,
        str(bridge),
        "--model-name",
        str(parent),
        "--train",
        "--output-dir",
        str(output),
        "--run-card",
        str(output / "spiraltorch-hf-finetune-run-card.json"),
        "--trainer-trace-jsonl",
        str(output / "spiraltorch-hf-finetune-trainer-trace.jsonl"),
        "--finetune-mode",
        "lora",
        "--max-steps",
        "1",
        "--max-train-samples",
        "8",
        "--adapter-promotion-gate",
        "--eval-before-train",
    ]


def _write_promoted_adapter(
    output: Path,
    parent: Path,
    *,
    weights: bytes,
    before: float,
    after: float,
    launch_command: Sequence[str] | None = None,
) -> dict[str, object]:
    adapter = _write_adapter(output, weights)
    command = list(launch_command or _launch_command(parent, output))
    run_card_path = adapter / "spiraltorch-hf-finetune-run-card.json"
    card = _run_card(
        parent,
        before=before,
        after=after,
        launch_command=command,
    )
    lineage = st.write_hf_adapter_lineage(
        adapter,
        parent_adapter=parent,
        run_card=card,
        run_card_path=run_card_path,
    )
    card["adapter_lineage"] = lineage
    promotion = st.write_hf_adapter_promotion(
        adapter,
        card,
        parent_adapter=parent,
    )
    card["adapter_promotion"] = promotion
    run_card_path.write_text(json.dumps(card), encoding="utf-8")
    return lineage


def _seed_chain(tmp_path: Path, *, improvement: float = 0.1) -> tuple[Path, Path]:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child = tmp_path / "child"
    _write_promoted_adapter(
        child,
        root,
        weights=b"child",
        before=1.0,
        after=1.0 - improvement,
    )
    return root, child


class FakeFineTuneRunner:
    def __init__(
        self,
        improvements: Sequence[float],
        *,
        fail_returncode: int | None = None,
        weight_prefix: str = "generated",
    ) -> None:
        self.improvements = list(improvements)
        self.fail_returncode = fail_returncode
        self.weight_prefix = weight_prefix
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> int:
        values = [str(item) for item in command]
        self.commands.append(values)
        if self.fail_returncode is not None:
            return self.fail_returncode
        output = Path(_flag(values, "--output-dir"))
        parent = Path(_flag(values, "--model-name"))
        improvement = self.improvements[len(self.commands) - 1]
        before = 0.9 - (len(self.commands) - 1) * 0.1
        _write_promoted_adapter(
            output,
            parent,
            weights=f"{self.weight_prefix}-{len(self.commands)}".encode(),
            before=before,
            after=before - improvement,
            launch_command=values,
        )
        return 0


def test_executor_dry_run_writes_replayable_state_and_cli(
    tmp_path: Path,
    capsys,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=16,
    )
    loaded = st.load_hf_adapter_continuation_executor(state_path)
    code = hf_cli.adapter_continuation_executor_main(
        [
            str(child),
            "--output-root",
            str(output_root),
            "--state",
            str(state_path),
            "--max-lineage-depth",
            "3",
            "--max-steps",
            "2",
            "--max-train-samples",
            "16",
        ]
    )
    output = capsys.readouterr().out

    assert report["status"] == "ready"
    assert report["action"] == "run_generation"
    assert report["pending_generation"]["lineage_depth"] == 2
    assert report["pending_generation"]["preflight"]["ready"] is True
    assert report["generation_attempt_count"] == 0
    assert not (output_root / "generation-002").exists()
    assert loaded["status"] == "ready"
    assert loaded["invocation_count"] == 1
    assert code == 0
    assert "status=ready" in output
    assert "depth=2" in output
    assert "hf_adapter_executor" in st.__all__
    assert "status=ready" in st.hf_adapter_continuation_executor_lines(report)[0]


def test_executor_runs_multiple_generations_until_depth_policy_stops(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    runner = FakeFineTuneRunner([0.08, 0.07])

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        run=True,
        max_generations=5,
        max_lineage_depth=3,
        min_eval_improvement=0.01,
        plateau_patience=2,
        command_runner=runner,
    )

    assert report["status"] == "stopped"
    assert report["reason"] == "continuation_policy_stop"
    assert report["continuation_policy"]["stop_reason_codes"] == [
        "max_lineage_depth_reached"
    ]
    assert report["generations_executed_this_invocation"] == 2
    assert report["promoted_generation_count"] == 2
    assert [row["status"] for row in report["generations"]] == [
        "promoted",
        "promoted",
    ]
    assert [row["lineage_depth"] for row in report["generations"]] == [2, 3]
    assert all(row["postflight"]["ready"] for row in report["generations"])
    assert (output_root / "generation-002" / st.HF_ADAPTER_LINEAGE_FILENAME).is_file()
    assert (output_root / "generation-003" / st.HF_ADAPTER_PROMOTION_FILENAME).is_file()
    assert len(runner.commands) == 2


def test_executor_stops_after_new_generation_completes_plateau(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path, improvement=0.01)
    runner = FakeFineTuneRunner([0.005])

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        run=True,
        max_generations=4,
        min_eval_improvement=0.02,
        plateau_patience=2,
        command_runner=runner,
    )

    assert report["status"] == "stopped"
    assert report["generations_executed_this_invocation"] == 1
    assert report["promoted_generation_count"] == 1
    assert report["continuation_policy"]["stop_reason_codes"] == [
        "eval_improvement_plateau"
    ]
    assert report["continuation_policy"]["consecutive_below_min_eval_improvement"] == 2


def test_executor_records_failure_and_resumes_same_generation(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    failed_runner = FakeFineTuneRunner([0.05], fail_returncode=7)

    failed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=failed_runner,
    )
    resumed_runner = FakeFineTuneRunner([0.05])
    resumed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=resumed_runner,
    )

    assert failed["status"] == "failed"
    assert failed["reason"] == "generation_command_returncode_7"
    assert failed["generations"][0]["returncode"] == 7
    assert resumed["status"] == "stopped"
    assert resumed["invocation_count"] == 2
    assert [row["status"] for row in resumed["generations"]] == [
        "failed",
        "promoted",
    ]
    assert resumed["promoted_generation_count"] == 1
    assert resumed["generations"][1]["lineage_depth"] == 2
    assert len(resumed_runner.commands) == 1


def test_executor_blocks_partial_output_left_by_failed_command(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"

    def partial_failure(command: Sequence[str]) -> int:
        output = Path(_flag(command, "--output-dir"))
        output.mkdir(parents=True)
        (output / "partial.txt").write_text("incomplete", encoding="utf-8")
        return 9

    failed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=partial_failure,
    )
    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=FakeFineTuneRunner([0.05]),
    )
    partial_output = output_root / "generation-002"
    shutil.rmtree(partial_output)
    resumed_runner = FakeFineTuneRunner([0.05], weight_prefix="after-partial")
    resumed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=resumed_runner,
    )

    assert failed["status"] == "failed"
    assert blocked["status"] == "blocked"
    assert blocked["action"] == "resolve_failed_generation_output"
    assert blocked["unresolved_generation"]["output_dir"] == str(partial_output)
    assert resumed["status"] == "stopped"
    assert [row["status"] for row in resumed["generations"]] == [
        "failed",
        "promoted",
    ]
    assert len(resumed_runner.commands) == 1


def test_executor_resumes_after_per_invocation_generation_limit(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    first_runner = FakeFineTuneRunner([0.06])

    first = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_generations=1,
        max_lineage_depth=3,
        command_runner=first_runner,
    )
    second_runner = FakeFineTuneRunner([0.05], weight_prefix="resumed")
    second = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_generations=1,
        max_lineage_depth=3,
        command_runner=second_runner,
    )

    assert first["status"] == "generation_limit_reached"
    assert first["selected_lineage_depth"] == 2
    assert first["promoted_generation_count"] == 1
    assert second["status"] == "stopped"
    assert second["selected_lineage_depth"] == 3
    assert second["invocation_count"] == 2
    assert second["promoted_generation_count"] == 2
    assert [row["lineage_depth"] for row in second["generations"]] == [2, 3]
    assert len(first_runner.commands) == 1
    assert len(second_runner.commands) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_generations": 1.5},
        {"max_lineage_depth": -1},
        {"target_eval_loss": float("nan")},
        {"plateau_patience": 0},
        {"max_steps_multiplier": 0.0},
        {"output_prefix": "../generation"},
    ],
)
def test_executor_rejects_invalid_configuration_before_writing_state(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"

    with pytest.raises(ValueError):
        st.run_hf_adapter_continuation_executor(
            child,
            output_root=output_root,
            state_path=state_path,
            **kwargs,
        )

    assert not state_path.exists()


def test_executor_fails_closed_on_interrupted_attempt_then_explicitly_retries(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    planned = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pending = planned["pending_generation"]
    state["generations"] = [
        {
            "attempt_id": "interrupted-test",
            "status": "running",
            "parent_adapter_id": pending["parent_adapter_id"],
            "lineage_depth": pending["lineage_depth"],
            "output_dir": pending["output_dir"],
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    runner = FakeFineTuneRunner([0.05], weight_prefix="retried")
    retried = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        retry_interrupted=True,
        max_lineage_depth=2,
        command_runner=runner,
    )

    assert blocked["status"] == "blocked"
    assert blocked["action"] == "audit_interrupted_generation"
    assert "unresolved" in blocked["reason"]
    assert retried["status"] == "stopped"
    assert [row["status"] for row in retried["generations"]] == [
        "interrupted_retry",
        "promoted",
    ]
    assert retried["promoted_generation_count"] == 1
    assert len(runner.commands) == 1


def test_executor_recovers_completed_output_after_explicit_selection_interrupt(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    chain = st.hf_adapter_promotion_chain_report(child)
    selected_adapter_id = chain["selected_adapter_id"]
    planned = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        select_adapter_id=selected_adapter_id,
        max_lineage_depth=2,
    )
    pending = planned["pending_generation"]
    output_dir = Path(pending["output_dir"])
    command = pending["command"]["command"]
    _write_promoted_adapter(
        output_dir,
        child,
        weights=b"recovered-explicit-selection",
        before=0.9,
        after=0.84,
        launch_command=command,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["generations"] = [
        {
            "attempt_id": "interrupted-after-command",
            "status": "running",
            "parent_adapter_id": pending["parent_adapter_id"],
            "lineage_depth": pending["lineage_depth"],
            "output_dir": pending["output_dir"],
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    recovered = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        select_adapter_id=selected_adapter_id,
        max_lineage_depth=2,
    )

    assert recovered["status"] == "stopped"
    assert recovered["selected_lineage_depth"] == 2
    assert recovered["promoted_generation_count"] == 1
    assert recovered["generations"][0]["status"] == "promoted_recovered"
    assert recovered["generations"][0]["postflight"]["ready"] is True
