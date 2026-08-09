from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pytest

import spiraltorch as st
from spiraltorch import hf_cli
from spiraltorch import hf_optimizer_study as study


def _bridge_args(*, max_steps: int = 4) -> list[str]:
    return [
        "--model-name",
        "local-model",
        "--train-file",
        "corpus.txt",
        "--train",
        "--max-steps",
        str(max_steps),
        "--eval-before-train",
        "--eval-after-train-policy",
        "always",
    ]


def _fake_bridge(path: Path) -> Path:
    path.write_text("print('fake bridge')\n", encoding="utf-8")
    return path


def _trajectory_id(seed: int) -> str:
    return f"sha256:{seed + 1:064x}"


def _write_completed_run(run: Mapping[str, object]) -> None:
    seed = int(run["seed"])
    arm = str(run["arm"])
    trajectory_id = _trajectory_id(seed)
    trajectory_path = Path(str(run["trajectory"]))
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(
        json.dumps({"trajectory_id": trajectory_id}) + "\n",
        encoding="utf-8",
    )
    optimizer_trace = Path(str(run["optimizer_trace"]))
    trainer_trace = Path(str(run["trainer_trace"]))
    optimizer_trace.parent.mkdir(parents=True, exist_ok=True)
    optimizer_trace.write_text('{"event":"optimizer"}\n', encoding="utf-8")
    trainer_trace.write_text('{"event":"trainer"}\n', encoding="utf-8")
    Path(str(run["output_dir"])).mkdir(parents=True, exist_ok=True)
    command = [str(value) for value in run["command"]]  # type: ignore[index]
    max_steps = int(command[command.index("--max-steps") + 1])
    receipt_arm = "raw" if arm == "observe" else arm
    card = {
        "row_type": "hf_finetune_run_card",
        "failure_stage": None,
        "failure_error": None,
        "train_requested": True,
        "launch_command": command,
        "training_recipe_identity": {
            "status": "ready",
            "identity_payload": {"training_arguments": {"seed": seed}},
        },
        "finetune_execution_identity_after_model": {
            "status": "ready",
            "identity_verified": True,
            "path_independent": True,
            "observed_identity_id": _trajectory_id(100),
        },
        "model_runtime_identity_after_model": {
            "status": "ready",
            "identity_verified": True,
            "path_independent": True,
            "observed_identity_id": _trajectory_id(101),
        },
        "training_input_identity_after_load": {
            "status": "ready",
            "identity_verified": True,
            "path_independent": True,
            "observed_input_id": _trajectory_id(102),
        },
        "zspace_optimizer_control_receipt": {
            "status": "ready",
            "mode": "observe" if arm == "observe" else "apply",
            "trajectory_arm": receipt_arm,
            "evidence_blockers": [],
            "schedule_evidence_complete": True,
            "trajectory_validated": True,
            "planned_update_count": max_steps,
            "realized_update_count": max_steps,
            "trajectory_step_count": max_steps,
            "training_horizon_complete": True,
            "trajectory_horizon_complete": True,
            "trajectory_id": trajectory_id,
            "trace_sha256": study._sha256_file(optimizer_trace),
        },
        "trainer_trace_segment_receipt": {
            "status": "ready",
            "ready": True,
            "trace_sha256": study._sha256_file(trainer_trace),
        },
        "eval_before_train": {"status": "ok", "eval_loss": 2.0},
        "eval_after_train": {
            "status": "ok",
            "eval_loss": 1.9
            + 0.01 * list(study.HF_ZSPACE_FACTORIZED_STUDY_ARMS).index(arm),
        },
    }
    card_path = Path(str(run["run_card"]))
    card_path.write_text(json.dumps(card) + "\n", encoding="utf-8")


def _patch_ready_comparator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        study,
        "compare_hf_zspace_optimizer_factorized_run_cards",
        lambda paths: {
            "status": "ready",
            "matched_seed_count": len(paths) // 4,
            "error_count": 0,
            "errors": [],
        },
    )


def test_factorized_study_plan_is_deterministic_and_owns_arm_flags(
    tmp_path: Path,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    kwargs = {
        "study_dir": tmp_path / "study",
        "seeds": [23, 13],
        "bridge_args": _bridge_args(),
        "bridge_script": bridge,
        "python_executable": sys.executable,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
    }

    first = st.build_hf_zspace_optimizer_factorized_study_plan(**kwargs)
    second = st.build_hf_zspace_optimizer_factorized_study_plan(**kwargs)

    assert first == second
    assert first["study_id"].startswith("sha256:")
    assert first["scientific_spec"]["seeds"] == [13, 23]  # type: ignore[index]
    assert first["run_count"] == 8
    assert [run["arm"] for run in first["runs"][:4]] == list(  # type: ignore[index]
        st.HF_ZSPACE_FACTORIZED_STUDY_ARMS
    )
    observe = first["runs"][0]  # type: ignore[index]
    raw = first["runs"][2]  # type: ignore[index]
    assert "--zspace-optimizer-trajectory-out" in observe["command"]
    assert "--zspace-optimizer-trajectory-json" in raw["command"]
    assert raw["command"].count("--seed") == 1


@pytest.mark.parametrize(
    "managed",
    [
        ["--seed", "99"],
        ["--output-dir=elsewhere"],
        ["--zspace-optimizer-control", "observe"],
    ],
)
def test_factorized_study_rejects_caller_owned_managed_flags(
    tmp_path: Path,
    managed: list[str],
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")

    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="study-managed"):
        st.build_hf_zspace_optimizer_factorized_study_plan(
            study_dir=tmp_path / "study",
            seeds=[13],
            bridge_args=[*_bridge_args(), *managed],
            bridge_script=bridge,
            launch_cwd=tmp_path,
            min_free_disk_gb=0.0,
        )


def test_factorized_study_plan_is_immutable_in_one_directory(tmp_path: Path) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    common = {
        "study_dir": tmp_path / "study",
        "seeds": [13],
        "bridge_script": bridge,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
    }
    planned = st.run_hf_zspace_optimizer_factorized_study(
        **common,
        bridge_args=_bridge_args(max_steps=4),
    )

    assert planned["status"] == "planned"
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="immutable plan"):
        st.run_hf_zspace_optimizer_factorized_study(
            **common,
            bridge_args=_bridge_args(max_steps=8),
        )


def test_factorized_study_recovers_after_parent_crash_and_reuses_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    calls: list[str] = []

    def crash_after_child(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        calls.append(str(run["label"]))
        _write_completed_run(run)
        raise KeyboardInterrupt

    monkeypatch.setattr(study, "_execute_run", crash_after_child)
    common = {
        "study_dir": tmp_path / "study",
        "seeds": [13],
        "bridge_args": _bridge_args(),
        "bridge_script": bridge,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
        "execute": True,
    }
    with pytest.raises(KeyboardInterrupt):
        st.run_hf_zspace_optimizer_factorized_study(**common)

    def complete(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        calls.append(str(run["label"]))
        _write_completed_run(run)
        return 0, 0.25

    monkeypatch.setattr(study, "_execute_run", complete)
    _patch_ready_comparator(monkeypatch)
    summary = st.run_hf_zspace_optimizer_factorized_study(**common)

    assert calls == [
        "s13-observe",
        "s13-dose_matched_constant",
        "s13-raw",
        "s13-dose_normalized",
    ]
    assert summary["status"] == "ready"
    assert summary["completed_run_count"] == 4
    assert summary["run_statuses"]["s13-observe"]["status"] == "recovered"  # type: ignore[index]

    calls.clear()
    reused = st.run_hf_zspace_optimizer_factorized_study(**common)
    assert calls == []
    assert reused["status"] == "ready"
    assert {
        value["status"]
        for value in reused["run_statuses"].values()  # type: ignore[union-attr]
    } == {"reused"}

    inspected = st.run_hf_zspace_optimizer_factorized_study(
        **{**common, "execute": False}
    )
    assert inspected == reused

    card_path = Path(
        str(
            st.build_hf_zspace_optimizer_factorized_study_plan(
                **{key: value for key, value in common.items() if key != "execute"}
            )["runs"][0]["run_card"]  # type: ignore[index]
        )
    )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["eval_after_train"]["eval_loss"] = 1.7
    card_path.write_text(json.dumps(card) + "\n", encoding="utf-8")
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="changed after receipt"):
        st.run_hf_zspace_optimizer_factorized_study(**common)


def test_factorized_study_rejects_cross_run_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")

    def execute(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        _write_completed_run(run)
        if run["arm"] == "raw":
            card_path = Path(str(run["run_card"]))
            card = json.loads(card_path.read_text(encoding="utf-8"))
            card["finetune_execution_identity_after_model"]["observed_identity_id"] = (
                _trajectory_id(999)
            )
            card_path.write_text(json.dumps(card) + "\n", encoding="utf-8")
        return 0, 0.25

    monkeypatch.setattr(study, "_execute_run", execute)
    with pytest.raises(
        st.HFZSpaceFactorizedStudyError,
        match="execution_identity_id",
    ):
        st.run_hf_zspace_optimizer_factorized_study(
            study_dir=tmp_path / "study",
            seeds=[13],
            bridge_args=_bridge_args(),
            bridge_script=bridge,
            launch_cwd=tmp_path,
            min_free_disk_gb=0.0,
            execute=True,
        )


def test_factorized_study_journal_detects_event_tampering(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    study_id = _trajectory_id(200)
    events: list[dict[str, object]] = []
    study._append_event(
        path,
        events,
        study_id=study_id,
        event_type="study_started",
    )
    study._append_event(
        path,
        events,
        study_id=study_id,
        event_type="run_started",
        details={"run_id": _trajectory_id(201)},
    )
    assert len(study._load_events(path, study_id=study_id)) == 2

    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["event_type"] = "study_completed"
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="event identity"):
        study._load_events(path, study_id=study_id)


def test_factorized_study_quarantines_partial_observe_trajectory(
    tmp_path: Path,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    plan = st.build_hf_zspace_optimizer_factorized_study_plan(
        study_dir=tmp_path / "study",
        seeds=[13],
        bridge_args=_bridge_args(),
        bridge_script=bridge,
        launch_cwd=tmp_path,
        min_free_disk_gb=0.0,
    )
    observe = plan["runs"][0]  # type: ignore[index]
    trajectory = Path(str(observe["trajectory"]))
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text('{"partial":true}\n', encoding="utf-8")

    destination = study._quarantine_run_artifacts(
        Path(str(plan["study_dir"])),
        observe,
    )

    assert not trajectory.exists()
    assert (destination / "trajectory.json").is_file()


def test_factorized_study_does_not_break_another_hosts_lock(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    (root / ".study.lock").write_text(
        json.dumps(
            {
                "study_id": _trajectory_id(300),
                "pid": 1,
                "hostname": "another-host",
                "token": "remote",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="another host"):
        with study._study_lock(root, _trajectory_id(300)):
            raise AssertionError("lock should not be acquired")


def test_factorized_study_excludes_its_artifacts_from_git_provenance(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    bridge = _fake_bridge(repository / "bridge.py")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "bridge.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SpiralTorch Test",
            "-c",
            "user.email=spiraltorch@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    kwargs = {
        "study_dir": repository / "runs" / "study",
        "seeds": [13],
        "bridge_args": _bridge_args(),
        "bridge_script": bridge,
        "launch_cwd": repository,
        "min_free_disk_gb": 0.0,
    }

    st.run_hf_zspace_optimizer_factorized_study(**kwargs)
    persisted = json.loads(
        (repository / "runs" / "study" / "study-plan.json").read_text(encoding="utf-8")
    )
    rebuilt = st.build_hf_zspace_optimizer_factorized_study_plan(**kwargs)

    assert rebuilt == persisted
    assert persisted["git_source_provenance"]["dirty"] is False
    assert persisted["git_source_provenance"]["excluded_paths"] == ["runs/study"]


def test_factorized_study_cli_writes_a_plan(tmp_path: Path, capsys) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    status = hf_cli.zspace_optimizer_factorized_study_main(
        [
            "--study-dir",
            str(tmp_path / "study"),
            "--seed",
            "13",
            "--bridge-script",
            str(bridge),
            "--launch-cwd",
            str(tmp_path),
            "--min-free-disk-gb",
            "0",
            "--",
            *_bridge_args(),
        ]
    )

    assert status == 0
    assert "status=planned" in capsys.readouterr().out
    assert (
        tmp_path / "study" / study.HF_ZSPACE_FACTORIZED_STUDY_PLAN_FILENAME
    ).is_file()


def test_factorized_gain_studies_require_shared_evidence_and_report_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")

    def execute(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        _write_completed_run(run)
        return 0, 0.25

    def compare(paths: list[Path]) -> dict[str, object]:
        card = json.loads(paths[0].read_text(encoding="utf-8"))
        command = card["launch_command"]
        gain = float(command[command.index("--zspace-optimizer-control-gain") + 1])
        contrast_arms = {
            "dose_effect": ("dose_matched_constant", "observe", 0.75),
            "dose_normalized_shape_effect": ("dose_normalized", "observe", 0.25),
            "raw_total_effect": ("raw", "observe", 1.0),
            "shape_effect_at_raw_dose": (
                "raw",
                "dose_matched_constant",
                0.25,
            ),
        }
        return {
            "schema": "spiraltorch.hf_zspace_factorized_ablation.v1",
            "status": "ready",
            "matched_seed_count": 1,
            "seeds": [13],
            "error_count": 0,
            "errors": [],
            "factorized_seeds": [
                {
                    "status": "ready",
                    "seed": 13,
                    "trajectory_id": _trajectory_id(13),
                    "eval_before_losses": {"observe": 2.0},
                    "eval_after_losses": {"observe": 1.9},
                }
            ],
            "contrasts": {
                name: {
                    "left_arm": left,
                    "right_arm": right,
                    "lower_is_better": True,
                    "mean": gain * multiplier,
                    "values": [gain * multiplier],
                    "bounded_trend_direction": "right_arm_better",
                }
                for name, (left, right, multiplier) in contrast_arms.items()
            },
        }

    monkeypatch.setattr(study, "_execute_run", execute)
    monkeypatch.setattr(
        study,
        "compare_hf_zspace_optimizer_factorized_run_cards",
        compare,
    )
    directories = []
    for gain in (0.25, 0.5, 1.0):
        directory = tmp_path / f"gain-{gain}"
        directories.append(directory)
        summary = st.run_hf_zspace_optimizer_factorized_study(
            study_dir=directory,
            seeds=[13],
            bridge_args=[
                *_bridge_args(),
                "--zspace-optimizer-control-gain",
                str(gain),
            ],
            bridge_script=bridge,
            launch_cwd=tmp_path,
            min_free_disk_gb=0.0,
            execute=True,
        )
        assert summary["status"] == "ready"

    report = st.compare_hf_zspace_optimizer_factorized_gain_studies(directories)

    assert report["status"] == "ready"
    assert report["gains"] == [0.25, 0.5, 1.0]
    assert report["observe_baseline_exact_match"] is True
    assert report["bounded_gain_correlated_loss_degradation_observed"] is True
    assert report["bounded_gain_correlated_loss_improvement_observed"] is False
    assert all("study_dir" not in source for source in report["source_studies"])
    assert report["contrasts"]["raw_total_effect"][  # type: ignore[index]
        "ordinary_least_squares"
    ]["r_squared"] == pytest.approx(1.0)

    output = tmp_path / "gain-response.json"
    status = hf_cli.zspace_optimizer_factorized_gain_compare_main(
        [*[str(path) for path in directories], "--out", str(output)]
    )
    assert status == 0
    assert output.is_file()
    assert "gain_correlated_loss_degradation=True" in capsys.readouterr().out

    source_report = directories[0] / study.HF_ZSPACE_FACTORIZED_STUDY_REPORT_FILENAME
    source_report.write_text(source_report.read_text(encoding="utf-8") + "\n")
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="receipt drifted"):
        st.compare_hf_zspace_optimizer_factorized_gain_studies(directories)
