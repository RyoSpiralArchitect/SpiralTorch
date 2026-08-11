from __future__ import annotations

import json
import subprocess
import sys
import types
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


def _shared_corpus_bridge_args(*, max_steps: int = 4) -> list[str]:
    arguments = _bridge_args(max_steps=max_steps)
    source_index = arguments.index("--train-file")
    del arguments[source_index : source_index + 2]
    return arguments


def _fake_bridge(path: Path) -> Path:
    path.write_text("print('fake bridge')\n", encoding="utf-8")
    return path


def _trajectory_id(seed: int) -> str:
    return f"sha256:{seed + 1:064x}"


def test_path_is_within_is_python_38_compatible(tmp_path: Path) -> None:
    parent = (tmp_path / "parent").resolve()

    assert study._path_is_within(parent / "child" / "artifact.json", parent)
    assert study._path_is_within(parent, parent)
    assert not study._path_is_within(tmp_path / "sibling" / "artifact.json", parent)


def _feedback_config() -> dict[str, object]:
    return {
        "loss_ema_alpha": 0.2,
        "relative_delta_ema_alpha": 0.5,
        "loss_floor": 1.0e-8,
        "regression_threshold": 0.01,
        "halt_threshold": 0.05,
        "recovery_threshold": 0.0025,
        "attenuation_rate": 0.25,
        "recovery_rate": 0.125,
        "halt_regression_streak": 2,
        "resume_improvement_streak": 2,
        "warmup_observations": 2,
        "max_stale_updates": 0,
        "maximum_gate": 1.0,
    }


def _patch_feedback_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        study,
        "_resolved_feedback_config",
        lambda requested: dict(_feedback_config() if requested is None else requested),
    )


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
    receipt_arm = str(
        run.get("expected_trajectory_arm", "raw" if arm == "observe" else arm)
    )
    expected_mode = str(
        run.get("expected_mode", "observe" if arm == "observe" else "apply")
    )
    expected_feedback = str(run.get("expected_feedback_mode", "off"))
    if arm == "dose_preserving_complement":
        arms = study.HF_ZSPACE_POLARITY_STUDY_ARMS
    elif arm in study.HF_ZSPACE_FACTORIZED_STUDY_ARMS:
        arms = study.HF_ZSPACE_FACTORIZED_STUDY_ARMS
    else:
        arms = study.HF_ZSPACE_FEEDBACK_STUDY_ARMS
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
            "mode": expected_mode,
            "feedback_mode": expected_feedback,
            "recipe": {
                "feedback_config": (
                    _feedback_config() if expected_feedback == "loss_guard" else None
                ),
            },
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
            "eval_loss": 1.9 + 0.01 * list(arms).index(arm),
        },
    }
    if receipt_arm == "dose_preserving_complement":
        policy_id = _trajectory_id(seed + 200)
        policy_path = Path(str(run["output_dir"])) / "trajectory-policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "policy_id": policy_id,
                    "source_trajectory_id": trajectory_id,
                    "policy_validated": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        receipt = card["zspace_optimizer_control_receipt"]
        assert isinstance(receipt, dict)
        receipt.update(
            {
                "trajectory_policy_id": policy_id,
                "trajectory_policy_path": str(policy_path),
                "trajectory_policy_validated": True,
                "trajectory_policy_source_trajectory_id": trajectory_id,
                "trajectory_policy_sha256": study._sha256_file(policy_path),
                "trajectory_policy_size_bytes": policy_path.stat().st_size,
            }
        )
    card_path = Path(str(run["run_card"]))
    card_path.write_text(json.dumps(card) + "\n", encoding="utf-8")


def _write_failed_run(run: Mapping[str, object]) -> None:
    card_path = Path(str(run["run_card"]))
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(
        json.dumps(
            {
                "row_type": "hf_finetune_run_card",
                "failure_stage": "train",
                "failure_error": "synthetic child failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )


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


def _patch_ready_feedback_comparator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        study,
        "compare_hf_zspace_optimizer_feedback_run_cards",
        lambda paths: {
            "schema": "spiraltorch.hf_zspace_feedback_ablation.v1",
            "status": "ready",
            "matched_seed_count": len(paths) // 3,
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


def test_polarity_study_plan_is_deterministic_and_owns_policy_arm(
    tmp_path: Path,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    kwargs = {
        "study_dir": tmp_path / "polarity-study",
        "seeds": [23, 13],
        "bridge_args": _bridge_args(),
        "bridge_script": bridge,
        "python_executable": sys.executable,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
    }

    first = st.build_hf_zspace_optimizer_polarity_study_plan(**kwargs)
    second = st.build_hf_zspace_optimizer_polarity_study_plan(**kwargs)

    assert first == second
    assert first["schema"] == st.HF_ZSPACE_POLARITY_STUDY_SCHEMA
    assert first["scientific_spec"]["seeds"] == [13, 23]
    assert first["run_count"] == 6
    seed_runs = first["runs"][:3]
    assert [run["arm"] for run in seed_runs] == list(st.HF_ZSPACE_POLARITY_STUDY_ARMS)
    assert "--zspace-optimizer-trajectory-out" in seed_runs[0]["command"]
    for run in seed_runs[1:]:
        assert "--zspace-optimizer-trajectory-json" in run["command"]
    complement = seed_runs[-1]
    arm_index = complement["command"].index("--zspace-optimizer-trajectory-arm")
    assert complement["command"][arm_index + 1] == "dose_preserving_complement"
    assert first["artifacts"]["polarity_report"].endswith("polarity-report.json")


def test_polarity_corpus_study_plan_seals_sources_and_substudies(
    tmp_path: Path,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    corpus_a = tmp_path / "a.txt"
    corpus_b = tmp_path / "b.txt"
    corpus_a.write_text("alpha corpus\n", encoding="utf-8")
    corpus_b.write_text("beta corpus\n", encoding="utf-8")
    kwargs = {
        "study_dir": tmp_path / "corpus-study",
        "corpora": {"beta": corpus_b, "alpha": corpus_a},
        "seeds": [23, 13],
        "bridge_args": _shared_corpus_bridge_args(),
        "bridge_script": bridge,
        "python_executable": sys.executable,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
    }

    first = st.build_hf_zspace_optimizer_polarity_corpus_study_plan(**kwargs)
    second = st.build_hf_zspace_optimizer_polarity_corpus_study_plan(**kwargs)

    assert first == second
    assert first["schema"] == st.HF_ZSPACE_POLARITY_CORPUS_STUDY_SCHEMA
    assert first["corpus_count"] == 2
    assert first["run_count"] == 12
    assert first["scientific_spec"]["protocol"]["corpus_source_flags"] == [  # type: ignore[index]
        "--train-file"
    ]
    assert [substudy["label"] for substudy in first["substudies"]] == [  # type: ignore[index]
        "alpha",
        "beta",
    ]
    assert all(
        substudy["sha256"].startswith("sha256:")
        for substudy in first["substudies"]  # type: ignore[index]
    )
    for substudy in first["substudies"]:  # type: ignore[index]
        scientific_spec = substudy["plan"]["scientific_spec"]
        assert scientific_spec["bridge_args"].count("--train-file") == 1
        assert scientific_spec["seeds"] == [13, 23]

    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="owns dataset source"):
        st.build_hf_zspace_optimizer_polarity_corpus_study_plan(
            **{**kwargs, "bridge_args": _bridge_args()}
        )

    assert (
        st._rs.ZSPACE_POLARITY_EVIDENCE_MAX_SAFE_SEED
        == st.ZSPACE_POLARITY_EVIDENCE_MAX_SAFE_SEED
    )
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="cross-client maximum"):
        st.build_hf_zspace_optimizer_polarity_corpus_study_plan(
            **{
                **kwargs,
                "seeds": [st.ZSPACE_POLARITY_EVIDENCE_MAX_SAFE_SEED + 1],
            }
        )


def test_study_runtime_fingerprint_seals_loaded_native_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = tmp_path / "libspiraltorch.dylib"
    native.write_bytes(b"rust-semantic-core")
    monkeypatch.setattr(st, "_rs", types.SimpleNamespace(__file__=str(native)))

    fingerprint = study._runtime_source_fingerprint()

    sealed = fingerprint["loaded_native_extension"]
    assert sealed["status"] == "ready"
    assert sealed["filename"] == native.name
    assert sealed["size_bytes"] == len(b"rust-semantic-core")
    assert sealed["sha256"] == study._sha256_file(native)


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
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="changed after receipt"):
        st.run_hf_zspace_optimizer_factorized_study(
            **common,
            retry_failed=True,
        )


def test_factorized_study_quarantines_failed_run_card_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    calls: list[str] = []

    def fail(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        calls.append(str(run["label"]))
        _write_failed_run(run)
        return 1, 0.1

    monkeypatch.setattr(study, "_execute_run", fail)
    common = {
        "study_dir": tmp_path / "study",
        "seeds": [13],
        "bridge_args": _bridge_args(),
        "bridge_script": bridge,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
        "execute": True,
    }
    failed = st.run_hf_zspace_optimizer_factorized_study(**common)
    assert failed["status"] == "failed"
    assert calls == ["s13-observe"]

    def complete(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        calls.append(str(run["label"]))
        _write_completed_run(run)
        return 0, 0.25

    monkeypatch.setattr(study, "_execute_run", complete)
    _patch_ready_comparator(monkeypatch)
    summary = st.run_hf_zspace_optimizer_factorized_study(
        **common,
        retry_failed=True,
    )

    assert summary["status"] == "ready"
    assert calls == [
        "s13-observe",
        "s13-observe",
        "s13-dose_matched_constant",
        "s13-raw",
        "s13-dose_normalized",
    ]
    quarantine = Path(str(common["study_dir"])) / "quarantine"
    quarantined_cards = list(quarantine.glob("s13-observe-*/run-card.json"))
    assert len(quarantined_cards) == 1
    quarantined = json.loads(quarantined_cards[0].read_text(encoding="utf-8"))
    assert quarantined["failure_error"] == "synthetic child failure"
    event_path = (
        Path(str(common["study_dir"]))
        / study.HF_ZSPACE_FACTORIZED_STUDY_EVENTS_FILENAME
    )
    events = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    quarantine_events = [
        event for event in events if event["event_type"] == "run_quarantined"
    ]
    assert len(quarantine_events) == 1
    assert "training failure" in quarantine_events[0]["validation_error"]


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


def test_polarity_corpus_study_reuses_outer_git_provenance_on_resume(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    bridge = _fake_bridge(repository / "bridge.py")
    corpus_a = repository / "a.txt"
    corpus_b = repository / "b.txt"
    corpus_a.write_text("alpha corpus\n", encoding="utf-8")
    corpus_b.write_text("beta corpus\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "add", "bridge.py", "a.txt", "b.txt"],
        cwd=repository,
        check=True,
    )
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
    root = repository / "runs" / "corpus-study"
    kwargs = {
        "study_dir": root,
        "corpora": {"alpha": corpus_a, "beta": corpus_b},
        "seeds": [13],
        "bridge_args": _shared_corpus_bridge_args(),
        "bridge_script": bridge,
        "launch_cwd": repository,
        "min_free_disk_gb": 0.0,
    }

    first = st.run_hf_zspace_optimizer_polarity_corpus_study(**kwargs)
    second = st.run_hf_zspace_optimizer_polarity_corpus_study(**kwargs)
    persisted = json.loads(
        (root / study.HF_ZSPACE_POLARITY_CORPUS_STUDY_PLAN_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    assert second == first
    outer_provenance = persisted["git_source_provenance"]
    assert outer_provenance["dirty"] is False
    assert outer_provenance["excluded_paths"] == ["runs/corpus-study"]
    assert all(
        substudy["plan"]["git_source_provenance"] == outer_provenance
        for substudy in persisted["substudies"]
    )


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


def test_polarity_study_cli_writes_a_plan(tmp_path: Path, capsys) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    status = hf_cli.zspace_optimizer_polarity_study_main(
        [
            "--study-dir",
            str(tmp_path / "polarity-study"),
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
        tmp_path / "polarity-study" / study.HF_ZSPACE_POLARITY_STUDY_PLAN_FILENAME
    ).is_file()


def test_polarity_corpus_study_cli_writes_a_meta_plan(
    tmp_path: Path,
    capsys,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    corpus_a = tmp_path / "a.txt"
    corpus_b = tmp_path / "b.txt"
    corpus_a.write_text("alpha corpus\n", encoding="utf-8")
    corpus_b.write_text("beta corpus\n", encoding="utf-8")
    root = tmp_path / "corpus-study"

    status = hf_cli.zspace_optimizer_polarity_corpus_study_main(
        [
            "--study-dir",
            str(root),
            "--corpus",
            f"alpha={corpus_a}",
            "--corpus",
            f"beta={corpus_b}",
            "--seed",
            "13",
            "--bridge-script",
            str(bridge),
            "--launch-cwd",
            str(tmp_path),
            "--min-free-disk-gb",
            "0",
            "--",
            *_shared_corpus_bridge_args(),
        ]
    )

    assert status == 0
    assert "status=planned" in capsys.readouterr().out
    assert (root / study.HF_ZSPACE_POLARITY_CORPUS_STUDY_PLAN_FILENAME).is_file()


def test_feedback_study_plan_seals_one_rust_config_and_shared_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_feedback_config(monkeypatch)
    bridge = _fake_bridge(tmp_path / "bridge.py")
    kwargs = {
        "study_dir": tmp_path / "study",
        "seeds": [23, 13],
        "bridge_args": _bridge_args(max_steps=8),
        "bridge_script": bridge,
        "launch_cwd": tmp_path,
        "min_free_disk_gb": 0.0,
    }

    first = st.build_hf_zspace_optimizer_feedback_study_plan(**kwargs)
    second = st.build_hf_zspace_optimizer_feedback_study_plan(**kwargs)

    assert first == second
    assert first["schema"] == st.HF_ZSPACE_FEEDBACK_STUDY_SCHEMA
    assert first["scientific_spec"]["seeds"] == [13, 23]
    assert first["scientific_spec"]["logging_steps"] == 1
    assert first["scientific_spec"]["require_eval_dataset"] is True
    assert first["scientific_spec"]["feedback_config"] == _feedback_config()
    assert first["scientific_spec"]["bridge_argument_validation"]["status"] == (
        "not_run_custom_bridge"
    )
    assert first["run_count"] == 6
    seed_runs = first["runs"][:3]
    assert [run["arm"] for run in seed_runs] == list(st.HF_ZSPACE_FEEDBACK_STUDY_ARMS)
    assert len({run["trajectory"] for run in seed_runs}) == 1
    assert all(run["command"].count("--logging-steps") == 1 for run in seed_runs)
    assert all(run["command"].count("--require-eval-dataset") == 1 for run in seed_runs)
    assert all(
        run["command"][run["command"].index("--logging-steps") + 1] == "1"
        for run in seed_runs
    )
    guarded = seed_runs[-1]
    assert guarded["expected_feedback_mode"] == "loss_guard"
    assert (
        guarded["expected_feedback_config_id"]
        == first["scientific_spec"]["feedback_config_id"]
    )
    assert "--zspace-optimizer-feedback-warmup-observations" in guarded["command"]
    assert "--zspace-optimizer-trajectory-json" in guarded["command"]


def test_feedback_study_default_bridge_argument_validation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_bridge = tmp_path / "bridge.py"
    default_bridge.write_text("print('bridge')\n", encoding="utf-8")
    monkeypatch.setattr(study, "_default_bridge_script", lambda: default_bridge)
    monkeypatch.setattr(
        study.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="",
            stderr="bridge.py: error: unrecognized arguments: --stale-name\n",
        ),
    )

    with pytest.raises(
        st.HFZSpaceFactorizedStudyError,
        match="unrecognized arguments: --stale-name",
    ):
        study._validate_feedback_bridge_arguments(
            bridge_script=default_bridge,
            python_executable=Path(sys.executable),
            bridge_args=["--stale-name"],
            launch_cwd=tmp_path,
        )


def test_feedback_study_rejects_run_card_config_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_feedback_config(monkeypatch)
    bridge = _fake_bridge(tmp_path / "bridge.py")
    plan = st.build_hf_zspace_optimizer_feedback_study_plan(
        study_dir=tmp_path / "study",
        seeds=[13],
        bridge_args=_bridge_args(),
        bridge_script=bridge,
        launch_cwd=tmp_path,
        min_free_disk_gb=0.0,
    )
    guarded = plan["runs"][-1]
    _write_completed_run(guarded)
    card = study._read_json(Path(guarded["run_card"]))
    card["zspace_optimizer_control_receipt"]["recipe"]["feedback_config"][
        "maximum_gate"
    ] = 0.5

    with pytest.raises(
        st.HFZSpaceFactorizedStudyError,
        match="feedback config does not match",
    ):
        study._validate_completed_run_card(guarded, card)


@pytest.mark.parametrize(
    "managed",
    [
        ["--logging-steps", "2"],
        ["--zspace-optimizer-feedback", "loss_guard"],
        ["--zspace-optimizer-feedback-maximum-gate", "0.5"],
        ["--require-eval-dataset"],
        ["--validate-args-only"],
    ],
)
def test_feedback_study_rejects_caller_owned_feedback_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed: list[str],
) -> None:
    _patch_feedback_config(monkeypatch)
    bridge = _fake_bridge(tmp_path / "bridge.py")

    with pytest.raises(
        st.HFZSpaceFactorizedStudyError,
        match="feedback study-managed",
    ):
        st.build_hf_zspace_optimizer_feedback_study_plan(
            study_dir=tmp_path / "study",
            seeds=[13],
            bridge_args=[*_bridge_args(), *managed],
            bridge_script=bridge,
            launch_cwd=tmp_path,
            min_free_disk_gb=0.0,
        )


def test_feedback_study_uses_shared_resumable_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_feedback_config(monkeypatch)
    _patch_ready_feedback_comparator(monkeypatch)
    bridge = _fake_bridge(tmp_path / "bridge.py")
    calls: list[str] = []

    def execute(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        calls.append(str(run["label"]))
        _write_completed_run(run)
        return 0, 0.25

    monkeypatch.setattr(study, "_execute_run", execute)
    summary = st.run_hf_zspace_optimizer_feedback_study(
        study_dir=tmp_path / "study",
        seeds=[13],
        bridge_args=_bridge_args(),
        bridge_script=bridge,
        launch_cwd=tmp_path,
        min_free_disk_gb=0.0,
        execute=True,
    )

    assert calls == [
        "s13-observe",
        "s13-raw_unguarded",
        "s13-raw_loss_guard",
    ]
    assert summary["status"] == "ready"
    assert summary["feedback_status"] == "ready"
    assert summary["feedback_report_sha256"].startswith("sha256:")
    event_path = tmp_path / "study" / study.HF_ZSPACE_FEEDBACK_STUDY_EVENTS_FILENAME
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert all(
        event["schema"] == study.HF_ZSPACE_FEEDBACK_STUDY_EVENT_SCHEMA
        for event in events
    )

    calls.clear()
    reused = st.run_hf_zspace_optimizer_feedback_study(
        study_dir=tmp_path / "study",
        seeds=[13],
        bridge_args=_bridge_args(),
        bridge_script=bridge,
        launch_cwd=tmp_path,
        min_free_disk_gb=0.0,
        execute=True,
    )
    assert calls == []
    assert reused["status"] == "ready"
    assert {row["status"] for row in reused["run_statuses"].values()} == {"reused"}


def test_feedback_study_cli_writes_a_rust_resolved_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    _patch_feedback_config(monkeypatch)
    bridge = _fake_bridge(tmp_path / "bridge.py")
    config_path = tmp_path / "feedback.json"
    config_path.write_text(json.dumps(_feedback_config()), encoding="utf-8")

    status = hf_cli.zspace_optimizer_feedback_study_main(
        [
            "--study-dir",
            str(tmp_path / "study"),
            "--seed",
            "13",
            "--feedback-config-json",
            str(config_path),
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
    assert (tmp_path / "study" / study.HF_ZSPACE_FEEDBACK_STUDY_PLAN_FILENAME).is_file()


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


def _polarity_corpus_bundle(label: str) -> dict[str, object]:
    corpus_id = "sha256:" + label * 64
    shared_evidence = {
        "trajectory_id": "sha256:" + "3" * 64,
        "trajectory_policy_id": "sha256:" + "4" * 64,
        "control_sequence_id": "sha256:" + "5" * 64,
        "nominal_schedule_sequence_id": "sha256:" + "6" * 64,
    }
    rows = []
    for seed in (13, 17, 23):
        normalized = 0.002 + seed * 1e-7
        complement = -0.001 - seed * 1e-7
        rows.append(
            {
                "corpus_id": corpus_id,
                "seed": seed,
                "dose_normalized_shape_effect": normalized,
                "complement_shape_effect": complement,
                "polarity_effect": complement - normalized,
            }
        )
    return {
        "label": label,
        "study_dir": f"/study/{label}",
        "study_id": "sha256:" + "7" * 64,
        "corpus_id": corpus_id,
        "runtime_identity_id": "sha256:" + "2" * 64,
        "protocol_payload": {"schema": "test.protocol.v1"},
        "protocol_id": "sha256:" + "1" * 64,
        "source_args": ["--train-file", f"{label}.txt"],
        "shared_evidence": shared_evidence,
        "rows": rows,
        "seeds": [13, 17, 23],
        "plan_sha256": "sha256:" + "8" * 64,
        "summary_sha256": "sha256:" + "9" * 64,
        "completion_event_id": "sha256:" + "b" * 64,
        "polarity_report_sha256": "sha256:" + "a" * 64,
    }


def test_polarity_corpus_comparison_delegates_balanced_semantics_to_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {label: _polarity_corpus_bundle(label) for label in ("a", "b", "c")}
    monkeypatch.setattr(
        study,
        "_polarity_corpus_study_bundle",
        lambda label, _path: bundles[label],
    )

    report = st.compare_hf_zspace_optimizer_polarity_studies(
        {label: Path(f"/study/{label}") for label in bundles}
    )

    assert report["schema"] == st.HF_ZSPACE_POLARITY_CORPUS_REPORT_SCHEMA
    assert report["status"] == "ready"
    assert report["corpus_count"] == 3
    assert report["seed_count_per_corpus"] == 3
    assert report["observation_count"] == 9
    assert report["bounded_polarity_improvement_observed"] is True
    assert report["efficacy_claim_ready"] is False
    rust = report["rust_evidence"]
    assert rust["semantic_owner"] == "st-core::runtime::zspace_evidence"
    assert (
        rust["contrasts"]["polarity_effect"][  # type: ignore[index]
            "corpus_left_arm_win_count"
        ]
        == 3
    )
    assert report["report_id"].startswith("sha256:")
    assert report["corpora"][0]["completion_event_id"] == "sha256:" + "b" * 64
    assert "summary_sha256" not in report["corpora"][0]


def test_polarity_corpus_report_ignores_mutable_summary_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {label: _polarity_corpus_bundle(label) for label in ("a", "b", "c")}
    monkeypatch.setattr(
        study,
        "_polarity_corpus_study_bundle",
        lambda label, _path: bundles[label],
    )
    studies = {label: Path(f"/study/{label}") for label in bundles}

    first = st.compare_hf_zspace_optimizer_polarity_studies(studies)
    for index, bundle in enumerate(bundles.values()):
        bundle["summary_sha256"] = "sha256:" + str(index + 1) * 64
    second = st.compare_hf_zspace_optimizer_polarity_studies(studies)

    assert second == first


def test_polarity_corpus_report_id_ignores_presentation_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {label: _polarity_corpus_bundle(label) for label in ("a", "b", "c")}
    monkeypatch.setattr(
        study,
        "_polarity_corpus_study_bundle",
        lambda label, _path: bundles[label],
    )
    studies = {label: Path(f"/study/{label}") for label in bundles}

    first = st.compare_hf_zspace_optimizer_polarity_studies(studies)
    for index, bundle in enumerate(bundles.values()):
        bundle["label"] = f"alias-{index}"
        bundle["study_dir"] = f"/relocated/{index}"
        bundle["source_args"] = ["--train-file", f"/relocated/{index}.txt"]
        bundle["study_id"] = "sha256:" + str(index + 1) * 64
        bundle["plan_sha256"] = "sha256:" + str(index + 4) * 64
        bundle["completion_event_id"] = "sha256:" + str(index + 7) * 64
        bundle["polarity_report_sha256"] = "sha256:" + chr(ord("d") + index) * 64
    second = st.compare_hf_zspace_optimizer_polarity_studies(studies)

    assert second["corpora"] != first["corpora"]
    assert second["report_identity"] == first["report_identity"]
    assert second["report_id"] == first["report_id"]
    assert all(
        set(corpus) == {"corpus_id"}
        for corpus in second["report_identity"]["corpora"]
    )


def test_polarity_corpus_comparison_rejects_protocol_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {label: _polarity_corpus_bundle(label) for label in ("a", "b", "c")}
    bundles["c"]["protocol_id"] = "sha256:" + "f" * 64
    monkeypatch.setattr(
        study,
        "_polarity_corpus_study_bundle",
        lambda label, _path: bundles[label],
    )

    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="protocol_id"):
        st.compare_hf_zspace_optimizer_polarity_studies(
            {label: Path(f"/study/{label}") for label in bundles}
        )


def test_polarity_corpus_bundle_requires_a_sealed_completion_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")

    def execute(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
        del cwd
        _write_completed_run(run)
        return 0, 0.25

    def compare(_paths: list[Path]) -> dict[str, object]:
        return {
            "schema": "spiraltorch.hf_zspace_polarity_ablation.v1",
            "status": "ready",
            "matched_seed_count": 1,
            "seeds": [13],
            "polarity_seeds": [
                {
                    "seed": 13,
                    "status": "ready",
                    "trajectory_id": "sha256:" + "3" * 64,
                    "trajectory_policy_id": "sha256:" + "4" * 64,
                    "control_sequence_id": "sha256:" + "5" * 64,
                    "nominal_schedule_sequence_id": "sha256:" + "6" * 64,
                    "contrasts": {
                        "dose_normalized_shape_effect": 0.002,
                        "complement_shape_effect": -0.001,
                        "polarity_effect": -0.003,
                    },
                }
            ],
            "error_count": 0,
            "errors": [],
        }

    monkeypatch.setattr(study, "_execute_run", execute)
    monkeypatch.setattr(
        study,
        "compare_hf_zspace_optimizer_polarity_run_cards",
        compare,
    )
    root = tmp_path / "polarity"
    summary = st.run_hf_zspace_optimizer_polarity_study(
        study_dir=root,
        seeds=[13],
        bridge_args=_bridge_args(),
        bridge_script=bridge,
        launch_cwd=tmp_path,
        min_free_disk_gb=0.0,
        execute=True,
    )

    assert summary["status"] == "ready"
    bundle = study._polarity_corpus_study_bundle("alpha", root)
    assert bundle["study_id"] == summary["study_id"]
    assert bundle["protocol_payload"]["corpus_source_flags"] == ["--train-file"]  # type: ignore[index]
    event_path = root / study.HF_ZSPACE_POLARITY_STUDY_EVENTS_FILENAME
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(st.HFZSpaceFactorizedStudyError, match="blank row"):
        study._polarity_corpus_study_bundle("alpha", root)


def test_polarity_corpus_compare_cli_parses_labeled_studies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def compare(studies: Mapping[str, Path]) -> dict[str, object]:
        captured.update(studies)
        return {
            "status": "ready",
            "report_id": "sha256:" + "1" * 64,
            "corpus_count": 3,
            "seed_count_per_corpus": 3,
        }

    monkeypatch.setattr(hf_cli, "compare_hf_zspace_optimizer_polarity_studies", compare)
    monkeypatch.setattr(
        hf_cli,
        "write_hf_zspace_optimizer_polarity_corpus_report",
        lambda report, path: Path(path).write_text(
            json.dumps(report), encoding="utf-8"
        ),
    )
    output = tmp_path / "corpus-report.json"

    status = hf_cli.zspace_optimizer_polarity_corpus_compare_main(
        [
            "--study",
            "alpha=/study/a",
            "--study",
            "beta=/study/b",
            "--study",
            "gamma=/study/c",
            "--out",
            str(output),
        ]
    )

    assert status == 0
    assert captured == {
        "alpha": Path("/study/a"),
        "beta": Path("/study/b"),
        "gamma": Path("/study/c"),
    }
    assert output.is_file()
    assert "corpora=3" in capsys.readouterr().out


def test_polarity_corpus_runner_reuses_substudy_executor_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _fake_bridge(tmp_path / "bridge.py")
    corpora = {}
    for label in ("a", "b", "c"):
        path = tmp_path / f"{label}.txt"
        path.write_text(f"{label} corpus\n", encoding="utf-8")
        corpora[label] = path
    executed: list[str] = []

    def run_substudy(plan, **_kwargs):
        label = Path(str(plan["study_dir"])).name
        executed.append(label)
        return {
            "status": "ready",
            "completed_run_count": 9,
            "run_count": 9,
        }

    rust_evidence = {
        "status": "ready",
        "evidence_id": "sha256:" + "e" * 64,
        "evidence_boundary": "bounded test evidence",
    }
    aggregate = {
        "status": "ready",
        "report_id": "sha256:" + "f" * 64,
        "rust_evidence": rust_evidence,
    }
    monkeypatch.setattr(study, "_run_hf_zspace_optimizer_study_plan", run_substudy)
    monkeypatch.setattr(
        study,
        "compare_hf_zspace_optimizer_polarity_studies",
        lambda _studies: aggregate,
    )

    summary = st.run_hf_zspace_optimizer_polarity_corpus_study(
        study_dir=tmp_path / "corpus-study",
        corpora=corpora,
        seeds=[13, 17, 23],
        bridge_args=_shared_corpus_bridge_args(),
        bridge_script=bridge,
        launch_cwd=tmp_path,
        min_free_disk_gb=0.0,
        execute=True,
    )

    assert executed == ["a", "b", "c"]
    assert summary["status"] == "ready"
    assert summary["completed_corpus_count"] == 3
    assert summary["completed_run_count"] == 27
    assert summary["remaining_run_count"] == 0
    assert summary["polarity_corpus_report_id"] == aggregate["report_id"]
    assert summary["rust_evidence_id"] == rust_evidence["evidence_id"]
    report_path = Path(str(summary["polarity_corpus_report"]))
    assert json.loads(report_path.read_text(encoding="utf-8")) == aggregate
