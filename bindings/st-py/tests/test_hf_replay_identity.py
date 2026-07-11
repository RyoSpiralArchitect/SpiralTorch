from __future__ import annotations

from spiraltorch.hf_replay_identity import (
    HF_FINETUNE_REPLAY_IDENTITY_SCHEMA,
    hf_finetune_replay_identity_lines,
    hf_finetune_replay_identity_report,
)


def _identity(seed: str) -> str:
    return f"sha256:{seed * 64}"


def _report(
    schema: str,
    identity_id: str | None,
    *,
    id_field: str = "observed_identity_id",
    status: str = "ready",
    verified: bool | None = True,
    **extra: object,
) -> dict[str, object]:
    return {
        "schema": schema,
        "status": status,
        id_field: identity_id,
        "identity_verified": verified,
        "captured_at": "ignored",
        "resolved_path": "/ignored/path",
        **extra,
    }


def _kwargs() -> dict[str, object]:
    return {
        "training_input_identity": _report(
            "training-input.v1",
            None,
            id_field="observed_input_id",
            status="not_applicable",
            verified=None,
        ),
        "dataset_input_identity": _report("dataset-input.v1", _identity("1")),
        "dataset_materialization_identity": _report(
            "dataset-materialization.v1",
            _identity("2"),
        ),
        "tokenized_dataset_identity": _report(
            "tokenized-dataset.v1",
            _identity("3"),
        ),
        "model_runtime_identity": _report("model-runtime.v1", _identity("4")),
        "execution_identity": _report("execution.v1", _identity("5")),
        "training_recipe_identity": _report("training-recipe.v1", _identity("6")),
    }


def test_replay_identity_binds_ready_remote_run_components() -> None:
    report = hf_finetune_replay_identity_report(**_kwargs())

    assert report["schema"] == HF_FINETUNE_REPLAY_IDENTITY_SCHEMA
    assert report["status"] == "ready"
    assert report["identity_verified"] is True
    assert report["path_independent"] is True
    assert report["component_count"] == 8
    assert report["ready_component_count"] == 6
    assert report["components"]["training_input"]["status"] == "not_applicable"
    line = hf_finetune_replay_identity_lines(report)[0]
    assert "status=ready" in line
    assert "ready=6/6" in line
    assert "applicable=6/8" in line


def test_replay_identity_is_stable_across_report_paths_and_timestamps() -> None:
    first = hf_finetune_replay_identity_report(**_kwargs())
    relocated = _kwargs()
    relocated["model_runtime_identity"] = {
        **relocated["model_runtime_identity"],
        "captured_at": "different",
        "resolved_path": "/relocated/model",
    }
    second = hf_finetune_replay_identity_report(**relocated)

    assert first["observed_identity_id"] == second["observed_identity_id"]


def test_replay_identity_accepts_local_input_without_remote_dataset() -> None:
    kwargs = _kwargs()
    kwargs["training_input_identity"] = _report(
        "training-input.v1",
        _identity("7"),
        id_field="observed_input_id",
    )
    kwargs["dataset_input_identity"] = _report(
        "dataset-input.v1",
        None,
        status="not_applicable",
        verified=False,
    )

    report = hf_finetune_replay_identity_report(**kwargs)

    assert report["status"] == "ready"
    assert report["components"]["training_input"]["status"] == "ready"


def test_replay_identity_requires_at_least_one_source_identity() -> None:
    kwargs = _kwargs()
    kwargs["dataset_input_identity"] = _report(
        "dataset-input.v1",
        None,
        status="not_applicable",
        verified=False,
    )

    report = hf_finetune_replay_identity_report(**kwargs)

    assert report["status"] == "blocked"
    assert report["observed_identity_id"] is None
    assert "must provide a ready source identity" in str(report["errors"])


def test_replay_identity_fails_closed_on_missing_or_unverified_component() -> None:
    missing = _kwargs()
    missing["tokenized_dataset_identity"] = None
    unverified = _kwargs()
    unverified["execution_identity"] = _report(
        "execution.v1",
        _identity("5"),
        verified=False,
    )

    missing_report = hf_finetune_replay_identity_report(**missing)
    unverified_report = hf_finetune_replay_identity_report(**unverified)

    assert missing_report["status"] == "blocked"
    assert "tokenized_dataset: identity report is required" in str(
        missing_report["errors"]
    )
    assert unverified_report["status"] == "blocked"
    assert "execution: identity is not verified" in str(unverified_report["errors"])


def test_replay_identity_verifies_exact_replay_and_rejects_component_drift() -> None:
    adopted = hf_finetune_replay_identity_report(**_kwargs())
    expected = str(adopted["observed_identity_id"])
    replay = hf_finetune_replay_identity_report(
        **_kwargs(),
        expected_identity_id=expected,
    )
    drifted = _kwargs()
    drifted["training_recipe_identity"] = _report(
        "training-recipe.v1",
        _identity("8"),
    )
    blocked = hf_finetune_replay_identity_report(
        **drifted,
        expected_identity_id=expected,
    )

    assert replay["status"] == "ready"
    assert replay["expected_identity_verified"] is True
    assert blocked["status"] == "blocked"
    assert blocked["expected_identity_verified"] is False


def test_replay_identity_binds_required_adapter_lineage() -> None:
    adapter = _report(
        "adapter-input.v1",
        _identity("9"),
        id_field="observed_adapter_id",
        observed_lineage_depth=3,
        observed_root_adapter_id=_identity("a"),
    )
    first = hf_finetune_replay_identity_report(
        **_kwargs(),
        adapter_input_identity=adapter,
        adapter_input_required=True,
    )
    changed = dict(adapter, observed_lineage_depth=4)
    second = hf_finetune_replay_identity_report(
        **_kwargs(),
        adapter_input_identity=changed,
        adapter_input_required=True,
    )
    missing = hf_finetune_replay_identity_report(
        **_kwargs(),
        adapter_input_required=True,
    )

    assert first["status"] == "ready"
    assert first["observed_identity_id"] != second["observed_identity_id"]
    assert missing["status"] == "blocked"
    assert "adapter_input: identity report is required" in str(missing["errors"])


def test_replay_identity_rejects_malformed_component_identity() -> None:
    kwargs = _kwargs()
    kwargs["model_runtime_identity"] = _report("model-runtime.v1", "sha256:nope")

    report = hf_finetune_replay_identity_report(**kwargs)

    assert report["status"] == "blocked"
    assert report["observed_identity_id"] is None
    assert "model_runtime identity id must be" in str(report["errors"])
