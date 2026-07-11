from __future__ import annotations

from types import SimpleNamespace

from spiraltorch.hf_dataset_identity import (
    HF_DATASET_INPUT_IDENTITY_SCHEMA,
    HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA,
    HF_TOKENIZED_DATASET_IDENTITY_SCHEMA,
    hf_dataset_input_identity_lines,
    hf_dataset_input_identity_report,
    hf_dataset_materialization_identity_lines,
    hf_dataset_materialization_identity_report,
    hf_tokenized_dataset_identity_lines,
    hf_tokenized_dataset_identity_report,
)


COMMIT = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"


class _HubApi:
    def __init__(self, commit: str = COMMIT, dataset_id: str = "org/corpus") -> None:
        self.commit = commit
        self.dataset_id = dataset_id
        self.calls: list[tuple[str, str | None]] = []

    def dataset_info(self, dataset_name: str, *, revision: str | None = None):
        self.calls.append((dataset_name, revision))
        return SimpleNamespace(id=self.dataset_id, sha=self.commit)


class _FailingHubApi:
    def dataset_info(self, dataset_name: str, *, revision: str | None = None):
        raise RuntimeError(f"offline: {dataset_name}@{revision}")


def _report(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "dataset_name": "org/corpus",
        "dataset_config": "clean",
        "requested_revision": "main",
        "train_split": "train[:90%]",
        "eval_split": "train[90%:]",
        "text_column": "text",
        "hub_api": _HubApi(),
    }
    kwargs.update(overrides)
    return hf_dataset_input_identity_report(**kwargs)


def test_dataset_identity_resolves_branch_and_is_commit_stable() -> None:
    api = _HubApi()
    branch = _report(hub_api=api)
    pinned = _report(requested_revision=COMMIT, hub_api=_FailingHubApi())

    assert branch["schema"] == HF_DATASET_INPUT_IDENTITY_SCHEMA
    assert branch["status"] == "ready"
    assert branch["effective_revision"] == COMMIT
    assert branch["effective_dataset_name"] == "org/corpus"
    assert branch["revision_resolution_source"] == "hub_dataset_info"
    assert api.calls == [("org/corpus", "main")]
    assert pinned["status"] == "ready"
    assert pinned["revision_resolution_source"] == "requested_commit"
    assert pinned["observed_identity_id"] == branch["observed_identity_id"]
    assert "status=ready" in hf_dataset_input_identity_lines(branch)[0]


def test_dataset_identity_canonicalizes_hub_alias_before_pinning() -> None:
    resolved = _report(
        dataset_name="wikitext",
        hub_api=_HubApi(dataset_id="Salesforce/wikitext"),
    )
    pinned = _report(
        dataset_name="Salesforce/wikitext",
        requested_revision=COMMIT,
        hub_api=_FailingHubApi(),
    )

    assert resolved["dataset_name"] == "wikitext"
    assert resolved["effective_dataset_name"] == "Salesforce/wikitext"
    assert resolved["observed_identity_id"] == pinned["observed_identity_id"]


def test_dataset_identity_rejects_commit_or_selection_drift() -> None:
    ready = _report()
    expected = str(ready["observed_identity_id"])
    verified = _report(requested_revision=COMMIT, expected_identity_id=expected)
    drifted_commit = _report(
        requested_revision="0" * 40,
        expected_identity_id=expected,
    )
    drifted_split = _report(
        requested_revision=COMMIT,
        train_split="validation",
        expected_identity_id=expected,
    )

    assert verified["status"] == "ready"
    assert verified["expected_identity_verified"] is True
    assert drifted_commit["status"] == "blocked"
    assert drifted_split["status"] == "blocked"


def test_dataset_identity_reports_resolution_and_local_scope_honestly() -> None:
    incomplete = _report(hub_api=_FailingHubApi())
    blocked = _report(
        hub_api=_FailingHubApi(),
        expected_identity_id=f"sha256:{'0' * 64}",
    )
    local = _report(local_files=True, hub_api=_FailingHubApi())

    assert incomplete["status"] == "evidence_incomplete"
    assert incomplete["observed_identity_id"] is None
    assert blocked["status"] == "blocked"
    assert local["status"] == "not_applicable"
    assert local["coverage"] == "delegated_to_local_training_input_identity"


def test_materialization_identity_hashes_exact_selected_rows_in_order() -> None:
    train = [{"text": "alpha"}, {"text": "beta\n"}, {"text": "猫"}]
    evaluation = [{"text": "held out"}]

    first = hf_dataset_materialization_identity_report(
        train_dataset=train,
        eval_dataset=evaluation,
    )
    replay = hf_dataset_materialization_identity_report(
        train_dataset=[dict(row) for row in train],
        eval_dataset=[dict(row) for row in evaluation],
        expected_identity_id=str(first["observed_identity_id"]),
    )

    assert first["schema"] == HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA
    assert first["status"] == "ready"
    assert first["materialized_rows_verified"] is True
    assert first["total_rows"] == 4
    assert first["total_utf8_bytes"] == len("alphabeta\n猫held out".encode())
    assert replay["status"] == "ready"
    assert replay["expected_identity_verified"] is True
    assert replay["observed_identity_id"] == first["observed_identity_id"]
    line = hf_dataset_materialization_identity_lines(replay)[0]
    assert "status=ready" in line
    assert "rows=4" in line


def test_materialization_identity_rejects_content_order_and_split_drift() -> None:
    train = [{"text": "alpha"}, {"text": "beta"}]
    baseline = hf_dataset_materialization_identity_report(train_dataset=train)
    expected = str(baseline["observed_identity_id"])

    content_drift = hf_dataset_materialization_identity_report(
        train_dataset=[{"text": "alpha"}, {"text": "beta "}],
        expected_identity_id=expected,
    )
    order_drift = hf_dataset_materialization_identity_report(
        train_dataset=list(reversed(train)),
        expected_identity_id=expected,
    )
    split_drift = hf_dataset_materialization_identity_report(
        train_dataset=train,
        eval_dataset=[],
        expected_identity_id=expected,
    )

    assert content_drift["status"] == "blocked"
    assert order_drift["status"] == "blocked"
    assert split_drift["status"] == "blocked"


def test_materialization_identity_blocks_unhashable_text_rows() -> None:
    missing = hf_dataset_materialization_identity_report(
        train_dataset=[{"body": "alpha"}],
    )
    non_text = hf_dataset_materialization_identity_report(
        train_dataset=[{"text": None}],
    )

    assert missing["status"] == "blocked"
    assert missing["observed_identity_id"] is None
    assert missing["materialized_rows_verified"] is False
    assert "does not contain text column" in str(missing["errors"])
    assert non_text["status"] == "blocked"
    assert "must be str" in str(non_text["errors"])


def test_materialization_identity_contains_dataset_access_failures() -> None:
    class BrokenDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            raise OSError(f"row {index} unavailable")

    report = hf_dataset_materialization_identity_report(
        train_dataset=BrokenDataset(),
    )

    assert report["status"] == "blocked"
    assert report["observed_identity_id"] is None
    assert "row 0 unavailable" in str(report["errors"])


def test_tokenized_identity_hashes_every_block_column_and_split_in_order() -> None:
    train = [
        {
            "input_ids": [10, 11, 12],
            "attention_mask": [1, 1, 1],
            "labels": [10, 11, 12],
        },
        {
            "input_ids": [20, 21, 22],
            "attention_mask": [1, 1, 1],
            "labels": [20, 21, 22],
        },
    ]
    evaluation = [
        {
            "input_ids": [30, 31],
            "attention_mask": [1, 1],
            "labels": [30, 31],
        }
    ]

    first = hf_tokenized_dataset_identity_report(
        train_dataset=train,
        eval_dataset=evaluation,
    )
    replay = hf_tokenized_dataset_identity_report(
        train_dataset=[dict(row) for row in train],
        eval_dataset=[dict(row) for row in evaluation],
        expected_identity_id=str(first["observed_identity_id"]),
    )

    assert first["schema"] == HF_TOKENIZED_DATASET_IDENTITY_SCHEMA
    assert first["status"] == "ready"
    assert first["tokenized_rows_verified"] is True
    assert first["total_rows"] == 3
    assert first["total_input_tokens"] == 8
    assert first["total_label_tokens"] == 8
    assert replay["status"] == "ready"
    assert replay["expected_identity_verified"] is True
    assert replay["observed_identity_id"] == first["observed_identity_id"]
    line = hf_tokenized_dataset_identity_lines(replay)[0]
    assert "status=ready" in line
    assert "input_tokens=8" in line


def test_tokenized_identity_rejects_value_order_column_and_split_drift() -> None:
    train = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
        {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [3, 4]},
    ]
    baseline = hf_tokenized_dataset_identity_report(train_dataset=train)
    expected = str(baseline["observed_identity_id"])

    value_drift = [dict(row) for row in train]
    value_drift[1] = dict(value_drift[1], labels=[3, 9])
    extra_column = [dict(row, token_type_ids=[0, 0]) for row in train]
    reports = [
        hf_tokenized_dataset_identity_report(
            train_dataset=value_drift,
            expected_identity_id=expected,
        ),
        hf_tokenized_dataset_identity_report(
            train_dataset=list(reversed(train)),
            expected_identity_id=expected,
        ),
        hf_tokenized_dataset_identity_report(
            train_dataset=extra_column,
            expected_identity_id=expected,
        ),
        hf_tokenized_dataset_identity_report(
            train_dataset=train,
            eval_dataset=[],
            expected_identity_id=expected,
        ),
    ]

    assert all(report["status"] == "blocked" for report in reports)


def test_tokenized_identity_blocks_invalid_trainer_rows() -> None:
    missing_labels = hf_tokenized_dataset_identity_report(
        train_dataset=[{"input_ids": [1, 2]}],
    )
    non_integer_ids = hf_tokenized_dataset_identity_report(
        train_dataset=[{"input_ids": [1, 2.5], "labels": [1, 2]}],
    )
    inconsistent_columns = hf_tokenized_dataset_identity_report(
        train_dataset=[
            {"input_ids": [1], "labels": [1]},
            {"input_ids": [2], "labels": [2], "attention_mask": [1]},
        ],
    )

    assert missing_labels["status"] == "blocked"
    assert "missing required columns" in str(missing_labels["errors"])
    assert non_integer_ids["status"] == "blocked"
    assert "integer token id" in str(non_integer_ids["errors"])
    assert inconsistent_columns["status"] == "blocked"
    assert "columns changed" in str(inconsistent_columns["errors"])
