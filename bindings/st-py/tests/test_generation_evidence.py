from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import spiraltorch as st


GENERATION_SWEEP_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_gpt2_zspace_generation_control_sweep.py"
)
PYTHIA_PILOT_PROMPTS = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_generation_evidence_pythia70m_pilot_prompts.json"
)
MODEL_CONFIGS = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hf_finetune_model_configs.example.json"
)
PYTHIA_PILOT_PROTOCOL = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "benchmarks"
    / "hf_zspace_pythia70m_polarity_pilot_protocol_v1.json"
)


def _load_generation_sweep_example() -> Any:
    spec = importlib.util.spec_from_file_location(
        "spiraltorch_generation_evidence_sweep_test",
        GENERATION_SWEEP_EXAMPLE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load generation-control sweep example")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_id(character: str) -> str:
    return "sha256:" + character * 64


def _report(
    samples: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    return st.zspace_generation_evidence(
        protocol_id=_sha_id("a"),
        runtime_identity_id=_sha_id("b"),
        model_artifact_id=_sha_id("c"),
        prompt_set_id=_sha_id("d"),
        decoding_config_id=_sha_id("e"),
        samples=(
            samples
            if samples is not None
            else [
                {
                    "prompt_id": _sha_id("1"),
                    "seed": 13,
                    "continuation_token_ids": (1, 2, 1, 2, 1, 2),
                },
                {
                    "prompt_id": _sha_id("2"),
                    "seed": 13,
                    "continuation_token_ids": [7, 8, 9],
                },
            ]
        ),
    )


def test_generation_evidence_is_rust_owned_and_tamper_evident() -> None:
    report = _report()

    assert report["contract_version"] == st.ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION
    assert report["kind"] == st.ZSPACE_GENERATION_EVIDENCE_KIND
    assert report["semantic_owner"] == st.ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER
    assert report["semantic_backend"] == "rust"
    assert report["sample_count"] == 2
    assert report["ngram_orders"] == [1, 2, 3, 4]
    assert report["efficacy_claim_ready"] is False
    assert report["loop_score_rule"] == st.ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE
    assert report["request"]["samples"][0]["prompt_id"] == _sha_id("1")  # type: ignore[index]
    repeated = report["samples"][0]  # type: ignore[index]
    assert repeated["periodic_loop_detected"] is True
    assert repeated["periodic_suffix_period"] == 2
    assert repeated["periodic_suffix_repeated_token_count"] == 4
    assert report["aggregate"]["periodic_loop_sample_count"] == 1  # type: ignore[index]
    assert report["aggregate"]["periodic_loop_sample_ratio"] == 0.5  # type: ignore[index]
    assert st.validate_zspace_generation_evidence(report) == report

    tampered = copy.deepcopy(report)
    tampered["aggregate"]["periodic_loop_sample_count"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical Rust generation evidence"):
        st.validate_zspace_generation_evidence(tampered)


def test_generation_evidence_canonicalizes_sample_order() -> None:
    samples = [
        {
            "prompt_id": _sha_id("2"),
            "seed": 17,
            "continuation_token_ids": [2],
        },
        {
            "prompt_id": _sha_id("1"),
            "seed": 13,
            "continuation_token_ids": [1],
        },
    ]

    assert _report(samples) == _report(list(reversed(samples)))


def test_generation_evidence_records_empty_continuations() -> None:
    report = _report(
        [
            {
                "prompt_id": _sha_id("1"),
                "seed": 13,
                "continuation_token_ids": [],
            }
        ]
    )

    assert report["samples"][0]["empty_continuation"] is True  # type: ignore[index]
    assert report["aggregate"]["empty_sample_count"] == 1  # type: ignore[index]
    assert report["aggregate"]["total_token_count"] == 0  # type: ignore[index]
    assert (
        report["aggregate"]["periodic_suffix_repeated_token_ratio"] is None  # type: ignore[index]
    )


def test_generation_evidence_rejects_duplicate_prompt_seed() -> None:
    samples = [
        {
            "prompt_id": _sha_id("1"),
            "seed": 13,
            "continuation_token_ids": [1],
        },
        {
            "prompt_id": _sha_id("1"),
            "seed": 13,
            "continuation_token_ids": [2],
        },
    ]

    with pytest.raises(ValueError, match="duplicate generation evidence sample"):
        _report(samples)


def test_generation_evidence_rejects_cross_client_unsafe_token_id() -> None:
    with pytest.raises(ValueError, match="exceeds the cross-client maximum"):
        _report(
            [
                {
                    "prompt_id": _sha_id("1"),
                    "seed": 13,
                    "continuation_token_ids": [
                        st.ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER + 1
                    ],
                }
            ]
        )


def test_generation_evidence_bounds_facade_snapshots_before_native_work() -> None:
    sample = {
        "prompt_id": _sha_id("1"),
        "seed": 13,
        "continuation_token_ids": [1],
    }
    with pytest.raises(ValueError, match="sample count exceeds maximum 10000"):
        _report([sample] * (st.ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES + 1))

    overwide = dict(sample)
    overwide["client_metric"] = "local"
    with pytest.raises(ValueError, match="field count exceeds maximum 3"):
        _report([overwide])


def test_generation_evidence_native_ingress_rejects_deep_reports() -> None:
    nested: object = None
    for _ in range(40):
        nested = [nested]

    with pytest.raises(ValueError, match="too deeply nested"):
        st.validate_zspace_generation_evidence({"nested": nested})


def test_generation_evidence_dict_subclasses_cannot_bypass_snapshot_bounds() -> None:
    class HostileDict(dict[str, object]):
        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

        def copy(self) -> dict[str, object]:
            raise AssertionError("overridden copy must not run")

        def items(self):
            raise AssertionError("overridden items must not run")

    report = _report(
        [
            HostileDict(
                {
                    "prompt_id": _sha_id("1"),
                    "seed": 13,
                    "continuation_token_ids": [1, 2],
                }
            )
        ]
    )
    assert st.validate_zspace_generation_evidence(HostileDict(report)) == report


def test_generation_evidence_rejects_active_container_hooks() -> None:
    class ActiveMapping(Mapping[str, object]):
        def __getitem__(self, _key: str) -> object:
            raise AssertionError("custom __getitem__ must not run")

        def __iter__(self):
            raise AssertionError("custom __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("custom __len__ must not run")

        def items(self):
            raise AssertionError("custom items must not run")

    class ActiveSequence(Sequence[Mapping[str, object]]):
        def __getitem__(self, _index: int) -> Mapping[str, object]:
            raise AssertionError("custom __getitem__ must not run")

        def __iter__(self):
            raise AssertionError("custom __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("custom __len__ must not run")

    class HostileList(list[Mapping[str, object]]):
        def __iter__(self):
            raise AssertionError("overridden __iter__ must not run")

        def __len__(self) -> int:
            raise AssertionError("overridden __len__ must not run")

    sample = {
        "prompt_id": _sha_id("1"),
        "seed": 13,
        "continuation_token_ids": [1, 2],
    }
    assert _report(HostileList([sample]))["sample_count"] == 1

    with pytest.raises(TypeError, match="list or tuple for bounded admission"):
        _report(ActiveSequence())
    with pytest.raises(TypeError, match="dict-backed mapping"):
        _report([ActiveMapping()])
    with pytest.raises(TypeError, match="dict-backed mapping"):
        st.validate_zspace_generation_evidence(ActiveMapping())


def test_generation_evidence_public_surface_is_exported() -> None:
    expected = {
        "ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION",
        "ZSPACE_GENERATION_EVIDENCE_KIND",
        "ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE",
        "ZSPACE_GENERATION_EVIDENCE_METRIC_RULE",
        "validate_zspace_generation_evidence",
        "zspace_generation_evidence",
    }

    assert expected <= set(st.__all__)


def test_generation_sweep_routes_continuation_tokens_through_rust() -> None:
    module = _load_generation_sweep_example()
    args = module.parse_args(
        [
            "--prompt",
            "SpiralTorch is",
            "--max-new-tokens",
            "6",
            "--seed",
            "17",
        ]
    )
    seeded: list[int] = []

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeTorch:
        cuda = SimpleNamespace(manual_seed_all=lambda seed: seeded.append(seed))

        @staticmethod
        def manual_seed(seed: int) -> None:
            seeded.append(seed)

        @staticmethod
        def no_grad() -> NoGrad:
            return NoGrad()

    class FakeInput:
        shape = (1, 2)

    class FakeModel:
        def parameters(self):
            return iter(())

        def generate(self, **_kwargs: object) -> list[list[int]]:
            return [[10, 11, 1, 2, 1, 2, 1, 2]]

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        @staticmethod
        def decode(_tokens: object, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return "SpiralTorch is looping"

    run = {"name": "baseline-greedy", "kind": "baseline", "config": {}}
    evidence_context = {
        "protocol_id": _sha_id("a"),
        "runtime_identity_id": _sha_id("b"),
        "model_artifact_id": _sha_id("c"),
        "prompt_set_id": _sha_id("d"),
        "prompt_id": _sha_id("e"),
    }

    row = module._generate_one(
        run=run,
        transformers=SimpleNamespace(),
        torch=FakeTorch(),
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        encoded={"input_ids": FakeInput()},
        args=args,
        evidence_context=evidence_context,
        prompt="SpiralTorch is",
        prompt_label="prompt-0001",
    )

    evidence = row["generation_evidence"]
    assert evidence["request"]["samples"][0]["continuation_token_ids"] == [  # type: ignore[index]
        1,
        2,
        1,
        2,
        1,
        2,
    ]
    assert evidence["samples"][0]["periodic_loop_detected"] is True  # type: ignore[index]
    assert row["repetition"]["metric_backend"] == "rust"  # type: ignore[index]
    assert seeded == [17, 17]


def test_generation_sweep_plan_freezes_seed_prompt_and_decoding_ids() -> None:
    module = _load_generation_sweep_example()
    args = module.parse_args(
        ["--dry-run", "--prompt", "SpiralTorch is", "--seed", "23"]
    )
    runs = module.build_control_runs(args)

    first = module._generation_evidence_plan(args, runs)
    second = module._generation_evidence_plan(args, list(reversed(runs)))

    assert first["semantic_owner"] == st.ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER
    assert first["seed"] == 23
    assert first["protocol_id"] == second["protocol_id"]
    assert first["prompt_set_id"] == second["prompt_set_id"]
    assert first["decoding_config_ids"] == second["decoding_config_ids"]


def test_generation_sweep_binds_a_prespecified_protocol_id() -> None:
    module = _load_generation_sweep_example()
    protocol_id = _sha_id("f")
    args = module.parse_args(
        [
            "--dry-run",
            "--prompt",
            "SpiralTorch is",
            "--generation-evidence-protocol-id",
            protocol_id,
        ]
    )

    plan = module._generation_evidence_plan(args, module.build_control_runs(args))

    assert plan["protocol_id"] == protocol_id
    assert plan["protocol_binding"] == "prespecified_cli_override"

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--dry-run",
                "--prompt",
                "SpiralTorch is",
                "--generation-evidence-protocol-id",
                "sha256:not-a-digest",
            ]
        )


def test_generation_sweep_remote_adapter_identity_requires_a_pinned_revision() -> None:
    module = _load_generation_sweep_example()
    summary = {
        "artifact_kind": "peft_adapter",
        "base_model_name_or_path": "org/base",
        "base_model_revision": None,
        "base_model_commit": "a" * 40,
    }
    first_args = SimpleNamespace(
        model_name="org/adapter",
        adapter_revision="b" * 40,
    )
    second_args = SimpleNamespace(
        model_name="org/adapter",
        adapter_revision="c" * 40,
    )

    first = module._model_artifact_id(first_args, summary, _sha_id("d"))
    second = module._model_artifact_id(second_args, summary, _sha_id("d"))

    assert first != second
    assert module._adapter_loader_kwargs(first_args) == {"revision": "b" * 40}
    first_args.adapter_revision = None
    with pytest.raises(RuntimeError, match="requires --adapter-revision"):
        module._model_artifact_id(first_args, summary, _sha_id("d"))


def test_generation_sweep_validates_fixed_prompt_set_identity(tmp_path: Path) -> None:
    module = _load_generation_sweep_example()
    args = module.parse_args(
        [
            "--dry-run",
            "--model-configs",
            str(MODEL_CONFIGS),
            "--model-profile",
            "pythia-70m-local-smoke",
            "--prompt-set",
            str(PYTHIA_PILOT_PROMPTS),
            "--baseline-only",
            "--no-do-sample",
        ]
    )
    report = module.run_sweep(args)

    assert report["run_count"] == 1
    assert report["prompt_count"] == 8
    assert report["do_sample"] is False
    assert report["prompt_set"]["prompt_set_id"] == (  # type: ignore[index]
        "sha256:4a5029e46c21915c4982c7eb005978ff7759cb04901c49f7d0e9f4b6aaaf94ed"
    )

    payload = {
        "schema": "spiraltorch.hf_generation_prompt_set.v1",
        "prompt_set_id": _sha_id("f"),
        "prompts": [{"label": "fixed", "text": "A fixed prompt"}],
    }
    tampered = tmp_path / "tampered-prompts.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit):
        module.parse_args(["--dry-run", "--prompt-set", str(tampered)])


def test_generation_sweep_rejects_failed_prompt_before_aggregation() -> None:
    module = _load_generation_sweep_example()
    args = module.parse_args(["--prompt", "SpiralTorch is"])

    with pytest.raises(RuntimeError, match="did not complete successfully"):
        module._combine_prompt_generations(
            run={"name": "baseline-greedy", "kind": "baseline", "config": {}},
            generations=[{"status": "blocked"}],
            args=args,
            evidence_context={
                "protocol_id": _sha_id("a"),
                "runtime_identity_id": _sha_id("b"),
                "model_artifact_id": _sha_id("c"),
                "prompt_set_id": _sha_id("d"),
            },
        )


def test_generation_sweep_combines_prompts_with_token_exact_identity() -> None:
    module = _load_generation_sweep_example()
    args = module.parse_args(["--prompt", "SpiralTorch is", "--baseline-only"])
    run = {"name": "baseline-greedy", "kind": "baseline", "config": {}}
    context = {
        "protocol_id": _sha_id("a"),
        "runtime_identity_id": _sha_id("b"),
        "model_artifact_id": _sha_id("c"),
        "prompt_set_id": _sha_id("d"),
    }
    rows = []
    for prompt_id, token_ids, calls in (
        (_sha_id("2"), [7, 8, 9], 3),
        (_sha_id("1"), [1, 2, 1, 2, 1, 2], 7),
    ):
        evidence = st.zspace_generation_evidence(
            **context,
            decoding_config_id=module._decoding_config_id(args, run),
            samples=[
                {
                    "prompt_id": prompt_id,
                    "seed": args.seed,
                    "continuation_token_ids": token_ids,
                }
            ],
        )
        rows.append(
            {
                "status": "ok",
                "generation_evidence": evidence,
                "generation": {"generation_control": {"calls": calls}},
            }
        )

    first = module._combine_prompt_generations(
        run=run,
        generations=rows,
        args=args,
        evidence_context=context,
    )
    second = module._combine_prompt_generations(
        run=run,
        generations=list(reversed(rows)),
        args=args,
        evidence_context=context,
    )

    assert (
        first["generated_continuation_set_id"]
        == second["generated_continuation_set_id"]
    )
    assert first["generation_evidence"]["sample_count"] == 2  # type: ignore[index]
    assert first["generation_evidence"]["request"]["samples"][0][  # type: ignore[index]
        "prompt_id"
    ] == _sha_id("1")
    assert module._summary([first])["max_control_calls"] == 7.0


def test_pythia_pilot_protocol_is_prespecified_and_content_addressed() -> None:
    protocol = json.loads(PYTHIA_PILOT_PROTOCOL.read_text(encoding="utf-8"))
    prompt_set = json.loads(PYTHIA_PILOT_PROMPTS.read_text(encoding="utf-8"))
    scientific_spec = protocol["scientific_spec"]
    encoded = json.dumps(
        scientific_spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert protocol["status"] == "prespecified"
    assert protocol["protocol_id"] == "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert scientific_spec["implementation"]["generation_evidence_git_commit"] == (
        "32b552670bde887102420d340a7769d47e451e89"
    )
    assert (
        scientific_spec["generation_endpoint"]["prompt_set_id"]
        == prompt_set["prompt_set_id"]
    )
    assert scientific_spec["efficacy_claim_ready"] is False
    assert (
        scientific_spec["interpretation"][
            "pilot_rows_reusable_as_final_efficacy_evidence"
        ]
        is False
    )
