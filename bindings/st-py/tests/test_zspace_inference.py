from __future__ import annotations

import math
import types
from typing import Any, Mapping

import pytest

pytest.importorskip("spiraltorch")

import spiraltorch as st

from spiraltorch import (
    ZSpaceControlGradient,
    ZSpaceInferencePipeline,
    ZSpaceInferenceRuntime,
    ZSpacePosterior,
    ZSpacePartialBundle,
    ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS,
    canvas_partial_from_snapshot,
    canvas_coherence_partial,
    coherence_partial_from_diagnostics,
    compile_inference,
    decode_zspace_embedding,
    elliptic_partial_from_telemetry,
    blend_zspace_partials,
    infer_canvas_snapshot,
    infer_canvas_transformer,
    infer_coherence_diagnostics,
    infer_coherence_from_sequencer,
    infer_canvas_with_coherence,
    infer_with_partials,
    infer_from_partial,
    inference_to_mapping,
    weights_partial_from_dlpack,
    weights_partial_from_compat,
    infer_weights_from_dlpack,
    infer_weights_from_compat,
    validate_zspace_coherence_distribution_witness,
    zspace_coherence_distribution_witness,
    zspace_coherence_project,
    zspace_posterior_decode,
    zspace_posterior_project,
    zspace_metric_gradient_projection,
    zspace_partial_fusion,
    zspace_telemetry_fusion,
)


def test_decode_produces_expected_structure():
    vector = [0.12, -0.03, 0.48, -0.2]
    decoded = decode_zspace_embedding(vector)
    assert set(decoded.metrics.keys()) == {"speed", "memory", "stability", "frac", "drs"}
    assert len(decoded.gradient) == len(vector)
    assert math.isclose(sum(decoded.barycentric), 1.0, rel_tol=1e-6)
    assert decoded.spectral_bins == len(vector) // 2 + 1


def test_posterior_decode_exposes_rust_owned_golden_contract():
    contract = zspace_posterior_decode([0.12, -0.03, 0.48, -0.2])

    assert contract["kind"] == "spiraltorch.zspace_posterior_decode"
    assert contract["contract_version"] == "spiraltorch.zspace_posterior.v2"
    assert contract["semantic_owner"] == "st-core::inference::zspace_posterior"
    assert contract["semantic_backend"] == "rust"
    assert "DFT" in contract["fractional_formula"]
    assert "Parseval" in contract["spectral_formula"]
    assert "zero exterior boundaries" in contract["gradient_formula"]
    assert "softplus" in contract["barycentric_formula"]
    assert contract["gradient_basis"] == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    assert contract["energy"] == pytest.approx(0.2857)
    assert contract["spectral_energy"] == pytest.approx(contract["energy"])
    assert contract["parseval_relative_error"] <= 1.0e-14
    assert contract["frac_energy"] == pytest.approx(0.2210090973787923)
    assert contract["fractional_energy_ratio"] == pytest.approx(
        0.7735705193517406
    )
    assert contract["spectral_centroid"] == pytest.approx(0.7415120756037802)
    assert contract["spectral_bins"] == 3
    assert contract["gradient"] == pytest.approx(
        [-0.015, 0.18, -0.085, -0.24]
    )
    assert contract["metrics"] == pytest.approx(
        {
            "speed": 0.4463024613684294,
            "memory": 0.10991043037290935,
            "stability": 0.672934550609384,
            "drs": 0.6413201743341207,
            "frac": 0.6490008507207798,
        }
    )


def test_posterior_projection_delegates_aliases_and_telemetry_to_rust():
    contract = zspace_posterior_project(
        [0.12, -0.03, 0.48, -0.2],
        {"speed": 0.3, "mem": -0.2, "gradient": [2.0, -1.0]},
        gradient_basis="test.posterior.control.v1",
        telemetry={"psi": {"energy": 2.0, "focus": 0.4}},
    )

    assert contract["kind"] == "spiraltorch.zspace_posterior_projection"
    assert "rms_observed" in contract["projection_formula"]
    assert "variance" in contract["telemetry_formula"]
    assert contract["applied"]["memory"] == pytest.approx(-0.2)
    assert "gradient" not in contract["applied"]
    assert contract["gradient"] == pytest.approx([-0.015, 0.18, -0.085, -0.24])
    assert contract["gradient_basis"] == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    assert contract["gradient_source"] == "latent_state"
    control = contract["control_gradient"]
    assert control["source"] == "partial"
    assert control["basis"] == "test.posterior.control.v1"
    assert control["values"] == pytest.approx([2.0, -1.0])
    assert control["dimension"] == 2
    assert control["l2"] == pytest.approx(math.sqrt(5.0))
    assert control["linf"] == pytest.approx(2.0)
    assert contract["telemetry"]["semantic_backend"] == "rust"
    assert contract["telemetry"]["summary"]["variance"] == pytest.approx(0.64)
    assert contract["residual_metric_count"] == 2
    assert contract["residual"] == pytest.approx(0.24233126609703365)
    assert contract["telemetry_reliability"] == pytest.approx(1.0 / 1.64)
    assert contract["confidence"] == pytest.approx(0.4785342427598339)


def test_posterior_contract_rejects_ambiguous_or_invalid_requests():
    with pytest.raises(ValueError, match="both resolve to 'speed'"):
        zspace_posterior_project([0.1], {"speed": 0.2, "velocity": 0.3})
    with pytest.raises(ValueError, match="smoothing"):
        zspace_posterior_project([0.1], smoothing=1.1)
    with pytest.raises(ValueError, match="explicit gradient basis"):
        zspace_posterior_project([0.1], {"gradient": [0.2]})
    with pytest.raises(ValueError, match="at least one"):
        zspace_posterior_decode([])
    with pytest.raises(ValueError, match="fractional order"):
        zspace_posterior_decode([0.1], alpha=0.0)


def test_posterior_requires_the_native_semantic_core(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(st, "_rs", object())

    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        zspace_posterior_decode([0.1])
    with pytest.raises(RuntimeError, match="compiled Rust semantic core"):
        zspace_posterior_project([0.1], {"speed": 0.2})


def test_posterior_rejects_an_untrusted_native_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    native = types.SimpleNamespace(
        _zspace_posterior_decode=lambda _request: {
            "kind": "spiraltorch.zspace_posterior_decode",
            "contract_version": "spiraltorch.zspace_posterior.v2",
            "semantic_owner": "python",
            "semantic_backend": "rust",
        }
    )
    monkeypatch.setattr(st, "_rs", native)

    with pytest.raises(RuntimeError, match="untrusted contract"):
        zspace_posterior_decode([0.1])


@pytest.mark.parametrize(
    "partial",
    [
        {"speed": 0.3, "stab": 0.7},
        {"mem": -0.2, "gradient": [0.1, -0.05, 0.0, 0.2]},
    ],
)
def test_infer_from_partial_overrides_requested_metrics(partial):
    vector = [0.2, -0.1, 0.45, -0.05]
    gradient_basis = "test.partial.control.v1" if "gradient" in partial else None
    result = infer_from_partial(vector, partial, gradient_basis=gradient_basis)
    for key, value in partial.items():
        canonical = {
            "mem": "memory",
            "stab": "stability",
        }.get(key, key)
        if canonical == "gradient":
            assert len(result.gradient) == len(vector)
            assert result.gradient_basis == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
            assert isinstance(result.control_gradient, ZSpaceControlGradient)
            assert result.control_gradient.basis == gradient_basis
            assert result.control_gradient.values == pytest.approx(value)
            continue
        assert math.isclose(result.metrics[canonical], float(value))
    assert 0.0 <= result.confidence <= 1.0
    assert math.isclose(sum(result.barycentric), 1.0, rel_tol=1e-6)


def test_infer_from_partial_accepts_mapping_proxy():
    vector = [0.18, -0.07, 0.29, -0.14]
    gradient_mapping = types.MappingProxyType({0: 0.3, 1: -0.15, 2: 0.05, 3: -0.02})
    partial = types.MappingProxyType({"stab": 0.55, "gradient": gradient_mapping})

    result = infer_from_partial(
        vector, partial, gradient_basis="test.mapping.control.v1"
    )

    assert math.isclose(result.metrics["stability"], 0.55)
    assert result.gradient_basis == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    assert result.control_gradient is not None
    assert result.control_gradient.values == pytest.approx([0.3, -0.15, 0.05, -0.02])
    assert result.control_gradient.basis == "test.mapping.control.v1"


def test_posterior_project_matches_helper():
    vector = [0.42, 0.1, -0.25, 0.08]
    partial = {"speed": 0.1, "mem": 0.05}
    posterior = ZSpacePosterior(vector)
    direct = posterior.project(partial)
    helper = infer_from_partial(vector, partial)
    assert math.isclose(direct.residual, helper.residual)
    assert math.isclose(direct.confidence, helper.confidence)
    assert direct.metrics == helper.metrics


def test_runtime_accumulates_partial_updates():
    vector = [0.3, -0.05, 0.22, -0.11]
    runtime = ZSpaceInferenceRuntime(vector, smoothing=0.25)
    first = runtime.update({"speed": 0.45})
    assert math.isclose(first.metrics["speed"], 0.45)
    second = runtime.update({"stab": 0.4})
    assert math.isclose(second.metrics["speed"], 0.45)
    assert math.isclose(second.metrics["stability"], 0.4)
    runtime.clear()
    assert len(runtime.cached_observations) == 0
    reset = runtime.update()
    assert "speed" in reset.metrics


def test_runtime_without_accumulation_replaces_previous_observations():
    vector = [0.15, -0.2, 0.05, 0.18]
    runtime = ZSpaceInferenceRuntime(vector, accumulate=False)
    runtime.update({"speed": 0.6})
    runtime.update({"stab": 0.2})
    assert "speed" not in runtime.cached_observations


def test_blend_partials_supports_weighted_mean_and_gradient():
    blended = blend_zspace_partials(
        [
            {"speed": 0.2, "gradient": [0.1, -0.1]},
            ZSpacePartialBundle({"speed": 0.6, "gradient": [0.5, 0.1]}, weight=2.0),
        ]
    )
    assert math.isclose(blended["speed"], (0.2 + 0.6 * 2.0) / 3.0, rel_tol=1e-6)
    assert math.isclose(blended["gradient"][0], (0.1 + 0.5 * 2.0) / 3.0, rel_tol=1e-6)
    assert len(blended["gradient"]) == 2


def test_flattened_fusion_preserves_gradient_basis_for_composition():
    basis = "test.compose.control.v1"
    bundle = ZSpacePartialBundle(
        {"speed": 0.4, "gradient": [0.25, -0.1]},
        gradient_basis=basis,
    )

    resolved = bundle.resolved()
    blended = blend_zspace_partials([bundle])

    for payload in (resolved, blended):
        assert payload["gradient_basis"] == basis
        inference = infer_from_partial([0.3, -0.05, 0.22, -0.11], payload)
        assert inference.control_gradient is not None
        assert inference.control_gradient.basis == basis
        assert inference.control_gradient.values == pytest.approx([0.25, -0.1])


def test_runtime_keeps_fused_gradient_basis_out_of_metric_cache():
    basis = "test.runtime.control.v1"
    runtime = ZSpaceInferenceRuntime([0.3, -0.05, 0.22, -0.11])
    partial = blend_zspace_partials(
        [
            ZSpacePartialBundle(
                {"speed": 0.4, "gradient": [0.25, -0.1]},
                gradient_basis=basis,
            )
        ]
    )

    inference = runtime.update(partial)

    assert runtime.gradient_basis == basis
    assert "gradient_basis" not in runtime.cached_observations
    assert inference.control_gradient is not None
    assert inference.control_gradient.basis == basis


def test_telemetry_fusion_exposes_rust_owned_contract():
    contract = zspace_telemetry_fusion(
        {"psi": {"energy": 2.0, "ready": True}, "ignored": [1.0]},
        {"psi.energy": "4.0", "note": "ignored"},
    )

    assert contract["contract_version"] == "spiraltorch.zspace_telemetry_fusion.v1"
    assert contract["semantic_owner"] == "st-core::telemetry::zspace_fusion"
    assert contract["semantic_backend"] == "rust"
    assert contract["payload"] == {"psi.energy": 4.0, "psi.ready": 1.0}
    assert contract["summary"]["count"] == 2
    assert contract["conflict_count"] == 1
    assert contract["ignored_value_count"] == 2

    collection_contract = zspace_telemetry_fusion(
        [{"psi": {"energy": 2.0}}, {"psi.energy": 4.0}]
    )
    assert collection_contract["payload"] == {"psi.energy": 4.0}


def test_partial_fusion_overrides_bundle_weights_and_suppresses_telemetry():
    contract = zspace_partial_fusion(
        [
            ZSpacePartialBundle(
                {"velocity": 1.0, "gradient": [1.0, 2.0]},
                weight=100.0,
                origin="suppressed",
                telemetry={"suppressed_only": 1.0},
            ),
            ZSpacePartialBundle(
                {"speed": 5.0, "gradient": [5.0]},
                weight=1.0,
                origin="active",
                telemetry={"source": 2.0},
            ),
        ],
        weights=[0.0, 3.0],
        telemetry={"external": 7.0},
    )

    assert contract["contract_version"] == "spiraltorch.zspace_partial_fusion.v3"
    assert contract["metrics"] == {"speed": 5.0}
    assert contract["gradient"] == [5.0]
    assert contract["suppressed_count"] == 1
    assert contract["gradient_alignment"] == "strict"
    assert contract["gradient_input_count"] == 1
    assert contract["gradient_dim"] == 1
    assert contract["gradient_padding_applied"] is False
    assert contract["telemetry"]["payload"] == {"external": 7.0, "source": 2.0}
    assert contract["sources"][0]["status"] == "suppressed"


def test_metric_gradient_projection_exposes_rust_owned_basis():
    contract = zspace_metric_gradient_projection(
        {
            "speed": 0.1,
            "memory": 0.2,
            "stability": 0.3,
            "frac": 0.4,
            "drs": -0.5,
        },
        gradient_dim=7,
    )

    assert contract["contract_version"] == (
        "spiraltorch.zspace_metric_gradient_projection.v1"
    )
    assert contract["semantic_owner"] == "st-core::telemetry::zspace_fusion"
    assert contract["semantic_backend"] == "rust"
    assert contract["basis"] == st.ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS
    assert contract["coordinate_channels"] == [
        "speed",
        "memory",
        "stability",
        "frac",
        "drs",
        "speed",
        "memory",
    ]
    assert contract["gradient"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, -0.5, 0.1, 0.2]
    )

    with pytest.raises(ValueError, match="dimension must be in"):
        zspace_metric_gradient_projection(
            {
                "speed": 0.1,
                "memory": 0.2,
                "stability": 0.3,
                "frac": 0.4,
                "drs": -0.5,
            },
            gradient_dim=0,
        )
    with pytest.raises(TypeError, match="dimension must be an integer"):
        zspace_metric_gradient_projection(
            {
                "speed": 0.1,
                "memory": 0.2,
                "stability": 0.3,
                "frac": 0.4,
                "drs": -0.5,
            },
            gradient_dim=1.5,
        )


def test_partial_fusion_rejects_equal_length_gradients_from_different_bases():
    with pytest.raises(ValueError, match="does not match basis"):
        zspace_partial_fusion(
            [
                ZSpacePartialBundle(
                    {"speed": 0.2, "gradient": [0.1, 0.2]},
                    gradient_basis="basis.a.v1",
                ),
                ZSpacePartialBundle(
                    {"speed": 0.4, "gradient": [0.3, 0.4]},
                    gradient_basis="basis.b.v1",
                ),
            ]
        )


def test_partial_fusion_can_replace_heterogeneous_gradients_from_named_metrics():
    contract = zspace_partial_fusion(
        [
            ZSpacePartialBundle(
                {
                    "speed": 1.0,
                    "memory": 2.0,
                    "stability": 3.0,
                    "frac": 4.0,
                    "drs": 5.0,
                    "gradient": [99.0, 98.0],
                },
                gradient_basis="feature.a.v1",
            ),
            ZSpacePartialBundle(
                {
                    "speed": 3.0,
                    "memory": 4.0,
                    "stability": 5.0,
                    "frac": 6.0,
                    "drs": 7.0,
                    "gradient": [-9.0, -8.0, -7.0],
                },
                gradient_basis="feature.b.v1",
            ),
        ],
        metric_gradient_dimension=4,
    )

    assert contract["gradient_source"] == "canonical_metrics"
    assert contract["gradient_basis"] == st.ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS
    assert contract["gradient"] == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert contract["gradient_replaced_source_count"] == 2
    assert contract["metric_gradient_projection_applied"] is True


def test_partial_fusion_requires_explicit_legacy_gradient_padding():
    with pytest.raises(ValueError, match="does not match dimension"):
        zspace_partial_fusion(
            [
                {"speed": -2.0, "gradient": [-2.0, 4.0]},
                {"speed": 3.0, "gradient": [3.0]},
            ],
            strategy="max",
        )

    contract = zspace_partial_fusion(
        [
            {"speed": -2.0, "gradient": [-2.0, 4.0]},
            {"speed": 3.0, "gradient": [3.0]},
        ],
        strategy="max",
        gradient_alignment="pad_zero",
    )

    assert contract["metrics"]["speed"] == 3.0
    assert contract["gradient"] == [3.0, 4.0]
    assert contract["gradient_alignment"] == "pad_zero"
    assert contract["gradient_padding_applied"] is True
    assert contract["gradient_padded_source_count"] == 1
    assert contract["sources"][1]["gradient_padded"] is True


def test_partial_fusion_applies_median_and_sum_to_scalars_and_gradients():
    partials = [
        {"speed": 0.0, "gradient": [0.0, 6.0]},
        {"speed": 2.0, "gradient": [2.0, 4.0]},
        {"speed": 10.0, "gradient": [10.0, 8.0]},
    ]

    median = zspace_partial_fusion(partials, strategy="median")
    total = zspace_partial_fusion(partials, strategy="sum")

    assert median["metrics"]["speed"] == 2.0
    assert median["gradient"] == [2.0, 6.0]
    assert total["metrics"]["speed"] == 12.0
    assert total["gradient"] == [12.0, 18.0]

    weighted_partials = [
        {"speed": 0.0, "gradient": [0.0]},
        {"speed": 10.0, "gradient": [10.0]},
    ]
    weighted_sum = zspace_partial_fusion(
        weighted_partials, weights=[1.0, 3.0], strategy="sum"
    )
    weighted_median = zspace_partial_fusion(
        weighted_partials, weights=[1.0, 3.0], strategy="median"
    )
    assert weighted_sum["metrics"]["speed"] == 30.0
    assert weighted_sum["gradient"] == [30.0]
    assert weighted_median["metrics"]["speed"] == 10.0
    assert weighted_median["gradient"] == [10.0]


def test_elliptic_telemetry_delegates_scalar_and_gradient_reduction_to_rust():
    telemetry = [
        {"curvature_radius": 0.0, "rotor_transport": [0.0, 6.0]},
        {"curvature_radius": 2.0, "rotor_transport": [2.0, 4.0]},
        {"curvature_radius": 10.0, "rotor_transport": [10.0, 8.0]},
    ]

    bundle = elliptic_partial_from_telemetry(telemetry, aggregate="median")

    assert bundle.metrics["elliptic_curvature"] == 2.0
    assert bundle.metrics["gradient"] == [2.0, 6.0]
    assert bundle.gradient_basis == "spiraltorch.elliptic.rotor_transport.v1"
    inference = infer_from_partial(
        [0.12, -0.03, 0.48, -0.2],
        bundle.metrics,
        gradient_basis=bundle.gradient_basis,
    )
    assert inference.control_gradient is not None
    assert inference.control_gradient.basis == bundle.gradient_basis


def test_elliptic_custom_gradient_sources_require_an_explicit_basis():
    telemetry = [{"curvature_radius": 2.0, "custom_vector": [1.0, -1.0]}]

    with pytest.raises(ValueError, match="gradient_basis is required"):
        elliptic_partial_from_telemetry(
            telemetry,
            gradient_source="custom_vector",
        )

    bundle = elliptic_partial_from_telemetry(
        telemetry,
        gradient_source="custom_vector",
        gradient_basis="example.custom_elliptic.v1",
    )
    assert bundle.gradient_basis == "example.custom_elliptic.v1"


def test_partial_fusion_fails_closed_on_ambiguous_or_invalid_requests():
    with pytest.raises(ValueError, match="both resolve to 'speed'"):
        zspace_partial_fusion([{"speed": 1.0, "velocity": 2.0}])
    with pytest.raises(ValueError, match="weights length"):
        zspace_partial_fusion([{"speed": 1.0}], weights=[])
    with pytest.raises(ValueError, match="unknown variant"):
        zspace_partial_fusion([{"speed": 1.0}], strategy="product")
    with pytest.raises(ValueError, match="unknown variant"):
        zspace_partial_fusion(
            [{"speed": 1.0}], gradient_alignment="truncate"
        )


def test_zspace_fusion_functions_are_public():
    assert st.zspace_partial_fusion is zspace_partial_fusion
    assert st.zspace_telemetry_fusion is zspace_telemetry_fusion
    assert "zspace_partial_fusion" in st.__all__
    assert "zspace_telemetry_fusion" in st.__all__


def test_zspace_posterior_functions_are_public():
    assert st.zspace_posterior_decode is zspace_posterior_decode
    assert st.zspace_posterior_project is zspace_posterior_project
    assert "zspace_posterior_decode" in st.__all__
    assert "zspace_posterior_project" in st.__all__
    assert st.ZSpaceControlGradient is ZSpaceControlGradient
    assert (
        st.ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
        == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    )


def test_z_notation_delegates_partial_semantics_to_rust():
    assert st.z.bundle({"velocity": 0.5}) == {"speed": 0.5}
    with pytest.raises(ValueError, match="both resolve to 'speed'"):
        st.z.partial({"speed": 1.0, "velocity": 2.0})
    with pytest.raises(ValueError, match="unknown Z-space metric"):
        st.z.bundle({"not_a_metric": 1.0})


def test_infer_with_partials_honours_last_strategy():
    vector = [0.4, -0.12, 0.33, -0.09]
    result = infer_with_partials(
        vector,
        {"speed": 0.15},
        {"speed": 0.9},
        strategy="last",
    )
    assert math.isclose(result.metrics["speed"], 0.9)


def test_infer_with_partials_preserves_post_fusion_projection_receipt():
    result = infer_with_partials(
        [0.3, -0.05, 0.21, 0.08],
        {
            "speed": 0.1,
            "memory": 0.2,
            "stability": 0.3,
            "frac": 0.4,
            "drs": -0.5,
            "gradient": [9.0],
        },
        metric_gradient_dimension=4,
    )

    assert result.fusion is not None
    assert result.fusion["gradient_source"] == "canonical_metrics"
    assert result.fusion["gradient"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert result.gradient_basis == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    assert result.control_gradient is not None
    assert result.control_gradient.basis == st.ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS
    assert result.control_gradient.values == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_infer_with_partials_merges_telemetry_payloads():
    vector = [0.3, -0.05, 0.21, 0.08]
    bundle = ZSpacePartialBundle({"speed": 0.4}, telemetry={"psi.mean": 0.5})
    result = infer_with_partials(
        vector,
        bundle,
        telemetry={"psi.offset": 0.2},
    )
    assert result.telemetry is not None
    assert result.fusion is not None
    assert result.fusion["semantic_owner"] == "st-core::telemetry::zspace_fusion"
    payload = result.telemetry.payload
    assert "psi.mean" in payload and "psi.offset" in payload


def test_pipeline_blends_and_clears_partials():
    vector = [0.22, -0.11, 0.31, -0.07]
    pipeline = ZSpaceInferencePipeline(vector, strategy="mean", smoothing=0.4)
    pipeline.add_partial({"speed": 0.5})
    pipeline.add_partial(ZSpacePartialBundle({"mem": -0.1}, weight=2.0))
    first = pipeline.infer(clear=False)
    assert math.isclose(first.metrics["speed"], 0.5, rel_tol=1e-6)
    assert math.isclose(first.metrics["memory"], -0.1, rel_tol=1e-6)
    assert first.fusion is not None
    assert first.fusion["gradient_source"] == "none"
    second = pipeline.infer()
    assert "speed" in second.metrics
    assert pipeline.posterior is not None


def test_pipeline_does_not_hide_invalid_fusion_overrides():
    pipeline = ZSpaceInferencePipeline(
        [0.22, -0.11, 0.31, -0.07], gradient_alignment="pad_zero"
    )
    pipeline.add_partial(
        {"speed": 0.2, "gradient": [0.1, -0.1]},
        gradient_basis="test.pipeline.control.v1",
    )
    pipeline.add_partial(
        ZSpacePartialBundle(
            {"speed": 0.6, "gradient": [0.5]},
            gradient_basis="test.pipeline.control.v1",
        )
    )

    assert pipeline.gradient_alignment == "pad_zero"
    pipeline.infer(clear=False)
    with pytest.raises(ValueError, match="unknown variant"):
        pipeline.infer(gradient_alignment="", clear=False)
    with pytest.raises(ValueError, match="unknown variant"):
        pipeline.infer(strategy="", clear=False)


def test_pipeline_can_queue_geometry_probe_context():
    probe = {
        "kind": "spiraltorch.wasm_fractal_field_probe",
        "source_crate": "st-frac::fractal_field",
        "mode": "branching_field",
        "generator": {"octaves": 3, "lacunarity": 2.0, "gain": 0.5, "iterations": 16},
        "log_lattice": {"log_start": -2.0, "log_step": 0.25, "len": 4},
        "sample_count": 4,
        "preview_count": 1,
        "energy": 0.4,
        "mean_abs": 0.3,
        "max_abs": 0.7,
        "phase_drift": 0.2,
        "total_variation": 0.15,
        "coherence_score": 0.86,
        "samples": [{"index": 0, "re": 0.1, "im": 0.0, "abs": 0.1}],
    }
    pipeline = ZSpaceInferencePipeline([0.22, -0.11, 0.31, -0.07])

    partials, metadata = pipeline.add_geometry_probes(
        {"field": probe},
        gradient_dim=4,
        include_consensus=True,
        consensus_only=True,
        return_metadata=True,
    )
    inference = pipeline.infer()

    assert len(partials) == 1
    assert metadata["source_origins"] == ["geometry:field"]
    assert metadata["context_origins"] == ["geometry:consensus"]
    assert metadata["consensus"]["gradient_preserved"] is True
    assert inference.control_gradient is not None
    assert inference.control_gradient.basis == partials[0].gradient_basis
    assert "speed" in inference.metrics
    assert inference.telemetry is not None
    assert inference.telemetry.payload["geometry.consensus.probe_count"] == pytest.approx(1.0)
    assert "geometry.fractal_field.1.energy" not in inference.telemetry.payload


def test_compile_inference_wraps_callable():
    calls: list[dict[str, float]] = []

    def collect(sample: Mapping[str, float]) -> Mapping[str, float]:
        calls.append(dict(sample))
        return sample

    compiled = compile_inference(collect)
    vector = [0.21, -0.04, 0.33, -0.12]
    partial = {"speed": 0.35}
    result = compiled(vector, partial)
    assert calls == [partial]
    assert math.isclose(result.metrics["speed"], 0.35)


def test_compile_inference_supports_decorator_usage():
    @compile_inference(alpha=0.5)
    def produce(speed: float, bias: float = 0.0) -> Mapping[str, float]:
        return {"speed": speed + bias}

    vector = [0.11, 0.02, -0.07, 0.19]
    output = produce(vector, 0.25, bias=0.1)
    assert math.isclose(output.metrics["speed"], 0.35)


def test_compile_inference_preserves_an_explicit_control_basis():
    @compile_inference(gradient_basis="test.compiled.control.v1")
    def produce() -> Mapping[str, Any]:
        return {"gradient": [0.2, -0.1]}

    output = produce([0.11, 0.02, -0.07, 0.19])

    assert output.control_gradient is not None
    assert output.control_gradient.values == pytest.approx([0.2, -0.1])
    assert output.control_gradient.basis == "test.compiled.control.v1"


def test_inference_mapping_preserves_the_latent_gradient_basis():
    inference = infer_from_partial([0.11, 0.02, -0.07, 0.19], {"speed": 0.3})

    payload = inference_to_mapping(inference)

    assert payload["gradient_basis"] == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    assert st.ZMetrics.from_mapping(payload).gradient_basis == (
        ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
    )

    remapped = inference_to_mapping(payload)
    assert remapped["gradient_basis"] == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS


def test_embedded_gradient_basis_round_trips_through_partial_projection():
    inference = infer_from_partial([0.12, -0.03, 0.48, -0.2], {"speed": 0.2})
    payload = inference_to_mapping(inference)

    projected = infer_from_partial([0.12, -0.03, 0.48, -0.2], payload)

    assert projected.control_gradient is not None
    assert projected.control_gradient.basis == ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS


def test_compile_inference_validates_return_type():
    @compile_inference
    def invalid(_: Mapping[str, float]):
        return 1

    vector = [0.2, -0.1, 0.05, -0.03]
    with pytest.raises(TypeError):
        invalid(vector, {"speed": 0.1})


def _matrix_stats(matrix: list[list[float]]) -> dict[str, float]:
    flat = [value for row in matrix for value in row]
    if not flat:
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0}
    l1 = sum(abs(value) for value in flat)
    l2 = math.sqrt(sum(value * value for value in flat))
    linf = max(abs(value) for value in flat)
    mean = sum(flat) / len(flat)
    return {"l1": l1, "l2": l2, "linf": linf, "mean": mean}


class _DummyCanvasSnapshot:
    def __init__(self) -> None:
        self.canvas = [[0.2, -0.1], [0.05, 0.3]]
        self.hypergrad = [[0.08, -0.04], [0.01, -0.02]]
        self.realgrad = [[0.03, -0.01], [0.02, 0.04]]
        self.summary = {
            "hypergrad": _matrix_stats(self.hypergrad),
            "realgrad": _matrix_stats(self.realgrad),
        }
        self.patch = [[0.12, -0.05], [0.02, 0.06]]


class _DummyCanvas:
    def __init__(self, snapshot: _DummyCanvasSnapshot) -> None:
        self._snapshot = snapshot
        self.snapshot_calls = 0

    def snapshot(self) -> _DummyCanvasSnapshot:
        self.snapshot_calls += 1
        return self._snapshot


def test_canvas_snapshot_inference_derives_augmented_metrics():
    snapshot = _DummyCanvasSnapshot()
    vector = [0.18, -0.07, 0.32, -0.11]
    partial = canvas_partial_from_snapshot(snapshot)
    assert {"canvas_energy", "hypergrad_norm", "realgrad_norm"}.issubset(partial)
    inference = infer_canvas_snapshot(vector, snapshot)
    assert "canvas_energy" in inference.metrics
    assert "hypergrad_norm" in inference.metrics
    assert math.isfinite(inference.metrics["canvas_energy"])


def test_infer_canvas_transformer_uses_snapshot_method():
    snapshot = _DummyCanvasSnapshot()
    canvas = _DummyCanvas(snapshot)
    vector = [0.11, 0.07, -0.05, 0.2]
    inference = infer_canvas_transformer(vector, canvas, smoothing=0.4)
    assert canvas.snapshot_calls == 1
    assert "canvas_mean" in inference.metrics
    assert math.isfinite(inference.metrics["canvas_mean"])


def test_canvas_coherence_partial_combines_sources():
    snapshot = _DummyCanvasSnapshot()
    diagnostics = _DummyDiagnostics()
    partial = canvas_coherence_partial(
        snapshot,
        diagnostics,
        coherence=[0.6, 0.25, 0.1, 0.05],
        weights=(2.0, 1.0),
    )
    assert "canvas_energy" in partial
    assert "coherence_mean" in partial


class _DummyDiagnostics:
    def __init__(self) -> None:
        self.mean_coherence = 0.25
        self.coherence_entropy = 1.0296530140645737
        self.energy_ratio = 0.7
        self.z_bias = -0.12
        self.fractional_order = 0.4
        self.normalized_weights = [0.5, 0.3, 0.2, 0.0]
        self.preserved_channels = 3
        self.discarded_channels = 1
        self.dominant_channel = 0


class _DummyContour:
    def coherence_strength(self) -> float:
        return 0.58

    def prosody_index(self) -> float:
        return 0.41

    def articulation_bias(self) -> float:
        return 0.19


class _DummySequencer:
    def __init__(self, diagnostics: _DummyDiagnostics, contour: _DummyContour) -> None:
        self._diagnostics = diagnostics
        self._contour = contour
        self.calls: list[Any] = []

    def forward_with_diagnostics(self, tensor: object):
        self.calls.append(tensor)
        return ("tensor", [0.62, 0.23, 0.1, 0.05], self._diagnostics)

    def emit_linguistic_contour(self, tensor: object) -> _DummyContour:
        self.calls.append(("contour", tensor))
        return self._contour


def test_coherence_diagnostics_projection_infers_metrics():
    diagnostics = _DummyDiagnostics()
    contract = zspace_coherence_project(
        diagnostics,
        coherence=[0.6, 0.25, 0.1, 0.05],
        contour=_DummyContour(),
    )
    assert contract["kind"] == "spiraltorch.zspace_coherence_projection"
    assert (
        contract["contract_version"]
        == "spiraltorch.zspace_coherence_projection.v2"
    )
    assert contract["semantic_owner"] == "st-core::inference::zspace_coherence"
    assert contract["semantic_backend"] == "rust"
    assert "speed=tanh(speed_gain*C)" in contract["projection_formula"]
    assert "coherence_entropy~=H(p)" in contract["evidence_validation_formula"]
    assert contract["derived"]["distribution_source"] == "normalized_weights"
    assert contract["derived"]["normalized_entropy"] == pytest.approx(
        diagnostics.coherence_entropy / math.log(4.0)
    )
    assert contract["classification"] == {
        "kind": "spiraltorch.zspace_coherence_classification",
        "contract_version": "spiraltorch.zspace_coherence_classification.v1",
        "semantic_owner": "st-core::inference::zspace_coherence",
        "semantic_backend": "rust",
        "classification_formula": contract["classification"]["classification_formula"],
        "label": "cascade_imbalance",
        "reason": "dominant_energy_ratio_at_or_above_cascade_min",
        "energy_ratio": pytest.approx(0.7),
        "swap_invariant": False,
        "policy": {
            "background_energy_ratio_max": pytest.approx(1.0e-5),
            "cascade_energy_ratio_min": pytest.approx(0.7),
        },
    }
    assert contract["classification"]["classification_formula"]
    assert contract["control"]["kind"] == "spiraltorch.zspace_coherence_control"
    assert (
        contract["control"]["contract_version"]
        == "spiraltorch.zspace_coherence_control.v1"
    )
    assert contract["control"]["semantic_backend"] == "rust"
    assert contract["control"]["spectral_radius"] == pytest.approx(
        contract["derived"]["concentration"]
    )
    assert contract["control"]["spectral_entropy"] == pytest.approx(
        contract["derived"]["normalized_entropy"]
    )
    assert contract["control"]["spectral_pressure"] == pytest.approx(
        diagnostics.energy_ratio * (1.0 - contract["derived"]["normalized_entropy"])
    )
    assert contract["control"]["control_formula"]
    custom = zspace_coherence_project(
        diagnostics,
        coherence=[0.6, 0.25, 0.1, 0.05],
        cascade_energy_ratio_min=0.8,
    )
    assert custom["classification"]["label"] == "diffuse_drift"
    assert custom["classification"]["policy"]["cascade_energy_ratio_min"] == (
        pytest.approx(0.8)
    )
    partial = coherence_partial_from_diagnostics(
        diagnostics, coherence=[0.6, 0.25, 0.1, 0.05]
    )
    assert math.isclose(partial["coherence_mean"], diagnostics.mean_coherence, rel_tol=1e-6)
    assert "coherence_weight_entropy" in partial
    inference = infer_coherence_diagnostics(
        [0.2, -0.05, 0.41, -0.09],
        diagnostics,
        coherence=[0.6, 0.25, 0.1, 0.05],
        contour=_DummyContour(),
    )
    assert "coherence_strength" in inference.metrics


def test_coherence_distribution_witness_is_built_and_validated_in_rust():
    witness = zspace_coherence_distribution_witness([0.6, 0.3, 0.1])
    assert witness == {
        "kind": "spiraltorch.zspace_coherence_distribution_witness",
        "contract_version": "spiraltorch.zspace_coherence_distribution_witness.v1",
        "semantic_owner": "st-core::inference::zspace_coherence",
        "semantic_backend": "rust",
        "witness_formula": witness["witness_formula"],
        "normalized_weights": pytest.approx([0.6, 0.3, 0.1]),
    }
    assert "complete simplex witness" in witness["witness_formula"]

    summary = validate_zspace_coherence_distribution_witness(witness)
    assert summary["channels"] == 3
    assert summary["weight_mass"] == pytest.approx(1.0)
    assert summary["effective_channels"] == pytest.approx(
        math.exp(summary["weight_entropy"])
    )

    wrong_version = dict(witness)
    wrong_version["contract_version"] = (
        "spiraltorch.zspace_coherence_distribution_witness.v0"
    )
    with pytest.raises(ValueError, match="contract_version"):
        validate_zspace_coherence_distribution_witness(wrong_version)

    invalid_simplex = dict(witness)
    invalid_simplex["normalized_weights"] = [0.7, 0.3, 0.1]
    with pytest.raises(ValueError, match="sum to one"):
        validate_zspace_coherence_distribution_witness(invalid_simplex)

    with pytest.raises(TypeError, match="must be a mapping"):
        validate_zspace_coherence_distribution_witness([0.6, 0.3, 0.1])


def test_coherence_projection_accepts_trace_entropy_alias_and_rejects_bad_gain():
    contract = zspace_coherence_project(
        {
            "mean_coherence": 0.7 / 3.0,
            "entropy": 0.3,
            "energy_ratio": 0.6,
            "z_bias": 0.1,
            "fractional_order": 0.5,
        },
        coherence=[0.1, 0.4, 0.2],
    )
    assert contract["partial"]["coherence_entropy"] == pytest.approx(0.3)
    assert contract["partial"]["coherence_response_peak"] == pytest.approx(0.4)
    assert "control" not in contract

    with pytest.raises(ValueError, match="speed_gain"):
        zspace_coherence_project(_DummyDiagnostics(), speed_gain=-1.0)
    with pytest.raises(ValueError, match="background energy-ratio maximum"):
        zspace_coherence_project(
            _DummyDiagnostics(),
            background_energy_ratio_max=0.8,
            cascade_energy_ratio_min=0.7,
        )
    with pytest.raises(ValueError, match="coherence.*3 channels.*4"):
        zspace_coherence_project(
            _DummyDiagnostics(),
            coherence=[0.6, 0.3, 0.1],
        )

    inconsistent_entropy = _DummyDiagnostics()
    inconsistent_entropy.coherence_entropy = 0.5
    with pytest.raises(ValueError, match="coherence_entropy.*inconsistent"):
        zspace_coherence_project(inconsistent_entropy)

    inconsistent_support = _DummyDiagnostics()
    inconsistent_support.preserved_channels = 4
    inconsistent_support.discarded_channels = 0
    with pytest.raises(ValueError, match="preserved_channels.*4.*observed 3"):
        zspace_coherence_project(inconsistent_support)

    inconsistent_dominant = _DummyDiagnostics()
    inconsistent_dominant.dominant_channel = 1
    with pytest.raises(ValueError, match="dominant coherence channel 1"):
        zspace_coherence_project(inconsistent_dominant)

    with pytest.raises(ValueError, match="mean_coherence.*inconsistent"):
        zspace_coherence_project(
            _DummyDiagnostics(),
            coherence=[0.8, 0.2, 0.1, 0.1],
        )


def test_infer_coherence_from_sequencer_can_return_outputs():
    diagnostics = _DummyDiagnostics()
    contour = _DummyContour()
    sequencer = _DummySequencer(diagnostics, contour)
    vector = [0.22, -0.15, 0.38, -0.04]
    sample = object()
    inference, outputs = infer_coherence_from_sequencer(
        vector,
        sequencer,
        sample,
        include_contour=True,
        return_outputs=True,
    )
    assert len(outputs) == 3
    assert ("contour", sample) in sequencer.calls
    assert "coherence_mean" in inference.metrics


def test_infer_canvas_with_coherence_projects_blended_partial():
    snapshot = _DummyCanvasSnapshot()
    diagnostics = _DummyDiagnostics()
    vector = [0.25, -0.08, 0.34, -0.12]
    inference = infer_canvas_with_coherence(
        vector,
        snapshot,
        diagnostics,
        coherence=[0.7, 0.15, 0.1, 0.05],
        strategy="mean",
    )
    assert "canvas_energy" in inference.metrics
    assert "coherence_mean" in inference.metrics


class _FakeDlpackWeights:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    def __dlpack__(self) -> list[float]:
        return list(self._data)


class _CompatCarrier:
    def __init__(self, data: list[float] | list[list[float]]) -> None:
        self._data = data


def test_weights_partial_from_dlpack_populates_import_metrics():
    weights = _FakeDlpackWeights([0.4, -0.2, 0.1, 0.05])
    bundle = weights_partial_from_dlpack(
        weights,
        bundle_weight=2.0,
        telemetry={"psi.extra": 0.3},
    )
    metrics = bundle.resolved()
    assert metrics["import_l2"] > 0.0
    telemetry_map = dict(bundle.telemetry_payload() or {})
    assert "psi.extra" in telemetry_map


def test_infer_weights_from_dlpack_exposes_telemetry():
    weights = _FakeDlpackWeights([0.2, -0.1, 0.3])
    result = infer_weights_from_dlpack(
        [0.1, -0.2, 0.05],
        weights,
        telemetry={"psi.offset": 0.15},
    )
    assert result.telemetry is not None
    assert "psi.offset" in result.telemetry.payload


def test_weights_partial_from_compat_uses_adapter(monkeypatch):
    carrier = _CompatCarrier([[0.25, -0.12], [0.06, 0.18]])
    monkeypatch.setattr(
        st.compat,
        "torch",
        types.SimpleNamespace(to_tensor=lambda value: value._data),
        raising=False,
    )
    bundle = weights_partial_from_compat(
        carrier,
        adapter="torch",
        telemetry={"psi.bridge": 0.4},
    )
    metrics = bundle.resolved()
    assert metrics["import_count"] == 4.0
    telemetry_map = dict(bundle.telemetry_payload() or {})
    assert "psi.bridge" in telemetry_map


def test_infer_weights_from_compat_merges_telemetry(monkeypatch):
    carrier = _CompatCarrier([0.3, -0.18, 0.07, 0.22])
    monkeypatch.setattr(
        st.compat,
        "torch",
        types.SimpleNamespace(to_tensor=lambda value: value._data),
        raising=False,
    )
    result = infer_weights_from_compat(
        [0.2, -0.05, 0.19, -0.08],
        carrier,
        adapter="torch",
        telemetry={"psi.hint": 0.6},
    )
    assert result.telemetry is not None
    assert "psi.hint" in result.telemetry.payload
