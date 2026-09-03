"""Thin Python client for Rust-owned Z-space dynamical protocols."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Optional


def _native_constant(name: str, fallback: object) -> object:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    return getattr(native, name, fallback)


ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION",
        "spiraltorch.zspace_stochastic_schrodinger_forward.v1",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND",
        "spiraltorch.zspace_stochastic_schrodinger_forward",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION",
        "spiraltorch.zspace_stochastic_schrodinger_vjp.v1",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND",
        "spiraltorch.zspace_stochastic_schrodinger_vjp",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER",
        "st-core::runtime::zspace_stochastic_schrodinger",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER",
        "st-core::dynamics::stochastic_schrodinger",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND = str(
    _native_constant("ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND", "rust")
)
ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE",
        "sha256(contract_version UTF-8 || NUL || compact canonical request JSON)",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS",
        "vector-Jacobian product of output_real with respect to input and potential "
        "only; standard_normal and config are fixed witnesses; phase is recomputed "
        "from the canonical forward request and is never accepted as external evidence",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY = str(
    _native_constant(
        "ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY",
        "this receipt certifies one bounded real-quadrature numerical transition "
        "and its analytic input/potential VJP; it does not establish physical "
        "fidelity beyond the stated equation, semantic quality, or training efficacy",
    )
)
ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES = int(
    _native_constant("ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES", 262_144)
)
ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS = int(
    _native_constant("ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS", 262_144)
)
ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES = int(
    _native_constant("ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES", 65_536)
)

_COMPLEX_VERSION = str(_native_constant(
    "ZSPACE_SCHRODINGER_COMPLEX_CONTRACT_VERSION",
    "spiraltorch.zspace_stochastic_schrodinger_complex_step.v1",
))
_COMPLEX_KIND = str(_native_constant(
    "ZSPACE_SCHRODINGER_COMPLEX_KIND",
    "spiraltorch.zspace_stochastic_schrodinger_complex_step",
))
_COMPLEX_GRADIENT_SEMANTICS = str(_native_constant(
    "ZSPACE_SCHRODINGER_COMPLEX_GRADIENT_SEMANTICS",
    "real Euclidean VJP of output_real and output_imaginary with respect to "
    "input, input_imaginary, and shared potential; config and standard_normal "
    "are fixed witnesses",
))
_COMPLEX_EVIDENCE_BOUNDARY = str(_native_constant(
    "ZSPACE_SCHRODINGER_COMPLEX_EVIDENCE_BOUNDARY",
    "one complex Strang-split transition and optional analytic input/potential "
    "VJP; preserving both quadratures permits composition but does not establish "
    "semantic quality or training efficacy",
))

__all__ = [
    "zspace_stochastic_schrodinger_complex_step",
    "validate_zspace_stochastic_schrodinger_complex",
    "ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY",
    "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION",
    "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND",
    "ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE",
    "ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES",
    "ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS",
    "ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES",
    "ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER",
    "ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND",
    "ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER",
    "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION",
    "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND",
    "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS",
    "validate_zspace_stochastic_schrodinger_forward",
    "validate_zspace_stochastic_schrodinger_vjp",
    "zspace_stochastic_schrodinger_forward",
    "zspace_stochastic_schrodinger_vjp",
]


_MAX_NATIVE_ROOT_FIELDS = 32
_MAX_CONFIG_FIELDS = 8


def _bounded_mapping_snapshot(
    value: dict[str, object],
    *,
    maximum: int,
    label: str,
) -> dict[str, object]:
    try:
        field_count = dict.__len__(value)
    except TypeError:
        raise TypeError(
            f"{label} must be a dict-backed mapping for bounded admission"
        ) from None
    if field_count > maximum:
        raise ValueError(f"{label} field count exceeds maximum {maximum}")
    return dict.copy(value)


def _bounded_vector(
    values: list[float] | tuple[float, ...],
    *,
    maximum: int,
    label: str,
) -> list[float]:
    try:
        value_count = list.__len__(values)
        iterator = list.__iter__(values)
    except TypeError:
        try:
            value_count = tuple.__len__(values)
            iterator = tuple.__iter__(values)
        except TypeError:
            raise TypeError(
                f"{label} must be a list or tuple for bounded admission"
            ) from None
    if value_count > maximum:
        raise ValueError(f"{label} length exceeds maximum {maximum}")
    return list(iterator)


def _native_operation(name: str, payload: dict[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space stochastic Schrodinger dynamics require the compiled Rust "
            f"semantic core; rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(
        _bounded_mapping_snapshot(
            payload,
            maximum=_MAX_NATIVE_ROOT_FIELDS,
            label=f"native {name} payload",
        )
    )
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_receipt(result)
    return result


def _content_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
    )


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    kind = receipt.get("kind")
    evidence_boundary = ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY
    if kind == ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND:
        validated = receipt.get("forward_validated") is True
        contract_version = ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION
        identity = receipt.get("forward_id")
    elif kind == ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND:
        validated = receipt.get("vjp_validated") is True
        contract_version = ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION
        identity = receipt.get("vjp_id")
        if (
            receipt.get("gradient_semantics")
            != ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS
            or receipt.get("output_observable") != "real_quadrature"
            or not _content_id(receipt.get("forward_id"))
        ):
            raise RuntimeError(
                "native Z-space core returned an untrusted stochastic Schrodinger VJP"
            )
    elif kind == _COMPLEX_KIND:
        contract_version = _COMPLEX_VERSION
        identity = receipt.get("evaluation_id")
        evidence_boundary = _COMPLEX_EVIDENCE_BOUNDARY
        validated = receipt.get("gradient_semantics") == _COMPLEX_GRADIENT_SEMANTICS
    else:
        raise RuntimeError(
            "native Z-space core returned an unknown stochastic Schrodinger receipt"
        )
    if (
        receipt.get("contract_version") != contract_version
        or receipt.get("semantic_owner")
        != ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER
        or receipt.get("semantic_backend")
        != ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND
        or receipt.get("protocol_owner")
        != ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER
        or receipt.get("id_rule") != ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE
        or receipt.get("status") != "ready"
        or receipt.get("efficacy_claim_ready") is not False
        or receipt.get("evidence_boundary")
        != evidence_boundary
        or not validated
        or not _content_id(identity)
        or not isinstance(receipt.get("request"), Mapping)
    ):
        raise RuntimeError(
            "native Z-space core returned an untrusted stochastic Schrodinger receipt"
        )


def zspace_stochastic_schrodinger_forward(
    input_values: list[float] | tuple[float, ...],
    potential: list[float] | tuple[float, ...],
    *,
    rows: Optional[int] = None,
    features: Optional[int] = None,
    standard_normal: Optional[list[float] | tuple[float, ...]] = None,
    config: Optional[dict[str, object]] = None,
) -> dict[str, Any]:
    """Run one deterministic, audited Rust-owned real-time dynamics step.

    ``standard_normal`` is the explicit noise witness. Omitting it records a
    zero witness rather than invoking a hidden Python or Rust RNG.
    """

    input_vector = _bounded_vector(
        input_values,
        maximum=ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        label="Z-space stochastic Schrodinger input",
    )
    potential_vector = _bounded_vector(
        potential,
        maximum=ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES,
        label="Z-space stochastic Schrodinger potential",
    )
    resolved_features = len(potential_vector) if features is None else features
    if rows is None:
        if not isinstance(resolved_features, int) or isinstance(
            resolved_features, bool
        ):
            raise TypeError("features must be an integer")
        if resolved_features <= 0:
            raise ValueError("features must be positive when rows are inferred")
        if len(input_vector) % resolved_features:
            raise ValueError(
                "input length must be divisible by features when rows are inferred"
            )
        resolved_rows = len(input_vector) // resolved_features
    else:
        resolved_rows = rows
    if standard_normal is None:
        noise_vector = [0.0] * len(input_vector)
    else:
        noise_vector = _bounded_vector(
            standard_normal,
            maximum=ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
            label="Z-space stochastic Schrodinger standard_normal",
        )
    config_payload = (
        {}
        if config is None
        else _bounded_mapping_snapshot(
            config,
            maximum=_MAX_CONFIG_FIELDS,
            label="Z-space stochastic Schrodinger config",
        )
    )
    return _native_operation(
        "_zspace_stochastic_schrodinger_forward",
        {
            "input": input_vector,
            "potential": potential_vector,
            "standard_normal": noise_vector,
            "rows": resolved_rows,
            "features": resolved_features,
            "config": config_payload,
        },
    )


def validate_zspace_stochastic_schrodinger_forward(
    receipt: dict[str, object],
) -> dict[str, Any]:
    """Recompute a serialized forward receipt in Rust and reject any drift."""

    return _native_operation(
        "_zspace_stochastic_schrodinger_forward_validate", receipt
    )


def zspace_stochastic_schrodinger_vjp(
    forward_receipt: dict[str, object],
    grad_output_real: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Compute the analytic VJP after Rust revalidates the complete forward."""

    validated_forward = validate_zspace_stochastic_schrodinger_forward(
        forward_receipt
    )
    request = validated_forward.get("request")
    if not isinstance(request, dict):
        raise RuntimeError("validated stochastic Schrodinger forward lost its request")
    gradient = _bounded_vector(
        grad_output_real,
        maximum=ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        label="Z-space stochastic Schrodinger grad_output_real",
    )
    return _native_operation(
        "_zspace_stochastic_schrodinger_vjp",
        {"forward_request": request, "grad_output_real": gradient},
    )


def validate_zspace_stochastic_schrodinger_vjp(
    receipt: dict[str, object],
) -> dict[str, Any]:
    """Recompute a serialized VJP receipt in Rust and reject any drift."""

    return _native_operation("_zspace_stochastic_schrodinger_vjp_validate", receipt)


def zspace_stochastic_schrodinger_complex_step(
    request: dict[str, object],
) -> dict[str, Any]:
    """Evolve both quadratures using the same owned request as Rust/WASM.

    ``forward_request`` contains the real input, potential, shape, explicit
    Gaussian witness and optional config; ``input_imaginary`` is mandatory.
    Optional ``cotangent={"real": [...], "imaginary": [...]}`` requests the
    real Euclidean VJP of both outputs, with config/noise held fixed.
    Keep both output arrays when constructing a subsequent step.
    """
    return _native_operation("_zspace_stochastic_schrodinger_complex_step", request)


def validate_zspace_stochastic_schrodinger_complex(
    receipt: dict[str, object],
) -> dict[str, Any]:
    """Replay a complex-state step, including optional gradients, in Rust."""
    return _native_operation("_zspace_stochastic_schrodinger_complex_validate", receipt)
