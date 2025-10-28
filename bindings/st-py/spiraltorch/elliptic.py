"""Autograd-aware helpers around the Rust-backed elliptic warp."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - torch optional during build
    import torch
except Exception:  # pragma: no cover - degrade gracefully without torch
    torch = None  # type: ignore

try:
    from . import EllipticTelemetry as _EllipticTelemetry
    from . import EllipticWarp as _EllipticWarp
except Exception:  # pragma: no cover - module import failure propagates later
    _EllipticTelemetry = None  # type: ignore
    _EllipticWarp = None  # type: ignore

__all__ = [
    "EllipticWarpFunction",
    "elliptic_warp_autograd",
    "elliptic_warp_features",
    "elliptic_warp_partial",
]


_TORCH_ERROR_MESSAGE = (
    "PyTorch is required for elliptic autograd support; install the 'torch' package to"
    " enable this functionality."
)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(_TORCH_ERROR_MESSAGE)
    if _EllipticWarp is None:
        raise RuntimeError("Rust elliptic warp bindings are unavailable")


def _reshape(values: List[Any], shape: Sequence[int]) -> Any:
    if not shape:
        return values[0]
    if len(shape) == 1:
        step = 1
        return values[: shape[0]]
    step = int(len(values) / shape[0]) if shape[0] else 0
    return [
        _reshape(values[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])
    ]


if torch is None:

    class EllipticWarpFunction:  # type: ignore[misc,too-many-ancestors]
        """Placeholder that raises a clear error when PyTorch is unavailable."""

        _last_telemetry: Optional[List[Optional[_EllipticTelemetry]]] = None
        _last_shape: Optional[Tuple[int, ...]] = None

        @staticmethod
        def apply(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(_TORCH_ERROR_MESSAGE)

        @staticmethod
        def forward(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(_TORCH_ERROR_MESSAGE)

        @staticmethod
        def backward(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(_TORCH_ERROR_MESSAGE)

        @classmethod
        def last_telemetry(cls, *, as_dict: bool = False) -> Optional[Any]:
            raise RuntimeError(_TORCH_ERROR_MESSAGE)

else:

    class EllipticWarpFunction(torch.autograd.Function):  # type: ignore[misc]
        """torch.autograd.Function that exposes the elliptic warp features."""

        _last_telemetry: Optional[List[Optional[_EllipticTelemetry]]] = None
        _last_shape: Optional[Tuple[int, ...]] = None

        @staticmethod
        def forward(  # type: ignore[override]
            ctx: torch.autograd.FunctionCtx, warp: "_EllipticWarp", orientation: "torch.Tensor"
        ) -> "torch.Tensor":
            _require_torch()
            if not isinstance(warp, _EllipticWarp):  # pragma: no cover - defensive guard
                raise TypeError("warp must be a spiraltorch.EllipticWarp instance")
            if orientation.size(-1) != 3:
                raise ValueError("orientation tensor must have last dimension == 3")

            device = orientation.device
            dtype = orientation.dtype
            flat = orientation.reshape(-1, orientation.size(-1))
            features: List[torch.Tensor] = []
            jacobian: List[torch.Tensor] = []
            telemetry: List[Optional[_EllipticTelemetry]] = []

            for row in flat:
                vec = row.detach().to(torch.float32).cpu().tolist()
                result = warp.map_orientation_differential(vec)
                if result is None:
                    features.append(torch.zeros(9, device=device, dtype=dtype))
                    jacobian.append(torch.zeros((9, row.numel()), device=device, dtype=dtype))
                    telemetry.append(None)
                    continue
                tele, feats, jac = result
                features.append(torch.tensor(feats, device=device, dtype=dtype))
                jacobian.append(torch.tensor(jac, device=device, dtype=dtype))
                telemetry.append(tele)

            feature_tensor = torch.stack(features)
            jac_tensor = torch.stack(jacobian)
            ctx.save_for_backward(jac_tensor)
            ctx._input_shape = orientation.shape  # type: ignore[attr-defined]
            EllipticWarpFunction._last_shape = orientation.shape[:-1]
            EllipticWarpFunction._last_telemetry = telemetry
            return feature_tensor.reshape(*orientation.shape[:-1], -1)

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: torch.autograd.FunctionCtx, grad_output: "torch.Tensor"
        ) -> Tuple[None, "torch.Tensor"]:
            (jac_tensor,) = ctx.saved_tensors
            grad = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.bmm(grad.unsqueeze(1), jac_tensor).squeeze(1)
            input_shape = getattr(ctx, "_input_shape")
            grad_input = grad_input.reshape(*input_shape)
            return None, grad_input

        @classmethod
        def last_telemetry(cls, *, as_dict: bool = False) -> Optional[Any]:
            data = cls._last_telemetry
            if data is None:
                return None
            shape = cls._last_shape or ()
            if as_dict:
                converted = [tele.as_dict() if tele is not None else None for tele in data]
            else:
                converted = data
            leading = list(shape)
            if not leading:
                return converted[0] if converted else None
            return _reshape(converted, leading)


def elliptic_warp_autograd(
    warp: "_EllipticWarp", orientation: "torch.Tensor", *, return_telemetry: bool = False
) -> Any:
    """Apply the elliptic warp with autograd support.

    Args:
        warp: Rust-backed :class:`EllipticWarp` instance.
        orientation: Tensor whose final dimension enumerates the 3D orientation.
        return_telemetry: When True, also return the rich telemetry objects generated by the
            forward pass. The telemetry mirrors the leading batch dimensions.
    """

    _require_torch()
    features = EllipticWarpFunction.apply(warp, orientation)
    if not return_telemetry:
        return features
    telemetry = EllipticWarpFunction.last_telemetry(as_dict=False)
    return features, telemetry


def elliptic_warp_features(
    warp: "_EllipticWarp", orientations: Iterable[Sequence[float]], *, as_dict: bool = True
) -> List[Tuple[List[float], Optional[Any]]]:
    """Compute elliptic features and telemetry for a list of orientations."""

    if _EllipticWarp is None:
        raise RuntimeError("Rust elliptic warp bindings are unavailable")
    results: List[Tuple[List[float], Optional[Any]]] = []
    for orientation in orientations:
        result = warp.map_orientation_differential(list(orientation))
        if result is None:
            results.append(([0.0] * 9, None))
            continue
        telemetry, features, _jac = result
        results.append((features, telemetry.as_dict() if as_dict else telemetry))
    return results


def elliptic_warp_partial(
    warp: "_EllipticWarp",
    orientation: "torch.Tensor",
    *,
    bundle_weight: float = 1.0,
    origin: str | None = "elliptic",
    telemetry_prefix: str = "elliptic",
    aggregate: str = "mean",
    gradient_source: str = "rotor_transport",
    extra_telemetry: Mapping[str, Any] | None = None,
    return_features: bool = False,
):
    """Run the elliptic warp and convert its telemetry into a Z-space bundle.

    This helper stitches the autograd-enabled warp with the Z-space inference
    pipeline by translating the generated telemetry into a
    :class:`~spiraltorch.zspace_inference.ZSpacePartialBundle`.

    Args:
        warp: Rust-backed :class:`EllipticWarp` instance.
        orientation: Orientation tensor provided to :func:`elliptic_warp_autograd`.
        bundle_weight: Weight assigned to the resulting partial bundle.
        origin: Optional origin label recorded on the partial bundle.
        telemetry_prefix: Prefix applied to flattened telemetry payload keys.
        aggregate: Reduction strategy (``"mean"``, ``"max"``, ``"min"`` or
            ``"last"``) applied when multiple telemetry samples are provided.
        gradient_source: Telemetry vector used to seed the gradient channel;
            defaults to ``rotor_transport``.
        extra_telemetry: Additional telemetry mapping merged into the bundle.
        return_features: When ``True``, also return the raw feature tensor.

    Returns:
        Either the constructed :class:`ZSpacePartialBundle` or a tuple of the
        feature tensor and bundle when ``return_features`` is ``True``.
    """

    _require_torch()
    features, telemetry = elliptic_warp_autograd(warp, orientation, return_telemetry=True)
    from .zspace_inference import elliptic_partial_from_telemetry

    bundle = elliptic_partial_from_telemetry(
        telemetry,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix,
        aggregate=aggregate,
        gradient_source=gradient_source,
        extra_telemetry=extra_telemetry,
    )
    if return_features:
        return features, bundle
    return bundle
