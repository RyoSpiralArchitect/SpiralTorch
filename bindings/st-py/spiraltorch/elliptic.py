"""Autograd-aware helpers around the Rust-backed elliptic warp."""

from __future__ import annotations

from contextvars import ContextVar
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
_LAST_TELEMETRY = ContextVar("spiraltorch_elliptic_telemetry", default=None)


def _require_native() -> None:
    global _EllipticWarp, _EllipticTelemetry
    if _EllipticWarp is None:
        # The facade imports helpers before all native classes are registered.
        try:
            from . import EllipticWarp as _EllipticWarp
            from . import EllipticTelemetry as _EllipticTelemetry
        except (ImportError, AttributeError) as error:
            raise RuntimeError("Rust elliptic warp bindings are unavailable") from error


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(_TORCH_ERROR_MESSAGE)
    _require_native()


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
            ctx: torch.autograd.FunctionCtx,
            warp: "_EllipticWarp",
            orientation: "torch.Tensor",
        ) -> "torch.Tensor":
            _require_torch()
            if not isinstance(
                warp, _EllipticWarp
            ):  # pragma: no cover - defensive guard
                raise TypeError("warp must be a spiraltorch.EllipticWarp instance")
            if (
                not isinstance(orientation, torch.Tensor)
                or orientation.dtype != torch.float32
            ):
                raise TypeError("elliptic autograd requires float32 tensors")
            if orientation.ndim < 1 or orientation.size(-1) != 3:
                raise ValueError("orientation tensor must have last dimension == 3")
            if orientation.numel() // 3 > 65_536:
                raise ValueError("elliptic autograd exceeds the 65536-row budget")

            device = orientation.device
            dtype = orientation.dtype
            batch = warp.map_orientations_batch(
                orientation.detach().cpu().reshape(-1).tolist()
            )
            feature_tensor = torch.tensor(batch.features, device=device, dtype=dtype)
            ctx.save_for_backward(orientation)
            ctx.save_for_forward(orientation)
            ctx.batch = batch
            _LAST_TELEMETRY.set((orientation.shape[:-1], batch.telemetry()))
            return feature_tensor.reshape(*orientation.shape[:-1], 9)

        @staticmethod
        def jvp(ctx, _warp, tangent):
            (orientation,) = ctx.saved_tensors
            if tangent is None:
                tangent = torch.zeros_like(orientation)
            values = ctx.batch.jvp(tangent.detach().cpu().reshape(-1).tolist())
            return torch.tensor(
                values, device=orientation.device, dtype=orientation.dtype
            ).reshape(*orientation.shape[:-1], 9)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(  # type: ignore[override]
            ctx: torch.autograd.FunctionCtx, grad_output: "torch.Tensor"
        ) -> Tuple[None, "torch.Tensor"]:
            (orientation,) = ctx.saved_tensors
            values = ctx.batch.vjp(grad_output.detach().cpu().reshape(-1).tolist())
            grad_input = torch.tensor(
                values, device=orientation.device, dtype=orientation.dtype
            ).reshape(orientation.shape)
            return None, grad_input

        @classmethod
        def last_telemetry(cls, *, as_dict: bool = False) -> Optional[Any]:
            state = _LAST_TELEMETRY.get()
            if state is None:
                return None
            shape, data = state
            if as_dict:
                converted = [
                    tele.as_dict() if tele is not None else None for tele in data
                ]
            else:
                converted = data
            leading = list(shape)
            if not leading:
                return converted[0] if converted else None
            return _reshape(converted, leading)


def elliptic_warp_autograd(
    warp: "_EllipticWarp",
    orientation: "torch.Tensor",
    *,
    return_telemetry: bool = False,
) -> Any:
    """Apply the Rust batched f32 map and first-order VJP/JVP (CPU transport).

    Degenerate rows, chart poles and the azimuth cut raise rather than silently
    replacing features/gradients by zero. ``torch.autograd.forward_ad`` uses the
    native JVP. Higher-order gradients and ``torch.func`` transforms are unsupported.

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
    warp: "_EllipticWarp",
    orientations: Iterable[Sequence[float]],
    *,
    as_dict: bool = True,
) -> List[Tuple[List[float], Optional[Any]]]:
    """Compute elliptic features and telemetry for a list of orientations."""

    _require_native()
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
    gradient_alignment: str = "strict",
    gradient_source: str = "rotor_transport",
    gradient_basis: str | None = None,
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
        aggregate: Rust-owned reduction strategy applied when multiple
            telemetry samples are provided.
        gradient_alignment: ``"strict"`` rejects ragged gradient vectors;
            ``"pad_zero"`` explicitly enables legacy zero padding.
        gradient_source: Telemetry vector used to seed the gradient channel;
            defaults to ``rotor_transport``.
        gradient_basis: Explicit coordinate identity for custom gradient
            sources. Known elliptic vectors use their canonical basis.
        extra_telemetry: Additional telemetry mapping merged into the bundle.
        return_features: When ``True``, also return the raw feature tensor.

    Returns:
        Either the constructed :class:`ZSpacePartialBundle` or a tuple of the
        feature tensor and bundle when ``return_features`` is ``True``.
    """

    _require_torch()
    features, telemetry = elliptic_warp_autograd(
        warp, orientation, return_telemetry=True
    )
    from .zspace_inference import elliptic_partial_from_telemetry

    bundle = elliptic_partial_from_telemetry(
        telemetry,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix,
        aggregate=aggregate,
        gradient_alignment=gradient_alignment,
        gradient_source=gradient_source,
        gradient_basis=gradient_basis,
        extra_telemetry=extra_telemetry,
    )
    if return_features:
        return features, bundle
    return bundle
