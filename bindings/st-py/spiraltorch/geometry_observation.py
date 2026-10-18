"""Bounded scalar observations of actual adapter calls; no training policy."""

from copy import deepcopy

from ._geometry_conditioning_capture import capture_native_conditioning
from ._torch_transport import torch


class GeometryObservation:
    """Created by GeometryAdapterStack.observe; reports never retain tensors.

    Native conditioning is currently provided by WaveGate. Generic residual
    statistics measure actual float32 input/output differences in CPU float64.
    They are observations, not replacement forwards, VJPs or quality scores.
    """

    def __init__(self, *, max_events=128, max_values=1_048_576):
        if torch is None:
            raise RuntimeError("PyTorch is required for geometric observations")
        for name, value, maximum in (
            ("max_events", max_events, 4096),
            ("max_values", max_values, 1_048_576),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        self._max_events = max_events
        self._max_values = max_values
        self._calls = 0
        self._omitted = 0
        self._events = []
        self._status = "not_started"

    def to_dict(self):
        """Return an owned JSON-compatible snapshot, including coverage gaps."""
        return {
            "schema": "spiraltorch.geometry_observation.v1",
            "status": self._status,
            "max_events": self._max_events,
            "max_values": self._max_values,
            "calls_seen": self._calls,
            "events_omitted": self._omitted,
            "coverage_complete": self._status == "completed"
            and self._omitted == 0
            and all(
                e["status"] == "observed" and e["native_reports_omitted"] == 0
                for e in self._events
            ),
            "events": deepcopy(self._events),
        }

    @staticmethod
    def _residual_stats(before, after):
        after = after.detach().to(device="cpu").to(dtype=torch.float64)
        finite = bool(torch.isfinite(before).all() and torch.isfinite(after).all())
        result = {
            "finite": finite,
            "input_l2": None,
            "output_l2": None,
            "residual_l2": None,
            "relative_residual_l2": None,
            "residual_max_abs": None,
        }
        if finite:
            delta = after - before
            source = torch.linalg.vector_norm(before).item()
            residual = torch.linalg.vector_norm(delta).item()
            result.update(
                input_l2=source,
                output_l2=torch.linalg.vector_norm(after).item(),
                residual_l2=residual,
                relative_residual_l2=residual / source if source else None,
                residual_max_abs=delta.abs().max().item() if delta.numel() else 0.0,
            )
        return result

    def _call(self, path, adapter, value, invoke):
        self._calls += 1
        if self._calls > self._max_events:
            self._omitted += 1
            return invoke()
        event = {
            "ordinal": self._calls,
            "path": path,
            "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
            "shape": list(value.shape),
            "device": str(value.device),
            "training": adapter.training,
            "grad_enabled": torch.is_grad_enabled(),
            "status": "pending",
            "native_reports": [],
            "native_reports_omitted": 0,
        }
        self._events.append(event)
        try:
            if value.numel() > self._max_values:
                result = invoke()
                event["status"] = "skipped_value_budget"
                return result
            # Copy before the call so in-place custom adapters cannot rewrite
            # the observation. Only scalars survive this invocation.
            before = value.detach().to(device="cpu").to(dtype=torch.float64)
            with capture_native_conditioning() as native:
                result = invoke()
            event.update(
                status="observed",
                native_reports=native.reports,
                native_reports_omitted=native.omitted,
                residual=self._residual_stats(before, result),
            )
            return result
        except BaseException as error:
            event.update(status="failed", error_type=type(error).__name__)
            raise
