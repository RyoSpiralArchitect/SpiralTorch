"""Explicit model placement for geometric adapters, never replacement mathematics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import inspect
from typing import Any
from weakref import WeakKeyDictionary

from .geometry_autograd import _input, _require_torch, torch

__all__ = ["GeometryAdapterStack"]
_SCHEMA = "spiraltorch.geometry_adapter_stack.v1"
_ACTIVE_TARGETS: Any = WeakKeyDictionary()


if torch is None:
    class GeometryAdapterStack:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
else:
    class GeometryAdapterStack(torch.nn.Module):
        """Own adapters independently of a model; attach them within one scope.

        Mapping order fixes optimizer parameter order and is checkpointed with
        exact module paths and adapter types. No model weights, module names,
        train/eval state, devices or requires_grad flags are changed. Optimize
        this container's parameters explicitly, not model.parameters().

        Targets return shape-preserving float32 tensors. Native adapters keep
        their existing CPU/host-transfer boundary and full-prefix requirements.
        Autograd must finish inside the attachment scope. Cached generation,
        activation checkpointing, concurrent use, compile and distributed model
        wrappers are not supported by this initial placement API.
        """

        def __init__(self, placements: Mapping[str, Any]) -> None:
            super().__init__()
            if not isinstance(placements, Mapping) or not placements:
                raise ValueError("placements must be a nonempty path-to-adapter mapping")
            paths = tuple(placements)
            if any(not isinstance(path, str) or not path or not all(path.split(".")) for path in paths):
                raise ValueError("placements need nonempty dotted module paths")
            if any(a.startswith(b + ".") for a in paths for b in paths if a != b):
                raise ValueError("nested insertion paths are ambiguous")
            if any(not isinstance(module, torch.nn.Module) for module in placements.values()):
                raise TypeError("every adapter must be a torch.nn.Module")
            self._paths = paths
            self.adapters = torch.nn.ModuleList(placements.values())
            self._active_token: Any = None
            self._validate_ownership()

        @property
        def paths(self) -> tuple[str, ...]:
            return self._paths

        def _validate_ownership(self) -> tuple[set[int], set[int]]:
            if len(self.adapters) != len(self.paths):
                raise ValueError("adapter count no longer matches placements")
            modules: set[int] = set()
            values: set[int] = set()
            for adapter in self.adapters:
                owned_modules = {id(module) for module in adapter.modules()}
                owned_values = {id(value) for value in (*adapter.parameters(), *adapter.buffers())}
                if modules & owned_modules or values & owned_values:
                    raise ValueError("insertion sites must not share adapter modules or parameters")
                modules.update(owned_modules)
                values.update(owned_values)
            return modules, values

        def get_extra_state(self) -> dict[str, Any]:
            self._validate_ownership()
            return {"schema": _SCHEMA, "paths": list(self.paths),
                    "adapter_types": [f"{type(a).__module__}.{type(a).__qualname__}" for a in self.adapters]}

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if state != self.get_extra_state():
                raise ValueError("incompatible geometry adapter placements or types")

        def _load_from_state_dict(self, state_dict: Any, prefix: str, local_metadata: Any,
                                  strict: bool, missing_keys: Any, unexpected_keys: Any,
                                  error_msgs: Any) -> None:
            # Validate routing before Torch descends into any adapter parameters,
            # including when this stack is loaded as a child of another module.
            if self._active_token is not None:
                raise RuntimeError("detach geometry adapters before loading a checkpoint")
            self.set_extra_state(state_dict.get(prefix + "_extra_state"))
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                         missing_keys, unexpected_keys, error_msgs)

        @staticmethod
        def _check_model_mode(model: Any) -> None:
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                raise ValueError("attach geometry to an unwrapped model; parallel wrappers are unsupported")
            if getattr(model, "is_gradient_checkpointing", False):
                raise ValueError("geometry attachment does not support activation checkpointing")
            if getattr(getattr(model, "config", None), "use_cache", False):
                raise ValueError("set model.config.use_cache=False for complete-prefix geometry")

        def _check_call(self, model: Any, args: Any, kwargs: Any) -> None:
            self._check_model_mode(model)
            arguments = dict(kwargs)
            bound = inspect.signature(model.forward).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arguments.update(bound.arguments)
            cache = arguments.get("use_cache")
            if (cache is not None and cache is not False) or any(
                arguments.get(key) is not None for key in ("past_key_values", "past_key_value", "cache_params")
            ):
                raise ValueError("cached model calls are not supported by geometry attachment")

        def _hook(self, index: int, token: Any) -> Any:
            adapter = self.adapters[index]

            def apply(_module: Any, _args: Any, value: Any) -> Any:
                if (self._active_token is not token or len(self.adapters) != len(self.paths)
                        or self.adapters[index] is not adapter):
                    raise RuntimeError("geometry attachment changed during execution")
                _input(value)
                result = adapter(value)
                if (not isinstance(result, torch.Tensor) or result.shape != value.shape
                        or result.dtype != value.dtype or result.device != value.device):
                    raise ValueError("geometry adapters must preserve tensor shape, dtype and device")
                if result.requires_grad:
                    def in_scope(gradient: Any) -> Any:
                        if self._active_token is not token:
                            raise RuntimeError("finish backward before detaching geometry adapters")
                        return gradient
                    result.register_hook(in_scope)
                return result

            return apply

        @contextmanager
        def attach(self, model: Any) -> Iterator[GeometryAdapterStack]:
            """Install forward hooks and remove only these hooks on every exit.

            Target module call signatures, including keyword arguments, are
            unchanged. Existing user hooks remain installed. Tensor-valued
            targets must be chosen explicitly; tuple/dict extraction, padding,
            packed-document boundaries and model/data identity are caller-owned.
            """
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must be a torch.nn.Module")
            if self._active_token is not None:
                raise RuntimeError("this geometry stack is already attached")
            self._check_model_mode(model)
            adapter_modules, adapter_values = self._validate_ownership()
            named = list(model.named_modules(remove_duplicate=False))
            if (adapter_modules & {id(module) for _, module in named}
                    or adapter_values & {id(value) for value in (*model.parameters(), *model.buffers())}):
                raise ValueError("adapters must be owned separately from the base model")
            targets = [model.get_submodule(path) for path in self.paths]
            for target in targets:
                if sum(module is target for _, module in named) != 1:
                    raise ValueError("aliased insertion modules are ambiguous")
                if target in _ACTIVE_TARGETS:
                    raise RuntimeError("an insertion module already has an active geometry stack")
            token, handles, reserved = object(), [], []
            self._active_token = token
            try:
                for index, target in enumerate(targets):
                    _ACTIVE_TARGETS[target] = token
                    reserved.append(target)
                    handles.append(target.register_forward_hook(self._hook(index, token)))
                handles.append(model.register_forward_pre_hook(self._check_call, with_kwargs=True))
                yield self
            finally:
                self._active_token = None
                for handle in reversed(handles):
                    handle.remove()
                for target in reserved:
                    if _ACTIVE_TARGETS.get(target) is token:
                        del _ACTIVE_TARGETS[target]
