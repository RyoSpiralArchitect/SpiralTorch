from __future__ import annotations

import cmath as _cmath
import math as _math
import types as _types
import sys
from collections import deque as _deque
from dataclasses import dataclass as _dataclass
from importlib import import_module
from typing import (
    Any as _Any,
    Callable as _Callable,
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
    Mapping as _Mapping,
    MutableSequence as _MutableSequence,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
)
from importlib.metadata import version as _pkg_version, PackageNotFoundError

from ._meta import (
    BUILD_FINGERPRINT,
    BUILD_ID,
    BUILD_MANIFEST,
    BUILD_MANIFEST_JSON,
)

_rs: _types.ModuleType | None = None

_PREDECLARED_SUBMODULES: list[tuple[str, str]] = [
    ("nn", "SpiralTorch neural network primitives"),
    ("frac", "Fractal & fractional tools"),
    ("dataset", "Datasets & loaders"),
    ("linalg", "Linear algebra utilities"),
    ("planner", "Planning & device heuristics"),
    ("spiralk", "SpiralK DSL & hint bridges"),
    ("spiral_rl", "Reinforcement learning components"),
    ("rec", "Reconstruction / signal processing"),
    ("telemetry", "Telemetry / dashboards / metrics"),
    ("ecosystem", "Integrations & ecosystem glue"),
    ("selfsup", "Self-supervised objectives"),
    ("export", "Model export & compression"),
    ("compat", "Interoperability bridges"),
    ("hpo", "Hyper-parameter optimization tools"),
    ("inference", "Safety inference runtime & auditing"),
    ("zspace", "Z-space training helpers"),
    ("vision", "SpiralTorchVision orchestration"),
    ("canvas", "Canvas transformer utilities"),
]

_RENAMED_EXPORTS: dict[str, str] = {
    "DqnAgent": "stAgent",
}


def _safe_getattr(obj: _Any, name: str, default: _Any = None) -> _Any:
    if obj is None or not name:
        return default
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


def _resolve_rs_attr(candidate: str) -> _Any | None:
    if not candidate:
        return None
    target: _Any = _rs
    for part in candidate.split("."):
        target = _safe_getattr(target, part, None)
        if target is None:
            return None
    return target


_parent_module = sys.modules[__name__]
for _name, _doc in _PREDECLARED_SUBMODULES:
    _fq = f"{__name__}.{_name}"
    _module = sys.modules.get(_fq)
    if _module is None:
        _module = _types.ModuleType(_fq, _doc)
        sys.modules[_fq] = _module
    elif _doc and not getattr(_module, "__doc__", None):
        _module.__doc__ = _doc
    setattr(_parent_module, _name, _module)
    globals()[_name] = _module

# --- begin: preseed shim for legacy init that looks for `spiral_rl.DqnAgent` ---
# 一部の初期化コードが `spiral_rl.DqnAgent` を触るので、先に偽モジュールを噛ませる
if "spiral_rl" not in sys.modules:
    _shim = _types.ModuleType("spiral_rl")
    # 参照される両方の候補名を用意しておく（実体は後で本物に差し替え）
    _shim.DqnAgent = type("DqnAgent", (), {})  # placeholder
    _shim.PyDqnAgent = type("PyDqnAgent", (), {})  # placeholder
    sys.modules["spiral_rl"] = _shim
# ついでに第三者パッケージの `rl` が入り込む事故を防止
if "rl" not in sys.modules:
    sys.modules["rl"] = _types.ModuleType("rl")
# --- end: preseed shim ---

# Rust拡張の本体
try:
    _rs = import_module("spiraltorch.spiraltorch")
except ModuleNotFoundError as exc:
    if exc.name not in {"spiraltorch.spiraltorch", "spiraltorch"}:
        raise
    try:
        _rs = import_module("spiraltorch.spiraltorch_native")
    except ModuleNotFoundError:
        _rs = import_module("spiraltorch_native")

# --- begin: promote real rl submodule & alias DqnAgent->stAgent ---
try:
    _spiral_rl = globals().get("spiral_rl")
    if isinstance(_spiral_rl, _types.ModuleType):
        sys.modules["spiral_rl"] = _spiral_rl
        if hasattr(_spiral_rl, "stAgent") and not hasattr(_spiral_rl, "DqnAgent"):
            setattr(_spiral_rl, "DqnAgent", getattr(_spiral_rl, "stAgent"))
except Exception:
    # フェイルセーフ（失敗しても致命ではない）
    pass
# --- end: promote ---

# パッケージ版
try:
    __version__ = _pkg_version("spiraltorch")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


def print_build_id(*, verbose: bool = False) -> None:
    """Display the build identifier embedded in the wheel."""

    if verbose:
        print(f"[SpiralTorch] Build manifest: {BUILD_MANIFEST_JSON}")
    else:
        print(f"[SpiralTorch] Build ID: {BUILD_ID} ({BUILD_FINGERPRINT})")


def build_manifest() -> dict[str, _Any]:
    """Return a copy of the structured build metadata."""

    return dict(BUILD_MANIFEST)

# 追加API（Rust側でエクスポート済みのやつだけ拾う）
_EXTRAS = [
    "golden_ratio","golden_angle","set_global_seed",
    "capture","share","compat",
    "fibonacci_pacing","pack_nacci_chunks",
    "pack_tribonacci_chunks","pack_tetranacci_chunks",
    "generate_plan_batch_ex","plan","plan_topk",
    "describe_device","hip_probe","z_space_barycenter",
]
for _n in _EXTRAS:
    _value = _safe_getattr(_rs, _n, None)
    if _value is not None:
        globals()[_n] = _value

# 後方互換の別名（存在する方を公開名にバインド）
_COMPAT_ALIAS = {
    "Tensor":   ("Tensor", "PyTensor"),
    "Device":   ("Device", "PyDevice"),
    "Dataset":  ("Dataset", "PyDataset"),
    "Plan":     ("Plan", "PyPlan"),
}
for _pub, _cands in _COMPAT_ALIAS.items():
    for _c in _cands:
        _value = _safe_getattr(_rs, _c, None)
        if _value is not None:
            globals()[_pub] = _value
            break

_FORWARDING_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "nn": {
        "Dataset": ("_NnDataset",),
        "DataLoader": ("_NnDataLoader",),
        "DataLoaderIter": ("_NnDataLoaderIter",),
        "from_samples": ("nn_from_samples", "dataset_from_samples"),
    },
    "compat": {
        "capture": ("capture",),
        "share": ("share",),
    },
    "planner": {
        "RankPlan": ("PyRankPlan",),
        "plan": (),
        "plan_topk": (),
        "describe_device": (),
        "hip_probe": (),
        "generate_plan_batch_ex": (),
    },
    "spiralk": {
        "FftPlan": (),
        "MaxwellBridge": (),
        "MaxwellHint": (),
        "MaxwellFingerprint": (),
        "MeaningGate": (),
        "SequentialZ": (),
        "MaxwellPulse": (),
        "MaxwellProjector": (),
        "required_blocks": (),
    },
    "compat.torch": {
        "to_torch": ("compat_to_torch", "to_torch"),
        "from_torch": ("compat_from_torch", "from_torch"),
    },
    "compat.jax": {
        "to_jax": ("compat_to_jax", "to_jax"),
        "from_jax": ("compat_from_jax", "from_jax"),
    },
    "compat.tensorflow": {
        "to_tensorflow": ("compat_to_tensorflow", "to_tensorflow"),
        "from_tensorflow": ("compat_from_tensorflow", "from_tensorflow"),
    },
    "spiral_rl": {
        "stAgent": ("PyDqnAgent", "DqnAgent", "StAgent"),
        "PpoAgent": ("PyPpoAgent",),
        "SacAgent": ("PySacAgent",),
    },
    "rec": {
        "QueryPlan": ("PyQueryPlan",),
        "RecEpochReport": ("PyRecEpochReport",),
        "Recommender": ("PyRecommender",),
    },
    "telemetry": {
        "DashboardMetric": ("PyDashboardMetric",),
        "DashboardEvent": ("PyDashboardEvent",),
        "DashboardFrame": ("PyDashboardFrame",),
        "DashboardRing": ("PyDashboardRing",),
        "DashboardRingIter": ("PyDashboardRingIter",),
    },
    "export": {
        "QatObserver": ("PyQatObserver",),
        "QuantizationReport": ("PyQuantizationReport",),
        "StructuredPruningReport": ("PyStructuredPruningReport",),
        "CompressionReport": ("PyCompressionReport",),
        "structured_prune": (),
        "compress_weights": (),
    },
    "hpo": {
        "SearchLoop": ("PySearchLoop",),
    },
    "selfsup": {
        "info_nce": ("selfsup.info_nce",),
        "masked_mse": ("selfsup.masked_mse",),
    },
}


@_dataclass
class ZMetrics:
    """Typed metrics container fed into :class:`ZSpaceTrainer`."""

    speed: float
    memory: float
    stability: float
    gradient: _Optional[_Sequence[float]] = None
    drs: float = 0.0


def _clone_volume(volume: _Sequence[_Sequence[_Sequence[float]]]) -> _List[_List[_List[float]]]:
    return [[list(row) for row in slice_] for slice_ in volume]


def _coerce_slice(
    data: _Sequence[_Sequence[float]] | _Any,
    height: _Optional[int] = None,
    width: _Optional[int] = None,
) -> _List[_List[float]]:
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, _Sequence):
        raise TypeError("slice must be a sequence of rows")
    rows_seq = list(data)
    rows: _List[_List[float]] = []
    if height is None:
        height = len(rows_seq)
    if len(rows_seq) != height:
        raise ValueError(f"expected {height} rows, received {len(rows_seq)}")
    for row in rows_seq:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, _Sequence):
            raise TypeError("slice rows must be sequences")
        values = [float(v) for v in row]
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError(f"expected row width {width}, received {len(values)}")
        rows.append(values)
    return rows


def _coerce_volume(
    volume: _Sequence[_Sequence[_Sequence[float]]],
    depth: int,
    height: int,
    width: int,
) -> _List[_List[_List[float]]]:
    if hasattr(volume, "tolist"):
        volume = volume.tolist()  # type: ignore[assignment]
    if len(volume) != depth:
        raise ValueError(f"expected {depth} slices, received {len(volume)}")
    slices: _List[_List[_List[float]]] = []
    for slice_data in volume:
        slices.append(_coerce_slice(slice_data, height, width))
    return slices


def _spectral_window(name: str | None, depth: int) -> _List[float]:
    if depth <= 0:
        return []
    if name is None:
        return [1.0] * depth
    key = name.lower()
    if key == "hann":
        return [0.5 - 0.5 * _math.cos(2.0 * _math.pi * n / max(1, depth - 1)) for n in range(depth)]
    if key == "hamming":
        return [0.54 - 0.46 * _math.cos(2.0 * _math.pi * n / max(1, depth - 1)) for n in range(depth)]
    if key == "blackman":
        return [
            0.42
            - 0.5 * _math.cos(2.0 * _math.pi * n / max(1, depth - 1))
            + 0.08 * _math.cos(4.0 * _math.pi * n / max(1, depth - 1))
            for n in range(depth)
        ]
    if key == "gaussian":
        centre = 0.5 * (depth - 1)
        sigma = max(depth * 0.17, 1.0)
        return [
            _math.exp(-0.5 * ((n - centre) / sigma) ** 2)
            for n in range(depth)
        ]
    raise ValueError(f"unknown spectral window '{name}'")


def _blend_volumes(
    current: _Sequence[_Sequence[_Sequence[float]]],
    update: _Sequence[_Sequence[_Sequence[float]]],
    alpha: float,
) -> _List[_List[_List[float]]]:
    blended: _List[_List[_List[float]]] = []
    for cur_slice, upd_slice in zip(current, update):
        upd_rows = _coerce_slice(upd_slice)
        width = len(upd_rows[0]) if upd_rows else None
        cur_rows = _coerce_slice(cur_slice, len(upd_rows), width)
        rows: _List[_List[float]] = []
        for cur_row, upd_row in zip(cur_rows, upd_rows):
            rows.append([
                (1.0 - alpha) * cur_val + alpha * upd_val
                for cur_val, upd_val in zip(cur_row, upd_row)
            ])
        blended.append(rows)
    return blended


class TemporalResonanceBuffer:
    """Maintains an exponential moving average over recent Z-space volumes."""

    def __init__(self, capacity: int = 4, alpha: float = 0.2) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._history: _deque[_List[_List[_List[float]]]] = _deque(maxlen=capacity)
        self._alpha = max(1e-6, min(1.0, float(alpha)))
        self._ema: _Optional[_List[_List[_List[float]]]] = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def capacity(self) -> int:
        maxlen = self._history.maxlen
        return maxlen if maxlen is not None else len(self._history)

    def update(self, volume: _Sequence[_Sequence[_Sequence[float]]]) -> _List[_List[_List[float]]]:
        snapshot = _clone_volume(volume)
        self._history.append(snapshot)
        if self._ema is None:
            self._ema = snapshot
        else:
            self._ema = _blend_volumes(self._ema, snapshot, self._alpha)
        return _clone_volume(self._ema)

    def state(self) -> _Optional[_List[_List[_List[float]]]]:
        if self._ema is not None:
            return _clone_volume(self._ema)
        if self._history:
            return _clone_volume(self._history[-1])
        return None

    def history(self) -> _List[_List[_List[_List[float]]]]:
        return [_clone_volume(volume) for volume in self._history]

    def state_dict(self) -> _Dict[str, _Any]:
        return {
            "capacity": self.capacity,
            "alpha": self._alpha,
            "history": self.history(),
            "ema": _clone_volume(self._ema) if self._ema is not None else None,
        }

    def load_state_dict(self, state: _Mapping[str, _Any]) -> None:
        if not isinstance(state, _Mapping):
            raise TypeError("state must be a mapping")
        capacity = int(state.get("capacity", self.capacity) or self.capacity)
        if capacity <= 0:
            raise ValueError("state capacity must be positive")
        self._alpha = max(1e-6, min(1.0, float(state.get("alpha", self._alpha))))
        self._history = _deque(maxlen=capacity)
        history = state.get("history", [])
        if history:
            if not isinstance(history, _Sequence):
                raise TypeError("history must be a sequence of volumes")
            for volume in history:
                self._history.append(_clone_volume(volume))
        ema = state.get("ema")
        self._ema = _clone_volume(ema) if ema is not None else None


@_dataclass
class SliceProfile:
    mean: float
    std: float
    energy: float


class SpiralTorchVision:
    """Minimal Python orchestrator for SpiralTorchVision pipelines."""

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        *,
        alpha: float = 0.2,
        window: str | None = "hann",
        temporal: int = 4,
    ) -> None:
        if depth <= 0 or height <= 0 or width <= 0:
            raise ValueError("depth, height, and width must be positive")
        self.depth = depth
        self.height = height
        self.width = width
        self._alpha = max(1e-6, min(1.0, float(alpha)))
        self._window_name = window
        self._window = _spectral_window(window, depth)
        self._buffer_capacity = max(1, int(temporal))
        self._volume: _List[_List[_List[float]]] = [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(depth)
        ]
        self._buffer = TemporalResonanceBuffer(capacity=self._buffer_capacity, alpha=self._alpha)

    @property
    def volume(self) -> _List[_List[_List[float]]]:
        return _clone_volume(self._volume)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def temporal_capacity(self) -> int:
        return self._buffer_capacity

    @property
    def temporal_state(self) -> _Optional[_List[_List[_List[float]]]]:
        return self._buffer.state()

    @property
    def window(self) -> _List[float]:
        return list(self._window)

    def reset(self) -> None:
        for slice_ in self._volume:
            for row in slice_:
                for idx in range(len(row)):
                    row[idx] = 0.0
        self._buffer = TemporalResonanceBuffer(capacity=self._buffer_capacity, alpha=self._alpha)

    def update_window(self, window: str | _Sequence[float] | None) -> None:
        if window is None or isinstance(window, str):
            self._window_name = window
            self._window = _spectral_window(window, self.depth)
            if not self._window:
                self._window = [1.0] * self.depth
            return
        values = [float(v) for v in window]
        if values and len(values) != self.depth:
            raise ValueError(
                f"expected window with {self.depth} coefficients, received {len(values)}"
            )
        if not values:
            values = [1.0] * self.depth
        self._window_name = None
        self._window = values

    def accumulate(self, volume: _Sequence[_Sequence[_Sequence[float]]], weight: float = 1.0) -> None:
        if hasattr(volume, "tolist"):
            volume = volume.tolist()
        if len(volume) != self.depth:
            raise ValueError(f"expected {self.depth} slices, received {len(volume)}")
        w = max(0.0, float(weight))
        alpha = self._alpha * (w if w else 1.0)
        for idx, slice_data in enumerate(volume):
            rows = _coerce_slice(slice_data, self.height, self.width)
            for r_idx, row in enumerate(rows):
                target_row = self._volume[idx][r_idx]
                for c_idx, value in enumerate(row):
                    target_row[c_idx] = (1.0 - alpha) * target_row[c_idx] + alpha * value
        self._buffer.update(self._volume)

    def accumulate_slices(self, slices: _Sequence[_Sequence[_Sequence[float]]]) -> None:
        self.accumulate(slices)

    def accumulate_sequence(
        self,
        frames: _Iterable[_Sequence[_Sequence[_Sequence[float]]]],
        weights: _Optional[_Sequence[float]] = None,
    ) -> None:
        if weights is None:
            for frame in frames:
                self.accumulate(frame)
            return
        for frame, weight in zip(frames, weights):
            self.accumulate(frame, weight)

    def project(self, *, normalise: bool = True) -> _List[_List[float]]:
        window = self._window or [1.0] * self.depth
        if not window:
            window = [1.0] * self.depth
        total: _List[_List[float]] = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        weight_sum = 0.0
        for coeff, slice_ in zip(window, self._volume):
            if coeff == 0.0:
                continue
            weight_sum += coeff
            for r_idx, row in enumerate(slice_):
                target = total[r_idx]
                for c_idx, value in enumerate(row):
                    target[c_idx] += coeff * value
        if normalise and weight_sum:
            inv = 1.0 / weight_sum
            for row in total:
                for idx in range(len(row)):
                    row[idx] *= inv
        return total

    def volume_energy(self) -> float:
        acc = 0.0
        for slice_ in self._volume:
            for row in slice_:
                for value in row:
                    acc += value * value
        return acc

    def slice_profile(self) -> _List[SliceProfile]:
        profiles: _List[SliceProfile] = []
        for slice_ in self._volume:
            flat = [value for row in slice_ for value in row]
            if not flat:
                profiles.append(SliceProfile(0.0, 0.0, 0.0))
                continue
            mean = sum(flat) / len(flat)
            var = sum((value - mean) ** 2 for value in flat) / len(flat)
            energy = sum(value * value for value in flat) / len(flat)
            profiles.append(SliceProfile(mean, _math.sqrt(var), energy))
        return profiles

    def snapshot(self) -> _Dict[str, _Any]:
        return {
            "volume": self.volume,
            "profiles": self.slice_profile(),
            "energy": self.volume_energy(),
            "temporal": self._buffer.state(),
        }

    def state_dict(self) -> _Dict[str, _Any]:
        return {
            "depth": self.depth,
            "height": self.height,
            "width": self.width,
            "alpha": self._alpha,
            "window": list(self._window),
            "window_name": self._window_name,
            "buffer": self._buffer.state_dict(),
            "volume": self.volume,
        }

    def load_state_dict(self, state: _Mapping[str, _Any], *, strict: bool = True) -> None:
        if not isinstance(state, _Mapping):
            raise TypeError("state must be a mapping")
        depth = int(state.get("depth", self.depth))
        height = int(state.get("height", self.height))
        width = int(state.get("width", self.width))
        if strict and (depth != self.depth or height != self.height or width != self.width):
            raise ValueError("state dimensions do not match the vision volume")
        alpha = float(state.get("alpha", self._alpha))
        self.depth = depth
        self.height = height
        self.width = width
        self._alpha = max(1e-6, min(1.0, alpha))
        buffer_state = state.get("buffer")
        capacity = self._buffer_capacity
        if isinstance(buffer_state, _Mapping):
            capacity = int(buffer_state.get("capacity", capacity) or capacity)
        if capacity <= 0:
            capacity = self._buffer_capacity
        if capacity != self._buffer_capacity:
            self._buffer_capacity = capacity
            self._buffer = TemporalResonanceBuffer(capacity=capacity, alpha=self._alpha)
        if isinstance(buffer_state, _Mapping):
            self._buffer.load_state_dict(buffer_state)
        else:
            self._buffer._alpha = self._alpha  # keep alpha in sync when no buffer state is supplied
        window_name = state.get("window_name")
        window_values = state.get("window")
        if window_name is not None:
            self.update_window(window_name)
        elif isinstance(window_values, _Sequence):
            self.update_window(list(window_values))
        volume_data = state.get("volume")
        if volume_data is not None:
            coerced = _coerce_volume(volume_data, self.depth, self.height, self.width)
            self._volume = coerced
        temporal_state = state.get("temporal")
        if temporal_state is not None:
            current = self._buffer.state_dict()
            current["ema"] = temporal_state
            self._buffer.load_state_dict(current)


class ZSpaceTrainer:
    """Lightweight Adam optimiser operating on a Z vector."""

    def __init__(
        self,
        z_dim: int = 4,
        *,
        alpha: float = 0.35,
        lam_speed: float = 0.5,
        lam_mem: float = 0.3,
        lam_stab: float = 0.2,
        lam_frac: float = 0.1,
        lam_drs: float = 0.0,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        self._z: _List[float] = [0.0] * z_dim
        self._alpha = max(1e-6, float(alpha))
        self._lam = (float(lam_speed), float(lam_mem), float(lam_stab), float(lam_frac), float(lam_drs))
        self._lr = float(lr)
        self._beta1 = float(beta1)
        self._beta2 = float(beta2)
        self._eps = float(eps)
        self._m: _List[float] = [0.0] * z_dim
        self._v: _List[float] = [0.0] * z_dim
        self._t = 0

    @property
    def state(self) -> _List[float]:
        return list(self._z)

    def reset(self) -> None:
        for arr in (self._z, self._m, self._v):
            for idx in range(len(arr)):
                arr[idx] = 0.0
        self._t = 0

    def state_dict(self) -> _Dict[str, _Any]:
        return {
            "z": list(self._z),
            "moment": list(self._m),
            "velocity": list(self._v),
            "step": self._t,
            "hyperparams": {
                "alpha": self._alpha,
                "lambda": self._lam,
                "lr": self._lr,
                "beta1": self._beta1,
                "beta2": self._beta2,
                "eps": self._eps,
            },
        }

    def load_state_dict(self, state: _Mapping[str, _Any], *, strict: bool = True) -> None:
        if not isinstance(state, _Mapping):
            raise TypeError("state must be a mapping")
        z = state.get("z")
        moment = state.get("moment")
        velocity = state.get("velocity")
        if z is None or moment is None or velocity is None:
            if strict:
                missing = [
                    key
                    for key, value in (("z", z), ("moment", moment), ("velocity", velocity))
                    if value is None
                ]
                raise KeyError(f"missing keys in state: {missing}")
            z = z or self._z
            moment = moment or self._m
            velocity = velocity or self._v
        self._assign_vector(self._z, z, strict)
        self._assign_vector(self._m, moment, strict)
        self._assign_vector(self._v, velocity, strict)
        self._t = int(state.get("step", self._t))
        hyper = state.get("hyperparams")
        if isinstance(hyper, _Mapping):
            alpha = float(hyper.get("alpha", self._alpha))
            self._alpha = max(1e-6, alpha)
            lam = hyper.get("lambda")
            if isinstance(lam, _Sequence) and len(lam) == 5:
                self._lam = tuple(float(value) for value in lam)
            self._lr = float(hyper.get("lr", self._lr))
            self._beta1 = float(hyper.get("beta1", self._beta1))
            self._beta2 = float(hyper.get("beta2", self._beta2))
            self._eps = float(hyper.get("eps", self._eps))

    def _assign_vector(self, target: _MutableSequence[float], values: _Any, strict: bool) -> None:
        data = [float(v) for v in values]
        if len(data) != len(target):
            if strict:
                raise ValueError(
                    f"expected vector of length {len(target)}, received {len(data)}"
                )
            if len(data) < len(target):
                data.extend(0.0 for _ in range(len(target) - len(data)))
            else:
                data = data[: len(target)]
        for idx, value in enumerate(data):
            target[idx] = value

    def step_batch(
        self,
        metrics: _Iterable[_Mapping[str, float] | ZMetrics],
    ) -> _List[float]:
        losses: _List[float] = []
        for sample in metrics:
            losses.append(self.step(sample))
        return losses

    def _rfft(self, values: _Sequence[float]) -> _List[complex]:
        n = len(values)
        if n == 0:
            return []
        freq: _List[complex] = []
        for k in range(n // 2 + 1):
            total = 0.0j
            for t, val in enumerate(values):
                angle = -2.0 * _math.pi * k * t / max(1, n)
                total += complex(val, 0.0) * _cmath.exp(1j * angle)
            freq.append(total)
        return freq

    def _frac_reg(self, values: _Sequence[float]) -> float:
        spectrum = self._rfft(values)
        n = len(spectrum)
        if n <= 1:
            return 0.0
        acc = 0.0
        for idx, coeff in enumerate(spectrum):
            omega = idx / max(1, n - 1)
            weight = omega ** (2.0 * self._alpha)
            acc += weight * abs(coeff) ** 2
        return acc / n

    def _frac_grad(self) -> _List[float]:
        grad: _List[float] = []
        base = self._frac_reg(self._z)
        step = 1e-4
        for i in range(len(self._z)):
            original = self._z[i]
            self._z[i] = original + step
            plus = self._frac_reg(self._z)
            self._z[i] = original - step
            minus = self._frac_reg(self._z)
            self._z[i] = original
            grad.append((plus - minus) / (2.0 * step))
        scale = max(1.0, max(abs(g) for g in grad) if grad else 1.0)
        return [g / scale for g in grad]

    def _normalise(self, value: float) -> float:
        return _math.tanh(value)

    def _normalise_gradient(self, grad: _Sequence[float] | None) -> _List[float]:
        if not grad:
            return [0.0] * len(self._z)
        grad_list = list(grad)
        if len(grad_list) == len(self._z):
            return [self._normalise(g) for g in grad_list]
        out: _List[float] = []
        for idx in range(len(self._z)):
            out.append(self._normalise(grad_list[idx % len(grad_list)]))
        return out

    def _adam_update(self, grad: _Sequence[float]) -> None:
        self._t += 1
        for i, g in enumerate(grad):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * (g * g)
            m_hat = self._m[i] / (1.0 - self._beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - self._beta2 ** self._t)
            self._z[i] -= self._lr * m_hat / (_math.sqrt(v_hat) + self._eps)

    def step(self, metrics: _Mapping[str, float] | ZMetrics) -> float:
        if isinstance(metrics, ZMetrics):
            speed = float(metrics.speed)
            memory = float(metrics.memory)
            stability = float(metrics.stability)
            gradient = metrics.gradient
            drs_signal = float(metrics.drs)
        else:
            speed = float(metrics.get("speed", 0.0))
            memory = float(metrics.get("mem", metrics.get("memory", 0.0)))
            stability = float(metrics.get("stab", metrics.get("stability", 0.0)))
            grad = metrics.get("gradient")
            gradient = grad if isinstance(grad, _Sequence) else None
            drs_signal = float(metrics.get("drs", 0.0))
        lam_speed, lam_mem, lam_stab, lam_frac, lam_drs = self._lam
        penalty = (
            lam_speed * self._normalise(speed)
            + lam_mem * self._normalise(memory)
            + lam_stab * self._normalise(stability)
        )
        if lam_drs:
            penalty += lam_drs * self._normalise(drs_signal)
        frac_reg = self._frac_reg(self._z)
        loss = penalty + lam_frac * frac_reg
        grad_metric = self._normalise_gradient(gradient)
        frac_grad = self._frac_grad()
        total_grad = [grad_metric[idx] + lam_frac * frac_grad[idx] for idx in range(len(self._z))]
        self._adam_update(total_grad)
        return loss


def step_many(trainer: ZSpaceTrainer, samples: _Iterable[_Mapping[str, float] | ZMetrics]) -> _List[float]:
    for metrics in samples:
        trainer.step(metrics)
    return trainer.state


def stream_zspace_training(
    trainer: ZSpaceTrainer,
    samples: _Iterable[_Mapping[str, float] | ZMetrics],
    *,
    on_step: _Optional[_Callable[[int, _List[float], float], None]] = None,
) -> _List[float]:
    losses: _List[float] = []
    for index, metrics in enumerate(samples):
        loss = trainer.step(metrics)
        losses.append(loss)
        if on_step is not None:
            on_step(index, trainer.state, loss)
    return losses


def _matrix_summary(matrix: _Sequence[_Sequence[float]]) -> _Dict[str, float]:
    flat = [float(value) for row in matrix for value in row]
    if not flat:
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0}
    l1 = sum(abs(value) for value in flat)
    l2 = _math.sqrt(sum(value * value for value in flat))
    linf = max(abs(value) for value in flat)
    mean = sum(flat) / len(flat)
    return {"l1": l1, "l2": l2, "linf": linf, "mean": mean}


def _coerce_matrix(matrix: _Any, height: int, width: int) -> _List[_List[float]]:
    rows = _coerce_slice(matrix, height, width)
    return rows


class _ForwardingModule(_types.ModuleType):
    """Module stub that forwards attribute lookups to the Rust backend."""

    def __init__(self, name: str, doc: str, key: str) -> None:
        super().__init__(name, doc)
        self.__dict__["_forward_key"] = key

    @property
    def _forward_key(self) -> str:
        return self.__dict__["_forward_key"]

    def __getattr__(self, attr: str) -> _Any:
        if attr.startswith("_"):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{attr}'")

        # Prefer already-exposed globals so top-level mirrors stay consistent.
        if attr in globals():
            value = globals()[attr]
            setattr(self, attr, value)
            _register_module_export(self, attr)
            return value

        hints = _FORWARDING_HINTS.get(self._forward_key, {})
        candidates: list[str] = []
        aliases = hints.get(attr)
        if aliases:
            candidates.extend(aliases)

        namespace_parts = self._forward_key.split(".")
        suffix = namespace_parts[-1]
        flat_suffix = "_".join(namespace_parts)
        candidates.extend(
            [
                attr,
                f"{suffix}_{attr}",
                f"{suffix}_{attr.lower()}",
                f"{flat_suffix}_{attr}",
                f"{flat_suffix}_{attr.lower()}",
            ]
        )

        for candidate in dict.fromkeys(candidates):
            value = _resolve_rs_attr(candidate)
            if value is not None:
                setattr(self, attr, value)
                _register_module_export(self, attr)
                return value

        raise AttributeError(f"module '{self.__name__}' has no attribute '{attr}'")

    def __dir__(self) -> list[str]:
        exported = set(getattr(self, "__all__", ()))
        exported.update(super().__dir__())
        hints = _FORWARDING_HINTS.get(self._forward_key, {})
        exported.update(hints.keys())
        suffix = self._forward_key.split(".")[-1] + "_"
        flat_suffix = "_".join(self._forward_key.split(".")) + "_"
        if _rs is not None:
            for name in dir(_rs):
                trimmed = None
                if name.startswith(suffix):
                    trimmed = name[len(suffix):]
                elif name.startswith(flat_suffix):
                    trimmed = name[len(flat_suffix):]
                if not trimmed:
                    continue
                trimmed = _RENAMED_EXPORTS.get(trimmed, trimmed)
                exported.add(trimmed)
        return sorted(exported)


def _register_module_export(module: _types.ModuleType, name: str) -> None:
    exported = set(getattr(module, "__all__", ()))
    exported.add(name)
    module.__all__ = sorted(exported)


def _ensure_submodule(name: str, doc: str = "") -> _types.ModuleType:
    """Return or create a synthetic child module without touching the native core."""

    parts = name.split(".")
    fq = __name__
    parent: _types.ModuleType = sys.modules[__name__]
    for idx, part in enumerate(parts):
        fq = f"{fq}.{part}"
        module = sys.modules.get(fq)
        final = idx == len(parts) - 1
        doc_for_part = doc if final else ""
        if module is None:
            key = ".".join(parts[: idx + 1])
            module = _ForwardingModule(fq, doc_for_part, key)
            sys.modules[fq] = module
        elif doc_for_part and not getattr(module, "__doc__", None):
            module.__doc__ = doc_for_part

        setattr(parent, part, module)
        if idx == 0:
            globals()[part] = module
        parent = module
    return parent


def _expose_from_rs(name: str, *aliases: str) -> None:
    if name in globals():
        return
    for candidate in (name, *aliases):
        value = _resolve_rs_attr(candidate)
        if value is not None:
            globals()[name] = value
            return


def _mirror_into_module(
    name: str,
    members: _Iterable[str] | _Mapping[str, _Iterable[str]],
    *,
    reexport: bool = True,
) -> _types.ModuleType:
    module = _ensure_submodule(name)
    exported: set[str] = set(getattr(module, "__all__", ()))
    items: _Iterable[tuple[str, _Iterable[str]]] \
        = members.items() if isinstance(members, _Mapping) else ((m, ()) for m in members)
    for member, aliases in items:
        value = None
        if reexport:
            _expose_from_rs(member, *aliases)
            value = globals().get(member)
        else:
            if member in globals():
                value = globals()[member]
            if value is None:
                for candidate in (member, *aliases):
                    value = _safe_getattr(_rs, candidate, None)
                    if value is not None:
                        break
        if value is None:
            continue
        if reexport:
            globals()[member] = value
        setattr(module, member, value)
        exported.add(member)
    if exported:
        module.__all__ = sorted(exported)
    return module


for _name, _doc in _PREDECLARED_SUBMODULES:
    _module = _ensure_submodule(_name, _doc)
    if not isinstance(_module, _ForwardingModule):
        _fq = f"{__name__}.{_name}"
        _forward = _ForwardingModule(_fq, getattr(_module, "__doc__", _doc), _name)
        for _key, _value in vars(_module).items():
            if _key in {"__dict__", "__weakref__"}:
                continue
            if _key == "__name__":
                continue
            setattr(_forward, _key, _value)
        sys.modules[_fq] = _forward
        setattr(sys.modules[__name__], _name, _forward)
        globals()[_name] = _forward


_compat_children = {
    "torch": "PyTorch interoperability helpers",
    "jax": "JAX interoperability helpers",
    "tensorflow": "TensorFlow interoperability helpers",
}
for _child, _doc in _compat_children.items():
    _ensure_submodule(f"compat.{_child}", _doc)
_compat_module = globals().get("compat")
if isinstance(_compat_module, _types.ModuleType):
    _compat_exports = set(getattr(_compat_module, "__all__", ()))
    _compat_exports.update(_compat_children.keys())
    _compat_module.__all__ = sorted(_compat_exports)


_mirror_into_module(
    "inference",
    [
        "SafetyViolation","SafetyVerdict","AuditEvent","AuditLog",
        "InferenceResult","InferenceRuntime",
    ],
)


_mirror_into_module(
    "hpo",
    {
        "SearchLoop": ("PySearchLoop",),
    },
)


_mirror_into_module(
    "export",
    {
        "QatObserver": ("PyQatObserver",),
        "QuantizationReport": ("PyQuantizationReport",),
        "StructuredPruningReport": ("PyStructuredPruningReport",),
        "CompressionReport": ("PyCompressionReport",),
        "structured_prune": (),
        "compress_weights": (),
    },
)


_mirror_into_module(
    "nn",
    {
        "Dataset": ("_NnDataset",),
        "DataLoader": ("_NnDataLoader",),
        "DataLoaderIter": ("_NnDataLoaderIter",),
        "from_samples": ("nn_from_samples", "dataset_from_samples"),
    },
)
_mirror_into_module(
    "frac",
    {
        "gl_coeffs_adaptive": ("frac.gl_coeffs_adaptive",),
        "fracdiff_gl_1d": ("frac.fracdiff_gl_1d",),
    },
)
_mirror_into_module(
    "spiral_rl",
    {
        "stAgent": ("PyDqnAgent", "DqnAgent", "StAgent"),
        "PpoAgent": ("PyPpoAgent",),
        "SacAgent": ("PySacAgent",),
    },
)
_mirror_into_module(
    "rec",
    {
        "QueryPlan": ("PyQueryPlan",),
        "RecEpochReport": ("PyRecEpochReport",),
        "Recommender": ("PyRecommender",),
    },
)
_mirror_into_module(
    "telemetry",
    {
        "DashboardMetric": ("PyDashboardMetric",),
        "DashboardEvent": ("PyDashboardEvent",),
        "DashboardFrame": ("PyDashboardFrame",),
        "DashboardRing": ("PyDashboardRing",),
    },
)


_mirror_into_module(
    "compat",
    [
        "capture",
        "share",
    ],
    reexport=False,
)
_mirror_into_module(
    "compat.torch",
    {
        "to_torch": ("compat_to_torch", "to_torch"),
        "from_torch": ("compat_from_torch", "from_torch"),
    },
    reexport=False,
)
_mirror_into_module(
    "compat.jax",
    {
        "to_jax": ("compat_to_jax", "to_jax"),
        "from_jax": ("compat_from_jax", "from_jax"),
    },
    reexport=False,
)
_mirror_into_module(
    "compat.tensorflow",
    {
        "to_tensorflow": ("compat_to_tensorflow", "to_tensorflow"),
        "from_tensorflow": ("compat_from_tensorflow", "from_tensorflow"),
    },
    reexport=False,
)


_mirror_into_module(
    "zspace",
    [
        "ZMetrics",
        "ZSpaceTrainer",
        "step_many",
        "stream_zspace_training",
    ],
    reexport=False,
)
_mirror_into_module(
    "vision",
    [
        "SpiralTorchVision",
        "TemporalResonanceBuffer",
        "SliceProfile",
    ],
    reexport=False,
)
_mirror_into_module(
    "canvas",
    [
        "CanvasTransformer",
        "CanvasSnapshot",
        "apply_vision_update",
    ],
    reexport=False,
)


_mirror_into_module(
    "selfsup",
    {
        "info_nce": ("selfsup.info_nce",),
        "masked_mse": ("selfsup.masked_mse",),
    },
)


_mirror_into_module(
    "spiralk",
    {
        "SpiralKFftPlan": (),
        "MaxwellSpiralKBridge": (),
        "MaxwellSpiralKHint": (),
        "SpiralKContext": (),
        "SpiralKWilsonMetrics": (),
        "SpiralKHeuristicHint": (),
        "wilson_lower_bound": (),
        "should_rewrite": (),
        "synthesize_program": (),
        "rewrite_with_wilson": (),
    },
    reexport=False,
)


_mirror_into_module(
    "planner",
    {
        "RankPlan": (),
        "plan": (),
        "plan_topk": (),
        "describe_device": (),
        "hip_probe": (),
        "generate_plan_batch_ex": (),
    },
    reexport=False,
)


_mirror_into_module(
    "spiralk",
    {
        "FftPlan": (),
        "MaxwellBridge": (),
        "MaxwellHint": (),
        "required_blocks": (),
    },
    reexport=False,
)


class SpiralSession:
    """Lightweight execution context for quick experimentation."""

    backend: str
    seed: int | None
    device: str

    def __init__(self, backend: str = "auto", seed: int | None = None) -> None:
        self.backend = backend
        self.seed = seed
        self.device = "wgpu" if backend == "wgpu" else "cpu"

    def plan_topk(self, rows: int, cols: int, k: int):
        return plan_topk(rows, cols, k, backend=self.backend)

    def close(self) -> None:
        """Release any session-scoped resources (currently a no-op)."""


_EXTRAS.append("SpiralSession")


for _key, _hint in _FORWARDING_HINTS.items():
    _module = _ensure_submodule(_key)
    if not _hint:
        continue
    _exports = set(getattr(_module, "__all__", ()))
    _exports.update(_hint.keys())
    _module.__all__ = sorted(_exports)


_CORE_EXPORTS = [
    "Tensor","ComplexTensor","OpenCartesianTopos","LanguageWaveEncoder",
    "GradientSummary","Hypergrad","TensorBiome",
    "LinearModel",
    "BarycenterIntermediate","ZSpaceBarycenter",
    "QueryPlan","RecEpochReport","Recommender",
    "stAgent","PpoAgent","SacAgent",
    "DashboardMetric","DashboardEvent","DashboardFrame","DashboardRing",
    "AuditEvent","AuditLog","InferenceResult","InferenceRuntime",
    "SafetyVerdict","SafetyViolation",
    "SearchLoop",
    "QatObserver","QuantizationReport","StructuredPruningReport","CompressionReport",
    "structured_prune","compress_weights",
    "ModuleTrainer","ZSpaceTrainer","ZSpaceCoherenceSequencer","TemporalResonanceBuffer","SpiralTorchVision",
    "CanvasTransformer","CanvasSnapshot","apply_vision_update",
    "ZMetrics","SliceProfile","step_many","stream_zspace_training",
    "info_nce","masked_mse","mean_squared_error",
]
for _name in _CORE_EXPORTS:
    _expose_from_rs(_name)


def __getattr__(name: str) -> _Any:
    """Defer missing attributes to the Rust extension module.

    This keeps the Python façade lightweight while still exposing the rich
    surface area implemented in Rust.
    """

    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    redirect = _RENAMED_EXPORTS.get(name)
    if redirect is not None:
        _expose_from_rs(redirect)
        if redirect in globals():
            return globals()[redirect]
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    _expose_from_rs(name)
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    _public = set(__all__)
    if _rs is not None:
        for _name in dir(_rs):
            if _name.startswith("_"):
                continue
            _public.add(_RENAMED_EXPORTS.get(_name, _name))
    return sorted(_public)


_EXPORTED = {
    *_EXTRAS,
    *_CORE_EXPORTS,
    *[n for n in _COMPAT_ALIAS if n in globals()],
    "nn","frac","dataset","linalg","spiral_rl","rec","telemetry","ecosystem",
    "selfsup","export","compat","hpo","inference","zspace","vision","canvas",
    "planner","spiralk",
    "__version__",
}
_EXPORTED.update(
    n
    for n in _safe_getattr(_rs, "__all__", ())
    if isinstance(n, str) and not n.startswith("_")
)
__all__ = sorted(_EXPORTED)
