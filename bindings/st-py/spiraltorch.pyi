from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from types import ModuleType

class Tensor:
    def __init__(self, rows: int, cols: int, data: Optional[Sequence[float]] = ...) -> None: ...
    @staticmethod
    def zeros(rows: int, cols: int) -> Tensor: ...
    @staticmethod
    def from_dlpack(capsule: object) -> Tensor: ...
    def to_dlpack(self) -> object: ...
    def __dlpack__(self, *, stream: object | None = ...) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    def shape(self) -> tuple[int, int]: ...
    def tolist(self) -> List[List[float]]: ...

def from_dlpack(capsule: object) -> Tensor: ...

def to_dlpack(tensor: Tensor) -> object: ...

class _CompatTorch(ModuleType):
    def to_torch(
        tensor: Tensor,
        *,
        dtype: object | None = ...,
        device: object | None = ...,
        requires_grad: bool | None = ...,
        copy: bool | None = ...,
        memory_format: object | None = ...,
    ) -> object: ...

    def from_torch(
        tensor: object,
        *,
        dtype: object | None = ...,
        device: object | None = ...,
        ensure_cpu: bool | None = ...,
        copy: bool | None = ...,
        require_contiguous: bool | None = ...,
    ) -> Tensor: ...
    def to_torch(tensor: Tensor) -> object: ...
    def from_torch(tensor: object) -> Tensor: ...

class _CompatJax(ModuleType):
    def to_jax(tensor: Tensor) -> object: ...
    def from_jax(array: object) -> Tensor: ...

class _CompatTensorFlow(ModuleType):
    def to_tensorflow(tensor: Tensor) -> object: ...
    def from_tensorflow(value: object) -> Tensor: ...

class _CompatNamespace(ModuleType):
    torch: _CompatTorch
    jax: _CompatJax
    tensorflow: _CompatTensorFlow

compat: _CompatNamespace

    def capture(value: object, /) -> Tensor: ...
    def share(value: object, target: str, /) -> object: ...

compat: _CompatNamespace

def set_global_seed(seed: int) -> None: ...

def golden_ratio() -> float: ...

def golden_angle() -> float: ...

def fibonacci_pacing(total_steps: int) -> List[int]: ...

def pack_nacci_chunks(order: int, total_steps: int) -> List[int]: ...

def pack_tribonacci_chunks(total_steps: int) -> List[int]: ...

def pack_tetranacci_chunks(total_steps: int) -> List[int]: ...

def generate_plan_batch_ex(
    n: int,
    total_steps: int,
    base_radius: float,
    radial_growth: float,
    base_height: float,
    meso_gain: float,
    micro_gain: float,
    seed: Optional[int] = ...,
) -> List[object]: ...


class _NnDataset:
    def __init__(self) -> None: ...

    @staticmethod
    def from_pairs(samples: Sequence[Tuple[Tensor, Tensor]]) -> _NnDataset: ...

    def push(self, input: Tensor, target: Tensor) -> None: ...

    def len(self) -> int: ...

    def is_empty(self) -> bool: ...

    def loader(self) -> _NnDataLoader: ...

    def __len__(self) -> int: ...


class _NnDataLoader:
    def len(self) -> int: ...

    def __len__(self) -> int: ...

    def is_empty(self) -> bool: ...

    def batch_size(self) -> int: ...

    def prefetch_depth(self) -> int: ...

    def shuffle(self, seed: int) -> _NnDataLoader: ...

    def batched(self, batch_size: int) -> _NnDataLoader: ...

    def dynamic_batch_by_rows(self, max_rows: int) -> _NnDataLoader: ...

    def prefetch(self, depth: int) -> _NnDataLoader: ...

    def iter(self) -> _NnDataLoaderIter: ...

    def __iter__(self) -> _NnDataLoaderIter: ...


class _NnDataLoaderIter(Iterable[Tuple[Tensor, Tensor]]):
    def __iter__(self) -> _NnDataLoaderIter: ...

    def __next__(self) -> Tuple[Tensor, Tensor]: ...


class _NnModule(ModuleType):
    Dataset: type[_NnDataset]
    DataLoader: type[_NnDataLoader]
    DataLoaderIter: type[_NnDataLoaderIter]

    def from_samples(samples: Sequence[Tuple[Tensor, Tensor]]) -> _NnDataLoader: ...


nn: _NnModule


class _FracModule(ModuleType):
    def gl_coeffs_adaptive(alpha: float, tol: float = ..., max_len: int = ...) -> List[float]: ...

    def fracdiff_gl_1d(
        x: Sequence[float],
        alpha: float,
        kernel_len: int,
        pad: str = ...,
        pad_constant: Optional[float] = ...,
        scale: Optional[float] = ...,
    ) -> List[float]: ...


frac: _FracModule

dataset: ModuleType

linalg: ModuleType

rl: ModuleType

rec: ModuleType

telemetry: ModuleType

ecosystem: ModuleType

class QueryPlan:
    def __init__(self, query: str) -> None: ...
    @property
    def query(self) -> str: ...
    def selects(self) -> List[str]: ...
    def limit(self) -> Optional[int]: ...
    def order(self) -> Optional[Tuple[str, str]]: ...
    def filters(self) -> List[Tuple[str, str, float]]: ...

class RecEpochReport:
    rmse: float
    samples: int
    regularization_penalty: float

class Recommender:
    def __init__(self, users: int, items: int, factors: int, learning_rate: float = ..., regularization: float = ..., curvature: float = ...) -> None: ...
    def predict(self, user: int, item: int) -> float: ...
    def train_epoch(self, ratings: Sequence[Tuple[int, int, float]]) -> RecEpochReport: ...
    def recommend_top_k(self, user: int, k: int, exclude: Optional[Sequence[int]] = ...) -> List[Tuple[int, float]]: ...
    def recommend_query(
        self, user: int, query: QueryPlan, exclude: Optional[Sequence[int]] = ...
    ) -> List[Dict[str, float]]: ...
    def user_embedding(self, user: int) -> Tensor: ...
    def item_embedding(self, item: int) -> Tensor: ...
    @property
    def users(self) -> int: ...
    @property
    def items(self) -> int: ...
    @property
    def factors(self) -> int: ...

class DqnAgent:
    def __init__(self, state_dim: int, action_dim: int, discount: float, learning_rate: float) -> None: ...
    def select_action(self, state: int) -> int: ...
    def update(self, state: int, action: int, reward: float, next_state: int) -> None: ...

class PpoAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float, clip_range: float) -> None: ...
    def score_actions(self, state: Sequence[float]) -> List[float]: ...
    def value(self, state: Sequence[float]) -> float: ...
    def update(self, state: Sequence[float], action: int, advantage: float, old_log_prob: float) -> None: ...

class SacAgent:
    def __init__(self, state_dim: int, action_dim: int, temperature: float) -> None: ...
    def sample_action(self, state: Sequence[float]) -> int: ...
    def jitter(self, entropy_target: float) -> None: ...

class DashboardMetric:
    name: str
    value: float
    unit: Optional[str]
    trend: Optional[float]

class DashboardEvent:
    message: str
    severity: str

class DashboardFrame:
    timestamp: float
    metrics: List[DashboardMetric]
    events: List[DashboardEvent]

class DashboardRing:
    def __init__(self, capacity: int) -> None: ...
    def push(self, frame: DashboardFrame) -> None: ...
    def latest(self) -> Optional[DashboardFrame]: ...
    def __iter__(self) -> Iterable[DashboardFrame]: ...

__all__ = [
    "Tensor",
    "compat",
    "capture",
    "share",
    "from_dlpack",
    "to_dlpack",
    "nn",
    "frac",
    "dataset",
    "linalg",
    "rl",
    "rec",
    "telemetry",
    "ecosystem",
    "compat",
    "set_global_seed",
    "golden_ratio",
    "golden_angle",
    "fibonacci_pacing",
    "pack_nacci_chunks",
    "pack_tribonacci_chunks",
    "pack_tetranacci_chunks",
    "generate_plan_batch_ex",
    "gl_coeffs_adaptive",
    "fracdiff_gl_1d",
    "QueryPlan",
    "RecEpochReport",
    "Recommender",
    "DqnAgent",
    "PpoAgent",
    "SacAgent",
    "DashboardMetric",
    "DashboardEvent",
    "DashboardFrame",
    "DashboardRing",
]
