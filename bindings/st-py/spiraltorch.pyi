from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from types import ModuleType

class Tensor:
    def __init__(self, rows: int, cols: int, data: Optional[Sequence[float]] = ...) -> None: ...
    @staticmethod
    def zeros(rows: int, cols: int) -> Tensor: ...
    @staticmethod
    def randn(
        rows: int,
        cols: int,
        mean: float = ...,
        std: float = ...,
        seed: int | None = ...,
    ) -> Tensor: ...
    @staticmethod
    def rand(
        rows: int,
        cols: int,
        min: float = ...,
        max: float = ...,
        seed: int | None = ...,
    ) -> Tensor: ...
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
    def matmul(self, other: Tensor, *, backend: str | None = ...) -> Tensor: ...
    def add(self, other: Tensor) -> Tensor: ...
    def sub(self, other: Tensor) -> Tensor: ...
    def scale(self, value: float) -> Tensor: ...
    def hadamard(self, other: Tensor) -> Tensor: ...
    def add_scaled_(self, other: Tensor, scale: float) -> None: ...
    def add_row_inplace(self, bias: Sequence[float]) -> None: ...
    def transpose(self) -> Tensor: ...
    def reshape(self, rows: int, cols: int) -> Tensor: ...
    def sum_axis0(self) -> List[float]: ...
    def squared_l2_norm(self) -> float: ...
    def project_to_poincare(self, curvature: float) -> Tensor: ...
    def hyperbolic_distance(self, other: Tensor, curvature: float) -> float: ...
    @staticmethod
    def cat_rows(tensors: Sequence[Tensor]) -> Tensor: ...

class ComplexTensor:
    def __init__(self, rows: int, cols: int, data: Optional[Sequence[complex]] = ...) -> None: ...
    @staticmethod
    def zeros(rows: int, cols: int) -> ComplexTensor: ...
    def shape(self) -> tuple[int, int]: ...
    def to_tensor(self) -> Tensor: ...
    def data(self) -> List[complex]: ...
    def matmul(self, other: ComplexTensor) -> ComplexTensor: ...

class OpenCartesianTopos:
    def __init__(
        self,
        curvature: float,
        tolerance: float,
        saturation: float,
        max_depth: int,
        max_volume: int,
    ) -> None: ...
    def curvature(self) -> float: ...
    def tolerance(self) -> float: ...
    def saturation(self) -> float: ...
    def max_depth(self) -> int: ...
    def max_volume(self) -> int: ...
    def ensure_loop_free(self, depth: int) -> None: ...
    def saturate(self, value: float) -> float: ...

class LanguageWaveEncoder:
    def __init__(self, curvature: float, temperature: float) -> None: ...
    def curvature(self) -> float: ...
    def temperature(self) -> float: ...
    def encode_wave(self, text: str) -> ComplexTensor: ...
    def encode_z_space(self, text: str) -> Tensor: ...

class GradientSummary:
    def l1(self) -> float: ...
    def l2(self) -> float: ...
    def linf(self) -> float: ...
    def count(self) -> int: ...
    def mean_abs(self) -> float: ...
    def rms(self) -> float: ...
    def sum_squares(self) -> float: ...

class Hypergrad:
    def __init__(
        self,
        curvature: float,
        learning_rate: float,
        rows: int,
        cols: int,
        topos: OpenCartesianTopos | None = None,
    ) -> None: ...
    def curvature(self) -> float: ...
    def learning_rate(self) -> float: ...
    def shape(self) -> tuple[int, int]: ...
    def gradient(self) -> List[float]: ...
    def summary(self) -> GradientSummary: ...
    def scale_learning_rate(self, factor: float) -> None: ...
    def reset(self) -> None: ...
    def retune(self, curvature: float, learning_rate: float) -> None: ...
    def accumulate_wave(self, tensor: Tensor) -> None: ...
    def accumulate_complex_wave(self, wave: ComplexTensor) -> None: ...
    def absorb_text(self, encoder: LanguageWaveEncoder, text: str) -> None: ...
    def accumulate_pair(self, prediction: Tensor, target: Tensor) -> None: ...
    def apply(self, weights: Tensor) -> None: ...
    def accumulate_barycenter_path(self, intermediates: Sequence[BarycenterIntermediate]) -> None: ...
    def topos(self) -> OpenCartesianTopos: ...
    def accumulate_barycenter_path(
        self, intermediates: Sequence[BarycenterIntermediate]
    ) -> None: ...

class LinearModel:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def forward(
        self,
        inputs: Tensor | Sequence[Sequence[float]],
    ) -> Tensor: ...

    def train_batch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
    ) -> float: ...

    def train_batch_tensor(
        self,
        inputs: Tensor,
        targets: Tensor,
        learning_rate: float = ...,
    ) -> float: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

    def state_dict(self) -> Dict[str, object]: ...

class ModuleTrainer:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def train_epoch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
        batch_size: int = ...,
    ) -> float: ...

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float: ...

class SpiralKFftPlan:
    def __init__(self, radix: int, tile_cols: int, segments: int, subgroup: bool) -> None: ...
    @classmethod
    def from_rank_plan(cls, plan: RankPlan) -> SpiralKFftPlan: ...
    @property
    def radix(self) -> int: ...
    @property
    def tile_cols(self) -> int: ...
    @property
    def segments(self) -> int: ...
    @property
    def subgroup(self) -> bool: ...
    def workgroup_size(self) -> int: ...
    def wgsl(self) -> str: ...
    def spiralk_hint(self) -> str: ...

class MaxwellSpiralKHint:
    @property
    def channel(self) -> str: ...
    @property
    def blocks(self) -> int: ...
    @property
    def z_score(self) -> float: ...
    @property
    def z_bias(self) -> float: ...
    @property
    def weight(self) -> float: ...
    def script_line(self) -> str: ...

class MaxwellSpiralKBridge:
    def __init__(self) -> None: ...
    def set_base_program(self, program: str | None) -> None: ...
    def set_weight_bounds(self, min_weight: float, max_weight: float) -> None: ...
    def is_empty(self) -> bool: ...
    def len(self) -> int: ...
    def push_pulse(
        self,
        channel: str,
        blocks: int,
        mean: float,
        standard_error: float,
        z_score: float,
        band_energy: tuple[float, float, float],
        z_bias: float,
    ) -> MaxwellSpiralKHint: ...
    def hints(self) -> List[MaxwellSpiralKHint]: ...
    def script(self) -> str | None: ...
    def reset(self) -> None: ...

class SpiralKContext:
    def __init__(
        self,
        rows: int,
        cols: int,
        k: int,
        subgroup: bool,
        subgroup_capacity: int,
        kernel_capacity: int,
        tile_cols: int,
        radix: int,
        segments: int,
    ) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    @property
    def k(self) -> int: ...
    @property
    def subgroup(self) -> bool: ...
    @property
    def subgroup_capacity(self) -> int: ...
    @property
    def kernel_capacity(self) -> int: ...
    @property
    def tile_cols(self) -> int: ...
    @property
    def radix(self) -> int: ...
    @property
    def segments(self) -> int: ...

class SpiralKWilsonMetrics:
    def __init__(
        self,
        baseline_latency: float,
        candidate_latency: float,
        wins: int,
        trials: int,
    ) -> None: ...
    @property
    def baseline_latency(self) -> float: ...
    @property
    def candidate_latency(self) -> float: ...
    @property
    def wins(self) -> int: ...
    @property
    def trials(self) -> int: ...
    def gain(self) -> float: ...

class SpiralKHeuristicHint:
    def __init__(self, field: str, value_expr: str, weight: float, condition_expr: str) -> None: ...
    @property
    def field(self) -> str: ...
    @property
    def value_expr(self) -> str: ...
    @property
    def weight_expr(self) -> str: ...
    @property
    def condition_expr(self) -> str: ...

def wilson_lower_bound(wins: int, trials: int, z: float) -> float: ...

def should_rewrite(
    metrics: SpiralKWilsonMetrics,
    min_gain: float = ...,
    min_confidence: float = ...,
) -> bool: ...

def synthesize_program(
    base_src: str,
    hints: Sequence[SpiralKHeuristicHint],
) -> str: ...

def rewrite_with_wilson(
    base_src: str,
    ctx: SpiralKContext,
    metrics: SpiralKWilsonMetrics,
    hints: Sequence[SpiralKHeuristicHint],
    min_gain: float = ...,
    min_confidence: float = ...,
) -> tuple[Dict[str, object], str]: ...

    def predict(self, inputs: Sequence[Sequence[float]]) -> Tensor: ...

    def predict_tensor(self, inputs: Tensor) -> Tensor: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

class LinearModel:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def forward(
        self,
        inputs: Tensor | Sequence[Sequence[float]],
    ) -> Tensor: ...

    def train_batch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
    ) -> float: ...

    def train_batch_tensor(
        self,
        inputs: Tensor,
        targets: Tensor,
        learning_rate: float = ...,
    ) -> float: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

    def state_dict(self) -> Dict[str, object]: ...

class ModuleTrainer:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def train_epoch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
        batch_size: int = ...,
    ) -> float: ...

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float: ...

    def predict(self, inputs: Sequence[Sequence[float]]) -> Tensor: ...

    def predict_tensor(self, inputs: Tensor) -> Tensor: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

class LinearModel:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def forward(
        self,
        inputs: Tensor | Sequence[Sequence[float]],
    ) -> Tensor: ...

    def train_batch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
    ) -> float: ...

    def train_batch_tensor(
        self,
        inputs: Tensor,
        targets: Tensor,
        learning_rate: float = ...,
    ) -> float: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

    def state_dict(self) -> Dict[str, object]: ...

class ModuleTrainer:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def train_epoch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
        batch_size: int = ...,
    ) -> float: ...

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float: ...

    def predict(self, inputs: Sequence[Sequence[float]]) -> Tensor: ...

    def predict_tensor(self, inputs: Tensor) -> Tensor: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

class LinearModel:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def forward(
        self,
        inputs: Tensor | Sequence[Sequence[float]],
    ) -> Tensor: ...

    def train_batch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
    ) -> float: ...

    def train_batch_tensor(
        self,
        inputs: Tensor,
        targets: Tensor,
        learning_rate: float = ...,
    ) -> float: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

    def state_dict(self) -> Dict[str, object]: ...

class ModuleTrainer:
    def __init__(self, input_dim: int, output_dim: int) -> None: ...

    def train_epoch(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        learning_rate: float = ...,
        batch_size: int = ...,
    ) -> float: ...

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float: ...

    def predict(self, inputs: Sequence[Sequence[float]]) -> Tensor: ...

    def predict_tensor(self, inputs: Tensor) -> Tensor: ...

    def weights(self) -> Tensor: ...

    def bias(self) -> List[float]: ...

    def input_dim(self) -> int: ...

    def output_dim(self) -> int: ...

class TensorBiome:
    def __init__(self, topos: OpenCartesianTopos) -> None: ...
    def topos(self) -> OpenCartesianTopos: ...
    def len(self) -> int: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def total_weight(self) -> float: ...
    def weights(self) -> List[float]: ...
    def absorb(self, tensor: Tensor) -> None: ...
    def absorb_weighted(self, tensor: Tensor, weight: float) -> None: ...
    def clear(self) -> None: ...
    def canopy(self) -> Tensor: ...

class BarycenterIntermediate:
    @property
    def interpolation(self) -> float: ...
    @property
    def density(self) -> Tensor: ...
    @property
    def kl_energy(self) -> float: ...
    @property
    def entropy(self) -> float: ...
    @property
    def objective(self) -> float: ...

class ZSpaceBarycenter:
    @property
    def density(self) -> Tensor: ...
    @property
    def kl_energy(self) -> float: ...
    @property
    def entropy(self) -> float: ...
    @property
    def coupling_energy(self) -> float: ...
    @property
    def objective(self) -> float: ...
    @property
    def effective_weight(self) -> float: ...
    def intermediates(self) -> List[BarycenterIntermediate]: ...

class ZMetrics:
    speed: float
    memory: float
    stability: float
    gradient: Optional[Sequence[float]]
    drs: float

class ZSpaceTrainer:
    def __init__(
        self,
        z_dim: int = ...,
        *,
        alpha: float = ...,
        lam_speed: float = ...,
        lam_mem: float = ...,
        lam_stab: float = ...,
        lam_frac: float = ...,
        lam_drs: float = ...,
        lr: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        eps: float = ...,
    ) -> None: ...
    @property
    def state(self) -> List[float]: ...
    def step(self, metrics: Mapping[str, float] | ZMetrics) -> float: ...
    def reset(self) -> None: ...
    def state_dict(self) -> Dict[str, object]: ...
    def load_state_dict(self, state: Dict[str, object], *, strict: bool = ...) -> None: ...
    def step_batch(self, metrics: Iterable[Mapping[str, float] | ZMetrics]) -> List[float]: ...

def step_many(trainer: ZSpaceTrainer, samples: Iterable[Mapping[str, float] | ZMetrics]) -> List[float]: ...

def stream_zspace_training(
    trainer: ZSpaceTrainer,
    samples: Iterable[Mapping[str, float] | ZMetrics],
    *,
    on_step: Optional[Callable[[int, List[float], float], None]] = ...,
) -> List[float]: ...

class RankPlan:
    kind: str
    rows: int
    cols: int
    k: int
    workgroup: int
    lanes: int
    channel_stride: int
    merge_strategy: str
    merge_detail: str
    use_two_stage: bool
    subgroup: bool
    tile: int
    compaction_tile: int
    fft_tile: int
    fft_radix: int
    fft_segments: int

    def latency_window(self) -> Optional[Tuple[int, int, int, int, int, int, int]]: ...
    def to_unison_script(self) -> str: ...
    def fft_wgsl(self) -> str: ...
    def fft_spiralk_hint(self) -> str: ...

def from_dlpack(capsule: object) -> Tensor: ...

def to_dlpack(tensor: Tensor) -> object: ...

def z_space_barycenter(
    weights: Sequence[float],
    densities: Sequence[Tensor],
    entropy_weight: float,
    beta_j: float,
    coupling: Tensor | None = ...,
) -> ZSpaceBarycenter: ...

def plan(
    kind: str,
    rows: int,
    cols: int,
    k: int,
    *,
    backend: Optional[str] = ...,
    lane_width: Optional[int] = ...,
    subgroup: Optional[bool] = ...,
    max_workgroup: Optional[int] = ...,
    shared_mem_per_workgroup: Optional[int] = ...,
) -> RankPlan: ...

def plan_topk(
    rows: int,
    cols: int,
    k: int,
    *,
    backend: Optional[str] = ...,
    lane_width: Optional[int] = ...,
    subgroup: Optional[bool] = ...,
    max_workgroup: Optional[int] = ...,
    shared_mem_per_workgroup: Optional[int] = ...,
) -> RankPlan: ...

class SpiralSession:
    backend: str
    seed: int | None
    device: str

    def __init__(self, backend: str = ..., seed: int | None = ...) -> None: ...

    def plan_topk(self, rows: int, cols: int, k: int) -> RankPlan: ...

    def close(self) -> None: ...

def describe_device(
    backend: str = ...,
    *,
    lane_width: Optional[int] = ...,
    subgroup: Optional[bool] = ...,
    max_workgroup: Optional[int] = ...,
    shared_mem_per_workgroup: Optional[int] = ...,
    workgroup: Optional[int] = ...,
    cols: Optional[int] = ...,
    tile_hint: Optional[int] = ...,
    compaction_hint: Optional[int] = ...,
) -> Dict[str, object]: ...

def hip_probe() -> Dict[str, object]: ...

def gl_coeffs_adaptive(alpha: float, tol: float = ..., max_len: int = ...) -> List[float]: ...

def fracdiff_gl_1d(
    xs: Sequence[float],
    alpha: float,
    kernel_len: int,
    pad: str = ...,
    pad_constant: Optional[float] = ...,
) -> List[float]: ...

def mean_squared_error(predictions: Tensor, targets: Tensor) -> float: ...

def info_nce(
    anchors: Sequence[Sequence[float]],
    positives: Sequence[Sequence[float]],
    temperature: float = ...,
    normalize: bool = ...,
) -> Dict[str, object]: ...

def masked_mse(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    mask_indices: Sequence[Sequence[int]],
) -> Dict[str, object]: ...

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

class TemporalResonanceBuffer:
    def __init__(self, capacity: int = ..., alpha: float = ...) -> None: ...
    @property
    def alpha(self) -> float: ...
    @property
    def capacity(self) -> int: ...
    def update(self, volume: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]: ...
    def state(self) -> Optional[List[List[List[float]]]]: ...
    def history(self) -> List[List[List[List[float]]]]: ...
    def state_dict(self) -> Dict[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...

class SliceProfile:
    mean: float
    std: float
    energy: float

class SpiralTorchVision:
    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        *,
        alpha: float = ...,
        window: Optional[str] = ...,
        temporal: int = ...,
    ) -> None: ...
    @property
    def volume(self) -> List[List[List[float]]]: ...
    @property
    def alpha(self) -> float: ...
    @property
    def temporal_capacity(self) -> int: ...
    @property
    def temporal_state(self) -> Optional[List[List[List[float]]]]: ...
    @property
    def window(self) -> List[float]: ...
    def reset(self) -> None: ...
    def update_window(self, window: Optional[str] | Sequence[float]) -> None: ...
    def accumulate(self, volume: Sequence[Sequence[Sequence[float]]], weight: float = ...) -> None: ...
    def accumulate_slices(self, slices: Sequence[Sequence[Sequence[float]]]) -> None: ...
    def accumulate_sequence(
        self,
        frames: Iterable[Sequence[Sequence[Sequence[float]]]],
        weights: Optional[Sequence[float]] = ...,
    ) -> None: ...
    def project(self, *, normalise: bool = ...) -> List[List[float]]: ...
    def volume_energy(self) -> float: ...
    def slice_profile(self) -> List[SliceProfile]: ...
    def snapshot(self) -> Dict[str, object]: ...
    def state_dict(self) -> Dict[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object], *, strict: bool = ...) -> None: ...

class CanvasTransformer:
    def __init__(self, width: int, height: int, *, smoothing: float = ...) -> None: ...
    @property
    def smoothing(self) -> float: ...
    def refresh(self, projection: Sequence[Sequence[float]]) -> List[List[float]]: ...
    def accumulate_hypergrad(self, gradient: Sequence[Sequence[float]]) -> None: ...
    def accumulate_realgrad(self, gradient: Sequence[Sequence[float]]) -> None: ...
    def reset(self) -> None: ...
    def gradient_summary(self) -> Dict[str, Dict[str, float]]: ...
    def emit_zspace_patch(self, vision: SpiralTorchVision, weight: float = ...) -> List[List[float]]: ...
    def canvas(self) -> List[List[float]]: ...
    def hypergrad(self) -> List[List[float]]: ...
    def realgrad(self) -> List[List[float]]: ...
    def state_dict(self) -> Dict[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object], *, strict: bool = ...) -> None: ...
    def snapshot(self) -> CanvasSnapshot: ...

class CanvasSnapshot:
    canvas: List[List[float]]
    hypergrad: List[List[float]]
    realgrad: List[List[float]]
    summary: Dict[str, Dict[str, float]]
    patch: Optional[List[List[float]]]

def apply_vision_update(
    vision: SpiralTorchVision,
    canvas: CanvasTransformer,
    *,
    hypergrad: Optional[Sequence[Sequence[float]]] = ...,
    realgrad: Optional[Sequence[Sequence[float]]] = ...,
    weight: float = ...,
    include_patch: bool = ...,
) -> CanvasSnapshot: ...

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

class CoherenceChannelReport:
    channel: int
    weight: float
    backend: str
    dominant_concept: str | None
    emphasis: float
    descriptor: str | None


class CoherenceDiagnostics:
    channel_weights: List[float]
    normalized_weights: List[float]
    normalization: float
    fractional_order: float
    dominant_channel: int | None
    mean_coherence: float
    z_bias: float
    energy_ratio: float
    coherence_entropy: float
    aggregated: Tensor
    coherence: List[float]
    channel_reports: List[CoherenceChannelReport]
    preserved_channels: int
    discarded_channels: int
    pre_discard: PreDiscardTelemetry | None


class PreDiscardTelemetry:
    dominance_ratio: float
    energy_floor: float
    discarded: int
    preserved: int
    used_fallback: bool


class PreDiscardPolicy:
    def __init__(
        self,
        dominance_ratio: float,
        *,
        energy_floor: float | None = ...,
        min_channels: int | None = ...,
    ) -> None: ...

    dominance_ratio: float
    energy_floor: float
    min_channels: int


class _ZSpaceCoherenceSequencer:
    def __init__(
        self,
        dim: int,
        num_heads: int,
        curvature: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...

    def forward_with_coherence(self, x: Tensor) -> Tuple[Tensor, List[float]]: ...

    def forward_with_diagnostics(
        self, x: Tensor
    ) -> Tuple[Tensor, List[float], CoherenceDiagnostics]: ...

    def project_to_zspace(self, x: Tensor) -> Tensor: ...

    def configure_pre_discard(
        self,
        dominance_ratio: float,
        *,
        energy_floor: float | None = ...,
        min_channels: int | None = ...,
    ) -> None: ...

    def disable_pre_discard(self) -> None: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def dim(self) -> int: ...

    def num_heads(self) -> int: ...

    def pre_discard_policy(self) -> PreDiscardPolicy | None: ...

    def curvature(self) -> float: ...

    def maxwell_channels(self) -> int: ...

    def topos(self) -> OpenCartesianTopos: ...


class _NnModule(ModuleType):
    Dataset: type[_NnDataset]
    DataLoader: type[_NnDataLoader]
    DataLoaderIter: type[_NnDataLoaderIter]
    CoherenceDiagnostics: type[CoherenceDiagnostics]
    ZSpaceCoherenceSequencer: type[_ZSpaceCoherenceSequencer]

    def from_samples(samples: Sequence[Tuple[Tensor, Tensor]]) -> _NnDataLoader: ...


nn: _NnModule


class ZSpaceCoherenceSequencer(_ZSpaceCoherenceSequencer):
    ...


class _FracModule(ModuleType):
    def gl_coeffs_adaptive(alpha: float, tol: float = ..., max_len: int = ...) -> List[float]: ...

    def fracdiff_gl_1d(
        xs: Sequence[float],
        alpha: float,
        kernel_len: int,
        pad: str = ...,
        pad_constant: Optional[float] = ...,
    ) -> List[float]: ...


frac: _FracModule

dataset: ModuleType

linalg: ModuleType

spiral_rl: ModuleType

rec: ModuleType

telemetry: ModuleType

ecosystem: ModuleType

class _ZSpaceModule(ModuleType):
    ZMetrics: type[ZMetrics]
    ZSpaceTrainer: type[ZSpaceTrainer]
    step_many: staticmethod
    stream_zspace_training: staticmethod

zspace: _ZSpaceModule

class _VisionModule(ModuleType):
    SpiralTorchVision: type[SpiralTorchVision]
    TemporalResonanceBuffer: type[TemporalResonanceBuffer]
    SliceProfile: type[SliceProfile]

vision: _VisionModule

class _CanvasModule(ModuleType):
    CanvasTransformer: type[CanvasTransformer]
    CanvasSnapshot: type[CanvasSnapshot]

    def apply_vision_update(
        vision: SpiralTorchVision,
        canvas: CanvasTransformer,
        *,
        hypergrad: Sequence[Sequence[float]] | None = ...,
        realgrad: Sequence[Sequence[float]] | None = ...,
        weight: float = ...,
        include_patch: bool = ...,
    ) -> CanvasSnapshot: ...

canvas: _CanvasModule

class _SelfSupModule(ModuleType):
    def info_nce(
        anchors: Sequence[Sequence[float]],
        positives: Sequence[Sequence[float]],
        temperature: float = ...,
        normalize: bool = ...,
    ) -> Dict[str, object]: ...

    def masked_mse(
        predictions: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        mask_indices: Sequence[Sequence[int]],
    ) -> Dict[str, object]: ...

selfsup: _SelfSupModule

class _PlannerModule(ModuleType):
    RankPlan: type[RankPlan]

    def plan(
        kind: str,
        rows: int,
        cols: int,
        k: int,
        *,
        backend: Optional[str] = ...,
        lane_width: Optional[int] = ...,
        subgroup: Optional[bool] = ...,
        max_workgroup: Optional[int] = ...,
        shared_mem_per_workgroup: Optional[int] = ...,
    ) -> RankPlan: ...

    def plan_topk(
        rows: int,
        cols: int,
        k: int,
        *,
        backend: Optional[str] = ...,
        lane_width: Optional[int] = ...,
        subgroup: Optional[bool] = ...,
        max_workgroup: Optional[int] = ...,
        shared_mem_per_workgroup: Optional[int] = ...,
    ) -> RankPlan: ...

    def describe_device(
        backend: str = ...,
        *,
        lane_width: Optional[int] = ...,
        subgroup: Optional[bool] = ...,
        max_workgroup: Optional[int] = ...,
        shared_mem_per_workgroup: Optional[int] = ...,
        workgroup: Optional[int] = ...,
        cols: Optional[int] = ...,
        tile_hint: Optional[int] = ...,
        compaction_hint: Optional[int] = ...,
    ) -> Dict[str, object]: ...

    def hip_probe() -> Dict[str, object]: ...
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

planner: _PlannerModule

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
    def __init__(
        self,
        users: int,
        items: int,
        factors: int,
        learning_rate: float = ...,
        regularization: float = ...,
        curvature: float | None = ...,
    ) -> None: ...
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

class stAgent:
    def __init__(self, state_dim: int, action_dim: int, discount: float, learning_rate: float) -> None: ...
    def select_action(self, state: int) -> int: ...
    def update(self, state: int, action: int, reward: float, next_state: int) -> None: ...

DqnAgent = stAgent

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
    "ModuleTrainer",
    "SpiralSession",
    "from_dlpack",
    "to_dlpack",
    "ZSpaceBarycenter",
    "BarycenterIntermediate",
    "z_space_barycenter",
    "ZMetrics",
    "ZSpaceTrainer",
    "ZSpaceCoherenceSequencer",
    "step_many",
    "stream_zspace_training",
    "compat",
    "capture",
    "share",
    "from_dlpack",
    "to_dlpack",
    "nn",
    "frac",
    "dataset",
    "linalg",
    "spiral_rl",
    "rec",
    "telemetry",
    "ecosystem",
    "selfsup",
    "planner",
    "zspace",
    "vision",
    "canvas",
    "compat",
    "set_global_seed",
    "golden_ratio",
    "golden_angle",
    "fibonacci_pacing",
    "pack_nacci_chunks",
    "pack_tribonacci_chunks",
    "pack_tetranacci_chunks",
    "generate_plan_batch_ex",
    "info_nce",
    "masked_mse",
    "gl_coeffs_adaptive",
    "fracdiff_gl_1d",
    "QueryPlan",
    "RecEpochReport",
    "Recommender",
    "stAgent",
    "DqnAgent",
    "PpoAgent",
    "SacAgent",
    "TemporalResonanceBuffer",
    "SpiralTorchVision",
    "SliceProfile",
    "CanvasTransformer",
    "CanvasSnapshot",
    "apply_vision_update",
    "DashboardMetric",
    "DashboardEvent",
    "DashboardFrame",
    "DashboardRing",
]
