# NOTE: This stub lives next to the runtime package as `spiraltorch/__init__.pyi`,
# so it is shipped in the published wheel (PEP 561).
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, overload
from types import ModuleType

from .optim import Amegagrad, amegagrad

def init_backend(backend: str) -> bool: ...

def load_zspace_trace_events(path: str, *, event_type: str = ...) -> List[Dict[str, Any]]: ...

def write_zspace_trace_html(
    trace_jsonl: str,
    html_path: str | None = ...,
    *,
    title: str = ...,
    event_type: str = ...,
) -> str: ...

def zspace_trace_event_to_atlas_frame(
    event: Mapping[str, Any],
    *,
    district: str = ...,
    timestamp_base: float | None = ...,
    step_seconds: float = ...,
) -> "telemetry.AtlasFrame" | None: ...

def zspace_trace_to_atlas_route(
    trace: str | Iterable[Mapping[str, Any]],
    *,
    district: str = ...,
    bound: int = ...,
    event_type: str = ...,
    timestamp_base: float | None = ...,
    step_seconds: float = ...,
) -> "telemetry.AtlasRoute": ...

class ZSpaceTraceLiveServer:
    url: str
    event_type: str
    record_jsonl: str | None
    def close(self) -> None: ...
    def join(self, timeout: float | None = ...) -> None: ...

def serve_zspace_trace(
    *,
    event_type: str = ...,
    host: str = ...,
    port: int = ...,
    title: str = ...,
    maxlen: int = ...,
    poll_interval: float = ...,
    max_batch: int = ...,
    buffer: int = ...,
    record_jsonl: str | None = ...,
    open_browser: bool = ...,
    background: bool = ...,
) -> ZSpaceTraceLiveServer: ...

class Axis:
    name: str
    size: int | None

    def __init__(self, name: str, size: int | None = ...) -> None: ...
    def with_size(self, size: int) -> Axis: ...


class LabeledTensor:
    tensor: Tensor
    axes: tuple[Axis, Axis]

    def __init__(self, data: Tensor | Sequence[Sequence[float]] | Sequence[float], axes: Sequence[Axis | str]) -> None: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    def to_tensor(self) -> Tensor: ...
    def tolist(self) -> List[List[float]]: ...
    def rename(self, axes: Sequence[Axis | str]) -> LabeledTensor: ...
    def with_axes(self, axes: Sequence[Axis | str]) -> LabeledTensor: ...
    def transpose(self) -> LabeledTensor: ...
    def row_softmax(self, *, backend: str | None = ...) -> LabeledTensor: ...
    def describe(self) -> Dict[str, object]: ...
    def axis_names(self) -> tuple[str, str]: ...
    def __matmul__(self, other: LabeledTensor) -> LabeledTensor: ...


@overload
def tensor(data: Sequence[Sequence[float]] | Sequence[float]) -> Tensor: ...


@overload
def tensor(data: Sequence[Sequence[float]] | Sequence[float] | Tensor, *, axes: Sequence[Axis | str]) -> LabeledTensor: ...


def label_tensor(tensor_obj: Tensor | Sequence[Sequence[float]] | Sequence[float], axes: Sequence[Axis | str]) -> LabeledTensor: ...


class ScaleStack:
    @property
    def threshold(self) -> float: ...
    @property
    def mode(self) -> str: ...
    def samples(self) -> List[Tuple[float, float]]: ...
    def persistence(self) -> List[Tuple[float, float, float]]: ...
    def interface_density(self) -> Optional[float]: ...
    def moment(self, order: int) -> float: ...
    def boundary_dimension(self, ambient_dim: float, window: int) -> Optional[float]: ...
    def coherence_break_scale(self, level: float) -> Optional[float]: ...
    def coherence_profile(self, levels: Sequence[float]) -> List[Optional[float]]: ...

def scalar_scale_stack(
    field: Sequence[float],
    shape: Sequence[int],
    scales: Sequence[float],
    threshold: float,
) -> ScaleStack: ...

def semantic_scale_stack(
    embeddings: Sequence[Sequence[float]],
    scales: Sequence[float],
    threshold: float,
    metric: str = ...,
) -> ScaleStack: ...

class Tensor:
    def __init__(self, *args: object, **kwargs: object) -> None: ...
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
    def row_softmax(self, *, backend: str | None = ...) -> Tensor: ...
    def scaled_dot_attention(
        self,
        keys: Tensor,
        values: Tensor,
        *,
        contexts: int,
        sequence: int,
        scale: float,
        z_bias: Tensor | None = ...,
        attn_bias: Tensor | None = ...,
        backend: str | None = ...,
    ) -> Tensor: ...
    def layer_norm_affine(
        self,
        gamma: Tensor,
        beta: Tensor,
        *,
        epsilon: float = ...,
    ) -> Tensor: ...
    def layer_norm_affine_add(
        self,
        residual: Tensor,
        gamma: Tensor,
        beta: Tensor,
        *,
        epsilon: float = ...,
    ) -> Tensor: ...
    def add(self, other: Tensor) -> Tensor: ...
    def sub(self, other: Tensor) -> Tensor: ...
    def scale(self, value: float) -> Tensor: ...
    def hadamard(self, other: Tensor) -> Tensor: ...
    def add_scaled_(self, other: Tensor, scale: float) -> None: ...
    def add_row_inplace(self, bias: Sequence[float]) -> None: ...
    def transpose(self) -> Tensor: ...
    def reshape(self, rows: int, cols: int) -> Tensor: ...
    def sum_axis0(self) -> List[float]: ...
    def sum_axis1(self) -> List[float]: ...
    def squared_l2_norm(self) -> float: ...
    def project_to_poincare(self, curvature: float) -> Tensor: ...
    def hyperbolic_distance(self, other: Tensor, curvature: float) -> float: ...
    @staticmethod
    def cat_rows(tensors: Sequence[Tensor]) -> Tensor: ...

def lorentzian_metric_scaled(
    components: Sequence[Sequence[float]],
    scale: float,
) -> Dict[str, object]: ...

def assemble_zrelativity_model(
    base_metric: Sequence[Sequence[float]],
    internal_metric: Sequence[Sequence[float]],
    *,
    mixed: Sequence[Sequence[float]] | None = ...,
    warp: float | None = ...,
    first_derivatives: Sequence[Sequence[Sequence[float]]] | None = ...,
    second_derivatives: Sequence[Sequence[Sequence[Sequence[float]]]] | None = ...,
    gravitational_constant: float,
    speed_of_light: float,
    internal_volume: float,
    cosmological_constant: float = ...,
    symmetry: str | None = ...,
    topology: str | None = ...,
    boundary_conditions: Sequence[str] | None = ...,
) -> ZRelativityModel: ...

class ZRelativityModel:
    def as_tensor(self) -> Tensor: ...
    def to_dlpack(self) -> object: ...
    def effective_metric(self) -> Tensor: ...
    def gauge_tensor(self) -> Tensor: ...
    def scalar_moduli(self) -> Tensor: ...
    def field_equations(self) -> Tensor: ...
    def tensor_bundle(self) -> Dict[str, object]: ...
    def torch_bundle(self) -> Dict[str, object]: ...
    def reduction_summary(self) -> Dict[str, object]: ...
    def curvature_diagnostics(self) -> Dict[
        str,
        float | complex | List[List[complex]] | List[complex],
    ]: ...
    def learnable_flags(self) -> Tuple[bool, bool, bool]: ...
    def warp_scale(self) -> float | None: ...
    def internal_volume_density(self) -> float: ...
    def field_prefactor(self) -> float: ...
    def total_dimension(self) -> int: ...

class ZRelativityModule:
    def __init__(self, model: ZRelativityModel) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...
    def parameter_tensor(self) -> Tensor: ...
    def torch_parameters(self) -> object: ...
    def parameter_dimension(self) -> int: ...
    def model(self) -> ZRelativityModel: ...

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
    def porosity(self) -> float: ...
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
    @staticmethod
    def from_values(values: Sequence[float]) -> GradientSummary: ...
    def l1(self) -> float: ...
    def l2(self) -> float: ...
    def linf(self) -> float: ...
    def count(self) -> int: ...
    def sum(self) -> float: ...
    def mean_abs(self) -> float: ...
    def rms(self) -> float: ...
    def sum_squares(self) -> float: ...
    def sum_cubes(self) -> float: ...
    def sum_quartic(self) -> float: ...
    def mean(self) -> float: ...
    def min(self) -> float: ...
    def max(self) -> float: ...
    def support_width(self) -> float: ...
    def positive_count(self) -> int: ...
    def negative_count(self) -> int: ...
    def zero_count(self) -> int: ...
    def near_zero_count(self) -> int: ...
    def positive_fraction(self) -> float: ...
    def negative_fraction(self) -> float: ...
    def zero_fraction(self) -> float: ...
    def near_zero_fraction(self) -> float: ...
    def activation(self) -> float: ...
    def sign_lean(self) -> float: ...
    def sign_entropy(self) -> float: ...
    def variance(self) -> float: ...
    def std(self) -> float: ...
    def skewness(self) -> float: ...
    def kurtosis(self) -> float: ...

class HypergradTelemetry:
    def summary(self) -> GradientSummary: ...
    def curvature(self) -> float: ...
    def learning_rate(self) -> float: ...
    def saturation(self) -> float: ...
    def porosity(self) -> float: ...
    def tolerance(self) -> float: ...
    def max_depth(self) -> int: ...
    def max_volume(self) -> int: ...
    def shape(self) -> tuple[int, int]: ...
    def volume(self) -> int: ...
    def finite_count(self) -> int: ...
    def non_finite_count(self) -> int: ...
    def non_finite_ratio(self) -> float: ...

class DesireGradientInterpretation:
    def hyper_pressure(self) -> float: ...
    def real_pressure(self) -> float: ...
    def balance(self) -> float: ...
    def stability(self) -> float: ...
    def saturation(self) -> float: ...
    def hyper_std(self) -> float: ...
    def real_std(self) -> float: ...
    def sharpness(self) -> float: ...
    def penalty_gain(self) -> float: ...
    def activation(self) -> float: ...
    def sign_alignment(self) -> float: ...
    def sign_entropy(self) -> float: ...

class DesireGradientControl:
    def penalty_gain(self) -> float: ...
    def bias_mix(self) -> float: ...
    def observation_gain(self) -> float: ...
    def damping(self) -> float: ...
    def hyper_rate_scale(self) -> float: ...
    def real_rate_scale(self) -> float: ...
    def operator_mix(self) -> float: ...
    def operator_gain(self) -> float: ...
    def tuning_gain(self) -> float: ...
    def target_entropy(self) -> float: ...
    def learning_rate_eta(self) -> float: ...
    def learning_rate_min(self) -> float: ...
    def learning_rate_max(self) -> float: ...
    def learning_rate_slew(self) -> float: ...
    def clip_norm(self) -> float: ...
    def clip_floor(self) -> float: ...
    def clip_ceiling(self) -> float: ...
    def clip_ema(self) -> float: ...
    def temperature_kappa(self) -> float: ...
    def temperature_slew(self) -> float: ...
    def quality_gain(self) -> float: ...
    def quality_bias(self) -> float: ...
    def events(self) -> List[str]: ...

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
    def finite_count(self) -> int: ...
    def non_finite_count(self) -> int: ...
    def non_finite_ratio(self) -> float: ...
    def has_non_finite(self) -> bool: ...
    def telemetry(self) -> HypergradTelemetry: ...
    def scale_learning_rate(self, factor: float) -> None: ...
    def scale_gradient(self, factor: float) -> None: ...
    def rescale_rms(self, target_rms: float) -> float: ...
    def reset(self) -> None: ...
    def retune(self, curvature: float, learning_rate: float) -> None: ...
    def accumulate_wave(self, tensor: Tensor) -> None: ...
    def accumulate_complex_wave(self, wave: ComplexTensor) -> None: ...
    def absorb_text(self, encoder: LanguageWaveEncoder, text: str) -> None: ...
    def accumulate_pair(self, prediction: Tensor, target: Tensor) -> None: ...
    def apply(self, weights: Tensor) -> None: ...
    def accumulate_barycenter_path(self, intermediates: Sequence[BarycenterIntermediate]) -> None: ...
    def topos(self) -> OpenCartesianTopos: ...
    def desire_control(
        self, real: GradientSummary, gain: float | None = ...
    ) -> DesireGradientControl: ...
    def desire_interpretation(
        self, real: GradientSummary
    ) -> DesireGradientInterpretation: ...


class Realgrad:
    def __init__(self, learning_rate: float, rows: int, cols: int) -> None: ...
    def learning_rate(self) -> float: ...
    def shape(self) -> tuple[int, int]: ...
    def gradient(self) -> List[float]: ...
    def summary(self) -> GradientSummary: ...
    def scale_learning_rate(self, factor: float) -> None: ...
    def reset(self) -> None: ...
    def accumulate_wave(self, tensor: Tensor) -> None: ...
    def accumulate_complex_wave(self, wave: ComplexTensor) -> None: ...
    def absorb_text(self, encoder: LanguageWaveEncoder, text: str) -> None: ...
    def accumulate_pair(self, prediction: Tensor, target: Tensor) -> None: ...
    def apply(self, weights: Tensor) -> None: ...

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

    def train_zrelativity_step(
        self,
        module: ZRelativityModule,
        targets: Tensor,
        learning_rate: float,
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

class SpiralKAiRewriteConfig:
    def __init__(
        self,
        model: str,
        max_hints: int | None = ...,
        default_weight: float | None = ...,
        eta_floor: float | None = ...,
    ) -> None: ...
    @property
    def model(self) -> str: ...
    @property
    def max_hints(self) -> int: ...
    @property
    def default_weight(self) -> float: ...
    @property
    def eta_floor(self) -> float: ...

class SpiralKAiRewritePrompt:
    def __init__(
        self,
        base_program: str,
        ctx: SpiralKContext,
        eta_bar: float = ...,
        device_guard: str | None = ...,
    ) -> None: ...
    def set_metrics(self, metrics: SpiralKWilsonMetrics) -> None: ...
    def clear_metrics(self) -> None: ...
    def add_note(self, note: str) -> None: ...
    @property
    def base_program(self) -> str: ...
    @property
    def eta_bar(self) -> float: ...
    @property
    def device_guard(self) -> str | None: ...

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

def rewrite_with_ai(
    base_src: str,
    ctx: SpiralKContext,
    config: SpiralKAiRewriteConfig,
    prompt: SpiralKAiRewritePrompt,
    generator: Callable[
        [SpiralKAiRewriteConfig, SpiralKAiRewritePrompt],
        Sequence[SpiralKHeuristicHint],
    ]
    | None = ...,
) -> tuple[Dict[str, object], str, Sequence[SpiralKHeuristicHint]]: ...

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
    telemetry: Optional[Mapping[str, float]]

class ZSpaceDecoded:
    z_state: Tuple[float, ...]
    metrics: Mapping[str, float]
    gradient: Tuple[float, ...]
    barycentric: Tuple[float, float, float]
    energy: float
    frac_energy: float

    def as_dict(self) -> Dict[str, object]: ...


class ZSpaceInference:
    metrics: Mapping[str, float]
    gradient: Tuple[float, ...]
    barycentric: Tuple[float, float, float]
    residual: float
    confidence: float
    prior: ZSpaceDecoded
    applied: Mapping[str, object]
    telemetry: Optional[ZSpaceTelemetryFrame]

    def as_dict(self) -> Dict[str, object]: ...


class ZSpacePosterior:
    def __init__(self, z_state: Sequence[float], *, alpha: float = ...) -> None: ...

    @classmethod
    def from_state(
        cls,
        source: object,
        *,
        alpha: float | None = ...,
    ) -> ZSpacePosterior: ...

    @property
    def z_state(self) -> List[float]: ...

    @property
    def alpha(self) -> float: ...

    def decode(self) -> ZSpaceDecoded: ...

    def project(
        self,
        partial: Mapping[str, object] | None,
        *,
        smoothing: float = ...,
    ) -> ZSpaceInference: ...


def decode_zspace_embedding(
    z_state: Sequence[float] | ZSpacePosterior | object,
    *,
    alpha: float = ...,
) -> ZSpaceDecoded: ...


def infer_from_partial(
    z_state: Sequence[float] | ZSpacePosterior | object,
    partial: Mapping[str, object] | None,
    *,
    alpha: float = ...,
    smoothing: float = ...,
) -> ZSpaceInference: ...


def inference_to_mapping(
    inference: ZSpaceInference | Mapping[str, object] | ZMetrics,
    *,
    prefer_applied: bool = ...,
    canonical: bool = ...,
    include_gradient: bool = ...,
) -> Dict[str, object]: ...


def inference_to_zmetrics(
    inference: ZSpaceInference | Mapping[str, object] | ZMetrics,
    *,
    prefer_applied: bool = ...,
    include_telemetry: bool = ...,
) -> ZMetrics: ...


def prepare_trainer_step_payload(
    trainer: object,
    inference: ZSpaceInference,
    *,
    payload: None | str | Callable[[ZSpaceInference], object] = ...,
    prefer_applied: bool = ...,
    canonical_mapping: bool = ...,
) -> object: ...


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
    def spiralk_context(self) -> SpiralKContext: ...
    def rewrite_with_spiralk(self, script: str) -> RankPlan: ...

def from_dlpack(capsule: object) -> Tensor: ...

def to_dlpack(tensor: Tensor) -> object: ...

def z_space_barycenter(
    weights: Sequence[float],
    densities: Sequence[Tensor],
    entropy_weight: float,
    beta_j: float,
    coupling: Tensor | None = ...,
) -> ZSpaceBarycenter: ...

def zspace_eval(
    real: Sequence[float],
    imag: Sequence[float],
    z_re: Sequence[float],
    z_im: Sequence[float],
) -> List[Tuple[float, float]]: ...

def plan(
    kind: Literal["topk", "midk", "bottomk", "fft"],
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

    def dataset(
        self,
        samples: Optional[Iterable[Tuple[object, object]]] = ...,
    ) -> "dataset.Dataset": ...

    def dataloader(
        self,
        samples: object,
        *,
        batch_size: Optional[int] = ...,
        shuffle: bool | int = ...,
        seed: Optional[int] = ...,
        prefetch: Optional[int] = ...,
        max_rows: Optional[int] = ...,
    ) -> "dataset.DataLoader": ...

    def plan_topk(self, rows: int, cols: int, k: int) -> RankPlan: ...

    def close(self) -> None: ...

def hypergrad(
    *shape_args: Any,
    curvature: float = ...,
    learning_rate: float = ...,
    shape: Any | None = ...,
    rows: Any | None = ...,
    cols: Any | None = ...,
    topos: Any | None = ...,
) -> Hypergrad: ...

def realgrad(
    *shape_args: Any,
    learning_rate: float = ...,
    shape: Any | None = ...,
    rows: Any | None = ...,
    cols: Any | None = ...,
) -> Realgrad: ...

def hypergrad_topos(
    *,
    curvature: float = ...,
    tolerance: float = ...,
    saturation: float = ...,
    max_depth: int = ...,
    max_volume: int = ...,
) -> OpenCartesianTopos: ...

def encode_zspace(
    text: str,
    *,
    curvature: float = ...,
    temperature: float = ...,
    encoder: LanguageWaveEncoder | None = ...,
) -> Tensor: ...

class HypergradSession(SpiralSession):
    hyper: Hypergrad
    weights: Tensor
    route: AtlasRoute | None

    def __init__(
        self,
        *shape_args: Any,
        curvature: float = ...,
        learning_rate: float = ...,
        backend: str = ...,
        seed: int | None = ...,
        shape: Any | None = ...,
        rows: Any | None = ...,
        cols: Any | None = ...,
        topos: Any | None = ...,
        weights: Any | None = ...,
        telemetry: bool = ...,
        telemetry_bound: int = ...,
    ) -> None: ...

    def shape(self) -> tuple[int, int]: ...
    def zero_grad(self) -> None: ...
    def reset(self) -> None: ...
    def accumulate_wave(self, wave: Any) -> None: ...
    def accumulate_pair(self, prediction: Any, target: Any) -> None: ...
    def summary(self) -> Any: ...
    def step(self, weights: Tensor | None = ..., *, note: str | None = ...) -> Tensor: ...

class AmegagradSession(SpiralSession):
    opt: Amegagrad
    hyper: Hypergrad
    real: Realgrad
    weights: Tensor
    ztrainer: ZSpaceTrainer | None
    route: AtlasRoute | None

    def __init__(
        self,
        *shape_args: Any,
        curvature: float = ...,
        hyper_learning_rate: float = ...,
        real_learning_rate: float = ...,
        backend: str = ...,
        seed: int | None = ...,
        shape: Any | None = ...,
        rows: Any | None = ...,
        cols: Any | None = ...,
        topos: Any | None = ...,
        gain: float = ...,
        weights: Any | None = ...,
        z_dim: int = ...,
        z_lr: float = ...,
        z_lam_frac: float = ...,
        telemetry: bool = ...,
        telemetry_bound: int = ...,
    ) -> None: ...

    def shape(self) -> tuple[int, int]: ...
    def zero_grad(self) -> None: ...
    def reset(self) -> None: ...
    def step_wave(
        self,
        wave: Any,
        *,
        tune: bool = ...,
        gain: float | None = ...,
        control: Any | None = ...,
        note: str | None = ...,
    ) -> Tensor: ...

    def step_pair(
        self,
        prediction: Any,
        target: Any,
        *,
        tune: bool = ...,
        gain: float | None = ...,
        control: Any | None = ...,
        note: str | None = ...,
    ) -> Tensor: ...

    def step_text(
        self,
        encoder: Any,
        text: str,
        *,
        tune: bool = ...,
        gain: float | None = ...,
        control: Any | None = ...,
        note: str | None = ...,
    ) -> Tensor: ...

def hypergrad_session(*shape_args: Any, **kwargs: Any) -> HypergradSession: ...

def amegagrad_session(*shape_args: Any, **kwargs: Any) -> AmegagradSession: ...

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

class InfiniteZSpacePatch:
    @property
    def dimension(self) -> float: ...
    @property
    def zoom(self) -> float: ...
    @property
    def support(self) -> Tuple[float, float]: ...
    @property
    def mellin_weights(self) -> List[float]: ...
    @property
    def density(self) -> List[float]: ...
    def eta_bar(self) -> float: ...

class FractalCanvas:
    def __init__(
        self,
        dimension: float = ...,
        capacity: int = ...,
        width: int = ...,
        height: int = ...,
    ) -> None: ...
    @property
    def dimension(self) -> float: ...
    def set_dimension(self, dimension: float) -> None: ...
    def emit_zspace_patch(
        self,
        dimension: Optional[float] = ...,
        zoom: Optional[float] = ...,
        steps: Optional[int] = ...,
    ) -> InfiniteZSpacePatch: ...
    def emit_zspace_infinite(
        self,
        dimension: Optional[float] = ...,
    ) -> InfiniteZSpacePatch: ...
    def emit_infinite_z(
        self,
        zoom: Optional[float] = ...,
        steps: Optional[int] = ...,
        dimension: Optional[float] = ...,
    ) -> InfiniteZSpacePatch: ...

class CanvasProjector:
    def __init__(
        self,
        width: int = ...,
        height: int = ...,
        *,
        capacity: int = ...,
        palette: str = ...,
    ) -> None: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def queue_len(self) -> int: ...
    def total_weight(self) -> float: ...

    def palette(self) -> str: ...
    def set_palette(self, name: str) -> None: ...
    def reset_normalizer(self) -> None: ...

    def push_patch(
        self,
        relation: Tensor | Sequence[Sequence[float]],
        *,
        coherence: float = ...,
        tension: float = ...,
        depth: int = ...,
    ) -> None: ...

    def emit_zspace_patch(
        self,
        *,
        coherence: float = ...,
        tension: float = ...,
        depth: int = ...,
    ) -> Dict[str, object]: ...

    def emit_wasm_trail(self, curvature: float = ...) -> Dict[str, object]: ...

    def emit_atlas_frame(
        self,
        *,
        prefix: str = ...,
        refresh: bool = ...,
        timestamp: float | None = ...,
    ) -> AtlasFrame: ...

    def clear_queue(self) -> None: ...

    def rgba(self) -> bytes: ...
    def refresh_rgba(self) -> bytes: ...

    def tensor(self) -> Tensor: ...
    def refresh_tensor(self) -> Tensor: ...

    def refresh_vector_fft_tensor(self, *, inverse: bool = ...) -> Tensor: ...
    def refresh_vector_fft_power_db_tensor(self, *, inverse: bool = ...) -> Tensor: ...

class CanvasZSpacePatch:
    relation: Tensor
    coherence: float
    tension: float
    depth: int
    weight: float

    def to_dict(self) -> Dict[str, object]: ...

class CanvasWasmTrail:
    curvature: float
    width: int
    height: int
    samples: Tensor

    def to_dict(self) -> Dict[str, object]: ...

class QuantumOverlayConfig:
    def __init__(
        self,
        curvature: float = ...,
        qubits: int = ...,
        packing_bias: float = ...,
        leech_shells: int = ...,
    ) -> None: ...
    @property
    def curvature(self) -> float: ...
    def set_curvature(self, curvature: float) -> None: ...
    @property
    def qubits(self) -> int: ...
    def set_qubits(self, qubits: int) -> None: ...
    @property
    def packing_bias(self) -> float: ...
    def set_packing_bias(self, packing_bias: float) -> None: ...
    @property
    def leech_shells(self) -> int: ...
    def set_leech_shells(self, leech_shells: int) -> None: ...

class ZResonance:
    def __init__(
        self,
        spectrum: Optional[Sequence[float]] = ...,
        eta_hint: float = ...,
        shell_weights: Optional[Sequence[float]] = ...,
    ) -> None: ...
    @staticmethod
    def from_spectrum(
        spectrum: Sequence[float],
        eta_hint: Optional[float] = ...,
    ) -> ZResonance: ...
    @staticmethod
    def from_pulses(pulses: Sequence[MaxwellPulse]) -> ZResonance: ...
    @property
    def spectrum(self) -> List[float]: ...
    @property
    def eta_hint(self) -> float: ...
    @property
    def shell_weights(self) -> List[float]: ...

class QuantumMeasurement:
    @property
    def active_qubits(self) -> List[int]: ...
    @property
    def eta_bar(self) -> float: ...
    @property
    def policy_logits(self) -> List[float]: ...
    @property
    def packing_pressure(self) -> float: ...
    def top_qubits(self, count: Optional[int] = ...) -> List[Tuple[int, float]]: ...
    def activation_density(self) -> float: ...
    def to_policy_update(self, base_rate: float = ...) -> Dict[str, float]: ...

class ZOverlayCircuit:
    def weights(self) -> List[float]: ...
    def eta_bar(self) -> float: ...
    def packing_pressure(self) -> float: ...
    def measure(self, threshold: float = ...) -> QuantumMeasurement: ...

class QuantumRealityStudio:
    def __init__(
        self,
        curvature: float = ...,
        qubits: int = ...,
        packing_bias: float = ...,
        leech_shells: int = ...,
    ) -> None: ...
    @property
    def config(self) -> QuantumOverlayConfig: ...
    def configure(
        self,
        *,
        curvature: Optional[float] = ...,
        qubits: Optional[int] = ...,
        packing_bias: Optional[float] = ...,
        leech_shells: Optional[int] = ...,
    ) -> None: ...
    def overlay_zspace(self, resonance: ZResonance) -> ZOverlayCircuit: ...
    def overlay(self, resonance: ZResonance) -> ZOverlayCircuit: ...
    def record_quantum_policy(
        self,
        pulses: Sequence[MaxwellPulse],
        *,
        threshold: float = ...,
    ) -> QuantumMeasurement: ...

class FractalQuantumSession:
    def __init__(
        self,
        studio: QuantumRealityStudio,
        *,
        threshold: float = ...,
        eta_scale: float = ...,
    ) -> None: ...
    @property
    def threshold(self) -> float: ...
    @property
    def eta_scale(self) -> float: ...
    @property
    def ingested(self) -> int: ...
    def ingest(
        self,
        patch: InfiniteZSpacePatch,
        *,
        weight: float = ...,
    ) -> ZResonance: ...
    def resonance(self) -> ZResonance: ...
    def measure(self, *, threshold: Optional[float] = ...) -> QuantumMeasurement: ...
    def clear(self) -> None: ...

def resonance_from_fractal_patch(
    patch: InfiniteZSpacePatch,
    eta_scale: float = ...,
) -> ZResonance: ...

def quantum_measurement_from_fractal(
    studio: QuantumRealityStudio,
    patch: InfiniteZSpacePatch,
    threshold: float = ...,
    eta_scale: float = ...,
) -> QuantumMeasurement: ...

def quantum_measurement_from_fractal_sequence(
    studio: QuantumRealityStudio,
    patches: Sequence[InfiniteZSpacePatch],
    *,
    weights: Optional[Sequence[float]] = ...,
    threshold: float = ...,
    eta_scale: float = ...,
) -> QuantumMeasurement: ...

class ZTigerOptim:
    def __init__(self, curvature: float = ...) -> None: ...
    @property
    def curvature(self) -> float: ...
    @property
    def gain(self) -> float: ...
    def update(self, lora_pid: float, resonance: Sequence[float]) -> float: ...

def tempo_latency_score(tile: int, slack: int) -> float: ...

def apply_vision_update(
    vision: SpiralTorchVision,
    canvas: CanvasTransformer,
    *,
    hypergrad: Optional[Sequence[Sequence[float]]] = ...,
    realgrad: Optional[Sequence[Sequence[float]]] = ...,
    weight: float = ...,
    include_patch: bool = ...,
) -> CanvasSnapshot: ...

def zrelativity_heatmap(
    model: ZRelativityModel,
    field: str,
) -> List[List[float]]: ...

def set_global_seed(seed: int) -> None: ...

def golden_ratio() -> float: ...

def golden_angle() -> float: ...

def fibonacci_pacing(total_steps: int) -> List[int]: ...

def pack_nacci_chunks(order: int, total_steps: int) -> List[int]: ...

def pack_tribonacci_chunks(total_steps: int) -> List[int]: ...

def pack_tetranacci_chunks(total_steps: int) -> List[int]: ...

def build_info() -> Dict[str, object]: ...

def generate_plan_batch_ex(
    n: int,
    total_steps: int,
    base_radius: float,
    radial_growth: float,
    base_height: float,
    meso_gain: float,
    micro_gain: float,
    seed: Optional[int] = ...,
) -> List[SoT3DPlan]: ...


class SoT3DStep:
    @property
    def index(self) -> int: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def height(self) -> float: ...

    @property
    def macro_index(self) -> int: ...

    @property
    def meso_role(self) -> str: ...

    @property
    def micro_role(self) -> str: ...

    def as_dict(self) -> Dict[str, object]: ...


class MacroSummary:
    @property
    def index(self) -> int: ...

    @property
    def start(self) -> int: ...

    @property
    def length(self) -> int: ...

    def as_dict(self) -> Dict[str, object]: ...


class SoT3DPlan:
    @property
    def total_steps(self) -> int: ...

    @property
    def base_radius(self) -> float: ...

    @property
    def radial_growth(self) -> float: ...

    def steps(self) -> List[SoT3DStep]: ...

    def as_dicts(self) -> List[Dict[str, object]]: ...

    def as_tensor(self) -> Tensor: ...

    def feature_tensor(self) -> Tensor: ...

    def role_tensor(self) -> Tensor: ...

    def reflection_tensor(self) -> Tensor: ...

    def macro_summary_tensor(self) -> Tensor: ...

    def macro_summaries(self) -> List[MacroSummary]: ...

    def polyline(self) -> List[Tuple[float, float, float]]: ...

    def reflection_points(self) -> List[Tuple[int, str]]: ...

    def grow_biome(
        self,
        topos: OpenCartesianTopos,
        label_prefix: str | None = ...,
        include_reflections: bool = ...,
        include_roles: bool = ...,
    ) -> TensorBiome: ...

    def infuse_biome(
        self,
        biome: TensorBiome,
        label_prefix: str | None = ...,
        include_reflections: bool = ...,
        include_roles: bool = ...,
    ) -> None: ...


class _NnNonLiner:
    def __init__(
        self,
        name: str,
        features: int,
        *,
        activation: str = ...,
        slope: float = ...,
        gain: float = ...,
        bias: float = ...,
        curvature: float | None = ...,
        z_scale: float | None = ...,
        retention: float = ...,
    ) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def reset_metrics(self) -> None: ...

    def configure_geometry(
        self,
        *,
        curvature: float | None = ...,
        z_scale: float | None = ...,
        retention: float | None = ...,
    ) -> None: ...

    def attach_hypergrad(
        self,
        curvature: float,
        learning_rate: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...

    def attach_realgrad(self, learning_rate: float) -> None: ...

    def zero_accumulators(self) -> None: ...

    def apply_step(self, fallback_lr: float) -> None: ...

    def state_dict(self) -> List[Tuple[str, Tensor]]: ...

    def load_state_dict(self, state: Sequence[Tuple[str, Tensor]]) -> None: ...

    @property
    def activation(self) -> str: ...

    @property
    def curvature(self) -> float | None: ...

    @property
    def z_scale(self) -> float | None: ...

    @property
    def retention(self) -> float | None: ...

    @property
    def psi_drift(self) -> float | None: ...

    @property
    def last_hyperbolic_radius(self) -> float | None: ...

    @property
    def gain(self) -> Tensor: ...

    @property
    def slope(self) -> Tensor: ...

    @property
    def bias(self) -> Tensor: ...

    def gradients(self) -> Tuple[Tensor | None, Tensor | None, Tensor | None]: ...


class _NnDropout:
    def __init__(self, probability: float, *, seed: int | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...

    @property
    def probability(self) -> float: ...

    @property
    def training(self) -> bool: ...

    @training.setter
    def training(self, value: bool) -> None: ...


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


class CoherenceSignature:
    dominant_channel: int | None
    energy_ratio: float
    entropy: float
    mean_coherence: float
    swap_invariant: bool


class CoherenceObservation:
    is_signature: bool
    label: str
    signature: CoherenceSignature | None


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
    observation: CoherenceObservation


class PreDiscardTelemetry:
    dominance_ratio: float
    energy_floor: float
    discarded: int
    preserved: int
    used_fallback: bool
    total: int
    preserved_ratio: float
    discarded_ratio: float
    survivor_energy: float
    discarded_energy: float
    total_energy: float
    survivor_energy_ratio: float
    discarded_energy_ratio: float
    dominant_weight: float


class PreDiscardSnapshot:
    step: int
    telemetry: PreDiscardTelemetry
    survivors: List[int]
    discarded: List[int]
    filtered: List[float]


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


class ZConv:
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel: Tuple[int, int],
        *,
        stride: Tuple[int, int] = ...,
        padding: Tuple[int, int] = ...,
        dilation: Tuple[int, int] = ...,
        input_hw: Tuple[int, int],
        layout: Literal["NCHW", "NHWC"] = ...,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...
    def backward(self, x: Tensor, grad_output: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def attach_hypergrad(
        self,
        curvature: float,
        learning_rate: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...
    def attach_realgrad(self, learning_rate: float) -> None: ...
    def zero_accumulators(self) -> None: ...
    def apply_step(self, fallback_lr: float) -> None: ...
    def state_dict(self) -> List[Tuple[str, Tensor]]: ...
    def load_state_dict(self, state: Sequence[Tuple[str, Tensor]]) -> None: ...

    @property
    def layout(self) -> Literal["NCHW", "NHWC"]: ...

    @property
    def in_channels(self) -> int: ...

    @property
    def out_channels(self) -> int: ...

    @property
    def input_hw(self) -> Tuple[int, int]: ...

    @property
    def output_hw(self) -> Tuple[int, int]: ...

    @property
    def output_shape(self) -> Tuple[int, int, int]: ...

    @property
    def kernel(self) -> Tuple[int, int]: ...

    @property
    def stride(self) -> Tuple[int, int]: ...

    @property
    def padding(self) -> Tuple[int, int]: ...

    @property
    def dilation(self) -> Tuple[int, int]: ...

    @property
    def psi_drift(self) -> float | None: ...


class ZConv6DA:
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        grid: Tuple[int, int, int],
        *,
        leech_rank: int = ...,
        leech_weight: float = ...,
        layout: Literal["NCDHW", "NDHWC"] = ...,
        neighbors: Sequence[Tuple[int, int, int]] | None = ...,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...
    def backward(self, x: Tensor, grad_output: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def attach_hypergrad(
        self,
        curvature: float,
        learning_rate: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...
    def attach_realgrad(self, learning_rate: float) -> None: ...
    def zero_accumulators(self) -> None: ...
    def apply_step(self, fallback_lr: float) -> None: ...
    def state_dict(self) -> List[Tuple[str, Tensor]]: ...
    def load_state_dict(self, state: Sequence[Tuple[str, Tensor]]) -> None: ...
    def leech_enrich(self, geodesic: float) -> float: ...

    @staticmethod
    def ramanujan_pi_boost() -> float: ...

    @property
    def layout(self) -> Literal["NCDHW", "NDHWC"]: ...

    @property
    def grid(self) -> Tuple[int, int, int]: ...

    @property
    def in_channels(self) -> int: ...

    @property
    def out_channels(self) -> int: ...

    @property
    def neighbor_count(self) -> int: ...

    @property
    def neighbor_offsets(self) -> List[Tuple[int, int, int]]: ...

    @property
    def leech_rank(self) -> int: ...

    @property
    def leech_weight(self) -> float: ...

    @property
    def input_shape(self) -> Tuple[int, int, int, int]: ...

    @property
    def output_shape(self) -> Tuple[int, int, int, int]: ...

    @property
    def ramanujan_pi_delta(self) -> float: ...


class Pool2d:
    def __init__(
        self,
        mode: Literal["max", "avg"],
        channels: int,
        height: int,
        width: int,
        kernel: Tuple[int, int],
        *,
        stride: Tuple[int, int] | None = ...,
        padding: Tuple[int, int] | None = ...,
        layout: Literal["NCHW", "NHWC"] = ...,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...
    def backward(self, x: Tensor, grad_output: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...

    @property
    def mode(self) -> Literal["max", "avg"]: ...

    @property
    def layout(self) -> Literal["NCHW", "NHWC"]: ...

    @property
    def kernel(self) -> Tuple[int, int]: ...

    @property
    def stride(self) -> Tuple[int, int]: ...

    @property
    def padding(self) -> Tuple[int, int]: ...

    @property
    def input_shape(self) -> Tuple[int, int, int]: ...

    @property
    def output_shape(self) -> Tuple[int, int, int]: ...

    def set_layout(self, layout: Literal["NCHW", "NHWC"]) -> None: ...


class ZPooling:
    def __init__(
        self,
        channels: int,
        kernel: Tuple[int, int],
        input_hw: Tuple[int, int],
        *,
        stride: Tuple[int, int] | None = ...,
        padding: Tuple[int, int] = ...,
        layout: Literal["NCHW", "NHWC"] = ...,
        mode: Literal["max", "avg"] = ...,
    ) -> None: ...

    def forward(self, x: Tensor) -> Tensor: ...
    def backward(self, x: Tensor, grad_output: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...

    @property
    def mode(self) -> Literal["max", "avg"]: ...

    @property
    def layout(self) -> Literal["NCHW", "NHWC"]: ...

    @property
    def channels(self) -> int: ...

    @property
    def input_hw(self) -> Tuple[int, int]: ...

    @property
    def output_hw(self) -> Tuple[int, int]: ...

    @property
    def kernel(self) -> Tuple[int, int]: ...

    @property
    def stride(self) -> Tuple[int, int]: ...

    @property
    def padding(self) -> Tuple[int, int]: ...


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

    def configure_pre_discard_memory(self, limit: int) -> None: ...

    def clear_pre_discard_snapshots(self) -> None: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def dim(self) -> int: ...

    def num_heads(self) -> int: ...

    def pre_discard_policy(self) -> PreDiscardPolicy | None: ...

    def pre_discard_snapshots(self) -> List[PreDiscardSnapshot]: ...

    def curvature(self) -> float: ...

    def maxwell_channels(self) -> int: ...

    def topos(self) -> OpenCartesianTopos: ...

    def install_trace_recorder(
        self,
        *,
        capacity: int = ...,
        max_vector_len: int = ...,
        publish_plugin_events: bool = ...,
    ) -> ZSpaceTraceRecorder: ...

    def trace_recorder(self) -> ZSpaceTraceRecorder | None: ...


class _ZSpaceTraceRecorder:
    def snapshot(self) -> Dict[str, Any]: ...
    def clear(self) -> None: ...
    def write_jsonl(self, path: str) -> None: ...


def is_swap_invariant(arrangement: Sequence[float]) -> bool: ...


class _NnIdentity:
    def __init__(self) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...


class _NnLinear:
    def __init__(self, name: str, input_dim: int, output_dim: int) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def attach_hypergrad(
        self,
        curvature: float,
        learning_rate: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...

    def attach_realgrad(self, learning_rate: float) -> None: ...

    def zero_accumulators(self) -> None: ...

    def apply_step(self, fallback_lr: float) -> None: ...

    def state_dict(self) -> List[Tuple[str, Tensor]]: ...

    def load_state_dict(self, state: Sequence[Tuple[str, Tensor]]) -> None: ...


class _NnRelu:
    def __init__(self) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...


class _NnSequential:
    def __init__(self) -> None: ...

    def add(self, layer: object) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def attach_hypergrad(
        self,
        curvature: float,
        learning_rate: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...

    def attach_realgrad(self, learning_rate: float) -> None: ...

    def zero_accumulators(self) -> None: ...

    def apply_step(self, fallback_lr: float) -> None: ...

    def state_dict(self) -> List[Tuple[str, Tensor]]: ...

    def load_state_dict(self, state: Sequence[Tuple[str, Tensor]]) -> None: ...

    def len(self) -> int: ...

    def is_empty(self) -> bool: ...

    def __len__(self) -> int: ...


class _NnMeanSquaredError:
    def __init__(self) -> None: ...

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...

class _NnHyperbolicCrossEntropy:
    def __init__(self, curvature: float, *, epsilon: float | None = ...) -> None: ...

    @property
    def curvature(self) -> float: ...

    @property
    def epsilon(self) -> float: ...

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...

class _NnFocalLoss:
    def __init__(self, alpha: float = ..., gamma: float = ..., *, epsilon: float | None = ...) -> None: ...

    @property
    def alpha(self) -> float: ...

    @property
    def gamma(self) -> float: ...

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...

class _NnContrastiveLoss:
    def __init__(self, margin: float = ...) -> None: ...

    @property
    def margin(self) -> float: ...

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...

class _NnTripletLoss:
    def __init__(self, margin: float = ...) -> None: ...

    @property
    def margin(self) -> float: ...

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def backward(self, prediction: Tensor, target: Tensor) -> Tensor: ...

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...

class _NnRoundtableConfig:
    def __init__(
        self,
        *,
        top_k: int = ...,
        mid_k: int = ...,
        bottom_k: int = ...,
        here_tolerance: float = ...,
    ) -> None: ...

    @property
    def top_k(self) -> int: ...

    @property
    def mid_k(self) -> int: ...

    @property
    def bottom_k(self) -> int: ...

    @property
    def here_tolerance(self) -> float: ...


class _NnRoundtableSchedule:
    def above(self) -> RankPlan: ...
    def here(self) -> RankPlan: ...
    def beneath(self) -> RankPlan: ...


class _NnEpochStats:
    @property
    def batches(self) -> int: ...

    @property
    def total_loss(self) -> float: ...

    @property
    def average_loss(self) -> float: ...


class _NnModuleTrainer:
    def __init__(
        self,
        *,
        backend: str = ...,
        curvature: float = ...,
        hyper_learning_rate: float = ...,
        fallback_learning_rate: float = ...,
        lane_width: int | None = ...,
        subgroup: bool | None = ...,
        max_workgroup: int | None = ...,
        shared_mem_per_workgroup: int | None = ...,
    ) -> None: ...

    def roundtable(
        self,
        rows: int,
        cols: int,
        config: _NnRoundtableConfig | None = ...,
    ) -> _NnRoundtableSchedule: ...

    def train_epoch(
        self,
        module: object,
        loss: object,
        batches: Iterable[Tuple[Tensor, Tensor]],
        schedule: _NnRoundtableSchedule,
    ) -> _NnEpochStats: ...


class _NnScaler:
    def __init__(self, name: str, features: int) -> None: ...

    @staticmethod
    def from_gain(name: str, gain: Tensor) -> _NnScaler: ...

    def forward(self, input: Tensor) -> Tensor: ...

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor: ...

    def __call__(self, x: Tensor) -> Tensor: ...

    def calibrate(self, samples: Tensor, epsilon: float) -> None: ...

    def attach_hypergrad(self, curvature: float, learning_rate: float) -> None: ...

    def attach_hypergrad_with_topos(
        self,
        curvature: float,
        learning_rate: float,
        *,
        topos: OpenCartesianTopos | None = ...,
    ) -> None: ...

    def attach_realgrad(self, learning_rate: float) -> None: ...

    def zero_accumulators(self) -> None: ...

    def apply_step(self, fallback_lr: float) -> None: ...

    def state_dict(self) -> List[Tuple[str, Tensor]]: ...

    def load_state_dict(self, state: Sequence[Tuple[str, Tensor]]) -> None: ...

    @property
    def gain(self) -> Tensor: ...

    @property
    def baseline(self) -> Tensor: ...

    def gradient(self) -> Tensor | None: ...

    def psi_probe(self) -> float | None: ...

    def psi_components(self) -> Tensor | None: ...


class _NnModule(ModuleType):
    Identity: type[_NnIdentity]
    Linear: type[_NnLinear]
    Relu: type[_NnRelu]
    Sequential: type[_NnSequential]
    MeanSquaredError: type[_NnMeanSquaredError]
    HyperbolicCrossEntropy: type[_NnHyperbolicCrossEntropy]
    CrossEntropy: type[_NnHyperbolicCrossEntropy]
    FocalLoss: type[_NnFocalLoss]
    ContrastiveLoss: type[_NnContrastiveLoss]
    TripletLoss: type[_NnTripletLoss]
    RoundtableConfig: type[_NnRoundtableConfig]
    RoundtableSchedule: type[_NnRoundtableSchedule]
    EpochStats: type[_NnEpochStats]
    ModuleTrainer: type[_NnModuleTrainer]
    Scaler: type[_NnScaler]
    NonLiner: type[_NnNonLiner]
    Dropout: type[_NnDropout]
    Dataset: type[_NnDataset]
    DataLoader: type[_NnDataLoader]
    DataLoaderIter: type[_NnDataLoaderIter]
    CoherenceDiagnostics: type[CoherenceDiagnostics]
    ZSpaceCoherenceSequencer: type[_ZSpaceCoherenceSequencer]
    PreDiscardTelemetry: type[PreDiscardTelemetry]
    PreDiscardPolicy: type[PreDiscardPolicy]
    PreDiscardSnapshot: type[PreDiscardSnapshot]
    ZRelativityModule: type[ZRelativityModule]
    ZConv: type[ZConv]
    ZPooling: type[ZPooling]
    Pool2d: type[Pool2d]

    def from_samples(samples: Sequence[Tuple[Tensor, Tensor]]) -> _NnDataLoader: ...
    def save_json(
        target: object | Mapping[str, Tensor] | Sequence[Tuple[str, Tensor]],
        path: str,
    ) -> None: ...
    def load_json(
        target: object | None,
        path: str,
    ) -> List[Tuple[str, Tensor]] | None: ...
    def save(
        path: Any,
        target: object | Mapping[str, Tensor] | Sequence[Tuple[str, Tensor]],
    ) -> None: ...
    def load(
        path: Any,
        target: object | None = ...,
    ) -> List[Tuple[str, Tensor]] | None: ...
    def save_bincode(
        target: object | Mapping[str, Tensor] | Sequence[Tuple[str, Tensor]],
        path: str,
    ) -> None: ...
    def load_bincode(
        target: object | None,
        path: str,
    ) -> List[Tuple[str, Tensor]] | None: ...


nn: _NnModule


class Identity(_NnIdentity):
    ...


class Linear(_NnLinear):
    ...


class Relu(_NnRelu):
    ...


class Sequential(_NnSequential):
    ...


class MeanSquaredError(_NnMeanSquaredError):
    ...

class HyperbolicCrossEntropy(_NnHyperbolicCrossEntropy):
    ...

class CrossEntropy(_NnHyperbolicCrossEntropy):
    ...

class FocalLoss(_NnFocalLoss):
    ...

class ContrastiveLoss(_NnContrastiveLoss):
    ...

class TripletLoss(_NnTripletLoss):
    ...

class Scaler(_NnScaler):
    ...


class NonLiner(_NnNonLiner):
    ...


class Dropout(_NnDropout):
    ...


class ZSpaceCoherenceSequencer(_ZSpaceCoherenceSequencer):
    ...


class ZSpaceTraceRecorder(_ZSpaceTraceRecorder):
    ...


class MetaMembConfig:
    def __init__(
        self,
        *,
        delta: Sequence[float] | None = ...,
        omega: Sequence[float] | None = ...,
        theta: Sequence[float] | None = ...,
    ) -> None: ...

    @staticmethod
    def default() -> MetaMembConfig: ...

    @property
    def delta(self) -> Tuple[float, float, float]: ...

    @property
    def omega(self) -> Tuple[float, float, float]: ...

    @property
    def theta(self) -> Tuple[float, float, float]: ...


class CircleLockMapConfig:
    def __init__(
        self,
        *,
        lam_min: float = ...,
        lam_max: float = ...,
        lam_bins: int = ...,
        wd_min: float = ...,
        wd_max: float = ...,
        wd_bins: int = ...,
        burn_in: int = ...,
        samples: int = ...,
        qmax: int = ...,
    ) -> None: ...

    @staticmethod
    def default() -> CircleLockMapConfig: ...

    @property
    def lam_min(self) -> float: ...

    @property
    def lam_max(self) -> float: ...

    @property
    def lam_bins(self) -> int: ...

    @property
    def wd_min(self) -> float: ...

    @property
    def wd_max(self) -> float: ...

    @property
    def wd_bins(self) -> int: ...

    @property
    def burn_in(self) -> int: ...

    @property
    def samples(self) -> int: ...

    @property
    def qmax(self) -> int: ...


class PsiTelemetryConfig:
    def __init__(
        self,
        *,
        emit_atlas: bool = ...,
        atlas_timestamp: float | None = ...,
        emit_psi: bool = ...,
        psi_step_base: int = ...,
        emit_golden: bool = ...,
        golden_baseline_interval: float = ...,
        golden_baseline_window: int = ...,
    ) -> None: ...

    @property
    def emit_atlas(self) -> bool: ...

    @property
    def atlas_timestamp(self) -> float | None: ...

    @property
    def emit_psi(self) -> bool: ...

    @property
    def psi_step_base(self) -> int: ...

    @property
    def emit_golden(self) -> bool: ...

    @property
    def golden_baseline_interval(self) -> float: ...

    @property
    def golden_baseline_window(self) -> int: ...


class PsiSynchroConfig:
    def __init__(
        self,
        *,
        step: float = ...,
        samples: int = ...,
        ticker_interval: float | None = ...,
        min_ident_points: int = ...,
        max_ident_points: int = ...,
        metamemb: MetaMembConfig | None = ...,
        circle_map: CircleLockMapConfig | None = ...,
        telemetry: PsiTelemetryConfig | None = ...,
    ) -> None: ...

    @staticmethod
    def default() -> PsiSynchroConfig: ...

    @property
    def step(self) -> float: ...

    @property
    def samples(self) -> int: ...

    @property
    def ticker_interval(self) -> float | None: ...

    @property
    def min_ident_points(self) -> int: ...

    @property
    def max_ident_points(self) -> int: ...

    @property
    def metamemb(self) -> MetaMembConfig: ...

    @property
    def circle_map(self) -> CircleLockMapConfig: ...

    @property
    def telemetry(self) -> PsiTelemetryConfig | None: ...


class PsiBranchState:
    def __init__(
        self,
        branch_id: str,
        *,
        gamma: float = ...,
        lambda_: float = ...,
        wd: float = ...,
        omega0: float = ...,
        drift_coupled: float = ...,
        phase0: float = ...,
    ) -> None: ...

    @property
    def branch_id(self) -> str: ...

    @property
    def gamma(self) -> float: ...

    @property
    def lambda_(self) -> float: ...

    @property
    def wd(self) -> float: ...

    @property
    def omega0(self) -> float: ...

    @property
    def drift_coupled(self) -> float: ...

    @property
    def phase0(self) -> float: ...

    def poincare_period(self) -> float: ...


class ZPulseSnapshot:
    @property
    def source(self) -> str: ...

    @property
    def ts(self) -> int: ...

    @property
    def tempo(self) -> float: ...

    @property
    def band_energy(self) -> Tuple[float, float, float]: ...

    @property
    def drift(self) -> float: ...

    @property
    def z_bias(self) -> float: ...

    @property
    def density_fluctuation(self) -> float: ...

    @property
    def support(self) -> Tuple[float, float, float]: ...

    @property
    def scale(self) -> Tuple[float, float] | None: ...

    @property
    def quality(self) -> float: ...

    @property
    def stderr(self) -> float: ...

    @property
    def latency_ms(self) -> float: ...


class ContextualPulseFrame:
    @property
    def summary(self) -> str: ...

    @property
    def highlights(self) -> List[str]: ...

    @property
    def label(self) -> str | None: ...

    @property
    def lexical_weight(self) -> float: ...

    @property
    def signature(self) -> Tuple[int, int, int] | None: ...

    @property
    def support(self) -> int: ...

    @property
    def pulse(self) -> ZPulseSnapshot: ...


class ContextualLagrangianGate:
    def __init__(
        self,
        curvature: float,
        temperature: float,
        *,
        gauge: str = ...,
        tempo_normaliser: float | None = ...,
        energy_gain: float = ...,
        drift_gain: float = ...,
        bias_gain: float = ...,
        support_gain: float = ...,
        scale: Tuple[float, float] | None = ...,
        quality_floor: float = ...,
        stderr_gain: float = ...,
    ) -> None: ...

    def project(
        self,
        placements: Sequence[int],
        edges: Optional[Sequence[Tuple[int, int]]] = ...,
        *,
        gauge: str | None = ...,
        ts: int = ...,
    ) -> ContextualPulseFrame: ...

    @property
    def gauge(self) -> str: ...


class ArnoldTonguePeak:
    @property
    def ratio_p(self) -> int: ...

    @property
    def ratio_q(self) -> int: ...

    @property
    def rotation(self) -> float: ...

    @property
    def lam(self) -> float: ...

    @property
    def wd(self) -> float: ...

    @property
    def strength(self) -> float: ...

    @property
    def peak_strength(self) -> float: ...

    @property
    def error(self) -> float: ...

    @property
    def ratio(self) -> float: ...


class HeatmapAnalytics:
    @property
    def total_energy(self) -> float: ...

    @property
    def leading_sum(self) -> float: ...

    @property
    def central_sum(self) -> float: ...

    @property
    def trailing_sum(self) -> float: ...

    @property
    def leading_norm(self) -> float: ...

    @property
    def central_norm(self) -> float: ...

    @property
    def trailing_norm(self) -> float: ...

    @property
    def dominant_lam(self) -> float: ...

    @property
    def dominant_wd(self) -> float: ...

    @property
    def peak_value(self) -> float: ...

    @property
    def peak_ratio(self) -> float: ...

    @property
    def radius(self) -> float: ...

    @property
    def log_radius(self) -> float: ...

    @property
    def bias(self) -> float: ...

    @property
    def drift(self) -> float: ...

    @property
    def quality(self) -> float: ...

    @property
    def stderr(self) -> float: ...

    @property
    def entropy(self) -> float: ...

    def band_energy(self) -> Tuple[float, float, float]: ...


class HeatmapResult:
    @property
    def branch_id(self) -> str: ...

    @property
    def gamma(self) -> float: ...

    @property
    def kappa_hat(self) -> float: ...

    @property
    def lam_grid(self) -> List[float]: ...

    @property
    def wd_grid(self) -> List[float]: ...

    @property
    def matrix(self) -> List[List[float]]: ...

    @property
    def tongues(self) -> List[ArnoldTonguePeak]: ...

    def dominant_tongue(self) -> ArnoldTonguePeak | None: ...

    def analyse(self) -> HeatmapAnalytics | None: ...

    def to_atlas_fragment(
        self,
        timestamp: float | None = ...,
    ) -> Dict[str, Any] | None: ...

    def to_psi_reading(self, step: int) -> Dict[str, Any] | None: ...

    def to_zpulse(self, ts: int) -> ZPulseSnapshot: ...


class PsiSynchroPulse:
    @property
    def branch_id(self) -> str: ...

    @property
    def pulse(self) -> ZPulseSnapshot: ...


class PsiSynchroResult:
    @property
    def heatmaps(self) -> List[HeatmapResult]: ...

    @property
    def pulses(self) -> List[PsiSynchroPulse]: ...

    def atlas_fragments(self) -> List[Tuple[str, Dict[str, object]]]: ...

    def psi_readings(self) -> List[Tuple[str, Dict[str, object]]]: ...

    def by_branch(self) -> List[Tuple[str, ZPulseSnapshot]]: ...


class _PsiModule(ModuleType):
    MetaMembConfig: type[MetaMembConfig]
    CircleLockMapConfig: type[CircleLockMapConfig]
    PsiTelemetryConfig: type[PsiTelemetryConfig]
    PsiSynchroConfig: type[PsiSynchroConfig]
    PsiBranchState: type[PsiBranchState]
    ArnoldTonguePeak: type[ArnoldTonguePeak]
    HeatmapAnalytics: type[HeatmapAnalytics]
    HeatmapResult: type[HeatmapResult]
    ZPulseSnapshot: type[ZPulseSnapshot]
    PsiSynchroPulse: type[PsiSynchroPulse]
    PsiSynchroResult: type[PsiSynchroResult]

    def run_multibranch_demo(
        branches: Sequence[PsiBranchState],
        config: PsiSynchroConfig | None = ...,
    ) -> PsiSynchroResult: ...

    def run_zspace_learning(
        branches: Sequence[PsiBranchState],
        config: PsiSynchroConfig | None = ...,
    ) -> PsiSynchroResult: ...


psi: _PsiModule


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

class _DatasetModule(ModuleType):
    class Dataset:
        def __init__(self) -> None: ...

        @staticmethod
        def from_samples(
            samples: Optional[Sequence[Tuple[Tensor, Tensor]]] = ...,
        ) -> "_DatasetModule.Dataset": ...

        def push(self, input: Tensor, target: Tensor) -> None: ...

        def len(self) -> int: ...

        def __len__(self) -> int: ...

        def is_empty(self) -> bool: ...

        def samples(self) -> List[Tuple[Tensor, Tensor]]: ...

        def iter(self) -> "_DatasetModule.DataLoaderIterator": ...

        def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]: ...

        def loader(self) -> "_DatasetModule.DataLoader": ...

    class DataLoader:
        @staticmethod
        def from_samples(
            samples: Optional[Sequence[Tuple[Tensor, Tensor]]] = ...,
        ) -> "_DatasetModule.DataLoader": ...

        def len(self) -> int: ...

        def __len__(self) -> int: ...

        def is_empty(self) -> bool: ...

        def batch_size(self) -> int: ...

        def prefetch_depth(self) -> int: ...

        def shuffle(self, seed: int) -> "_DatasetModule.DataLoader": ...

        def batched(self, batch_size: int) -> "_DatasetModule.DataLoader": ...

        def dynamic_batch_by_rows(self, max_rows: int) -> "_DatasetModule.DataLoader": ...

        def prefetch(self, depth: int) -> "_DatasetModule.DataLoader": ...

        def iter(self) -> "_DatasetModule.DataLoaderIterator": ...

        def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]: ...

    class DataLoaderIterator(Iterator[Tuple[Tensor, Tensor]]):
        def __iter__(self) -> "_DatasetModule.DataLoaderIterator": ...

        def __next__(self) -> Tuple[Tensor, Tensor]: ...


dataset: _DatasetModule

linalg: ModuleType

spiral_rl: ModuleType

rec: ModuleType

class _TelemetryModule(ModuleType):
    DashboardMetric: type[DashboardMetric]
    DashboardEvent: type[DashboardEvent]
    DashboardFrame: type[DashboardFrame]
    DashboardRing: type[DashboardRing]
    AtlasMetric: type[AtlasMetric]
    AtlasFragment: type[AtlasFragment]
    AtlasFrame: type[AtlasFrame]
    AtlasRoute: type[AtlasRoute]
    AtlasMetricFocus: type[AtlasMetricFocus]
    AtlasPerspective: type[AtlasPerspective]

    def current() -> SoftlogicZFeedback | None: ...
    def metric_root_token(name: str) -> str: ...
    def infer_district(name: str) -> str: ...
    def perspective_for_dict(
        route: AtlasRoute,
        district: str,
        focus_prefixes: Optional[Sequence[str]] = ...,
    ) -> Optional[Dict[str, object]]: ...
    def perspective_for_packet(
        route: AtlasRoute,
        district: str,
        focus_prefixes: Optional[Sequence[str]] = ...,
    ) -> Optional[AtlasPerspective]: ...

telemetry: _TelemetryModule

class _PluginModule(ModuleType):
    class PluginQueue:
        @property
        def event_type(self) -> str: ...

        @property
        def subscription_id(self) -> int: ...

        @property
        def maxlen(self) -> int: ...

        def poll(self) -> Optional[Dict[str, object]]: ...
        def drain(self, max_items: int | None = ...) -> List[Dict[str, object]]: ...
        def close(self) -> bool: ...
        def __len__(self) -> int: ...
        def __iter__(self) -> "_PluginModule.PluginQueue": ...
        def __next__(self) -> Dict[str, object]: ...

    class PluginRecorder:
        path: str
        event_types: List[str]
        closed: bool

        def close(self) -> bool: ...
        def __enter__(self) -> "_PluginModule.PluginRecorder": ...
        def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, object]], None]) -> int: ...
    def subscribe_many(
        self,
        event_types: Iterable[str],
        callback: Callable[[Dict[str, object]], None],
    ) -> List[Tuple[str, int]]: ...
    def unsubscribe(self, event_type: str, subscription_id: int) -> bool: ...
    def unsubscribe_many(self, subscriptions: Iterable[Tuple[str, int]]) -> int: ...
    def listen(self, event_type: str = ..., *, maxlen: int = ...) -> "_PluginModule.PluginQueue": ...
    def listen_stream(
        self,
        event_type: str = ...,
        *,
        maxlen: int = ...,
        poll_interval: float = ...,
        max_batch: int = ...,
    ) -> Iterator[Dict[str, object]]: ...
    def record(
        self,
        path: Any,
        event_types: str | Iterable[str] = ...,
        *,
        mode: str = ...,
        flush: bool = ...,
    ) -> "_PluginModule.PluginRecorder": ...
    def event_types(self) -> Dict[str, str]: ...

plugin: _PluginModule

ecosystem: ModuleType

class _OpsModule(ModuleType):
    @overload
    def register(
        self,
        name: str,
        num_inputs: int,
        num_outputs: int,
        forward: Callable[[Sequence[Tensor]], Tensor | Sequence[Tensor]],
        *,
        backward: Callable[[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]], Tensor | Sequence[Tensor]] | None = ...,
        description: str | None = ...,
        backends: Sequence[str] | None = ...,
        attributes: Mapping[str, object] | None = ...,
        supports_inplace: bool = ...,
        differentiable: bool | None = ...,
    ) -> None: ...

    @overload
    def register(
        self,
        name: str,
        forward: Callable[[Sequence[Tensor]], Tensor | Sequence[Tensor]],
        backward: Callable[[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]], Tensor | Sequence[Tensor]] | None = ...,
        *,
        num_inputs: int | None = ...,
        num_outputs: int | None = ...,
        description: str | None = ...,
        backends: Sequence[str] | None = ...,
        attributes: Mapping[str, object] | None = ...,
        supports_inplace: bool = ...,
        differentiable: bool | None = ...,
    ) -> None: ...

    def signature(
        self,
        num_inputs: int,
        num_outputs: int,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    @overload
    def execute(
        self,
        name: str,
        inputs: Sequence[Tensor],
        *,
        return_single: bool = ...,
    ) -> Tensor | List[Tensor]: ...

    @overload
    def execute(
        self,
        name: str,
        *inputs: Tensor,
        return_single: bool = ...,
    ) -> Tensor | List[Tensor]: ...

    def backward(
        self,
        name: str,
        inputs: Sequence[Tensor],
        outputs: Sequence[Tensor],
        grad_outputs: Sequence[Tensor],
        *,
        return_single: bool = ...,
    ) -> Tensor | List[Tensor]: ...

    def list_operators(self) -> List[str]: ...
    def unregister(self, name: str) -> bool: ...
    def metadata(self, name: str) -> Dict[str, object]: ...
    def describe(self, name: str) -> str: ...

ops: _OpsModule

text: ModuleType

theory: ModuleType

scale_stack: ModuleType

rl: ModuleType

robotics: ModuleType

class _SotModule(ModuleType):
    SoT3DPlan: type[SoT3DPlan]
    SoT3DStep: type[SoT3DStep]
    MacroSummary: type[MacroSummary]

    def generate_plan(
        total_steps: int,
        base_radius: float = ...,
        radial_growth: float = ...,
        base_height: float = ...,
        meso_gain: float = ...,
        micro_gain: float = ...,
    ) -> SoT3DPlan: ...

    def pack_tribonacci_chunks(length: int) -> List[int]: ...
    def pack_tetranacci_chunks(length: int) -> List[int]: ...
    def fibonacci_pacing(total_steps: int) -> List[int]: ...
    def golden_angle() -> float: ...
    def golden_ratio() -> float: ...

sot: _SotModule

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
    FractalCanvas: type[FractalCanvas]
    InfiniteZSpacePatch: type[InfiniteZSpacePatch]

vision: _VisionModule

class _CanvasModule(ModuleType):
    CanvasTransformer: type[CanvasTransformer]
    CanvasSnapshot: type[CanvasSnapshot]
    CanvasProjector: type[CanvasProjector]
    CanvasZSpacePatch: type[CanvasZSpacePatch]
    CanvasWasmTrail: type[CanvasWasmTrail]

    def apply_vision_update(
        vision: SpiralTorchVision,
        canvas: CanvasTransformer,
        *,
        hypergrad: Sequence[Sequence[float]] | None = ...,
        realgrad: Sequence[Sequence[float]] | None = ...,
        weight: float = ...,
        include_patch: bool = ...,
    ) -> CanvasSnapshot: ...

    def available_palettes() -> List[str]: ...
    def canonical_palette(name: str) -> str: ...
    def emit_zspace_patch_dict(
        projector: CanvasProjector,
        *,
        coherence: float = ...,
        tension: float = ...,
        depth: int = ...,
    ) -> Dict[str, object]: ...
    def emit_zspace_patch_packet(
        projector: CanvasProjector,
        *,
        coherence: float = ...,
        tension: float = ...,
        depth: int = ...,
    ) -> CanvasZSpacePatch: ...
    def emit_wasm_trail_dict(
        projector: CanvasProjector,
        curvature: float = ...,
    ) -> Dict[str, object]: ...
    def emit_wasm_trail_packet(
        projector: CanvasProjector,
        curvature: float = ...,
    ) -> CanvasWasmTrail: ...

canvas: _CanvasModule

class _QrModule(ModuleType):
    QuantumOverlayConfig: type[QuantumOverlayConfig]
    ZResonance: type[ZResonance]
    ZOverlayCircuit: type[ZOverlayCircuit]
    QuantumMeasurement: type[QuantumMeasurement]
    QuantumRealityStudio: type[QuantumRealityStudio]

qr: _QrModule

class _JuliaModule(ModuleType):
    ZTigerOptim: type[ZTigerOptim]

    def tempo_latency_score(self, tile: int, slack: int) -> float: ...

julia: _JuliaModule

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
        kind: Literal["topk", "midk", "bottomk", "fft"],
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
    ) -> List[SoT3DPlan]: ...

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

class EpsilonGreedy:
    def __init__(self, start: float, end: float, decay_steps: int) -> None: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def decay_steps(self) -> int: ...
    @property
    def step(self) -> int: ...
    def value(self) -> float: ...
    def advance(self) -> float: ...

class Replay:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        prioritized: bool = ...,
        alpha: float = ...,
        beta0: float = ...,
    ) -> None: ...
    @property
    def capacity(self) -> int: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def prioritized(self) -> bool: ...
    @property
    def alpha(self) -> float: ...
    @property
    def beta0(self) -> float: ...

class AgentConfig:
    def __init__(
        self,
        algo: str,
        state_dim: int,
        action_dim: int,
        gamma: float,
        lr: float,
        exploration: EpsilonGreedy | None = ...,
        optimizer: str = ...,
        clip_grad: float | None = ...,
        replay: Replay | None = ...,
        target_sync: int | None = ...,
        n_step: int | None = ...,
        seed: int | None = ...,
    ) -> None: ...
    @property
    def algo(self) -> str: ...
    @property
    def state_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def gamma(self) -> float: ...
    @property
    def lr(self) -> float: ...
    @property
    def optimizer(self) -> str: ...
    @property
    def clip_grad(self) -> float | None: ...
    @property
    def replay(self) -> Replay | None: ...
    @property
    def target_sync(self) -> int | None: ...
    @property
    def n_step(self) -> int | None: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def exploration(self) -> EpsilonGreedy | None: ...

class Agent:
    def __init__(self, config: AgentConfig) -> None: ...
    @property
    def config(self) -> AgentConfig: ...
    @property
    def algo(self) -> str: ...
    def select_action(self, state: int) -> int: ...
    def select_actions(self, states: Sequence[int]) -> List[int]: ...
    def update(self, state: int, action: int, reward: float, next_state: int) -> None: ...
    def update_batch(
        self,
        states: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        next_states: Sequence[int],
        dones: Optional[Sequence[bool]] = ...,
    ) -> None: ...
    @property
    def epsilon(self) -> float: ...
    def epsilon(self) -> float: ...
    def set_epsilon(self, epsilon: float) -> None: ...
    def set_exploration(self, schedule: EpsilonGreedy) -> None: ...
    def state_dict(self) -> Dict[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...

class stAgent:
    def __init__(self, state_dim: int, action_dim: int, discount: float, learning_rate: float) -> None: ...
    def select_action(self, state: int) -> int: ...
    def select_actions(self, states: Sequence[int]) -> List[int]: ...
    def update(self, state: int, action: int, reward: float, next_state: int) -> None: ...
    def update_batch(
        self,
        states: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        next_states: Sequence[int],
        dones: Optional[Sequence[bool]] = ...,
    ) -> None: ...
    @property
    def epsilon(self) -> float: ...
    def epsilon(self) -> float: ...
    def set_epsilon(self, epsilon: float) -> None: ...
    def set_exploration(self, schedule: EpsilonGreedy) -> None: ...
    def state_dict(self) -> Dict[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...

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

class AtlasMetric:
    name: str
    value: float
    district: Optional[str]

class AtlasFragment:
    def __init__(self, timestamp: float | None = ...) -> None: ...
    @property
    def timestamp(self) -> float | None: ...
    def set_timestamp(self, timestamp: float) -> None: ...
    def is_empty(self) -> bool: ...
    def push_metric(self, name: str, value: float, district: str | None = ...) -> None: ...
    def push_note(self, note: str) -> None: ...
    def metrics(self) -> List[AtlasMetric]: ...
    def notes(self) -> List[str]: ...
    def to_frame(self) -> AtlasFrame | None: ...

class AtlasFrame:
    @staticmethod
    def from_metrics(
        metrics: Mapping[str, float],
        *,
        timestamp: float | None = ...,
    ) -> AtlasFrame: ...
    @property
    def timestamp(self) -> float: ...
    def metric_value(self, name: str) -> float | None: ...
    def metrics_with_prefix(self, prefix: str) -> List[AtlasMetric]: ...
    def metrics(self) -> List[AtlasMetric]: ...
    def notes(self) -> List[str]: ...
    def districts(self) -> List[Dict[str, object]]: ...

class AtlasRoute:
    def __init__(self) -> None: ...
    def is_empty(self) -> bool: ...
    def len(self) -> int: ...
    def __len__(self) -> int: ...
    def latest_timestamp(self) -> float | None: ...
    def push_bounded(self, frame: AtlasFrame, bound: int) -> None: ...
    def summary(self) -> Dict[str, object]: ...
    def perspective_for(
        self, district: str, focus_prefixes: Optional[Sequence[str]] = ...
    ) -> Optional[Dict[str, object]]: ...
    def beacons(self, limit: int = ...) -> List[Dict[str, object]]: ...

class AtlasMetricFocus:
    name: str
    coverage: int
    mean: float
    latest: float
    delta: float
    momentum: float
    std_dev: float

class AtlasPerspective:
    district: str
    coverage: int
    mean: float
    latest: float
    delta: float
    momentum: float
    volatility: float
    stability: float
    guidance: str
    focus: List[AtlasMetricFocus]

    def to_dict(self) -> Dict[str, object]: ...

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

class ZSpaceSpinBand:
    name: str
    label: str

class ZSpaceRadiusBand:
    name: str
    label: str

class ZSpaceRegionKey:
    spin: ZSpaceSpinBand
    radius: ZSpaceRadiusBand
    label: str

class ZSpaceRegionDescriptor:
    spin_alignment: float
    normalized_radius: float
    curvature_radius: float
    geodesic_radius: float
    sheet_index: int
    sheet_count: int
    topological_sector: int
    @property
    def key(self) -> ZSpaceRegionKey: ...
    @property
    def spin_band(self) -> ZSpaceSpinBand: ...
    @property
    def radius_band(self) -> ZSpaceRadiusBand: ...
    @property
    def label(self) -> str: ...

class SoftlogicEllipticSample:
    curvature_radius: float
    geodesic_radius: float
    normalized_radius: float
    spin_alignment: float
    sheet_index: int
    sheet_position: float
    normal_bias: float
    sheet_count: int
    topological_sector: int
    homology_index: int
    rotor_field: tuple[float, float, float]
    flow_vector: tuple[float, float, float]
    curvature_tensor: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    resonance_heat: float
    noise_density: float
    quaternion: tuple[float, float, float, float]
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    def region_descriptor(self) -> ZSpaceRegionDescriptor: ...

class SoftlogicZFeedback:
    psi_total: float
    weighted_loss: float
    band_energy: tuple[float, float, float]
    drift: float
    z_signal: float
    scale: tuple[float, float] | None
    events: List[str]
    attributions: List[tuple[str, float]]
    elliptic: SoftlogicEllipticSample | None
    region: ZSpaceRegionDescriptor | None
    def has_event(self, tag: str) -> bool: ...
    def spin_band(self) -> ZSpaceSpinBand | None: ...
    def radius_band(self) -> ZSpaceRadiusBand | None: ...
    def label(self) -> str | None: ...

class CurvatureDecision:
    raw_pressure: float
    smoothed_pressure: float
    curvature: float
    changed: bool

class CurvatureScheduler:
    def __init__(
        self,
        *,
        initial: float | None = ...,
        min_curvature: float | None = ...,
        max_curvature: float | None = ...,
        min: float | None = ...,
        max: float | None = ...,
        target_pressure: float = ...,
        step: float | None = ...,
        tolerance: float | None = ...,
        smoothing: float | None = ...,
    ) -> None: ...
    @property
    def current(self) -> float: ...
    @property
    def min_curvature(self) -> float: ...
    @property
    def max_curvature(self) -> float: ...
    @property
    def target_pressure(self) -> float: ...
    @property
    def step(self) -> float: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def smoothing(self) -> float: ...
    @property
    def last_pressure(self) -> float | None: ...
    def set_bounds(self, min_curvature: float, max_curvature: float) -> None: ...
    def set_target_pressure(self, target: float) -> None: ...
    def set_step(self, step: float) -> None: ...
    def set_tolerance(self, tolerance: float) -> None: ...
    def set_smoothing(self, smoothing: float) -> None: ...
    def sync(self, curvature: float) -> None: ...
    def observe(self, summary: GradientSummary) -> CurvatureDecision: ...
    def observe_pressure(self, raw_pressure: float) -> CurvatureDecision: ...

def zspace_snapshot() -> Optional[ZSpaceRegionDescriptor]: ...
def softlogic_feedback() -> Optional[SoftlogicZFeedback]: ...
def describe_zspace(*, latest: bool = ..., feedback: bool = ...) -> Optional[object]: ...
def softlogic_signal() -> Optional[dict[str, object]]: ...

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
    "ZConv",
    "ZConv6DA",
    "ZPooling",
    "Pool2d",
    "compat",
    "capture",
    "share",
    "from_dlpack",
    "to_dlpack",
    "nn",
    "frac",
    "dataset",
    "linalg",
    "psi",
    "text",
    "theory",
    "scale_stack",
    "rl",
    "robotics",
    "spiral_rl",
    "rec",
    "telemetry",
    "ecosystem",
    "selfsup",
    "planner",
    "sot",
    "zspace",
    "vision",
    "canvas",
    "qr",
    "julia",
    "compat",
    "set_global_seed",
    "golden_ratio",
    "golden_angle",
    "fibonacci_pacing",
    "pack_nacci_chunks",
    "pack_tribonacci_chunks",
    "pack_tetranacci_chunks",
    "generate_plan_batch_ex",
    "SoT3DPlan",
    "SoT3DStep",
    "MacroSummary",
    "info_nce",
    "masked_mse",
    "gl_coeffs_adaptive",
    "fracdiff_gl_1d",
    "zspace_eval",
    "zspace_snapshot",
    "softlogic_feedback",
    "describe_zspace",
    "softlogic_signal",
    "QueryPlan",
    "RecEpochReport",
    "ContextualLagrangianGate",
    "ContextualPulseFrame",
    "Recommender",
    "Agent",
    "AgentConfig",
    "EpsilonGreedy",
    "Replay",
    "stAgent",
    "DqnAgent",
    "CurvatureScheduler",
    "CurvatureDecision",
    "ZSpaceSpinBand",
    "ZSpaceRadiusBand",
    "ZSpaceRegionKey",
    "ZSpaceRegionDescriptor",
    "SoftlogicEllipticSample",
    "SoftlogicZFeedback",
    "PpoAgent",
    "SacAgent",
    "TemporalResonanceBuffer",
    "SpiralTorchVision",
    "SliceProfile",
    "CanvasTransformer",
    "CanvasSnapshot",
    "FractalCanvas",
    "InfiniteZSpacePatch",
    "QuantumOverlayConfig",
    "ZResonance",
    "ZOverlayCircuit",
    "QuantumMeasurement",
    "QuantumRealityStudio",
    "FractalQuantumSession",
    "resonance_from_fractal_patch",
    "quantum_measurement_from_fractal",
    "quantum_measurement_from_fractal_sequence",
    "ZTigerOptim",
    "tempo_latency_score",
    "apply_vision_update",
    "DashboardMetric",
    "DashboardEvent",
    "DashboardFrame",
    "DashboardRing",
]
