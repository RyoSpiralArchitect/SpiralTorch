"""Model export pipeline orchestrating quantisation and pruning passes."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from spiraltorch import export as _export


class DeploymentTarget(str, Enum):
    """Supported on-device runtimes."""

    TFLITE = "tflite"
    ONNX = "onnx"


@dataclass
class ExportConfig:
    """Configuration driving the compression and export pipeline."""

    quantization_bit_width: int = 8
    ema_decay: float = 0.9
    clamp_value: Optional[float] = 6.0
    epsilon: float = 1e-6
    symmetric: bool = True
    pruning_block_size: int = 32
    target_sparsity: float = 0.5
    min_l2_keep: float = 1e-4
    latency_budget_ms: float = 12.0
    name: str = "spiraltorch-model"


class ExportPipeline:
    """Runs QAT + pruning before materialising deployment artefacts."""

    def __init__(self, weights: Sequence[float], config: ExportConfig):
        self._config = config
        self._original_weights: List[float] = [float(w) for w in weights]
        self._weights: List[float] = list(self._original_weights)
        self._observer = _export.PyQatObserver(
            bit_width=config.quantization_bit_width,
            ema_decay=config.ema_decay,
            clamp_value=config.clamp_value,
            epsilon=config.epsilon,
            symmetric=config.symmetric,
        )
        self._quant_report: Optional[Dict[str, float]] = None
        self._prune_report: Optional[Dict[str, float]] = None
        self._compression_report: Optional[Dict[str, float]] = None
        self._benchmark_report: Optional[Dict[str, float]] = None

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def quant_report(self) -> Optional[Dict[str, float]]:
        return None if self._quant_report is None else dict(self._quant_report)

    @property
    def prune_report(self) -> Optional[Dict[str, float]]:
        return None if self._prune_report is None else dict(self._prune_report)

    @property
    def compression_report(self) -> Optional[Dict[str, float]]:
        return None if self._compression_report is None else dict(self._compression_report)

    @property
    def benchmark_report(self) -> Optional[Dict[str, float]]:
        return None if self._benchmark_report is None else dict(self._benchmark_report)

    def run(self, apply_pruning: bool = True) -> Dict[str, float]:
        pruning_cfg = (
            self._config.pruning_block_size,
            self._config.target_sparsity,
            self._config.min_l2_keep,
        ) if apply_pruning else None
        weights, compression = _export.compress_weights(
            self._weights,
            self._observer,
            pruning_cfg,
            latency_hint=max(0.05, self._config.latency_budget_ms / 100.0),
        )
        self._weights = list(weights)
        compression_dict = dict(compression.as_dict())
        self._compression_report = compression_dict
        if "quantization" in compression_dict:
            self._quant_report = dict(compression_dict["quantization"])
        if "pruning" in compression_dict:
            self._prune_report = dict(compression_dict["pruning"])
        return compression_dict

    def benchmark(self, iterations: int = 1000) -> Dict[str, float]:
        if self._compression_report is None:
            raise RuntimeError("call run() before collecting benchmarks")
        start = time.perf_counter()
        acc = 0.0
        for _ in range(iterations):
            acc += sum(w * w for w in self._weights)
        elapsed = (time.perf_counter() - start) / max(1, iterations)
        simulated_latency = max(0.1, elapsed * 1000.0)
        theoretical = self._compression_report.get("estimated_latency_reduction", 0.0)
        realised = max(0.0, 1.0 - simulated_latency / (self._config.latency_budget_ms + 1e-6))
        self._benchmark_report = {
            "iterations": iterations,
            "average_latency_ms": simulated_latency,
            "realised_speedup": realised,
            "theoretical_speedup": theoretical,
            "accumulator": acc,
        }
        return dict(self._benchmark_report)

    def export(self, directory: Path, target: DeploymentTarget) -> Path:
        if self._compression_report is None:
            raise RuntimeError("run() must be executed prior to export")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artefact = {
            "config": asdict(self._config),
            "compression_report": self._compression_report,
            "benchmark_report": self._benchmark_report,
            "weights": self._weights,
            "target": target.value,
        }
        output_path = directory / f"{self._config.name}.{target.value}.json"
        output_path.write_text(json.dumps(artefact, indent=2))
        return output_path

    def generate_report(self) -> Dict[str, float]:
        report = {
            "config": asdict(self._config),
            "compression": self._compression_report,
            "quantization": self._quant_report,
            "pruning": self._prune_report,
            "benchmark": self._benchmark_report,
        }
        return report


def load_benchmark_report(path: Path | str) -> Dict[str, float]:
    path = Path(path)
    data = json.loads(path.read_text())
    return data.get("benchmark_report", {})


__all__ = [
    "DeploymentTarget",
    "ExportConfig",
    "ExportPipeline",
    "load_benchmark_report",
]
