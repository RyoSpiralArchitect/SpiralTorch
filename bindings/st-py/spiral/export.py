"""Model export pipeline orchestrating quantisation and pruning passes.

The current implementation emits JSON artefacts describing the compressed weights and
deployment metadata. Converting those artefacts into ONNX/TFLite binaries is planned.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from spiraltorch import export as _export


class DeploymentTarget(str, Enum):
    """Planned on-device runtimes (JSON scaffolding is emitted today)."""

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
    """Runs QAT + pruning before materialising JSON deployment artefacts."""

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
    "main",
]


def _parse_optional_float(text: str) -> Optional[float]:
    text = text.strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return float(text)


def _load_weights(source: str) -> List[float]:
    if source == "-":
        payload = sys.stdin.read()
    else:
        payload = Path(source).read_text()

    payload = payload.strip()
    if not payload:
        raise ValueError("weights input is empty")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [float(x) for x in parsed]

    # Fallback: whitespace/comma separated floats.
    tokens = payload.replace(",", " ").split()
    if not tokens:
        raise ValueError("weights input did not contain any numbers")
    return [float(tok) for tok in tokens]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="spiral-export",
        description=(
            "Run SpiralTorch compression (QAT observer + structured pruning) and emit a JSON artefact."
        ),
    )
    parser.add_argument(
        "weights",
        help="Path to weights (JSON array or whitespace/comma-separated floats). Use '-' for stdin.",
    )
    parser.add_argument(
        "--out",
        default="artefacts",
        help="Output directory for emitted JSON artefacts (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        default=DeploymentTarget.ONNX.value,
        choices=[t.value for t in DeploymentTarget],
        help="Deployment target label stored in the artefact.",
    )
    parser.add_argument("--name", default="spiraltorch-model", help="Artefact basename.")
    parser.add_argument("--no-prune", action="store_true", help="Skip structured pruning pass.")
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=0,
        help="If >0, run a lightweight latency simulation (default: 0).",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the export report JSON to stdout (in addition to writing the artefact).",
    )

    parser.add_argument("--bit-width", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--clamp-value", default="6.0", help="Float, or 'none' to disable clamping.")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--asymmetric", action="store_true", help="Use asymmetric quantization.")

    parser.add_argument("--pruning-block", type=int, default=32)
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--min-l2-keep", type=float, default=1e-4)
    parser.add_argument("--latency-budget-ms", type=float, default=12.0)

    args = parser.parse_args(list(argv) if argv is not None else None)

    weights = _load_weights(args.weights)
    config = ExportConfig(
        quantization_bit_width=args.bit_width,
        ema_decay=args.ema_decay,
        clamp_value=_parse_optional_float(args.clamp_value),
        epsilon=args.epsilon,
        symmetric=not args.asymmetric,
        pruning_block_size=args.pruning_block,
        target_sparsity=args.target_sparsity,
        min_l2_keep=args.min_l2_keep,
        latency_budget_ms=args.latency_budget_ms,
        name=args.name,
    )

    pipeline = ExportPipeline(weights, config)
    pipeline.run(apply_pruning=not args.no_prune)
    if args.benchmark_iterations and args.benchmark_iterations > 0:
        pipeline.benchmark(iterations=args.benchmark_iterations)

    out_path = pipeline.export(Path(args.out), DeploymentTarget(args.target))
    if args.print_report:
        print(json.dumps(pipeline.generate_report(), indent=2))
    print(out_path)
