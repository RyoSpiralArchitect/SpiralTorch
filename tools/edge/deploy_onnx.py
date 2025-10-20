#!/usr/bin/env python3
"""Export pipeline targeting ONNX Runtime deployments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "bindings" / "st-py" / "python"))
sys.path.append(str(Path(__file__).resolve().parent))

from spiral.export import DeploymentTarget, ExportConfig, ExportPipeline
from runtime import OnnxRuntimeEmulator


def _load_weights(path: Path | None) -> Sequence[float]:
    if path is None:
        return [((i * 3) % 17) / 19.0 for i in range(128)]
    data = json.loads(path.read_text())
    return data.get("weights", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory to store artefacts")
    parser.add_argument("--weights", type=Path, help="Optional JSON file containing weights")
    parser.add_argument("--bit-width", type=int, default=8)
    parser.add_argument("--target-sparsity", type=float, default=0.4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--latency-budget", type=float, default=10.0)
    parser.add_argument("--benchmark-iters", type=int, default=256)
    parser.add_argument("--name", type=str, default="spiraltorch-onnx")
    parser.add_argument("--disable-pruning", action="store_true")
    args = parser.parse_args()

    weights = _load_weights(args.weights)
    config = ExportConfig(
        quantization_bit_width=args.bit_width,
        target_sparsity=args.target_sparsity,
        pruning_block_size=args.block_size,
        latency_budget_ms=args.latency_budget,
        name=args.name,
    )

    pipeline = ExportPipeline(weights, config)
    compression = pipeline.run(apply_pruning=not args.disable_pruning)
    benchmark = pipeline.benchmark(iterations=args.benchmark_iters)
    artefact = pipeline.export(args.output, DeploymentTarget.ONNX)

    runtime = OnnxRuntimeEmulator(artefact)
    sample_output = runtime.run(weights[: len(runtime.weights)])

    summary = {
        "artefact": str(artefact),
        "compression": compression,
        "benchmark": benchmark,
        "sample_output": sample_output,
    }

    (args.output / "onnx-benchmark.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
