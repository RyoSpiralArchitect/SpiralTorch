#!/usr/bin/env python3
"""Export Spiral Self-Supervised checkpoints into the st-model-hub layout."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BINDINGS = REPO_ROOT / "bindings" / "st-py" / "python"
if str(PYTHON_BINDINGS) not in sys.path:
    sys.path.append(str(PYTHON_BINDINGS))

try:  # pragma: no cover - optional convenience import for downstream validation
    import spiraltorch as st  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 - best effort import only used for validation
    st = None


@dataclass
class Matrix:
    """Lightweight container describing a 2D tensor."""

    rows: int
    cols: int
    data: list[float]

    @classmethod
    def from_nested(cls, values: Iterable[Iterable[float]] | Iterable[float]) -> "Matrix":
        items = list(values)
        if not items:
            raise ValueError("Matrix must contain at least one row")
        first = items[0]
        if isinstance(first, Iterable) and not isinstance(first, (bytes, bytearray, str)):
            rows: list[list[float]] = []
            expected_len: int | None = None
            for row in items:
                if not isinstance(row, Iterable) or isinstance(row, (bytes, bytearray, str)):
                    raise TypeError("Expected nested iterable for matrix rows")
                normalized = [float(x) for x in row]
                if expected_len is None:
                    expected_len = len(normalized)
                elif len(normalized) != expected_len:
                    raise ValueError("All rows must share the same length")
                rows.append(normalized)
            cols = expected_len or 0
            flat = [value for row in rows for value in row]
            return cls(len(rows), cols, flat)

        normalized = [float(value) for value in items]
        return cls(1, len(normalized), normalized)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Matrix":
        if "weights" in mapping and "shape" in mapping:
            rows, cols = mapping["shape"]
            data = [float(value) for value in mapping["weights"]]
            if len(data) != rows * cols:
                raise ValueError("weights length does not match declared shape")
            return cls(int(rows), int(cols), data)
        if {"rows", "cols", "data"}.issubset(mapping):
            rows = int(mapping["rows"])
            cols = int(mapping["cols"])
            data = [float(value) for value in mapping["data"]]
            if len(data) != rows * cols:
                raise ValueError("data length does not match declared shape")
            return cls(rows, cols, data)
        if "values" in mapping and "rows" in mapping and "cols" in mapping:
            rows = int(mapping["rows"])
            cols = int(mapping["cols"])
            nested = mapping["values"]
            if isinstance(nested, list):
                matrix = cls.from_nested(nested)
                if matrix.rows != rows or matrix.cols != cols:
                    raise ValueError("values shape does not match rows/cols metadata")
                return matrix
        raise ValueError("Unsupported matrix encoding in checkpoint")

    def as_state_dict_entry(self) -> dict[str, Any]:
        return {"rows": self.rows, "cols": self.cols, "data": self.data}


@dataclass
class LinearHead:
    weight: Matrix
    bias: Matrix


@dataclass
class SelfSupCheckpoint:
    encoder: Matrix
    projector: Matrix
    head: LinearHead | Matrix | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "SelfSupCheckpoint":
        def extract_matrix(key: str) -> Matrix:
            if key not in payload:
                raise KeyError(f"Missing required key: {key}")
            value = payload[key]
            if isinstance(value, dict):
                return Matrix.from_mapping(value)
            if isinstance(value, list):
                return Matrix.from_nested(value)
            raise TypeError(f"Expected mapping or nested list for {key}")

        encoder = extract_matrix("encoder")
        projector_key = "projector" if "projector" in payload else "projection"
        projector = extract_matrix(projector_key)

        head_value: LinearHead | Matrix | None = None
        for candidate in ("linear_head", "classification_head", "head"):
            if candidate in payload:
                value = payload[candidate]
                if isinstance(value, dict) and {"weight", "bias"}.issubset(value):
                    head_value = LinearHead(
                        weight=_coerce_matrix(value["weight"]),
                        bias=_coerce_matrix(value["bias"], allow_vector=True),
                    )
                elif isinstance(value, dict):
                    head_value = Matrix.from_mapping(value)
                else:
                    head_value = Matrix.from_nested(value)
                break

        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be an object when provided")

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object when provided")

        return cls(encoder=encoder, projector=projector, head=head_value, metrics=metrics, metadata=metadata)


def _load_checkpoint(path: Path) -> SelfSupCheckpoint:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        candidate = path / "checkpoint.json"
        if not candidate.exists():
            raise FileNotFoundError("Directory checkpoints must contain checkpoint.json")
        path = candidate
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must decode to a JSON object")
    return SelfSupCheckpoint.parse(payload)


def _build_manifest(args: argparse.Namespace, ckpt: SelfSupCheckpoint) -> dict[str, Any]:
    encoder_dim = (ckpt.encoder.rows, ckpt.encoder.cols)
    projector_dim = (ckpt.projector.rows, ckpt.projector.cols)
    if isinstance(ckpt.head, LinearHead):
        head_dim = (ckpt.head.weight.rows, ckpt.head.weight.cols)
    elif isinstance(ckpt.head, Matrix):
        head_dim = (ckpt.head.rows, ckpt.head.cols)
    else:
        head_dim = None
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    downstream_tasks = [
        "st-vision/classification",
        "st-nn/linear-probe",
        "st-nn/fine-tune",
    ]
    manifest = {
        "format": "st-model-hub",
        "format_version": "1.0",
        "family": "spiral-selfsup",
        "variant": args.variant,
        "objective": args.objective,
        "source_checkpoint": str(args.checkpoint),
        "created_at": timestamp,
        "encoder": {"rows": encoder_dim[0], "cols": encoder_dim[1]},
        "projector": {"rows": projector_dim[0], "cols": projector_dim[1]},
        "head": {"rows": head_dim[0], "cols": head_dim[1]} if head_dim else None,
        "downstream": {
            "compatible": downstream_tasks,
            "recommended_head": {
                "type": "linear",
                "num_classes": args.num_classes,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "total_epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        },
        "metrics": ckpt.metrics,
        "checkpoint_metadata": ckpt.metadata,
    }
    return manifest


def _write_module_snapshot(matrix: Matrix, name: str) -> dict[str, Any]:
    weight_key = f"{name}::weight"
    stored = {weight_key: matrix.as_state_dict_entry()}
    return {"parameters": stored}


def _coerce_matrix(value: Any, *, allow_vector: bool = False) -> Matrix:
    if isinstance(value, Matrix):
        return value
    if isinstance(value, dict):
        return Matrix.from_mapping(value)
    if isinstance(value, list):
        if allow_vector and value and not isinstance(value[0], (list, tuple)):
            return Matrix.from_nested([value])
        return Matrix.from_nested(value)
    raise TypeError("Expected matrix-compatible structure")


def _write_head_snapshot(head: LinearHead | Matrix) -> dict[str, Any]:
    if isinstance(head, LinearHead):
        stored = {
            "linear_probe::weight": head.weight.as_state_dict_entry(),
            "linear_probe::bias": head.bias.as_state_dict_entry(),
        }
        return {"parameters": stored}

    rows, cols = head.rows, head.cols
    data = head.data
    if len(data) != rows * cols:
        raise ValueError("Head matrix data length mismatch")
    if rows < 2:
        raise ValueError("Head matrix must provide weight rows and bias row")
    weight_elements = (rows - 1) * cols
    weight_matrix = Matrix(rows - 1, cols, data[:weight_elements])
    bias_matrix = Matrix(1, cols, data[weight_elements:])
    stored = {
        "linear_probe::weight": weight_matrix.as_state_dict_entry(),
        "linear_probe::bias": bias_matrix.as_state_dict_entry(),
    }
    return {"parameters": stored}


def export_selfsup(args: argparse.Namespace) -> None:
    ckpt = _load_checkpoint(args.checkpoint)
    output_dir = args.output.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory {output_dir} already exists. Use --overwrite to replace it.")
    else:
        output_dir.mkdir(parents=True)

    manifest = _build_manifest(args, ckpt)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    encoder_snapshot = _write_module_snapshot(ckpt.encoder, "encoder")
    (output_dir / "encoder.json").write_text(json.dumps(encoder_snapshot, indent=2))

    projector_snapshot = _write_module_snapshot(ckpt.projector, "projector")
    (output_dir / "projector.json").write_text(json.dumps(projector_snapshot, indent=2))

    if ckpt.head is not None:
        head_snapshot = _write_head_snapshot(ckpt.head)
        (output_dir / "linear_head.json").write_text(json.dumps(head_snapshot, indent=2))

    summary = {
        "output": str(output_dir),
        "has_head": ckpt.head is not None,
        "encoder_shape": [ckpt.encoder.rows, ckpt.encoder.cols],
        "projector_shape": [ckpt.projector.rows, ckpt.projector.cols],
    }
    if isinstance(ckpt.head, LinearHead):
        summary["head_shape"] = [ckpt.head.weight.rows, ckpt.head.weight.cols]
    elif isinstance(ckpt.head, Matrix):
        summary["head_shape"] = [ckpt.head.rows, ckpt.head.cols]
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the spiral-selfsup checkpoint (JSON)")
    parser.add_argument("output", type=Path, help="Directory to write the st-model-hub artefact")
    parser.add_argument("--variant", default="resnet50-contrastive", help="Model variant name")
    parser.add_argument("--objective", default="info_nce", help="Self-supervised objective used during training")
    parser.add_argument("--num-classes", type=int, default=1000, help="Default downstream number of classes")
    parser.add_argument("--lr", type=float, default=0.005, help="Recommended learning rate for linear probing")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Recommended weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear probe warmup epochs")
    parser.add_argument("--epochs", type=int, default=90, help="Total fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Recommended batch size")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    export_selfsup(args)


if __name__ == "__main__":
    main()
