#!/usr/bin/env python3
"""Generate a Markdown storyboard from Quantum Reality Studio exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_frames(path: Path) -> List[Dict[str, object]]:
    if path.suffix == ".ndjson":
        frames = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
        return frames
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "frames" in data:
        return list(data["frames"])
    if isinstance(data, list):
        return list(data)
    raise ValueError("Unsupported storyboard input format")


def load_annotations(path: Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Annotations file must map channels to tag lists")
    annotations: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            annotations[key] = [str(tag) for tag in value if str(tag).strip()]
    return annotations


def summarise_frames(
    frames: Iterable[Dict[str, object]],
    annotations: Dict[str, List[str]],
) -> Tuple[List[str], int]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for frame in frames:
        channel = str(frame.get("channel", "unknown"))
        grouped.setdefault(channel, []).append(frame)
    total = 0
    lines: List[str] = ["# Quantum Reality Storyboard", ""]
    for channel in sorted(grouped):
        lines.append(f"## Channel `{channel}`")
        tags = annotations.get(channel)
        if tags:
            lines.append(f"Lexicon tags: {', '.join(tags)}")
        lines.append("")
        lines.append("| Ordinal | Timestamp (s) | Z-score | Z-bias | Band Energy |")
        lines.append("|---:|---:|---:|---:|:--|")
        for frame in sorted(grouped[channel], key=lambda f: f.get("ordinal", 0)):
            ordinal = int(frame.get("ordinal", total))
            timestamp = float(frame.get("timestamp", 0.0))
            z_score = float(frame.get("z_score", 0.0))
            z_bias = float(frame.get("z_bias", 0.0))
            band_energy = frame.get("band_energy", (0.0, 0.0, 0.0))
            lines.append(
                f"| {ordinal} | {timestamp:.3f} | {z_score:+.3f} | {z_bias:+.3f} | {band_energy} |"
            )
            total += 1
        lines.append("")
    return lines, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a storyboard JSON/NDJSON export")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional JSON file mapping channels to narrative tag lists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the Markdown storyboard",
    )
    args = parser.parse_args()

    frames = load_frames(args.input)
    annotations = load_annotations(args.annotations)
    lines, total = summarise_frames(frames, annotations)

    document = "\n".join(lines)
    if args.output:
        args.output.write_text(document, encoding="utf-8")
    else:
        print(document)

    print(f"Generated storyboard for {total} frames", flush=True)


if __name__ == "__main__":
    main()
