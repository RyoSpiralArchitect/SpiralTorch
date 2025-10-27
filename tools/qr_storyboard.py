#!/usr/bin/env python3
"""Generate a Markdown storyboard from Quantum Reality Studio exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion to float, ignoring empty or invalid entries."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion to int, ignoring empty or invalid entries."""

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return int(candidate)
        except ValueError:
            return None
    return None


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


def _format_overlay(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    glyphs = data.get("glyphs") or []
    formatted: List[str] = []
    if isinstance(glyphs, list):
        for glyph in glyphs:
            if not isinstance(glyph, dict):
                continue
            name = str(glyph.get("glyph", "")).strip()
            if not name:
                continue
            intensity = _coerce_float(glyph.get("intensity"))
            formatted.append(f"{name}({intensity:.2f})" if intensity is not None else name)
            if len(formatted) == 4:
                break
    if not formatted and data.get("glyph"):
        name = str(data["glyph"]).strip()
        if name:
            intensity = _coerce_float(data.get("intensity"))
            formatted.append(f"{name}({intensity:.2f})" if intensity is not None else name)
    return ", ".join(formatted)


def _format_narrative(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    tags = [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()]
    if not tags:
        return ""
    intensity = _coerce_float(data.get("intensity"))
    if intensity is not None:
        return f"{'/'.join(tags)} @ {intensity:.2f}"
    return "/".join(tags)


def _format_meta(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    tags = [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()]
    event = str(data.get("event_id", "")).strip()
    pieces: List[str] = []
    if event:
        pieces.append(f"event={event}")
    if tags:
        pieces.append("tags=" + "/".join(tags))
    intensity = _coerce_float(data.get("intensity"))
    if intensity is not None:
        pieces.append(f"intensity={intensity:.2f}")
    signature = data.get("signature")
    preview_parts: List[str] = []
    if isinstance(signature, list):
        for value in signature:
            coerce = _coerce_float(value)
            if coerce is None:
                continue
            preview_parts.append(f"{coerce:+.2f}")
            if len(preview_parts) == 3:
                break
    if preview_parts:
        suffix = ", …" if isinstance(signature, list) and len(signature) > len(preview_parts) else ""
        pieces.append(f"sheaf=[{', '.join(preview_parts)}{suffix}]")
    return "; ".join(pieces)


def _format_concept_window(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    weights = data.get("weights") or []
    formatted: List[str] = []
    if isinstance(weights, list):
        for entry in weights:
            if isinstance(entry, dict):
                idx = entry.get("index")
                weight = entry.get("weight")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                idx, weight = entry[0], entry[1]
            else:
                continue
            idx_int = _coerce_int(idx)
            weight_float = _coerce_float(weight)
            if idx_int is None or weight_float is None:
                continue
            formatted.append(f"#{idx_int}:{weight_float:.2f}")
            if len(formatted) == 4:
                break
    if not formatted:
        return ""
    magnitude = _coerce_float(data.get("magnitude"))
    suffix = f" |m|={magnitude:.2f}" if magnitude is not None else ""
    return ", ".join(formatted) + suffix


def _format_causal(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    parts: List[str] = []
    depth = _coerce_int(data.get("depth"))
    if depth is not None:
        parts.append(f"d={depth}")
    parents = [str(parent).strip() for parent in data.get("parents", []) if str(parent).strip()]
    if parents:
        parts.append("after:" + ",".join(parents))
    magnitude = _coerce_float(data.get("magnitude"))
    if magnitude is not None:
        parts.append(f"|Δ|={magnitude:.2f}")
    return " ".join(parts)


def _format_meaning_sheaf(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    signature = data.get("signature")
    if not isinstance(signature, list) or not signature:
        return ""
    preview: List[str] = []
    for value in signature:
        coerce = _coerce_float(value)
        if coerce is None:
            continue
        preview.append(f"{coerce:+.2f}")
        if len(preview) == 3:
            break
    if not preview:
        return ""
    suffix = ", …" if len(signature) > len(preview) else ""
    return f"[{', '.join(preview)}{suffix}]"


def _combine_group(label: str, parts: Iterable[str]) -> str:
    values = [part for part in parts if part]
    if not values:
        return ""
    return f"{label}: {'; '.join(values)}"


def _compose_highlights(frame: Dict[str, Any]) -> str:
    segments = [
        _combine_group(
            "Signals",
            [
                _format_overlay(frame.get("overlay")),
                _format_concept_window(frame.get("concept_window")),
            ],
        ),
        _combine_group(
            "Story",
            [
                _format_narrative(frame.get("narrative")),
                _format_meta(frame.get("meta")),
            ],
        ),
        _combine_group("Dynamics", [_format_causal(frame.get("causal"))]),
        _combine_group("Meaning", [_format_meaning_sheaf(frame.get("meaning_sheaf"))]),
    ]
    filtered = [segment for segment in segments if segment]
    return " · ".join(filtered) if filtered else "—"


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
        lines.append("| Ordinal | Timestamp (s) | Z-score | Z-bias | Band Energy | Highlights |")
        lines.append("|---:|---:|---:|---:|:--|:--|")
        for frame in sorted(grouped[channel], key=lambda f: f.get("ordinal", 0)):
            ordinal = int(frame.get("ordinal", total))
            timestamp = float(frame.get("timestamp", 0.0))
            z_score = float(frame.get("z_score", 0.0))
            z_bias = float(frame.get("z_bias", 0.0))
            band_energy = frame.get("band_energy", (0.0, 0.0, 0.0))
            if isinstance(band_energy, (list, tuple)) and len(band_energy) == 3:
                band_fmt = "[" + ", ".join(f"{float(value):+.3f}" for value in band_energy) + "]"
            else:
                band_fmt = str(band_energy)
            highlights = _compose_highlights(frame)
            lines.append(
                f"| {ordinal} | {timestamp:.3f} | {z_score:+.3f} | {z_bias:+.3f} | {band_fmt} | {highlights} |"
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
