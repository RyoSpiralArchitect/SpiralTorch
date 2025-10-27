#!/usr/bin/env python3
"""Generate a Markdown storyboard from Quantum Reality Studio exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
    for glyph in glyphs:
        if not isinstance(glyph, dict):
            continue
        name = str(glyph.get("glyph", "")).strip()
        if not name:
            continue
        intensity = float(glyph.get("intensity", 0.0))
        formatted.append(f"{name}({intensity:.2f})")
        if len(formatted) == 4:
            break
    if not formatted and data.get("glyph"):
        name = str(data["glyph"]).strip()
        if name:
            intensity = float(data.get("intensity", 0.0))
            formatted.append(f"{name}({intensity:.2f})")
    if not formatted:
        return ""
    return "Overlay: " + ", ".join(formatted)


def _format_narrative(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    tags = [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()]
    if not tags:
        return ""
    intensity = data.get("intensity")
    if intensity is not None:
        return f"Narrative: {'/'.join(tags)} @ {float(intensity):.2f}"
    return "Narrative: " + "/".join(tags)


def _format_meta(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    tags = [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()]
    event = data.get("event_id")
    pieces: List[str] = []
    if event:
        pieces.append(f"event={event}")
    if tags:
        pieces.append("tags=" + "/".join(tags))
    intensity = data.get("intensity")
    if intensity is not None:
        pieces.append(f"intensity={float(intensity):.2f}")
    signature = data.get("signature")
    if isinstance(signature, list) and signature:
        preview = ", ".join(f"{float(value):+.2f}" for value in signature[:3])
        if len(signature) > 3:
            preview += ", …"
        pieces.append(f"sheaf=[{preview}]")
    return "Meta: " + "; ".join(pieces) if pieces else ""


def _format_concept_window(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    weights = data.get("weights") or []
    formatted: List[str] = []
    for entry in weights:
        if isinstance(entry, dict):
            idx = entry.get("index")
            weight = entry.get("weight")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            idx, weight = entry[0], entry[1]
        else:
            continue
        try:
            idx_int = int(idx)
            weight_float = float(weight)
        except (TypeError, ValueError):
            continue
        formatted.append(f"#{idx_int}:{weight_float:.2f}")
        if len(formatted) == 4:
            break
    if not formatted:
        return ""
    magnitude = data.get("magnitude")
    suffix = f" |m|={float(magnitude):.2f}" if magnitude is not None else ""
    return "Concept: " + ", ".join(formatted) + suffix


def _format_causal(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    parts: List[str] = []
    depth = data.get("depth")
    if depth is not None:
        try:
            parts.append(f"d={int(depth)}")
        except (TypeError, ValueError):
            pass
    parents = [str(parent).strip() for parent in data.get("parents", []) if str(parent).strip()]
    if parents:
        parts.append("after:" + ",".join(parents))
    magnitude = data.get("magnitude")
    if magnitude is not None:
        try:
            parts.append(f"|Δ|={float(magnitude):.2f}")
        except (TypeError, ValueError):
            pass
    return "Causal: " + " ".join(parts) if parts else ""


def _format_meaning_sheaf(data: Dict[str, Any] | None) -> str:
    if not data:
        return ""
    signature = data.get("signature")
    if not isinstance(signature, list) or not signature:
        return ""
    preview = ", ".join(f"{float(value):+.2f}" for value in signature[:3])
    if len(signature) > 3:
        preview += ", …"
    return f"Sheaf: [{preview}]"


def _compose_highlights(parts: List[str]) -> str:
    filtered = [part for part in parts if part]
    return "; ".join(filtered) if filtered else "—"


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
            highlights = _compose_highlights(
                [
                    _format_overlay(frame.get("overlay")),
                    _format_narrative(frame.get("narrative")),
                    _format_meta(frame.get("meta")),
                    _format_concept_window(frame.get("concept_window")),
                    _format_causal(frame.get("causal")),
                    _format_meaning_sheaf(frame.get("meaning_sheaf")),
                ]
            )
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
