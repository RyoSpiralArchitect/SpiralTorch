from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from typing import Iterator

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st


def _volume_frame(
    step: int,
    *,
    depth: int,
    height: int,
    width: int,
    jitter: float,
    rng: random.Random,
) -> list[list[list[float]]]:
    frame: list[list[list[float]]] = []
    for z in range(depth):
        z_phase = 0.21 * step + 0.37 * z
        rows: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                carrier = math.sin(z_phase + 0.15 * y + 0.09 * x)
                envelope = 0.5 + 0.5 * math.cos(0.13 * step + 0.11 * y - 0.07 * x)
                noise = rng.uniform(-jitter, jitter) if jitter > 0.0 else 0.0
                row.append(float(carrier * envelope + noise))
            rows.append(row)
        frame.append(rows)
    return frame


def _iter_frames(
    *,
    steps: int,
    depth: int,
    height: int,
    width: int,
    jitter: float,
    seed: int,
) -> Iterator[list[list[list[float]]]]:
    rng = random.Random(seed)
    for step in range(steps):
        yield _volume_frame(
            step,
            depth=depth,
            height=height,
            width=width,
            jitter=jitter,
            rng=rng,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online Z-space stream recipe with SpiralTorchVision + ZSpaceTrainer"
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--temporal", type=int, default=6)
    parser.add_argument("--flush-every", type=int, default=2)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lam-frac", type=float, default=0.08)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional JSON path for full per-step update traces",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    vision = st.vision.SpiralTorchVision(
        depth=int(args.depth),
        height=int(args.height),
        width=int(args.width),
        alpha=float(args.alpha),
        temporal=int(args.temporal),
    )
    trainer = st.zspace.ZSpaceTrainer(
        z_dim=int(args.z_dim),
        lr=float(args.lr),
        lam_frac=float(args.lam_frac),
    )

    updates = st.vision.stream_vision_training(
        vision,
        _iter_frames(
            steps=int(args.steps),
            depth=int(args.depth),
            height=int(args.height),
            width=int(args.width),
            jitter=float(args.jitter),
            seed=int(args.seed),
        ),
        trainer=trainer,
        flush_every=max(1, int(args.flush_every)),
        final_flush=True,
    )

    losses = [float(update["loss"]) for update in updates if update.get("loss") is not None]
    summary = {
        "recipe": "zspace_stream_online_vision",
        "steps": int(args.steps),
        "updates": len(updates),
        "flush_every": int(args.flush_every),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
        "final_energy": float(vision.volume_energy()),
        "z_state": trainer.state,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.output is not None:
        payload = dict(summary)
        payload["updates_trace"] = updates
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
