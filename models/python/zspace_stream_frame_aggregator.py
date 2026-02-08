from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from typing import Any, Iterator

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
    rng: random.Random,
) -> list[list[list[float]]]:
    frame: list[list[list[float]]] = []
    for z in range(depth):
        z_phase = 0.18 * step + 0.31 * z
        rows: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                base = math.sin(z_phase + 0.14 * y + 0.08 * x)
                contrast = math.cos(0.10 * step - 0.05 * y + 0.03 * x)
                noise = rng.uniform(-0.025, 0.025)
                row.append(float(0.65 * base + 0.35 * contrast + noise))
            rows.append(row)
        frame.append(rows)
    return frame


def _frame_energy(frame: list[list[list[float]]]) -> float:
    total = 0.0
    count = 0
    for slice_ in frame:
        for row in slice_:
            for value in row:
                total += float(value) * float(value)
                count += 1
    return total / float(count) if count else 0.0


def _make_stream_frame(
    volume: list[list[list[float]]],
    *,
    step: int,
    dt: float,
) -> Any:
    frame_cls = getattr(st.vision, "ZSpaceStreamFrame", None)
    chrono_cls = getattr(st.vision, "ChronoSnapshot", None)
    if not callable(frame_cls) or chrono_cls is None:
        return volume
    if not hasattr(chrono_cls, "from_values"):
        return volume

    energy = _frame_energy(volume)
    latest_timestamp = float(step + 1) * dt
    summary = chrono_cls.from_values(
        frames=int(step + 1),
        duration=max(dt, latest_timestamp),
        latest_timestamp=latest_timestamp,
        mean_drift=0.01 * (step + 1),
        mean_abs_drift=0.012 * (step + 1),
        drift_std=0.002 + 0.0005 * (step % 7),
        mean_energy=energy,
        energy_std=0.05 + 0.01 * (step % 5),
        mean_decay=0.03,
        min_energy=max(0.0, energy - 0.1),
        max_energy=energy + 0.1,
        dt=dt,
    )
    return frame_cls(volume, chrono_snapshot=summary)


def _iter_frames(
    *,
    steps: int,
    depth: int,
    height: int,
    width: int,
    seed: int,
    dt: float,
    native_frames: bool,
) -> Iterator[Any]:
    rng = random.Random(seed)
    for step in range(steps):
        volume = _volume_frame(
            step,
            depth=depth,
            height=height,
            width=width,
            rng=rng,
        )
        if native_frames:
            yield _make_stream_frame(volume, step=step, dt=dt)
        else:
            yield volume


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chrono/aggregator-based online stream recipe for SpiralTorchVision"
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--temporal", type=int, default=6)
    parser.add_argument("--flush-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--native-frames", action="store_true")
    parser.add_argument("--z-dim", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output", type=pathlib.Path, default=None)
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
    )

    aggregator = None
    if hasattr(st.vision, "ZSpaceStreamFrameAggregator"):
        try:
            aggregator = st.vision.ZSpaceStreamFrameAggregator(max_depth=int(args.depth))
        except Exception:
            aggregator = None

    updates = st.vision.stream_vision_training(
        vision,
        _iter_frames(
            steps=int(args.steps),
            depth=int(args.depth),
            height=int(args.height),
            width=int(args.width),
            seed=int(args.seed),
            dt=float(args.dt),
            native_frames=bool(args.native_frames),
        ),
        aggregator=aggregator,
        trainer=trainer,
        flush_every=max(1, int(args.flush_every)),
        keep_depth=int(args.depth),
        final_flush=True,
    )

    losses = [float(update["loss"]) for update in updates if update.get("loss") is not None]
    summary = {
        "recipe": "zspace_stream_frame_aggregator",
        "steps": int(args.steps),
        "updates": len(updates),
        "flush_every": int(args.flush_every),
        "native_frames": bool(args.native_frames),
        "aggregator_used": aggregator is not None,
        "final_loss": losses[-1] if losses else None,
        "final_energy": float(vision.volume_energy()),
        "last_stream_metadata": vision.last_stream_metadata,
        "z_state": trainer.state,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.output is not None:
        payload = dict(summary)
        payload["updates_trace"] = updates
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
