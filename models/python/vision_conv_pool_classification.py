from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import pathlib
import random
import sys
from typing import Any

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st

import _softlogic_cli

FORMAT = "st-vision-conv-pool-classification-v1"
RUN_SCHEMA = "st.modelzoo.run.v1"


def _timestamp_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_run_dir() -> pathlib.Path:
    return _ROOT / "models" / "runs" / _timestamp_slug()


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _native_feature_flags() -> dict[str, bool]:
    try:
        info = st.build_info()
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    features = info.get("features")
    if not isinstance(features, dict):
        return {}
    out: dict[str, bool] = {}
    for key, value in features.items():
        out[str(key)] = bool(value)
    return out


def _require_backend_available(backend: str) -> None:
    backend = str(backend).strip().lower()
    if backend in {"cpu"}:
        return

    flags = _native_feature_flags()
    if backend in {"wgpu", "webgpu", "auto"}:
        if flags.get("wgpu") or flags.get("wgpu-rt"):
            return
        raise RuntimeError(
            "backend=wgpu requested but this SpiralTorch build lacks the 'wgpu' feature. "
            "Rebuild the extension with `--features wgpu` (or `wgpu-rt`)."
        )
    if backend == "cuda":
        if flags.get("cuda"):
            return
        raise RuntimeError(
            "backend=cuda requested but this SpiralTorch build lacks the 'cuda' feature. "
            "Rebuild the extension with `--features cuda`."
        )
    if backend in {"hip", "rocm"}:
        if flags.get("hip") or flags.get("hip-real"):
            return
        raise RuntimeError(
            "backend=hip requested but this SpiralTorch build lacks the 'hip' feature. "
            "Rebuild the extension with `--features hip` (or `hip-real`)."
        )
    raise ValueError(f"unknown --backend: {backend} (expected cpu|wgpu|cuda|hip|auto)")


def _build_batch(batch: int, hw: tuple[int, int], seed: int) -> tuple[st.Tensor, st.Tensor]:
    rng = random.Random(seed)
    pixels = hw[0] * hw[1]

    data: list[float] = []
    targets: list[float] = []

    for idx in range(batch):
        sample = [rng.random() * 0.20 for _ in range(pixels)]
        class_one = idx % 2 == 1
        if class_one:
            for r in range(hw[0] // 2, hw[0]):
                for c in range(hw[1] // 2, hw[1]):
                    sample[r * hw[1] + c] += 0.9
            targets.append(1.0)
        else:
            for r in range(0, hw[0] // 2):
                for c in range(0, hw[1] // 2):
                    sample[r * hw[1] + c] += 0.9
            targets.append(0.0)
        data.extend(sample)

    x = st.Tensor(batch, pixels, data)
    y = st.Tensor(batch, 1, targets)
    return x, y


def _build_model(
    input_hw: tuple[int, int],
    out_channels: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    pool_kernel: tuple[int, int],
    pool_stride: tuple[int, int],
) -> st.nn.Sequential:
    conv_out_hw = st.nn.conv_output_shape(
        input_hw, kernel, stride=stride, padding=padding, dilation=(1, 1)
    )
    pool_out_hw = st.nn.pool_output_shape(conv_out_hw, pool_kernel, stride=pool_stride, padding=(0, 0))
    features = out_channels * pool_out_hw[0] * pool_out_hw[1]

    model = st.nn.Sequential()
    model.add(
        st.nn.ZConv(
            "conv1",
            1,
            out_channels,
            input_hw,
            kernel,
            stride=stride,
            padding=padding,
            layout="NCHW",
        )
    )
    model.add(st.nn.Relu())
    model.add(
        st.nn.Pool2d(
            "max",
            out_channels,
            conv_out_hw[0],
            conv_out_hw[1],
            pool_kernel,
            stride=pool_stride,
            padding=(0, 0),
            layout="NCHW",
        )
    )
    model.add(st.nn.Relu())
    model.add(st.nn.Linear("head", features, 1))
    return model


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/vision_conv_pool_classification.py "
            "[--load weights.json] [--save weights.json] "
            "[--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] "
            "[--lr-schedule constant|linear|cosine] [--lr-min F] "
            "[--lr-warmup-epochs N] [--lr-warmup-start F] [--grad-clip F] "
            "[--hw H,W] [--out-channels N] [--kernel H,W] [--stride H,W] [--padding H,W] "
            "[--pool-kernel H,W] [--pool-stride H,W] "
            "[--backend cpu|wgpu|cuda|hip|auto] "
            "[--events PATH] [--events-types A,B,C] "
            "[--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--run-dir PATH] "
            f"{_softlogic_cli.usage_flags()}"
        )
        return

    load_weights: pathlib.Path | None = None
    save_weights: pathlib.Path | None = None
    run_dir: pathlib.Path | None = None

    epochs = 8
    batches_per_epoch = 4
    batch = 8
    lr = 2e-2
    curvature = -1.0
    lr_schedule = "constant"
    lr_min: float | None = None
    lr_warmup_epochs = 0
    lr_warmup_start: float | None = None
    grad_clip = 0.0

    input_hw = (8, 8)
    out_channels = 4
    kernel = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    pool_kernel = (2, 2)
    pool_stride = (2, 2)

    backend = "cpu"
    events_path: pathlib.Path | None = None
    events_types = [
        "EpochStart",
        "EpochEnd",
        "RoundtablePlanned",
        "TrainerStep",
        "TrainerPhase",
    ]
    atlas = False
    atlas_bound = 512
    atlas_district = "Training"

    args = list(sys.argv[1:])
    softlogic_cli = _softlogic_cli.pop_softlogic_flags(args)
    it = iter(args)
    for flag in it:
        if flag == "--load":
            load_weights = pathlib.Path(next(it))
        elif flag == "--save":
            save_weights = pathlib.Path(next(it))
        elif flag == "--epochs":
            epochs = int(next(it))
        elif flag == "--batches":
            batches_per_epoch = int(next(it))
        elif flag == "--batch":
            batch = int(next(it))
        elif flag == "--lr":
            lr = float(next(it))
        elif flag == "--lr-schedule":
            lr_schedule = str(next(it)).strip().lower()
        elif flag == "--lr-min":
            lr_min = float(next(it))
        elif flag == "--lr-warmup-epochs":
            lr_warmup_epochs = int(next(it))
        elif flag == "--lr-warmup-start":
            lr_warmup_start = float(next(it))
        elif flag == "--grad-clip":
            grad_clip = float(next(it))
        elif flag == "--curvature":
            curvature = float(next(it))
        elif flag == "--hw":
            raw = str(next(it))
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            input_hw = (int(parts[0]), int(parts[1]))
        elif flag == "--out-channels":
            out_channels = int(next(it))
        elif flag == "--kernel":
            raw = str(next(it))
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            kernel = (int(parts[0]), int(parts[1]))
        elif flag == "--stride":
            raw = str(next(it))
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            stride = (int(parts[0]), int(parts[1]))
        elif flag == "--padding":
            raw = str(next(it))
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            padding = (int(parts[0]), int(parts[1]))
        elif flag == "--pool-kernel":
            raw = str(next(it))
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            pool_kernel = (int(parts[0]), int(parts[1]))
        elif flag == "--pool-stride":
            raw = str(next(it))
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            pool_stride = (int(parts[0]), int(parts[1]))
        elif flag == "--backend":
            backend = str(next(it)).strip().lower()
        elif flag == "--events":
            events_path = pathlib.Path(next(it))
        elif flag == "--events-types":
            raw = str(next(it))
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if parts:
                events_types = parts
        elif flag == "--atlas":
            atlas = True
        elif flag == "--atlas-bound":
            atlas_bound = int(next(it))
        elif flag == "--atlas-district":
            atlas_district = str(next(it))
        elif flag == "--run-dir":
            run_dir = pathlib.Path(next(it))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if lr_schedule not in {"constant", "linear", "cosine"}:
        raise ValueError(
            f"unknown --lr-schedule: {lr_schedule} (expected constant|linear|cosine)"
        )
    if lr_min is not None and (not math.isfinite(lr_min) or lr_min <= 0.0):
        raise ValueError("--lr-min must be a positive, finite float")
    if lr_warmup_epochs < 0:
        raise ValueError("--lr-warmup-epochs must be >= 0")
    if lr_warmup_epochs > epochs:
        raise ValueError("--lr-warmup-epochs must be <= --epochs")
    if lr_warmup_start is not None and (not math.isfinite(lr_warmup_start) or lr_warmup_start <= 0.0):
        raise ValueError("--lr-warmup-start must be a positive, finite float")
    if not math.isfinite(grad_clip) or grad_clip < 0.0:
        raise ValueError("--grad-clip must be a finite float >= 0")

    _require_backend_available(backend)

    if run_dir is None:
        run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    if atlas and events_path is None:
        events_path = run_dir / "events.jsonl"

    resolved_lr_min: float | None = None
    if lr_schedule != "constant":
        resolved_lr_min = float(lr_min) if lr_min is not None else float(lr) * 0.1
    resolved_warmup_start = (
        float(lr_warmup_start)
        if lr_warmup_start is not None
        else (float(resolved_lr_min) if resolved_lr_min is not None else float(lr) * 0.1)
    )

    model = _build_model(input_hw, out_channels, kernel, stride, padding, pool_kernel, pool_stride)
    if load_weights is not None:
        st.nn.load(str(load_weights), model)

    model.attach_hypergrad(curvature=curvature, learning_rate=lr)
    trainer = st.nn.ModuleTrainer(
        backend=backend,
        curvature=curvature,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    if grad_clip > 0.0:
        trainer.set_grad_clip_max_norm(grad_clip)
    softlogic_meta = _softlogic_cli.apply_softlogic_cli(trainer, softlogic_cli)
    schedule = trainer.roundtable(
        batch,
        1,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    loss = st.nn.CrossEntropy(curvature=curvature)

    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "vision_conv_pool_classification",
        "format": FORMAT,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch": batch,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "lr_min": resolved_lr_min,
        "lr_warmup_epochs": lr_warmup_epochs if lr_warmup_epochs > 0 else None,
        "lr_warmup_start": resolved_warmup_start if lr_warmup_epochs > 0 else None,
        "grad_clip": grad_clip if grad_clip > 0.0 else None,
        "curvature": curvature,
        "backend": backend,
        "input_hw": input_hw,
        "out_channels": out_channels,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "pool_kernel": pool_kernel,
        "pool_stride": pool_stride,
        "events_path": str(events_path) if events_path is not None else None,
        "events_types": events_types,
        "atlas": atlas,
        "atlas_bound": atlas_bound if atlas else None,
        "atlas_district": atlas_district if atlas else None,
        "softlogic": softlogic_meta,
        "weights_loaded_from": str(load_weights) if load_weights is not None else None,
    }
    _write_json(run_dir / "run.json", run_meta)

    print(
        f"arch=vision_conv_pool hw={input_hw} out_ch={out_channels} kernel={kernel} pool={pool_kernel} "
        f"epochs={epochs} batch={batch} lr={lr:.3e} schedule={lr_schedule} warmup={lr_warmup_epochs} "
        f"grad_clip={grad_clip:.3g} "
        f"curvature={curvature} backend={backend} run_dir={run_dir}"
    )

    metrics_path = run_dir / "metrics.jsonl"
    record_ctx = (
        st.plugin.record(str(events_path), event_types=events_types)
        if events_path is not None
        else contextlib.nullcontext()
    )
    lr_active = float(lr)
    lr_floor = float(resolved_lr_min) if resolved_lr_min is not None else float(lr) * 0.1

    def _scheduled_lr(epoch_idx: int) -> float:
        if lr_warmup_epochs > 0 and epoch_idx < lr_warmup_epochs:
            if lr_warmup_epochs == 1:
                return float(lr)
            t = float(epoch_idx) / float(max(1, lr_warmup_epochs - 1))
            return resolved_warmup_start + (float(lr) - resolved_warmup_start) * t
        decay_epochs = max(0, epochs - lr_warmup_epochs)
        if lr_schedule == "constant" or decay_epochs <= 1:
            return float(lr)
        t = float(max(0, epoch_idx - lr_warmup_epochs)) / float(max(1, decay_epochs - 1))
        if lr_schedule == "linear":
            return float(lr) + (lr_floor - float(lr)) * t
        return lr_floor + 0.5 * (float(lr) - lr_floor) * (1.0 + math.cos(math.pi * t))

    with record_ctx:
        for epoch in range(max(0, epochs)):
            target_lr = _scheduled_lr(epoch)
            if lr_active > 0.0 and target_lr > 0.0:
                factor = float(target_lr) / float(lr_active)
                if factor == factor and abs(factor - 1.0) > 1e-9:
                    trainer.mul_learning_rate(model, factor)
                    lr_active = float(target_lr)

            batches = []
            base_seed = 777 + epoch * 10_000
            for b in range(batches_per_epoch):
                batches.append(_build_batch(batch, input_hw, seed=base_seed + b))
            stats = trainer.train_epoch(model, loss, batches, schedule)
            avg_loss = float(stats.average_loss)
            _append_jsonl(
                metrics_path,
                {
                    "epoch": epoch,
                    "batches": int(stats.batches),
                    "average_loss": avg_loss,
                    "learning_rate": lr_active,
                },
            )
            print(f"epoch[{epoch}] batches={stats.batches} lr={lr_active:.3e} avg_loss={avg_loss:.6f}")

    if atlas and events_path is not None:
        try:
            route = st.trainer_events_to_atlas_route(
                events_path,
                district=atlas_district,
                bound=atlas_bound,
            )
            _write_json(run_dir / "atlas_summary.json", route.summary())
        except Exception as exc:
            _write_json(run_dir / "atlas_summary.json", {"error": str(exc)})

    weights_path = run_dir / "weights.json"
    st.nn.save(str(weights_path), model)
    if save_weights is not None:
        save_weights.parent.mkdir(parents=True, exist_ok=True)
        st.nn.save(str(save_weights), model)


if __name__ == "__main__":
    main()
