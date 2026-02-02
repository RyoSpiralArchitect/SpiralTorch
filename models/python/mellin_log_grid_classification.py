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

FORMAT = "st-frac-mellin-log-grid-classification-v1"
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


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + step * float(i) for i in range(count)]


def _evaluate_features(
    plan: Any,
    grid: Any,
    *,
    feature: str,
    epsilon: float,
) -> list[float]:
    if feature == "mag":
        return list(plan.evaluate_magnitude(grid))
    if feature == "logmag":
        return list(plan.evaluate_log_magnitude(grid, epsilon=epsilon))
    raise ValueError(f"unknown --feature: {feature} (expected mag|logmag)")


def _build_batch(
    batch: int,
    plan: Any,
    *,
    log_start: float,
    log_step: float,
    series_len: int,
    feature: str,
    epsilon: float,
    low_rate_min: float,
    low_rate_max: float,
    high_rate_min: float,
    high_rate_max: float,
    seed: int,
) -> tuple[st.Tensor, st.Tensor]:
    rng = random.Random(seed)
    pixels = int(plan.len())
    if pixels <= 0:
        raise ValueError("plan produced zero features; check --real-count/--imag-count")

    data: list[float] = []
    targets: list[float] = []

    for idx in range(batch):
        class_one = idx % 2 == 1
        if class_one:
            rate = rng.uniform(high_rate_min, high_rate_max)
            targets.extend([0.0, 1.0])
        else:
            rate = rng.uniform(low_rate_min, low_rate_max)
            targets.extend([1.0, 0.0])

        grid = st.frac.MellinLogGrid.exp_decay_scaled(log_start, log_step, series_len, rate)
        feats = _evaluate_features(plan, grid, feature=feature, epsilon=epsilon)
        if len(feats) != pixels:
            raise RuntimeError(f"feature length mismatch: expected {pixels}, got {len(feats)}")
        data.extend(float(v) if math.isfinite(v) else 0.0 for v in feats)

    x = st.Tensor(batch, pixels, data)
    y = st.Tensor(batch, 2, targets)
    return x, y


def _build_model(
    input_hw: tuple[int, int],
    *,
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
    model.add(st.nn.Linear("head", features, 2))
    return model


def _evaluate_loss(model: st.nn.Sequential, loss: Any, batches: list[tuple[st.Tensor, st.Tensor]]) -> float:
    if not batches:
        return float("nan")
    total = 0.0
    for x, y in batches:
        pred = model.forward(x)
        value = loss.forward(pred, y).tolist()
        total += float(value[0][0])
    return total / float(len(batches))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/mellin_log_grid_classification.py "
            "[--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] "
            "[--feature mag|logmag] [--epsilon F] "
            "[--log-start F] [--log-step F] [--series-len N] "
            "[--real-min F] [--real-max F] [--real-count N] "
            "[--imag-min F] [--imag-max F] [--imag-count N] "
            "[--low-rate-min F] [--low-rate-max F] [--high-rate-min F] [--high-rate-max F] "
            "[--backend cpu|wgpu|cuda|hip|auto] "
            "[--events PATH] [--events-types A,B,C] "
            "[--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--run-dir PATH] [--seed N] [--val-batches N]"
        )
        return

    run_dir: pathlib.Path | None = None

    epochs = 8
    batches_per_epoch = 8
    batch = 8
    lr = 2e-2
    curvature = -1.0
    seed = 777

    feature = "logmag"
    epsilon = 1e-6

    log_start = -6.0
    log_step = 0.05
    series_len = 256

    real_min = 0.5
    real_max = 3.0
    real_count = 8

    imag_min = -12.0
    imag_max = 12.0
    imag_count = 16

    low_rate_min = 0.6
    low_rate_max = 1.0
    high_rate_min = 1.6
    high_rate_max = 2.6

    backend = "cpu"
    events_path: pathlib.Path | None = None
    events_types = ["EpochStart", "EpochEnd", "TrainerStep", "TrainerPhase"]
    atlas = False
    atlas_bound = 512
    atlas_district = "Training"
    val_batches = 0

    args = list(sys.argv[1:])
    it = iter(args)
    for flag in it:
        if flag == "--epochs":
            epochs = int(next(it))
        elif flag == "--batches":
            batches_per_epoch = int(next(it))
        elif flag == "--batch":
            batch = int(next(it))
        elif flag == "--lr":
            lr = float(next(it))
        elif flag == "--curvature":
            curvature = float(next(it))
        elif flag == "--seed":
            seed = int(next(it))
        elif flag == "--feature":
            feature = str(next(it)).strip().lower()
        elif flag == "--epsilon":
            epsilon = float(next(it))
        elif flag == "--log-start":
            log_start = float(next(it))
        elif flag == "--log-step":
            log_step = float(next(it))
        elif flag == "--series-len":
            series_len = int(next(it))
        elif flag == "--real-min":
            real_min = float(next(it))
        elif flag == "--real-max":
            real_max = float(next(it))
        elif flag == "--real-count":
            real_count = int(next(it))
        elif flag == "--imag-min":
            imag_min = float(next(it))
        elif flag == "--imag-max":
            imag_max = float(next(it))
        elif flag == "--imag-count":
            imag_count = int(next(it))
        elif flag == "--low-rate-min":
            low_rate_min = float(next(it))
        elif flag == "--low-rate-max":
            low_rate_max = float(next(it))
        elif flag == "--high-rate-min":
            high_rate_min = float(next(it))
        elif flag == "--high-rate-max":
            high_rate_max = float(next(it))
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
        elif flag == "--val-batches":
            val_batches = int(next(it))
        elif flag == "--run-dir":
            run_dir = pathlib.Path(next(it))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if feature not in {"mag", "logmag"}:
        raise ValueError("--feature must be mag|logmag")
    if series_len <= 1:
        raise ValueError("--series-len must be >= 2")
    if real_count <= 0 or imag_count <= 0:
        raise ValueError("--real-count/--imag-count must be >= 1")
    if not (low_rate_min > 0.0 and low_rate_max >= low_rate_min):
        raise ValueError("--low-rate-min/max must satisfy 0 < min <= max")
    if not (high_rate_min > 0.0 and high_rate_max >= high_rate_min):
        raise ValueError("--high-rate-min/max must satisfy 0 < min <= max")

    _require_backend_available(backend)

    real_values = _linspace(real_min, real_max, real_count)
    imag_values = _linspace(imag_min, imag_max, imag_count)
    plan = st.frac.MellinEvalPlan.mesh(log_start, log_step, real_values, imag_values)
    input_hw = (len(real_values), len(imag_values))

    if run_dir is None:
        run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    if atlas and events_path is None:
        events_path = run_dir / "events.jsonl"

    model = _build_model(
        input_hw,
        out_channels=4,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        pool_kernel=(2, 2),
        pool_stride=(2, 2),
    )
    model.attach_hypergrad(curvature=curvature, learning_rate=lr)
    trainer = st.nn.ModuleTrainer(
        backend=backend,
        curvature=curvature,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    schedule = trainer.roundtable(
        batch,
        2,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    loss = st.nn.CrossEntropy(curvature=curvature)

    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "mellin_log_grid_classification",
        "format": FORMAT,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch": batch,
        "lr": lr,
        "curvature": curvature,
        "seed": seed,
        "backend": backend,
        "feature": feature,
        "epsilon": epsilon,
        "lattice": {"log_start": log_start, "log_step": log_step, "len": series_len},
        "mesh": {
            "real_min": real_min,
            "real_max": real_max,
            "real_count": real_count,
            "imag_min": imag_min,
            "imag_max": imag_max,
            "imag_count": imag_count,
        },
        "rates": {
            "low_min": low_rate_min,
            "low_max": low_rate_max,
            "high_min": high_rate_min,
            "high_max": high_rate_max,
        },
        "events_path": str(events_path) if events_path is not None else None,
        "events_types": events_types,
        "atlas": atlas,
        "atlas_bound": atlas_bound if atlas else None,
        "atlas_district": atlas_district if atlas else None,
        "val_batches": val_batches if val_batches > 0 else None,
    }
    _write_json(run_dir / "run.json", run_meta)

    print(
        f"arch=mellin_log_grid_classification mesh={input_hw} feature={feature} series_len={series_len} "
        f"epochs={epochs} batch={batch} lr={lr:.3e} curvature={curvature} backend={backend} run_dir={run_dir}"
    )

    metrics_path = run_dir / "metrics.jsonl"
    record_ctx = (
        st.plugin.record(str(events_path), event_types=events_types)
        if events_path is not None
        else contextlib.nullcontext()
    )

    val_batches_cache: list[tuple[st.Tensor, st.Tensor]] = []
    if val_batches > 0:
        val_seed = seed + 123_456
        for b in range(val_batches):
            val_batches_cache.append(
                _build_batch(
                    batch,
                    plan,
                    log_start=log_start,
                    log_step=log_step,
                    series_len=series_len,
                    feature=feature,
                    epsilon=epsilon,
                    low_rate_min=low_rate_min,
                    low_rate_max=low_rate_max,
                    high_rate_min=high_rate_min,
                    high_rate_max=high_rate_max,
                    seed=val_seed + b,
                )
            )

    with record_ctx:
        for epoch in range(max(0, epochs)):
            batches: list[tuple[st.Tensor, st.Tensor]] = []
            base_seed = seed + epoch * 10_000
            for b in range(batches_per_epoch):
                batches.append(
                    _build_batch(
                        batch,
                        plan,
                        log_start=log_start,
                        log_step=log_step,
                        series_len=series_len,
                        feature=feature,
                        epsilon=epsilon,
                        low_rate_min=low_rate_min,
                        low_rate_max=low_rate_max,
                        high_rate_min=high_rate_min,
                        high_rate_max=high_rate_max,
                        seed=base_seed + b,
                    )
                )
            stats = trainer.train_epoch(model, loss, batches, schedule)
            avg_loss = float(stats.average_loss)
            val_loss = _evaluate_loss(model, loss, val_batches_cache) if val_batches_cache else float("nan")
            _append_jsonl(
                metrics_path,
                {
                    "epoch": epoch,
                    "batches": int(stats.batches),
                    "average_loss": avg_loss,
                    "val_loss": val_loss if val_loss == val_loss else None,
                },
            )
            if val_loss == val_loss:
                print(f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f} val_loss={val_loss:.6f}")
            else:
                print(f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f}")

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


if __name__ == "__main__":
    main()

