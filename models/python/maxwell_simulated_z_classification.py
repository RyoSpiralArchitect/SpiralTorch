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

FORMAT = "st-core-maxwell-simulated-z-classification-v1"
RUN_SCHEMA = "st.modelzoo.run.v1"


def _meta_path_for_weights(weights_path: pathlib.Path) -> pathlib.Path:
    name = weights_path.name
    if name.endswith(".json"):
        return weights_path.with_name(name[: -len(".json")] + ".meta.json")
    return weights_path.with_name(name + ".meta.json")


def _save_meta(meta_path: pathlib.Path, meta: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


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


def _build_batch(
    batch: int,
    *,
    blocks: int,
    sigma: float,
    kappa: float,
    low_lambda_min: float,
    low_lambda_max: float,
    high_lambda_min: float,
    high_lambda_max: float,
    seed: int,
) -> tuple[st.Tensor, st.Tensor]:
    rng = random.Random(seed)
    data: list[float] = []
    targets: list[float] = []

    for idx in range(batch):
        class_one = idx % 2 == 1
        if class_one:
            lam = rng.uniform(high_lambda_min, high_lambda_max)
            targets.extend([0.0, 1.0])
        else:
            lam = rng.uniform(low_lambda_min, low_lambda_max)
            targets.extend([1.0, 0.0])

        curve = st.spiralk.simulate_z_curve(blocks, sigma, kappa, lam, seed + idx + 10_000)
        if len(curve) != blocks:
            raise RuntimeError(f"simulate_z_curve returned {len(curve)} points (expected {blocks})")
        data.extend(float(v) if math.isfinite(v) else 0.0 for v in curve)

    x = st.Tensor(batch, blocks, data)
    y = st.Tensor(batch, 2, targets)
    return x, y


def _build_model(in_dim: int, *, norm: str = "none", curvature: float = -1.0) -> st.nn.Sequential:
    model = st.nn.Sequential()
    if norm == "none":
        pass
    elif norm == "layer":
        model.add(st.nn.LayerNorm("norm1", in_dim, curvature=curvature, epsilon=1e-5))
    elif norm == "zspace":
        model.add(st.nn.ZSpaceLayerNorm("norm1", in_dim, curvature=curvature, epsilon=1e-5))
    elif norm == "batch":
        model.add(st.nn.BatchNorm1d("norm1", in_dim, momentum=0.1, epsilon=1e-5))
    elif norm == "zbatch":
        model.add(st.nn.ZSpaceBatchNorm1d("norm1", in_dim, curvature=curvature, momentum=0.1, epsilon=1e-5))
    else:
        raise ValueError(f"unknown --norm: {norm} (expected none|layer|zspace|batch|zbatch)")
    model.add(st.nn.Linear("head", in_dim, 2))
    return model


def _evaluate_loss(model: st.nn.Sequential, loss: Any, batches: list[tuple[st.Tensor, st.Tensor]]) -> float:
    if not batches:
        return float("nan")
    total = 0.0
    with st.nn.eval_mode(model):
        for x, y in batches:
            pred = model.forward(x)
            value = loss.forward(pred, y).tolist()
            total += float(value[0][0])
    return total / float(len(batches))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/maxwell_simulated_z_classification.py "
            "[--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--norm none|layer|zspace|batch|zbatch] "
            "[--lr-schedule constant|linear|cosine] [--lr-min F] "
            "[--lr-warmup-epochs N] [--lr-warmup-start F] [--grad-clip F] "
            "[--blocks N] [--sigma F] [--kappa F] "
            "[--low-lambda-min F] [--low-lambda-max F] [--high-lambda-min F] [--high-lambda-max F] "
            "[--backend cpu|wgpu|cuda|hip|auto] "
            "[--events PATH] [--events-types A,B,C] "
            "[--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--run-dir PATH] [--seed N] [--val-batches N] "
            f"{_softlogic_cli.usage_flags()}"
        )
        return

    run_dir: pathlib.Path | None = None

    epochs = 8
    batches_per_epoch = 8
    batch = 8
    lr = 2e-2
    curvature = -1.0
    norm = "none"
    lr_schedule = "constant"
    lr_min: float | None = None
    lr_warmup_epochs = 0
    lr_warmup_start: float | None = None
    grad_clip = 0.0
    seed = 777
    val_batches = 0

    blocks = 48
    sigma = 1.25
    kappa = 1.0

    low_lambda_min = 0.4
    low_lambda_max = 0.7
    high_lambda_min = 0.9
    high_lambda_max = 1.3

    backend = "cpu"
    events_path: pathlib.Path | None = None
    events_types = ["EpochStart", "EpochEnd", "TrainerStep", "TrainerPhase"]
    atlas = False
    atlas_bound = 512
    atlas_district = "Training"

    args = list(sys.argv[1:])
    softlogic_cli = _softlogic_cli.pop_softlogic_flags(args)
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
        elif flag == "--norm":
            norm = str(next(it)).strip().lower()
        elif flag == "--seed":
            seed = int(next(it))
        elif flag == "--val-batches":
            val_batches = int(next(it))
        elif flag == "--blocks":
            blocks = int(next(it))
        elif flag == "--sigma":
            sigma = float(next(it))
        elif flag == "--kappa":
            kappa = float(next(it))
        elif flag == "--low-lambda-min":
            low_lambda_min = float(next(it))
        elif flag == "--low-lambda-max":
            low_lambda_max = float(next(it))
        elif flag == "--high-lambda-min":
            high_lambda_min = float(next(it))
        elif flag == "--high-lambda-max":
            high_lambda_max = float(next(it))
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

    if blocks <= 1:
        raise ValueError("--blocks must be >= 2")
    if not (sigma > 0.0 and math.isfinite(sigma)):
        raise ValueError("--sigma must be finite and > 0")
    if not math.isfinite(kappa):
        raise ValueError("--kappa must be finite")
    if not (low_lambda_min > 0.0 and low_lambda_max >= low_lambda_min):
        raise ValueError("--low-lambda-min/max must satisfy 0 < min <= max")
    if not (high_lambda_min > 0.0 and high_lambda_max >= high_lambda_min):
        raise ValueError("--high-lambda-min/max must satisfy 0 < min <= max")
    if norm not in {"none", "layer", "zspace", "batch", "zbatch"}:
        raise ValueError(f"unknown --norm: {norm} (expected none|layer|zspace|batch|zbatch)")
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

    model = _build_model(blocks, norm=norm, curvature=curvature)
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
        2,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    loss = st.nn.CrossEntropy(curvature=curvature)

    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "maxwell_simulated_z_classification",
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
        "norm": norm,
        "seed": seed,
        "backend": backend,
        "blocks": blocks,
        "sigma": sigma,
        "kappa": kappa,
        "lambdas": {
            "low_min": low_lambda_min,
            "low_max": low_lambda_max,
            "high_min": high_lambda_min,
            "high_max": high_lambda_max,
        },
        "events_path": str(events_path) if events_path is not None else None,
        "events_types": events_types,
        "atlas": atlas,
        "atlas_bound": atlas_bound if atlas else None,
        "atlas_district": atlas_district if atlas else None,
        "softlogic": softlogic_meta,
        "val_batches": val_batches if val_batches > 0 else None,
    }
    _write_json(run_dir / "run.json", run_meta)

    print(
        f"arch=maxwell_simulated_z_classification blocks={blocks} sigma={sigma:.3g} kappa={kappa:.3g} norm={norm} "
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

    val_batches_cache: list[tuple[st.Tensor, st.Tensor]] = []
    if val_batches > 0:
        val_seed = seed + 123_456
        for b in range(val_batches):
            val_batches_cache.append(
                _build_batch(
                    batch,
                    blocks=blocks,
                    sigma=sigma,
                    kappa=kappa,
                    low_lambda_min=low_lambda_min,
                    low_lambda_max=low_lambda_max,
                    high_lambda_min=high_lambda_min,
                    high_lambda_max=high_lambda_max,
                    seed=val_seed + b,
                )
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

            batches: list[tuple[st.Tensor, st.Tensor]] = []
            base_seed = seed + epoch * 10_000
            for b in range(batches_per_epoch):
                batches.append(
                    _build_batch(
                        batch,
                        blocks=blocks,
                        sigma=sigma,
                        kappa=kappa,
                        low_lambda_min=low_lambda_min,
                        low_lambda_max=low_lambda_max,
                        high_lambda_min=high_lambda_min,
                        high_lambda_max=high_lambda_max,
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
                    "learning_rate": lr_active,
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
    meta = dict(run_meta)
    meta.update(
        {
            "weights_path": weights_path.name,
            "weights_format": "json",
            "model": {
                "blocks": blocks,
                "norm": norm,
            },
        }
    )
    _save_meta(_meta_path_for_weights(weights_path), meta)


if __name__ == "__main__":
    main()
