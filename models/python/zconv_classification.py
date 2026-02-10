from __future__ import annotations

import math
import pathlib
import sys

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st
from spiraltorch import plugin

import _softlogic_cli


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


def build_batch(batch: int, input_hw: tuple[int, int], seed: int) -> tuple[st.Tensor, st.Tensor]:
    cols = input_hw[0] * input_hw[1]
    x = st.Tensor.rand(batch, cols, seed=seed)
    targets: list[float] = []
    for row in x.tolist():
        mid = len(row) // 2
        score = sum(row[:mid]) - sum(row[mid:])
        if score >= 0.0:
            targets.extend([1.0, 0.0])
        else:
            targets.extend([0.0, 1.0])
    y = st.Tensor(batch, 2, targets)
    return x, y


def build_model(
    input_hw: tuple[int, int],
    out_channels: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> st.nn.Sequential:
    out_hw = st.nn.conv_output_shape(
        input_hw, kernel, stride=stride, padding=padding, dilation=(1, 1)
    )
    features = out_channels * out_hw[0] * out_hw[1]
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
    model.add(st.nn.NonLiner("nl1", features, activation="tanh"))
    model.add(st.nn.Scaler("sc1", features))
    model.add(st.nn.Linear("head", features, 2))
    return model


def main() -> None:
    backend = "cpu"
    epochs = 3
    batch = 6
    lr = 1e-2
    curvature = -1.0
    lr_schedule = "constant"
    lr_min: float | None = None
    grad_clip = 0.0
    args = list(sys.argv[1:])
    _softlogic_cli_state = _softlogic_cli.pop_softlogic_flags(args)
    it = iter(args)
    for flag in it:
        if flag == "--backend":
            backend = str(next(it)).strip().lower()
        elif flag == "--epochs":
            epochs = int(next(it))
        elif flag == "--batch":
            batch = int(next(it))
        elif flag == "--lr":
            lr = float(next(it))
        elif flag == "--curvature":
            curvature = float(next(it))
        elif flag == "--lr-schedule":
            lr_schedule = str(next(it)).strip().lower()
        elif flag == "--lr-min":
            lr_min = float(next(it))
        elif flag == "--grad-clip":
            grad_clip = float(next(it))
        elif flag in {"-h", "--help"}:
            print(
                "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/zconv_classification.py "
                "[--backend cpu|wgpu|cuda|hip|auto] "
                "[--epochs N] [--batch N] [--lr F] [--curvature F] "
                "[--lr-schedule constant|linear|cosine] [--lr-min F] [--grad-clip F] "
                f"{_softlogic_cli.usage_flags()}"
            )
            return
        else:
            raise ValueError(f"unknown flag: {flag}")

    if epochs < 0:
        raise ValueError("--epochs must be >= 0")
    if batch <= 0:
        raise ValueError("--batch must be > 0")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("--lr must be a positive, finite float")
    if lr_schedule not in {"constant", "linear", "cosine"}:
        raise ValueError(
            f"unknown --lr-schedule: {lr_schedule} (expected constant|linear|cosine)"
        )
    if lr_min is not None and (not math.isfinite(lr_min) or lr_min <= 0.0):
        raise ValueError("--lr-min must be a positive, finite float")
    if not math.isfinite(grad_clip) or grad_clip < 0.0:
        raise ValueError("--grad-clip must be a finite float >= 0")

    _require_backend_available(backend)

    input_hw = (4, 4)
    kernel = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    out_channels = 2

    resolved_lr_min: float | None = None
    if lr_schedule != "constant":
        resolved_lr_min = float(lr_min) if lr_min is not None else float(lr) * 0.1

    trainer = st.nn.ModuleTrainer(
        backend=backend,
        curvature=curvature,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    if grad_clip > 0.0:
        trainer.set_grad_clip_max_norm(grad_clip)
    _softlogic_cli.apply_softlogic_cli(trainer, _softlogic_cli_state)
    schedule = trainer.roundtable(
        batch,
        2,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )

    model = build_model(input_hw, out_channels, kernel, stride, padding)
    model.attach_hypergrad(curvature=curvature, learning_rate=lr)
    loss = st.nn.CrossEntropy(curvature=curvature)

    x, y = build_batch(batch, input_hw, seed=11)

    weights_dir = pathlib.Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    record_path = weights_dir / "zconv_classification_events.jsonl"

    with plugin.record(str(record_path), event_types=["EpochEnd", "TensorOp"]):
        lr_active = float(lr)
        lr_floor = float(resolved_lr_min) if resolved_lr_min is not None else float(lr) * 0.1

        def _scheduled_lr(epoch_idx: int) -> float:
            if lr_schedule == "constant" or epochs <= 1:
                return float(lr)
            t = float(epoch_idx) / float(max(1, epochs - 1))
            if lr_schedule == "linear":
                return float(lr) + (lr_floor - float(lr)) * t
            return lr_floor + 0.5 * (float(lr) - lr_floor) * (1.0 + math.cos(math.pi * t))

        for epoch in range(epochs):
            target_lr = _scheduled_lr(epoch)
            if lr_active > 0.0 and target_lr > 0.0:
                factor = float(target_lr) / float(lr_active)
                if factor == factor and abs(factor - 1.0) > 1e-9:
                    trainer.mul_learning_rate(model, factor)
                    lr_active = float(target_lr)
            stats = trainer.train_epoch(model, loss, [(x, y)], schedule)
            print(f"epoch[{epoch}] lr={lr_active:.3e} stats:", stats)

    weights_path = weights_dir / "zconv_classification.json"
    st.nn.save(str(weights_path), model)
    manifest_path = weights_path.with_suffix(".manifest.json")

    reloaded = build_model(input_hw, out_channels, kernel, stride, padding)
    st.nn.load(str(manifest_path), reloaded)
    _ = reloaded.forward(x)


if __name__ == "__main__":
    main()
