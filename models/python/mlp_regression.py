from __future__ import annotations

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


def main() -> None:
    backend = "cpu"
    activation = "relu"
    norm = "none"
    args = list(sys.argv[1:])
    _softlogic_cli_state = _softlogic_cli.pop_softlogic_flags(args)
    it = iter(args)
    for flag in it:
        if flag == "--backend":
            backend = str(next(it)).strip().lower()
        elif flag == "--activation":
            activation = str(next(it)).strip().lower()
        elif flag == "--norm":
            norm = str(next(it)).strip().lower()
        elif flag in {"-h", "--help"}:
            print(
                "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/mlp_regression.py "
                "[--backend cpu|wgpu|cuda|hip|auto] "
                "[--activation relu|gelu] "
                "[--norm none|layer|zspace|batch|zbatch] "
                f"{_softlogic_cli.usage_flags()}"
            )
            return
        else:
            raise ValueError(f"unknown flag: {flag}")

    _require_backend_available(backend)

    # Subscribe to epoch end events (from st_nn::ModuleTrainer).
    sub_id = plugin.subscribe(
        "EpochEnd",
        lambda e: print(f"[epoch={e['epoch']}] avg_loss={e['loss']:.6f}"),
    )

    batch = 8
    in_dim = 2
    out_dim = 1
    hidden = 8

    trainer = st.nn.ModuleTrainer(
        backend=backend,
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    _softlogic_cli.apply_softlogic_cli(trainer, _softlogic_cli_state)
    schedule = trainer.roundtable(
        batch,
        out_dim,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", in_dim, hidden))
    if activation == "relu":
        model.add(st.nn.Relu())
    elif activation == "gelu":
        model.add(st.nn.Gelu())
    else:
        raise ValueError(f"unknown --activation: {activation} (expected relu|gelu)")

    if norm == "none":
        pass
    elif norm == "layer":
        model.add(st.nn.LayerNorm("ln1", hidden, curvature=-1.0, epsilon=1e-5))
    elif norm == "zspace":
        model.add(st.nn.ZSpaceLayerNorm("ln1", hidden, curvature=-1.0, epsilon=1e-5))
    elif norm == "batch":
        model.add(st.nn.BatchNorm1d("bn1", hidden, momentum=0.1, epsilon=1e-5))
    elif norm == "zbatch":
        model.add(st.nn.ZSpaceBatchNorm1d("bn1", hidden, curvature=-1.0, momentum=0.1, epsilon=1e-5))
    else:
        raise ValueError(f"unknown --norm: {norm} (expected none|layer|zspace|batch|zbatch)")

    model.add(st.nn.Linear("l2", hidden, out_dim))
    model.attach_hypergrad(curvature=-1.0, learning_rate=1e-2)

    loss = st.nn.MeanSquaredError()

    x = st.Tensor.rand(batch, in_dim, seed=1)
    y_data = [0.7 * row[0] - 0.2 * row[1] for row in x.tolist()]
    y = st.Tensor(batch, out_dim, y_data)

    # Train a few epochs on the same batch (model-zoo smoke).
    for _ in range(3):
        stats = trainer.train_epoch(model, loss, [(x, y)], schedule)
        print("stats:", stats)

    # Serialize weights and reload via the manifest helpers.
    weights_dir = pathlib.Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "mlp_regression.json"
    st.nn.save(str(weights_path), model)
    manifest_path = weights_path.with_suffix(".manifest.json")
    restored = st.nn.load(str(manifest_path))

    reloaded = st.nn.Sequential()
    reloaded.add(st.nn.Linear("l1", in_dim, hidden))
    if activation == "relu":
        reloaded.add(st.nn.Relu())
    else:
        reloaded.add(st.nn.Gelu())
    if norm == "layer":
        reloaded.add(st.nn.LayerNorm("ln1", hidden, curvature=-1.0, epsilon=1e-5))
    elif norm == "zspace":
        reloaded.add(st.nn.ZSpaceLayerNorm("ln1", hidden, curvature=-1.0, epsilon=1e-5))
    elif norm == "batch":
        reloaded.add(st.nn.BatchNorm1d("bn1", hidden, momentum=0.1, epsilon=1e-5))
    elif norm == "zbatch":
        reloaded.add(st.nn.ZSpaceBatchNorm1d("bn1", hidden, curvature=-1.0, momentum=0.1, epsilon=1e-5))
    reloaded.add(st.nn.Linear("l2", hidden, out_dim))
    reloaded.load_state_dict(restored)
    _ = reloaded.forward(x)

    plugin.unsubscribe("EpochEnd", sub_id)


if __name__ == "__main__":
    main()
