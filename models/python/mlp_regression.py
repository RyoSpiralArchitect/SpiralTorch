from __future__ import annotations

import pathlib
import sys

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st
from spiraltorch import plugin


def main() -> None:
    # Subscribe to epoch end events (from st_nn::ModuleTrainer).
    sub_id = plugin.subscribe(
        "EpochEnd",
        lambda e: print(f"[epoch={e['epoch']}] avg_loss={e['loss']:.6f}"),
    )

    batch = 8
    in_dim = 2
    out_dim = 1

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    schedule = trainer.roundtable(
        batch,
        out_dim,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", in_dim, 8))
    model.add(st.nn.Relu())
    model.add(st.nn.Linear("l2", 8, out_dim))
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
    reloaded.add(st.nn.Linear("l1", in_dim, 8))
    reloaded.add(st.nn.Relu())
    reloaded.add(st.nn.Linear("l2", 8, out_dim))
    reloaded.load_state_dict(restored)
    _ = reloaded.forward(x)

    plugin.unsubscribe("EpochEnd", sub_id)


if __name__ == "__main__":
    main()
