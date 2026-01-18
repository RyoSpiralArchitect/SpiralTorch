from __future__ import annotations

import pathlib
import sys

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st
from spiraltorch import plugin


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
    batch = 6
    input_hw = (4, 4)
    kernel = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    out_channels = 2

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    schedule = trainer.roundtable(
        batch,
        2,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )

    model = build_model(input_hw, out_channels, kernel, stride, padding)
    model.attach_hypergrad(curvature=-1.0, learning_rate=1e-2)
    loss = st.nn.CrossEntropy(curvature=-1.0)

    x, y = build_batch(batch, input_hw, seed=11)

    weights_dir = pathlib.Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    record_path = weights_dir / "zconv_classification_events.jsonl"

    with plugin.record(str(record_path), event_types=["EpochEnd", "TensorOp"]):
        for _ in range(3):
            stats = trainer.train_epoch(model, loss, [(x, y)], schedule)
            print("stats:", stats)

    weights_path = weights_dir / "zconv_classification.json"
    st.nn.save(str(weights_path), model)
    manifest_path = weights_path.with_suffix(".manifest.json")

    reloaded = build_model(input_hw, out_channels, kernel, stride, padding)
    st.nn.load(str(manifest_path), reloaded)
    _ = reloaded.forward(x)


if __name__ == "__main__":
    main()
