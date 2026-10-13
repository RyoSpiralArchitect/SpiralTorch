"""Small Rust-owned geometric byte LM. No Torch, tokenizer or model download."""
import argparse
import json
import math
from pathlib import Path

import spiraltorch as st

METRICS = (
    "poincare_squared.v1",
    "euclidean_chord_squared.v1",
    "categorical_fisher_rao_squared.v1",
    "complex_projective_chord_squared.v1",
)


def make_plan(mode="query_mixture", batch=2, steps=4):
    """Build a tiny fixed-window plan on CPU; this does not acquire a GPU."""
    width, heads, hidden = 4, 2, 7
    shape = [batch, steps, width]

    def weights(count, phase):
        return [0.15 * math.sin(i * 0.73 + phase) for i in range(count)]

    def linear(inner, cols, phase, input_shape):
        return st.nn.InferencePlan.from_json(json.dumps({
            "schema": "spiraltorch.nn.inference_plan.v1",
            "input_shape": input_shape,
            "stages": [{"inner": inner, "cols": cols,
                        "weight": weights(inner * cols, phase),
                        "bias": weights(cols, phase + 1.0), "gelu": False}],
        }))

    pre = st.nn.Sequential()
    pre.add(st.nn.LayerNorm("pre", width, -1.0, 1e-5))
    attention = st.nn.AttentionInferencePlan.from_projection_plans(
        *(linear(width, width, phase, shape) for phase in (1., 2., 3., 4.)),
        heads=heads, causal_offset=0)
    feed = st.nn.InferencePlan.from_json(json.dumps({
        "schema": "spiraltorch.nn.inference_plan.v5",
        "input_shape": shape,
        "parameters": [
            {"role": "gain", "shape": [width], "values": [1.] * width},
            {"role": "bias", "shape": [width], "values": [0.] * width},
            {"role": "weight", "shape": [width, hidden], "values": weights(width * hidden, 5.)},
            {"role": "bias", "shape": [hidden], "values": weights(hidden, 6.)},
            {"role": "gate", "shape": [hidden], "values": [0.5] * hidden},
            {"role": "weight", "shape": [hidden, width], "values": weights(hidden * width, 7.)},
            {"role": "bias", "shape": [width], "values": weights(width, 8.)},
        ],
        "stages": [
            {"kind": "layer_norm", "gain": 0, "bias": 1, "epsilon": 1e-5},
            {"kind": "linear", "weight": 2, "bias": 3, "gelu": True},
            {"kind": "topos_resonator", "gate": 4, "coupling": 0.2,
             "iterations": 4, "saturation": 0.12, "porosity": 0.3,
             "max_volume": batch * steps * hidden},
            {"kind": "linear", "weight": 5, "bias": 6, "gelu": False},
        ],
    }))
    block = st.nn.ResidualAttentionPlan.from_plans(pre.inference_plan(shape), attention, feed)
    plan = st.nn.ByteDecoderPlan.from_plans(
        st.Tensor(256, width, weights(256 * width, 9.)),
        st.Tensor(steps, width, weights(steps * width, 10.)),
        [block], linear(width, 256, 11., shape))
    if mode == "none":
        return plan
    geometry = st.nn.ByteDecoderGeometryPlan(
        linear(width, 2, 12., shape), [0.2], [0.1], [[-0.5, 0.25]], -1.0)
    if mode in METRICS:
        geometry = geometry.with_pair_metric(mode)
    elif mode == "headwise":
        geometry = geometry.with_head_pair_metrics([[METRICS[0], METRICS[2]]])
    elif mode in ("mixture", "query_mixture"):
        geometry = geometry.with_mixture(list(METRICS), [[0.1, -0.2, 0.3, -0.1] * heads], [[1.] * (heads * len(METRICS))])
        if mode == "query_mixture":
            geometry = geometry.with_query_projections([
                linear(2, heads * len(METRICS), 13., [batch, steps, 2])])
    else:
        raise ValueError("unknown geometry mode")
    return plan.with_causal_geometry(geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("none", "headwise", "mixture", "query_mixture") + METRICS,
                        default="query_mixture")
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    model = make_plan(args.mode).compile_training_wgpu()
    batch = model.prepare_batch(st.nn.ByteLmBatch.from_documents(
        [b"SpiralTorch learns bytes.", b"Geometry changes attention."], [(0, 0), (1, 0)], 4))
    losses = []
    for _ in range(args.updates):
        forward = model.forward(batch)
        loss = forward.next_byte_loss()
        gradients = model.backward(forward, loss.prediction_gradient_tensor())
        update = model.sgd(gradients, 0.125)
        update.read()  # Submission alone is not proof that an update was accepted.
        losses.append(loss.loss_tensor().snapshot().read_values()[0])
    checkpoint = model.checkpoint_snapshot().read()
    if args.checkpoint:
        with args.checkpoint.open("x") as output:
            output.write(checkpoint.to_json() + "\n")
    restored = checkpoint.restore_wgpu()
    restored_batch = restored.prepare_batch(st.nn.ByteLmBatch.from_documents(
        [b"SpiralTorch learns bytes.", b"Geometry changes attention."], [(0, 0), (1, 0)], 4))
    expected = model.forward(batch).prediction_tensor().snapshot().read_values()
    observed = restored.forward(restored_batch).prediction_tensor().snapshot().read_values()
    if observed != expected:
        raise RuntimeError("checkpoint restore changed the prediction")
    print(json.dumps({"mode": args.mode, "losses": losses,
                      "attempted_revision": checkpoint.attempted_revision,
                      "resume_prediction_exact": True,
                      "scope": "tiny repeated-window mechanics, not language quality"}))


if __name__ == "__main__":
    main()
