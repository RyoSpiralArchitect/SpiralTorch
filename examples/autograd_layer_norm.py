"""A bounded native LayerNorm learning fixture, not a model-quality benchmark."""

import json
import spiraltorch as st


def run():
    values = st.Tensor(3, 3, [0.4, -0.8, 1.2, -0.3, 0.9, -1.1, 0.7, 0.1, -0.2])
    inputs = st.AutogradTensor.constant(values)
    targets = st.AutogradTensor.constant(values.layer_norm_affine(
        st.Tensor(1, 3, [1.7, 0.5, -0.8]), st.Tensor(1, 3, [0.2, -0.3, 0.6]),
    ))
    optimizer = st.AutogradSgd([
        st.AutogradTensor.variable(st.Tensor(1, 3, [1.0, 1.0, 1.0])),
        st.AutogradTensor.variable(st.Tensor.zeros(1, 3)),
    ], learning_rate=0.1)
    losses = []
    for _ in range(400):
        gamma, beta = optimizer.parameters()
        prediction = inputs.layer_norm_affine(gamma, beta)
        loss = prediction.mean_squared_error(targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 1e-4
    return {"steps": 400, "initial_loss": losses[0], "final_loss": losses[-1],
            "semantic_owner": "st-tensor", "backward_backend": "cpu"}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
