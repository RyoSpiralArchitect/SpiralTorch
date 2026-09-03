"""Train a nonlinear fixture through native Rust autograd, without PyTorch/NumPy."""
import json

import spiraltorch as st


def run():
    if getattr(st, "_rs", None) is None:
        raise RuntimeError("This example requires a freshly built native SpiralTorch wheel")
    variable = st.AutogradTensor.variable
    constant = st.AutogradTensor.constant
    inputs = constant(st.Tensor(4, 2, [0., 0., 0., 1., 1., 0., 1., 1.]))
    target = constant(st.Tensor(4, 2, [1., 0., 0., 1., 0., 1., 1., 0.]))
    parameters = [
        variable(st.Tensor.randn(2, 8, std=0.6, seed=19)),
        variable(st.Tensor.zeros(1, 8)),
        variable(st.Tensor.randn(8, 2, std=0.6, seed=29)),
        variable(st.Tensor.zeros(1, 2)),
    ]

    def predict():
        return (
            inputs.matmul(parameters[0]).add_row(parameters[1]).gelu()
            .matmul(parameters[2]).add_row(parameters[3]).row_softmax()
        )

    initial_loss = predict().mean_squared_error(target).item()
    for _ in range(600):
        loss = predict().mean_squared_error(target)
        receipt = loss.backward()
        assert receipt["leaf_gradient_count"] == 4
        parameters = [
            variable(parameter.value().sub(parameter.grad().scale(0.8)))
            for parameter in parameters
        ]
    output = predict()
    final_loss = output.mean_squared_error(target).item()
    predicted = [int(row[1] > row[0]) for row in output.value().tolist()]
    assert final_loss < 0.02 and final_loss < initial_loss / 10
    assert predicted == [0, 1, 1, 0]
    return {
        "fixture": "xor_nonlinear_autograd",
        "steps": 600,
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "predicted": predicted,
        "semantic_owner": st.AUTOGRAD_SEMANTIC_OWNER,
        "evidence_scope": "four-example learning fixture; not an HF/FT benchmark",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
