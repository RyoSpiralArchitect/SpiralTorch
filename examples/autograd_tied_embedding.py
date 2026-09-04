"""Synthetic token-transition learning, not evidence of LLM/FT quality gains."""

import json
import spiraltorch as st


def run(steps=400):
    def variable(rows, cols, values):
        return st.AutogradTensor.variable(st.Tensor(rows, cols, values))

    ids = [0, 1, 2, 3] * 2
    labels = [1, 2, 3, 0] * 2
    optimizer = st.AutogradSgd([
        variable(4, 3, [0.2, -0.1, 0.3, -0.3, 0.2, 0.1, 0.1, 0.3, -0.2, -0.2, -0.3, 0.2]),
        variable(3, 3, [0.2, 0.1, -0.1, -0.2, 0.3, 0.1, 0.1, -0.2, 0.2]),
    ], learning_rate=0.15)
    gamma = st.AutogradTensor.constant(st.Tensor(1, 3, [1, 1, 1]))
    beta = st.AutogradTensor.constant(st.Tensor.zeros(1, 3))

    def logits():
        table, transition = optimizer.parameters()
        hidden = table.gather_rows(ids).matmul(transition).gelu()
        hidden = hidden.layer_norm_affine(gamma, beta)
        return hidden.matmul(table.transpose())  # One table, two gradient paths.

    initial = logits().cross_entropy_with_logits(labels).item()
    for _ in range(steps):
        loss = logits().cross_entropy_with_logits(labels)
        loss.backward()
        optimizer.step()
    final_logits = logits()
    final = final_logits.cross_entropy_with_logits(labels).item()
    predictions = [max(range(4), key=row.__getitem__) for row in final_logits.value().tolist()]
    accuracy = sum(a == b for a, b in zip(predictions, labels)) / len(labels)
    assert final < initial * 0.1, (initial, final)
    assert accuracy == 1.0, predictions
    return {"steps": steps, "initial_loss": initial, "final_loss": final,
            "accuracy": accuracy, "predictions": predictions, "tied_weights": True,
            "semantic_owner": "st-tensor", "backward_backend": "cpu",
            "evidence": "synthetic_mechanics_only"}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
