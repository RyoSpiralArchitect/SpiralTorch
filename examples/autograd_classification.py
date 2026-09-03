"""Bounded three-class learning fixture, entirely in native SpiralTorch autograd."""
import json
import math

import spiraltorch as st


def samples(per_class, phase):
    values, labels = [], []
    for label in range(3):
        center = 2.0 * math.pi * label / 3.0
        for index in range(per_class):
            angle = 2.0 * math.pi * (index + phase) / per_class
            values.extend([
                1.4 * math.cos(center) + 0.35 * math.cos(angle),
                1.4 * math.sin(center) + 0.35 * math.sin(angle),
            ])
            labels.append(label)
    return st.Tensor(len(labels), 2, values), labels


def run(steps=300):
    variable = st.AutogradTensor.variable
    constant = st.AutogradTensor.constant
    train, labels = samples(24, 0.0)
    validation, validation_labels = samples(12, 0.5)
    inputs = constant(train)
    held_out = constant(validation)
    parameters = [
        variable(st.Tensor.randn(2, 12, std=0.4, seed=31)),
        variable(st.Tensor.zeros(1, 12)),
        variable(st.Tensor.randn(12, 3, std=0.4, seed=37)),
        variable(st.Tensor.zeros(1, 3)),
    ]

    def predict(batch):
        return (batch.matmul(parameters[0]).add_row(parameters[1]).gelu()
                .matmul(parameters[2]).add_row(parameters[3]))

    initial = predict(inputs).cross_entropy_with_logits(labels, label_smoothing=0.05).item()
    for _ in range(steps):
        loss = predict(inputs).cross_entropy_with_logits(labels, label_smoothing=0.05)
        receipt = loss.backward()
        assert receipt["leaf_gradient_count"] == 4
        parameters = [
            variable(parameter.value().sub(parameter.grad().scale(0.2)))
            for parameter in parameters
        ]
    final = predict(inputs).cross_entropy_with_logits(labels, label_smoothing=0.05).item()
    output = predict(held_out)
    validation_nll = output.cross_entropy_with_logits(validation_labels).item()
    predicted = [max(range(3), key=row.__getitem__) for row in output.value().tolist()]
    accuracy = sum(a == b for a, b in zip(predicted, validation_labels)) / len(predicted)
    assert final < initial / 3.0 and validation_nll < 0.1 and accuracy == 1.0
    return {
        "fixture": "three_class_native_autograd", "steps": steps,
        "train_samples": len(labels), "validation_samples": len(validation_labels),
        "initial_smoothed_ce": initial, "final_smoothed_ce": final,
        "validation_nll": validation_nll, "validation_accuracy": accuracy,
        "semantic_owner": st.AUTOGRAD_SEMANTIC_OWNER,
        "evidence_scope": "synthetic pipeline fixture, not an HF/FT quality benchmark",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
