"""Independent CPU-f32 embedding/CE/SGD oracle. Refuses to replace a fixture."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def values(tensor):
    return tensor.detach().reshape(-1).tolist()


def case(name, rows, width, shape, ids):
    table = ((torch.arange(rows * width, device="cpu", dtype=torch.float32) % 23 - 11) / 16)
    table = table.reshape(rows, width).requires_grad_()
    indices = torch.tensor(ids, device="cpu", dtype=torch.long).reshape(shape)
    output = F.embedding(indices, table)
    seed = ((torch.arange(output.numel(), device="cpu", dtype=torch.float32) % 19 - 9) / 8)
    seed = seed.reshape(output.shape)
    gradient, = torch.autograd.grad(output, table, seed)
    return {"name": name, "table_shape": [rows, width], "index_shape": shape,
            "indices": ids, "table": values(table), "cotangent": values(seed),
            "output_shape": list(output.shape), "output": values(output),
            "gradient": values(gradient)}


def learning():
    ids = torch.tensor([[4, 1, 4], [0, 1, 2]], device="cpu", dtype=torch.long)
    target = torch.tensor([[2, 0, 2], [1, 0, 1]], device="cpu", dtype=torch.long)
    table = ((torch.arange(15, device="cpu", dtype=torch.float32) - 7) / 16)
    table = table.reshape(5, 3).requires_grad_()
    initial = values(table)
    trace = []
    rate, steps = 0.125, 16
    for _ in range(steps):
        logits = F.embedding(ids, table)
        loss = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))
        gradient, = torch.autograd.grad(loss, table)
        trace.append({"loss": loss.item(), "logits": values(logits), "gradient": values(gradient)})
        with torch.no_grad():
            table -= rate * gradient
    return {"table_shape": [5, 3], "index_shape": [2, 3], "indices": values(ids),
            "target": values(target), "initial": initial, "steps": steps,
            "learning_rate": rate, "trace": trace, "final": values(table)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = [
        case("duplicates", 9, 5, [2, 3], [8, 0, 8, 2, 0, 1]),
        case("multi_workgroup", 29, 11, [2, 17], [(i * 7) % 13 for i in range(34)]),
        case("scalar", 5, 3, [], [3]),
        case("empty_samples", 5, 3, [2, 0], []),
        case("empty_table", 0, 3, [0], []),
        case("zero_width", 5, 0, [3], [4, 0, 4]),
    ]
    payload = {"schema": "spiraltorch.resident_embedding.torch_fixture.v1",
               "torch_version": torch.__version__, "device": "cpu", "dtype": "float32",
               "threads": 1, "tolerance": {"atol": 3e-6, "rtol": 5e-5},
               "cases": cases, "learning": learning()}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
