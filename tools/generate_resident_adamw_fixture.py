"""Independent CPU PyTorch AdamW oracle; zero-rate freeze means grad=None."""
import json

import torch


def fixture():
    torch.set_num_threads(1)
    initial = [[1.0, -2.0, -0.0, 0.25], [-0.5, 2.0, -0.0], [0.125]]
    shapes = [[2, 2], [3], [1]]
    config = [float(torch.tensor(x, dtype=torch.float32)) for x in (0.9, 0.999, 1e-8)]
    params = [torch.tensor(p, dtype=torch.float32).reshape(s).requires_grad_() for p, s in zip(initial, shapes)]
    optimizer = torch.optim.AdamW([{"params": [p]} for p in params], lr=0.01,
                                 betas=tuple(config[:2]), eps=config[2], foreach=False, fused=False)
    trajectory = []
    for update in range(12):
        rates = [0.01, 0.0 if update in (0, 1, 4, 5) else 0.02, 0.03]
        if update == 8:
            rates = [0.0] * 3
        decay = [0.1, 0.0, 0.02]
        rates, decay = [[float(torch.tensor(x, dtype=torch.float32)) for x in xs] for xs in (rates, decay)]
        gradients = [[((update + 2 * j + k) % 7 - 3) * 0.125 for k in range(p.numel())]
                     for j, p in enumerate(params)]
        for p, group, g, lr, wd in zip(params, optimizer.param_groups, gradients, rates, decay):
            p.grad = torch.tensor(g, dtype=torch.float32).reshape(p.shape) if lr else None
            group["lr"], group["weight_decay"] = lr, wd
        optimizer.step()
        slots = []
        for p in params:
            state = optimizer.state[p]
            slots.append({"first": state.get("exp_avg", torch.zeros_like(p)).flatten().tolist(),
                          "second": state.get("exp_avg_sq", torch.zeros_like(p)).flatten().tolist(),
                          "steps": int(state.get("step", 0))})
        trajectory.append({"rates": rates, "decay": decay, "gradients": gradients,
                           "values": [p.detach().flatten().tolist() for p in params], "slots": slots})
    return {"schema": "spiraltorch.resident_adamw.torch_fixture.v1", "torch": torch.__version__,
            "device": "cpu", "dtype": "float32", "freeze": "grad_none", "config": config,
            "initial": initial, "shapes": shapes, "trajectory": trajectory}


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2, allow_nan=False))
