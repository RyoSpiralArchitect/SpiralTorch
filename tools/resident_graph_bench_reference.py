"""Independent eager Torch oracle for the bounded v2 graph timing fixture."""
import json
import math
import time


def compare(actual, expected):
    maximum = 0.0

    def check(a, b):
        nonlocal maximum
        if len(a) != len(b):
            raise ValueError("graph result length differs")
        for x, y in zip(a, b):
            if isinstance(x, list):
                check(x, y)
            elif (not math.isfinite(x) or not math.isfinite(y)
                  or abs(x-y) > 2e-5+2e-4*abs(y)):
                raise ValueError(f"graph result differs: {x} != {y}")
            else:
                maximum = max(maximum, abs(x-y))

    check([actual["loss"]], [expected["loss"]])
    for key in ("prediction", "input_gradient", "parameters", "raw_gradients", "effective_gradients"):
        check(actual[key], expected[key])
    return maximum


def torch_sample(torch, fixture, device, cadence, synchronize):
    plan = json.loads(fixture["plan_json"])
    if plan["schema"] != "spiraltorch.nn.inference_plan.v2":
        raise ValueError("v2 graph required")
    shape = fixture["config"]["shape"]
    x = torch.tensor(fixture["input"], dtype=torch.float32, device=device).reshape(shape).requires_grad_()
    target = torch.tensor(fixture["target"], dtype=torch.float32, device=device).reshape(shape)
    parameters = [torch.tensor(p["values"], dtype=torch.float32, device=device)
                  .reshape(p["shape"]).requires_grad_() for p in plan["parameters"]]
    optimizer = torch.optim.SGD(parameters, lr=fixture["learning_rate"], foreach=False)

    def evaluate():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        current = x
        for stage in plan["stages"]:
            if stage["kind"] == "linear":
                current = current @ parameters[stage["weight"]] + parameters[stage["bias"]]
                if stage["gelu"]:
                    current = torch.nn.functional.gelu(current, approximate="tanh")
            elif stage["kind"] == "pointwise":
                inputs = [current, *(parameters[i] for i in stage["parameters"])]
                for step in stage["steps"]:
                    op = step["op"]
                    if op == "multiply": current = current * inputs[step["rhs"]]
                    elif op == "add": current = current + inputs[step["rhs"]]
                    elif op == "relu": current = torch.relu(current)
                    elif op == "gelu": current = torch.nn.functional.gelu(current, approximate="tanh")
                    elif op != "identity": raise ValueError("unknown pointwise operation")
            else:
                raise ValueError("unknown graph node")
        objective = torch.nn.functional.mse_loss(current, target, reduction="mean")
        objective.backward()
        return objective, current

    initial, _ = evaluate()
    initial_loss = initial.detach().cpu().item()
    losses = []
    synchronize()
    start = time.perf_counter()
    for _ in range(fixture["config"]["steps"]):
        objective, _ = evaluate()
        optimizer.step()
        if cadence == "immediate":
            synchronize()
            losses.append(objective.detach().cpu().item())
        else:
            losses.append(objective.detach())
    if cadence == "deferred":
        losses = torch.stack(losses).cpu().tolist()
    synchronize()
    elapsed_ms = (time.perf_counter()-start)*1000
    objective, prediction = evaluate()
    def values(t): return t.detach().cpu().reshape(-1).tolist()
    raw = [values(p.grad) for p in parameters]
    state = dict(loss=objective.detach().cpu().item(), prediction=values(prediction),
                 input_gradient=values(x.grad), parameters=[values(p) for p in parameters],
                 raw_gradients=raw, effective_gradients=raw)
    return dict(status="passed", cadence=cadence, steps=fixture["config"]["steps"],
                elapsed_ms=elapsed_ms, losses=losses, initial_loss=initial_loss, state=state)
