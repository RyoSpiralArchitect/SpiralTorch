"""Independent eager Torch oracle for the bounded v2 graph timing fixture."""
import json
import math
import time


def require_fusion_equivalent(source_json, fused_json):
    """Test-only structural proof: same ordered SSA operations/operands, not a rewrite."""
    def program(payload):
        plan = json.loads(payload)
        if plan["schema"] != "spiraltorch.nn.inference_plan.v2":
            raise ValueError("fusion comparison requires v2 graphs")
        operations = []
        for stage in plan["stages"]:
            if stage["kind"] == "linear":
                operations.append(("linear", len(operations), stage["weight"], stage["bias"]))
                if stage["gelu"]:
                    operations.append(("gelu", len(operations), None))
            elif stage["kind"] == "pointwise":
                inputs = [("activation", len(operations)),
                          *(("parameter", i) for i in stage["parameters"])]
                for step in stage["steps"]:
                    binary = step["op"] in ("add", "multiply")
                    if step["op"] not in ("identity", "relu", "gelu", "add", "multiply"):
                        raise ValueError("unknown pointwise op")
                    slot = step["rhs"]
                    if binary:
                        if type(slot) is not int or not 0 <= slot < len(inputs):
                            raise ValueError("invalid pointwise rhs")
                    elif slot is not None:
                        raise ValueError("unary rhs is not empty")
                    operations.append((step["op"], len(operations), inputs[slot] if binary else None))
            else:
                raise ValueError("unknown graph stage")
        return plan["input_shape"], plan["parameters"], operations
    if program(source_json) != program(fused_json):
        raise ValueError("fusion changed parameter identity or ordered operations/operands")


def match_fused_fixture(source, candidate):
    if candidate["config"] != dict(source["config"], fuse_pointwise=True):
        raise ValueError("fusion recipe differs")
    for key in ("input", "target", "learning_rate", "kernel", "accumulation", "adapter"):
        if source[key] != candidate[key]:
            raise ValueError("fusion fixture differs: " + key)
    if candidate["source_plan_json"] != source["plan_json"]:
        raise ValueError("fusion source plan differs")
    require_fusion_equivalent(source["plan_json"], candidate["plan_json"])


def compare(actual, expected, fields=("prediction", "input_gradient", "parameters", "raw_gradients", "effective_gradients")):
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
    for key in fields:
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
