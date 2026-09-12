"""Independent eager Torch weighted-learning oracle, not a guard/rollback model."""
import json
import time
import resident_graph_bench_reference as graph


def compare(actual, expected):
    keys = ("prediction", "input_gradients", "parameters", "raw_gradients")
    if ("momentum" in actual) != ("momentum" in expected):
        raise ValueError("optimizer history missing or unexpectedly present")
    return graph.compare(actual, expected, keys + (("momentum",) if "momentum" in expected else ()))


def torch_sample(torch, fixture, device, cadence, synchronize):
    from validate_resident_graph_training_vs_torch import evaluate
    plan = json.loads(fixture["plan_json"])
    if plan["schema"] != "spiraltorch.nn.inference_plan.v2":
        raise ValueError("v2 graph required")
    shape = fixture["config"]["shape"]
    x = torch.tensor(fixture["input"], dtype=torch.float32, device=device).reshape(shape).requires_grad_()
    target = torch.tensor(fixture["target"], dtype=torch.float32, device=device).reshape(shape)
    parameters = [torch.tensor(p["values"], dtype=torch.float32, device=device).reshape(p["shape"]).requires_grad_()
                  for p in plan["parameters"]]
    optimizer = fixture["config"].get("learner_optimizer")
    if optimizer not in (None, "topos_ema", "clipped_topos_ema"):
        raise ValueError("unknown learner optimizer")
    history = [torch.zeros_like(p) for p in parameters] if optimizer else None
    damping = 0.5 if optimizer else None
    clip = 1.0 / 1024 if optimizer == "clipped_topos_ema" else None

    def vjps():
        prediction = evaluate(plan, x, parameters)
        e = prediction - target
        norm = 1.0 / e.numel()
        a = torch.autograd.grad(prediction, [x, *parameters], grad_outputs=e * norm, retain_graph=True)
        b = torch.autograd.grad(prediction, [x, *parameters], grad_outputs=e * e * e * norm)
        return prediction, a, b

    def update(rate):
        prediction, a, b = vjps()
        with torch.no_grad():
            gradients = [0.75 * first + 0.25 * second for first, second in zip(a[1:], b[1:])]
            if clip is not None:
                norm = torch.cat([g.reshape(-1) for g in gradients]).norm()
                scale = torch.clamp(clip / norm, max=1.)
                eps = torch.finfo(torch.float32).eps
                scale = torch.where((norm <= eps) | ((scale - 1.).abs() <= eps), torch.ones_like(scale), scale)
                gradients = [g * scale for g in gradients]
            for i, (p, gradient) in enumerate(zip(parameters, gradients)):
                next_gradient = gradient if history is None else damping * history[i] + (1. - damping) * gradient
                p.sub_(rate * next_gradient)
                # Match the explicit zero-rate warmup contract, not Torch SGD.
                if history is not None and rate != 0.:
                    history[i] = next_gradient
        return prediction

    def objective(prediction):
        e = prediction.detach() - target
        return (0.75 * 0.5 * e.square() + 0.25 * 0.25 * e.pow(4)).mean().cpu().item()

    initial_loss = objective(update(0.0))
    synchronize()
    start = time.perf_counter()
    for _ in range(fixture["config"]["steps"]):
        update(fixture["learning_rate"])
        if cadence == "immediate": synchronize()
    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    prediction, a, b = vjps()
    values = lambda t: t.detach().cpu().reshape(-1).tolist()
    state = dict(loss=objective(prediction), prediction=values(prediction),
        parameters=[values(p) for p in parameters], input_gradients=[values(a[0]), values(b[0])],
        raw_gradients=[[values(p) for p in term[1:]] for term in [a, b]])
    if history is not None:
        state["momentum"] = [values(p) for p in history]
    return dict(status="passed", learner=True, cadence=cadence, steps=fixture["config"]["steps"],
        learner_optimizer=optimizer, momentum_damping=damping, grad_clip_max_norm=clip,
        completed_updates=fixture["config"]["steps"], acceptance="synchronized_only", losses=[],
        initial_loss=initial_loss, final_loss=state["loss"], state=state, elapsed_ms=elapsed_ms)
