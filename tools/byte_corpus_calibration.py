"""Independent measurement only. Fitted weights are prepared by Rust, not here."""
import math


def verify_qualification(request, record):
    """Validate recorded independent measurements without trusting pass flags."""
    if (not isinstance(record, dict) or set(record) != {"passed", "no_fitting", "no_updates_consumed", "cases"}
            or any(record.get(flag) is not True for flag in ("passed", "no_fitting", "no_updates_consumed"))):
        raise ValueError("missing or failed independent calibration qualification")
    cases = record["cases"]
    seeds = sorted({c["seed"] for c in request["cases"]})
    if (type(cases) is not list or len(cases) != len(seeds)
            or any(type(c.get("seed")) is not int or c["seed"] != seed for c, seed in zip(cases, seeds))):
        raise ValueError("independent calibration seed coverage")
    cfg = request["config"]
    spec = request["bias_calibration"]
    count = len(spec["train_batch_indices"]) * cfg["batch"] * cfg["steps"] * (cfg["steps"] + 1) // 2
    for case in cases:
        if type(case.get("valid_pairs_per_head")) is not int or case["valid_pairs_per_head"] != count:
            raise ValueError("independent calibration pair coverage")
        for field in ("reference_rms", "fitted_rms", "realized_relative_errors"):
            value = case.get(field)
            if (type(value) is not list or len(value) != len(cfg["blocks"])
                    or any(type(row) is not list or len(row) != cfg["heads"] for row in value)
                    or any(type(x) not in (float, int) or not math.isfinite(x)
                           or (x < 0 if field == "realized_relative_errors" else x <= 0)
                           for row in value for x in row)):
                raise ValueError("independent calibration block/head statistic")
        for reference, fitted, stated in zip(case["reference_rms"], case["fitted_rms"], case["realized_relative_errors"]):
            for target, actual, claimed in zip(reference, fitted, stated):
                error = abs(actual / target - 1)
                if not math.isfinite(error) or error > spec["relative_tolerance"] or abs(error - claimed) > 1e-14:
                    raise ValueError("independent calibration RMS gate or arithmetic")


def verify_initial_rms(request, wave, metric):
    import torch
    import torch.nn.functional as functional

    config = request["config"]
    batch, t = config["batch"], config["steps"]
    spec = request["bias_calibration"]

    def moments(case):
        p = [torch.tensor(v["values"], device="cpu", dtype=torch.float32).reshape(v["shape"])
             for v in case["parameters"]]
        energy = torch.zeros((len(config["blocks"]), config["heads"]), device="cpu", dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for index in spec["train_batch_indices"]:
                rows = [request["train_documents"][d][s:s+t] for d, s in request["train_batches"][index]]
                ids = torch.tensor(rows, device="cpu", dtype=torch.long)
                positions = torch.arange(t, device="cpu").expand(batch, t)
                embedded = functional.embedding(ids, p[0]) + functional.embedding(positions, p[1])
                coordinates, _ = wave(embedded @ p[2] + p[3], p[4], p[5],
                    torch.zeros((batch, config["geometry_cols"]), device="cpu"), config["curvature"])
                for block in range(len(config["blocks"])):
                    if case["pair_metric"] == "euclidean_chord_squared.v1":
                        distance = (coordinates[:, :, None, :] - coordinates[:, None, :, :]).square().sum(-1)
                        scores = (-functional.softplus(p[6+block])[None, :, None, None] * (4 * distance[:, None])).tril()
                    else:
                        scores = metric(coordinates, p[6+block], config["curvature"])
                    for q in range(t):
                        row = scores[:, :, q, :q+1].double()
                        centered = row - row.mean(-1, keepdim=True)
                        energy[block] += centered.square().sum((0, 2))
                count += batch * t * (t + 1) // 2
        return (energy / count).sqrt().tolist(), count

    reports = []
    for seed in sorted({c["seed"] for c in request["cases"]}):
        group = [c for c in request["cases"] if c["seed"] == seed and c["geometry_update"] == "train"]
        target = next(c for c in group if c.get("pair_metric") == "poincare_squared.v1")
        fitted = next(c for c in group if c["bias_initialization"] == "matched_poincare_rms")
        reference, count = moments(target)
        observed, other_count = moments(fitted)
        if count != other_count:
            raise ValueError("calibration measurement coverage differs")
        errors = []
        for reference_row, observed_row in zip(reference, observed):
            row = []
            for target, actual in zip(reference_row, observed_row):
                if not math.isfinite(target) or not math.isfinite(actual) or min(target, actual) <= 0:
                    raise ValueError("unqualified initial RMS")
                error = abs(actual / target - 1)
                if error > spec["relative_tolerance"]:
                    raise ValueError("frozen initial RMS gate failed; do not fit or relax it")
                row.append(error)
            errors.append(row)
        reports.append(dict(seed=seed, reference_rms=reference, fitted_rms=observed,
                            valid_pairs_per_head=count, realized_relative_errors=errors))
    result = dict(passed=True, no_fitting=True, no_updates_consumed=True, cases=reports)
    verify_qualification(request, result)
    return result
