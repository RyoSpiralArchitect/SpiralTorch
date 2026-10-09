"""Independent Torch-only initialization oracle, never a production learner."""
import torch
import torch.nn.functional as F


def rms(batches):
    values = torch.stack(batches).double()
    count, batch, heads, steps, _ = values.shape
    energy = torch.zeros(heads, dtype=torch.float64, device="cpu")
    for q in range(steps):
        row = values[:, :, :, q, :q + 1]
        centered = row - row.mean(dim=-1, keepdim=True)
        energy += centered.square().sum(dim=(0, 1, 3))
    return (energy / (count * batch * steps * (steps + 1) / 2)).sqrt()


def calibrate(parameters, config, training_windows, wave, poincare):
    batch, steps = config["batch"], config["steps"]
    if (batch, steps) != (2, 4):
        raise ValueError("This frozen oracle uses two four-position rows")
    windows = [training_windows, [[13, 19, 5, 2, 255], [1, 3, 15, 8, 254]]]
    blocks = len(config["blocks"])
    cols = config["causal_geometry"]["cols"]
    curvature = config["causal_geometry"]["curvature"]

    def scores(flat_metric):
        result = []
        for rows in windows:
            ids = torch.tensor([row[:steps] for row in rows], dtype=torch.long, device="cpu")
            positions = torch.arange(steps, device="cpu").expand(batch, steps)
            embedded = F.embedding(ids, parameters[0]) + F.embedding(positions, parameters[1])
            drive = embedded @ parameters[2] + parameters[3]
            coordinates, _ = wave(drive, parameters[4], parameters[5],
                                  torch.zeros((batch, cols), device="cpu"), curvature)
            distances = 4 * (coordinates[:, :, None, :] - coordinates[:, None, :, :]).square().sum(-1)
            row = []
            for i in range(blocks):
                gain = parameters[6 + i]
                row.append((-F.softplus(gain)[None, :, None, None] * distances[:, None]).tril()
                           if flat_metric else poincare(coordinates, gain, curvature))
            result.append(row)
        return result

    with torch.no_grad():
        old = [parameters[6 + i].clone() for i in range(blocks)]
        reference, candidate = scores(False), scores(True)
        reference_rms = [rms([b[i] for b in reference]) for i in range(blocks)]
        candidate_rms = [rms([b[i] for b in candidate]) for i in range(blocks)]
        for i, (target, observed) in enumerate(zip(reference_rms, candidate_rms)):
            if not bool(((target > 0) & (observed > 0)).all()):
                raise ValueError("degenerate calibration input")
            gain = F.softplus(old[i].double()) * target / observed
            parameters[6 + i].copy_((gain + torch.log(-torch.expm1(-gain))).float())
        fitted = scores(True)
        fitted_rms = [rms([b[i] for b in fitted]) for i in range(blocks)]
        if any(not bool(((actual / target - 1).abs() <= 1e-5).all())
               for actual, target in zip(fitted_rms, reference_rms)):
            raise ValueError("calibration did not meet the fixed realized-RMS gate")
    flatten = lambda values: [[v.flatten().tolist() for v in batch] for batch in values]
    return {"windows": windows, "shape": [batch, config["heads"], steps, steps],
            "relative_tolerance": 1e-5, "valid_pairs_per_head": len(windows) * batch * steps * (steps + 1) // 2,
            "reference_scores": flatten(reference), "candidate_scores": flatten(candidate),
            "fitted_scores": flatten(fitted), "reference_rms": [x.tolist() for x in reference_rms],
            "candidate_rms": [x.tolist() for x in candidate_rms], "fitted_rms": [x.tolist() for x in fitted_rms],
            "old_raw_gains": [x.tolist() for x in old], "raw_gains": [parameters[6 + i].detach().tolist() for i in range(blocks)],
            "scope": "Training-only synthetic calibration windows; no labels used, no update consumed, no quality claim"}
