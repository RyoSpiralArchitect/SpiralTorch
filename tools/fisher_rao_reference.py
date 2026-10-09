"""Independent Torch/autograd categorical Fisher-Rao attention bias oracle."""
import torch
import torch.nn.functional as F


def metric(logits, raw_gain):
    roots = torch.exp(0.5 * F.log_softmax(logits, dim=-1))
    chord = (roots[:, :, None, :] - roots[:, None, :, :]).square().sum(-1).clamp_max(2.)
    # The analytic continuation removes sqrt's singular intermediate at self
    # pairs; it is not a constant-distance epsilon or a detached branch.
    local = 4 * chord * (1 + chord / 12 + chord.square() / 90
                        + chord.pow(3) / 560 + chord.pow(4) / 3150)
    regular = 16 * torch.asin(0.5 * torch.sqrt(chord.clamp_min(0.0004))).square()
    distance = torch.where(chord < 0.0004, local, regular)
    steps = logits.shape[1]
    causal = torch.ones((steps, steps), device=logits.device, dtype=torch.bool).tril()
    return (-F.softplus(raw_gain)[None, :, None, None] * distance[:, None]).masked_fill(~causal, 0.)
