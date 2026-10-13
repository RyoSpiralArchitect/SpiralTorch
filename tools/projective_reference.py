"""Independent CPU Torch projector-distance oracle, not a production backend."""
import torch
import torch.nn.functional as F


def metric(coordinates, raw_gain):
    """Anchored complex projectors; half their squared Frobenius distance."""
    if coordinates.ndim != 3 or coordinates.shape[-1] == 0 or coordinates.shape[-1] % 2:
        raise ValueError("expected nonempty interleaved complex chart")
    z = torch.view_as_complex(coordinates.reshape(*coordinates.shape[:-1], -1, 2).contiguous())
    state = torch.cat((torch.ones_like(z[..., :1]), z), dim=-1)
    unit = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)
    projector = unit[..., :, None] * unit.conj()[..., None, :]
    difference = projector[:, :, None] - projector[:, None, :]
    distance = 0.5 * (difference.real.square() + difference.imag.square()).sum((-2, -1))
    steps = coordinates.shape[1]
    causal = torch.ones((steps, steps), dtype=torch.bool, device=coordinates.device).tril()
    return (-F.softplus(raw_gain)[None, :, None, None] * distance[:, None]).masked_fill(~causal, 0.)
