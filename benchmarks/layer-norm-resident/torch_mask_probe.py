"""Diagnostic only: distinguish LayerNorm output-mask correctness from speed."""
import json
import math
import os

import torch


def main():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("MPS fallback must remain disabled")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("real MPS device required")
    x = torch.tensor([[0.4, -0.8, 1.2], [-0.3, 0.9, -1.1], [0.7, 0.1, -0.2]])
    gamma, beta = torch.ones(3), torch.zeros(3)
    seed = torch.tensor([[0.2, -0.1, 0.4], [0.1, 0.3, -0.2], [-0.1, 0.2, 0.3]])
    _, mean, rstd = torch.ops.aten.native_layer_norm(x, [3], gamma, beta, 1e-5)
    oracle = torch.ops.aten.native_layer_norm_backward(seed, x, [3], mean, rstd, gamma, beta, [True] * 3)
    cases = []
    for device in ("cpu", "mps"):
        a, g, b, s = [v.to(device, copy=True) for v in (x, gamma, beta, seed)]
        _, mean, rstd = torch.ops.aten.native_layer_norm(a, [3], g, b, 1e-5)
        for mask in range(1, 8):
            requested = [bool(mask & (1 << i)) for i in range(3)]
            output = torch.ops.aten.native_layer_norm_backward(s, a, [3], mean, rstd, g, b, requested)
            gradients = []
            for index, (value, expected, needed) in enumerate(zip(output, oracle, requested)):
                if not needed:
                    gradients.append(dict(index=index, requested=False, valid=value is None))
                    continue
                if value is None:
                    gradients.append(dict(index=index, requested=True, valid=False, reason="missing gradient"))
                    continue
                actual = value.detach().to("cpu", copy=True).double()
                finite = torch.isfinite(actual).all().item()
                error = ((actual - expected.double()).abs() / (2e-5 * (1 + expected.double().abs()))).max().item()
                values = [v if math.isfinite(v) else str(v) for v in actual.flatten().tolist()]
                gradients.append(dict(index=index, requested=True, valid=finite and error <= 1,
                                      max_scaled_error=error if math.isfinite(error) else str(error),
                                      actual=values, expected=expected.flatten().tolist()))
            cases.append(dict(device=device, mask=requested, gradients=gradients,
                              valid=all(g["valid"] for g in gradients)))
    print(json.dumps(dict(schema="spiraltorch.layer_norm.torch_mask_probe.v1", diagnostic_only=True,
                         torch_version=torch.__version__, operator=str(torch.ops.aten.native_layer_norm_backward.default._schema),
                         all_masks_valid=all(c["valid"] for c in cases), cases=cases), allow_nan=False))


if __name__ == "__main__":
    main()
