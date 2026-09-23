"""Independent high-precision oracle for the cancellation-sensitive review case."""
from decimal import Decimal, localcontext
import json
import struct


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


with localcontext() as context:
    context.prec = 100
    x = [Decimal(f32(v)) for v in (0, -1e10, 1192.0928955078125, 0)]
    gamma = [Decimal(f32(v)) for v in (1, 1e20, 1, 1)]
    epsilon = Decimal(f32(1e-5))
    mean = sum(x) / len(x)
    centered = [v - mean for v in x]
    variance = sum(v * v for v in centered) / len(x)
    inverse_std = 1 / (variance + epsilon).sqrt()
    mean_gradient = sum(gamma) / len(x)
    covariance = sum(c * g for c, g in zip(centered, gamma)) / len(x)
    dx = [(g - mean_gradient - c * covariance / (variance + epsilon)) * inverse_std
          for c, g in zip(centered, gamma)]
    print(json.dumps(dict(schema="spiraltorch.layer_norm.decimal_probe.v1", precision=100,
                          exact_f32_input=[str(v) for v in x],
                          exact_f32_gamma=[str(v) for v in gamma],
                          exact_f32_epsilon=str(epsilon), upstream=[1] * 4,
                          dx_decimal=[str(v) for v in dx], dx_f32=[f32(v) for v in dx]), indent=2))
