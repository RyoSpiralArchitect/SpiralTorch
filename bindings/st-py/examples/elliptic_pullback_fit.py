"""Bounded matrix-free Gauss-Newton fitting through native JVP/VJP snapshots.

The map and both derivatives are Rust-owned. This client only orchestrates a
damped linear solve and an actual-loss line search. Synthetic feature fitting
is not evidence for pretrained language-model quality or throughput.
"""

import json
import math

import spiraltorch as st


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def chart(vector):
    # Hold the first coordinate at one in the primal and zero in directions.
    return [0.0 if i % 3 == 0 else v for i, v in enumerate(vector)]


def solve(operator, rhs):
    value, residual, direction = [0.0] * len(rhs), list(rhs), list(rhs)
    energy = dot(residual, residual)
    for _ in range(16):
        if energy < 1e-20:
            break
        product = operator(direction)
        curvature = dot(direction, product)
        if not math.isfinite(curvature) or curvature <= 0.0:
            raise ValueError("damped pullback must be finite and positive")
        alpha = energy / curvature
        value = [x + alpha * p for x, p in zip(value, direction)]
        residual = [r - alpha * a for r, a in zip(residual, product)]
        next_energy = dot(residual, residual)
        direction = [r + next_energy / energy * p for r, p in zip(residual, direction)]
        energy = next_energy
    actual_residual = [a - b for a, b in zip(operator(value), rhs)]
    return value, math.sqrt(dot(actual_residual, actual_residual))


def run():
    warp = st.EllipticWarp(1.3, 3, 2)
    desired = [1.0, 0.35, -0.2, 1.0, -0.4, 0.6]
    target = warp.map_orientations_batch(desired).features
    orientation = [1.0, 3.0, -2.0, 1.0, -2.0, 2.0]
    damping = 0.001

    def forward(value):
        snapshot = warp.map_orientations_batch(value)
        residual = [a - b for a, b in zip(snapshot.features, target)]
        return snapshot, residual, 0.5 * dot(residual, residual)

    initial_loss = forward(orientation)[2]
    losses, linear_residuals = [initial_loss], []
    for _ in range(24):
        snapshot, residual, loss = forward(orientation)
        if loss < 1e-12:
            break

        def normal(direction):
            product = chart(snapshot.vjp(snapshot.jvp(chart(direction))))
            return [a + damping * v for a, v in zip(product, direction)]

        step, residual_norm = solve(normal, [-v for v in chart(snapshot.vjp(residual))])
        linear_residuals.append(residual_norm)
        for exponent in range(16):
            candidate = [x + 2.0**-exponent * v for x, v in zip(orientation, step)]
            candidate_loss = forward(candidate)[2]
            if candidate_loss < loss:
                orientation = candidate
                losses.append(candidate_loss)
                break
        else:
            raise RuntimeError("no decreasing step; no successful fit is reported")
    final_loss = forward(orientation)[2]
    if not math.isfinite(final_loss) or final_loss >= initial_loss * 1e-4:
        raise RuntimeError("bounded synthetic fit did not converge")
    return {
        "schema": "spiraltorch.elliptic_pullback_fit.v1",
        "scope": "Synthetic feature fitting, no LM-quality or speed claim",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "accepted_updates": len(losses) - 1,
        "losses": losses,
        "max_linear_system_residual": max(linear_residuals),
        "damping": damping,
        "orientation": orientation,
        "target_orientation": desired,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
