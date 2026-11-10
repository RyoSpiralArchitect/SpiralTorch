"""Materialize rank-one factors using only the fit-direction file, not check scores or model calls."""

import hashlib
import json
import math
import sys
from pathlib import Path


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def close(a, b):
    return math.isclose(a, b, rel_tol=3e-9, abs_tol=8e-13)


def check(a, msg):
    if not a:
        raise ValueError(msg)


def top_rank_one(weight):
    # Cyclic symmetric Jacobi diagonalization of the normalized training Gram matrix.
    w = 32
    check(len(weight) == w * w and all(map(math.isfinite, weight)), "invalid input-major weight direction")
    n = math.hypot(*weight)
    if n == 0:
        return None
    cols = [[weight[i * w + j] / n for i in range(w)] for j in range(w)]
    gram = [[dot(x, y) for y in cols] for x in cols]
    orig = [r[:] for r in gram]
    vec = [[float(i == j) for j in range(w)] for i in range(w)]
    for _ in range(40):
        peak = 0.0
        for a in range(w):
            for b in range(a + 1, w):
                v = gram[a][b]
                peak = max(peak, abs(v))
                if abs(v) < 2e-15:
                    continue
                z = (gram[b][b] - gram[a][a]) / (2 * v)
                t = math.copysign(1.0, z) / (abs(z) + math.hypot(1.0, z))
                c = 1 / math.hypot(1.0, t)
                s = t * c
                gram[a][a] -= t * v
                gram[b][b] += t * v
                gram[a][b] = gram[b][a] = 0.0
                for i in range(w):
                    if i not in (a, b):
                        x, y = gram[i][a], gram[i][b]
                        gram[i][a] = gram[a][i] = c * x - s * y
                        gram[i][b] = gram[b][i] = s * x + c * y
                    x, y = vec[i][a], vec[i][b]
                    vec[i][a] = c * x - s * y
                    vec[i][b] = s * x + c * y
        if peak < 2e-15:
            break
    k = max(range(w), key=lambda j: gram[j][j])
    v = [r[k] for r in vec]
    lam = gram[k][k]
    upper = max(r[i] + math.fsum(abs(x) for j, x in enumerate(r) if j != i) for i, r in enumerate(gram))
    residual = math.hypot(*(dot(row, v) - lam * x for row, x in zip(orig, v, strict=True)))
    check(
        upper - lam < 1e-11 and residual < 1e-11 and close(math.hypot(*v), 1.0),
        "unresolved rank-one direction",
    )
    u = [dot(weight[i * w : (i + 1) * w], v) for i in range(w)]
    sigma = math.hypot(*u)
    axis = [x / sigma * y for x in u for y in v]
    check(close(math.hypot(*axis), 1.0) and close(dot(weight, axis), sigma), "wrong rank-one normalization")
    first = max(range(w), key=lambda j: abs(v[j]))
    sg = 1.0 if v[first] >= 0.0 else -1.0
    return dict(
        left=[sg * x / sigma for x in u],
        right=[sg * x for x in v],
        fraction_dense_fit_norm=sigma,
        relative_gram_residual=residual,
    )


def freeze_roster(source, expected_sha256, out):
    with source.open("rb") as f:
        raw = f.read((4 << 20) + 1)
    pin = hashlib.sha256(raw).hexdigest()
    check(len(raw) <= 4 << 20 and pin == expected_sha256, "direction pin or size changed")
    d = json.loads(raw)
    roles = {
        s: {k: top_rank_one(v[32:]) for k, v in ds.items() if k.endswith("/dense") and v is not None}
        for s, ds in d["roles"].items()
    }
    check(len(roles) == 4 and all(len(v) == 8 for v in roles.values()), "incomplete fit-only roster")
    result = dict(
        kind="fit_only_rank_one_no_bias.v1",
        plan=d["plan"],
        fit=d["fit"],
        directions=pin,
        roles=roles,
        model_calls=0,
        map="zero bias; unit Frobenius W[i,j] = left[i]*right[j]. No physical test, "
        "optimizer step, artifact promotion or check-fold selection.",
    )
    data = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with out.open("xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: rank_one_roster.py DIRECTIONS SHA256 NEW_OUTPUT")
    print(freeze_roster(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])))
