"""Fixed protocol, pure data operations and bounded local receipts."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "plan.json").read_bytes())
SPLITS = ("external", "development")
FORBIDDEN = {"torch", "numpy", "transformers", "peft", "accelerate", "safetensors"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_bytes())


def midpoint_indices(total, count):
    require(0 < count <= total, "not enough disjoint windows for declared sample")
    return [(2 * i + 1) * total // (2 * count) for i in range(count)]


def full_windows(ids, width):
    return [ids[i : i + width] for i in range(0, len(ids) - width + 1, width)]


def ordering(seed, count):
    return sorted(
        range(count), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).digest()
    )


def cases():
    arms = PLAN["arms"]
    return [
        (seed, a) for i, seed in enumerate(PLAN["seeds"]) for a in arms[i:] + arms[:i]
    ]


def check_ids(groups):
    seen = set()
    for split, windows in groups.items():
        require(windows, f"empty {split}")
        for ids in windows:
            require(
                len(ids) == PLAN["recipe"]["context_length"]
                and all(type(n) is int and 0 <= n < 50257 for n in ids),
                "invalid ID window",
            )
            key = tuple(ids)
            require(key not in seen, "duplicate complete window within/across splits")
            seen.add(key)


def weighted(rows):
    count = sum(targets for _, targets in rows)
    require(
        count > 0 and all(math.isfinite(nll) and targets > 0 for nll, targets in rows),
        "invalid NLL or denominator",
    )
    return math.fsum(nll * targets for nll, targets in rows) / count


def guard_ml():
    import sys

    require(
        not (FORBIDDEN & {k.partition(".")[0] for k in sys.modules}),
        "foreign ML already loaded",
    )
    attempts = []

    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] in FORBIDDEN:
                attempts.append(fullname)
                raise ModuleNotFoundError(
                    "foreign ML disabled for native study", name=fullname
                )

    sys.meta_path.insert(0, Guard())
    return attempts
