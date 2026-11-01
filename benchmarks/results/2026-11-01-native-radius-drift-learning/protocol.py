"""Frozen sample preparation and paired comparisons, independent of the optimizer."""

import argparse
import math
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "2026-10-31-native-book-ablation"
sys.path.insert(0, str(PRIOR))
from common import (  # noqa: E402
    PLAN as ORIGINAL,
    guard_ml,
    midpoint_indices,
    read,
    require,
    save,
    sha,
)

PLAN = read(HERE / "plan.json")
ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX".split()
ROSTER = [
    (s, a)
    for i, s in enumerate(PLAN["seeds"])
    for a in PLAN["arms"][i:] + PLAN["arms"][:i]
]
KEYS = [f"{s}-{a}" for s, a in ROSTER]
STOPS = list(range(PLAN["observe_every"], PLAN["updates"] + 1, PLAN["observe_every"]))


def chapters(text, book):
    start, end = (
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
    )
    require(text.count(start) == text.count(end) == 1, "wrong Gutenberg frame")
    body = text.split(start)[1].split("\n", 1)[1].split(end)[0]
    if book == "emma":
        labels = ROMAN[:18] + ROMAN[:18] + ROMAN[:19]
        pattern = r"(?m)^CHAPTER ([IVX]+)[ \t]*$"
    else:
        require(
            book == "darwin" and len(re.findall(r"(?m)^INDEX\.[ \t]*$", body)) == 1,
            "missing Darwin index boundary",
        )
        body = re.split(r"(?m)^INDEX\.[ \t]*$", body)[0]
        labels = ROMAN[:14]
        pattern = r"(?m)^CHAPTER ([IVX]+)\.[ \t]*$"
    pairs = re.split(pattern, body)
    require(pairs[1::2] == labels, "actual chapter headings/order differ")
    return [
        " ".join(re.split(r"(?m)^VOLUME [IVX]+\.?[ \t]*$", part)[0].split())
        for part in pairs[2::2]
    ]


def train_windows(path):
    require(sha(path) == PLAN["train_sha256"], "training bytes changed")
    docs = read(path)
    require(
        len(docs) == 512
        and all(len(x) == 64 for x in docs)
        and len({tuple(x) for x in docs}) == len(docs),
        "wrong training windows",
    )
    return docs


def prepare(a):
    guard_ml()
    sys.path.append(str(a.tokenizers_site.resolve(strict=True)))
    import tokenizers

    require(
        sha(HERE / "plan.json") == a.plan_pin
        and tokenizers.__version__ == ORIGINAL["tokenizers"]
        and sha(a.model / "tokenizer.json") == ORIGINAL["model"]["tokenizer.json"],
        "wrong tokenizer/protocol",
    )
    tokenizer = tokenizers.Tokenizer.from_file(str(a.model / "tokenizer.json"))
    seen = {tuple(x) for x in train_windows(a.train)}
    docs, roster = {}, {}
    for book, spec in PLAN["books"].items():
        p = a.books / f"pg{spec['id']}.txt"
        require(sha(p) == spec["sha256"], "evaluation book bytes changed: " + book)
        parts = chapters(p.read_text(encoding="utf-8-sig"), book)
        require(len(parts) == spec["chapters"], "unexpected chapter count")
        groups = docs[book] = {}
        roster[book] = [i + 1 for i in midpoint_indices(len(parts), 14)]
        for num in roster[book]:
            ids = tokenizer.encode(parts[num - 1], add_special_tokens=False).ids
            windows = [
                ids[j * 64 : (j + 1) * 64] for j in midpoint_indices(len(ids) // 64, 8)
            ]
            for row in windows:
                require(
                    len(row) == 64 and tuple(row) not in seen,
                    "overlapping/duplicate full sample",
                )
                seen.add(tuple(row))
            groups[str(num)] = windows
    a.output.mkdir(mode=0o700, exist_ok=False)
    save(a.output / "evaluation.json", docs)
    save(
        a.output / "manifest.json",
        {
            "plan_sha256": a.plan_pin,
            "train_sha256": PLAN["train_sha256"],
            "evaluation_sha256": sha(a.output / "evaluation.json"),
            "chapters": roster,
            "targets_per_book": 7056,
        },
    )


def criterion(fixed, free, bounded, retention_free, retention_bounded, projections):
    require(
        all(
            math.isfinite(x)
            for x in [fixed, free, bounded, retention_free, retention_bounded]
        ),
        "nonfinite score",
    )
    gain = fixed - free
    return {
        "emma_bounded_minus_free": bounded - free,
        "darwin_bounded_minus_free": retention_bounded - retention_free,
        "emma_free_gain_over_fixed": gain,
        "emma_bounded_gain_over_fixed": fixed - bounded,
        "joint_pass": gain > 0
        and fixed - bounded >= 0.5 * gain
        and retention_bounded < retention_free
        and projections > 0,
    }


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("books", "train", "model", "output", "tokenizers-site"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--plan-pin", required=True)
    prepare(p.parse_args())
