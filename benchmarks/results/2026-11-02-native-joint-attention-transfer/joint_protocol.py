"""Fixed native joint-adaptation protocol; dataset preparation never loads the LM."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-10-31-native-book-ablation"
TOOLS = HERE.parents[2] / "tools"
sys.path[:0] = [str(OLD), str(TOOLS)]
from common import (
    PLAN as ORIGINAL,
    check_ids,
    digest,
    full_windows,
    guard_ml,
    midpoint_indices,
    ordering,
    read,
    require,
    save,
    sha,
    weighted,
)

PLAN = read(HERE / "plan.json")
SEEDS, ARMS = PLAN["seeds"], PLAN["arms"]
CASES = [(s, a) for i, s in enumerate(SEEDS) for a in ARMS[i:] + ARMS[:i]]
KEYS = [f"{s}-{a}" for s, a in CASES]
STATES = ["base", *["initial-" + a for a in ARMS], *KEYS]
BOOKS = list(PLAN["books"])


def roman(n):
    out = ""
    for v, name in [
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]:
        times, n = divmod(n, v)
        out += name * times
    return out


def chapters(text, book):
    start, end = (
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
    )
    require(text.count(start) == text.count(end) == 1, "Gutenberg framing changed")
    body = text.split(start)[1].split("\n", 1)[1].split(end)[0]
    if book == "worlds":
        volumes = re.split(r"(?m)^BOOK (ONE|TWO)$", body)
        require(volumes[1::2] == ["ONE", "TWO"], "unexpected Wells volumes")
        parts = []
        for text, size in zip(volumes[2::2], [17, 10], strict=True):
            matches = re.split(r"(?m)^([IVX]+)\.$", text)
            require(
                matches[1::2] == [roman(i) for i in range(1, size + 1)],
                "unexpected Wells chapters",
            )
            parts.extend(matches[2::2])
    else:
        require(book in ("mansfield", "beagle"), "unknown book")
        matches = re.split(r"(?m)^CHAPTER ([IVXL]+)$", body)
        require(
            matches[1::2]
            == [roman(i) for i in range(1, PLAN["books"][book]["chapters"] + 1)],
            "unexpected chapter order",
        )
        parts = matches[2::2]
    return [" ".join(p.split()) for p in parts]


def prepare(raw, train, model, site, out):
    guard_ml()
    sys.path.append(str(site.resolve(strict=True)))
    import tokenizers

    require(
        tokenizers.__version__ == ORIGINAL["tokenizers"]
        and sha(model / "tokenizer.json") == ORIGINAL["model"]["tokenizer.json"]
        and sha(train) == PLAN["train_sha256"],
        "wrong tokenizer/training windows",
    )
    tokenizer = tokenizers.Tokenizer.from_file(str(model / "tokenizer.json"))
    seen = {"train": read(train)}
    groups, indices = {}, {}
    for book, spec in PLAN["books"].items():
        path = raw / f"pg{spec['id']}.txt"
        require(sha(path) == spec["sha256"], "sample source changed: " + book)
        parts = chapters(path.read_text(encoding="utf-8-sig"), book)
        groups[book], indices[book] = {}, {}
        for i in midpoint_indices(len(parts), 6):
            pool = full_windows(
                tokenizer.encode(parts[i], add_special_tokens=False).ids, 64
            )
            key = str(i + 1)
            selected = indices[book][key] = midpoint_indices(len(pool), 12)
            groups[book][key] = seen[f"{book}/{key}"] = [pool[j] for j in selected]
    check_ids(seen)
    prompts = [
        tokenizer.encode(p, add_special_tokens=False).ids for p in PLAN["prompts"]
    ]
    out.mkdir(mode=0o700, exist_ok=False)
    save(out / "windows.json", groups)
    save(out / "prompts.json", prompts)
    save(
        out / "manifest.json",
        {
            "plan_sha256": sha(HERE / "plan.json"),
            "windows_sha256": sha(out / "windows.json"),
            "prompts_sha256": sha(out / "prompts.json"),
            "offsets": indices,
            "targets_per_book": 4536,
        },
    )


def source_pins():
    paths = [
        HERE / n
        for n in ("plan.json", "joint_protocol.py", "joint_worker.py", "joint_study.py")
    ]
    paths += [
        OLD / n for n in ("common.py", "worker.py", "plan.json", "execution.json")
    ]
    paths += [TOOLS / n for n in ("owned_process.py", "byte_release_artifacts.py")]
    return {str(p.relative_to(HERE.parents[2])): sha(p) for p in paths}


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("raw", "train", "model", "site", "out"):
        p.add_argument(key, type=Path)
    prepare(**vars(p.parse_args()))
