"""Prepare a fixed held-out sample before loading any language model."""

import argparse
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-10-31-native-book-ablation"
sys.path.insert(0, str(OLD))
from common import (
    PLAN as ORIGINAL,
    check_ids,
    full_windows,
    guard_ml,
    midpoint_indices,
    read,
    require,
    save,
    sha,
)

PLAN = read(HERE / "plan.json")


def roman(num):
    text = ""
    for value, digits in [
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]:
        n, num = divmod(num, value)
        text += n * digits
    return text


def chapters(text, book):
    start = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK"
    require(text.count(start) == text.count(end) == 1, "unexpected Gutenberg framing")
    body = text.split(start)[1].split("\n", 1)[1].split(end)[0]
    size = PLAN["books"][book]["chapters"]
    labels = [roman(i) for i in range(1, size + 1)]
    if book == "austen":
        pattern = r"(?m)^CHAPTER ([IVXL]+)\.$"
        body = re.sub(r"(?m)^END OF THE (?:FIRST|SECOND|THIRD) VOLUME$", "", body)
    elif book == "shelley":
        pattern = r"(?m)^Chapter ([1-9][0-9]*)$"
        labels = [str(i) for i in range(1, size + 1)]
    else:
        require(book == "faraday", "unknown book")
        boundary = r"(?m)^THE CHEMICAL HISTORY OF A CANDLE$"
        # The body title follows the contents; an additional platinum lecture is not in the series.
        body = re.split(boundary, body)[-1]
        stop = r"(?m)^LECTURE ON PLATINUM\.$"
        require(
            len(re.findall(stop, body)) == 1, "missing end of the main lecture series"
        )
        body = re.split(stop, body)[0]
        pattern = r"(?m)^LECTURE ([IV]+)\.$"
    parts = re.split(pattern, body)
    require(parts[1::2] == labels, "unexpected section count or order: " + book)
    return [" ".join(part.split()) for part in parts[2::2]]


def prepare(raw, model, site, train_file, output):
    guard_ml()
    sys.path.append(str(site.resolve(strict=True)))
    import tokenizers

    require(
        tokenizers.__version__ == ORIGINAL["tokenizers"]
        and sha(model / "tokenizer.json") == ORIGINAL["model"]["tokenizer.json"]
        and sha(train_file) == PLAN["train_sha256"],
        "wrong tokenizer or training inputs",
    )
    tokenizer = tokenizers.Tokenizer.from_file(str(model / "tokenizer.json"))
    groups, offsets = {}, {}
    unique = {"train": read(train_file)}
    for book, spec in PLAN["books"].items():
        source = raw / f"pg{spec['id']}.txt"
        require(sha(source) == spec["sha256"], "book hash mismatch: " + book)
        sections = chapters(source.read_text(encoding="utf-8-sig"), book)
        groups[book], offsets[book] = {}, {}
        for pos in midpoint_indices(len(sections), 6):
            windows = full_windows(
                tokenizer.encode(sections[pos], add_special_tokens=False).ids, 64
            )
            indices = midpoint_indices(len(windows), 12)
            key = str(pos + 1)
            offsets[book][key] = indices
            groups[book][key] = unique[f"{book}/{key}"] = [windows[i] for i in indices]
    check_ids(unique)
    output.mkdir(mode=0o700, exist_ok=False)
    save(output / "windows.json", groups)
    save(
        output / "manifest.json",
        {
            "plan_sha256": sha(HERE / "plan.json"),
            "sections": offsets,
            "windows_sha256": sha(output / "windows.json"),
            "targets_per_book": 4536,
        },
    )


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("raw", "model", "site", "train_file", "output"):
        parser.add_argument(arg, type=Path)
    prepare(**vars(parser.parse_args()))
