"""Prepare chapter-bounded private ID windows; no model forward or optimizer."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    PLAN,
    check_ids,
    full_windows,
    guard_ml,
    midpoint_indices,
    require,
    save,
    sha,
)

ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII XXIII XXIV".split()


def persuasion(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    begin = "*** START OF THE PROJECT GUTENBERG EBOOK PERSUASION ***"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK PERSUASION ***"
    require(text.count(begin) == text.count(end) == 1, "unexpected book envelope")
    body = text.split(begin, 1)[1].split(end, 1)[0].rstrip()
    require(body.endswith("\nFinis"), "unexpected book ending")
    parts = re.split(r"(?m)^CHAPTER ([IVX]+)\.\n", body[: -len("\nFinis")])
    require(
        parts[1::2] == ROMAN and all(s.strip() for s in parts[2::2]),
        "chapter roster differs",
    )
    return [piece.strip() for piece in parts[2::2]]


def main(args):
    os.umask(0o077)
    guard_ml()
    if args.tokenizers_site:
        sys.path.append(str(args.tokenizers_site.resolve(strict=True)))
    import tokenizers

    require(
        tokenizers.__version__ == PLAN["tokenizers"],
        "tokenizer implementation version differs",
    )
    tokpath = args.model / "tokenizer.json"
    require(sha(tokpath) == PLAN["model"]["tokenizer.json"], "unrecognized tokenizer")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokpath))
    manifest = args.chapters / "manifest.json"
    require(sha(manifest) == PLAN["book_manifest_sha256"], "old book manifest changed")
    require(
        sha(args.external) == PLAN["external"]["raw_sha256"], "new book bytes differ"
    )
    require(
        not any(p.is_symlink() for p in [args.chapters, *args.chapters.parents]),
        "linked chapter root",
    )
    width = PLAN["recipe"]["context_length"]
    pool, evals = [], {"development": {}, "external": {}}
    train_chapters, dev_chapters = [], []
    train_map = []
    rows = json.loads(manifest.read_bytes())["documents"]
    for doc in rows:
        chapter, split = doc["chapter"], doc["split"]
        if split == "reserve":
            continue
        require(
            (split == "train" and 1 <= chapter <= 49)
            or (split == "development" and 50 <= chapter <= 55),
            "unapproved book partition",
        )
        name = f"{split}/chapter-{chapter:02d}.txt"
        require(doc["relative_path"] == name, "chapter name differs")
        path = args.chapters / name
        require(
            not path.parent.is_symlink()
            and not path.is_symlink()
            and sha(path) == doc["sha256"],
            "chapter changed",
        )
        windows = full_windows(
            tokenizer.encode(
                path.read_text(encoding="utf-8"), add_special_tokens=False
            ).ids,
            width,
        )
        if split == "train":
            train_chapters.append(chapter)
            train_map.extend((chapter, i) for i in range(len(windows)))
            pool.extend(windows)
        else:
            dev_chapters.append(chapter)
            evals[split][chapter] = [
                windows[i]
                for i in midpoint_indices(
                    len(windows), PLAN["same_book"]["windows_per_development_chapter"]
                )
            ]
    require(
        train_chapters == list(range(1, 50)) and dev_chapters == list(range(50, 56)),
        "training/development roster differs",
    )
    for number, chapter in enumerate(persuasion(args.external.read_bytes()), 1):
        windows = full_windows(
            tokenizer.encode(chapter, add_special_tokens=False).ids, width
        )
        evals["external"][number] = [
            windows[i]
            for i in midpoint_indices(
                len(windows), PLAN["external"]["windows_per_chapter"]
            )
        ]
    selected = midpoint_indices(len(pool), PLAN["train_windows"])
    train = [pool[i] for i in selected]
    require(
        {train_map[i][0] for i in selected} == set(train_chapters),
        "training sample omits a chapter",
    )
    check_ids(
        {
            "train": train,
            **{
                s: [w for rows in chapters.values() for w in rows]
                for s, chapters in evals.items()
            },
        }
    )
    out = args.output
    out.mkdir(mode=0o700, exist_ok=False)
    save(out / "train.json", train)
    save(out / "evaluation.json", evals)
    save(out / "train-window-map.json", [train_map[i] for i in selected])
    manifest = {
        "plan_sha256": sha(Path(__file__).with_name("plan.json")),
        "train_windows": len(train),
        "training_chapters": len(train_chapters),
        "evaluation_windows": {
            s: sum(map(len, chapters.values())) for s, chapters in evals.items()
        },
        "files": {
            n: sha(out / n)
            for n in ("train.json", "evaluation.json", "train-window-map.json")
        },
        "reserve_read": False,
        "model_executed": False,
    }
    save(out / "data-manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "chapters", "external", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tokenizers-site", type=Path)
    main(p.parse_args())
