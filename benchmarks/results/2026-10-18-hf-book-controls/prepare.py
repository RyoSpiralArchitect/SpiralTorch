#!/usr/bin/env python3
"""Prepare pinned novel chapters; no tokenizer, model, network or training code."""
import argparse
import hashlib
import json
from pathlib import Path
import re

START = "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"
HEADING = re.compile(r"^chapter[ \t]+([IVXLCDM]+)\.?\]?[ \t]*$", re.I | re.M)
ILLUSTRATION = re.compile(r"\[Illustration\b.*?\]", re.S)
DATA_POLICY = {"normalization":"utf8_bom_optional_crlf_to_lf","body":"numbered_chapters_only_drop_illustration_blocks_and_final_colophon","expected_chapters":61,"train_chapters":[1,49],"development_chapters":[50,55],"reserve_chapters":[56,61],"reserve_policy":"write_only_no_tokenization_training_evaluation_or_selection","window_policy":"Rust HfGeometryDataset disjoint full contexts per chapter; drop tails"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bounded(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("input exceeds the preparation byte budget")
    return data


def roman(number):
    result = ""
    for value, letters in ((50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while number >= value:
            result += letters
            number -= value
    return result


def chapters(raw, expected_hash):
    if len(raw) > 5_000_000 or sha(raw) != expected_hash:
        raise ValueError("source differs from the pinned bounded corpus")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) >= text.index(END):
        raise ValueError("exactly one ordered Gutenberg body is required")
    body = text[text.index(START) + len(START):text.index(END)]
    headings = list(HEADING.finditer(body))
    if [m.group(1).upper() for m in headings] != [roman(i) for i in range(1, 62)]:
        raise ValueError("expected exactly chapters I through LXI in order")
    result = []
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        content = body[match.end():end]
        content, removed = ILLUSTRATION.subn("", content)
        if i == 60:
            marker = "CHISWICK PRESS:--"
            if content.count(marker) != 1:
                raise ValueError("expected exactly one final colophon")
            content = content[:content.index(marker)]
        content = content.strip() + "\n"
        if "[Illustration" in content or len(content) < 100:
            raise ValueError("incomplete illustration or implausibly short chapter")
        result.append((content, removed))
    return result


def prepare(raw_path, plan_path, destination):
    if destination.exists():
        raise ValueError("destination must be a new private directory")
    raw, plan_bytes = read_bounded(raw_path, 5_000_000), read_bounded(plan_path, 65_536)
    plan = json.loads(plan_bytes)
    policy = plan["data_policy"]
    expected = {"train": [1, 49], "development": [50, 55], "reserve": [56, 61]}
    if (plan["schema"] != "spiraltorch.hf_book_control_plan.v1"
            or policy != DATA_POLICY):
        raise ValueError("this preparation only implements the declared 49/6/6 chapter policy")
    records = chapters(raw, plan["source"]["raw_sha256"])
    if len({sha(text.encode("utf-8")) for text, _ in records}) != 61:
        raise ValueError("duplicate chapter content")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "spiraltorch.hf_book_preparation.v1", "plan_id": "sha256:" + sha(plan_bytes),
                "raw_source_id": "sha256:" + sha(raw), "documents": [], "reserve_used": False}
    for chapter, (text, removed) in enumerate(records, 1):
        split = next(name for name, (lo, hi) in expected.items() if lo <= chapter <= hi)
        encoded = text.encode("utf-8")
        identity = sha(encoded)
        relative = f"{split}/chapter-{chapter:02d}.txt"
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        manifest["documents"].append({"chapter": chapter, "split": split, "relative_path": relative,
                                      "bytes": len(encoded), "characters": len(text),
                                      "sha256": identity, "illustration_blocks_removed": removed})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.raw, args.plan, args.output)
    print(json.dumps({"plan_id": result["plan_id"], "source_id": result["raw_source_id"],
                      "chapters": len(result["documents"]), "reserve_used": False}))


if __name__ == "__main__":
    main()
