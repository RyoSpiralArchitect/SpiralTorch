"""Render ALL retained continuations privately; does not load a model or score text."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import ORIGINAL, PLAN, ROSTER, audit, guard_ml, read, require, sha  # noqa: E402


def render(run, model, site, output):
    report = audit(run)
    guard_ml()
    sys.path.append(str(site.resolve(strict=True)))
    import tokenizers

    tokfile = model / "tokenizer.json"
    require(
        tokenizers.__version__ == ORIGINAL["tokenizers"]
        and sha(tokfile) == ORIGINAL["model"]["tokenizer.json"],
        "wrong decoder for retained IDs",
    )
    tok = tokenizers.Tokenizer.from_file(str(tokfile))
    require(
        not output.resolve().is_relative_to(run.resolve()),
        "do not write into the recorded attempt",
    )
    lines = [
        "Private continuations from a completed native radius interchange",
        "NLL scores the supplied reference; displayed text is separate, free greedy generation.",
        "",
    ]
    for prompt in PLAN["prompts"]:
        pid = prompt["id"]
        lines.extend(
            [
                f"[{pid} | {prompt['group']}]",
                "Prompt: " + prompt["prefix"],
                "Reference: " + prompt["continuation"],
            ]
        )
        for case in ROSTER:
            score = report["models"][case]["prompts"][pid]
            path = run / case / (pid + ".json")
            require(sha(path) == score["raw_sha256"], "retained generation changed")
            ids = read(path)["generated"]
            text = json.dumps(
                tok.decode(ids, skip_special_tokens=False), ensure_ascii=False
            )
            lines.append(f"{case:12s}  target NLL {score['nll']:.6f}  {text}")
        lines.append("")
    os.umask(0o077)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "model", "tokenizers-site", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    render(args.run, args.model, args.tokenizers_site, args.output)
