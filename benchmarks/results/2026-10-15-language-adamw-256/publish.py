"""Publish bounded results/hashes, never corpus requests, weights or wheels."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualification = json.loads((args.root / "qualification.json").read_bytes())
    selection = json.loads((args.root / "selection.json").read_bytes())
    browser = json.loads((args.root / "browser-selection.json").read_bytes())
    comparison = json.loads((args.root / "comparison.json").read_bytes())
    records = []
    for case in qualification["cases"]:
        generated = case["generation"]["generated_bytes"]
        pairs = [(a, b) for row in generated for a, b in zip(row, row[1:])]
        records.append(dict(**case, repetition=dict(
            adjacent_repeat_fraction=sum(a == b for a, b in pairs) / len(pairs),
            mean_repeated_bigram_fraction=statistics.mean(
                1 - len(set(zip(row, row[1:]))) / (len(row) - 1) for row in generated),
            max_constant_run=max(len(list(group)) for row in generated for _, group in itertools.groupby(row)))))
    groups = []
    for arm in ("ordinary", "geometry", "geometry_frozen"):
        group = [r for r in records if r["cursor"] == 256 and r["case_name"].split("_", 1)[1] == arm]
        groups.append(dict(arm=arm, seeds=len(group),
                           development_mean_ce=statistics.mean(r["development"]["mean_ce"] for r in group),
                           probe_mean_ce=statistics.mean(r["probe"]["mean_ce"] for r in group),
                           mean_space_fraction=statistics.mean(r["space_fraction"] for r in group),
                           mean_repeated_bigram_fraction=statistics.mean(r["repetition"]["mean_repeated_bigram_fraction"] for r in group)))
    summary = dict(schema="spiraltorch.byte_corpus.language_budget_summary.v1",
                   request_sha256=qualification["request_sha256"], means=groups,
                   baselines=qualification["baselines"], probe=qualification["probe"],
                   prompt_sha256=qualification["prompt_sha256"], cases=records,
                   scope="Three-seed related-document pilot, not tuned optimizer evidence, independent language benchmark or serving promotion")
    write(args.output / "results.json", summary)
    write(args.output / "selection.json", selection)
    write(args.output / "comparison.json", comparison)
    write(args.output / "browser-selection.json", browser)
    write(args.output / "preparation.json", json.loads((args.root / "paired/preparation.json").read_bytes()))
    files = ["PLAN.md", "request.json", "paired/request.json", "paired/preparation.json", "torch.json",
             "comparison.json", "native/claim.json", "native/completed.json", "selection.json",
             "qualification.json", "evaluation/claim.json", "browser-selection.json"]
    files += [f"native/{name}-{stop}.json" for stop in (0,64,128,256)
              for name in ("checkpoint", "report", "segment")]
    files += ["selection/" + item["file"] for item in selection["artifacts"]]
    files += [f"evaluation/case-{i}.json" for i in range(len(records))]
    write(args.output / "private-receipts.json", {
        "schema":"spiraltorch.byte_corpus.private_receipts.v1",
        "records":[dict(path=name, bytes=(args.root/name).stat().st_size, sha256=digest(args.root/name)) for name in files]})
    print(json.dumps(dict(cases=len(records), means=groups), indent=2))


if __name__ == "__main__":
    main()
