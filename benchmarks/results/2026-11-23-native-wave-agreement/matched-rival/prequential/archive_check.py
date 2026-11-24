"""Pack only the authenticated training inputs, read them back and replay both reports."""
import argparse
import json
import math
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

T = runpy.run_path(str(Path(__file__).with_name("study.py")))
need, sha = T["need"], T["sha"]
ALLOWED = re.compile(r"(?:identity\.json|sealed\.json|receipt\.json|states/[0-9a-f]{64}\.json|"
                     r"(?:11|17|31|73)-(?:affine|wave)_(?:rival|full)\.(?:jsonl|g32))")


def check_numbers(report, raw, capture_prefix, journal_prefix):
    initial = {}
    for line in raw(capture_prefix+"attempt/rows.jsonl").splitlines():
        r = json.loads(line)
        c = r["receipt"]["contrast"]
        initial[r["host"], r["key"]] = c["target_logit"]-c["alternative_logit"]
    for host, family in (("affine", "affine"), ("both", "wave")):
        for s, group in report["summary"][host]["orders"].items():
            path = journal_prefix+s+"-"+family+"_"
            rivals, full = ([json.loads(x) for x in raw(path+a+".jsonl").splitlines()] for a in ("rival", "full"))
            rows = [r for r in report["rows"] if r["host"] == host and r["seed"] == int(s)]
            need(len(rows) == len(rivals) == len(full), "partial journal")
            for j, (r, cr, fr) in enumerate(zip(rows, rivals, full)):
                history = rivals[j-1]["step"]["result_id"] if j else r["initial"]
                observed = cr["declared"]["mean_target_minus_declared_rival"]
                need(r["before_rival"] == history and r["key"] == cr["key"] == fr["key"]
                     and r["prior"] == j and r["change"] == observed-initial[host, r["key"]]
                     and r["rival_minus_full_nll"] == T["f32"](cr["declared"]["full_ce"])-T["f32"](fr["loss"]),
                     "different recorded margin, full NLL or incoming history")
            for k, stats in group["forecasts"].items():
                if stats is None:
                    need(all(r[k] is None for r in rows), "missing control data")
                else:
                    err = sum((r["change"]-r[k])**2 for r in rows[1:])
                    need(math.isclose(math.sqrt(err/(len(rows)-1)), stats["rmse"], rel_tol=2e-15),
                         "RMSE disagrees with raw noninitial rows")


def run(capture, matched, docs, archive):
    roots = {"capture": capture, "matched": matched}
    pins = {"capture/SHA256SUMS": T["S"]["ARCHIVE"], "matched/SHA256SUMS": T["E"]["ROOTS"][2]}
    for kind, raw in docs.items():
        r = json.loads(raw)
        need(r["kind"] == kind and set(r["capture_inputs"]) ==
             {"plan.json", "attempt/rows.jsonl", "attempt/result.json", "attempt/receipt.json"}, "capture scope")
        for base, items in (("capture/"+T["S"]["SUBDIR"][kind]+"/", r["capture_inputs"]),
                            ("matched/"+kind+"/attempt/", r["training_inputs"])):
            for k, v in items.items():
                if base.startswith("matched/"):
                    need(ALLOWED.fullmatch(k) is not None, "not a training artifact")
                pins[base+k] = v
    with ZipFile(archive, "x", ZIP_DEFLATED) as z:
        for name, pin in sorted(pins.items()):
            top, rest = name.split("/", 1)
            data = (roots[top]/rest).read_bytes()
            need(sha(data) == pin, "changed source: "+name)
            z.writestr(name, data)
    with TemporaryDirectory(dir=archive.parent) as tmp, ZipFile(archive) as z:
        need(sorted(z.namelist()) == sorted(pins), "unexpected recovery members")
        root = Path(tmp)
        for name, pin in pins.items():
            data = z.read(name)
            need(sha(data) == pin, "changed recovery member: "+name)
            out = root/name
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("xb") as f:
                f.write(data)
        for kind, original in docs.items():
            check_numbers(json.loads(original), z.read, "capture/"+T["S"]["SUBDIR"][kind]+"/",
                          "matched/"+kind+"/attempt/")
            replay = T["enc"](T["run"](root/"capture", root/"matched", kind))
            need(replay == original, "training-only replay differs: "+kind)
    print(json.dumps(dict(members=len(pins), archive_sha256=sha(archive.read_bytes()),
                          report_sha256={k:sha(v) for k, v in docs.items()}, byte_identical=True), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("capture", "matched", "rehearsal_report", "gpt2_report", "archive"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    run(a.capture, a.matched, {"rehearsal": a.rehearsal_report.read_bytes(), "gpt2": a.gpt2_report.read_bytes()},
        a.archive)
