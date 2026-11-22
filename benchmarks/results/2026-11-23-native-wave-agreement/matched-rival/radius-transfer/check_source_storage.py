"""Rehearse a source-built wheel against the sealed miniature decoder, never GPT-2."""
import argparse
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys


HERE = Path(__file__).resolve().parent
STEP = runpy.run_path(str(HERE / "check_source_step.py"))
E, PLAN, RESULT = (STEP[k] for k in ("E", "PLAN", "RESULT"))
need, sha, enc = (E[k] for k in ("need", "sha", "enc"))


def checked_plan(toy, extension):
    raw = (toy / "plan.json").read_bytes()
    need(sha(raw) == PLAN, "only the pinned artificial rehearsal is accepted")
    p = json.loads(raw)
    need(p["kind"] == "rehearsal" and p["model_pins"] is None and
         (p["shape"], p["updates"], p["scored_rows"], p["seconds"]) == ([2, 16, 2, 32], 8, 16, 60)
         and p["sources"] == E["sources"](), "wrong synthetic scope or changed protocol")
    need(re.fullmatch("[0-9a-f]{64}", extension) and extension != p["identity"]["extension"],
         "supply the distinct source-built extension digest")
    original = (toy / "attempt" / "result.json").read_bytes()
    need(sha(original) == RESULT and original == enc(E["replay"](p, toy / "attempt")),
         "synthetic reference does not reproduce")
    # Only the execution identity changes; keep the original plan and attempt read-only.
    p["identity"]["extension"] = extension
    return p


def check_outputs(toy, twin, p):
    old, new = toy / "attempt", twin / "attempt"
    raw = (new / "result.json").read_bytes()
    need(raw == enc(E["replay"](p, new)), "candidate result does not replay")
    result = json.loads(raw)
    need(result.pop("plan") == sha(enc(p)), "wrong candidate plan")
    ref_raw = (old / "result.json").read_bytes()
    need(sha(ref_raw) == RESULT, "changed reference result")
    ref = json.loads(ref_raw)
    need(ref.pop("plan") == PLAN and result == ref, "synthetic summary differs")
    # Include intermediate state blobs, not just the endpoint identities.
    def files(root):
        return {f.relative_to(root) for pat in ("*.g32", "*.jsonl", "*.f32le", "states/*.json", "sealed.json")
                for f in root.glob(pat)}
    members = files(old)
    need(members == files(new) and len(members) == 19, "unexpected numerical evidence set")
    digests = {}
    for name in sorted(members):
        data = (old / name).read_bytes()
        need((new / name).read_bytes() == data, "raw evidence differs: " + str(name))
        digests[str(name)] = sha(data)
    return dict(reference_plan=PLAN, reference_result=RESULT, candidate_plan=sha(enc(p)),
                candidate_extension=p["identity"]["extension"], artificial_updates=result["updates"],
                artificial_predictions=result["scored"], pretrained_calls=0, matching_files=digests)


def run(toy, extension, twin):
    p = checked_plan(toy, extension)
    twin.mkdir(mode=0o700, exist_ok=False)
    E["save"](twin / "plan.json", p)
    args = [sys.executable, "-I", "-B", str(HERE / "experiment.py")]
    scope = ["rehearsal", str(twin.resolve()), sha(enc(p))]
    # The frozen CLI itself supervises one CPU worker for at most sixty seconds.
    subprocess.run(args + ["run"] + scope, check=True)
    subprocess.run(args + ["verify"] + scope, check=True)
    report = check_outputs(toy, twin, p)
    E["save"](twin / "comparison.json", report)
    print("%d artificial updates, %d predictions, %d byte-identical files" %
          (report["artificial_updates"], report["artificial_predictions"], len(report["matching_files"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("toy", type=Path)
    ap.add_argument("extension")
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    run(args.toy, args.extension, args.output)
