"""Compact publication and byte/structure verification, not numerical replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import analyze, describe, summarize
from contract import ATOL, RTOL, KEYS, key

STAGES = ["final-format", "final-admission", "final-native-clippy", "final-wasm-clippy",
          "final-backend-tests", "final-native-build", "final-freeze-native", "final-wasm-build", "final-bindgen"]
STAGES += [f"round-{r}-{family}" for r in range(3) for family in ("native", "browser", "torch")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def files(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}


def raw_summary(raw):
    reports = [[read(raw / (f"round-{r}-browser.json" if family == "browser" else f"round-{r}-{family}/stdout.log"))
                for r in range(3)] for family in ("native", "browser", "torch")]
    return analyze(*reports)


def publish(raw, output):
    source = read(raw / "round-0-native/receipt.json")["source"]
    if source["status"]:
        raise ValueError("final source must be committed and clean")
    receipts = []
    for stage in STAGES:
        receipt = read(raw / stage / "receipt.json")
        if receipt.pop("source") != source or receipt["exit_code"] != 0 or receipt["source_unchanged"] is not True:
            raise ValueError(f"unaccepted source/validation: {stage}")
        receipts.append({"stage":stage, **receipt})
    result = raw_summary(raw)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "results.json", result)
    write(output / "source.json", source)
    write(output / "validation.json", receipts)
    negative = read(raw / "torch-preflight-v1/receipt.json")
    if negative["exit_code"] == 0 or negative["source_unchanged"] is not True:
        raise ValueError("missing original MPS failure")
    write(output / "negative-attempt.json", {
        "receipt":negative,
        "stderr":(raw / "torch-preflight-v1/stderr.log").read_text(),
        "per_case_diagnosis":read(raw / "torch-diagnose-v1/stdout.log"),
        "stage_diagnosis":read(raw / "torch-stages-v1/stdout.log"),
        "resolution":"Keep the tolerance and independent f64 oracle; stabilize f32 thin alpha with a fourth-order polynomial. These changed eager controls are the timed controls, not the rejected original MPS expm1 path.",
    })
    shutil.copyfile(raw / "final-backend-tests/stdout.log", output / "backend-tests.log")
    shutil.copyfile(raw / "final-admission/stderr.log", output / "admission-tests.log")
    write(output / "local-raw-manifest.json", {str(p.relative_to(raw)):{"sha256":sha(p),"bytes":p.stat().st_size}
                                              for p in sorted(raw.rglob("*")) if p.is_file()})
    write(output / "manifest.json", files(output))
    return result["descriptive_summary"]


def verify(root, raw=None, source=None):
    if read(root / "manifest.json") != files(root):
        raise ValueError("archive bytes differ")
    result = read(root / "results.json")
    if result["status"] != "passed" or result["schema"] != "spiraltorch.nerf_direct_summary.v1" or result["rounds"] != 3:
        raise ValueError("archive schema/status")
    cases = result["cases"]
    if len(cases) != len(KEYS) or {key(c) for c in cases} != KEYS:
        raise ValueError("archive condition grid")
    if result["tolerance"] != {"atol":ATOL,"rtol":RTOL} or result["descriptive_summary"] != describe(cases):
        raise ValueError("changed tolerance or aggregate summary")
    for case in cases:
        if case["summary"] != summarize(case["intervals"]):
            raise ValueError("summary does not match published timings")
        if set(case["max_scaled_errors"]) != {"native_staged", "native_direct", "browser_staged", "browser_direct", "cpu", "mps"}:
            raise ValueError("missing numerical route")
        if any(type(v) not in (int,float) or not 0 <= v <= 1 for v in case["max_scaled_errors"].values()):
            raise ValueError("unaccepted reported numerical error")
    receipts = read(root / "validation.json")
    if [r["stage"] for r in receipts] != STAGES or any(r["exit_code"] != 0 or r["source_unchanged"] is not True for r in receipts):
        raise ValueError("incomplete validation records")
    if raw is not None:
        for name, record in read(root / "local-raw-manifest.json").items():
            path = safe_path(raw, name)
            if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
                raise ValueError(f"raw bytes differ: {name}")
        if result != raw_summary(raw):
            raise ValueError("raw report recomputation differs")
    if source is not None:
        for name, expected in read(root / "source.json")["files"].items():
            if sha(safe_path(source, name)) != expected:
                raise ValueError(f"measured source differs: {name}")
    return {"status":"passed", "cases":len(cases), "raw_fixity_and_summary_recomputed":raw is not None,
            "source_checked":source is not None, "numerical_reexecution":False}


def safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name:
        raise ValueError("non-relative artifact path")
    return root / path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["publish", "verify", "seal"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.mode == "publish":
        if args.raw_root is None:
            parser.error("publish requires --raw-root")
        value = publish(args.raw_root, args.root)
    elif args.mode == "seal":
        write(args.root / "manifest.json", files(args.root))
        value = {"status":"sealed"}
    else:
        value = verify(args.root, args.raw_root, args.source_root)
    print(json.dumps(value, allow_nan=False))
