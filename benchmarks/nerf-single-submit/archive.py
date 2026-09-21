"""Compact submission-comparison evidence, not numerical reexecution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import analyze, validate_summary

STAGES = [
    "format", "admission", "archive-tests", "native-clippy", "wasm-clippy",
    "backend-tests", "native-build", "freeze-native", "wasm-build", "bindgen",
    "legacy-native", "legacy-browser",
] + [f"round-{r}-{family}" for r in range(3) for family in ("native", "browser", "torch")]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}


def safe_path(root, name):
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError("non-relative artifact path")
    return root / path


def summary(root, exploratory=False):
    def path(family, repeat):
        stem = f"screen-{family}-{repeat + 1}" if exploratory else f"round-{repeat}-{family}"
        return root / (stem + ".json" if family == "browser" else stem + "/stdout.log")
    return analyze(*[[read(path(family, repeat)) for repeat in range(3)]
                     for family in ("native", "browser", "torch")])


def exploration(raw):
    sources, attempts = {}, []
    for path in sorted(raw.glob("*/receipt.json")):
        receipt = read(path)
        source = receipt.pop("source")
        identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        sources[identity] = source
        attempts.append({"stage": path.parent.name, "source_sha256": identity, **receipt,
                         "failed_stderr": (path.parent / "stderr.log").read_text()
                         if receipt["exit_code"] else None})
    return {
        "role": "Exploratory, not pooled with final clean-source measurements. All recorded attempts retained.",
        "sources": sources, "attempts": attempts, "screening": summary(raw, True),
        "decision": "Do not replace the existing default based on submission count; expose explicit scheduling.",
    }


def publish(raw, output):
    accepted = raw / "accepted"
    source = read(accepted / "round-0-native/receipt.json")["source"]
    if source["status"]:
        raise ValueError("accepted source must be committed and clean")
    receipts = []
    for stage in STAGES:
        receipt = read(accepted / stage / "receipt.json")
        if receipt.pop("source") != source or receipt["exit_code"] != 0 or receipt["source_unchanged"] is not True:
            raise ValueError(f"unaccepted source/validation: {stage}")
        receipts.append({"stage": stage, **receipt})
    result = summary(accepted)
    validate_summary(result)
    earlier = exploration(raw)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "results.json", result)
    write(output / "source.json", source)
    write(output / "validation.json", receipts)
    write(output / "exploration.json", earlier)
    shutil.copyfile(accepted / "backend-tests/stdout.log", output / "backend-tests.log")
    for stage in ("admission", "archive-tests"):
        shutil.copyfile(accepted / stage / "stderr.log", output / (stage + ".log"))
    write(output / "local-raw-manifest.json", {
        str(p.relative_to(raw)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(raw.rglob("*")) if p.is_file()
    })
    write(output / "manifest.json", files(output))
    return result["descriptive_summary"]


def verify(root, raw=None, source=None):
    if read(root / "manifest.json") != files(root):
        raise ValueError("archive bytes differ")
    result = read(root / "results.json")
    validate_summary(result)
    earlier = read(root / "exploration.json")
    validate_summary(earlier["screening"])
    for attempt in earlier["attempts"]:
        recorded = earlier["sources"][attempt["source_sha256"]]
        if hashlib.sha256(json.dumps(recorded, sort_keys=True).encode()).hexdigest() != attempt["source_sha256"]:
            raise ValueError("exploratory source identity differs")
    receipts = read(root / "validation.json")
    if [r["stage"] for r in receipts] != STAGES or any(
        r["exit_code"] != 0 or r["source_unchanged"] is not True for r in receipts
    ):
        raise ValueError("incomplete validation records")
    if read(root / "source.json")["status"]:
        raise ValueError("accepted source was not clean")
    if raw is not None:
        for name, record in read(root / "local-raw-manifest.json").items():
            path = safe_path(raw, name)
            if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
                raise ValueError(f"raw bytes differ: {name}")
        if result != summary(raw / "accepted") or earlier != exploration(raw):
            raise ValueError("raw summaries differ")
    if source is not None:
        for name, expected in read(root / "source.json")["files"].items():
            if sha(safe_path(source, name)) != expected:
                raise ValueError(f"measured source differs: {name}")
    return {"status": "passed", "cases": len(result["cases"]), "numerical_reexecution": False,
            "raw_fixity_and_summary_recomputed": raw is not None, "source_checked": source is not None}


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
        value = {"status": "sealed"}
    else:
        value = verify(args.root, args.raw_root, args.source_root)
    print(json.dumps(value, allow_nan=False))
