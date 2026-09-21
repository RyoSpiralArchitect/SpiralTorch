"""Verify archived bytes and condition gates, not a replay of Rust/GPU computation."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

sys.dont_write_bytecode = True


def load_measure(root):
    spec = importlib.util.spec_from_file_location("source_crosscut_measure", root / "measure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "spiraltorch.source_crosscut.archive.v1":
        raise ValueError("Unexpected manifest schema")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Archive file set differs from manifest")
    for name, record in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or PureWindowsPath(name).drive or relative.as_posix() != name:
            raise ValueError("Unsafe manifest path")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Archive symlink escapes root")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Hash/size mismatch: " + name)
    report = load_measure(root).report(root)
    receipt = json.loads((root / "verification/receipt.json").read_text())
    source = json.loads((root / "provenance.json").read_text())
    if receipt["status"] != "passed" or receipt["source"]["commit"] != source["candidate_commit"]:
        raise ValueError("Verification source/status mismatch")
    if len(receipt["steps"]) != 13 or any(s["exit_code"] != 0 for s in receipt["steps"]):
        raise ValueError("Incomplete verification steps")
    return dict(files=len(actual), candidate_numerical_passes=report["candidate_numerical_passes"],
                baseline_numerical_failures=report["baseline_numerical_failures"],
                numerical_replay=False, status="passed")


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
