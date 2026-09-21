"""Verify archive bytes, complete conditions and source identity; no GPU replay."""
import hashlib
import itertools
import json
import math
from pathlib import Path

GUARDS = {"shape_rejected", "zero_rejected", "subnormal_width_rejected",
          "inherited_error_rejected", "retained_version"}
STAGES = {"format", "inventory", "browser-syntax", "admission", "native-clippy",
          "wasm-clippy", "backend-tests", "native", "freeze-native", "wasm",
          "freeze-wasm", "bindgen", "browser", "torch"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root, raw=False, source_root=None):
    files = {str(p.relative_to(root)): p for p in root.rglob("*")
             if p.is_file() and p.name != "manifest.json"}
    manifest = read(root / "manifest.json")
    assert set(files) == set(manifest)
    for name, digest in manifest.items():
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert not files[name].is_symlink() and sha(files[name]) == digest, name
    source, result = read(root / "source.json"), read(root / "results.json")
    assert source["status"] == "" and result["measured_commit"] == source["commit"]
    assert result["status"] == "passed"
    cases = result["cases"]
    assert len(cases) == 36
    assert {(c["rays"], c["samples"], c["varying"], c["seed"]) for c in cases} == set(
        itertools.product([1, 65, 256], [1, 8, 64], [False, True], [None, 17]))
    for name in ["native", "browser"]:
        assert set(result["guards"][name]) == GUARDS
        assert all(value is True for value in result["guards"][name].values())
        errors = [c["max_abs_errors"][name] for c in cases]
        assert all(math.isfinite(e) and e >= 0 for e in errors)
        assert result["max_abs_errors"][name] == max(errors)
    validation = read(root / "validation.json")
    assert len(validation) == len(STAGES) and {v["stage"] for v in validation} == STAGES
    assert all(v["exit_code"] == 0 and v["source_unchanged"] is True for v in validation)
    failure = read(root / "failed-long-prefix/receipt.json")
    assert failure["exit_code"] != 0 and failure["source_unchanged"] is True
    if raw:
        for name, item in read(root / "local-raw-manifest.json").items():
            path = Path(name)
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], name
    if source_root:
        for name, digest in source["files"].items():
            assert sha(source_root / name) == digest, name
    return len(files)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    count = verify(args.archive, args.raw, args.source_root)
    print(f"Verified {count} files, 36 conditions and 14 stages; numerical replay not performed")
