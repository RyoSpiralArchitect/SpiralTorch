"""Freeze a selection-only site for the existing allowlisted loopback server."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def identity(path):
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "package", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, exist_ok=False)
    site = args.output / "site"
    target = site / "bindings/st-wasm/tests"
    target.mkdir(parents=True)
    mapping = {
        "byte_lm_browser.html": "byte_lm_selection_browser.html",
        "byte_lm_browser.mjs": "byte_lm_selection.mjs",
        "byte_lm_boundaries.mjs": "byte_lm_boundaries.mjs",
    }
    originals = {}
    for destination, source in mapping.items():
        path = args.repo / "bindings/st-wasm/tests" / source
        shutil.copyfile(path, target / destination)
        originals[str(path.relative_to(args.repo))] = identity(path)
        if identity(target / destination) != identity(path):
            raise ValueError("source changed while staging")
    shutil.copyfile(args.fixture, args.output / "input.json")
    manifest = dict(
        schema="spiraltorch.wasm_checkpoint_selection.inputs.v1",
        original_source=originals,
        source={str(path.relative_to(site)): identity(path) for path in target.iterdir()},
        package={str(path.relative_to(args.package)): identity(path)
                 for path in args.package.rglob("*") if path.is_file()},
        fixture=identity(args.output / "input.json"))
    with (args.output / "inputs.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(json.dumps(dict(frozen_assets=len(manifest["source"]),
                          fixture_sha256=manifest["fixture"]["sha256"])))


if __name__ == "__main__":
    main()
