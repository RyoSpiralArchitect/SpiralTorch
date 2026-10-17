"""Stage only synthetic test assets for the existing frozen loopback server."""
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
    for name in ("repo", "package", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, exist_ok=False)
    site = args.output / "site"
    target = site / "bindings/st-wasm/tests"
    target.mkdir(parents=True)
    mapping = {
        "byte_lm_browser.html": "crates/st-backend-wgpu/tests/parameter_clip_browser.html",
        "byte_lm_browser.mjs": "bindings/st-wasm/tests/byte_lm_browser.mjs",
        "byte_lm_boundaries.mjs": "bindings/st-wasm/tests/byte_lm_boundaries.mjs",
    }
    original = {}
    for destination, source in mapping.items():
        path = args.repo / source
        before = identity(path)
        shutil.copyfile(path, target / destination)
        if identity(path) != before or identity(target / destination) != before:
            raise ValueError("staged source changed")
        original[source] = before
    with (args.output / "input.json").open("x", encoding="utf-8") as output:
        json.dump(dict(schema="synthetic_in_module_no_private_fixture.v1"), output)
    manifest = dict(
        original_source=original,
        source={str(path.relative_to(site)): identity(path) for path in target.iterdir()},
        package={str(path.relative_to(args.package)): identity(path)
                 for path in args.package.rglob("*") if path.is_file()},
        fixture=identity(args.output / "input.json"))
    with (args.output / "inputs.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(json.dumps(dict(frozen_assets=len(manifest["source"]))))


if __name__ == "__main__":
    main()
