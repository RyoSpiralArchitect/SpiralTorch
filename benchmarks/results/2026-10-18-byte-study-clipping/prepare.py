"""Generate tiny synthetic requests and stage frozen, allowlisted browser assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import test_byte_corpus_gradient_clip as fixtures


def identity(path):
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, exist_ok=False)
    requests = []
    for name, optimizer in (("sgd", {"kind": "stateless_sgd.v1"}), ("adamw", fixtures.OPT_TEST.recipe())):
        v9 = fixtures.OPT_TEST.wrap(fixtures.learning_design(), optimizer)
        for suffix, request in (("v9", v9), ("off", fixtures.bind(v9, {"kind": "disabled.v1"})),
                                ("active", fixtures.bind(v9))):
            label = name + "-" + suffix
            path = args.output / (label + ".json")
            fixtures.STUDY.write_new(path, request)
            requests.append(dict(name=label, request_json=path.read_text(encoding="utf-8")))
    fixtures.STUDY.write_new(args.output / "input.json", dict(requests=requests))
    site = args.output / "site"
    tests = site / "bindings/st-wasm/tests"
    tests.mkdir(parents=True)
    mapping = {"byte_lm_browser.html": "byte_lm_clipping.html",
               "byte_lm_browser.mjs": "byte_lm_clipping.mjs",
               "byte_lm_boundaries.mjs": "byte_lm_boundaries.mjs"}
    original = {}
    for destination, source in mapping.items():
        path = ROOT / "bindings/st-wasm/tests" / source
        before = identity(path)
        shutil.copyfile(path, tests / destination)
        if identity(tests / destination) != before or identity(path) != before:
            raise ValueError("source changed while staging")
        original[str(path.relative_to(ROOT))] = before
    manifest = dict(original_source=original,
        source={str(p.relative_to(site)): identity(p) for p in tests.iterdir()},
        package={str(p.relative_to(args.package)): identity(p) for p in args.package.rglob("*") if p.is_file()},
        fixture=identity(args.output / "input.json"))
    fixtures.STUDY.write_new(args.output / "inputs.json", manifest)
    print(json.dumps(dict(requests=len(requests), fixture=manifest["fixture"])))


if __name__ == "__main__":
    main()
