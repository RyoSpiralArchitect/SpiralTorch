"""Publish an explicit metrics-only allowlist, never private corpus or weights."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "repo", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    root, repo, output = (getattr(args,k).resolve() for k in ("raw","repo","output"))
    output.mkdir(exist_ok=False)
    files = {
        "comparison-baseline.json":"local-before.json",
        "comparison-reviewed.json":"local-after.json",
        "review-before.json":"review-before.json",
        "review-after.json":"review-after.json",
        "learning/comparison.json":"language-intermediate.json",
        "learning-reviewed/comparison.json":"language-final.json",
        "browser-reviewed.json":"browser.json",
    }
    manifests = {}
    for source, target in files.items():
        raw = (root/source).read_bytes()
        data = json.loads(raw)
        write(output/target, data)
        manifests[source] = {"sha256":sha(raw),"bytes":len(raw),"public":target}
    lines = (root/"attention-integration-final.log").read_text().splitlines()
    reports = [json.loads(line[line.index("{"):]) for line in lines
               if '"schema":"spiraltorch.resident_attention_vjp.validation.v1"' in line]
    if len(reports) != 1:
        raise ValueError("expected one complete native integration report")
    write(output/"native.json",reports[0])
    for report in (reports[0], json.loads((root/"browser-reviewed.json").read_bytes())):
        if len(report["checks"]) != 36 or len(report["range_checks"]["cases"]) != 46 or len(report["shift_checks"]["cases"]) != 13:
            raise ValueError("shared fixture coverage drift")
    for lane in ("learning", "learning-reviewed"):
        raw = (root/lane/"build-receipt.json").read_bytes()
        receipt = json.loads(raw)
        receipt["command"] = [s.replace(str(repo),"<repo>") for s in receipt["command"]]
        receipt["private_receipt_sha256"] = sha(raw)
        write(output/(lane+"-build.json"),receipt)
    # Hashes preserve failures and complete private runs without publishing
    # their operands, checkpoint tensors, raw gradients or source corpus.
    preserved = ["fixture.json", "native-baseline.json", "native-reviewed.json",
                 "range-before.log", "range-baseline.log", "range-after.log",
                 "range-shift-tail-before.log", "range-tail-after.log", "denominator-before.log",
                 "attention-runtime-reviewed.log", "attention-runtime-guarded.log", "attention-integration-final.log",
                 "backend-static-final.log", "byte-study-static-final.log", "python-tests.log",
                 "probe-baseline", "probe-tail", "probe-reviewed",
                 "build-baseline.jsonl", "build-reviewed.jsonl", "build-wasm-reviewed.jsonl",
                 "web-reviewed/target/resident-attention-vjp-web/resident_attention_vjp_browser.js",
                 "web-reviewed/target/resident-attention-vjp-web/resident_attention_vjp_browser_bg.wasm",
                 "web-reviewed/crates/st-backend-wgpu/tests/attention_vjp_browser.html"]
    preserved += [f"{lane}/{name}" for lane in ("learning","learning-reviewed")
                  for name in ("request.json","native.json","checkpoint.json","build-receipt.json","started.json","runner.py","resident_byte_learning")]
    preserved += ["learning-reviewed/restored.json", "learning-reviewed/restored-checkpoint.json"]
    for source in preserved:
        raw = (root/source).read_bytes()
        manifests[source] = {"sha256":sha(raw),"bytes":len(raw),"public":False}
    public_files = {p.name:sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    restored = {name:(root/f"learning-reviewed/{name}.json").read_bytes() ==
                     (root/f"learning-reviewed/{restored}.json").read_bytes()
                for name,restored in (("native","restored"),("checkpoint","restored-checkpoint"))}
    write(output/"manifest.json", {"schema":"spiraltorch.language_vjp_isolation.publication.v1",
                                  "private_artifacts":manifests, "public_sha256":public_files,
                                  "completed_checkpoint_reload_byte_equal":restored,
                                  "scope":"local reproduction record; not independent attestation; all strict outcomes retained"})


if __name__ == "__main__":
    main()
