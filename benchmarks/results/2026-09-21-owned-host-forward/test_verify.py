"""Positive control and rehashed semantic-tamper tests of the public verifier."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
def run(root):
    return subprocess.run([sys.executable, "-B", "-I", str(root / "verify.py"), str(root)], capture_output=True).returncode
def change(root, name, transform, rehash=True):
    path = root / name
    obj = json.loads(path.read_text())
    transform(obj)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    if rehash:
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
cases = [
    ("hash", lambda r: change(r, "provenance.json", lambda o: o.update(candidate="wrong"), False)),
    ("source", lambda r: change(r, "candidate-native-build/receipt.json", lambda o: o["source"].update(commit="wrong"))),
    ("failed-check", lambda r: change(r, "verification/nn/receipt.json", lambda o: o.update(exit_code=1))),
    ("invalid-case", lambda r: change(r, "native/candidate-a.json", lambda o: o["cases"][0].update(valid=False))),
    ("missing-case", lambda r: change(r, "native/candidate-a.json", lambda o: o["cases"].pop())),
    ("layout-bits", lambda r: change(r, "wasm/candidate-a.json", lambda o: o["contracts"][0].update(output_sha256="wrong"))),
    ("gradient-bits", lambda r: change(r, "wasm/candidate-a.json", lambda o: o["contracts"][0].update(gradient_sha256="wrong"))),
    ("missing-guard", lambda r: change(r, "wasm/candidate-a.json", lambda o: o.update(error_cases=0))),
    ("allocations", lambda r: change(r, "native/candidate-a.json", lambda o: o["cases"][1].update(allocation_calls=12))),
    ("negative-erasure", lambda r: change(r, "baseline-contract/receipt.json", lambda o: o.update(exit_code=0))),
    ("ratio", lambda r: change(r, "native/comparison.json", lambda o: o["comparisons"][0].update(ratio=999))),
    ("worker", lambda r: change(r, "nn/receipt.json", lambda o: o["sha256"].update(candidate="wrong"))),
]
assert run(source) == 0, "positive control failed"
for name, mutation in cases:
    with tempfile.TemporaryDirectory(prefix="gelu-evidence-") as temporary:
        root = Path(temporary) / "evidence"
        shutil.copytree(source, root)
        mutation(root)
        assert run(root) != 0, name + " was accepted"
print(json.dumps({"passed": len(cases) + 1, "positive_control": True, "rejected": [n for n, _ in cases]}))
