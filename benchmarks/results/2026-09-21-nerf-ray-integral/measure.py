"""Six balanced worker orders, serial and source/binary bound; no retries."""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze import digest, validate

root, target, verified, output = map(Path, sys.argv[1:5])
output.mkdir()
driver = Path(__file__).with_name("run.py")
stages = json.loads((verified / "stages.json").read_text())
assert len(stages) == 12 and all(s["exit_code"] == 0 for s in stages)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
for stage in stages:
    receipt = json.loads((verified / stage["stage"] / "receipt.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["source_unchanged"] is True
    assert receipt["source"]["commit"] == commit
    assert all(digest(root / p) == sha for p, sha in receipt["source"]["files"].items())
frozen = output / "workers"
frozen.mkdir()
shutil.copy2(target / "release/nerf-ray-integral-fixture", frozen / "native")
shutil.copytree(verified / "wasm", frozen / "wasm")
for name in ["wasm.cjs", "torch_reference.py"]:
    shutil.copy2(root / "benchmarks/nerf-ray-integral" / name, frozen / name)
workers = {"native": [str(frozen / "native")],
    "wasm": ["node", str(frozen / "wasm.cjs"), str(frozen / "wasm/nerf_ray_integral_fixture.js")],
    "torch": [sys.executable, "-I", str(frozen / "torch_reference.py"), str(verified / "native-preflight/stdout.log")]}
files = sorted(p for p in frozen.rglob("*") if p.is_file())
before = {str(p): digest(p) for p in files if p.is_file()}
reference = json.loads((verified / "native-preflight/stdout.log").read_text())
preflight = []
for name in workers:
    raw = verified / (name + "-preflight") / "stdout.log"
    preflight.append({"worker": name, "raw_path": str(raw), "raw_sha256": digest(raw),
                      **validate(json.loads(raw.read_text()), reference)})
(output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
results = []
for block, order in enumerate(itertools.permutations(workers)):
    for position, name in enumerate(order):
        name_out = output / f"block-{block}-{name}"
        print("START", block, name, flush=True)
        subprocess.run([sys.executable, "-I", "-B", str(driver), str(root), str(target),
                        str(name_out), *workers[name]], check=True, stdout=subprocess.DEVNULL)
        assert before == {str(p): digest(p) for p in files if p.is_file()}
        raw = name_out / "stdout.log"
        result = validate(json.loads(raw.read_text()), reference)
        results.append({"block": block, "position": position, "order": list(order), "worker": name,
                        "raw_path": str(raw), "raw_sha256": digest(raw), **result})
        (output / "results.json").write_text(json.dumps({"commit": commit, "workers": before,
            "results": results}, indent=2) + "\n")
        print("END", block, name, flush=True)
