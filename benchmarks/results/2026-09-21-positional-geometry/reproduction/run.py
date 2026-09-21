"""Record bounded commands and source identity without normalizing their output."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root, target, output = map(Path, sys.argv[1:4])
command = sys.argv[4:]
output.mkdir()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
files = ["crates/st-core/src/util/rope_lru.rs", "crates/st-vision/src/nerf/mod.rs",
         "crates/st-vision/src/nerf/encoding.rs", "crates/st-vision/src/nerf/field.rs",
         "crates/st-vision/src/nerf/trainer.rs", "crates/st-vision/Cargo.toml",
         "crates/st-vision/tests/nerf_geometry.rs", "crates/st-vision/tests/nerf_regression.rs",
         "crates/st-nn/src/layers/linear.rs",
         "crates/st-tensor/src/pure.rs", "crates/st-nn/src/module.rs",
         "crates/st-nn/src/layers/sequential.rs", ".github/workflows/ci.yml", "Cargo.lock"]
source = {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
          "files": {f: sha(root / f) for f in files}}
fixture = Path(__file__).parent / "fixture"
fixture_files = [p for p in fixture.rglob("*") if p.is_file() and p.suffix in {".rs", ".toml", ".lock"}]
source["fixture_files"] = {str(p.relative_to(fixture)): sha(p) for p in fixture_files}
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4")
for key in ["SPIRALTORCH_AUTOTUNE_STORE", "SPIRALTORCH_AUTOTUNE", "SPIRAL_DETERMINISTIC", "SPIRAL_DETERMINISTIC_REDUCTION"]:
    env.pop(key, None)
start = time.monotonic()
with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
    result = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
receipt = {"source": source, "command": command, "exit_code": result.returncode,
           "seconds": time.monotonic() - start, "environment": {k: env[k] for k in ["CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "RAYON_NUM_THREADS"]},
           "source_unchanged": source["files"] == {f: sha(root / f) for f in files}
               and source["fixture_files"] == {str(p.relative_to(fixture)): sha(p) for p in fixture_files}}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
sys.exit(result.returncode or (0 if receipt["source_unchanged"] else 1))
