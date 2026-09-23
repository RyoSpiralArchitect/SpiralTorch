"""Append-only source-stable execution stages; never overwrite prior evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root, target, output = map(Path, sys.argv[1:4])
command = sys.argv[4:]
if not command:
    raise SystemExit("expected ROOT TARGET NEW_OUTPUT COMMAND...")
output.mkdir(parents=True, exist_ok=False)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=root)


def identity():
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    prefixes = ("crates/st-backend-wgpu/", "crates/st-kernel-contracts/", "crates/st-tensor/",
                "crates/st-nn/", "benchmarks/layer-norm-resident/", "bindings/st-wasm/tests/")
    exact = {"Cargo.toml", "Cargo.lock", ".github/workflows/ci.yml", "tools/test_resident_browser.cjs"}
    selected = sorted(n for n in set(names) if n in exact or n.startswith(prefixes))
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in selected if (root / n).is_file()}


before = identity()
(output / "runner.py").write_bytes(Path(__file__).read_bytes())
source = {"commit": git("rev-parse", "HEAD").decode().strip(),
          "status": git("status", "--porcelain").decode(), "files": before}
(output / "source.patch").write_bytes(git("diff", "HEAD", "--", *before))
for name in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"):
    if name in before:
        path = output / "untracked" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / name).read_bytes())
env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="4")
start = time.monotonic()
started_at = datetime.now(timezone.utc).isoformat()
with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
    try:
        code = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err).returncode
    except OSError as error:
        err.write(str(error).encode())
        code = 127
receipt = {"source": source, "command": command, "exit_code": code,
           "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
           "seconds": time.monotonic() - start, "source_unchanged": before == identity(),
           "environment": {k: env[k] for k in ["CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "RAYON_NUM_THREADS"]}}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "source"}))
sys.exit(code or (0 if receipt["source_unchanged"] else 1))
