"""Run new integration regressions against the previously admitted main rlibs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root, deps, output = map(Path, sys.argv[1:4])
output.mkdir()
source = root / "crates/st-vision/tests/nerf_ray_integral.rs"
shutil.copy2(source, output / source.name)
inputs = {"test_source": output / source.name}
for crate in ["st_core", "st_nn", "st_tensor", "st_vision"]:
    matches = list(deps.glob("lib" + crate + "*.rlib"))
    assert len(matches) == 1, (crate, matches)
    inputs[crate] = matches[0]
def hashes():
    return {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in inputs.items()}
before = hashes()
command = ["rustc", "+1.98.0", "--edition=2021", "-O", "--test", "--cfg", 'feature="nerf"',
           str(inputs["test_source"]), "-L", "dependency=" + str(deps), "-o", str(output / "tests")]
for crate, path in inputs.items():
    if crate != "test_source":
        command += ["--extern", crate + "=" + str(path)]
with (output / "build.stdout").open("xb") as out, (output / "build.stderr").open("xb") as err:
    build = subprocess.run(command, cwd=root, stdout=out, stderr=err)
receipt = {"command": command, "inputs_sha256": before, "build_exit": build.returncode,
           "runtime_commit": "2ab49c1ffa2f05a9df7461e9981ec1ca788f5174",
           "note": "Read-only linkage to admitted #2109 rlibs with the same source tree as main"}
if build.returncode == 0:
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        run = subprocess.run([str(output / "tests"), "--nocapture", "--test-threads=1"],
                             cwd=root, stdout=out, stderr=err)
    receipt.update(test_exit=run.returncode, binary_sha256=hashlib.sha256((output / "tests").read_bytes()).hexdigest())
receipt["inputs_unchanged"] = before == hashes()
(output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
sys.exit(build.returncode)
