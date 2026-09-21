"""Exercise normal Python invocations in retained isolated archive copies."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

source, output = map(Path, sys.argv[1:3])
output.mkdir()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
before = {name: sha(source / name) for name in ["manifest.json", "test_verify.py", "test_analyze.py", "verify.py", "analyze.py"]}
cases = {
    "plain-archive-tests": lambda p: [sys.executable, str(p / "test_verify.py")],
    "plain-numeric-tests": lambda p: [sys.executable, str(p / "test_analyze.py")],
    "unittest-discovery-with-cache": lambda p: [sys.executable, "-m", "unittest", "discover", "-s", str(p), "-p", "test_*.py"],
}
receipts = []
for name, command in cases.items():
    path = output / name
    path.mkdir()
    archive = path / "archive"
    shutil.copytree(source, archive)
    if "with-cache" in name:
        cache = archive / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "unrelated.pyc").write_bytes(b"inert synthetic cache fixture")
    argv = command(archive)
    with (path / "stdout.log").open("xb") as out, (path / "stderr.log").open("xb") as err:
        result = subprocess.run(argv, cwd=path, stdout=out, stderr=err)
    receipt = {"case": name, "command": argv, "exit_code": result.returncode, "source_sha256": before,
               "source_unchanged": before == {k: sha(source / k) for k in before}}
    (path / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipts.append(receipt)
    print(name, result.returncode, flush=True)
(output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
