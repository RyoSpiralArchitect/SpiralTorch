"""Record the review-only CI wiring follow-up without relabeling measured sources."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo, out = map(Path, sys.argv[1:3])
git = lambda *args: subprocess.check_output(["git", *args], cwd=repo)
provenance = json.loads((out / "provenance.json").read_bytes())
parent = provenance["validated_source"]
source = git("rev-parse", "HEAD").decode().strip()
assert git("diff", "--name-only", parent, source).decode().splitlines() == [".github/workflows/ci.yml"]
patch = git("diff", parent, source, "--", ".github/workflows/ci.yml")
(out / "ci-followup.patch").write_bytes(patch)
provenance["ci_followup"] = {"parent": parent, "source": source, "sha256": hashlib.sha256(patch).hexdigest()}
(out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
