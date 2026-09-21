"""Hash exact bytes and complete archive inventory, excluding only this manifest."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = {str(p.relative_to(root)): {"bytes": p.stat().st_size,
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob("*"))
    if p.is_file() and p != root / "manifest.json"}
(root / "manifest.json").write_text(json.dumps({
    "schema": "spiraltorch.positional_geometry.evidence.v1", "numerical_replay": False,
    "files": files}, indent=2) + "\n")
