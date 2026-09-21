"""Bind the final publication bytes, including documentation and the verifier."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
         for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"]
(root / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
