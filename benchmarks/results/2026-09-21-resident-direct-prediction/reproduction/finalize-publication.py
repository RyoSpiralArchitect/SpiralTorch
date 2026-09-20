"""Generate the public archive's file manifest after manual documentation edits."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
summary = json.loads((root / "summary.json").read_bytes())
records = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        data = path.read_bytes()
        records.append(dict(path=str(path.relative_to(root)), bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest()))
manifest = dict(schema="spiraltorch.direct_prediction_manifest.v1",
    source=summary["source"], files=records)
with (root / "manifest.json").open("w" if "--replace" in sys.argv[2:] else "x") as stream:
    json.dump(manifest, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(dict(files=len(records), bytes=sum(r["bytes"] for r in records),
    manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())))
