"""Bounded replay of frozen studies; does not introduce a new quality study."""
import hashlib
import json
from pathlib import Path
import sys
import time

import spiraltorch as st

root, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "tools"))
import byte_corpus_study as protocol
import verify_byte_corpus_resume as resume

archive = output.parent
pilot = archive / "2026-10-09-byte-corpus-matched-learning"
query = archive / "2026-10-13-byte-corpus-query-controls-v8"
summary = {"scope": "Python client numerical replay and checkpoint transport; not a new quality or speed study",
           "installed_module": st.__file__, "version_metadata": st.__version__,
           "inputs": {}, "stages": []}


def read(path):
    raw = path.read_bytes()
    summary["inputs"][str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def write(name, text):
    with (output / name).open("xb") as stream:
        stream.write(text.encode() if isinstance(text, str) else protocol.encoded(text))


def stage(name, operation):
    start = time.monotonic()
    value = operation()
    item = {"name": name, "elapsed_seconds": time.monotonic() - start}
    summary["stages"].append(item)
    print(json.dumps(item), flush=True)
    return value


raw = read(pilot / "frozen/request.json")
request = protocol.decode_json(raw)
study = st.nn.ByteCorpusStudy.from_json(raw)
study.validate_segment(37)
full = stage("v1_full_128", study.run_wgpu)
write("v1-full.json", full)
reference = protocol.decode_json(read(pilot / "torch-reference.json"))
comparison = protocol.compare(raw, reference, protocol.decode_json(full))
write("v1-torch-comparison.json", comparison)
first = stage("v1_pause_37", lambda: study.advance_wgpu(37))
write("v1-partial-37.json", first.report_json)
saved = first.checkpoint.to_json()
write("v1-checkpoint-37.json", saved)
other = st.nn.ByteCorpusStudy.from_json(raw)
loaded = other.checkpoint_from_json(saved.encode())
finished = stage("v1_resume_to_128", lambda: other.advance_wgpu(resume=loaded))
write("v1-resumed.json", finished.report_json)
write("v1-checkpoint-128.json", finished.checkpoint.to_json())
verification = resume.verify(request, study.request_sha256, protocol.decode_json(full),
    protocol.decode_json(first.report_json), protocol.decode_json(finished.report_json),
    protocol.decode_json(saved))
write("v1-resume-verification.json", verification)
assert first.checkpoint.to_json() == saved
historical_native = protocol.decode_json(read(pilot / "native-final.json"))
summary["v1"] = {
    "request_sha256": study.request_sha256,
    "cases": len(request["cases"]), "updates_per_case": study.total_updates,
    "numerical_checks_passed": True, "split_resume_exact": True,
    "historical_native_cases_exact": resume.exact(historical_native["cases"], protocol.decode_json(full)["cases"]),
    "comparison": comparison,
}
assert summary["v1"]["historical_native_cases_exact"]

# Cross-client completed checkpoint imports exercise transport, not retraining.
raw = read(query / "input/request.json")
study = st.nn.ByteCorpusStudy.from_json(raw)
reference = protocol.decode_json(read(query / "torch.json"))
summary["v8_completed_handoffs"] = []
for client in ("native", "browser"):
    checkpoint = study.checkpoint_from_json(read(query / f"{client}-checkpoint-128.json"))
    before = checkpoint.to_json()
    result = stage(f"v8_{client}_completed_handoff", lambda: study.advance_wgpu(resume=checkpoint))
    assert result.checkpoint.to_json() == before
    actual = protocol.decode_json(result.report_json)
    baseline = protocol.decode_json(read(query / f"{client}.json"))
    assert resume.exact(actual["cases"], baseline["cases"])
    comparison = protocol.compare(raw, reference, actual)
    write(f"v8-{client}-handoff.json", result.report_json)
    write(f"v8-{client}-comparison.json", comparison)
    summary["v8_completed_handoffs"].append({
        "source_client": client, "request_sha256": study.request_sha256,
        "completed_updates": checkpoint.completed_updates, "cases": len(actual["cases"]),
        "checkpoint_round_trip_exact": True, "historical_client_cases_exact": True,
        "numerical_checks_passed": True, "new_updates": 0,
    })
write("qualification.json", summary)
print(json.dumps({"qualification": "PASS", "v1_cases": summary["v1"]["cases"],
                  "handoffs": summary["v8_completed_handoffs"]}), flush=True)
