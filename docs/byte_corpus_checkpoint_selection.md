# Saved-Model Selection For Language Studies

Source API after PyPI 0.4.31; build the current source. Selection belongs to the
same Rust owner as training, geometry, optimizer history and artifact creation.
It does not add an independent Python scoring policy or automatically deploy a
model. It works without a GPU, including in a CPU-only Python wheel.

## Keep The Weights You Evaluate

`checkpoint_every` schedules validation, not permanent retention of every model.
Use `advance` / `advance_wgpu` to retain actual study checkpoints at chosen
evaluation boundaries. An early minimum in a loss history is not enough: its
weights may not exist. Decide the saved cursors before inspecting the results.

```python
import json
from pathlib import Path
import spiraltorch as st

study = st.nn.ByteCorpusStudy.from_json(Path("request.json").read_bytes())
saved = [study.checkpoint_from_json(Path(name).read_bytes())
         for name in ("checkpoint-0.json", "checkpoint-64.json", "checkpoint-128.json")]
artifact, decision_json = study.select_checkpoint(
    saved, "seed11_geometry", policy="fixed_window_frozen.v1"
)
decision = json.loads(decision_json)
print(decision["selected"], artifact.artifact_id)
inference = artifact.restore_wgpu()  # optional, requires the WGPU build
```

The fixed rule is the smallest mean validation CE for one exact case name, with
exact ties resolved to the earliest supplied revision. The mean is accumulated
in float64 from stored float32 batch losses, as in study reports. Batch sizes and
windows are fixed by the request. No rounding threshold, seed ranking, hidden
patience or automatic rate tuning is applied.

Candidates must be 1..32 strictly increasing saved cursors, at evaluation
boundaries, with at most 64 MiB combined canonical checkpoint JSON. Every case
in every snapshot is validated, not only the selected case. Training and
validation history prefixes must match bit-for-bit across the sequence. Foreign
request bytes, forked histories, duplicate/reversed cursors, and models saved
between evaluation boundaries reject. A single supplied snapshot is allowed;
the record explicitly lists that limited selection domain.

The returned artifact is the existing `ByteCorpusModelArtifact`, byte-identical
to extracting it directly from the winning source checkpoint. It retains learned
Poincare/Fisher/mixture/query state and complete AdamW history when present.
Serving uploads weights only; explicit training restore retains moments/clocks.
Keep the original request and all supplied checkpoints as private companions.
The decision JSON contains hashes and metrics, not weights or corpus bytes.

## Same Choice From Rust And Browser

Rust calls `study.select_checkpoint(&[&early, &later], case_name, policy)` and
receives `ByteCorpusCheckpointSelection { artifact, report }`. The packaged
[WASM client](wasm_byte_language_mlops.md#select-actual-saved-checkpoints) exposes
`study.checkpointCandidates()` with incremental `appendJson` imports and
`candidates.selectCheckpoint(caseName, policy)`. It returns an owned artifact and
the same decision JSON without a device, including in an `nn`-only build.

The standalone browser example also exports `select_resident_byte_learning_checkpoint(requestJson,
checkpointJsonStrings, caseName, policy)`, returning JSON with `artifact_json`
and `report`. Keep each checkpoint/model JSON as an opaque string through
JavaScript so signed-zero float32 bits are not reconstructed differently.

Both browser paths bound raw input strings before parsing; the shared Rust
selector additionally bounds their canonical representation. Neither selection
entry point requests an adapter. Actual inference is still an explicit GPU call.

## Selection Is Not Promotion

Once used to select checkpoints, these validation windows are **development
data**, not a final untouched test. Retain every seed and geometry control. Use
a separate predeclared probe for selected-versus-final evaluation and generation;
related repository prose still cannot establish broad language quality.

The score is a validated stored observation, not an authenticated measurement
or proof that training occurred. Existing serving gates still need fresh
evaluation, geometry safety and rollback policy. A good CE or artifact hash
does not establish coherent free-running text, production readiness, or a
geometric advantage over ordinary attention.

The installed `spiral-byte-corpus-run` command and
`spiraltorch.run_byte_corpus_segments` retain opaque checkpoints in a
[durable local store](python/byte-study-runs.md). Each run claims a new
owner-only output directory before GPU execution. The historical
`tools/run_byte_corpus_segments.py` is a compatibility wrapper, not a second
trainer. Neither entry point retries a failed directory automatically:

```bash
spiral-byte-corpus-run request.json private-run \
  --stop 0 --stop 64 --stop 128 --stop 256
```

The last cursor must equal the request's total update budget. Parent directories
must exist. Checkpoint validation/training remain Rust-owned. If a process is
interrupted, retain its claim and partial files. Recovery explicitly selects a
sealed segment and forks into a new directory; it never adopts unsealed files.
