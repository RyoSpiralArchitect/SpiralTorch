# Language And Geometry Serving Gates

**Source-only after 0.4.31.** This is a local, measured model-selection primitive,
not a registry, network service, HF executor or durable deployment manager.
Build an installed candidate using the [build guide](build-and-validation.md).

An [executable byte-LM artifact](byte-model-artifacts.md) carries learned weights,
geometry and routers together. `ByteCorpusServingGate` fixes evaluation bytes,
windows and all limits **before** an initial model becomes active. Rust measures
both incumbent and candidate with fresh owners on the same runtime and windows.
Python and the standalone WASM client never reconstruct acceptance semantics.

## Pin The Evaluation Policy

```python
import json
from pathlib import Path
import spiraltorch as st

# Obtain IDs independently from a trusted selection record, not by hashing an
# untrusted download and treating its self-consistency as authenticity.
initial = st.nn.ByteCorpusModelArtifact.from_json(
    Path("initial-artifact.json").read_bytes(), expected_id=trusted_initial_id,
)
candidate = st.nn.ByteCorpusModelArtifact.from_json(
    Path("candidate-artifact.json").read_bytes(), expected_id=trusted_candidate_id,
)
gate = st.nn.ByteCorpusServingGate.from_json(Path("serving-gate.json").read_bytes())
gate.validate_artifact(initial)  # CPU preflight: fixed batch/window and policy.
slot = gate.bootstrap_wgpu(initial)  # Actual inference; absolute health required.
```

The request has this shape. Numbers below illustrate the schema, **not** suitable
production thresholds. Predeclare limits and evaluation selection independently
of candidate results; changing either requires a new slot and bootstrap.

```json
{
  "schema": "spiraltorch.byte_corpus.serving_gate.v1",
  "documents": [[7, 8, 9, 10, 11]],
  "batches": [[[0, 0]], [[0, 2]]],
  "steps": 2,
  "limits": {
    "max_mean_ce": 6.0,
    "max_mean_ce_regression": 0.01,
    "max_batch_ce_regression": 0.1,
    "max_geometry_centered_rms": 0.5,
    "max_router_mass_error": 0.00001,
    "min_router_normalized_entropy": 0.1
  }
}
```

Each `[document_index, byte_offset]` selects `steps + 1` bytes without crossing a
document boundary. Every batch has the same row count and must match the model's
fixed window. Bounds: 64 MiB JSON, 1..1024 documents, 1..128 batches, 1..32 rows,
1..512 steps. Unknown fields, missing limits and nonfinite values are rejected.
The request SHA256 covers the exact UTF-8 bytes, including data and limits;
`to_json()` preserves those bytes. Keep the gate file private if its data are private.

The caller owns rights, held-outness, split selection and candidate authenticity.
Windows may overlap. Neither a digest nor acceptance proves statistical
independence, training provenance or generalization. The gate does not enforce
an architecture/source allowlist: compatible fixed-window byte models may differ
in geometry or architecture. Apply any such selection rules before admission.

## Protect Named Evaluation Groups

Use `spiraltorch.byte_corpus.serving_gate.v2` when language, task or data-source
groups must each meet their own limits. It replaces the top-level `batches` and
`limits` with `cohorts`; `documents` and `steps` remain shared. Each cohort has
`name`, `batches` and all six explicit `limits` from v1. No global-average rule
can override a failing cohort.

```python
# Start from a reviewed v1 request. These labels are caller declarations, not
# language detection. Choose the actual windows and thresholds before measuring.
cohort_request = {
    "schema": "spiraltorch.byte_corpus.serving_gate.v2",
    "documents": request_v1["documents"],
    "steps": request_v1["steps"],
    "cohorts": [
        {"name": "general-prose", "batches": prose_batches, "limits": prose_limits},
        {"name": "code", "batches": code_batches, "limits": code_limits},
    ],
}
gate = st.nn.ByteCorpusServingGate.from_json(json.dumps(cohort_request))
assert gate.schema_version == 2
assert gate.cohort_names == ["general-prose", "code"]
slot = gate.bootstrap_wgpu(initial)
permit = slot.prepare_candidate(candidate)
assessment = json.loads(permit.to_json())
for cohort in assessment["cohorts"]:
    print(cohort["name"], cohort["mean_ce_delta"], cohort["failed_checks"])
# Only permit.accepted can authorize slot.activate(permit).
```

Bounds are 1..32 unique names, 1..128 **total** batches and the same fixed batch
shape across every cohort. Names are nonempty, trimmed, at most 128 UTF-8 bytes
and contain no control characters. Unicode names are supported. Each cohort
needs at least one batch. Data/windows may overlap, including across cohorts;
the core does not assert independent samples or detect near-duplicate text.
Adding/removing a cohort or changing its policy requires a new gate and slot.

Measurements reset between cohorts: mean and worst paired batch CE, merged
centered RMS per block/head, and router row count/mass/entropy all belong to
that cohort alone. Bootstrap and incumbent rechecks require absolute health
in **every** cohort; candidate admission additionally requires all paired
regression limits. No weights or normalization mix cohorts together.

v2 produces `serving_bootstrap.v2` with a `cohorts` array of
`{name, limits, observation}`, and `candidate_assessment.v2` with one full
paired assessment per cohort. Root `failed_checks` contains
`{cohort, checks}` entries only for failing cohorts. There is intentionally no
root candidate mean. v1 keeps its original report shape and numbers and exposes
one implicit `cohort_names == ["default"]`, `schema_version == 1`.
Activation/rollback events remain `serving_transition.v1`; they carry the exact
gate hash. Neither report version is an importable activation permit.

## Measure, Activate, Roll Back

```python
permit = slot.prepare_candidate(candidate)  # Real paired inference, no updates.
assessment = json.loads(permit.to_json())
if permit.accepted:
    event = json.loads(slot.activate(permit))
    host = st.nn.ByteLmBatch.from_documents(documents, selections, steps)
    forward = slot.forward(slot.prepare_batch(host))
    assert forward.artifact_id == candidate.artifact_id
    logits = forward.prediction_tensor().snapshot().read_values()

    # An explicit operator decision, not an automatic response to one metric.
    rollback = json.loads(slot.rollback(slot.generation))
    assert slot.artifact_id == initial.artifact_id
```

The assessment reports actual artifact IDs/revisions, gate SHA256, adapter,
per-batch CE, mean CE, target-byte count and exact f32 little-endian logit hashes.
All logits/losses must be finite. No corpus text or model weights are included.

- Absolute CE checks the mean over fixed-size batches, equivalent to equal
  weighting of target bytes. Relative CE checks both mean and the worst paired
  batch delta against the incumbent. Negative regression limits require an
  improvement; equality passes. A good mean cannot hide a failing batch, but
  improvement in one row can offset another row within that batch.
- Geometry health reuses Rust `CausalBiasMoments`: remove each causal row's mean,
  exclude future positions from moments, merge sums/counts across evaluation
  batches within a cohort, then take RMS per block/head. The threshold is **cohort-merged RMS**,
  not a per-window peak bound. Full bias tensors must still be finite.
- Router health covers every observed row in each block, including all heads
  and contextual query positions. It independently checks maximum probability
  mass error and minimum entropy of `p_i / sum(p)`, divided by `ln(K)`. Zero mass,
  nonfinite or out-of-range weights are rejected. Ordinary models have empty
  geometry/router observations, not invented zeros or perfect scores.

Bootstrap applies absolute limits. Preparation rechecks incumbent absolute
health, measures the candidate, and adds relative checks. A failed preparation
or activation does not change the active owner, generation or rollback history.

The candidate is an unforgeable-by-report, process-local typed handle bound to
this exact slot, generation and incumbent. JSON is only an observation: there
is no candidate import API. Cross-slot, stale, reused or rejected permits cannot
activate. One activation attempt consumes the handle, even if rejected; Python
and WASM retain its read-only report. This is not a cryptographic authorization
boundary against code already running inside the process.

Activation swaps complete owners after validation and keeps the exact prior
owner for one rollback. Rollback checks the expected generation, consumes that
history and invalidates earlier permits. Retained forwards keep their original
artifact ID and output. No retraining, reset or checkpoint reconstruction occurs.
State/history is not persisted across a process restart; start with a newly
validated bootstrap, not an old report or permit.

The active slot also supports [bounded greedy byte generation](byte-generation.md)
with an expected serving generation. The same Rust decoder is exposed to Python
and the standalone browser client; adoption and rollback now select actual
generated output as well as fixed-window logits. Teacher-forced gate acceptance
does not certify free-running generation quality or health on unseen contexts.

## Rust And Browser Clients

Rust uses `ByteCorpusServingGate::from_json`,
`ResidentByteServing::bootstrap(runtime, artifact, gate).await`,
`prepare_candidate(artifact).await`, `activate(candidate)` and
`rollback(expected_generation)`. CPU-only builds support gate parsing/preflight;
Python `bootstrap_wgpu()` raises `NotImplementedError` without WGPU, not a CPU
fallback. Resident execution remains fixed-window frozen inference with retained
backward tapes, not a KV-cache or serving-throughput optimization.

The standalone `resident_byte_learning_browser` WASM example exposes
`ByteCorpusServingGate`, `ResidentByteServing` and `ResidentByteCandidate`.
Browser imports accept opaque artifact JSON plus a separately pinned ID;
`forward(Uint8Array)` maps B windows of T+1 bytes to logits/loss/identity.
Generation counters are canonical decimal strings to avoid JS rounding.
The gate exposes `schema_version`, ordered `cohort_names`, `batch_count` and
`batch_shape` in Rust, Python and the standalone browser client.
`byte_serving_browser.html` runs positive, negative, stale, cross-slot and exact
rollback checks using a private `target/resident-byte-learning-web/serving-input.json`.
See the [browser build recipe](../byte_corpus_learning.md#browser-execution).
These are **not yet packaged `st-wasm` exports**.

Existing HF adapter lineage, promotion reports and PEFT execution are separate
clients with different artifact types. This slice does not retrofit them or
claim that native byte-model artifacts are HF adapters. The next integration
should reuse this measured boundary rather than add another Python policy
engine. Mechanics success is not a language-quality gain or evidence for any
default geometry; that still requires independent, matched-budget evaluation.
