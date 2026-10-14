# Executable Byte-LM Artifacts

**Source-only after 0.4.31.** Build an installed candidate using the
[build guide](build-and-validation.md). This is the next step after the
[study-to-model handoff](byte-corpus-studies.md#from-a-study-to-inference),
not an HF adapter format, model registry or promotion service.

## Weights And Geometry Travel Together

`ByteCorpusModelArtifact` binds one actual model checkpoint, its learned geometry
and mixture/query routers, its selected study case and an explicit inference
policy. The same Rust implementation owns creation, validation and restore for
Rust, Python and the standalone WASM qualification client. Python does not
reconstruct the model identity or enforce a different inference policy.

```python
from pathlib import Path
import spiraltorch as st

study = st.nn.ByteCorpusStudy.from_json(Path("request.json").read_bytes())
saved = study.checkpoint_from_json(Path("study-checkpoint.json").read_bytes())
artifact = study.case_artifact(
    saved, "my-explicit-case-name", policy="fixed_window_frozen.v1",
)
study.validate_artifact(saved, artifact)  # CPU-only; validates all source cases.
with open("selected-artifact.json", "x", encoding="utf-8") as out:
    out.write(artifact.to_json())
print(artifact.artifact_id)              # Pin this in the trusted selection record.
```

There is currently one explicit operating policy, `fixed_window_frozen.v1`:
the checkpoint's fixed batch/window shape, immutable model/geometry/router
parameters, reset positions and causal geometry on every window, and **no
external score biases**. The runtime uses the existing default resident kernels.
This policy does not choose temperature, a sampler or a deployment winner.

## Pinned Fresh Inference

Obtain `trusted_id` from your trusted selection record, independently of the
artifact download. Computing a digest from an untrusted file and passing it
back as the expected ID only checks self-consistency, not authenticity.

```python
loaded = st.nn.ByteCorpusModelArtifact.from_json(
    Path("selected-artifact.json").read_bytes(), expected_id=trusted_id,
)
model = loaded.restore_wgpu()  # ResidentByteInference, not a training handle.

# Use B selections and T steps matching the model's fixed window. Each selection
# reads T+1 bytes; the last shifted byte is a target for optional held-out loss.
host = st.nn.ByteLmBatch.from_documents(documents, selections, steps)
forward = model.forward(model.prepare_batch(host))
logits = forward.prediction_tensor().snapshot().read_values()
assert forward.artifact_id == loaded.artifact_id
loss = forward.next_byte_loss().loss_tensor().snapshot().read_values()[0]
```

The inference owner has no SGD, backward, parameter-tensor or external-bias API.
Each retained forward carries its originating `artifact_id`, even after its
owner is dropped or a different model is selected. The existing training API's
forwards have `artifact_id=None`. Invalid candidate imports leave an existing
owner unchanged. The separate [serving gate](byte-serving-gates.md) can measure
paired candidates, activate an accepted owner and explicitly roll back; it is
not an automatic promotion service or a durable deployment manager.

`loaded.checkpoint()` is a deliberate low-level escape hatch: restoring a
separate mutable training owner does not mutate this artifact. Later updates to
that separate owner must not be labelled with the old artifact ID.

The current inference wrapper reuses the existing forward, including retained
backward tapes. It is not an inference-kernel optimization, KV cache, streaming
decoder or text-generation API. Cross-device bitwise equality is not promised.
A CPU-only wheel can create, pin, transport and source-validate artifacts;
`restore_wgpu()` explicitly raises `NotImplementedError`, without CPU fallback.

## Identity And Provenance Boundaries

The ID is `sha256:<lowercase hex>` of the **exact complete UTF-8 artifact bytes**.
Whitespace or re-encoding changes it. Keep the string opaque through Python/JS;
do not parse and reserialize it during transport. Import is bounded to 64 MiB
and rejects mismatched IDs before JSON decoding or device acquisition.

The envelope includes the exact request SHA256 and a SHA256 of the source study
checkpoint's **Rust `to_json()` output**, not arbitrary outer file whitespace.
That latter digest includes all cases and histories, including unselected cases.
Keep the original request and full study checkpoint locally for resume and for
`study.validate_artifact(saved, loaded)`. The deployment artifact does not embed
the original corpus, initial study models, histories or frozen-rate training
policy. Weights themselves may still contain sensitive learned information.

Standalone import validates the pinned envelope, model and revision. Without
the private companions it does not independently verify the declared source.
Even companion validation is consistency checking, not a signature or proof
that historical training actually ran. Partial and revision-zero models may be
exported; acceptance never implies completion, quality or readiness to deploy.

## Shared Clients And Next Gates

Rust uses `ByteCorpusStudy::case_artifact` with
`ByteCorpusInferencePolicy::FixedWindowFrozenV1`,
`ByteCorpusModelArtifact::from_json(bytes, expected_id)` and
`ByteCorpusStudy::validate_artifact`. `restore_wgpu` returns a constrained
`ResidentByteInference`; its forwards retain their originating ID.

The standalone `resident_byte_learning_browser` WASM example exposes
`pack_resident_byte_learning_case_artifact`,
`verify_resident_byte_learning_artifact` and `infer_resident_byte_artifact`.
The last creates a fresh owner and maps logits/loss for qualification. The
[browser recipe](../byte_corpus_learning.md#browser-execution) and
`byte_learning_resume_browser.html` exercise them. These functions are **not**
yet exports of the packaged `st-wasm` module.

The [serving gate](byte-serving-gates.md) pins language and geometry-health
criteria, binds actual paired inference to the artifact/operating conditions,
and provides accepted-candidate activation plus one-step rollback. Held-outness
and threshold selection remain caller responsibilities. Reuse existing HF lineage
and execution management; do not silently change established HF adapter IDs or
pretend native byte-model artifacts and HF adapters are interchangeable.
Correct restore and identity checks are not evidence that geometry improves
language quality. That requires independent, matched-budget evaluation.
