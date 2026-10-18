# Portable HF Geometry Adapters

[Language MLOps](../language_mlops.md) | [Placement and learning](../geometry_adapter_stack.md)

**Source-candidate API after 0.4.31; not a published wheel release.**

`GeometryAdapterStack.export_artifact()` packages learned geometric tensors,
native recipes, ordered HF insertion paths, parameter/buffer roles, trainability
and train/eval flags.
`GeometryAdapterStack.from_artifact()` reconstructs working adapters, not an
intervention report. Rust owns validation and immutable state; Python owns
Torch modules and placement. WASM `nn` builds expose the same artifact contract
without claiming browser HF execution.

## Export And Use

Start with a float32 HF model loaded through `load_hf_causal_lm_artifact`, and an
explicitly placed geometry stack as in the [placement guide](../geometry_adapter_stack.md).
Use the verified runtime identity (base model and tokenizer), not a directory name:

```python
from pathlib import Path
import spiraltorch as st

# model, tokenizer, config, load_report = st.load_hf_causal_lm_artifact(...)
identity = load_report["runtime_identity_after_model"]
if identity is None or identity["status"] != "ready":
    raise ValueError("a verified HF runtime identity is required")
base_id = identity["observed_identity_id"]

# Finish backward and exit geometry.attach(model) before exporting.
artifact = geometry.export_artifact(base_model_identity=base_id)
Path("private/geometry.json").write_text(artifact.to_json(), encoding="utf-8")
selected_artifact_id = artifact.artifact_id
# Retain selected_artifact_id and base_id in the experiment's trusted selection.
```

On a later process, load the base with `expected_runtime_identity_id=base_id`,
then separately load its geometry:

```python
artifact = st.GeometryAdapterArtifact.from_json(
    Path("private/geometry.json").read_text(encoding="utf-8"),
    selected_artifact_id, base_id,
)
geometry = st.GeometryAdapterStack.from_artifact(
    artifact, expected_base_model_identity=base_id, device="cpu",
)
model.config.use_cache = False
model.eval().requires_grad_(False)
geometry.eval()
with geometry.attach(model):
    logits = model(input_ids, use_cache=False).logits
```

For generation, supply each complete unpadded single-document prefix. To learn,
optimize `geometry.parameters()` explicitly and finish backward inside the
attachment scope. Restoring does not consume the CPU RNG. Keep geometry and
hidden states on the same device; native geometry still runs on Rust f32 CPU
with explicit transfers, not a new resident GPU path.

The base ID is caller-provided, not a fingerprint of a live Python object.
Verify the HF load and prevent subsequent base-weight mutation. Do not supply
an arbitrary hash or take the expected artifact ID from an untrusted payload.
Matching hashes establish neither authentication nor model quality.

## Contract And Limits

Supported exact built-ins: WaveGate (including fixed/learned radius), Topos,
all four Elliptic residual variants, FractionalMemory, History, L2History,
GainHistory and AngleGainHistory, including configured lag windows. Custom
subclasses and arbitrary code imports are rejected; no pickle is used.

- `full_prefix_f32_no_cache.v1` keeps existing full-prefix requirements. No
  padding/packed documents, KV cache, AMP, distributed or compiled path is added.
  Unpadded document boundaries remain the caller's obligation.
- Rust rejects unknown schemas/fields, duplicate/nested paths, wrong tensor
  names/shapes/roles, nonfinite tensors and invalid native parameter domains.
  Fractional recipes also check input-independent coefficient/scale overflow
  through the actual st-frac builders, preserving normalized/empty zero maps. It does
  not prove finite activations for every possible input.
- v1 allows 1..128 placements, widths up to 65,536, 1,048,576 total f32 values
  and 64 MiB JSON. Fractional kernels allow 65,536 taps each and 1,048,576 total
  taps per artifact; remaining kernel sizes must fit portable u32 budgets.
- `to_json()` preserves exact UTF-8 bytes for the pinned SHA256. `state_json()`
  returns typed normalized state, not replacement hash bytes. `create_json()`
  issues a new artifact from a local record, not a trusted external load.

This is **not a full training checkpoint**. Base weights, tokenizer, optimizer
moments/groups, schedule, data cursor, RNG, pending gradients and promotion
permissions remain separate HF-orchestrator companions. Export only at an
explicit completed-update boundary; export success does not prove that boundary.
Tiny GPT-2/Llama tests check exact continuation with separately restored AdamW,
dropout disabled and identical data/base weights, not universal resume or quality.

## Rust And WASM

Rust callers create `st_nn::geometry_artifact::GeometryAdapterRecord`, call
`GeometryAdapterArtifact::create(record)`, inspect immutable typed `record()`,
and verify `from_json(bytes, expected_id, expected_base_id)`. Validation uses
existing native WaveGate/Topos/Elliptic/Fractional constructors and domain guards.

WASM `nn` exports `GeometryAdapterArtifact.createJson`, `fromJson`, `toJson`,
`stateJson`, `artifactId` and `baseModelIdentity`. These transport the same state,
not Torch hooks or Transformers execution. Rust, installed Python and actual
WASM/Node test a shared 13-placement synthetic fixture.

See the [qualification record](../../benchmarks/results/2026-10-18-portable-geometry/README.md).
Real artifacts contain learned weights; keep them private unless intentionally
publishing them.
