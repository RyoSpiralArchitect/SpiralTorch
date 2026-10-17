# Optimizer-Aware Byte-Language Studies

Source API after PyPI 0.4.31; build the current source for this interface.
The v9 request binds a declared optimizer to an unchanged v1-v8 geometry design.
V10 additionally binds an explicit active-gradient clipping policy.
This separates **what geometry is learned** from **how parameters are updated**.
The same Rust runner owns native, Python and browser execution.

## Recipe, Not A Second Runtime

Complete any RMS calibration, headwise, mixture or contextual preparation first.
Then wrap that request without changing its data, cases or initial parameters:

```json
{
  "schema": "spiraltorch.byte_corpus.request.v9",
  "design_schema": "spiraltorch.byte_corpus.request.v2",
  "optimizer": {
    "kind": "adamw_zero_rate_freeze.v1",
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 0.00001,
    "weight_decay": 0.01,
    "geometry_weight_decay": 0.0
  }
}
```

The excerpt omits the existing required config, rate, documents, windows and
cases. `design_schema` is mandatory and cannot recursively select v9 or v10.
`{"kind":"stateless_sgd.v1"}` supplies an explicit SGD control. Legacy v1-v8
requests retain their exact SGD meaning and reject optimizer/design metadata;
there is no inferred optimizer or silent history conversion.

```bash
python tools/byte_corpus_study.py bind-optimizer prepared-request.json \
  examples/configs/byte_corpus/adamw.json adamw-request.json
python tools/byte_corpus_study.py bind-optimizer prepared-request.json \
  examples/configs/byte_corpus/sgd.json sgd-request.json
```

This utility only assembles a request. Rust preflight remains authoritative.
The example epsilon is explicit, not an API default or a tuned quality result.
All optimizer numbers are normalized to float32; lossy nonzero-to-zero or
subnormal controls reject instead of changing the policy silently.

All non-`geometry.*` tensors use `weight_decay`, including biases and norm gains.
All `geometry.*` tensors use `geometry_weight_decay`. There are no implicit
decay exclusions. Existing geometry/router/query freeze policies override the
rate to zero and freeze **weights, both moments and the accepted tensor clock**.
Contextual projection biases remain frozen at zero. The independent PyTorch
reference uses `grad=None` for these tensors, not a present gradient with
`lr=0`.

This is [resident raw-coordinate AdamW](resident_adamw.md), not Riemannian
momentum transport. Geometric forward/VJP paths still supply the geometric
mechanism. No schedule, AMSGrad or gradient accumulation is implied. Clipping
is opt-in through the v10 recipe below, never inferred by v9.

## Request-Bound Clipping

After preparing the design and binding its optimizer, explicitly choose either
`{"kind":"disabled.v1"}` or `{"kind":"active_global_l2.v1","max_norm":1.0}`:

```bash
python tools/byte_corpus_study.py bind-gradient-clip adamw-request.json \
  examples/configs/byte_corpus/clip_active_global_l2.json clipped-request.json
python tools/byte_corpus_study.py bind-gradient-clip adamw-request.json \
  examples/configs/byte_corpus/clip_disabled.json unclipped-control.json
```

The result uses `spiraltorch.byte_corpus.request.v10` and requires all three
fields: `design_schema`, `optimizer`, `gradient_clip`. The eight geometry
designs are unchanged. Missing/null/unknown recipes reject, and older request
versions reject a clipping field rather than silently ignoring it. `max_norm`
must remain positive, normal and finite after float32 conversion. The example
limit is illustrative, not a tuned default.

Rust clips the joint norm of tensors whose effective learning rate is nonzero,
before SGD or AdamW moments. Frozen geometry/query tensors are excluded from
that budget but still checked for nonfinite gradients and inherited failures.
Weights, moments and clocks of frozen tensors remain unchanged. This uses the
[resident clipping contract](resident_byte_decoder.md#resident-active-gradient-clipping),
including its small-norm/near-one no-ops; it is not PyTorch's epsilon-adjusted
`clip_grad_norm_` formula or Riemannian momentum transport.

V10 report/checkpoint envelopes bind the exact request bytes, including the
clipping policy. Changing the limit or disabling clipping rejects resume.
Legacy v1-v9 retain unclipped execution; v10 `disabled.v1` takes that same path.
Model-only checkpoints still do not encode a caller's next-step policy: use the
complete study request/checkpoint pair for reproducible study continuation.

## Train, Resume, Serve

```python
import json
from pathlib import Path
import spiraltorch as st

study = st.nn.ByteCorpusStudy.from_json(Path("adamw-request.json").read_bytes())
print(study.design_schema, json.loads(study.optimizer_recipe_json))
print(json.loads(study.gradient_clip_recipe_json))  # disabled.v1 for legacy requests
study.validate_segment()  # CPU-only preflight, including expanded checkpoint size
first = study.advance_wgpu(3)
saved = first.checkpoint.to_json()
restored = study.checkpoint_from_json(saved)
continued = study.advance_wgpu(resume=restored)
case_name = json.loads(continued.report_json)["cases"][0]["name"]
artifact = study.case_artifact(
    continued.checkpoint, case_name, policy="fixed_window_frozen.v1"
)
inference = artifact.restore_wgpu()
```

Parsing, metadata inspection and checkpoint validation work in a CPU-only wheel.
Execution there raises `NotImplementedError`; it is not a hidden CPU fallback.
Rust exposes the same `design_schema()`, `optimizer_recipe_json()`,
`gradient_clip_recipe_json()`, `advance`
and checkpoint/artifact operations.

The browser example exports `inspect_resident_byte_learning(input)` without
requesting a GPU, plus the existing `advance_resident_byte_learning` and
artifact functions. Its learning/resume pages accept v9/v10. The packaged WASM
client exposes the readonly `gradientClipRecipeJson` getter. Keep request and
checkpoint/model JSON opaque through JavaScript transport: reconstruction can
lose signed-zero bits.

V9/v10 report/checkpoint envelopes retain the geometry design, exact request-byte
identity and all case histories. AdamW model checkpoints use model envelope v6,
preserving their topology-specific v1-v5 `model_schema`. Resume validates the
optimizer configuration and requires each active clock to equal the study
cursor, each frozen clock to remain zero, and every frozen parameter to retain
its original bits. Changing decay, data or any other request bytes invalidates
the original study checkpoint.

Inference artifacts retain complete history for lineage and explicit training
restoration. Their immutable serving owner uploads **weights only**, not unused
GPU moment buffers. Calling `artifact.checkpoint().restore_wgpu()` explicitly
creates a separate training owner with the complete optimizer state.
Hash binding and valid resume state do not attest that training really occurred,
and creating an artifact does not promote it through a quality gate.

For model choice across retained update boundaries, use the
[shared saved-checkpoint selector](byte_corpus_checkpoint_selection.md), not the
minimum of a loss history whose corresponding weights were never saved.

## Independent Comparison

```bash
python tools/byte_corpus_study.py reference adamw-request.json torch.json
target/release/examples/resident_byte_learning adamw-request.json > native.json
python tools/byte_corpus_study.py compare \
  adamw-request.json torch.json native.json comparison.json
```

The oracle uses real CPU PyTorch AdamW (`foreach=False`, `fused=False`), not a
copy of Rust optimizer arithmetic. Verification checks ordinary/geometry
parameters, held-out losses, geometric learning deltas, both moments and
accepted/frozen clocks. Every moment tensor also requires relative L2 error at
most 0.002, without an absolute floor that could hide erased tiny history.
At the final cursor, paused and no-op-resumed reports must match exactly.
V9/v10 retain the selected design's original controls and
contrasts, rather than treating all new requests as the most complex geometry.
The resume verifier also binds saved optimizer bits to the paused report.
For v10, the independent Torch reference computes a float64 norm from its own
float32 VJPs and scales before the optimizer; it does not copy Rust's reduction
or factor decomposition. Report comparison requires matching clipping metadata.

The rate remains part of the fixed request. Matching data, initial values and
update counts is not equal compute, tuned hyperparameters or equal model
capacity. A short comparison qualifies mechanisms for a tested recipe; it
does not establish language quality or that geometry beats ordinary training.

The [2026-10-15 qualification record](../benchmarks/results/2026-10-15-byte-study-optimizers/README.md)
keeps all three seeds, both optimizer recipes, native/browser comparisons,
resume checks and known limitations. Full corpus requests and model states
remain private; source revisions, hashes and reconstruction steps are public.

The [v10 client qualification](../benchmarks/results/2026-10-18-byte-study-clipping/README.md)
adds installed Python interruption recovery and packaged browser fresh-page
resume. It keeps the failed small-SGD minimum-delta control alongside passing
numerical conditions; choosing a tiny clip limit is not automatically a useful
learning recipe.
