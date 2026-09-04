# spiraltorch-wasm bindings

The `types/spiraltorch-wasm.d.ts` file ships hand-crafted TypeScript declarations for the
wasm-bindgen surface exposed by this crate. Copy the file next to the generated
JavaScript glue (e.g. into the `pkg/` directory produced by `wasm-pack`) and reference it
from your bundler's `types` field to enable editor completion and static checking.

## Building

Prerequisite: install `wasm-pack` and make sure it is available on `PATH`.

```bash
cargo install wasm-pack
```

Build the WebAssembly package (and copy the bundled TypeScript declarations) with:

```bash
./scripts/build_wasm_web.sh --dev
# or: ./scripts/build_wasm_web.sh --release
# optional modern-browser profile: ./scripts/build_wasm_web.sh --release --simd128
```

The portable profile remains the default. `--simd128` opts into WebAssembly SIMD
and produces a module that will not instantiate in engines without that feature;
serve a portable fallback when supporting older clients. On supported engines it
allows LLVM to vectorise Rust CPU kernels used by browser autograd and tensor ops.
The build helper sets the target-specific flag itself and strips ambient host or
target Rust flags. It explicitly sets `RUSTFLAGS` for both profiles so Cargo
configuration cannot silently change the selected target contract.

If you have `vcpkg`-style host linker flags exported in your shell (for example via
`RUSTFLAGS`), prefer the helper script above (it sanitises the environment for wasm
builds) or unset `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `CARGO_BUILD_RUSTFLAGS`,
target-specific Rust flags, `LIBRARY_PATH`, and `PKG_CONFIG_PATH` before invoking
`wasm-pack` directly.

## Examples

- COBOL dispatch console: `bindings/st-wasm/examples/cobol-console/`
- Canvas hypertrain demo (FractalCanvas + hypergradWave): `bindings/st-wasm/examples/canvas-hypertrain/`
- Mellin log grid demo (evaluateMany): `bindings/st-wasm/examples/mellin-log-grid/`

## Rust-owned runtime protocol catalog

Browser code can inspect and replay the exact cross-client protocol surface
without keeping a JavaScript registry of versions or operations:

```ts
import {
    validateZspaceRuntimeProtocolCatalogJson,
    zspaceRuntimeProtocolCatalogJson,
} from "spiraltorch-wasm";

const catalogJson = zspaceRuntimeProtocolCatalogJson();
const replay = JSON.parse(validateZspaceRuntimeProtocolCatalogJson(catalogJson));

console.log(replay.catalog_id, replay.protocols.map((protocol) => protocol.name));
```

The catalog is content-addressed by `st-core` and currently binds generation
evidence, token-periodicity analysis, stochastic Schrodinger real and complex forward/VJP,
repetition-unlikelihood planning, and the complete blinded semantic-review
lifecycle across Rust, Python, and WASM. Catalog v4 records the normal-admission
profile and guarantee for every client
surface; serialized Python/WASM surfaces also carry Rust-owned byte/node/depth
limits, while typed Rust admission has no serialized budget. Each catalogued
WASM operation is a bounded JSON admission route checked against the generated
export and bundled TypeScript declaration. Object APIs remain trusted-local
convenience transports rather than hostile-input boundaries. Trusted legacy
replay remains deliberately absent from the browser surface.

## Replayable stochastic Schrodinger dynamics

`zspaceStochasticSchrodingerComplexStepJson` preserves both quadratures and
optionally evaluates their VJP; `validateZspaceStochasticSchrodingerComplexJson`
replays the entire receipt. See the [complex dynamics guide](../../docs/zspace_complex_dynamics.md)
for the shared request and trajectory-backpropagation contract.

Browser workers and persisted runs can execute the same bounded real-time
transition and analytic VJP as direct Rust and Python callers. The catalogued
surface is JSON-only so JavaScript getters and Proxies are never a normal
protocol admission path:

```ts
import {
    validateZspaceStochasticSchrodingerForwardJson,
    validateZspaceStochasticSchrodingerVjpJson,
    zspaceStochasticSchrodingerForwardJson,
    zspaceStochasticSchrodingerVjpJson,
} from "spiraltorch-wasm";

const forward = JSON.parse(zspaceStochasticSchrodingerForwardJson(JSON.stringify({
    input: [1.0, 0.25, -0.5, 0.75],
    potential: [0.2, -0.1],
    standard_normal: [0.1, -0.3, 0.2, 0.0],
    rows: 2,
    features: 2,
    config: { time_step: 0.08, noise_scale: 0.15 },
})));
validateZspaceStochasticSchrodingerForwardJson(JSON.stringify(forward));

const vjp = JSON.parse(zspaceStochasticSchrodingerVjpJson(JSON.stringify({
    forward_request: forward.request,
    grad_output_real: [0.2, -0.4, 0.1, 0.3],
})));
validateZspaceStochasticSchrodingerVjpJson(JSON.stringify(vjp));
```

The standard-normal array and configuration are fixed stochastic witnesses. The
VJP differentiates `output_real` with respect to input and potential; its phase
is recomputed by `st-core` from the forward request and is never accepted from
the browser. Receipts certify the stated numerical transition and derivative,
not physical fidelity or training efficacy.

## Integer Row Indexing (Source Builds)

`AutogradTensor.gatherRows(ids)` accepts an Array of validated u32 integers or a
`Uint32Array`. `scatterAddRows(ids, outputRows)` adds rows into a new zero table.
Both preserve duplicate IDs and share Rust forward/VJP semantics. Unlike an ABI
`Vec<u32>` conversion, fractional/negative IDs and non-numeric Array elements
are rejected before coercion. Values already coerced by the caller's own
`Uint32Array` construction cannot be recovered or validated retroactively.
These graph operations execute on CPU; native Rust/Python Tensor utilities
separately expose strict WGPU dispatch. They do not imply a browser GPU graph.

`node bindings/st-wasm/tests/row_indexing.cjs /path/to/spiraltorch_wasm.js` executes
the generated WASM, checks boundary failures, and trains a shared embedding/output
table for 400 synthetic token-transition steps. Free returned handles normally.

## Shared reverse-mode autograd

`AutogradTensor` is a browser handle over the immutable graph implemented in
`st-tensor`; JavaScript does not rebuild gradient formulas or accumulation rules.
The v1 surface supports elementwise arithmetic, matrix multiplication, reductions,
dot products, and mean-squared error, with explicit seeds required for non-scalar
vector-Jacobian products:

```ts
import { AutogradTensor, autogradSemanticOwner } from "spiraltorch-wasm";

const x = new AutogradTensor(1, 3, new Float32Array([1, 2, -1]), true);
const loss = x.hadamard(x).add(x.scale(3)).sum();
const receipt = loss.backward();

console.log(x.gradientValues(), receipt.contract_version, autogradSemanticOwner());
```

Backward passes commit atomically, repeated passes accumulate explicitly, and
`zeroGradGraph()` clears every reachable node. Python, WASM, and direct Rust all
report `spiraltorch.autograd.v1` with semantic owner `st-tensor`. For probes that
must not touch accumulated state, use `output.vectorJacobianProduct(input, seed)`;
a disconnected input returns an all-zero gradient.

`rhs.prepackRhs()` returns a Rust-owned `AutogradPackedRhs` for repeated
`lhs.matmulPrepacked(packed)` calls. It retains the original source node and
its backward graph, even for trainable or non-leaf RHS values. Use `sourceId()`,
`rows()`, `cols()` and `requiresGrad()` to inspect its identity. Free packed
handles normally; outputs retain their own graph dependencies.

```ts
const weights = new AutogradTensor(3, 2, new Float32Array([1, 0, 0, 1, 1, 1]), false);
const packed = weights.prepackRhs();
const hidden = new AutogradTensor(1, 3, new Float32Array([1, 2, 3]), true);
const projected = hidden.matmulPrepacked(packed);
const projectedLoss = projected.sum();
projectedLoss.backward();
console.log(projected.values(), hidden.gradientValues());
projectedLoss.free(); projected.free(); hidden.free(); packed.free(); weights.free();
```

This is an immutable snapshot, not a live optimizer cache: after changing a
trainable RHS with `AutogradSgd.step()`, fetch and pack the new parameter.
Frozen projections may reuse one packed handle across training steps. The
existing Tensor prepacked Auto dispatch owns execution; browser autograd here
remains CPU, not a WebGPU-resident graph. Benchmark reuse and its one-time pack
cost with `tools/bench_wasm_vs_torch.py --prepacked` (and reverse the operation
order with `--reverse-operations` to check warmup effects).

Classification uses the same stable CPU kernels as native Rust and Python:

```ts
const logits = new AutogradTensor(2, 3, new Float32Array([2, 0, -1, 0, 2, -1]), true);
const loss = logits.crossEntropyWithLogits(new Int32Array([0, 1]), "mean", -100, 0.05);
loss.backward();
console.log(loss.values(), logits.gradientValues());
loss.free();
logits.free();
```

The arguments after `Int32Array` are optional: mean reduction, ignore index
`-100`, and no smoothing. `"none"` returns `(rows, 1)` and needs a matching
backward seed; an all-ignored mean is rejected. `rowLogSoftmax()` exposes stable
log-probabilities and their coupled VJP. These are CPU kernels compiled to WASM,
not WebGPU dispatch. Run `node bindings/st-wasm/tests/classification.cjs
/absolute/path/to/spiraltorch_wasm.js` against an actual nodejs-target build for
masking, gradient, and 180-step learning checks.

### Atomic parameter updates

`AutogradSgd` delegates plain CPU SGD to Rust and replaces all registered leaves
only when every update is valid. It does not mutate old graph handles. Fetch
`parameter(index)` again for every forward pass, then free temporary JS handles:

```ts
import { AutogradSgd, AutogradTensor } from "spiraltorch-wasm";

const optimizer = new AutogradSgd(0.1);
const initial = new AutogradTensor(1, 2, new Float32Array([2, -3]), true);
optimizer.addParameter(initial); // Borrowed, not consumed.
initial.free();
for (let step = 0; step < 100; step++) {
  const parameter = optimizer.parameter(0);
  const squared = parameter.hadamard(parameter);
  const loss = squared.mean();
  try {
    loss.backward();
    optimizer.step();
  } finally {
    loss.free();
    squared.free();
    parameter.free();
  }
}
optimizer.free();
```

Duplicate/non-leaf parameters, missing gradients, invalid learning rates and
overflow are errors, not partial updates. There is no momentum, clipping,
implicit averaging, or WebGPU optimizer dispatch. Run
`node bindings/st-wasm/tests/autograd_sgd.cjs /absolute/path/to/spiraltorch_wasm.js`
for the failure/ownership checks; `classification.cjs` also uses the optimizer.

### Numeric argument boundaries

`AutogradTensor` rows/columns, `AutogradSgd.parameter(index)`, and all scalar
indices and workload dimensions in `WasmTuner`/`baseChoice` require primitive
JavaScript numbers that are finite integers in `[0, 2**32 - 1]`. Strings,
booleans, boxed numbers, fractions, NaN, and overflowing numbers are rejected
before Rust receives an integer. Invalid tuner mutations leave its records
unchanged; valid out-of-bounds indices still return `undefined`/`false`.

`crossEntropyWithLogits` applies the same non-coercing check to `ignore_index`
with the signed i32 range. Omitted, `undefined`, or `null` keeps the `-100`
default. Labels remain an `Int32Array`; values truncated while constructing
that array cannot be recovered by the binding. TypeScript signatures remain
`number` (and optional `number | null` for the sentinel).

The tuner fallback still runs the shared Rust heuristics and records ecosystem
metrics. On WASM, these observations use the JavaScript host's `Date.now()`;
native Rust keeps its system wall clock. This is not a monotonic timing source.

Run `node bindings/st-wasm/tests/numeric_boundaries.cjs
/absolute/path/to/spiraltorch_wasm.js` against a nodejs-target build. CI runs
this regression suite and the SGD, nonlinear autograd, and classification
learning loops in the generated WASM, rather than checking compilation alone.

## FFT numerics

The low-level `fft_forward`, `fft_inverse`, and their `_in_place` variants share
the corrected `st-frac::fft` mixed radix-2/4 CPU kernel. Buffers are interleaved
real/imaginary `Float32Array`s with a positive power-of-two complex length;
length one is the identity. Forward uses a negative exponent and inverse applies
`1/n` normalization. Invalid lengths do not mutate in-place buffers. These
functions compute in WASM, independently of emitted WebGPU FFT plans.

`tests/fft_numerics.cjs` checks dense complex signals against an independent DFT,
exact four-point bin order, inverse transforms, in-place roundtrips, and circular
filtering. These checks replace reliance on origin-impulse fixtures alone.

## Shared Topos control and runtime routing

Browser clients can derive a topology control signal and project a runtime profile through
the same `st-tensor::pure::topos` contract used by native Rust and Python. WASM only
converts the object/JSON boundary; it does not maintain separate pressure, hint, or route
formulas:

For live device discovery, call `runtimeDeviceProbeObserveObject` (or its JSON
variant) with only `requested_backend`, optional `caps_overrides`, and workload
hints. Rust selects the effective backend, constructs any MPS surrogate overlay,
and commits both runtime observations. `runtimeDeviceProbeObject` remains the
resolved v1 evaluation surface for callers that already hold the full request;
`runtimeDeviceProbeValidate*` replays persisted contracts without probing mutable
hardware. JavaScript should not rebuild the live resolution policy.

```ts
import {
    toposControlSignalObject,
    toposOptimizerSnapshotObject,
    toposRuntimeRouteObject,
    toposZSpaceProjectionObject,
} from "spiraltorch-wasm";

const toposInput = {
    curvature: -1.0,
    porosity: 0.25,
    max_depth: 64,
    max_volume: 512,
    observed_depth: 12,
    visited_volume: 96,
};

const signal = toposControlSignalObject(toposInput);

const optimizerSnapshot = toposOptimizerSnapshotObject({
    signal: toposInput,
    sequence: 1,
    hyper_learning_rate: 0.04,
    real_learning_rate: 0.02,
    options: { training_gain: 0.75 },
});

const route = toposRuntimeRouteObject({
    closure_risk: 0.2,
    exploration_budget: 0.6,
    inference_temperature: 1.2,
    inference_top_p: 0.95,
});

const projection = toposZSpaceProjectionObject(toposInput, 8);

console.log(
    signal.closure_pressure,
    optimizerSnapshot.optimizer_application.hyper_learning_rate,
    route.mode,
    projection.gradient,
);
```

`toposControlSignalJson`, `toposOptimizerSnapshotJson`, `toposRuntimeRouteJson`, and
`toposZSpaceProjectionJson` expose the same contracts for storage and worker-message flows.
Optimizer snapshots bind a JavaScript-safe sequence, the complete Rust control bundle, and the
v3 learning-rate plus gradient-state application prescribed by that bundle. The payload includes
the same ten-axis bias basis, RMS-relative bias and clipping rules, and clipped-gradient momentum
transition consumed by native Amega tapes; browser code does not rebuild those equations. Topos
projection v2 names its six-axis basis, exact resize rule, and ordered channels; wider client
vectors are explicitly zero-filled by Rust. Results carry
`contract_version`, `semantic_owner`, and `semantic_backend` so Python, browser, and direct Rust
runs can be audited against one semantic core.

Latent posterior decoding and partial projection use the same boundary. The
browser passes state, partial observations, and telemetry to
`st-core::inference::zspace_posterior`; Rust alone owns the DFT-derived
fractional energy, gradient normalization, aliases, barycentric coordinates,
residual/confidence update, and telemetry adjustment. Contract v2 uses a
one-sided Parseval-normalized spectrum, reports its reconstruction error and
centroid, computes residual RMS only over observed metrics, and applies
telemetry as a confidence reliability no greater than one. Browser-supplied
gradients require an explicit basis and remain separate controls; they never
replace the Rust latent finite-difference gradient:

```ts
import {
    zspacePosteriorDecodeObject,
    zspacePosteriorProjectObject,
} from "spiraltorch-wasm";

const prior = zspacePosteriorDecodeObject({
    z_state: [0.12, -0.03, 0.48, -0.2],
    alpha: 0.35,
});
const projection = zspacePosteriorProjectObject({
    z_state: prior.z_state,
    partial: { speed: 0.3, mem: -0.2, gradient: [0.2, -0.1] },
    gradient_basis: "example.browser.control.v1",
    telemetry: [{ psi: { energy: 2.0, focus: 0.4 } }],
});

console.log(
    projection.residual,
    projection.gradient_basis,
    projection.control_gradient?.basis,
);
```

The corresponding `zspacePosteriorDecodeJson` and
`zspacePosteriorProjectJson` functions expose identical worker/persistence
contracts. WASM adds only `execution_client: "wasm"`; it has no JavaScript
posterior fallback. An untagged external gradient fails closed in Rust.

Coherence diagnostics use a peer contract as well. Browser code may carry
diagnostics produced elsewhere, but the projection formula and summaries are
owned by `st-core::inference::zspace_coherence`:

```ts
import { zspaceCoherenceProjectObject } from "spiraltorch-wasm";

const projection = zspaceCoherenceProjectObject({
    diagnostics: {
        mean_coherence: 1 / 3,
        coherence_entropy: 1.0296530140645737,
        energy_ratio: 0.7,
        z_bias: -0.12,
        fractional_order: 0.4,
        normalized_weights: [0.5, 0.3, 0.2],
    },
    coherence: [0.6, 0.3, 0.1],
    classification_policy: {
        background_energy_ratio_max: 1e-5,
        cascade_energy_ratio_min: 0.7,
    },
});

console.log(
    projection.partial.speed,
    projection.derived.normalized_entropy,
    projection.control?.spectral_pressure,
    projection.classification?.label,
    projection.classification?.reason,
    projection.semantic_owner,
);
```

`zspaceCoherenceProjectJson` exposes the same worker/persistence boundary.
Rust recomputes dimension-normalized entropy and normalized HHI concentration;
projection v2 also rejects entropy, support counts, dominant channels, or raw
means that contradict the supplied vectors. Browser code does not repair those
cross-field inconsistencies.
It can also persist and later revalidate the exact simplex witness without
implementing any distribution mathematics in JavaScript:

```ts
import {
    zspaceCoherenceDistributionValidateObject,
    zspaceCoherenceDistributionWitnessObject,
} from "spiraltorch-wasm";

const witness = zspaceCoherenceDistributionWitnessObject([0.5, 0.3, 0.2]);
const summary = zspaceCoherenceDistributionValidateObject(witness);
console.log(witness.contract_version, summary.concentration);
```

Its versioned control payload provides the same spectral radius, entropy, and
pressure consumed by native training, while its classification policy emits the
structural label, reason, formula, and thresholds. WASM adds only
`execution_client: "wasm"` and has no JavaScript coherence heuristic,
classifier, control formula, or normalization fallback.

Stateful temperature control follows the same contract. The browser carries
controller state between calls, while Rust alone validates probability mass and
computes entropy, Z-feedback, scale-memory, and gradient adjustments:

```ts
import { zspaceTemperatureControlObject } from "spiraltorch-wasm";

const transition = zspaceTemperatureControlObject({
    probabilities: [0.6, 0.4],
    config: {
        target_entropy: 0.8,
        eta: 0.2,
        min_temperature: 0.3,
        max_temperature: 2.0,
    },
    state: { temperature: 1.0 },
});

console.log(transition.entropy, transition.next_state.temperature);
```

`zspaceTemperatureControlJson` exposes the identical versioned payload for
worker-message and persistence paths. Neither entry point contains a browser
fallback for the transition formula.

Token periodicity uses the same bounded suffix kernel consumed by Rust training
and generation evidence. JavaScript transports token IDs and can persist the
content-addressed report, but it does not reimplement period search or
tie-breaking:

```ts
import {
    validateZspacePeriodicityObject,
    zspacePeriodicityObject,
} from "spiraltorch-wasm";

const report = zspacePeriodicityObject({
    token_ids: [9, 1, 2, 1, 2, 1],
    appended_token_id: 2,
    config: { maximum_period: 16, minimum_repetitions: 3 },
});
const replay = validateZspacePeriodicityObject(report);

console.log(replay.periodic_suffix?.period, replay.analysis_id);
```

`zspacePeriodicityJson` and `validateZspacePeriodicityJson` expose the same
contract for workers and storage. These JSON routes bound bytes before making a
Rust string, then reject duplicate keys and enforce node/depth limits before
serde materialization. Rust rejects unknown fields, unsafe token IDs, excessive
comparison work, and any report tampering. The Object helpers above are
trusted-local conveniences, not hostile-input boundaries. A positive suffix is
a structural token observation, not a semantic-quality or efficacy claim.

Held-out generation evidence is exposed through the same boundary. Browser
clients provide only content-addressed run identities and continuation token IDs;
Rust canonicalizes sample order and owns all n-gram, adjacent-repetition,
periodic-suffix, aggregate, and loop-score semantics:

```ts
import {
    validateZspaceGenerationEvidenceObject,
    zspaceGenerationEvidenceObject,
} from "spiraltorch-wasm";

const sha = (digit: string) => `sha256:${digit.repeat(64)}`;
const evidence = zspaceGenerationEvidenceObject({
    protocol_id: sha("a"),
    runtime_identity_id: sha("b"),
    model_artifact_id: sha("c"),
    prompt_set_id: sha("d"),
    decoding_config_id: sha("e"),
    samples: [{
        prompt_id: sha("f"),
        seed: 17,
        continuation_token_ids: [9, 1, 2, 1, 2, 1, 2],
    }],
});
const replay = validateZspaceGenerationEvidenceObject(evidence);

console.log(replay.aggregate.sample_mean_loop_score, replay.evidence_id);
```

The JSON variants carry the identical artifact through Web Workers or durable
storage. Validation replays the complete request in Rust and rejects a changed
metric, identity, ordering, or evidence boundary. As in Python and direct Rust,
these are structural token observations rather than semantic-quality claims.

Repetition-unlikelihood planning is available to browser training clients without
porting candidate semantics into TypeScript. Rust owns prior-continuation matching,
model-top-k history filtering, periodic-suffix gating, candidate ordering, and the
v3 plan identity:

```ts
import {
    validateZspaceRepetitionUnlikelihoodPlanObject,
    zspaceRepetitionUnlikelihoodPlanObject,
} from "spiraltorch-wasm";

const plan = zspaceRepetitionUnlikelihoodPlanObject({
    config: {
        strength: 0.1,
        candidate_source: { kind: "prior_continuation", ngram_order: 3 },
        context_window: 16,
        max_candidates_per_position: 8,
    },
    sequences: [{
        token_ids: [1, 2, 3, 1, 2, 4],
        token_mask: [true, true, true, true, true, true],
        label_mask: [true, true, true, true, true, true],
    }],
});
const replay = validateZspaceRepetitionUnlikelihoodPlanObject(plan);

console.log(replay.positions[0]?.candidates, replay.plan_id);
```

The JSON variants expose the same plan for worker messages and persistence. A
shared 64,000,000-unit Rust preflight bounds prior-history scans and the more
expensive proposal-by-period scans during both planning and browser validation.
An independent 32 MiB conservative materialization preflight accounts for the
request arrays plus every potentially emitted position and candidate before Rust
builds the plan or converts it to a JavaScript value. Existing v3 report fields
and plan IDs remain unchanged. Browser JSON/object ingress is independently
bounded to 64 MiB, 8,000,000 JSON nodes, and depth 32 before a JavaScript object
is duplicated into Rust. A historical v3 artifact above either newer plan limit
can be replayed only through Rust's explicitly trusted
`validate_zspace_repetition_unlikelihood_value_trusted_legacy_replay` or the
matching Python `validate_zspace_repetition_unlikelihood_plan_trusted_legacy_replay`;
WASM deliberately exposes no unbounded validator for attacker-controlled browser
input.

Blinded semantic review is also a complete Rust-owned browser lifecycle rather
than a report-only adapter. Rust seals packet fields and identities, commits the
candidate-to-arm map before review, validates resumable drafts, and performs
unblinding and arm/seed aggregation:

```ts
import {
    sealZspaceSemanticReviewPacketObject,
    summarizeZspaceSemanticReviewDraftObject,
    zspaceSemanticReviewMapIdObject,
} from "spiraltorch-wasm";

const mapId = zspaceSemanticReviewMapIdObject({ entries: sealedMapEntries });
const packetReceipt = sealZspaceSemanticReviewPacketObject({
    protocol_id: protocolId,
    prompt_set_id: promptSetId,
    blinding_key_sha256: blindingKeyDigest,
    blinding_map_id: mapId,
    instructions: "Score every candidate while blind.",
    rubric,
    groups,
});
const draftReceipt = summarizeZspaceSemanticReviewDraftObject({
    packet: packetReceipt.packet,
    draft,
});
```

JSON and object exports cover packet sealing and validation, packet-receipt
replay, draft summarization and replay, unblinding, and unblind-report replay.
Every object is preflighted before serde materialization with a 64 MiB,
1,000,000-node, depth-32 browser ingress bound. Properties are read once into a
bounded plain-data snapshot, so a getter or proxy cannot swap in an unchecked
second value during Rust conversion. JSON-string variants apply the same node
and depth limits in a duplicate-key-rejecting streaming Rust deserializer rather
than constructing an unbounded intermediate tree. The Rust contract additionally
limits packets to 10,000 groups and 32 MiB of aggregate JSON-encoded text, map entries
to 10,000, and arm names to 128 bytes. Existing packet and map identity rules
remain unchanged. Rust and Python provide explicitly named trusted-legacy replay
functions for locally held v1 evidence above newer admission budgets; WASM
deliberately exposes no unbounded replay route. Validation proves structural commitments and deterministic
aggregation; it does not prove reviewer blindness or model superiority.

Z-space meta-optimization follows the same state-carrying client model. The
browser stores the returned checkpoint, but Rust owns restore coercion,
observation normalisation, the FFT-derived fractional Sobolev gradient, bounded Topos
controls, and Adam:

```ts
import {
    zspaceMetaOptimizerInitObject,
    zspaceMetaOptimizerParameterControlObject,
    zspaceMetaOptimizerStepObject,
} from "spiraltorch-wasm";

const checkpoint = zspaceMetaOptimizerInitObject({
    dimension: 4,
    fractional_order: 0.35,
});
const transition = zspaceMetaOptimizerStepObject({
    config: checkpoint.config,
    state: checkpoint.state,
    observation: {
        speed: 0.2,
        memory: 0.1,
        stability: 0.7,
        gradient: [0.05, -0.02, 0.01, 0.0],
    },
});
const parameterControl = zspaceMetaOptimizerParameterControlObject(transition);

console.log(
    transition.state_after.z,
    parameterControl.absolute_learning_rate_scale,
    parameterControl.source_step,
    parameterControl.semantic_owner,
);
```

`zspaceMetaOptimizerInitJson`, `zspaceMetaOptimizerRestoreJson`, and
`zspaceMetaOptimizerStepJson` provide the identical transition contract for
workers and persistence. `zspaceMetaOptimizerParameterControlJson` verifies the
complete report and projects the same narrow parameter-control receipt. WASM
adds only `execution_client: "wasm"`; it does not carry a browser Adam, FFT
fallback, or model-parameter optimizer. The shared Rust contract bounds step
counters at JavaScript's largest exactly representable integer.

Partial-gradient construction is a peer contract too. Browser code supplies
named canonical metrics; Rust assigns their periodic optimizer coordinates and
returns the basis identity consumed by partial fusion:

```ts
import { zspaceMetricGradientProjectionObject } from "spiraltorch-wasm";

const projected = zspaceMetricGradientProjectionObject({
    metrics: { speed: 0.2, memory: 0.1, stability: 0.7, frac: 0.3, drs: -0.1 },
    dimension: 6,
});

console.log(projected.gradient, projected.basis);
```

Partial fusion v3 rejects tagged gradients from different bases, including
equal-length vectors. WASM therefore cannot silently fuse browser-local feature
orders with a Rust or Python gradient signal. Set
`metric_gradient_dimension` on a partial-fusion request when heterogeneous
clients should contribute named metrics instead: Rust fuses those scalars,
projects one canonical gradient, and reports which positional gradients were
replaced.

Concept diffusion is also a peer contract rather than a browser-side
approximation. Rust validates the labelled simplex and symmetric graph, applies
observation/Z-bias controls, selects CFL-safe heat-flow substeps, and audits
entropy plus Dirichlet energy:

```ts
import { zspaceConceptDiffusionObject } from "spiraltorch-wasm";

const transition = zspaceConceptDiffusionObject({
    tags: ["left", "right"],
    state: [1.0, 0.0],
    affinity: [[0.0, 1.0], [1.0, 0.0]],
    config: { timestep: 0.25 },
});

console.log(transition.next_state); // [0.75, 0.25]
```

`zspaceConceptDiffusionJson` returns the identical versioned payload for worker
and persistence boundaries. JavaScript never reconstructs the heat equation.

Imaginary-time Schrodinger evolution uses the same peer-client boundary. Rust
alone constructs the weighted graph Laplacian, shifts the scalar-potential
gauge, chooses a spectral-safe positive substep, evolves the amplitude in the
log domain, and audits the Rayleigh energy:

```ts
import { zspaceImaginaryTimeSchrodingerObject } from "spiraltorch-wasm";

const groundState = zspaceImaginaryTimeSchrodingerObject({
    tags: ["left", "right"],
    potential: [0.0, 2.0],
    edges: [{ left: 0, right: 1, weight: 1.0 }],
    config: { imaginary_time: 1.0 },
});

console.log(groundState.probability, groundState.effects.rayleigh_energy_drop);
```

`zspaceImaginaryTimeSchrodingerJson` exposes the exact same payload for workers
and persistence. Browser code never owns a second Hamiltonian or normalization
heuristic.

API LLM trace comparison follows the same boundary. Browser code can submit
the same measured rows used by Python, while
`st-core::runtime::api_llm_route_policy` alone owns normalization, health and
cost treatment, profile scores, deterministic winners, ranking, and near-best
membership:

```ts
import { apiLlmRoutePolicyEvaluateObject } from "spiraltorch-wasm";

const comparison = apiLlmRoutePolicyEvaluateObject({
    rows: [
        { label: "fast", count: 4, latency_ms_mean: 80, total_tokens: 32 },
        { label: "grounded", count: 4, text_quality_score: 0.9 },
    ],
    near_best_tolerance: 0.05,
});

console.log(comparison.profiles, comparison.winners, comparison.near_best);
```

Missing costs stay unknown instead of becoming zero-cost wins, sparse evidence
shrinks toward the neutral prior, aggregate token totals are scored per
observation, and zero-count rows remain inactive.
`apiLlmRoutePolicyEvaluateJson` exposes the identical contract for workers and
persistence.

Topos route-policy scoring follows the same boundary. Browser code supplies measured route rows,
while `st-core::runtime::topos_route_policy` alone owns profile normalization, scoring,
tie-breaking, reward projection, and selected-route resolution. The v2 contract treats
missing metrics as a neutral prior rather than a free latency/token win, shrinks scores by
the observed sample count, excludes zero-observation routes from rewards, and carries the
source row plus score evidence so resolution can reject drift. Stored v1 rewards must be
rebuilt from their original route rows because they do not carry this v2 witness; legacy
rows without a positive observation `count` must be remeasured:

```ts
import {
    toposRoutePolicyEvaluateObject,
    toposRoutePolicyResolveObject,
    toposRoutePolicyRewardsObject,
} from "spiraltorch-wasm";

const evaluation = toposRoutePolicyEvaluateObject({
    rows: [
        { label: "guarded", count: 3, trace_route_score: 0.8 },
        { label: "exploratory", count: 3, trace_route_score: 0.6 },
    ],
});
const projected = toposRoutePolicyRewardsObject({
    rows: evaluation.rows,
    profile: "grounded",
});
const selected = toposRoutePolicyResolveObject({
    rewards: projected.rewards,
    selected_label: evaluation.profiles.grounded.label,
});

console.log(selected.selected_label, selected.selected_reward);
```

The object and JSON entry points are transport adapters only. Their payloads retain the
Rust `semantic_owner` and add `execution_client: "wasm"` for audit provenance; no browser
fallback reimplements the route-policy formulas.

Runtime-device readiness follows that boundary too. Browser code may collect WebGPU or host
observations, but direct readiness, surrogate readiness, fallback identity, and requirement
gates are evaluated only by `st-core::backend::runtime_route`. When the observation is a
committed SpiralTorch probe, pass the whole payload back to Rust rather than copying fields:

```ts
import {
    runtimeDeviceProbeObserveObject,
    runtimeDeviceRouteFromProbesObject,
} from "spiraltorch-wasm";

const probe = runtimeDeviceProbeObserveObject({ requested_backend: "mps" });
const runtime = runtimeDeviceRouteFromProbesObject({
    probes: [probe],
    required_ready_backends: ["mps"],
});

console.log(runtime.routes[0].native_ready, runtime.selection?.effective_backend);
```

`runtimeDeviceRouteObject` remains the explicit compatibility ingress when the browser owns an
external evidence source rather than a committed probe:

```ts
import { runtimeDeviceRouteObject } from "spiraltorch-wasm";

const runtime = runtimeDeviceRouteObject({
    reports: [{
        requested_backend: "mps",
        effective_backend: "wgpu",
        runtime_ready: true,
        requested_backend_runtime_ready: false,
        effective_backend_runtime_ready: true,
        runtime_status: "kernel_wired",
    }],
    requested_backends: ["mps"],
    required_ready_backends: ["mps"],
});

console.log(runtime.routes[0].native_ready, runtime.selection?.effective_backend);
```

This reports native MPS honestly as unavailable while selecting its ready WGPU surrogate.
The browser binding adds only `execution_client: "wasm"`; it owns no readiness precedence
or fallback heuristic. Contract v5 separates probe success, native availability, and route
readiness; preserves absent evidence as `unknown`; commits ordered selection candidates and
the first ready route; binds the request and output with SHA-256; and rejects conflicting
claims about one effective backend. `runtimeDeviceRouteValidateObject` and
`runtimeDeviceRouteValidateAgainstObject` call the same Rust self-validation and replay path;
unknown routes remain fail-closed for execution.

Trainer optimizer preflight follows the same client boundary. Browsers can
validate a proposed curvature, learning-rate, realgrad, and gradient-clip
configuration before handing it to a native training runtime, but they do not
rebuild those admissibility rules:

```ts
import { trainerOptimizerConfigObject } from "spiraltorch-wasm";

const optimizer = trainerOptimizerConfigObject({
    curvature: -1.0,
    hyper_learning_rate: 0.02,
    fallback_learning_rate: 0.01,
    real_learning_rate: 0.005,
    grad_clip_max_norm: 1.0,
});

console.log(optimizer.contract_version, optimizer.semantic_owner);
```

`trainerOptimizerConfigJson` exposes the same fail-closed contract for worker
messages. Both entry points call `st-core::runtime::trainer_optimizer` and add
only `execution_client: "wasm"`; WASM does not own or execute the `st-nn`
parameter update loop.

WASM can also preflight a checkpoint produced by Rust or Python orchestration:

```ts
import { trainerOptimizerCheckpointObject } from "spiraltorch-wasm";

const receipt = trainerOptimizerCheckpointObject(checkpoint);
console.log(receipt.parameter_count, receipt.deterministic_resume_ready);
console.log(receipt.external_state_required);
```

`trainerOptimizerCheckpointJson` provides the equivalent worker-message path.
Both functions reject unknown fields, contract-version changes, non-finite
state, invalid tape shapes, and unsorted external-state requirements through
the Rust validator. They never restore model parameters in the browser.

Optimizer and external payloads can be transported as one integrity-bound
runtime bundle:

```ts
import { trainerRuntimeCheckpointBundleObject } from "spiraltorch-wasm";

const bundle = trainerRuntimeCheckpointBundleObject(runtimeCheckpointBundle);
console.log(bundle.optimizer_sha256, bundle.external_sha256);
console.log(bundle.unresolved_components, bundle.deterministic_resume_ready);
```

`trainerRuntimeCheckpointBundleJson` provides the worker-message equivalent.
Both entry points call `st-core::runtime::trainer_checkpoint`; WASM adds only
`execution_client: "wasm"`. The browser verifies metadata, child SHA-256
digests, and exact component alignment, but cannot claim that a native
distributed provider has been reattached or execute the `ModuleTrainer`
restore transaction.

External runtime state uses a separate preflight surface. The browser can
inspect exact component coverage and learn which concrete resources a native
orchestrator must reattach, but it cannot mark those resources as restored:

```ts
import { trainerExternalStateCheckpointObject } from "spiraltorch-wasm";

const external = trainerExternalStateCheckpointObject(externalCheckpoint);
console.log(external.unresolved_components);
console.log(external.reattach_required_components);
console.log(external.deterministic_resume_ready);
```

`trainerExternalStateCheckpointJson` exposes the same Rust validator to worker
messages. WASM adds only `execution_client: "wasm"`; a known distributed
provider remains not-ready until Python or another native orchestrator attaches
and Rust verifies the real resource. Counter, timestamp, rank, and world-size
fields are capped at JavaScript's largest exactly representable integer so the
Object and JSON entry points cannot disagree about checkpoint identity.
Contract v4 also carries trainer-pending and bridge-latest coherence evidence
plus subscription topology. The browser does not derive a label or control
metric: Rust validates the distribution witness, support, and dominant channel,
then reconstructs those semantics through the canonical coherence contract.
The same payload can preflight a GNN roundtable bridge snapshot: retained signal
history, latest signal, history limit, and trainer-last observation are checked
in Rust. WASM never derives graph multipliers or drift bias.

Rank planning uses the same boundary. `rankPlanObject` and `rankPlanJson` send shape and
capability observations to `st-core::ops::rank_entry`; Rust validates `rows`, `cols`, `k`,
device limits, runtime surrogate routing, and the complete unified heuristic choice:

```ts
import { rankPlanObject } from "spiraltorch-wasm";

const plan = rankPlanObject({
    kind: "midk",
    rows: 4,
    cols: 128,
    k: 8,
    backend: "wgpu",
    lane_width: 32,
    subgroup: true,
});

console.log(plan.choice.compaction_tile, plan.device_caps, plan.semantic_owner);
```

The browser owns no fallback table or capability clamp. The returned
`spiraltorch.rank_plan.v2` payload is the same audit snapshot available from Python via
`RankPlan.contract()`, with only client provenance added by each binding. A browser
worker can instead pass a previously committed `runtime_execution_plan`; Rust validates
its commitment and planning provenance, and the rank result carries the same parent
commitment:

```ts
const replayedRank = rankPlanObject({
    kind: "topk",
    rows: 4,
    cols: 128,
    k: 8,
    runtime_execution_plan: persistedRuntimePlan,
});

console.log(replayedRank.runtime_execution_plan_output_sha256);
```

`runtime_execution_plan` is mutually exclusive with backend, capability, and execution
overrides. Browser-side code cannot reinterpret a blocked plan. Rank replay is
planning-only, so it does not require Faer or WGPU kernels in the browser and does not
claim that the browser can execute the committed tensor policy; a session or trainer
must still materialize that policy against its local runtime before executing kernels.
The parent plan also records Rust's component-resolution contract. Concrete workload
preflight must carry a committed
`spiraltorch.runtime_component_capability_observation.v4` payload. Its
`ready_proof` records either a static host contract or Rust's accelerator
dispatch/readback sentinel; naked browser capability arrays are not execution-plan
input. A session-level `deferred` plan may
postpone unobserved shape checks without claiming those components are native. Strict
fallback, runtime readiness, and tensor-utility threshold behavior remain Rust
decisions in both cases, so JavaScript only transports the same contract used by
Python.

Stateful rank adaptation keeps that boundary intact. JavaScript supplies bounded
SpiralK candidates and measured elapsed time; Rust owns candidate compilation,
Black Cat selection, correctness-gated reward, and the pending-selection slot:

```ts
import { RankAdaptationSession } from "spiraltorch-wasm";

const session = new RankAdaptationSession({
  rank_plan: { kind: "topk", rows: 2, cols: 256, k: 8, backend: "wgpu" },
  scripts: ["u2: false;", "u2: true;"],
  policy: "ucb",
  seed: 17,
});
const selection = session.choose();
const result = await executeAndCheck(selection.plan);
const observation = session.observe(
  selection.selection_id,
  result.elapsedMs,
  result.correct,
);
console.log(selection.decision.arms, observation.credited, session.snapshot());
session.free();
```

UCB gives every candidate one selection attempt before posterior comparison.
`selection_attempts` therefore advances after a failed or abandoned run while
`observations` remains unchanged. Correctness failure quarantines that arm;
abandonment does not. These decision fields use Black Cat bandit witness contract
v3. Candidate receipts expose the effective execution signature, and Thompson
RNG seeds are decimal strings so values above JavaScript's safe-integer range
remain exact. Mutating calls commit the cloned Rust session only after their
receipt is serializable, so a transport error does not leave hidden state behind.
Neither JavaScript nor the WASM wrapper derives the reward or silently retries a
candidate.

## High-level Canvas utilities

`types/canvas-view.ts` implements an opinionated orchestration layer around the raw
`FractalCanvas` wasm bindings. It manages requestAnimationFrame loops, pointer-based
navigation (pan + zoom), palette overrides, and gradient statistic sampling. The helper
exposes getters and setters so UI layers can tune stats sampling, pointer navigation, and
device pixel ratios at runtime.

Add the file to your bundler entrypoint (for example by copying it next to the generated
JavaScript glue or importing it from your TypeScript project) and instantiate the helper:

```ts
import init, { FractalCanvas } from "spiraltorch-wasm";
import { SpiralCanvasView } from "./canvas-view";

await init();

const canvas = document.querySelector("#fractal") as HTMLCanvasElement;
const fractal = new FractalCanvas(64, 512, 320);
const view = new SpiralCanvasView(canvas, fractal, {
    palette: {
        stops: [
            { offset: 0, color: "#0b1026" },
            { offset: 0.5, color: "#3967ff" },
            { offset: 1, color: "#ffce69" },
        ],
        gamma: 0.8,
    },
});

view.on("stats", ({ summary }) => {
    console.log("hyper RMS", summary.hypergradRms);
});
```

The helper stays framework-agnostic so it can be wrapped inside React/Vue components or
connected to bespoke UI panels. Use the exposed `on("pointer", …)` hooks to keep external
controls synchronized with the navigation state.

For teams that want a ready-to-go diagnostics HUD, `types/canvas-dashboard.ts` exposes a
vanilla DOM controller that wires `SpiralCanvasView` up to palette selectors, curvature
sliders, pointer toggles, and a stats grid:

```ts
import { FractalCanvas } from "spiraltorch-wasm";
import { SpiralCanvasView } from "./canvas-view";
import { SpiralCanvasDashboard } from "./canvas-dashboard";

const canvas = document.querySelector("#fractal") as HTMLCanvasElement;
const fractal = new FractalCanvas(64, 512, 320);
const view = new SpiralCanvasView(canvas, fractal, {
    autoStart: false,
    statsInterval: 125,
});

const hudContainer = document.querySelector("#hud") as HTMLElement;
const dashboard = new SpiralCanvasDashboard(hudContainer, view, {
    customPalettes: {
        dusk: {
            stops: [
                { offset: 0, color: "#021024" },
                { offset: 0.55, color: "#6f8dd6" },
                { offset: 1, color: "#f4bc6d" },
            ],
            gamma: 0.85,
        },
    },
    showRecorder: true,
    snapshotFilename: "latest-frame.png",
    onRecordingComplete: async (clip) => {
        // implement uploadBlobToS3 to push the recording to your backend
        await uploadBlobToS3(clip);
    },
});

view.start();
```

The dashboard ships lightweight glassmorphism-inspired defaults, can be embedded in any
layout, and exposes options for supplying custom palette presets or toggling controls on
and off. Since it only depends on `SpiralCanvasView`, it can be further wrapped inside
framework components as needed.

## Browser-side report audit

The wasm package can audit exported Mellin and Canvas learning reports before they leave
the browser. Use `auditWasmReportObject` for one report or `compareWasmReportsObject` for
a batch; JSON-string variants are also exported for clipboard/storage flows:

```ts
import { auditWasmReportObject, compareWasmReportsObject } from "spiraltorch-wasm";

const audit = auditWasmReportObject(canvasReport);
if (audit.status === "ready") {
    console.log("promote as Z-space context", audit.readiness_score);
}

const comparison = compareWasmReportsObject({
    baseline: oldReport,
    candidate: canvasReport,
});
console.log("best browser context", comparison.best_readiness?.label);
```

Audits include runtime readiness, learning progress, risk flags, and recommendations, so
browser dashboards can decide whether a report is ready for Python/API-LLM handoff without
waiting for a server-side preflight.

## Fractal-field probes

`st-frac::fractal_field` is also available from the wasm surface, so browser-side demos
can generate deterministic branching fields for Mellin lattices and emit a compact probe
that Python Z-space runtimes can ingest later:

```ts
import { WasmFractalFieldGenerator, fractalFieldProbeObject } from "spiraltorch-wasm";

const generator = new WasmFractalFieldGenerator(4, 2.0, 0.55, 24);
const field = generator.branchingField(-2.0, 0.125, 64); // packed [re, im, ...]

const probe = fractalFieldProbeObject(4, 2.0, 0.55, 24, -2.0, 0.125, 64, 8);
console.log(probe.energy, probe.total_variation, probe.samples[0]);
```

The probe records generator hyper-parameters, log-lattice support, field energy, phase
drift, total variation, and a bounded preview of complex samples. Keeping this payload
source-crate tagged (`st-frac::fractal_field`) lets browser experiments, Python reports,
and future Rust backends share the same geometric context contract.

## Log-Z cosmology probes

`st-frac::cosmology::LogZSeries` can be projected in-browser too. This gives demos and
dashboards a compact way to summarise real-valued log-lattice series before handing the
same payload to Python-side Z-space partials:

```ts
import { WasmLogZSeries, logZSeriesProbeObject } from "spiraltorch-wasm";

const samples = new Float32Array([1.0, 1.2, 1.6, 2.1, 2.8]);
const zValues = new Float32Array([
    0.5, 0.0,
    0.2, 0.3,
]); // interleaved complex [re, im, ...]

const series = new WasmLogZSeries(0.0, 0.25, samples, "hann", "l1");
console.log(series.evaluateManyZ(zValues));

const probe = logZSeriesProbeObject(0.0, 0.25, samples, "hann", "l1", zValues, 4);
console.log(probe.sample_stats.energy, probe.projection.stability_score);
```

The probe records sample statistics, windowed weight statistics, projection energy,
phase drift, and a bounded projection preview. It complements Mellin grids by exposing
the real-series cosmology path as a first-class browser context signal.

## Scale-stack geometry probes

`st-frac::scale_stack` is now exposed directly in the wasm package. Browser demos can run
microlocal ⇄ macrolocal interface probes over scalar fields or token/semantic embeddings,
then hand the same JSON shape to Python-side Z-space pipelines:

```ts
import { WasmScaleStack, scalarScaleStackProbeObject } from "spiraltorch-wasm";

const field = new Float32Array([
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
]);
const shape = new Uint32Array([4, 4]);
const scales = new Float32Array([1, 2, 3]);
const levels = new Float32Array([0.25, 0.5, 0.75]);

const stack = WasmScaleStack.scalar(field, shape, scales, 0.01);
console.log(stack.samples()); // packed [scale, gate_mean, ...]

const probe = scalarScaleStackProbeObject(field, shape, scales, 0.01, 2, 3, levels);
console.log(probe.boundary_dimension, probe.coherence_profile);
```

The probe payload is intentionally source-crate tagged (`st-frac::scale_stack`) so browser
telemetry, Python reports, and future Rust backends can agree on where each geometric
signal came from.

### Frame capture and recording

`SpiralCanvasView` now exposes helpers for exporting the currently rendered frame:

```ts
const pixels = view.capturePixels({ applyPalette: true }); // Uint8ClampedArray copy
const pngBlob = await view.toBlob("image/png");
const dataUrl = view.toDataURL("image/webp", 0.9);
const bitmap = await view.toImageBitmap();
const stream = view.createCaptureStream(60); // MediaStream for WebRTC/recording
```

For time-based captures hook the new `types/canvas-recorder.ts` helper. It wraps the
browser's `MediaRecorder` and automatically wires it to the canvas capture stream:

```ts
import { SpiralCanvasRecorder } from "./canvas-recorder";

const recorder = new SpiralCanvasRecorder(view, {
    mimeType: "video/webm;codecs=vp9",
    videoBitsPerSecond: 6_000_000,
});

recorder.start();

// ... wait for a few seconds ...
const clip = await recorder.stop();
```

The vanilla dashboard exposes snapshot/recording buttons out of the box. Provide
`snapshotFilename`, `onSnapshot`, or `onRecordingComplete` callbacks to integrate the
capture workflow with your own UX (e.g. uploading to a backend or pushing into a React
state store).

### Collaborative control between trainers, models, and humans

Real-time co-creation becomes much easier with `types/canvas-collab.ts`. The new
`SpiralCanvasCollabSession` peers a `SpiralCanvasView` across any number of browser tabs
or devices using the `BroadcastChannel` API with automatic fallbacks. Every participant –
whether they're a human artist, a trainer supervising gradients, or the training run
itself – has the same authority to steer palettes, zoom levels, navigation toggles,
stats sampling, and render loop state. Updates are batched into 16 ms micro-windows,
governed by a 20 diff/s token bucket, and pointer broadcasts are throttled to 30 Hz so
the UI keeps a steady 60 fps even under heavy interaction.

```ts
import { SpiralCanvasCollabSession } from "./canvas-collab";

const session = new SpiralCanvasCollabSession(view, {
    sessionId: "lab-floor-7", // pick any shared identifier for the room
    participant: {
        role: "trainer", // "trainer" | "model" | "human" or your own identifier
        label: "Curator A",
        color: "#facc15",
        capabilities: {
            wgpu: typeof navigator !== "undefined" && "gpu" in navigator,
            wasm: true,
            controlSurface: "palette",
        },
    },
    patchRateHz: 20,
    pointerRateHz: 30,
    pointerTrailMs: 240,
    replayWindowMs: 12_000,
    telemetry: (event) => console.debug("collab", event),
    attributionSink: (sample) => conductor.step(sample), // pipe into your ZConductor
    rolePolicies: {
        trainer: { canPatch: true, canState: true, rateLimitHz: 30, gain: 1.2 },
        model: { canPatch: false, canState: true, gain: 0.4 },
    },
    defaultRolePolicy: { canPatch: true, canState: true, rateLimitHz: 10, gain: 0.7 },
});

// Surface shared presence, last input timestamps, and pointer motions inside the HUD.
dashboard.attachCollaboration(session);

// React to remote updates (for example to log attribution or build custom UI chrome).
session.on("state", ({ participant, origin }) => {
    console.info(`%s adjusted the view (%s)`, participant.label ?? participant.role, origin);
});

session.on("pointer", ({ participant, event, origin }) => {
    // Mirror pointer navigation in a minimap, trigger haptics, etc.
    highlightParticipantCursor(participant.id, event.offset);
    console.debug("pointer", origin, participant.id);
});
```

When `BroadcastChannel` is unavailable the helper degrades to a
`localStorage`/`storage`-event transport with a safety-net polling loop, so you can still
wire it into dashboards without special casing. The session emits presence heartbeats at
1 Hz, records join/leave/suppression events via the optional `telemetry` hook, and pipes
every patch (local or remote) through the `attributionSink` so it can be fused straight
into your `ZConductor` dashboards. Each message carries schema version tags, participant
metadata (including the resolved `gain` for each participant), and size guards, making it
straightforward to colour-code the HUD or enforce your own policies on top of the
symmetric default. Declarative `rolePolicies` let you switch individual roles between
read-only, bursty, or high-authority modes: the token bucket honours the narrowest
`rateLimitHz`, the optional `gain` flows through to attribution samples, and the telemetry
hook reports `policy-blocked` events whenever a disallowed patch/state arrives.

Participants can now advertise a structured capability surface via the optional
`capabilities` object. Keys are automatically trimmed to 64 characters, values are limited
to simple JSON primitives (boolean, finite number, string, or null), and the helper keeps
at most 16 entries per participant (512 UTF‑8 bytes per value) by default. The advertised
set shows up on `session.participants`, in presence heartbeats, and in the payload passed
to `attributionSink`, enabling downstream dashboards to fan out richer context (e.g. GPU
availability, palette control preference, experiment tags). Call `session.setCapabilities`
at runtime to push an updated advertisement without waiting for the next presence tick:

```ts
session.setCapabilities({
    wgpu: typeof navigator !== "undefined" && "gpu" in navigator,
    wasm: true,
    sandbox: "beta-2025-10",
});

// Clearing the object broadcasts `null` to peers so they can retire cached badges.
session.setCapabilities(null);
```

Capability propagation keeps the “safe-by-default” posture from earlier hardening work.
Each `CollabRolePolicy` can now declare:

* `allowedCapabilities`: allow-list of keys that the role may broadcast (omit or `null`
  for "anything goes").
* `blockedCapabilities`: deny-list enforced before the allow-list.
* `maxCapabilityEntries`: per-participant ceiling (default 16, hard cap 64).
* `maxCapabilityValueBytes`: UTF‑8 byte budget per value (default 512, hard cap 4096).

Capabilities that miss the allow-list, hit a deny-list, overflow the quota, or include an
unsupported primitive are dropped locally and emit `policy-blocked` telemetry with the
reason code `capability:<context>:<constraint>:<key>`. That keeps innovation room wide
open—roles can still introduce new markers on the fly—while preserving a deterministic
boundary around what crosses the wire and what reaches downstream attribution sinks.

Pointer broadcasts now include two companion surfaces that make spectator UX and replay
dashboards effortless:

* Set `pointerTrailMs` (defaults to `200`) to accumulate per-participant cursor trails.
  Every time a pointer update lands, the session emits a `pointerTrail` event with the
  most recent positions, and you can query the latest trail with
  `session.getPointerTrail(participantId)`.
* Configure `replayWindowMs`/`replayMaxEntries` to bound an in-memory timeline that keeps
  recent pointer, patch, and full-state events. Call `session.replay({ windowMs: 3_000,
  kinds: ["pointer", "patch"] })` to obtain chronologically ordered `CollabReplayFrame`
  records for scrubbers, instant replays, or audit tooling.

```ts
session.on("pointerTrail", ({ participant, trail }) => {
    // Fade a ghost cursor using the recent positions.
    paintTrail(participant.id, trail);
});

const frames = session.replay({ windowMs: 5_000, participantId: "trainer-1" });
frames.forEach((frame) => {
    if (frame.kind === "pointer") {
        timeline.push({ type: "cursor", at: frame.timestamp, offset: frame.pointer!.offset });
    }
});
```

Replay frames preserve the `origin` (local vs. remote), Lamport `clock`, `kind`, and
sanitised payloads, so dashboards can interleave them with attribution reports without
touching the internal queues. Pointer trail emissions are equally policy-aware—roles that
cannot patch still broadcast trails for spectating, while deny-listed capabilities never
reach the queue that feeds the replay log.
