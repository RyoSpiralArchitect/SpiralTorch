# The Original NN Model On Resident GPU Inputs

The high-level model now owns and reuses the existing Rust resident graph.
It does not require a second hand-built GPU model:

```text
explicit upload -> model.forward_resident -> more GPU operations -> explicit read
                        Linear + GELU
                      Scaler / ReLU / ...
```

Enable `st-nn/wgpu` in Rust, `wgpu` in the Python build, or `webgpu` in WASM.
These are current-source APIs, not a claim about older published wheels.

## Python

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear("up", 4, 7))
model.add(st.nn.Gelu())
model.add(st.nn.Scaler("gain", 7))
model.add(st.nn.Relu())
model.add(st.nn.Linear("down", 7, 4))

device = st.WgpuTensorDevice.create()
x = device.upload([2, 5, 4], [0.25] * 40)
for _ in range(20):
    x = model(x)  # same model, owning GPU output, no host readback
print(model.resident_cache_info())
print(x.snapshot().read_values())  # explicit terminal observation
```

`model.forward(x)` and `model(x)` select by input type, not a hidden fallback:
host `Tensor` keeps its existing host-returning route; `WgpuTensor` uses the
input's device and returns `WgpuTensor`. Supported modules are `Linear`,
`Scaler`, `Gelu`, `Relu`, and compositions of them in `Sequential`.
Unsupported layers reject before executing their host implementation.

When the next consumer is the host, `model.forward_snapshot(x)` returns a
`WgpuTensorSnapshot` directly. Packing, NN execution and the error guard are
submitted first; the terminal copy is prepared and submitted separately.
`capture.read_values()` explicitly waits and consumes it. This is not an
implicit read inside `model(x)`:

```python
capture = model.forward_snapshot(x)
values = capture.read_values()
```

The method supports the same five built-in module types, takes only `WgpuTensor`,
and rejects CPU-only builds. Parameters and cache selection are shared with
ordinary forwarding, not reconstructed by Python.

## Rust And Browser

In Rust import `st_nn::Module` and call
`model.forward_resident(&input)?` with a
`st_backend_wgpu::resident_tensor::ResidentTensor`. The same
`resident_forward_stats()` and `clear_resident_forward_cache()` methods are
available. Custom modules may explicitly opt into the shared
`ResidentForwardCache::forward(operations, input)`; descriptors must faithfully
represent their ordinary forward semantics.

Rust's `Module::forward_resident_snapshot` and the browser's
`Sequential.forwardSnapshot` return the same owning Rust tensor snapshot contract.
Custom modules can opt in through `ResidentForwardCache::snapshot`; the default
trait implementation rejects. At the explicit graph level, use
`forward_tensor_snapshot` (Rust/Python) or `forwardTensorSnapshot` (WASM).
These return tensor validity, not the stage-indexed `GraphInferenceSnapshot`;
the separate graph snapshot API retains its detailed stage errors.

The browser owns a real `st_nn::Sequential`, not a JavaScript implementation:

```javascript
import init, {Sequential, WgpuTensorDevice} from "./spiraltorch_wasm.js";
await init();
const model = new Sequential();
model.addLinear("up", 4, 7);
model.addGelu();
model.addScaler("gain", new Float32Array(7).fill(1));
model.addRelu();
model.addLinear("down", 7, 4);
const device = await WgpuTensorDevice.create();
const input = device.upload([2, 5, 4], new Float32Array(40).fill(0.25));
const output = model.forward(input);
const stats = model.residentCacheInfo(); // bigint counts
console.log(stats.compilations);
stats.free();
model.clearResidentCache();
model.free(); // output still owns its values
const snapshot = output.snapshot();
console.log(await snapshot.readValues());
snapshot.free();
output.free();
input.free();
device.free();
```

`model.inferencePlan(shape)` still exports a fixed portable plan. After resident
training, `baseline.applyParametersTo(model, updatedPlan)` performs the same
checked, baseline-matching weight handoff as Python's
`baseline.apply_parameters_to(model, updated_plan)`. The next forward follows
the changed weights. See [handoff rules](resident_module_handoff.md); this is not
optimizer-state resume.

## Reuse And Boundaries

- Each parameterized Module holds one bounded, replaceable graph, not a cache
  growing with every shape. Shape, device/queue, operation sequence, layout or
  parameter-bit changes rebuild it. Identical values reuse it.
- Unchanged native parameters use Rust-owned `TensorContentStamp` witnesses to
  skip repeated finite/equality scans. These weak witnesses do not retain tensor
  values or force a value-sized copy on mutation. A Rust mutation, different
  shape/layout, or later shared writable DLPack export invalidates them.
  Foreign storage is never trusted, including read-only imports. Externally
  shared parameters still require exact bit comparison on every reuse, and
  Linear still checks finite values. Copy-only exports do not revoke the source.
  Equal-value replacement is checked once and may refresh a native witness
  without graph recompilation. Signed zero changes remain significant.
- CPU Parameter prepacking uses the same revocable witnesses. A writable export
  made after pack creation cannot silently retain stale packed weights; untracked
  sources are repacked. External writes must be serialized between Rust calls,
  not concurrent with them. Witnesses are process-local change detectors, not
  hashes, portable revisions, or authorization to mutate shared storage.
- Built-in modules append descriptors to one shared vector rather than allocating
  a temporary vector per leaf. `Module::append_inference_ops` is an optional
  allocation-saving companion to `inference_ops`, with the same validation and
  descriptor semantics. Its default calls the existing custom module lowering;
  Sequential restores the caller's prefix if a child fails. Descriptor collection
  still allocates its final vector and clones parameter handles. Cold compilation
  still snapshots and uploads weights; this is not globally zero-copy execution.
- Cached graph forwards bind contiguous, offset-zero inputs directly and write
  the final stage directly into an owning output version. There are no full-sized
  input/output bridge copies on that path. Strided/offset views are packed only
  when needed. Packing, NN dispatches and the owning error guard share one queue
  submission; observing a snapshot is explicitly separate.
- Each graph retains at most four output slots and 32 MiB of output values/flags.
  A slot is reusable only when no other tensor/view or weak storage owner exists.
  Held outputs, bound consumers and the current graph state prevent recycling.
  Saturated or oversized outputs allocate separately, without waiting or fallback.
  The budget covers retained output data, not all model/scratch/binding memory.
- Each retained output keeps its final-stage and guard bindings. Multi-stage
  graphs also reuse the first-stage binding for the same packed input storage;
  successful host upload/set-input invalidates that key. Single-stage or changed
  input boundaries still rebind. Clearing the Module cache drops the pool without
  invalidating externally held outputs.
- Dense and pointwise stages write disjoint words of one shared graph guard.
  One clear and the upstream-input guard copy replace per-pointwise clears/copies;
  logical stage error indices and the owning output guard are preserved.
- Direct forwards append the output-guard capture dispatch to the same compute
  pass as the NN stages. Dispatch ordering preserves stage writes before the
  capture reads them; no flags or validation dispatches are removed. View packing
  may still require its own pass. Explicit snapshot/output-capture APIs retain
  their separate observation behavior.
- Packing, new/uncacheable output allocation, command encoding, the upstream
  guard copy and the final guard dispatch remain. This is not globally allocation-free
  execution. Legacy explicit
  `set_input_tensor` / `dispatch` / `output_tensor` APIs retain their semantics;
  switching from direct forwarding back to `dispatch` copies the current input
  into stable workspace storage once.
- The existing Linear/bias/GELU fusion is reused. Broader pointwise fusion is not
  enabled implicitly. Returned tensors survive cache reuse, clear and model drop.
  Invalid-input flags remain visible at explicit readback even after zero gains
  or ReLU could otherwise hide them.
- Cache counters describe compilation/selection/submission, not successful GPU
  completion or numerical validation. An empty Sequential returns its input
  handle without dispatch and leaves the counters at zero.
- This forward path has no backward tape or optimizer. Generic `ModuleTrainer`
  is not automatically moved to the GPU. An active committed tensor execution
  plan is rejected because it does not yet describe this composite resident
  route; uncommitted scopes do not override an explicitly supplied GPU input.

Explicit `WgpuTensor.snapshot()` captures retain their existing fresh staging
allocation and exclusive readback lease. Snapshots remain readable after the
originating tensor/device wrapper is dropped; cancelling a pending Rust read
unmaps its own buffer without invalidating other captures. New native, Python
and browser tests cover mixed shapes, retained invalid-value flags, dropped
wrappers, negative zero and browser cancellation. Explicit graph snapshots keep
their existing separate pool.

Terminal forwarding uses that same fresh staging and exclusive lease, not either
rejected cache prototype. Delayed reads survive later forwards, weight changes,
cache clear and module destruction. Empty Sequential captures its input and keeps
its NN execution counters at zero. Strided Tensor snapshots preserve the original
pack-then-copy submissions; contiguous snapshots need only the copy submission.
The explicit terminal API changes host-wrapper costs, so endpoint timings do not
isolate GPU queue cost. It does not add a backward tape or change ModuleTrainer
routing.

Combining the forward and terminal copy into one submission was implemented and
tested, but regressed every retained browser pair by about 2-7% at the per-shape
median. The [rejected prototype and complete measurements](../benchmarks/results/2026-09-12-module-terminal-capture/README.md)
are preserved. The current implementation keeps the explicit API but delegates
to the original forward and snapshot paths; fewer submissions are not assumed
to be faster.

One-slot and two-alternating-slot Tensor staging caches were implemented and
tested, but neither is enabled in the selected runtime. Both passed correctness
checks yet regressed the small/middle sustained browser fixtures by about 1-2%.
Their complete [positive and negative evidence](../benchmarks/results/2026-09-12-tensor-readback-cache/README.md)
is preserved; fewer buffer creations alone are not evidence of higher speed.

For timing, `tools/bench_graph_forward_paths.py --include-module` adds ordinary
`model(WgpuTensor)` calls to the existing matched fixture and PyTorch controls.
The module d2h route excludes input upload; compare it separately from h2h.
The fixed-input burst routes each perform eight independent forwards and one
terminal host read. Per-call parameter validation/selection and resident I/O are timed;
cold compilation is recorded separately.

`nn-module-matched` in `tools/test_resident_browser.cjs` loads two frozen WASM
packages in one page and rotates baseline/candidate model routes plus explicit
graph controls. `tools/validate_module_direct_io.py` checks the full nine-case
matrix, source/product hashes, captures and cache counters. The native versions
run in separate processes; browser physical GPU identity remains unknown.

For browser clocks too coarse for one call, `nn-module-intervals` runs a separate
fixed completed-read protocol: four matrices, the same nine shape/seed cases,
two warmup intervals and nine retained intervals per route. Each interval times
256 independent forwards, **each** followed by its own completed read and output
release. It is not a burst with one terminal read. Both original Module and
explicit graph routes are rotated across the two frozen WASM packages; context
creation and case order alternate between matrices. Typed output arrays are
retained until the interval ends, then every value is checked outside the timer.
Thus host output allocation/retention and release are included, but upload,
compilation and numerical comparison are not. This protocol measures sustained
completed-read calls, not isolated-call latency or GPU-only time.

`tools/validate_module_resident_intervals.py` verifies the streamed cases, frozen
products, route order, all read/submission counts and cache reuse. Its primary
endpoint is the median of 12 case-median candidate/baseline ratios per shape;
pooled totals, explicit-dispatch control drift and all slow intervals are retained
separately. Observed clock granularity is reported, not treated as an uncertainty
bound. The earlier single-call and burst measurements remain separate evidence.

`nn-module-terminal-intervals` compares the terminal API against ordinary
forward-then-snapshot. For a same-API comparison between frozen implementations,
use `nn-module-terminal-matched-intervals` and validate with both
`--terminal-capture --same-terminal-api`. Both packages must export
`forwardSnapshot`; the validator rejects mismatched API labels or fixtures.
This separates the API choice from the source-version comparison, without
claiming to isolate driver or GPU-only costs.

The [source-bound first record](../benchmarks/results/2026-09-12-module-resident-forward/README.md)
includes the small-model slowdown as well as the deeper-model wins, CPU-only
build checks, independent Torch replay and the rejected exploratory attempts.

The [direct-I/O follow-up](../benchmarks/results/2026-09-12-module-direct-io/README.md)
adds same-page WASM A/B timings. The small burst improves in this fixture;
larger and single-call cases do not uniformly improve. All control drift and
regressions remain in the record rather than being filtered out.

The [bounded-output follow-up](../benchmarks/results/2026-09-12-module-output-reuse/README.md)
checks safe reuse with live views, consumers and snapshots, and retains four
counterbalanced timing matrices. Small bursts improve in this fixture;
middle bursts are approximately unchanged, and large Python bursts regress.
Fewer allocations are not by themselves evidence of a universal speedup.

The [shared-stage-guard follow-up](../benchmarks/results/2026-09-12-module-shared-guard/README.md)
removes per-pointwise flag clears/copies. Four paired matrices show lower burst
medians across the tested sizes, with two small native regressions retained.
Browser single-call medians remain approximately unchanged. The explicit graph
reference also changes with this patch; only eager Torch is an unchanged implementation.

The [combined preparation experiment](../benchmarks/results/2026-09-12-module-descriptor-assembly/README.md)
verifies fewer CPU allocations and exact parameter-bit checks, but does not
establish a speed improvement: middle browser bursts and large native bursts
regress across the retained four-matrix comparison.

The [four-way isolation](../benchmarks/results/2026-09-12-module-preparation-factorial/README.md)
compares assembly-only, bulk-comparison-only, both, and the preceding baseline.
That selected source retained checked single-vector assembly but restored the
per-element bit comparison. Assembly-only burst medians are approximately at
baseline in the isolation worktree; that is not a universal speedup claim.

The [main-worktree confirmation](../benchmarks/results/2026-09-12-module-assembly-confirmation/README.md)
rebuilds this selected source and repeats the complete verification and four
paired timing matrices against the original baseline. Burst medians remain
approximately baseline, with fewer descriptor allocations and individual
regressions retained. At that baseline, parameter scans and finite checks were
still performed on each call.

The [content-stamp follow-up](../benchmarks/results/2026-09-12-parameter-content-stamps/README.md)
uses revocable weak ownership to avoid rescanning unchanged native parameters,
and fixes stale CPU packs after a later writable DLPack export. Four paired
matrices show the largest completed single-call route at 0.851x baseline in
Python and 0.900x in the browser; burst results are more modest or unchanged.
Small-case regressions, quantization and control drift remain in the record.

The [in-pass guard follow-up](../benchmarks/results/2026-09-12-module-inline-guard/README.md)
keeps the final validation dispatch but appends it to the graph compute pass.
Four paired matrices put native burst medians at 0.799/0.950/0.985x the preceding
baseline for the three fixed sizes. Browser burst medians are unchanged; the
small browser single-call metric is unresolved across coarse-clock statistics,
not a browser speedup claim. Multi-workgroup guard and output-lifetime tests pass
in the Rust, Python and browser clients.

The [sustained completed-read follow-up](../benchmarks/results/2026-09-12-module-completed-intervals/README.md)
times 256 calls per interval while still completing a read after every call.
The same frozen guard candidate/baseline pair is approximately unchanged in
the browser: 0.9984/0.9990/0.9985x primary ratios, with individual regressions and
different pooled-total ratios retained. All 405,504 warmup/measured reads passed
numerical checks. This longer-interval workload does not replace the earlier
isolated-call observations or establish a browser speedup.
