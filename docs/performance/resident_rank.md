# Resident Exact Rank

`st-backend-wgpu::rankk_exact_2ce::resident::ResidentRank` owns the input,
scratch, output buffers, bind group and pipelines for one fixed rank plan.
It uses the same exact two-stage shaders as `dispatch_host`; the existing host
API is unchanged. TopK is descending, BottomK ascending, and MidK selects the
center of the ascending finite candidates. Equal values prefer the lower input
index; float total ordering distinguishes signed zero. Non-finite candidates
are ignored and missing outputs are `(NaN, -1)`.

## WASM First

Build with the `webgpu` feature. Creation validates dimensions before narrowing
JS numbers and awaits WebGPU pipeline validation. No CPU fallback is used.

```js
const rank = await WgpuRank.create("topk", 1, 8, 3, 128);
rank.upload(new Float32Array([3, 1, 3, -2, 7, 4, 0, 1]));
rank.dispatch();
const {values, indices, generation} = await rank.readback();
// values: [7, 4, 3], indices: [4, 5, 0], generation: 1n
rank.free();
```

`dispatch(repetitions=1)` enqueues 1..1024 repetitions without reading back.
All sort/merge dispatches share one compute pass and queue submission. Each
dispatch is a separate [WebGPU usage scope](https://gpuweb.github.io/gpuweb/#synchronization);
wgpu inserts the storage dependencies between dispatches, including scratch
reuse. This avoids allocating a separate native pass for every kernel.
`synchronize()` waits using a four-byte map completion fence. `readback()`
immediately snapshots values and indices into one staging buffer, then returns
an asynchronous mapping promise. Later uploads, dispatches or `free()` do not
change that snapshot. Upload invalidates the current output; readback before a
new dispatch is rejected. Failed input-length validation preserves prior state.

Rank and matmul use the same Rust readback lease implementation. Each workspace
retains at most one idle staging buffer after a successful read (or a snapshot
dropped before mapping). Outstanding snapshots still own distinct buffers;
later reuse cannot mutate an already returned Python list/Tensor or JavaScript
typed array. Failed or cancelled mapping is discarded, not cached. A snapshot
does not keep a destroyed workspace's cache alive. This is staging allocation
reuse, not removal of the final copy, map, or host-owned result allocation.

The shared shader writes value bits as `u32`, avoiding browser rejection of a
constant NaN expression, and uses `workgroupUniformLoad` for the merge-loop bound.
Neither change disables shader validation or changes the CPU ordering contract.
For MidK with at most 32 tiles, candidates find their global rank through
parallel binary searches over the sorted tiles. This avoids serially discarding
half of each row. One merge workgroup owns each candidate tile, rather than one
workgroup scanning every tile in a row. The total (value, index) order gives each
valid candidate a unique output destination. Only the disjoint missing-value
tail is initialized, so one workgroup cannot overwrite another's valid output.
Host and resident execution share the same checked dispatch grid. More fragmented
geometries use one row workgroup. When at least 64 finite candidates precede the
retained band, it first seeks the band's exact starting position: 32 total-float
key probes and at most `ceil(log2(cols))` source-index probes count lower bounds
across sorted tiles. The largest key/index with at most `start` predecessors is
the first retained candidate; each tile cursor is set to its lower bound, then
the existing k-way merge emits only the retained band. Ties, signed zeros and
non-finite exclusion use the same ordering as the CPU reference. Short prefixes
and full-width bands retain the direct merge. No extra dispatch, storage buffer,
planner tile rewrite or language-specific selection policy is introduced.

The 256-lane merge workgroup skips leading reduction levels whose higher lanes
contain only zero counts or invalid candidates. The first stride depends on the
number of tile lanes, capped at 128; one tile needs no pairwise reduction, while
129 or more tiles retain every level. This also reduces fragmented MidK prefix
counting work without changing the <=32-tile parallel MidK branch. It does not
change workgroup size, tile choice, dispatch count or any retained ordering step.
The `--suite active-lanes` comparison crosses tile counts and small/wide k values;
its [recorded results](../../benchmarks/results/2026-09-06-rank-active-lanes/README.md)
include slower and unchanged controls, not an all-shape speedup claim.

To reproduce the fragmented rank comparison against PyTorch CUDA on the same
named GPU, build the native example and run:

```bash
cargo build -p st-core --no-default-features --features wgpu-rt --example resident_rank_bench --release
python tools/bench_resident_rank_vs_torch.py --executable target/release/examples/resident_rank_bench --suite midk-boundary --resident-only --output /path/to/new-result.json
```

The suite crosses 5/32/33/129/257 tiles, three seeds, and TopK/MidK/BottomK controls;
one seed is quantized to exercise ties. `--resident-only` excludes intervening
host maps/uploads from the fixed-input timing intervals. Correctness is checked
before and after all intervals. The default suite and rotated host/resident
comparison remain available. These are host-API/fence timings, not GPU events.
CUDA uses stable sort for MidK and for any TopK/BottomK row with tied values;
tie-free TopK/BottomK retain `torch.topk`. Every timed CUDA case checks exact
source indices as well as values before and after all intervals. Each report
records the chosen CUDA operation; tied controls cannot receive credit for a
weaker ordering contract.

Tiles with a padded stride up to 1024 now sort in 8 KiB of workgroup memory,
publishing their sorted run to global scratch only once. Larger tiles keep the
storage-memory sorting network. Both paths use the same total-order comparator;
requested tile geometry, output ordering, padding and snapshot semantics are
unchanged. Device admission checks the workgroup storage requirement explicitly.

## Python And Rust

```python
import spiraltorch as st

rank = st.WgpuRank("topk", 1, 8, 3, tile_cols=128)
rank.upload(st.Tensor(1, 8, [3, 1, 3, -2, 7, 4, 0, 1]))
rank.dispatch()
result = rank.readback()  # flat values/indices lists plus integer generation
assert result["indices"] == [4, 5, 0]
```

`st.wgpu.WgpuRank` is the same class. A CPU-only wheel exposes a constructor
that raises `NotImplementedError`, rather than silently executing on CPU.
Rust callers construct `ResidentRank` from a `WgpuRuntime` and checked `Plan`,
then use `upload`, `dispatch`, `snapshot().read()` and `synchronize` directly.

## Opt-In GPU Stage Diagnostics

Pass timestamps require a separate, explicitly profiled runtime. Default
workspaces never request the feature or silently substitute wall-clock timing
when it is unavailable. These factories do not replace the shared runtime.

```js
const rank = await WgpuRank.create("midk", 1, 8193, 65, 256, true);
rank.upload(Float32Array.from({length: 8193}, (_, i) => i));
const profile = await rank.profile(16);
console.log(profile.tile_sort_total_ns, profile.row_merge_total_ns);
rank.free();
```

```python
rank = st.WgpuRank("midk", 1, 8193, 65, timestamp_queries=True)
rank.upload(st.Tensor(1, 8193, list(map(float, range(8193)))))
profile = rank.profile(16)
```

Rust uses `ResidentRank::request_profiled(plan).await` or
`ResidentRank::request_profiled_blocking(plan)`, then
`dispatch_profiled(n)?.read()?` (or `read_async().await?` on WASM).
`createFromAdaptation(session, index, true)` / Python's
`from_adaptation(session, index, timestamp_queries=True)` retain the same Rust
candidate geometry, but do not feed diagnostic timings back to the policy.

The shared Rust `spiraltorch.rank_gpu_profile.v1` report keeps per-pass raw
ticks and generation as decimal strings, subtracts integer clocks before float
conversion, and records timestamp period, zero/quantized intervals, stage sums,
merge entry point, submission count and native host pacing. Pending reads own
their query storage even after later uploads or workspace destruction.

**This is a diagnostic execution path, not the ordinary fast path.** Portable
stage timestamps require separate passes. Up to 256 repetitions fit each
submission; larger calls use multiple submissions, and native Metal waits
between chunks with a timeout to avoid exhausting its command-buffer pool.
Browser timestamps may be quantized to zero. `gpu_span_ns` includes gaps between
passes/submissions; stage sums exclude those gaps and query readback. Neither
can be subtracted from host clocks to infer pure overhead. A profiled runtime
also does not share device handles with default matmul workspaces.

`resident_rank_profile_bench` pairs uninstrumented dispatch/completion with
instrumented query readback, validating exact outputs outside timing. The
isolated browser runner's `rank-profile` fixture exercises the same schema,
ownership and 1024-repetition boundaries. Native/Python live timestamp tests
require `SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS=1`; lack of capability is an error,
not a CPU fallback or a successful zero-timing measurement.

Query creation/encoding/submission errors are captured in owned error-scope
futures before another profile can begin. Reads reject validation, allocation
and internal failures instead of accepting cleared query buffers as zero-duration
evidence. The pinned browser backend maps `GPUInternalError` through its
`GPUError` base type, preserving a typed error instead of trapping during conversion.
A rejected `popErrorScope()` Promise also becomes an internal error: unavailable
validation is not a clean scope and cannot publish profiled output.
Browser query resources are explicitly destroyed after submission/readback or
cancellation, rather than waiting for JavaScript garbage collection. This uses
a narrow `destroy_webgpu` extension in the pinned wgpu dependency; normal
query-handle Drop and all non-profiled execution retain their existing behavior.

Rank profiling requires the dedicated factory's private device. Its runtime
handles never escape, so another workspace's ordinary construction/dispatch
cannot enter its device-wide scopes. `ResidentRank::new` with a shareable
timestamp-enabled runtime still supports ordinary work, but profiling rejects
with `ProfileRequiresPrivateDevice` before allocating queries or changing output
freshness. No lock or coordination cost was added to ordinary dispatch. Python
and WASM keep the same public `timestamp_queries` arguments and use the private
factory internally. Profile construction validates all three error classes too.
An internal device-scope lease remains as a defensive encoding check, not an
execution/readback lock. WASM pops scopes synchronously before returning a promise.

Workspace buffers, idle staging buffers and pending query storage retire before
their final owning native device. Keeping the device alive until those resources
drop prevents repeated private-workspace creation from leaking retired device
resources. Pending reads still outlive the workspace; the lifetime regression
exercises both successful reads and cancellation over 96 creation/drop cycles.

Profiled output remains stale until the corresponding query read and captured
validation succeed. `synchronize()` alone does not publish it. Failed or dropped
Rust readbacks cannot expose output, and an older profile cannot publish after
a newer profile, upload, copy or ordinary dispatch supersedes it.
Profiling detaches earlier publication state before fallible encoding/submission,
so a native intermediate-chunk timeout cannot leave prior output marked current.
Initial admission failures (invalid repetitions, missing input, unsupported or
shareable devices) remain transactional. Python's
blocking `profile()` and an awaited successful WASM `profile()` publish the
same state. Ordinary dispatch keeps its existing submission-freshness contract
and does not allocate or read a diagnostic publication token.

The [single-pass comparison and stage study](../../benchmarks/results/2026-09-07-rank-single-pass/README.md)
retain repeated RTX 5090/PyTorch CUDA controls, matched browser and A/A runs,
raw query intervals, and the rejected allocation-failure prototype. Native
fixed controls improve at the median, but CUDA remains faster and browser
timing noise prevents a general browser speedup claim.

## Projection To Rank Without Host Staging

`set_input_from_matmul` (`setInputFromMatmul` in JavaScript) copies current
matmul output into the rank input on the GPU. This is a device-to-device copy,
not zero-copy aliasing: later source updates or destruction cannot change the
copied input. The destination has a new input generation and requires dispatch.
Source shape and exact device/queue handles must match. Missing/stale source
output, mismatched shape/device, or generation overflow leave rank state intact.
Freshness means submitted logical output, not a completed GPU execution receipt.

```python
head = st.WgpuMatmul(1, 2, 4)
head.upload(st.Tensor(1, 2, [1, 0]), st.Tensor(2, 4, [2, 7, 3, 1, 0, 0, 0, 0]))
rank = st.WgpuRank("topk", 1, 4, 2)
head.dispatch()
rank.set_input_from_matmul(head)  # no logits readback or host upload
rank.dispatch()
assert rank.readback()["indices"] == [1, 2]
```

```js
const head = await WgpuMatmul.create(1, 2, 4);
head.upload(new Float32Array([1, 0]), new Float32Array([2, 7, 3, 1, 0, 0, 0, 0]));
const rank = await WgpuRank.create("topk", 1, 4, 2);
head.dispatch();
rank.setInputFromMatmul(head);
head.free();
rank.dispatch();
const result = await rank.readback(); // indices: [1, 2]
rank.free();
```

Rust exposes the same setter on `ResidentRank`. This composes a resident linear
projection with exact rank selection; it is not a complete language model or a
fused single-dispatch kernel. The copy and the following rank dispatch are
separate queue submissions.

### One-Submission Composition

`rank.dispatch_from_matmul(head, repetitions=1)` in Python, or
`rank.dispatchFromMatmul(head, repetitions)` in JavaScript, computes the head,
copies its output, and runs exact rank in one queue submission. No intermediate
map or host upload occurs. It uses the head's configured kernel, tile and
accumulation policy, rather than silently selecting a new one.
Rust uses `rank.dispatch_from_matmul(&mut head, repetitions)` with the same
implementation and state contract.

```python
head.upload(st.Tensor(1, 2, [1, 0]), st.Tensor(2, 4, [2, 7, 3, 1, 0, 0, 0, 0]))
generation = rank.dispatch_from_matmul(head)  # no separate head/rank dispatch
assert head.output_is_current and rank.output_is_current
assert rank.readback()["generation"] == generation
```

```js
const head = await WgpuMatmul.create(1, 2, 4);
head.upload(new Float32Array([1, 0]), new Float32Array([2, 7, 3, 1, 0, 0, 0, 0]));
const rank = await WgpuRank.create("topk", 1, 4, 2);
rank.dispatchFromMatmul(head);
head.free();
const result = await rank.readback(); // indices: [1, 2]
rank.free();
```

Both source operands must be uploaded, but its output need not be current:
this method recomputes it. Invalid repetitions, missing operands, incompatible
shape/device/queue, or exhausted rank generation leave both workspaces intact.
Only successful submission marks both outputs current; this is still logical
freshness, not completion attestation. The source input generation is unchanged,
and the rank input generation advances once per call.

Repetitions 1..1024 encode complete matmul/copy/rank chains with the **same
operands**, not a recurrent model or a sequence of different token inputs.
The rank input remains an owned copy, so later source updates/free do not affect
it. Readback snapshots keep their prior ownership contract. One submission does
not mean one shader dispatch or zero-copy aliasing.

```sh
cargo build --release --locked -p st-core --features wgpu-rt --example resident_matmul_rank_bench
python tools/bench_matmul_rank_vs_torch.py \
  --executable target/release/examples/resident_matmul_rank_bench \
  --output matmul-rank-comparison.json
```

This harness compares the intermediate host bridge against the GPU copy bridge,
including the same final rank readback. It also measures resident chains against
preallocated PyTorch CUDA matmul/stable-sort with TF32 disabled. All three rank
kinds check both values and canonical indices, including cutoff ties. Inputs are bounded
integers to make fp32 projection and stable tie-order checks exact; these
fixtures do not establish arbitrary floating-point rank stability or model
quality. Source/image binding and foreign-GPU-process gates match the rank-only
harness; no exclusive GPU reservation or interleaved framework timing is claimed.
The native diagnostic additionally rotates a one-submission/final-readback
bridge, 16 one-submission calls, and a batch of 16 complete chains in one
submission. All resident intervals end at a completion fence and exclude maps;
every mode is checked against the reference after timing. Separate calls and
batched replay remain distinct metrics rather than being pooled into one speedup.

With `--readback-probe`, the Python wrapper first completes **all** normal native
and CUDA comparisons, then starts a separate native process for diagnostics.
The native executable's flag is probe-only: it emits no comparison timings.
Diagnostics are stored separately under `readback_diagnostics`, never mixed
into comparison samples. Extra completion fences separate projection,
copy, rank and snapshot submission/read; these are not GPU-event times and do
not reproduce the unfenced critical path. The synthetic byte-copy control holds
another MAP_READ buffer alive in **all** modes, so its fresh-allocation cost is
not an isolated allocation baseline. The opt-in Rust test
`snapshot_copy_stages_match_output_when_enabled` separately compares snapshot
allocation with and without such an anchor. Keep these diagnostics separate
from the harness's normal end-to-end/resident comparisons.

`--resident-only` runs only sixteen composed calls plus a completion fence per
interval, rather than interleaving host-to-host modes. It performs no maps or
uploads between intervals; fixed-operand output is validated before and after
all samples, not after every sample. Reports label this boundary explicitly.
The Python wrapper can combine this option with a later, separate diagnostic
pass, but one native invocation cannot mix the two modes.

## Measurement Boundaries

The comparison separates the existing host API, a persistent-buffer
host-to-host call, and batched resident dispatch plus completion fence. The
PyTorch reference uses preallocated CUDA outputs, with the same 16 repetitions
and no upload/readback in the resident interval. MidK uses stable full sort in
PyTorch. This is an API throughput diagnostic, not GPU-event kernel timing or
an interleaved cross-framework experiment. Do not compare resident times with
PyTorch host-to-host times as a speedup claim.

```sh
cargo build --release --locked -p st-core --features wgpu-rt --example resident_rank_bench
python tools/bench_resident_rank_vs_torch.py \
  --executable target/release/examples/resident_rank_bench \
  --output resident-rank-comparison.json
```

The harness requires a clean source-bound build, preserves exact requests by
hash, verifies output against canonical PyTorch ranks, and checks source and
execution-image stability. It crosses three seeds, three tiles, three rank
kinds and two widths. Numerical failures are errors, not timing samples.
The runner rejects other CUDA compute PIDs before and after measurement. This
is a best-effort process check, not an exclusive GPU reservation; shared-load
measurements must still be treated as provisional.
