# Frozen Artifact-Pair Performance Screen

Declared before timing. A is the retained pre-batching contextual-model binary;
B is the correctness-checked batching candidate. Exact native, WASM, JS,
fixture and expected output hashes are in [performance-protocol.json](performance-protocol.json).
No threshold, order, repetitions, fixture or retained failure may be changed
after inspecting timings. Interrupted/failed trials stay in the record; do
not replace a slow or failed sample with a retry.

## Work And Timing Boundary

Each invocation runs the entire unchanged four-case contextual-model fixture:
zero/nonzero query projection, one/two blocks, all three metrics, 16 updates
per case, plus output/VJP, causal, guard and same-runtime resume checks.
No cases, readbacks or validation are removed to improve timing.

Native measures fresh-process wall time from launch to exit, including fixture
loading, runtime setup, compilation/cache effects, learning, checks, JSON output
and file writing. Browser measures the awaited Rust export from call to returned
JSON string, including GPU runtime creation, learning and checks. Fetch, WASM
initialization, JS parsing/hash checks and download are outside the browser timer.
These are different observables: compare A/B only within each runtime, not
native versus browser or SpiralTorch versus Torch.

Failure recording is outside the timers. Native records each attempted
invocation to an fsynced ledger before launch and records termination afterward.
Browser commits a strict-durability IndexedDB checkpoint before each invocation
and after its result; raw reports remain local to the loopback origin until
download. A repeated page/block recovers the stored record without running it
again. Interrupted attempts are ineligible, not replaced. Storage failure
stops the block. Imported JS comes from the verified bytes via a distinct blob
module per arm; the verified WASM bytes are supplied directly to initialization.

Three blocks per runtime. Every native invocation is a fresh process; each
browser block uses a fresh page and separate A/B WASM instances. Each block
first warms A then B, then measures A, B, B, A. Keep all six invocations and
their output checks. Only the four declared measured rows enter timing ratios.
Builds, other GPU studies and the two runtimes are serialized. No power-mode,
compiler, request or environment tuning between A and B.

## Decision Rule And Limits

For each block, divide median B time by median A time. A runtime passes this
screen only if every block ratio is at most 1.0 and the geometric mean of its
three ratios is at most 0.95. Both native and browser must pass for the combined
screen. Preserve all durations and ratios, including warmups and rejected runs.
This is a fixed engineering threshold, not a statistical significance test.

Every report, warmup included, must match the retained same-runtime bytes.
Browser comparisons use the existing page's JS-normalized JSON plus fixture
hash, not original Rust numeric text. Its original update-14 rounded-delta
qualification failure must remain visible; speed cannot waive it.

A passing artifact-pair screen alone does not adopt the candidate. Also require
candidate native/browser runs of the unchanged 75-case, 128-update v8 corpus to
match each retained baseline report exactly, and independent review of this
harness. If this screen fails, retain the candidate and results as rejected
under this protocol rather than relaxing the rule. A future experiment must
have a new protocol and retain this result.

This measures the specified full qualification workload, not isolated kernel
latency, large-model FT throughput or general hardware superiority. Captured
artifact pairs are compared; timings alone do not isolate compiler/layout
effects or prove that every change in duration is due to submission batching.
Only scalar timing/check records and hashes are public; raw reports stay local.
