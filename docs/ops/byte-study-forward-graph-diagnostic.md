# Selected forward graph diagnostic (unreleased)

This is opt-in instrumentation for a predeclared experiment, not a numerical
repair, ordinary replay qualification, or permission to publish a release.
Local feature checks, selected CPU tests and bounded macOS GPU checks have
passed for the exact artifacts described below. Preserve the first failure and use a
separately approved, process-owned deadline and one-attempt policy.
No automatic CI dispatch, experiment retry or runtime fallback is installed.

The [update-boundary probe](byte-study-boundary-probe.md) retains owner weights
and optimizer state after an update. This diagnostic instead copies actual
graph bindings and values **after a selected training forward and loss have
been submitted, before its backward**. Immediately after an optimizer update,
graph bindings can legitimately still contain the preceding revision; rebinding
occurs inside the next forward.

## Python client

The following assumes an already parsed `study`, an existing checkpoint if
resuming, and a new private output directory. Running this snippet starts the
whole study or its remaining updates. It is not a harmless read-only probe.

```python
case_name = "mixture-11-3-train-query-train"  # must exist in this exact request
resume = None  # or an already saved, all-case checkpoint for this request
study.validate_forward_diagnostic(case_name, 2, resume=resume)

# Only after authorizing this bounded instrumented run:
observed = study.run_wgpu_with_forward_diagnostic(
    case_name, 2, resume=resume, max_bytes=64 * 1024 * 1024
)
with (new_output_dir / "run-outcome.json").open("x", encoding="utf-8") as stream:
    stream.write(observed.run_outcome_json)
if observed.report_json is not None:
    with (new_output_dir / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(observed.report_json)

try:
    raw = observed.read_diagnostic_json()  # consumes once, including on failure
except RuntimeError as error:
    with (new_output_dir / "diagnostic-error.txt").open("x", encoding="utf-8") as stream:
        stream.write(str(error))
    raise
else:
    with (new_output_dir / "forward-graphs.json").open("x", encoding="utf-8") as stream:
        stream.write(raw)
```

Parsing and `validate_forward_diagnostic` do not select a runtime. The update
number is absolute and one-based, and must be after the saved cursor. Both
methods reject bool, floats, strings and coercible non-integer objects for the
update/budget. A CPU-only wheel exposes validation and constants, but valid GPU
execution requests raise `NotImplementedError` rather than falling back.
Actual graph sizes/tokens are checked at the chosen boundary before any copy.

All cases retain their original request order. `resume=None` starts from the
request's initialization; providing a checkpoint restores all cases. The outcome
binds the request digest, starting cursor, and SHA-256 of the checkpoint's Rust
`to_json()` representation (null for a fresh start). This is not a hash of
arbitrary original JSON whitespace. Capture occurs once for the chosen pair.

## Reading the evidence

`ByteCorpusForwardDiagnosticObservation` is opaque and immutable. Its
`run_outcome_json` keeps the observed study report or error independently of
capture status: `not_reached`, `copies_submitted`, or `error`. A capture error
does not replace the learning outcome or cause a retry. Device failure can leave
earlier copies already submitted. Reading consumes the saved handle even if
mapping or encoding fails; concurrent readers cannot consume it twice. Reading
uses that handle's context and does not request another runtime.

`report_json` is Rust's exact serialization of the completed, instrumented report,
or None for a failed run. Do not rebuild it through a Python float/JSON round trip
for exact-byte comparisons. Save the original outcome before any diagnostic read
and retain read errors separately.

The graph JSON schema is `spiraltorch.byte_decoder_graph_diagnostic.v1`:

- `parameter_bits` and `activation_bits` are integer u32 f32 bit patterns, not
  normalized floats. NaN payloads, infinities and signed zero remain intact.
- Guards are raw integer fields; nonzero guards are evidence, not discarded data.
- All u64 counters/revisions are decimal strings so browser JSON cannot round
  them. Model revision and each graph's rebinding counter are distinct fields.
- Each named graph lists its absolute `model_slots` in graph order. A contextual
  query appends its flattened router table after projection parameters.
- `direct_model_slots` are explicitly excluded values: embeddings, wave controls,
  gains and non-contextual shared router tables. Full wave/attention tapes,
  optimizer state, targets and loss are not included. Some wave features appear
  as query graph inputs. This is not a full checkpoint or a replayable tape.

Rust owns the same contract through `run_with_forward_diagnostic` and
`ByteDecoderGraphDiagnostic::to_json`. Native callers use `read`; WASM Rust
callers use `read_async().await`. No new JavaScript binding is introduced here.

The exported limits are 256 graphs, 64 MiB aggregate staging payload (16 MiB per
graph), and 256 MiB encoded JSON. The encoder checks bytes while streaming and
does not build a second value tree. These limits are **not** peak host-memory
limits; reading and JSON decoding require additional memory. Rust callers may
choose a smaller encoding cap with `to_json_with_limit`.

Copies change scheduling and resource lifetimes. A successful read, matching
bits or a passing instrumented run does not diagnose the historical failure,
prove the ordinary path deterministic, or relax publication gates. Every outcome
and graph payload retains `release_ready=false`.

## Validation scope (2026-10-25)

These are separate artifact-bound results, not interchangeable release gates.
The historical numerical mismatch remains unresolved. Passing local checks
does not authorize another experiment, CI dispatch or publication.

### Host-only checks

On native macOS arm64, Rust 1.97.0 and CPython 3.12.6:

- WGPU-feature type checks passed for `st-nn` (library and tests) and the Python
  binding. These were compilation checks, not GPU runtime tests.
- 22 selected Rust host-only diagnostic/observer tests passed with WGPU compiled
  in. Separately, 82 study tests passed with WGPU disabled. Counts are per feature
  configuration and include six selection tests exercised in both builds.
- A local CPU-only wheel built with `--no-default-features --features
  python-default,cpu` passed 21 portable Python tests without Torch or NumPy.
  Installed package/native bytes and import origins were verified against that
  wheel; there were no skipped tests. This development wheel was not published.

Those host-only checks did not acquire a GPU. The portable Python client suite
is also selected explicitly in the ordinary CI client lane. Its CPU-artifact
execution-refusal case skips on a WGPU-enabled wheel; the four export and
preflight cases still run. The separate real-GPU diagnostic suite remains a
two-opt-in, explicitly authorized experiment, not an automatic CI addition.

### Development-artifact diagnostic checks

The development-profile macOS arm64 wheel had SHA-256
`d9414031fa4ab10ff18a9fcc8e1428140e8ba2b936e7308ea92655d0a65256bf`;
its native extension had SHA-256
`e2d787fb5e1ddb7a8a69baa1f4d945a6ac361c8678ba289680308d9642cda2c7`.
Two predeclared `ForwardDiagnosticGpu` tests ran once and passed. They exercised
graph copying/readback, model-drop ownership, two competing consumers with
exactly one successful read, immutable outcome/report access, and one-byte
capture-budget failure without replacing the completed learning outcome.
The budget rejection is not a device-loss or mapping-failure simulation.

A later bounded localization used the same artifact and retained both ordinary
and instrumented full/resumed reports. All four reports were byte-identical
across 25 cases (3,220,433 bytes each; SHA-256
`825a442c8c25680c9fa31196d6dd8e3c719c7c628f842a83e8695e616f006aa6`).
At the selected second-update boundary, seven named graphs covered 21 parameter
tensors and 18 activation tensors with equal raw bits and zero guards. An
independent coverage check required every expected graph, shape and slot;
matching two incomplete captures alone was not accepted. Direct values and
non-graph tapes remain outside this diagnostic's scope.

### Release-profile ordinary-path checks

A different, local release-profile universal2 wheel had SHA-256
`86a928cc71f83231af4e920bbcdcec218668633f8d6428ff5db36bad37a8dd98`;
its native extension had SHA-256
`509a7a7b6fb33f622bc0f74ae1a976684a89c19d3ba473f39a8664e86f3de369`.
On Apple M4 Metal with CPython 3.12.6, it passed the six unchanged `StudyGpu`
tests and three `ByteGpu` clipping tests once, including four fresh-process
inference handoffs. There were no failed, errored or skipped GPU tests.
The same installed wheel passed the official inline smoke and 21 standalone
release checks: 97 unittest successes and one documented recursion-wrapper
sentinel skip. Torch and NumPy were absent.

These are ordinary learning/resume checks, not evidence that the diagnostic
GPU tests ran against this release-profile binary. The native bytes differ;
development-artifact diagnostic results cannot be transferred to this wheel.
The source is still an unmerged diagnostic candidate and the local 0.4.32
version label must not be uploaded or used to replace the published release.
Neither Intel execution, other platforms, WASM/browser behavior nor actual
device/mapping-error behavior was qualified here. No numerical repair,
cross-runtime determinism or general LLM-quality improvement is established.
