# Pre-Measurement Timing Harness Review

The independent local reviewer inspected `performance-protocol.json`,
`performance-protocol.md`, `bench_native.py`, `bench_browser.html`,
`summarize_performance.py`, and `test_timing_summary.py` without editing,
running GPU/browser work, building or accessing the network.

Initial actionable findings:

- P2: rejected exports, malformed output and interrupted native children could
  disappear from the sample list. Both harnesses now record invocation identity
  before execution and preserve elapsed time, failure and available output.
- P2: browser evidence existed only in memory. Strict-durability IndexedDB
  checkpoints now precede and follow each invocation; existing blocks recover
  without retries. An ownership flag prevents a failed concurrent add from
  overwriting another page's record.
- P2: fetched-and-hashed JavaScript was imported again by URL. Each arm now
  imports a blob made from the verified source and initializes verified WASM
  bytes directly, avoiding a second unchecked source load.

After the repairs, the reviewer reported no actionable P1/P2 findings in the
six files. Seven CPU-only tests passed in both the parent and reviewer runs;
the reviewer also exercised independent failure, recovery, race and aggregation
controls. These are synthetic tests, not measured timing or learning results.
Warmup exclusion, strict schedule/output binding, median/geometric-mean rules
and the original browser qualification failure remain in the contract.

At this pre-measurement review, actual browser storage durability, crash
recovery, blob-module execution and GPU timing remained **unverified**.
That review did not establish a speedup,
adoption, statistical significance or general native/browser equivalence.

## Completed Screen Review

Reviewer `01a1325b-0293-7471-96f6-847cbb2a61a9` found no actionable P1/P2 issue
in the subsequent README, checksum manifest, performance results and evidence
inventory. Independent recomputation confirmed all 36 samples in the frozen
order, all 18 native start/finish pairs, ratios `0.911304` native and `0.973660`
browser, the browser block-3 regression, and the combined rejection.

The reviewer verified 12 public checksums, six unchanged frozen harness files
and all 55 listed private-artifact hashes. Normalizing the retained browser
Rust strings reproduced every browser report hash. The original qualification
failure remained and no raw weights, reports, checkpoints or corpus content
were published. This section was appended after that returned review.

This was read-only retained-record verification, not execution attestation or
crash-durability testing. Source rollback verification remained parent-owned.
No files were written and no reruns, builds, GPU/browser actions, network
access or broad Git scans were performed by this reviewer.
