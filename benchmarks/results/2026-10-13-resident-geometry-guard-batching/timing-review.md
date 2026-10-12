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

Actual browser storage durability, crash recovery, blob-module execution and
GPU timing remain **unverified**. The review does not establish a speedup,
adoption, statistical significance or general native/browser equivalence.
