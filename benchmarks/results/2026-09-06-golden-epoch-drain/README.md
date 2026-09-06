# Golden Failed-Epoch Completion Boundary

This is a CPU training/concurrency correctness record, not a throughput or
GPU benchmark. It does not claim a Black Cat policy, WASM, or CUDA speedup.

## Frozen Sources

- Unmodified baseline: `1ef53ed0da3304de2839d01890a68fa09ff07994`.
- Regression tests with old runtime behavior: `e95f1f4a6bb376da92a2fc3e098f2d5e05e65c1d`.
- Runtime fix: `d02e0245bdc6e3b46aae1771cccaaba8cd1fcf04`.
- Final source, including documentation and test-constructor lint fix:
  `ce02926b69aa4531ac7889babba0a142eaf405ec`.
- Final `crates/st-nn/src/golden.rs` SHA-256, identical on macOS and Furnace:
  `09622add2f0f0e1d5de3e76bfc338fa4709b6c3658c69da854a68a48ade6e703`.

## Results

| Probe | Source | Result |
| --- | --- | --- |
| Task-error and panic gated-worker regressions | test baseline | Both fail: epoch returns while another worker is running |
| macOS ARM64, Rust 1.98.0, debug, `golden,kdsl`, no defaults | final | 700 passed, zero failed/ignored |
| Furnace Linux x86_64, Rust 1.98.1, debug, `golden,kdsl`, no defaults | runtime fix | 700 passed, zero failed/ignored |
| Furnace Linux x86_64, Rust 1.98.1, release, default features plus `golden` | final | 707 passed, zero failed/ignored |
| Two gated-worker regressions, 25 repetitions | final | 50 test cases / 100 gated-worker scenarios passed |
| Workspace rustfmt, `nightly-2026-04-15` | final | Passed |
| Strict st-nn Clippy with tests | baseline and final | Both fail with the same 49 distinct source-position diagnostics; zero new diagnostics |

The regression gates a second worker with channels while worker 0 errors or
panics. It tests both a successful and a failing second worker, retains the
first error, checks loss retirement before return, and preserves epoch/dropout
state. Cleanup releases the gate and drains the pool before asserting, including
on the red baseline. The 100 ms absence check is not a performance measurement;
five-second watchdogs bound test waits. Twenty-five repetitions are a bounded
regression check, not proof against every possible thread schedule.

The implementation also drains earlier submissions if enqueueing fails. That
rare enqueue-failure branch was inspected, not fault-injected. There is no new
cancellation, deadline, state rollback, global execution lock, or recovery of
failed model/loss objects. Existing multi-epoch sequential/parallel parameter
parity, survivor reordering, quorum, council, and Black Cat pulse tests pass.

## Reproduce

```bash
cargo test --locked -p st-nn --no-default-features --features golden,kdsl --lib
cargo test --locked -p st-nn --release --features golden --lib
cargo +nightly-2026-04-15 fmt --all -- --check
```

At the test-baseline commit, filter with
`epoch_waits_for_other_submitted_workers` to reproduce the two expected failures.

`raw-logs.tar.gz` retains 11 raw logs/reports, including both failed baseline
tests and the still-failing strict Clippy runs. The Clippy comparison maps final
diagnostic primary lines back to unchanged baseline lines and compares code,
message, source file, line/column ranges, and duplicate-target multiplicity.
The repeated-test report includes its pinned executable SHA-256.
Archive SHA-256:
`9599726d3d41d1a814161b97d4eadff01735d98a5c5902e19093b006ceae38bc`.
