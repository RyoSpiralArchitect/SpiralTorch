# Julia integration design note

## Target call sites in `crates/`

| Location | Responsibility | Inputs | Outputs | Performance considerations |
| --- | --- | --- | --- | --- |
| `st-core/src/backend/unison_heuristics.rs::choose_unified_rank` | Unified entry point that evaluates KDSL hints, KV overrides, generated WGPU heuristics, and the Rust fallback to emit a `Choice`. | `(rows, cols, k, caps, kind)` covering tensor geometry and backend caps. | `Choice` struct containing merge strategy, tiling, FFT knobs, and optional latency window. | Invoked during every rank planning path, so additional latency must stay sub-microsecond to avoid impacting planner hot paths.【F:crates/st-core/src/backend/unison_heuristics.rs†L1698-L1758】【F:crates/st-core/src/ops/rank_entry.rs†L47-L68】 |
| `st-core/src/backend/unison_heuristics.rs::fallback_with_scenario` | Baseline heuristic used when external signals are absent; computes tiles, compaction windows, merge strategy, and latency tuning hints. | `RankScenario` (derived from `rows/cols/k/caps/kind`). | Baseline `Choice` plus optional `LaneWindow`. | Runs for every plan when Julia override is missing, so the Julia path must match or beat its ~microsecond cost to justify externalizing the logic.【F:crates/st-core/src/backend/unison_heuristics.rs†L1141-L1233】 |
| `st-core/src/backend/unison_heuristics.rs::TempoLearner::{observe, merge, into_choice}` | Aggregates runtime feedback into smoothed heuristics and latency windows, acting as a temporal filter over historical `Choice`s. | Streaming `TempoFeedback` (choice, optional window, weight). | Weighted aggregate `Choice` with merged latency window. | Executes per iteration of adaptive tuning, so Julia integration must maintain amortized linear complexity over feedback samples and avoid heap churn for telemetry-heavy workloads.【F:crates/st-core/src/backend/unison_heuristics.rs†L1255-L1358】 |
| `st-core/src/backend/unison_heuristics.rs::AdaptiveWindowTuner::tune` | Adjusts latency window stride/slack based on temporal fusion metrics and gradient summaries. | Current `LaneWindow`, compaction bounds, `TemporalSpectralFusion`, optional `GradientSummary`. | Tuned `LaneWindow` with stride/slack adjustments. | Called whenever latency-sensitive plans update, so Julia routine must vectorize the smoothing math and minimize allocations to preserve the planner’s responsiveness.【F:crates/st-core/src/backend/unison_heuristics.rs†L679-L759】 |
| `st-core/src/backend/unison_heuristics.rs::latency_ctile_window_with_slack` + helpers | Computes compaction window bounds and snapping rules that guard `Choice::ctile`. | Geometry (`rows, cols, k, lanes`) plus min/max compaction tiles. | `LaneWindow` describing bounds, slack, stride. | Utility functions are on the hot path inside the fallback; Julia replacements must exploit broadcasted arithmetic to keep throughput comparable.【F:crates/st-core/src/backend/unison_heuristics.rs†L847-L1048】 |

These call sites all emit or consume the `Choice` struct, whose fields capture the full kernel selection envelope the Julia side must reproduce.【F:crates/st-core/src/backend/unison_heuristics.rs†L291-L370】 Interfacing with Julia therefore requires serializing/deserializing `Choice` and `LaneWindow` with minimal overhead.

## API surface proposal

The PoC crate `julia-ffi-poc` demonstrates the intended Rust → Julia call boundary. Its primary API, `tempo_latency_score(tile, slack)`, delegates to Julia when the `with-julia` feature is enabled and otherwise falls back to the pure-Rust reference implementation for environments without a Julia runtime.【F:crates/julia-ffi-poc/src/lib.rs†L1-L55】 The Julia routine is defined inline via `Value::eval_string` for now, but in production it would live in a precompiled Julia module exposed through `PackageCompiler`, as prototyped in `docs/julia/tempo_score.jl`.【F:docs/julia/tempo_score.jl†L1-L13】

Planned production APIs:

- `tempo_latency_score(choice: Choice, feedback: TempoFeedback) -> LaneWindowSnapshot`
- `choose_unified_rank(rows, cols, k, caps, kind) -> Choice`
- `refine_choice(choice: Choice, scenario: RankScenario) -> Choice`

Each function returns owned Rust structs so that existing planner call sites remain unchanged aside from conversion glue.

## Data exchange strategy

- **Struct layout**: expose mirror Julia `struct`s for `Choice`, `LaneWindow`, and `TempoFeedback`. Serialize/deserialize via `jlrs` `Value::new` / `Value::data_ptr` APIs, or pass through JSON for debugging builds. The PoC currently passes primitive `UInt32` arguments across the boundary to validate end-to-end wiring.【F:crates/julia-ffi-poc/src/lib.rs†L23-L55】
- **Lifetime management**: bootstrap Julia with `jlrs::Builder` in a long-lived runtime owned by the planner service. Use `jlrs` scopes to pin short-lived arrays while avoiding per-call initialization cost.
- **Threading**: planner runs on the CPU thread-pool; Julia runtime should run in single-threaded mode initially, with a queue for heuristic requests to honor Julia’s global interpreter lock semantics.

## Testing strategy

1. **Unit equivalence**: mirror Rust-only logic in Julia and assert equality against the baseline functions, similar to `rust_reference_matches_stubbed_julia`.【F:crates/julia-ffi-poc/src/lib.rs†L57-L70】
2. **Integration tests**: when a Julia runtime is available (`with-julia` feature + `JULIA_DIR` env), execute the actual Julia functions and ensure choices round-trip through serialization.
3. **Property tests**: fuzz the geometry inputs (`rows, cols, k`) to ensure Julia and Rust fallbacks stay within the same latency window bounds as enforced by `latency_ctile_*` helpers.【F:crates/st-core/src/backend/unison_heuristics.rs†L847-L1048】
4. **Performance regression**: run Criterion benchmarks comparing the Rust fallback and Julia-accelerated paths. The current PoC shows the stubbed Julia path is ~3× slower because it reuses the Rust computation; replacing the stub with real Julia code will let us measure interpreter overhead precisely.【c48016†L1-L5】【9fb06d†L1-L5】

## Benchmark snapshot

Running `cargo bench -p julia-ffi-poc` (without the `with-julia` feature) yields:

- `rust_latency_score`: 0.59 ns (median)
- `julia_latency_score_poc`: 1.93 ns (median)

Even the stub demonstrates the expected additional cost of routing through the FFI layer. Once Julia bindings are active we expect a larger gap; caching the runtime and precompiling the Julia module will be critical to stay within the planner’s latency envelope.

## Next steps

1. Stabilize the `Choice`/`LaneWindow` schema in both languages and auto-generate bindings (e.g., via `cbindgen` + Julia’s `StructTypes.jl`).
2. Extract the latency-window math (`AdaptiveWindowTuner`, `latency_ctile_*`) into Julia functions and replace the Rust implementations behind feature flags.
3. Add integration smoke tests that spin up Julia in CI environments where `JULIA_DIR` is configured; otherwise fall back to the Rust path with informative logging.
