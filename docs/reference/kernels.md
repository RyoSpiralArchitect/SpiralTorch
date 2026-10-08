# SpiralK, kernel heuristics, and Amega

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [Heuristics (SpiralK) — optional & powerful](#heuristics-spiralk--optional--powerful)
- [Regenerating the WASM table (optional)](#regenerating-the-wasm-table-optional)
- [Fractional FFT / SpiralK execution path](#fractional-fft--spiralk-execution-path)
- [Amega Hypergrad (unrolled / implicit)](#amega-hypergrad-unrolled--implicit)

## Heuristics (SpiralK) — optional & powerful

SpiralK is a tiny runtime DSL for device-aware choices. Flip it on, then shape the policy per device.

```bash
export SPIRAL_HEUR_SOFT=1
export SPIRAL_HEUR_K='
  # mk: 0=bitonic, 1=shared, 2=warp (subgroup path on WGPU)
  mk:   sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  # mkd: sub-strategy (auto/heap/kway/bitonic/warp_heap/warp_bitonic)
  mkd:  sel(mk==2,4, sel(mk==1,1,3));
  # TopK sweeping tile
  tile: sel(log2(c)>15.0, 2048,
        sel(log2(c)>13.0, 1024,
        sel(log2(c)>12.0,  512, 256)));
  # Mid/Bottom compaction tile
  ctile: sel(tile>=1024, tile/2, tile);

  # Soft hints (gently bias the solver)
  soft(mk,   2, 0.25, sg && (k<=128));
  soft(mk,   1, 0.20, (k>128)&&(k<=2048));
  soft(tile, 2048, 0.20, log2(c)>15.0);
  soft(tile, 1024, 0.15, (log2(c)>13.0)&&(log2(c)<=15.0));
'
```

Need LLM-assisted rewrites without leaving Rust? Build an
`AiRewritePrompt`/`AiRewriteConfig` pair and call `rewrite_with_ai(...)` to let
the Grok-style template generator emit `soft(...)` rules based on Wilson metrics
and device guards. Python bindings expose the same surface via
`spiraltorch.spiralk.rewrite_with_ai`, returning both the rewritten program and
the generated `SpiralKHeuristicHint` objects so you can audit or persist the
AI-authored tweaks before baking them into tuner tables.

Prefer to bring your own LLM or heuristic engine? Pass a `generator=` callable
to `spiraltorch.spiralk.rewrite_with_ai` and SpiralTorch will hand your function
fresh `SpiralKAiRewriteConfig`/`SpiralKAiRewritePrompt` clones. Return a list of
`SpiralKHeuristicHint` objects to drive the rewrite pipeline while preserving
the same safety checks (`max_hints`, Wilson guards, DSL parsing) enforced by the
Rust core.

**How the final choice is made (three-way roundtable)**

- **A** = SoftLogic best (your DSL soft + optional Redis soft)
- **B** = DSL **hard** assignment (if you set `mk:`/`tile:` explicitly, B wins)
- **C** = **Generated table** (tuner output)

Default policy: if **B** exists use it; otherwise the runtime invites **A** and **C** into a quick conversation. It scores both with backend-aware occupancy/tile metrics derived from `DeviceCaps`, then adds a gentle prior to **C** (`SPIRAL_HEUR_GEN_WEIGHT`, default `0.10`). When the discussion reaches a Wilson-backed agreement, **Self-Rewrite** appends the matching `soft(...)` into `~/.spiraltorch/heur.kdsl` so the next run starts from the shared insight.

Want to materialise the FFT path straight from the chosen plan? Call the new helpers and feed the result to your browser/WASM runtime:

```rust
use st_core::backend::wgpu_heuristics::{auto_fft_plan, auto_fft_spiralk};

let plan = auto_fft_plan(rows, cols, k, subgroup).expect("heuristics available");
let wgsl = plan.emit_wgsl();
let dispatches = plan.dispatches();
let spiralk = auto_fft_spiralk(rows, cols, k, subgroup).unwrap();
// Bind primary + scratch buffers, then issue `dispatches` in order with the
// listed entry point/radix/span. Persist `spiralk` if the DSL should learn it.
```

Prefer to cache the tuned plan for JavaScript without re-running the heuristics? The WASM bindings now serialise plans as JSON or plain JS objects:

```ts
import { auto_fft_plan_json, WasmFftPlan } from "spiraltorch_wasm";

const planJson = await auto_fft_plan_json(rows, cols, k, true);
if (planJson) {
  const plan = WasmFftPlan.fromJson(planJson);
  await persistPlan(plan.toJson());
  const wgsl = plan.wgsl();
  const manifest = plan.dispatchManifestObject();
  // Dispatch `manifest.dispatches` in order and ping-pong the listed buffers.
}
```

---

## Regenerating the WASM table (optional)

Run the offline baker to convert your latest measurements into a `WasmTunerTable`
that both native and browser builds can consume:
```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json \
  > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

The generated module keeps the JSON embedded verbatim, parses it via
`st-core::backend::wasm_tuner`, and exposes a `choose(...)` helper that the
runtime queries after SpiralK/SoftLogic have spoken. Because the JSON format is
portable, you can ship the same file to a WebWorker, bake a table offline, and
let the browser pick overrides without re-running the tuner in production.

### Fractional FFT / SpiralK execution path

- **Validated radix-2 / radix-4 pipeline**: `SpiralKFftPlan` rejects ambiguous
  shapes, emits bit-reversal plus mathematically checked butterfly stages, and
  publishes the same ordered dispatch manifest to Rust, Python, and WASM.
- **Wilson-aware automation**: `st-kdsl::auto` turns latency deltas into
  high-confidence `soft(...)` rewrites, wiring tuned `radix`, `tile_cols`, and
  `segments` into `heur.kdsl` without manual editing.
- **ND GPU indexer**: A dedicated WGSL kernel materialises strided indices and
  per-segment IDs, unlocking fast fractional/FFT dispatches from WASM → Canvas.
- **WASM tuner baking**: `tools/tuner/tuner_results.json` keeps the measured
  overrides (`tile_cols`, `radix`, `segments`, `mode_*`) in one place so the
  generator can bake them into Rust **and** expose them to the Web via JSON.

**Example JSON**
```json
[
  {"rows": 256,  "cols_min": 0,     "cols_max": 4095,   "k_max": 128,  "sg": true,
   "wg": 128,    "tile": 512,  "tile_cols": 512,  "radix": 2, "segments": 1},
  {"rows": 512,  "cols_min": 4096,  "cols_max": 16383,  "k_max": 256,  "sg": true,
   "wg": 256,    "tile": 1024, "tile_cols": 1024, "radix": 4, "segments": 2},
  {"rows": 512,  "cols_min": 16384, "cols_max": 65535,  "k_max": 2048, "sg": false,
   "wg": 128,    "tile": 2048, "tile_cols": 2048, "radix": 4, "segments": 4, "use_2ce": true},
  {"rows": 1024, "cols_min": 65536, "cols_max": 262143, "k_max": 4096, "sg": false,
   "wg": 128,    "tile": 4096, "tile_cols": 4096, "radix": 4, "segments": 4, "use_2ce": true,
   "mode_bottomk": 2}
]
```

The generator bakes FFT-oriented hints (`tile_cols`, `radix`, `segments`) and
the ND compaction settings into the Rust table, while the same JSON remains
available for WASM workers that want to replay the optimisation flow offline.

---

## Amega Hypergrad (unrolled / implicit)

Rust utilities for hyper-parameter gradients (continuous relaxation):
- **Unrolled**: expand T updates and backprop
- **Implicit**: Neumann or **CG** to solve `(I − J) v ≈ g` efficiently

> See `crates/st-core/src/autograd/hypergrad*.rs`.
> Python glue is kept minimal; wheels can expose helpers.

The pure `st-tensor::pure::AmegaHypergrad` tape mirrors the same mindset in a
dependency-free package, letting you stage language diffusion experiments in
Rust and then feed the resulting curvature-aligned hints back into SpiralK.

---
