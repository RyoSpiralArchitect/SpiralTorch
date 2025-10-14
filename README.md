
# 🌀🕯️SpiralTorch🕯️🌀

**SpiralK + SoftLogic + (optional) WASM tuner** collaborate to pick the fastest **merge kind** and **tile width** for your hardware—then **Self-Rewrite** locks the win back into your heuristics.
**WGPU** is the default path; **HIP/CUDA** absorb the **same unified choices**. Python wheels target **3.11–3.14**.

Beyond kernels, the project now incubates an ever-expanding pure Rust learning
stack: language stays raw, gradients stay hyperbolic, and meaning is sculpted
directly in Z-space without ever touching NumPy or PyTorch.

> **Why it’s different**
> - **Two-layer consensus:** SpiralK (runtime rules) + WASM table (offline measurements)  
> - **Unified heuristics:** One `Choice { mk, mkd, tile, ctile, … }` across WGPU / HIP / CUDA  
> - **1-CE Subgroup Top-K (WGPU):** candidates → final in a single compute pass  
> - **MidK/BottomK compaction:** 1-CE / 2-CE, tile-aware, same API  
> - **Amega Hypergrad:** unrolled / implicit (Neumann / CG) hyper-gradients that now sync with the pure tensor tape

---

## What you get

- **Rank-K family** (TopK / MidK / BottomK) with a **single entrypoint**
  Backends implement a `RankKExecutor`, decisions are made once via **unison heuristics**, everyone uses the same plan.
- **SpiralK DSL** (K×Lisp-inspired)
  Hard assigns (`mk:`, `tile:`) and soft rules (`soft(mk, …)`, `soft(tile, …)`) that blend with measurements.
- **SoftLogic (finite-domain solver)**
  Explores a tiny discrete space (merge kinds, tiles) and scores candidates with your soft rules.
- **Pure Rust training core**
  `st-tensor::pure` ships dependency-free tensors, hyperbolic Z-space encoders,
  an open-cartesian topos guard (`OpenCartesianTopos`) with rewrite monads that
  swallow NaNs before they propagate, a deterministic `ConjugateGradientSolver`
  with explicit tolerances, the Tokio-uring inspired `UringFractalScheduler`
  plus `FractalSafetyEnvelope`, and the `AmegaHypergrad` tape so you can
  iterate on learning logic without PyTorch/Numpy while staying inside
  non-Euclidean geometry.
- **Causal Graph Compiler**
  Describes *why* ops matter.  Builds dependency-aware execution plans that can
  skip low-impact stages, enforce latency ceilings, and adapt in-flight through
  runtime feedback from the `OpenCartesianTopos` guardians.
- **Distributed Ameba Autograd mesh**
  A wheel-friendly, serverless gradient swarm: agents push/pull updates only to
  their neighbors, respect damping/tolerance guardrails, and mirror the
  zero-traceback ethos while training across an unreliable network.
- **Optional WASM tuner table**
  Autogenerates a simple piecewise `choose(rows, cols, k, sg)` for your device; the runtime gently prefers measured defaults.
- **Self-Rewrite**
  A/B outcomes (Wilson CI) append `soft(...)` into `~/.spiraltorch/heur.kdsl` when the advantage is statistically significant.
  
---

### Features (opt-in)

- `wgpu` / `wgpu-rt`: WebGPU backends + runtime wiring
- `mps`: macOS Metal (MPS)
- `cuda`: CUDA (NVRTC/PTX loader expected)
- `hip`: ROCm HIP (stub-safe)
- **`hip-real`**: ROCm HIP + RCCL “real” path (requires ROCm toolchain & linker; gated on top of `hip`)
- **`kv-redis`**: enable Redis-backed consensus (soft hints); absent = **safe no-op**
- `logic` / `kdsl`: SoftLogic solver / SpiralK DSL

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/RyoSpiralArchitect/SpiralTorch.git
cd SpiralTorch
```

### 2) Build from source (Rust)

**CPU (default; no GPU deps)**
```bash
cargo build -p st-core --release
```

**WGPU (WebGPU; Windows/Linux/macOS)**
```bash
cargo build -p st-core --features wgpu --release
```

**MPS (macOS GPU)**
```bash
cargo build -p st-core --features mps --release
```

**CUDA (optional; needs NVRTC/Toolkit)**
```bash
cargo build -p st-core --features cuda --release
```

**HIP / ROCm (optional; real backend is feature-gated)**
```bash
export HIPCC=/opt/rocm/bin/hipcc
export ROCM_PATH=/opt/rocm
cargo build -p st-core --features hip,st-backend-hip/hip-real --release
```

### 3) Python wheels (optional)
```bash
pip install maturin==1.*
# CPU + WGPU wheel
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
# Add other backends via features (mps / cuda / hip)
```

---

## Minimal API

**Rust (TopK via unified entry)**
```rust
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::rank_entry::{RankKind, plan_rank, execute_rank};

// describe device
let caps = DeviceCaps::wgpu(32, true, 256); // lane, subgroups, max_wg
// plan once (decisions: mk/mkd/tile/ctile/use_2ce)
let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);

// choose a backend executor (WGPU/CUDA/HIP); CPU fallback exists
use st_core::backend::wgpu_exec::WgpuExecutor;
let exec = WgpuExecutor::default();

// launch
execute_rank(&exec, &plan)?;
```

`DeviceCaps` now ships backend-specific constructors (`wgpu`, `cuda`, `hip`, `cpu`) and
builder-style setters (`with_subgroup`, `with_max_workgroup`, `with_shared_mem`) so you
can describe GPUs with realistic limits while still feeding the unified heuristic
chooser a compact struct.  The helpers also expose higher level tuning hints such as
`recommended_workgroup`, `recommended_tiles`, and `preferred_k_loop` so backends can
query consistent defaults without duplicating the heuristic math.  Pair them with the
extended `prefers_two_stage(rows, cols, k)` signature when you want to peek at whether
the planner will promote the 2-pass compaction path for huge matrices.
chooser a compact struct.  It also exposes derived helpers such as
`recommended_workgroup`, `recommended_sweep_tile`, and `recommended_compaction_tile`
so you can introspect the policy or plug device-aware hints into custom tooling.
chooser a compact struct. The chooser normalizes the plans produced by the DSL, the
generated tables, and the fallback rules, aligning workgroup sizes to hardware warp
widths, honouring shared-memory budgets, and scoring each candidate before execution.
When the reported shared memory is too small for the shared-heap paths or two-stage
compaction, the planner now automatically falls back to bitonic variants so that the
plan always honours device limits.

**Python**
```python
import numpy as np, spiraltorch as st

x = np.random.randn(8, 65536).astype(np.float32)
vals, idx = st.topk2d(x, k=1024, device="auto")   # "wgpu > cuda > mps > cpu"
```

---

## Pure Rust training (zero PyTorch/Numpy deps)

Need a bootstrap-friendly learning loop without pulling in heavyweight
dependencies?  `st-tensor::pure` now ships with zero-panic tensors,
hyperbolic distance helpers, and complex-spectrum encoders so the stack keeps
accelerating without ever leaning on NumPy or PyTorch.

```rust
use st_tensor::pure::{LinearModel, PureResult, Tensor, mean_squared_error};

fn main() -> PureResult<()> {
    // Build a dataset for y = 2x + 1 using plain Rust vectors.
    let inputs = Tensor::from_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0])?;
    let targets = Tensor::from_vec(4, 1, vec![1.0, 3.0, 5.0, 7.0])?;

    let mut model = LinearModel::new(1, 1)?;
    for _ in 0..200 {
        model.train_batch(&inputs, &targets, 0.1)?;
    }

    let predictions = model.forward(&inputs)?;
    let mse = mean_squared_error(&predictions, &targets)?;
    println!("Final MSE: {mse:.6}");
    Ok(())
}
```

Everything runs with `cargo run -p st-tensor --example ...` or inside your own
binary crate—no Python wheels required. When you want to leave Euclidean space,
hand text straight to the Z-space encoder and stay in browser-friendly memory
limits without ever tokenizing:

```rust
use st_tensor::pure::{LanguageWaveEncoder, PureResult};

fn main() -> PureResult<()> {
    let encoder = LanguageWaveEncoder::new(-1.0, 0.75)?;
    let z_space = encoder.encode_z_space("SpiralTorch stays homotopy-free")?;
    println!("{} hyperbolic components", z_space.shape().1);
    Ok(())
}
```

Take it further by coupling the Z-space encoder with the brand-new `AmegaHypergrad`
tape: gradients stay conformal, curvature never drifts, and the entire pipeline
continues to run without touching NumPy or PyTorch.

```rust
use st_tensor::pure::{AmegaHypergrad, LanguageWaveEncoder, PureResult, Tensor};

fn main() -> PureResult<()> {
    let encoder = LanguageWaveEncoder::new(-1.0, 0.8)?;
    let wave = encoder.encode_z_space("hyperbolic language without tokens")?;
    let (rows, cols) = wave.shape();

    let mut hypergrad = AmegaHypergrad::new(encoder.curvature(), 0.03, rows, cols)?;
    hypergrad.accumulate_wave(&wave)?;

    let targets = Tensor::zeros(rows, cols)?;
    hypergrad.accumulate_pair(&wave, &targets)?;

    let mut weights = Tensor::zeros(rows, cols)?;
    hypergrad.apply(&mut weights)?;

    println!("updated weight energy = {:.6}", weights.squared_l2_norm());
    Ok(())
}
```

Because the optimiser keeps its own curvature-aware buffer, you can stream
text → wave → hypergrad endlessly without ever seeing a traceback. Non-Euclidean
geometry, imaginary spectra, and category-inspired language flows all feed the
same tape, letting SpiralTorch chase meaning directly in Z-space.

#### Pure Python interop (no NumPy, no Torch)

The new `st_tensor::pure::python` module exposes a C ABI that any Python 3
interpreter can reach with nothing more than `ctypes`. Build the cdylib and use
the helper wrapper that ships in `tools/python/pure_bridge.py`:

```bash
cargo build --release -p st-tensor
python - <<'PY'
from tools.python.pure_bridge import PurePythonBridge

bridge = PurePythonBridge()
encoder = bridge.encoder(curvature=-1.0, temperature=0.55)
hypergrad = bridge.hypergrad(curvature=-1.0, learning_rate=0.03, rows=1, cols=8)

hypergrad.absorb_text(encoder, "SpiralTorch weaves Z-space without tokens")
weights = hypergrad.apply([0.0] * 8)
gradient = hypergrad.gradient()

print("updated weights", weights)
print("gradient norm", sum(g * g for g in gradient) ** 0.5)
PY
```

Under the hood the bridge calls into `st_pure_hypergrad_new`,
`st_pure_hypergrad_apply`, and friends, while forwarding any errors via the
`st_pure_last_error` sentinel so Python never has to chase NaNs or undefined
behaviour. No wheels, no third-party modules—just CPython lists that round-trip
through the same open-cartesian safety net as the Rust stack.

#### Causal Graph Compiler (`st-core::causal`)

Map “why” as well as “how”.  Feed your operations into the causal graph
compiler and receive an execution plan that honours data dependencies while it
skips stages whose aggregated influence falls below your `skip_threshold`.
Runtime observers can extend latency budgets and feed new measurements back via
`CompiledPlan::adapt_with_observation`, letting you reshape the plan in the
middle of a run without breaking determinism.

#### Distributed Ameba Autograd (`st-core::distributed::autograd`)

Turn the network into the optimiser.  Register agents, connect them as a mesh,
and seed gradients locally: the Ameba swarm pushes damped updates peer-to-peer
until the residual falls below the configured tolerance. No parameter server,
no central barrier, and every edge obeys the same NaN-absorbing guardrails as
the pure stack, making it perfect for zero-dependency Python experiments or
browser-hosted WASM canvases that want to train alongside native Rust nodes.

#### Open-cartesian safety nets (no NaNs, no runaway loops)

Hyperbolic Jacobians now flow through an explicit `OpenCartesianTopos`. The
guard rewrites every scalar through a bounded saturation window, ensures tensor
volumes stay inside memory-friendly envelopes, and exposes loop ceilings so
fractal traversals cannot self-intersect. Couple it with the deterministic
`ConjugateGradientSolver` whenever you need implicit solves:

```rust
use st_tensor::pure::{topos::{OpenCartesianTopos, RewriteMonad}, ConjugateGradientSolver, PureResult};

fn main() -> PureResult<()> {
    let topos = OpenCartesianTopos::new(-1.0, 1e-6, 8.0, 256, 4096)?;
    let monad = RewriteMonad::new(&topos);

    // Rewrite overflowing values before they enter the optimiser.
    let mut weights = st_tensor::pure::Tensor::from_vec(1, 4, vec![f32::INFINITY, 2.0, -9.0, 0.5])?;
    monad.rewrite_tensor("weights", &mut weights)?;

    // Solve Ax = b with explicit tolerances (no hidden divergence).
    let solver = ConjugateGradientSolver::new(&topos, 1e-6, 32)?;
    let matrix = [
        4.0f32, 1.0, 0.0,
        1.0, 3.0, 0.0,
        0.0, 0.0, 2.0,
    ];
    let mut matvec = |src: &[f32], dst: &mut [f32]| {
        dst.fill(0.0);
        for row in 0..3 {
            for col in 0..3 {
                dst[row] += matrix[row * 3 + col] * src[col];
            }
        }
    };
    let mut x = [0.0f32; 3];
    solver.solve(&mut matvec, &[1.0, 2.0, 3.0], &mut x)?;
    println!("solution {:?}", x);
    Ok(())
}
```

The same topos feeds the `FractalSafetyEnvelope`, which blends relation patches
in streaming mode without allocating new buffers and refuses to cross the
configured depth horizon.

### Fractal uring scheduler + WASM canvas loop

Feed those spectra directly into an async-friendly fractal loop without ever
allocating more than a small ring buffer. The `UringFractalScheduler` keeps the
latest relation patches in a Tokio-uring style queue, blends them by coherence,
and now offers both `fold_coherence` and the zero-allocation
`fold_coherence_into` so browser/GPU front-ends can reuse their frame buffers.

```rust
use st_tensor::pure::{Tensor, PureResult};
use st_tensor::pure::fractal::{FractalPatch, UringFractalScheduler};

async fn stream_waveforms(samples: Vec<Tensor>) -> PureResult<Tensor> {
    let scheduler = UringFractalScheduler::new(32)?;
    for (depth, relation) in samples.into_iter().enumerate() {
        let patch = FractalPatch::new(relation, 0.9, 0.7, depth as u32)?;
        // Works on any executor; tokio-uring, tokio, or synchronous loops.
        scheduler.push_async(patch).await?;
    }
    scheduler.fold_coherence()
}
```

For browser builds, wire the folded relation into the dedicated WASM canvas
projector so we never allocate more than a single RGBA surface:

```rust
use js_sys::Uint8ClampedArray;
use st_tensor::pure::fractal::UringFractalScheduler;
use st_tensor::pure::wasm_canvas::CanvasProjector;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

#[wasm_bindgen]
pub struct FractalCanvas {
    projector: RefCell<CanvasProjector>,
}

#[wasm_bindgen]
impl FractalCanvas {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize, width: usize, height: usize) -> Result<FractalCanvas, JsValue> {
        let scheduler = UringFractalScheduler::new(capacity)
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        let projector = CanvasProjector::new(scheduler, width, height)
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        Ok(Self {
            projector: RefCell::new(projector),
        })
    }

    pub fn render(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")?
            .ok_or("missing 2d context")?
            .dyn_into()?;
        let mut projector = self.projector.borrow_mut();
        let rgba = projector
            .refresh()
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        let clamped = Uint8ClampedArray::from(rgba);
        let image = ImageData::new_with_u8_clamped_array_and_sh(
            clamped,
            projector.surface().width() as u32,
            projector.surface().height() as u32,
        )?;
        ctx.put_image_data(&image, 0.0, 0.0)?;
        Ok(())
    }
}
```

And keep the JavaScript glue feather-light:

```html
<canvas id="zspace" width="512" height="32"></canvas>
<script type="module">
import init, { FractalCanvas } from "./pkg/spiraltorch_wasm.js";
const wasm = await init();
const canvas = document.getElementById("zspace");
const fractal = new FractalCanvas(64, canvas.width, canvas.height);
await fractal.render(canvas);
</script>
```

Pixels become Z-space relations, the scheduler keeps memory bounded, and the
entire loop stays panic-free even under aggressive streaming. The RGBA buffer
that powers the `<canvas>` upload can also be shared with WGPU textures for a
fully unified compute + render stack when you want GPU-native presentation.

---

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

**How the final choice is made (two-layer consensus)**

- **A** = SoftLogic best (your DSL soft + optional Redis soft)
- **B** = DSL **hard** assignment (if you set `mk:`/`tile:` explicitly, B wins)
- **C** = **Generated table** (tuner output)

Default policy: if **B** exists use it; else score **A** and **C** with backend-aware
occupancy/tile metrics derived from `DeviceCaps`, then add a small prior to **C**
(`SPIRAL_HEUR_GEN_WEIGHT`, default `0.10`).
If the adopted choice wins locally (Wilson CI lower bound > 0.5 with min trials), **Self-Rewrite** appends matching `soft(...)` to `~/.spiraltorch/heur.kdsl`.

---

## Regenerating the WASM table (optional)

The repo includes a tiny generator that converts tuner JSON to a Rust table:
```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json \
  > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

### Fractional FFT / SpiralK roadmap

- **Radix-2 → Radix-4 pipeline**: The new `st-frac::fft` module mirrors the GPU
  butterfly structure so SpiralK can auto-emit subgroup-aware WGSL.
- **Wilson-aware automation**: `st-kdsl::auto` turns latency deltas into
  high-confidence `soft(...)` rewrites, wiring tuned `radix`, `tile_cols`, and
  `segments` into `heur.kdsl` without manual editing.
- **ND GPU indexer**: A dedicated WGSL kernel materialises strided indices and
  per-segment IDs, unlocking fast fractional/FFT dispatches from WASM → Canvas.
- **WASM tuner baking**: The generator now bakes `tile_cols`/`radix`/`segments`
  into the Rust table, ensuring the browser path stays in sync with native
  runners when driving SpiralK graphs.

**Example JSON**
```json
[
  {"rows": 1024, "cols_min": 4096,  "cols_max": 8191,   "k_max": 128,  "sg": true,  "mk": 2, "tile": 512,
   "tile_cols": 1024, "radix": 2, "segments": 1, "use_2ce": false},
  {"rows": 1024, "cols_min": 8192,  "cols_max": 65535,  "k_max": 2048, "sg": true,  "mk": 1, "tile": 1024,
   "tile_cols": 2048, "radix": 4, "segments": 2},
  {"rows": 1024, "cols_min": 65536, "cols_max": 262143, "k_max": 4096, "sg": true,  "mk": 1, "tile": 2048,
   "tile_cols": 4096, "radix": 4, "segments": 4, "use_2ce": true},
  {"rows": 1024, "cols_min": 4096,  "cols_max": 65535,  "k_max": 2048, "sg": false, "mk": 1, "tile": 1024,
   "tile_cols": 1024, "radix": 2, "segments": 1},
  {"rows": 1024, "cols_min": 65536, "cols_max": 262143, "k_max": 4096, "sg": false, "mk": 0, "tile": 2048,
   "tile_cols": 2048, "radix": 4, "segments": 2, "use_2ce": true}
]
```

The generator now bakes FFT-oriented hints (`tile_cols`, `radix`) and the ND GPU
segment count directly into the Rust table, so `st-core` can immediately expose
them to the SpiralK Wilson self-rewrite logic.

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

## Safety & fallbacks

- Builds **CPU-only** by default (no GPU toolchains required).
- WGPU / CUDA / HIP are **feature-gated** and degrade safely.
- Heuristic chooser always returns a **safe** `Choice` (fills mk/tile from table or conservative defaults).

---

## Contributing

Issues & PRs welcome—especially:
- Backend kernels (WGPU subgroup variants, HIP/CUDA heap/k-way merges)
- Tuner recipes & generated tables
- New SpiralK sugar (e.g., `penalty_if(...)`, device-aware bands)

Run tests/benches on your device and share logs (latency / shapes / adapter caps).  
**AGPL-3.0-or-later** keeps it open and remix-able.

---

## Social preview

Upload a social preview PNG via **Repo → Settings → Social preview** (1200×630).  
Suggested caption: **“SpiralTorch — WGPU-first, Self-Tuning GPU Top-K (Rank-K)”**.

---

### Troubleshooting

- **No Redis?**  
  Build without `kv-redis` or leave `REDIS_URL` unset. The consensus chooser
  skips network calls and falls back to SpiralK / Generated-table safely.

- **ROCm not installed but `hip` enabled?**  
  Use `--features hip` only (stub path). The **real** path needs `hip-real`
  and a working ROCm + RCCL toolchain.

- **Wheels red?**  
  First build CPU+WGPU only: `maturin build -m bindings/st-py/Cargo.toml --release --features wgpu`
  to decouple GPU toolchain issues.

---

## License

**AGPL-3.0-or-later** for every crate and Python wheel. See `LICENSE`.
