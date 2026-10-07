# Shared Topos Gates: Variable-Batch Learning

Correctness evidence for source `4809ba969bf56f173dc8fa43ac5247dc08cc0647`,
relative to parent `9b734752e61ec517f44af0da709d2cf429044157`. This is **not a
speed benchmark or a pretrained-model quality result**.

Rust `ToposResonator::with_shared_gate` owns `(1, F)` trainable parameters.
Forward expands that gate; backward sums per-element VJPs without another
row average. Input and upstream tensor layouts are normalized by logical
coordinates. Existing constructors retain elementwise gates.

## Evidence

- Full native CPU NN library: **772 tests pass**.
- Native WGPU Topos suite: **26 tests pass**, with actual execution markers.
  The new shared-gate test covers row counts 1, 3 and 257, five features, all
  four CPU/WGPU forward/backward combinations and two repeated pullbacks.
  Each of the 12 combinations checks gradients before and after accumulation;
  completed typed `sum_axis0` receipts confirm actual CPU/WGPU reduction and
  no fallback. This is numerical and routing evidence, not accelerator timing.
- Two synthetic **100-update SGD trajectories** run through Rust `Sequential`.
  Rows cycle through 1, 3, 8 and 2; five features share five learned gates.
  Row-major and column-major inputs/upstreams alternate. Both use coupling
  0.2, five Picard iterations, saturation 1.0 and learning rate 0.03; porosity
  is respectively 0.0 and 0.3. This is one deterministic run, not a seed study.
- Independent Torch 2.12.1 CPU f32 replay uses a `(1, F)` leaf gate, native
  broadcasting, autograd and mean loss. It retains its own weights through
  every update rather than resetting to the Rust trajectory. Every output,
  input VJP, shared-gate VJP, updated gate and loss passes the preselected
  `rtol=5e-4`, `atol=3e-5` comparison.
- NN-enabled wasm32 release compilation, formatter and **150** tool/result
  tests pass. The WASM result is compilation, not direct NN execution.
- Five deliberately corrupted probes are rejected under Python `-O`, without
  writing success reports: row-averaged gate gradient, modified weight,
  missing update, nonnumeric gradient and wrong gate layout. The 22 new
  stdlib receipt-guard tests do not substitute for the separate Torch replay.

| Field | Maximum absolute Torch difference |
| --- | ---: |
| Output | 1.1920928955078125e-7 |
| Input gradient | 5.960464477539063e-8 |
| Shared gate gradient | 1.1920928955078125e-7 |
| Updated gate | 2.9802322387695312e-8 |
| Loss | 1.1920928955078125e-7 |

`native-learning.json.gz` contains all synthetic numeric states, not model
weights or corpus text. `torch-check.json` binds the uncompressed receipt and
both checker sources. `verification.json` records source/binary/log hashes,
test scopes and rejected attempts. Full logs and the exact native executable
remain local. Hashes establish byte consistency, not independent provenance.

Initial test-only API-name mistakes and a metadata macro recursion error were
corrected before this successful run; their failing compile logs are retained
and hashed. The macro limit was not raised. Existing WGPU vendor warnings are
not suppressed. Prior capture benchmarks and signed-zero correction evidence
remain unchanged; no old timing is attributed to this shared-gate change.

## Reproduce

Use fresh output paths; the probe/checker refuse to overwrite existing files.
At the source revision above, with Rust dependencies already available:

```sh
cargo test --locked --offline --release -p st-nn --lib
cargo test --locked --offline --release -p st-nn --features wgpu --lib topos -- --nocapture
cargo run --locked --offline --release -p st-nn --example topos_shared_gate_probe -- /tmp/topos-shared-new.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-shared-new.json /tmp/topos-shared-check-new.json
cargo build --locked --offline --release -p spiraltorch-wasm --target wasm32-unknown-unknown --features nn
```

The Torch check needs CPU PyTorch. To independently replay the saved numeric
trajectory rather than generate a new native one:

```sh
gzip -dc benchmarks/results/2026-10-07-topos-shared-gate/native-learning.json.gz > /tmp/topos-shared-saved.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-shared-saved.json /tmp/topos-shared-saved-check-new.json
```

## Boundaries

This remains a host-Tensor NN route. CPU capture still retains expanded gates;
WGPU execution still includes host transfers. `Sequential::backward` recaptures
forward activations, so direct-layer capture savings do not establish graph
speed. Core audits describe expanded elementwise VJPs, not the reduced shared
gradient. There is no new shared-gate Python/WASM NN constructor, new native
Python package, pretrained training, heldout rescoring, cleanup or quality claim.
