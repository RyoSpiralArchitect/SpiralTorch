# Residual attention public clients

This is **post-0.4.29 source correctness evidence**, not a new PyPI release,
complete language model, quality comparison or throughput claim.

The public Python and browser/WASM APIs call the same Rust-owned residual
attention block. Tests use the unchanged independent Torch fixture from the
[parent core study](../2026-10-09-resident-residual-attention/README.md).
The base implementation is `bcddb25c76ada9f6b616bd0461e421555cafae64`.
[validation.json](validation.json) binds the tested client sources, fixture,
local binaries and logs by SHA-256. Large generated binaries and original logs
remain in the local validation archive; only results and verification records
are published here.

## Observations

| Route | Result |
| --- | --- |
| Python default-feature wheel, Metal / Apple M4 | 7 passed, 1 CPU-only skip; all 60 VJP conditions and both 32-update recipes |
| Python CPU-only wheel, separate isolated environment | 4 passed, 4 GPU-only skips; plan composition works and GPU compile fails explicitly |
| Browser public WebGPU package | 60 VJP conditions, two 32-update recipes and 16 guard checks passed |
| Node scalar WASM, `nn` without `webgpu` | Plain and Topos plan composition, exact layout rejection and explicit GPU rejection passed |
| TypeScript 5.9.2 | Generated and shipped whole-file syntax plus residual/module API-shape checks passed; malformed-class negative control rejected |
| Existing attention regression | Python: 5 passed, 1 CPU-only skip, 30 VJPs and 16 updates; TypeScript contract passed |
| Other regression | Existing Node NN-plan test, complete Python documentation recipe, pinned workspace rustfmt and diff checks passed |

The 32-update loops do not read losses or receipts back between updates.
Afterward they check every accepted revision, loss, final prediction and
parameter against the frozen fixture, plus source-plan/snapshot immutability.
Optional Z/pair derivatives, strided inputs, repeated cotangents, foreign/stale
owners and recovery are exercised through the public APIs.

Terminal prediction and terminal input-gradient overflow each reject the whole
VJP, including optional geometry gradients, and the all-or-none update.
The scalar repros explicitly require eight initial parameters, eight parameter
gradients and the same eight parameters after rejection. Empty/truncated
families cannot pass these checks.

Browser-specific coverage includes freeing a plan while compilation is pending,
retaining tensors after owner destruction, freeing receipts during readback,
strict `addLayerNorm` argument checks and real LayerNorm/Topos Module assembly.
The browser reports `BrowserWebGpu`, `Other`, and no hardware name; this is not
an independently identified Apple M4 browser measurement.

## Numerical Results

[The downloaded browser report](browser.json) retains all individual conditions
and update errors. Fixed acceptance remains
`abs(actual - reference) <= 3e-6 + 5e-5 * abs(reference)`.

| Browser quantity | Largest absolute error |
| --- | ---: |
| Forward | 1.1920928955078125e-7 |
| Input VJP | 5.960464477539063e-8 |
| Parameter VJP | 1.1920928955078125e-7 |
| Z-bias VJP | 8.381903171539307e-9 |
| Pair-bias VJP | 7.450580596923828e-9 |
| Final parameters after 32 updates, either recipe | 2.9802322387695312e-8 |

Python checks the same fixed numerical gates, but its unittest log does not
publish per-value maxima. The table above is browser evidence only.
Plain and Topos are separate correctness recipes: initialization draws differ,
so their losses must not be interpreted as a matched geometry-quality ablation.

## Repairs and Reproduction

The initial Python run exposed missing names in `spiraltorch.nn.__all__`.
That wheel and failing log were preserved locally. The names were added and a
fresh wheel was installed before all accepted native runs.

Independent read-only source review then found an unclosed class in the shipped
TypeScript file and the truncated-family loopholes in the terminal tests.
Both were fixed. TypeScript now parses the whole declarations, rather than relying
only on member regexes, and tests the formerly missed malformed-class case.
`skipLibCheck` deliberately limits this to syntax plus explicit API checks, not
a complete downstream application typecheck. The reviewer confirmed no remaining
actionable findings without independently repeating runtime tests.

See the [public client guide](../../../docs/resident_residual_attention_clients.md)
for the complete Python Module recipe and API ownership rules.
Use fresh default and CPU-only wheels in separate isolated environments;
CPU-only builds require `--no-default-features --features python-default`.
For WASM, generate web glue from the `webgpu` build and Node glue from the
`--no-default-features --features nn` build. Keep the generated directories
separate: Cargo uses the same output filename for these feature sets.
The wasm-bindgen CLI must match Cargo.lock (0.2.129).

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B \
  bindings/st-py/tests/test_nn_residual_attention_training.py -v
node bindings/st-wasm/tests/residual_attention_plan.cjs /path/to/cpu/spiraltorch_wasm.js
node bindings/st-wasm/tests/residual_attention_training_types.cjs /path/to/web/spiraltorch_wasm.js
```

The final command requires `tsc` on PATH. Serve the repository over loopback
and open `bindings/st-wasm/tests/residual_attention_training_clients.html`
with its same-origin `module` query parameter. Require all 60 conditions,
two complete 32-update sequences and 16 guards, then download the report.
Do not overwrite frozen fixture or prior result files.

This block still leaves embeddings, causal positional input, byte/token output
heads, decoder-level loss wiring, geometry-parameter ownership and a global
multi-block update to subsequent work. No synthetic block test proves language
model learning quality.
