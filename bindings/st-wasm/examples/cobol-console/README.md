# SpiralTorch COBOL dispatch console

This example shows how to orchestrate SpiralTorch's language narrators from a web UI and
forward the resulting envelope to a COBOL bridge. Humans and model agents share the same
browser surface: they configure narrator parameters through WebAssembly, review the payload,
and ship it to a z/OS entrypoint via HTTP (which can fan the buffer into MQ, CICS, or
datasets).

## Quickstart

From the repo root:

```bash
bash scripts/wasm_demo.sh cobol-console
```

## Prerequisites

1. Build the `spiraltorch-wasm` bindings once so the example can import them:

   ```bash
   ./scripts/build_wasm_web.sh --dev
   ```

   (If you prefer calling `wasm-pack` directly, make sure to unset any host-only linker
   flags like `RUSTFLAGS` / `LIBRARY_PATH` / `PKG_CONFIG_PATH` that point at native
   `vcpkg` archives.)

2. Install the frontend dependencies for the console:

   ```bash
   cd bindings/st-wasm/examples/cobol-console
   npm install
   ```

## Running the console

Start Vite's dev server (or build a static bundle) from the example directory:

```bash
npm run dev
# or: npm run build && npm run preview
```

Open the printed URL (defaults to `http://localhost:5173/`) and populate the form:

* **Envelope identity:** Provide a job identifier, release channel, and optional metadata.
  Humans and models can both register themselves—the UI calls
  `CobolDispatchPlanner.addHumanInitiator`/`addModelInitiator`, so COBOL receives a
  provenance trail.
* **Narrator configuration:** Adjust curvature/temperature, pick the encoder key, and paste
  resonance coefficients. Use the “Seed with demo buffer” button to generate a deterministic
  Float32 series inside the browser (through WASM) when you want to test the pipeline.
* **Mainframe routing:** Configure MQ/CICS/dataset sinks. Populate dataset members, DCB
  attributes (record format/length, block size), SMS classes, SPACE allocations, DSNTYPE,
  LIKE templates, DSORG overrides, VSAM key length/offset/CI sizes, share options,
  REUSE/LOG directives, UNIT/AVGREC hints, catalog options, unit counts, retention days,
  RLSE/ERASE flags, and expiration dates when staging GDGs or sequential files. Any field you
  leave blank is omitted from the envelope so legacy programs can decide how to route the
  narration.
* **Dispatch:** Supply the HTTP bridge endpoint that proxies requests into the mainframe. On
  submission the console shows the JSON envelope, a COBOL pointer preview, and the raw byte
  count. Hitting “Send to bridge” POSTs the `Uint8Array` returned by
  `CobolDispatchPlanner.toUint8Array()`.

The preview panel renders both the dispatch envelope and a compact COBOL pointer summary so
operators can confirm the payload before shipping it to z/OS. The HTTP bridge is expected to
forward the JSON buffer to the shared library introduced in `docs/cobol_integration.md`
(e.g. by placing the message on MQ or invoking a CICS transaction).

## Embedding in your own dashboards

The console code (`src/main.ts`) is intentionally framework-free. Import the
`CobolDispatchPlanner` from `spiraltorch_wasm`, feed it Float32Array coefficients produced by
models or humans, and call `toUint8Array()` to obtain the exact bytes SpiralTorch expects on
z/OS. The helper also exposes `coefficientsAsBytes()` when you want to inject the buffer into
pre-allocated COBOL storage or craft LE/370 pointer records directly.

Pair this workflow with the shared library from `docs/cobol_integration.md` to receive the
payload inside Enterprise COBOL or GnuCOBOL. The mainframe side simply reverses the JSON into
parameters for `st_cobol_new_resonator`/`st_cobol_describe`, returning the rendered summary
back to the web console, MQ subscribers, or downstream datasets.
