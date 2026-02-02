# SpiralTorch WASM examples

These demos run SpiralTorch geometry/training primitives directly in the browser via
`spiraltorch-wasm`.

## Quickstart (recommended)

From the repo root:

```bash
# One command: builds the wasm package, installs deps (first time), starts Vite.
bash scripts/wasm_demo.sh mellin-log-grid
```

Other demos:

```bash
bash scripts/wasm_demo.sh canvas-hypertrain
bash scripts/wasm_demo.sh cobol-console
```

## Static build + simple server

```bash
bash scripts/wasm_demo.sh mellin-log-grid build
python scripts/serve_wasm_demo.py bindings/st-wasm/examples/mellin-log-grid/dist --port 4174
```

## Notes

- If you see a browser error about `application/wasm`, use `scripts/serve_wasm_demo.py` (sets
  the correct MIME type + WASM-friendly headers).
- The wasm package is built by `scripts/build_wasm_web.sh` and synced into each example’s
  `pkg/` directory.

