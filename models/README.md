# SpiralTorch Model Zoo (WIP)

This directory is the start of a "model zoo" for SpiralTorch: small, runnable,
copy-pastable models that act as reference implementations for both Python and Rust.

## Layout

- `models/python/`: runnable scripts using `import spiraltorch as st`
- `examples/modelzoo_*.rs`: runnable Rust examples (`cargo run -p st-nn --example ...`)

## Running

- Python: `PYTHONNOUSERSITE=1 python3 -s models/python/mlp_regression.py`
- Python (classification): `PYTHONNOUSERSITE=1 python3 -s models/python/zconv_classification.py`
- Rust: `cargo run -p st-nn --example modelzoo_mlp_regression`
- Rust (classification): `cargo run -p st-nn --example modelzoo_zconv_classification`

## Goals

- Keep examples dependency-light (no NumPy / PyTorch required).
- Prefer deterministic runs (seeded tensors) so results are reproducible.
- Establish conventions for state dict export/import as the zoo grows.

## Serialization conventions (draft)

- Store weights as JSON or bincode using `st.nn.save(path, model)` from Python
  (auto-detects format) or `st_nn::save_json` / `st_nn::save_bincode` from Rust.
- The Python helper also writes a `*.manifest.json` file; load it via
  `st.nn.load(manifest_path, model)` or `st.nn.load(manifest_path)` to get a
  state dict.
- Keep filenames aligned with the example name, e.g. `models/weights/mlp_regression.json`.
