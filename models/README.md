# SpiralTorch Model Zoo (WIP)

This directory is the start of a "model zoo" for SpiralTorch: small, runnable,
copy-pastable models that act as reference implementations for both Python and Rust.

## Layout

- `models/python/`: runnable scripts using `import spiraltorch as st`
- `examples/modelzoo_*.rs`: runnable Rust examples (`cargo run -p st-nn --example ...`)

## Running

- Python: `PYTHONNOUSERSITE=1 python3 -S -s models/python/mlp_regression.py`
- Python (classification): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zconv_classification.py`
- Python (LLM char fine-tune): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_finetune.py <text.txt>`
- Python (LLM char coherence scan): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_scan.py <text.txt>`
- Python (LLM char coherence wave): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_wave.py <text.txt> [--infuse \"spiral\" --infuse-every batch]`
- Rust: `cargo run -p st-nn --example modelzoo_mlp_regression`
- Rust (classification): `cargo run -p st-nn --example modelzoo_zconv_classification`
- Rust (vision + pooling): `cargo run -p st-nn --example modelzoo_vision_conv_pool_classification`
- Rust (mixer): `cargo run -p st-nn --example modelzoo_zspace_mixer_regression`
- Rust (VAE): `cargo run -p st-nn --example modelzoo_zspace_vae_reconstruction`
- Rust (sequence): `cargo run -p st-nn --example modelzoo_wave_rnn_sequence`
- Rust (LLM char fine-tune): `cargo run -p st-nn --example modelzoo_llm_char_finetune -- <text.txt>`
- Rust (LLM char coherence scan): `cargo run -p st-nn --example modelzoo_llm_char_coherence_scan -- <text.txt>`
- Rust (LLM char coherence wave): `cargo run -p st-nn --example modelzoo_llm_char_coherence_wave -- <text.txt> [--infuse \"spiral\" --infuse-every batch]`
- Rust (GNN): `cargo run -p st-nn --example modelzoo_gnn_graph_regression`
- Rust (Z-RBA telemetry): `cargo run -p st-nn --example modelzoo_zrba_telemetry`
- Rust (Lightning + selfsup): `cargo run -p st-nn --example modelzoo_lightning_selfsup_minimal`

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
