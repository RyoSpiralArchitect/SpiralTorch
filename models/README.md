# SpiralTorch Model Zoo (WIP)

This directory is the start of a "model zoo" for SpiralTorch: small, runnable,
copy-pastable models that act as reference implementations for both Python and Rust.

## Layout

- `models/python/`: runnable scripts using `import spiraltorch as st`
- `examples/modelzoo_*.rs`: runnable Rust examples (`cargo run -p st-nn --example ...`)

## Running

- Demo texts: `models/samples/spiral_demo_en.txt`, `models/samples/spiral_demo_ja.txt`
- Demo corpus folder: `models/samples/spiral_corpus_en/` (multiple `.txt` files)
- Run outputs: `models/runs/<timestamp>/` (e.g. `run.json`, `metrics.jsonl`, `samples/`, `weights.json` / `weights.bin`)
- Python scripts accept `--backend cpu|wgpu|cuda|hip|auto`, `--events <path>`, `--atlas`, and `--desire` (applies offsets during sampling).
- WGPU quickstart (build + run): `bash scripts/wgpu_quickstart.sh`
- Python: `PYTHONNOUSERSITE=1 python3 -S -s models/python/mlp_regression.py`
- Python (classification): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zconv_classification.py`
- Python (vision + pooling): `PYTHONNOUSERSITE=1 python3 -S -s models/python/vision_conv_pool_classification.py`
- Python (VAE): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_vae_reconstruction.py`
- Python (Text→ZSpace VAE): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_text_vae.py models/samples/spiral_corpus_en --mellin ramp`
- Python (LLM char fine-tune): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_finetune.py <text_or_dir> [<text_or_dir> ...] [--desire --events runs.jsonl --atlas]`
- Python (LLM char coherence scan): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_scan.py <text_or_dir> [<text_or_dir> ...] [--desire --events runs.jsonl --atlas]`
- Python (LLM char coherence wave): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_wave.py <text_or_dir> [<text_or_dir> ...] [--infuse \"spiral\" --infuse-every batch --infuse-mode separate] [--desire --events runs.jsonl --atlas]`
- Python (LLM char WaveRnn+Mixer): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_wave_rnn_mixer.py <text_or_dir> [<text_or_dir> ...] [--preset base --weights-format auto --checkpoint-every 1 --val-batches 4] [--infuse \"spiral\" --infuse-every batch --infuse-mode separate] [--desire --events runs.jsonl --atlas]`
- Rust: `cargo run -p st-nn --example modelzoo_mlp_regression`
- Rust (classification): `cargo run -p st-nn --example modelzoo_zconv_classification`
- Rust (vision + pooling): `cargo run -p st-nn --example modelzoo_vision_conv_pool_classification`
- Rust (mixer): `cargo run -p st-nn --example modelzoo_zspace_mixer_regression`
- Rust (VAE): `cargo run -p st-nn --example modelzoo_zspace_vae_reconstruction`
- Rust (sequence): `cargo run -p st-nn --example modelzoo_wave_rnn_sequence`
- Rust (LLM char fine-tune): `cargo run -p st-nn --example modelzoo_llm_char_finetune -- <text.txt>`
- Rust (LLM char coherence scan): `cargo run -p st-nn --example modelzoo_llm_char_coherence_scan -- <text.txt>`
- Rust (LLM char coherence wave): `cargo run -p st-nn --example modelzoo_llm_char_coherence_wave -- <text.txt> [--infuse \"spiral\" --infuse-every batch --infuse-mode separate]`
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
