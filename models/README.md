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
- Rust char-LM examples also write `best_weights.json` plus `samples/best_epoch_*.txt` when validation improves; pass `--early-stop-patience N` to stop after N non-improving validation epochs.
- Rust char-LM examples default to `--char-feature token-bigram`, adding a trainable previous-token/current-token embedding on top of the token embedding. Use `--char-feature token` for the older token-only input; older checkpoints without this metadata load as `token`.
- Rust char-LM examples default to `--head-prior learned-unigram`, a trainable unigram-initialized logit bias. Use `--head-prior unigram` for the old fixed prior or `--head-prior none` to test the model without a frequency prior.
- Rust char-LM coherence examples default to `--self-score-scale 0.0` so the scan context is not dominated by the query token matching itself; use `1.0` for legacy self-inclusive scans.
- Coherence examples also expose `--query-residual-scale` (default `1.0` for new runs, `0.0` when loading older checkpoints without metadata) so the current token embedding can stay visible beside the Z-space context.
- Python scripts accept `--backend cpu|wgpu|cuda|hip|auto`, `--events <path>`, `--atlas`, `--desire` (applies offsets during sampling), and `--softlogic-*` tuning flags (captured in `run.json`).
- WGPU quickstart (build + run): `bash scripts/wgpu_quickstart.sh`
- Python: `PYTHONNOUSERSITE=1 python3 -S -s models/python/mlp_regression.py [--activation gelu --norm zspace]`
- Python (classification): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zconv_classification.py`
- Python (vision + pooling): `PYTHONNOUSERSITE=1 python3 -S -s models/python/vision_conv_pool_classification.py`
- Python (Mellin log-grid classification): `PYTHONNOUSERSITE=1 python3 -S -s models/python/mellin_log_grid_classification.py --val-batches 4`
- Python (Maxwell simulated Z classification): `PYTHONNOUSERSITE=1 python3 -S -s models/python/maxwell_simulated_z_classification.py --val-batches 4`
- Python (VAE): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_vae_reconstruction.py`
- Python (Text→ZSpace VAE): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_text_vae.py models/samples/spiral_corpus_en --mellin ramp --optimizer adam --batch-size 8 --val-batches 8`
- Python (Text VAE on/off comparison): `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_text_vae_compare.py models/samples/spiral_corpus_en --mellin ramp --epochs 3 --batches 24`
- Python (LLM char fine-tune): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_finetune.py <text_or_dir> [<text_or_dir> ...] [--desire --events runs.jsonl --atlas]`
- Python (LLM char VAE context): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_vae_context.py models/samples/spiral_corpus_en --features raw,reconstruction,latent,raw_latent,reconstruction_latent --feature-normalize-modes blocks,vector --hybrid-latent-scales 0.5,1.0 --seeds 7,13 --epochs 3 --vae-epochs 2`; multi-seed reports include aggregate ranking plus per-feature win/near-win stability, including hybrid raw+latent and reconstruction+latent context probes with normalization/latent-scale config grids, `best_config`, and a chained `next_follow_up_command.sh` fresh-seed confirmation script. Use `--head-init xavier` to rescale deterministic Linear head weights for high-scale hybrid probes that would otherwise start with very large logits. Use `--follow-up-from previous/summary.json` to replay the winning config and source feature set on fresh seeds while recording `follow_up_result` / `Follow-Up Result`, `follow_up_chain` / `Follow-Up Chain`, `follow_up_ancestors` / `Follow-Up Ancestors`, `follow_up_trajectory` / `Follow-Up Trajectory`, `follow_up_guidance` / `Follow-Up Guidance`, and `guided_next_follow_up_command` / `Guided Next Follow-Up Command`; trajectory reports include cumulative NLL movement plus conflict/action flags for source-feature tradeoffs, runner-up uncertainty, and unsafe trajectory promotions disable the guided next command until the feature swap is audited. If a retained feature regresses versus its source but still beats the raw baseline, guidance widens fresh-seed confirmation instead of silently discarding the raw-positive candidate. Add `--follow-up-fail-on-verdict regressed,unknown` to keep reports while returning non-zero for weak follow-up outcomes, and the generated script will preserve that gate via `FOLLOW_UP_FAIL_ON_VERDICT`.
- Python (LLM char VAE context chain): `PYTHONNOUSERSITE=1 python3 -P tools/run_char_vae_context_chain.py models/samples/spiral_corpus_en --preset small --follow-ups 1`; writes `chain.json` and `chain_report.md` for parent sweep plus guided fresh-seed follow-ups, preserving gate stops such as seed-shift regressions as inspectable artifacts instead of losing the run context. The `small`/`base` presets scout hybrid latent scales through `4.0`, `hybrid4` focuses scale `2.0,4.0` with `--head-init xavier`, `hybrid4_deep` fixes scale `4.0` with a heavier `10x24` head budget and `256` eval samples, and follow-up reports include `runner_up` / `margin` / `within_uncertainty` plus `delta_vs_raw` / `delta_vs_source` so "barely won", "still beats raw", and "softened versus source" are visible. If that best-vs-runner-up margin is inside combined seed uncertainty, generated follow-up commands boost confirmation to five fresh seeds; the chain runner records seed precedence in the chain artifacts and uses those command-default seeds unless a matching `--follow-up-seed-groups` entry explicitly overrides that follow-up, with `follow_up_seed_group_plan` mapping group slots to attempted / unused-after-stop / extra status, `follow_up_seed_resolution` plus its summary recording the actual per-follow-up seeds/source used, and `extra_explicit_seed_groups` / `unused_explicit_seed_groups` staying disjoint. Compare chain artifact trees with `tools/summarize_char_vae_context_chains.py --recursive`; comparison selection separates the accepted champion from any lower-NLL absolute best that still needs gate-stop review and emits the recommended next action plus follow-up/review summary paths and runnable command usage when available. Add `--allow-gate-stop` when a stopped promotion should still return success for evidence-gathering automation; with multiple `--follow-ups`, guided gate-stop confirmations continue with the next generated command-default seed set instead of stopping the chain early.
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
- Rust char-LM sweep smoke: `PYTHONNOUSERSITE=1 python3 -S -s tools/run_char_lm_sweep.py models/samples/spiral_corpus_en --preset smoke --architectures finetune --features token,token-bigram`
- Rust char-LM sweep compare: `PYTHONNOUSERSITE=1 python3 -S -s tools/run_char_lm_sweep.py models/samples/spiral_corpus_en --preset small --architectures finetune,scan,wave --features token,token-bigram --head-priors learned-unigram --seeds 42,43`
- Rust char-LM no-prior pressure compare: `PYTHONNOUSERSITE=1 python3 -S -s tools/run_char_lm_sweep.py models/samples/spiral_corpus_en --recipe no-prior-context-pressure`; follow with `--recipe no-prior-coherence-budget` to spend a longer cheap-training budget on scan/wave, `--recipe no-prior-coherence-frontier` to compare LSTM/scan/lite-wave finalists with scan/wave-only mix RMS normalization, or `--recipe no-prior-coherence-wave-promoted` to rerun the route-debt-selected single-branch wave shape.
- Rust char-LM ablation report: `PYTHONNOUSERSITE=1 python3 -S -s tools/run_char_lm_ablation_report.py models/samples/spiral_corpus_en --preset smoke` writes a compact benchmark report comparing LSTM, SpiralRNN, coherence scan/wave, and top-k guard variants; use `--head-priors learned-bigram,learned-unigram,none` to repeat the plan across prior strengths.
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
