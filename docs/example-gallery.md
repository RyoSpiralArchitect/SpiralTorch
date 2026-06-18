# SpiralTorch Example Gallery

This page lists examples that ship with the repository.

## Running from source

- Python: prefer `PYTHONNOUSERSITE=1 python3 …` (or `python3 -s …`) to avoid user-site `.pth` surprises.
- Rust: run examples via `cargo run -p <crate> --example <name>`.

## Python examples

### Python bindings showcase
**File:** `examples/python_bindings_showcase.py`

```bash
PYTHONNOUSERSITE=1 python3 examples/python_bindings_showcase.py
```

---

### System integration demo
**File:** `examples/system_integration_demo.py`

```bash
PYTHONNOUSERSITE=1 python3 examples/system_integration_demo.py
```

---

### Z-space coherence sequencer demo (new layers)
**File:** `examples/05_new_layers/zspace_coherence_demo.py`

```bash
python3 -s examples/05_new_layers/zspace_coherence_demo.py
```

---

## Rust examples

### Custom operator registry demo
**File:** `examples/custom_operator_demo.rs`

```bash
cargo run -p st-core --example custom_operator_demo
```

---

### Plugin system demo
**File:** `examples/plugin_system_demo.rs`

```bash
cargo run -p st-core --example plugin_system_demo
```

---

### Ecosystem integration demo
**File:** `examples/ecosystem_integration_demo.rs`

```bash
cargo run -p st-nn --example ecosystem_integration_demo
```

---

### Self-supervised fine-tuning demo
**File:** `examples/fine_tune_with_selfsup.rs`

```bash
cargo run -p st-nn --example fine_tune_with_selfsup -- <artefact_dir>
```

---

### Character LM fine-tuning demo
**File:** `examples/modelzoo_llm_char_finetune.rs`

```bash
cargo run -p st-nn --example modelzoo_llm_char_finetune -- models/samples/spiral_demo_en.txt --head-rms 0.1 --val-fraction 0.1 --eval-samples 256
```

Writes `run.json`, `metrics.jsonl`, `summary.json`, samples, and weights into the selected run directory.
Pass `--recurrent lstm` to swap the default SpiralRNN core for the stateless batched LSTM path; compare both with `tools/run_char_lm_sweep.py --architectures finetune,lstm ...`.
For recurrent shape sweeps, pass `--step-values`, `--hidden-values`, and `--embed-dim-values`; for training-budget sweeps, pass `--epoch-values` and `--batches-values`; for residual-logit pressure sweeps, pass `--head-residual-scale-values`; for bigram top-k preservation sweeps, pass `--bigram-topk-guard-values`. `compare.md` aggregates rows separately by those dimensions.
Top aggregate rows rank by validation NLL first and use trace latency plus CPU debt as tie-breaks, which makes equal-quality recurrent shapes easier to compare.
Use `tools/run_char_lm_sweep.py <text-or-dir> --recipe guarded-lstm` to reproduce the validated guarded LSTM/SpiralRNN sweet-spot comparison; explicit flags still override the recipe when you want a narrower probe.
When a sweep contains both `bigram_guard=0` and guarded rows, `compare_summary.md` adds `Bigram Guard Recommendations` and `Bigram Guard Deltas` sections that compare NLL, bigram lift, rank lift, and top-5 preservation against the unguarded baseline while separating clean candidates from mixed top-k tradeoffs.
Rust char-LM classifier heads use RMS-scaled initialization by default; tune with `--head-rms`, and with `--mix-rms` for coherence scan/wave mixers.
Rust char-LM examples also add a learned smoothed train-token unigram prior before the softmax by default; pass `--head-prior none` to start without that prior, or `--head-prior bigram|learned-bigram` to seed the head with a previous-token train bigram prior.
The residual context logits before that prior can be scaled with `--head-residual-scale`; set it to `0` for a pure-prior ablation, or try values above `1.0` when probing whether context can push beyond the frequency or bigram baseline.
Add `--bigram-topk-guard F --bigram-topk-guard-k N` when the residual context should learn beyond the bigram baseline without freely destroying the previous-token top-k candidate set.
The coherence scan and coherence wave char-LM examples accept the same bigram prior and top-k guard flags; for tiny smoke runs, keep `--memory <= --steps` so their rolling context windows fit the shortened sequence.
Validation summaries also include smoothed train-token unigram and bigram baselines, target-token rank, and unigram/bigram lift metrics, including bigram top-5 overlap; use `--compare-summary-sort-metric final_bigram_logprob_lift|final_bigram_rank_lift|final_top5_bigram_overlap` when a sweep should be shortlisted by conditional-baseline lift or rank/top-k preservation instead of raw NLL.

Compare several char-LM runs with:

```bash
PYTHONNOUSERSITE=1 python3 -S -s tools/compare_char_lm_runs.py --aggregate --curves --params 5 models/runs/<baseline> models/runs/<scan> models/runs/<wave>
```

---

### SpiralReality demo
**File:** `examples/spiral_reality_demo.rs`

```bash
cargo run -p st-zeta --example spiral_reality_demo
```

---

### GNN graph regression demo
**File:** `examples/modelzoo_gnn_graph_regression.rs`

```bash
cargo run -p st-nn --example modelzoo_gnn_graph_regression
```

Uses `ZSpaceGraphBatchRegressor` so normal `DataLoader::batched(N)` graph samples are read out as `(N, target_cols)` graph predictions. The companion `examples/gnn_trainer_band_trace_demo.rs` writes `gnn_band_trace.json` with per-graph readout boundaries, a probe-batch readout MSE, validation-wide readout MSE, and band-replay coefficient traces from `ModuleTrainer::train_epochs_loader()`.
Use `tools/run_gnn_band_trace_sweep.py --batch-values 1,2 --lr-values 0.03,0.05 --top-k-values 1,2 --seeds 9,10 ...` to produce a `compare.md` with per-run rows, group averages, top validation candidates, and roundtable-axis deltas across shape, training, and schedule axes. The generated `sweep.json` mirrors those candidates and deltas under `comparison`; run `tools/run_gnn_band_trace_sweep.py --follow-up-from <previous-sweep> --seeds 21,22 ...` to replay the best validation-wide GNN schedule with fresh seeds or explicit axis overrides. Add `--follow-up-neighborhood --follow-up-neighborhood-axes lr,top_k,bottom_k` when the next probe should test a compact local schedule neighborhood around the winning candidate; `Follow-Up Result` reports whether the new best improved, matched, or regressed, `Follow-Up Promotion` names whether to carry forward the new best or keep the source candidate, and `--follow-up-fail-on-verdict regressed,unknown` adds `Follow-Up Gate` while making matching verdicts fail the sweep. Subsequent `--follow-up-from` runs consume `Follow-Up Promotion` by default, while `--follow-up-source top-candidate` forces the raw top validation candidate instead; `Next Follow-Up Command` gives the next chained command template with `NEXT_RUN_ROOT` left for the caller, and `config.follow_up.lineage` records parent sweep/run-root and generation for chained follow-ups.
Multi-seed comparisons add `validation_mse_stddev`, min/max/spread, `validation_stability_score`, and `validation_stability_status` to the candidate records; `Stable Validation Candidates` ranks by average validation MSE plus stddev so a reproducible schedule can be spotted even when the raw top average is noisier. `validation_readout_nmse` and `avg_validation_readout_nmse` sit beside the raw MSE columns, normalizing by target mean-square energy so fresh-seed shifts can be separated from schedule changes.
Completed follow-up comparisons add `Follow-Up Chain`, `Follow-Up Ancestors`, `Follow-Up Chain Guidance`, and `Guided Next Follow-Up Command`, keeping parent verdicts, promotion actions, selected candidate sources, verdict streaks, candidate stability status, raw/NMSE replay deltas, and a replay/neighborhood command template with explicit placeholders visible without reopening each older run by hand. `improved` candidates that are still `single_seed_probe` or `volatile` are routed to fresh-seed confirmation before promotion is continued; repeated `volatile` improvements switch to `widen_stability_search`, adding `--follow-up-neighborhood --seeds NEW_SEEDS` so the next run searches nearby schedules instead of replaying the same unstable point forever. If that neighborhood pass still improves but remains volatile, guidance switches to `increase_sample_budget` and adds doubled `--epoch-values`, `--train-graph-values`, and `--validation-graph-values`; if the larger-budget result regresses on average but surfaces a more stable top candidate, `review_stability_tradeoff` reruns that top candidate with fresh seeds before discarding it. If that review only produces a tiny average improvement while the source remains more seed-stable, `keep_source_stability_guard` keeps the stable source and asks for another fresh-seed confirmation instead of promoting the volatile best. Once repeated improvements are stable, `explore_stable_neighborhood` adds `--follow-up-neighborhood --seeds NEW_SEEDS` so the next run searches nearby schedules from the promoted stable anchor; when that search finds a stable winner, `confirm_stable_promotion` reruns the promoted schedule with fresh seeds before broadening again. When a neighborhood includes the source schedule, `source_replay_*` fields show whether a regression is really a schedule loss or just a fresh-seed shift; `review_seed_shift_neighborhood` repeats the source-anchored neighborhood if the best neighbor only loses to the previous source but matches or beats the replayed source. If raw MSE regresses but source replay NMSE does not, `review_target_scale_shift` widens validation graph values with fresh seeds before treating the run as a schedule loss. If seed-shift evidence is volatile, `increase_seed_shift_validation_budget` drops the neighborhood and doubles validation graph values before another promotion decision; repeated seed-shift regressions use `widen_seed_shift_neighborhood` with fresh seeds to avoid replaying an old seed surface, and persistent seed-shift regressions use `audit_seed_sensitivity` to pause schedule search and remeasure the source with a broader seed list plus validation budget. The generated run directory also contains `next_follow_up_command.sh`, which accepts those placeholders through environment variables.
Use `tools/run_gnn_threshold_grid.py --thresholds 1,1024 --batches 1,2 ...` when the question is whether tensor-utility reductions route to CPU or WGPU at a given size threshold; WGPU rows preflight once by default, and failed/preflight rows are preserved in `grid.json` instead of erasing successful CPU evidence.

---

## Go bridge POC

**File:** `examples/go_bridge_poc/README.md`

---

## COBOL integration

**File:** `examples/cobol/st_dataset_writer.cbl`

---

## Planned examples

- Vision: MNIST classifier (conv + pooling stack)
- NLP: text classification (LanguageWaveEncoder + projector + head)
- RL: bandit / PPO training loops
- Export: JSON artefacts today; ONNX parity demos (CPU-first) planned
