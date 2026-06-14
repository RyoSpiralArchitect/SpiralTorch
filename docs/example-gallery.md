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
Rust char-LM classifier heads use RMS-scaled initialization by default; tune with `--head-rms`, and with `--mix-rms` for coherence scan/wave mixers.
Rust char-LM examples also add a fixed smoothed train-token unigram prior before the softmax by default; pass `--head-prior none` to start without that prior.
The residual context logits before that prior can be scaled with `--head-residual-scale`; try values above `1.0` when probing whether context can push beyond the frequency baseline.
Validation summaries also include a smoothed train-token unigram baseline, target-token rank, and context-lift metrics, which make it easier to tell whether the model is beating frequency-only prediction or merely drifting away from uniform output.

Compare several char-LM runs with:

```bash
PYTHONNOUSERSITE=1 python3 -S -s tools/compare_char_lm_runs.py --curves --params 5 models/runs/<baseline> models/runs/<scan> models/runs/<wave>
```

---

### SpiralReality demo
**File:** `examples/spiral_reality_demo.rs`

```bash
cargo run -p st-zeta --example spiral_reality_demo
```

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
