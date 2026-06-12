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
cargo run -p st-nn --example modelzoo_llm_char_finetune -- models/samples/spiral_demo_en.txt --val-fraction 0.1 --eval-samples 256
```

Writes `run.json`, `metrics.jsonl`, `summary.json`, samples, and weights into the selected run directory.

Compare several char-LM runs with:

```bash
PYTHONNOUSERSITE=1 python3 -S -s tools/compare_char_lm_runs.py --curves models/runs/<baseline> models/runs/<scan> models/runs/<wave>
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
