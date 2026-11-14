# Prospective GPT-2 QKV objective pilot

Status: **protocol only; no model execution or result**. This comparison is
held until the post-merge `main` CI for the QKV objective and its Rust 1.100
repair is green. The first failed attempt, if any, must be retained rather
than retried for a favorable score.

`protocol.py` deterministically selects 24 training and 48 evaluation
records from the synthetic field-extraction grid. The 72 selected triples
exclude all 288 triples used by the earlier pilot and affine-control study.
Each split is balanced by queried field, answer value, field position, and
prompt template. Three hashed training orders reuse the same 24 records;
they are not independent data replications.

Both arms start from the same pinned GPT-2 small and frozen LoRA + WaveGate
checkpoint. Only layer-11 QKV is updated. Each arm/order gets 24 updates,
with identical learning rate, AdamW defaults, weight decay, gradient clip,
answer mask, and record order. The sole intended difference is ordinary
cross-entropy versus the Rust-owned top-rival objective (`rivals=4`,
`mix=0.5`). Initial and final scores use the same fixed evaluation set.

The primary contrast is the difference in counts of evaluation answers for
which **every gold token is top-1 under teacher forcing**. Secondary results
are unsmoothed NLL, answer-level NLL, token-level top-1 count, and per-answer
win/tie/loss. This is not free-running generation, a geometry ablation, or
evidence of general language quality.

Model-free preflight:

```bash
python3 -I -S -B benchmarks/results/2026-11-14-qkv-rival-pilot/test_protocol.py -v
python3 -I -S -B benchmarks/results/2026-11-14-qkv-rival-pilot/protocol.py
```

The protocol lists SHA-256 pins for the source model, tokenizer, source
checkpoint, predecessor generators, record sets, and orders. The runner and
encoded-input manifest are not yet implemented; publishing this protocol
does not claim that a run took place.
