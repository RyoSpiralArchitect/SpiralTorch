# Document-Safe Inputs For Geometric Language Learning

[HF geometry Trainer](hf-geometry-training.md) | [Language MLOps](../language_mlops.md)

**Source candidate after 0.4.31.** `HfGeometryDataset` supplies an ordinary HF map
dataset, using Rust `CausalDocumentWindows` for every row boundary. It does not
concatenate unrelated texts: a Fractional history filter sees one document
window, never the tail of another document packed into that same row.

```python
import json
from spiraltorch import HfGeometryDataset
from transformers import default_data_collator

data = HfGeometryDataset.from_texts(tokenizer, documents, context_length=128)
print(json.loads(data.plan.to_json()))
# Pass data as train_dataset and default_data_collator as data_collator to
# hf_geometry_trainer_class(). No padding or further label rewriting is needed.
```

`documents` is a sequence of strings, not one bare string. Tokenization remains
the external tokenizer's responsibility; padding/truncation are disabled, and
`add_special_tokens=True` is explicit and can be changed by the caller. The
constructor also accepts pre-tokenized documents: `HfGeometryDataset(rows, 128)`.
Encoded documents are copied into immutable tuples; returned row lists are fresh.
This is an in-memory adapter, not a streaming corpus loader.
Spawned dataloader workers serialize the frozen documents and reconstruct the
same plan through Rust. Boolean NumPy/Torch scalars are not valid token IDs.

## Exact Row Policy

- Every window has exactly `context_length >= 2` tokens. Full windows are
  disjoint within each document; incomplete tails are dropped **per document**.
- Documents shorter than a complete window contribute no rows. They cannot be
  joined to form a row. A dataset with no full windows fails explicitly.
- Attention masks contain only ones. Labels equal input IDs except the first
  label, which is `-100`: the unpredicted context token must not inflate the
  causal loss denominator. A window contributes `context_length - 1` targets.
- History resets at every window boundary. This is a fixed-context experiment,
  not full-document recurrence or KV-cached decoding. Tail and context-token
  omissions must be included when describing coverage.
- Rust bounds both document and window counts to 1,048,576; all lengths/indices
  use portable u32 values. Browser JSON recipes additionally have a 16 MiB cap.

The native plan reports `total_tokens`, `dropped_tokens`, `target_tokens` and the
ordered `(document, start, length)` windows. Its `plan_id` binds **only lengths,
ordering and policy**, not text, tokens, tokenizer, split membership or rights.
Use the existing `hf_tokenized_dataset_identity_report` to pin actual rows;
keep source provenance and train/evaluation separation with the run owner.

## Same Rust Contract From Three Clients

```python
import spiraltorch as st
plan = st.CausalDocumentWindows([0, 3, 5, 12], 4)
assert plan.window(0) == (2, 0, 4)
assert len(plan) == 4
```

Rust: `st_nn::language_windows::CausalDocumentWindows::from_lengths(lengths, 4)`.
WASM `nn`: `CausalDocumentWindows.createJson(JSON.stringify({document_lengths:
[0,3,5,12], context_length:4}))`, then `window(i)`, `length`, `planId`, `toJson()`.
`createJson` consumes a recipe, not the output state JSON. Neither client adds a
second implementation of the window-selection rule. The shared synthetic
fixture checks native/Python/WASM identity and boundary agreement.

## Offline Pretrained-Model Example

[`hf_geometry_finetune.py`](../../bindings/st-py/examples/hf_geometry_finetune.py)
uses a local HF base, a separately pinned portable initial-geometry artifact,
explicit train/evaluation files, this dataset and the public HF geometry Trainer.
It measures a frozen bare-base reference, the initial geometric reference, and
the trained geometry, verifies frozen base weights, and reconstructs geometry
for inference after saving. Continuation requires both the checkpoint ID and
`--expected-dataset-id` retained from the previous run, before reusing its cursor.

The older `spiral-hf-finetune` full/LoRA CLI keeps its existing concatenated-block
policy. Do **not** attach history geometry to those packed blocks without
changing that data route. This example is an explicit safe route, not a silent
change to existing full/LoRA training behavior or an authenticated run executor.
All model/tokenizer loads are local-only with remote code disabled. Complete
HF companion integrity and an unchanged training recipe remain caller-owned.

Public qualifications retain metrics, hashes and replay instructions. Raw
corpus, learned artifacts, checkpoints and generated text remain local. A small
pretrained-model workflow probe does not establish general quality or speed.

See the [qualification and replay record](../../benchmarks/results/2026-10-18-hf-document-windows/README.md)
for the actual offline GPT-2 run, retained negative resume result and corrected
fresh-process continuation.
