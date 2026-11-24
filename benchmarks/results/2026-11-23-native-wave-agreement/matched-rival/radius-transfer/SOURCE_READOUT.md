# Native CPU readout: source-to-wheel check

The source-built `NativeCausalLm` again exposes the training wheel's
`with_forward_readout_storage` hint. It selects a weight layout, not a
different forward or backward precision. The frozen study still uses its
original wheel; the new development wheel is not a replacement for it.

| Hint | No backward cache | Warm cache, f32 forward |
| --- | --- | --- |
| `original_packed` | Original weights | Original weights |
| `prefer_cached_transpose` | Original weights | Borrow a transpose if present, or use an existing panel for 1-3 rows |
| `auto` (default) | Original weights | Use an existing panel for 1-3 rows; borrow a transpose only for the qualified Apple Silicon shapes |

The small automatic transpose window is 6-7 rows, width 96-128 and vocabulary
8,192-16,385. The only two larger cases are exactly 7 rows with width/vocabulary
384/32,768 or 768/50,257. Other cases fall back to the packed head. A fully f64
forward does not use an f32 cache kernel. The first valid vocabulary pullback
initializes one auxiliary layout shared across clones; inference never builds
one. The hint does not enter checkpoint or adapter identity.

## What matched

Rust tests observe the actual kernels and compare output bits for cold and
warm models, short rows, tails and precision changes. The installed Python
extension also checks that training resumes to identical checkpoints when the
process-local hint is reset.

On the two-layer, width-16, **artificial** decoder, the new wheel reproduced all
8 normalized steps (R=1 and R=4) and all 16 next-token score rows. The raw
comparison covered 19 files: two gradient tapes, two journals with update
receipts, ten state blobs, four full-vocabulary logit files and the endpoint
manifest. Every byte matched the sealed original. Only the extension and
therefore the plan fingerprint differ; the original wheel, plan and
observations were not overwritten. The artificial advance rule remains false.

The historical GPT-2 margin study is separate: replaying its saved observations
with its archived source reproduced 48 decoder calls / 192 local maps without
any model calls. The live checkout correctly rejected that old plan because
it pins the Rust sources. No new pretrained-model training, held-out scores,
GPU execution, timing comparison or release is part of this check.

Run from this directory with the newly built, installed CPU wheel:

```sh
"$VENV/bin/python" -I -B check_source_storage.py "$SEALED_TOY" "$EXTENSION_SHA256" "$NEW_OUTPUT"
```

The verifier accepts only the original artificial plan and result digests,
requires a fresh output directory, and recomputes both summaries from the raw
files. It has no pretrained-model argument. The development wheel built with
Rust 1.99.1 has SHA-256
`84630d4b9cc26fd1c914e18917741ef92c2b472a4e119c360aa39f7b9f6e9212`;
the installed extension has SHA-256
`25114241da9a58d00724ddf6dd17ae08db7daafd89f77dffa224335a4ea95cf2`.
The resulting comparison manifest has SHA-256
`d8d1ae88a08ef7e7de7f9a78977a70fc5bb06e08cfd4309e1c5c4675b9a2a7c2`.
