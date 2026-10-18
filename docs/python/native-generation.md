# Native language generation

Development API, newer than the published 0.4.32 wheel. Python transports IDs;
Rust owns the decoder, geometry, filtering, Z-Space control, EOS and seeded
selection. No Torch, NumPy, Transformers or remote code is needed on this path.
The optional HF path remains explicit comparison/interop, not a fallback.

```python
import json
from pathlib import Path
import spiraltorch as st

snapshot = Path("/local/gpt2-snapshot")
weights = st.SafeTensorArchive(snapshot / "model.safetensors")
base = st.NativeCausalLm.from_gpt2((snapshot / "config.json").read_text(), weights)
# Optional: base.with_geometry(artifact), or restore a pinned learned model:
# model = st.NativeCausalLm.from_training_checkpoint(base, checkpoint_json, checkpoint_id)
model = base
generator = st.NativeCausalLmGenerator(model, [15496, 11], json.dumps({
    "max_new_tokens": 16,
    "seed": 81,
    "eos_token_ids": [50256],
    "execution": {"kind": "kv_cache", "max_values": 16_777_216},
    "decoding": {
        "selection": "sample", "top_k": 40, "top_p": 0.9,
        "distribution": {"kind": "softmax", "temperature": 0.8},
    },
}))
while not generator.finished:
    step = generator.step()
    print(step.token_id, step.probability)
saved, pin = generator.to_json(), generator.checkpoint_id
restored = st.NativeCausalLmGenerator.from_json(model, saved, pin)
```

Use `{"kind": "zspace", "config": {"curvature": -0.04,
"temperature": 1.0, "repression_strength": 1.0, "last_token_repression": 0.5}}`
as `distribution` to select the existing `st-core::inference::generation_control`
contract. Entropy-target, temperature-bound and n-gram options are accepted in
`config`; no second Python formula or extra temperature stage is introduced.
Geometry inside the decoder and Z-Space output control are separate interventions.

## Selection contract

- Default: ordinary temperature-1 greedy, full vocabulary, no EOS IDs. `max_new_tokens` is required.
- Raw-logit top-k runs **before** control. A suppressed token cannot bring an excluded candidate back. `null` means full vocabulary; zero or values above the vocabulary are errors.
- Nucleus filtering runs **after** control, includes the first candidate crossing `top_p`, then renormalizes positive mass. `top_p=1` keeps all positive candidates.
- Descending logits/adjusted logits determine rank; exact ties use the lowest token ID, including signed zero. Rounded probability ties do not change greedy ranking.
- Sampling uses probabilities directly, not the older Z-Space reporting log-probability floor. Underflowed zero mass is never sampled. No Torch RNG equivalence is claimed.
- Step `probability` and `entropy` describe the final filtered distribution, not the likelihood of a deterministic greedy policy. `control_entropy` is before nucleus filtering. JSON also records raw/controlled top IDs, candidate counts, temperature and the optional Z-Space contract version.

## Cached execution

The example explicitly opts into Rust-owned K/V caching. Omitting `execution`
preserves full-prefix execution and the original v1 checkpoint serialization.
`{"kind": "full_prefix"}` also selects that default. Python does not manage a
Torch cache, attention math or geometry history.

The prompt is evaluated once. Subsequent appends compute one new query at its
original absolute position, reuse prior K/V, and project only the final row to
vocabulary logits. Frozen model and geometry identities travel with the context;
they cannot be changed underneath an existing cache. Row-local geometry evaluates
only the new row. Causal/history geometry retains its raw input prefix and replays
the existing Rust geometry operator, selecting the last output row. It is correct
prefix reuse, **not** a claim that every geometry operator has constant-cost
incremental execution. K/V transaction buffers also still copy prior storage.

For caller-controlled token insertion, use the same state independently:

```python
context = st.NativeCausalLmContext(model, [15496, 11], max_cache_values=16_777_216)
last = context.logits()       # NdTensor [1,1,vocab]
last = context.append(995)    # Admit and evaluate one token, then publish state.
print(context.token_ids, context.cache_values)
assert context.execution_backend == "rust_f32_cpu_prepacked_kv"
```

`max_values` / `max_cache_values` counts persistent **f32 values**, not bytes:
K/V plus retained geometry input histories. It excludes model weights, readout,
temporaries and concurrent old/new transaction storage; it is not an RSS cap.
Prefill still uses the square attention admission. Cached appends use one-query
attention admission, while sequence geometry keeps its own full-prefix bounds.
Unsupported or over-budget execution fails; it never silently switches backends.
Invalid/numerically failing appends leave tokens, logits and cache unchanged.

Cached generation checkpoints store execution policy and token history, not opaque
K/V buffers. Resume rebuilds the **original prompt prefill followed by chronological
appends**, preserving the cached numerical path and random position. This costs a
replay on the first resumed step. Simply re-prefilling the entire current history
would change the floating-point reduction path. Old full-prefix checkpoints remain
supported; older readers reject the new explicit execution field.

## State and bounds

Both modes use one unpadded sequence and the frozen f32 CPU model. This is not
arbitrary-HF support, a GPU generation path or a serving scheduler.
Tokenization/detokenization is external; use the imported model's tokenizer.
The requested execution bounds are admitted before copying prompt IDs.
There is no automatic cropping or position reset.

Caps: 1024 new tokens, 16384 total IDs, 1048576 vocabulary entries and 256 distinct
EOS IDs, subject also to the model's tighter position/activation/attention budgets.
The last emitted token is not itself evaluated: a prefix at the last position may
emit one token and then stop. N-gram comparison work is bounded.

Only **newly emitted** EOS stops the session; EOS wins over length if both occur
on the same step. Finished `step()` returns `None`. Numerical errors leave IDs
and random position unchanged. Sampling uses ChaCha20 from `rand_chacha 0.3`,
u64 seed expansion, one u64 draw per accepted token, upper 53 bits mapped to
`[0,1)`. The checkpoint seals the algorithm identifier and history; random
position is derived from generated count, not an editable extra cursor.

Checkpoints are capped at 1 MiB, pinned by byte SHA-256 and bound to the supplied
base **and exact geometry artifact**. They omit large model weights. Continuation
also requires the same [readout accumulator](native-causal-lm.md#explicit-readout-arithmetic).
The f32 default keeps the old checkpoint bytes; f64 is explicit and sealed.
They are not interchangeable even when frozen weights and geometry are identical.
Continuation is qualified on the tested runtime, not promised bit-identical across CPU/compiler
versions. Hashes are integrity pins, not provenance signatures; re-sealing valid
arbitrary history does not prove that it was generated.

`from_training_checkpoint` validates nested model/geometry/optimizer ownership
and optimizer clocks, but does **not** read or verify the corpus or prove training
completion. Use [`NativeCausalLmTrainer.from_json`](native-corpus-training.md)
for a data-verified training resume.

## Offline example

[`native_geometry_generation.py`](../../bindings/st-py/examples/native_geometry_generation.py)
accepts `--model-dir`, a JSON token list `--prompt-json`, a JSON configuration
`--recipe-json`, and a new `--output` directory. Optionally supply
`--training-checkpoint` with `--training-checkpoint-id`, or `--artifact` with
`--artifact-id`. `--stop-after N` pauses at an absolute generated-token count.
Resume with `--resume generation.json --resume-id sha256:...`, the same model
and geometry source, and **no** replacement prompt or recipe.

Output is a new POSIX-0700 directory containing step reports, a generation
checkpoint and a summary. Treat IDs, text and checkpoints as private experiment
data. The example does not promise power-loss durability or public serving.
The summary records the effective `execution`, including `full_prefix` when
the recipe omits it; a KV run is not reported as a no-cache run.

Rust exposes the same session and context through
`st_nn::causal_lm::{NativeCausalLmGenerator, CausalLmGenerationConfig, NativeCausalLmContext}` and pure
selection through `st_core::inference::token_decoding`. Successful execution and
resume are not evidence of language-quality uplift.

The [fixed local KV qualification](../../benchmarks/results/2026-10-18-native-lm-cache/README.md)
separates geometry-bearing numerical checks from ordinary, geometry-free CPU
timing against PyTorch. Cached/full-prefix logits are compared with a declared
tolerance; exact cached resume does not imply bit-identical execution between
different kernel shapes, runtimes or hardware.
