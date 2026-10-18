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

## State and bounds

One unpadded, full prefix is recomputed per step using the frozen f32 CPU model.
This is not arbitrary-HF support, a KV cache, a GPU generation path or a serving
scheduler. Tokenization/detokenization is external; use the imported model's
tokenizer. The largest evaluated prefix is admitted before copying prompt IDs.
There is no automatic cropping, position reset or cache substitution.

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
is qualified on the tested runtime, not promised bit-identical across CPU/compiler
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

Rust exposes the same session through
`st_nn::causal_lm::{NativeCausalLmGenerator, CausalLmGenerationConfig}` and pure
selection through `st_core::inference::token_decoding`. Successful execution and
resume are not evidence of language-quality uplift.
