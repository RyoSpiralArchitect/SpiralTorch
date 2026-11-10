# Paired projection language study

A small, CPU-only GPT-format model tests two distinctions: absolute versus
anchored WaveGate coordinates, and equal versus guarded two-example SGD. Its
seeded weights are partly structured to provide label and continuation cues.
This is **not** pretrained GPT-2, a throughput benchmark, or evidence of
general language capability. No downloads, tokenizers, NumPy or PyTorch.

```sh
python3 benchmarks/paired-prompt-contrast/projection_language_fixture.py /tmp/st-pair-input
cargo run --offline -p st-nn --example paired_projection_language -- train /tmp/st-pair-input /tmp/st-pair-train
cargo run --offline -p st-nn --example paired_projection_language -- score /tmp/st-pair-input /tmp/st-pair-train /tmp/st-pair-scores
python3 benchmarks/paired-prompt-contrast/projection_language_audit.py /tmp/st-pair-input /tmp/st-pair-train /tmp/st-pair-scores /tmp/st-pair-audit
```

All stage outputs require new paths. The driver accepts only the pinned synthetic
fixture. The commands above do not impose OS-level isolation or a wall-time cap.
The training stage opens no query or evaluation labels. For each of 18 pairs, it
supervises both answers **including their terminal LF**. Only the head row mixes
full-vocabulary and reciprocal two-class CE; subsequent rows use full CE.
It combines the two per-example gradients, not a token-count-weighted batch.
The absolute equal update supplies the actual Euclidean length for all other
arms. A predeclared numerical refusal retires that arm; no fallback or partial
endpoint scoring. Other errors, including losing the reference, stop the run.

The score stage restores the immutable endpoints, builds candidate documents
from the training menus, and ranks by **summed** full-line NLL. The native
answer scorer returns a mean per target, so the number of content tokens **plus
one** matters here. Independently retained unmodified prefix logits permit
readback of every candidate, including its stop. Only the final audit joins
the 18 gold answers. A new word order is not a new set of labels.

### Shared-prefix stop/continue control

For a source-built CPU Python wheel that exposes `geometry_line_menu_gradients`,
`shared_prefix_study.py` compares gold CE with a 50% whole-menu objective.
Every answer starts with ID 23; only the continuation and the stop distinguish
them. The fixture uses the same synthetic weight generator, **not** a pretrained
model. Both arms use anchored geometry, the same 12 training prompts and matched
step lengths. The scorer sees the 12 new-order prompts without gold labels.

```sh
cd benchmarks/paired-prompt-contrast
F=$(python3 shared_prefix_study.py prepare /tmp/st-lines-fixture)
T=$(python3 shared_prefix_study.py train /tmp/st-lines-fixture "$F" /tmp/st-lines-train)
S=$(python3 shared_prefix_study.py score /tmp/st-lines-fixture "$F" /tmp/st-lines-train "$T" /tmp/st-lines-scores)
python3 shared_prefix_readback.py /tmp/st-lines-fixture "$F" /tmp/st-lines-train "$T" /tmp/st-lines-scores "$S" /tmp/st-lines-audit
```

The printed outputs are exact-byte input/worker hashes, not paths. Use empty
destinations (the stages create their own directories). Read the separate
conditional-NLL and full-line accuracy gates in `summary.json`. Shell commands
alone do not enforce access separation or CPU/time limits.

Read `summary.json` for complete vs refused arms, actual vs first-order
descent, endpoint scores and the separate mechanism/quality gates;
`steps.json` and `joined.json` preserve the pair and query denominators.
