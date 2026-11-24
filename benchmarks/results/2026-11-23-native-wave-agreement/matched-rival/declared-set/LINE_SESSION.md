# Restarting complete-answer geometry training

This note describes the v1 geometry-only contract. The optional v2
[owner comparison](LINE_OWNERS.md) trains LoRA, geometry or both; it does not
change v1 checkpoints or the geometry-only step.

The Rust line session owns a fixed ordered list of prompts and terminated
answers. Each example specifies absolute set NLL, a gold/menu mixture, or
conditional accepted-subset NLL. Each call to `step` performs **one ordinary
geometry-only SGD update**, not a document-CE or a finite-decrease step.
An attached LoRA and base stay frozen. There is no shuffle, momentum,
validation pass or hidden replay of earlier examples.

## What is pinned

The SHA-256 of the exact UTF-8 plan binds all IDs (including stops), order,
losses, tile cap, epoch count and rate. Even extra whitespace changes its ID.
The caller retains the plan and the original adapted model. Only the cursor,
current geometry JSON, owner IDs and readout arithmetic go in the checkpoint.
Omitting raw examples is not a privacy guarantee about learned parameters.
Restoring requires the checkpoint's independently retained pin. A matching
hash authenticates the bytes against that pin, **not** who produced them or
whether training was actually performed. The starting geometry, frozen LoRA,
base (including key-lag architecture), readout precision and all frozen geometry
values must agree. Updated recipes, training flags and impossible cursors are rejected.

Admission checks every example's exact trie and cumulative decoder/readout
limits before the first forward. A plan is at most 1 MiB, 4,096 examples,
131,072 prompt/content/stop IDs and 1,048,576 updates. Existing per-example
model limits still apply. A step failure does not advance the session.
Rust/Python can stage an interval of up to 1,024 updates and publish one
checkpoint at a new path before committing the live cursor.

## Actual transfer

Four tiny two-layer hosts: ordinary/key-lag attention, each with WaveGate or
trainable history gain. One shared [three-example plan](../../../../../bindings/st-wasm/tests/native_line_session_plan.json),
two epochs, one-row readout tiles. All losses, stop IDs and updates were fixed
in advance. No pretrained weights, held-out data, GPU or external ML runtime.

| Boundary | What the checks established |
| --- | --- |
| Direct old CPython wheel vs native session | 24 updates matched the six-step direct SGD trajectories, including reports, post-update IDs and final logits. |
| Fresh CPython process at update 2 | 4 checkpoints restored; the 16 remaining step reports/owners and exhausted checkpoint pins matched uninterrupted learning exactly. |
| Browser / Node | Each checked 24 session updates against its own direct geometry SGD, 16 same-engine resumed steps and 16 steps resumed from Python checkpoints. Each rejected 24 invalid requests. |
| Python (step 2) to WASM (step 4) and back to Python (step 6) | All 4 inbound and return checkpoints retained their exact payload and SHA-256. The local Chrome download and Node output were byte-identical; final native geometry IDs matched the uninterrupted native reference for all four. |

Do not infer portable bitwise arithmetic from portable checkpoint bytes. Of
the eight suffix losses exported at steps 5 and 6, one differed from direct
native SGD by 1 f32 ULP (lag/history, step 5); all other suffix loss words
matched. The maximum absolute returned-parameter difference was zero for
these four hosts; the largest WASM-vs-native final logit difference was
1.49e-8. These are continuation checks, not evidence for validation quality,
long-run stability or convergence. [Per-host results, pins and raw input
hashes](line_session_results.json).

## Reproduce

From `bindings/st-wasm/tests`, using fresh output paths:

```sh
"$OLD_PY" -IB native_line_session_fixture.py reference "$DIR/direct.json"
"$NEW_PY" -IB native_line_session_fixture.py issue "$DIR/direct.json" "$DIR/bridge.json"
"$NEW_PY" -IB native_line_session_fixture.py resume "$DIR/direct.json" "$DIR/bridge.json"
node native_causal_lm_node.mjs "$GLUE" "$DIR/selected.json" "$DIR/from-node.json"
"$NEW_PY" -IB native_line_session_fixture.py take-back "$DIR/bridge.json" "$DIR/from-node.json" "$DIR/node-return.json"
```

The checked-in native bridge is the same as the generated one. Open the
local browser demo and download its four step-4 checkpoints, or capture
`native_causal_lm.html?capture` with headless Chrome `--dump-dom` and run
`native_line_session_capture.py DOM NEW_DOWNLOAD_JSON`. Repeat `take-back`
with the browser output and a fresh destination. From this results directory:

```sh
python3 -ISB line_session_audit.py "$PLAN" "$DIR/direct.json" "$DIR/bridge.json" "$DIR/from-node.json" "$DIR/from-chrome.json" "$DIR/chrome-return.json" "$DIR/audit.json"
```

The audit consumes only saved outputs; it never reruns a decoder. Rust also
compares every cursor cut (0 through 6) under three arithmetic policies,
including frozen-LoRA retention and a mid-segment fault that must not
publish or advance. See [public APIs](../../../../../bindings/st-wasm/tests/native_causal_lm.md#resume-complete-answer-training).
