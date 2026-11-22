# Which adapter learns from the complete answer?

This is an execution and gradient check on a tiny, artificial decoder. It does
not load GPT-2 small, use its agreement sentences, or test generalization.
Keep any previous real-model measurement separate.

## The controlled difference

All three runs start with identical base, nonzero LoRA A/B and WaveGate gate/bias
on the first attention output. They see the same three terminated-answer
examples, twice in the same order, at SGD rate 0.125 and vocabulary tile size
11. Only the updated group changes. Unselected adapters remain in the forward
function and the input derivative. Joint SGD uses the same pre-step weights for
both groups; it is not two successive optimizers.

The forward-only losses below are on the **same training examples**, before or
after six updates. Each column has a different objective; do not compare or
pool columns as if they were the same task.

| Updated weights | Set NLL | Gold/menu mixture | Accepted-subset NLL |
| --- | ---: | ---: | ---: |
| Initial (shared) | 4.568401 | 2.570731 | 0.114927 |
| LoRA only | 4.573523 | 2.561706 | 0.111868 |
| Geometry only | 4.568263 | 2.546018 | 0.114526 |
| Both | 4.573349 | 2.556907 | 0.113486 |

There is no uniformly best mode here. In particular the absolute set loss
increased in two arms. No independence, speed, pretrained efficacy or
general learning claim follows from these three examples.

For a matched-distance follow-up, see [six equal steps](LINE_OWNER_DISTANCE.md).
Holding the SGD rate fixed did not hold the movement of the weights fixed.

## What checks the implementation

- Rust compares joint with the disjoint same-state VJPs on all three losses,
  all three readout policies, and dense/one-row/two-row head tiles. The decoder
  block forward/replay counters are the same as for one selected group.
- An independent scalar oracle scores every token including the stop by full
  prefix. Perturbing an A, B, gate and bias coordinate checks each objective's
  gradient against a central difference (epsilon 0.002, absolute error < 0.00015).
- A frozen-but-attached WaveGate transmits the LoRA derivative. Removing it
  gives a different gradient. A model without geometry can still train LoRA.
- Rust v1 and its typed geometry receipts are unchanged. A v2 checkpoint binds
  plan bytes, arithmetic and both origins; only selected weights are stored.
  Re-pinning a changed LoRA layout or injecting weights for a frozen owner is
  rejected. Same-engine restored suffixes match their uninterrupted runs.
- CPython, Node and Chrome each compare explicit SGD with the Rust session.
  WASM imports each native update-2 checkpoint; its update-4 checkpoint goes
  back to a fresh Python process for the final two examples. Across engines,
  use absolute tolerance 0.000003 on numerical values, not assumed
  bitwise equality. For this fixture the maximum WASM/native per-step loss
  difference was 1.41e-7 or less; final parameter/logit difference < 2.01e-9.

The [Python fixture](../../../../../bindings/st-wasm/tests/native_line_owners_fixture.py)
builds the three-mode reference with the native CPU extension and no external
ML library. Use `issue NEW_PATH` and `resume NEW_PATH [RETURNED_PATH]` in
separate processes. The [client guide](../../../../../bindings/st-wasm/tests/native_causal_lm.md)
describes running the Node/browser tests. Node argument 5 (after the glue,
selected-output path and v1 return path) exports the three v2 checkpoints.
The browser's `?capture` page exports the same data through
`native_line_session_capture.py DOM NEW_V1_PATH NEW_V2_PATH`.
