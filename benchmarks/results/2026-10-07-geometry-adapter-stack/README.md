# Public Geometry Adapter Stack: Pretrained Connection

The [public placement API](../../../docs/geometry_adapter_stack.md) runs WaveGate,
Topos, anchored elliptic geometry and fractional history after GPT-2 MLPs 0-3.
The native binary is unchanged from the active-support optimization; only the
Python placement module and public exports are new. Rust owns all geometric
operators and their derivatives.

## Executed Result

- Local pretrained GPT-2, training-only Pride blocks, CPU float32, two Torch
  threads, hidden shape `[2,128,768]`, and 12,294 trainable parameters.
- Three updates by explicit module insertion and three through the public
  `GeometryAdapterStack.attach` API match bitwise for loss, every gradient,
  adapter state, named Adam state and RNG.
- A serialized midpoint from the explicit-insertion branch is reloaded into
  fresh adapters and Adam state, then continued for one identical public-API
  update. This is in-process continuation, not a fresh-process portability test.
- One process, seven executed updates, three unique trajectory updates. At
  update three all twelve trainable parameter tensors have nonzero gradients.
  Base parameters/buffers and base registration remain unchanged; no base
  parameter gets a gradient. The API does not freeze the base implicitly.

Tiny random GPT-2 and Llama tests independently exercise the same mixed families,
state continuation, keyword interfaces, ownership and failed-hook cleanup. The
pretrained run here is GPT-2 only. Nonzero gradients are connectivity evidence,
not an advantage over ordinary controls. There are no heldout scores, generated
text or throughput measurements. Losses use different training batches.

## Archive

`report.json` contains every numeric update record, model/data/source hashes and
private state/checkpoint receipts. `runtime-sha256.json` records the 72-file
package. `validation.json` retains validation status, frozen client hashes and
the initial incomplete-copy failure before training. `SHA256SUMS` seals this
public inventory. Raw states, weights, corpus text and raw logs remain private.

The public-only tests check receipt consistency, not the hidden tensor contents.
Actual tensor-byte comparisons were executed by the recorded offline example;
its states remain available locally. Reproduction commands and restrictions are
in the linked guide. No existing study or failure artifact was rewritten.
