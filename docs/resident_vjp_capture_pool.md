# Bounded owning VJP capture reuse

`ResidentGraphAutograd::backward` still returns immutable owning input and
parameter gradients. It still copies graph scratch into these outputs, checks
every value and preserves one whole-VJP failure guard. This change removes
repeated output-buffer and bind-group creation when old captures are no longer
observed; it does not eliminate gradient copies or introduce lazy host readback.

The private `PreparedCapture` retains at most four complete output versions,
within 32 MiB of combined values (including empty-buffer padding) and guards.
Reuse requires exclusive ownership of **every** member's storage, with no weak
storage owner and no extra shared/weak flag owner. One retained input-gradient
view therefore pins the entire version, including unrelated parameter values
and their shared guard. Independent members cannot be recycled separately.

The reused guard is cleared by an encoder command before the capture pass.
Every existing capture dispatch then checks the same source values and ORs the
same upstream guards. No shader math, pass grouping, VJP normalization or
optimizer rule changes. Already submitted tensor consumers and snapshot copies
precede reuse on the same queue; their values and failures remain immutable.
Callers of this private encoder method must retain its returned tensors until
submission, as the autograd implementation does.

Busy and oversized captures allocate separately. They do not wait for a free
slot, evict observable results, alias scratch or fall back to CPU. The bound is
per autograd workspace, not a global cap on user-held GPU memory or driver
metadata. `PreparedCapture::encode` remains an unpooled operation for explicit
EMA/accumulator snapshots; only the repeated autograd VJP path opts into reuse.

For the benchmark's two-VJP loop, releasing both gradient containers after an
update allows two versions to serve subsequent iterations. With 25 parameters,
each reused VJP avoids creating 27 buffers and 26 output bind groups. These are
structural allocation counts, not a claim about elapsed time, physical memory
traffic or GPU kernel speed.

Verification includes finite and inherited failure recovery, signed zero,
empty/scalar/tail shapes, detached views, pending readbacks, zero-multiplied
invalid consumers, weak owners, spills beyond four held versions, no-retention
behavior and use after workspace destruction. A shared native/browser fixture
additionally runs nine VJPs with pending reads and seven held versions.

See [host diagnosis](resident_learner_host_profile.md) for optional enqueue
measurements. Ordinary native/browser/PyTorch intervals, not instrumented host
phase totals, decide whether this reduces end-to-end training cost.
