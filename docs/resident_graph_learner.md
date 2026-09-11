# Loss-Independent Resident Learning

`InferencePlan::compile_graph_learner_wgpu` connects the NN graph's exact VJPs
to explicit weighted SGD without mapping gradients or parameters to the CPU.
It shares Rust graph preparation, derivatives, guarded tensor operations and
atomic parameter commits with the existing resident paths. Python and WASM
only expose that implementation; they do not reconstruct update semantics.

For reusable GPU-side cotangent arithmetic from Python or WASM, see the
[public pointwise plans](resident_pointwise_clients.md). They compose directly
with `backward` without returning the seed to host memory.

The frozen [autograd workspace](resident_graph_autograd.md) remains available
without optimizer-sized scratch. The existing mean-MSE+SGD workspace keeps its
prepared fast path. This learner allocates optimizer scratch, but no MSE target
or loss reduction. It does not replace `ModuleTrainer`, hypergrad, distributed
sync, clipping, roundtable hooks or adaptive learning-rate policy.

## Python

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [2.0])))
model.add(st.nn.Relu())
plan = model.inference_plan([2, 2, 1]).fuse_pointwise()
learner = plan.compile_graph_learner_wgpu(gradient_policy="exact")
device = learner.tensor_device()
learner.set_input_tensor(device.upload([2, 2, 1], [1.0] * 4))
negative_target = device.upload([], [-1.0])
normalizer = device.upload([], [0.25])

receipts = []
for _ in range(32):
    forward = learner.forward()
    error = forward.prediction_tensor().add(negative_target)
    # Mean [0.75 * e^2 / 2 + 0.25 * e^4 / 4], derivatives built on the GPU.
    quadratic = learner.backward(forward, error.mul(normalizer))
    quartic = learner.backward(forward, error.mul(error).mul(error).mul(normalizer))
    batch = st.nn.GraphGradientBatch()
    batch.add(quadratic, 0.75)
    batch.add(quartic, 0.25)
    learner.sgd_weighted(batch, 0.1)
    receipts.append(learner.update_snapshot())

# Observe only at the batch boundary, not between GPU operations.
for receipt in receipts:
    receipt.read()

saved = learner.parameter_snapshot().read_plan()
resumed = saved.compile_graph_learner_wgpu(gradient_policy="exact")
```

Use `sgd(gradients, rate)` for one VJP with coefficient one. Rust also accepts
borrowed contributions directly via `sgd_weighted(&[(&g1, a), (&g2, b)], rate)`;
the owning `GraphGradientBatch` has `add`, `len`, and `is_empty`, and is applied
with `sgd_batch`. WASM exposes `await plan.compileGraphLearnerWebGpu("exact")`,
`new GraphGradientBatch()`, `batch.add`, `learner.sgdWeighted`,
`learner.updateSnapshot()` and `await snapshot.read()` (a `bigint`).
`parameterSnapshot().readPlan()` exports an owning weight-only checkpoint.
Call `.free()` on WASM handles when finished, including intermediate tensors.

## Update Contract

- Contributions are an ordered sum of 1..=256 finite-weighted exact VJPs from
  the **same current forward token**. A batch owns its handles, not copies of
  GPU values; releasing the original gradient handles does not break it.
- `exact` uses the composed derivative directly. `module_compatible` additionally
  divides gain parameter gradients by the flattened input row count, once at
  update time. The returned VJPs remain exact in both policies.
- Rate must be finite and nonnegative. Invalid host arguments change neither
  counters, current tape nor weights. A submitted attempt, including zero-rate
  and rejected attempts, invalidates the tape; run forward again before backward.
- Every source, weighted product, intermediate sum and parameter candidate is
  checked. Zero coefficients cannot hide invalid gradients, and later cancellation
  cannot hide overflow. One invalid value rejects **all** parameter commits.
- `sgd*` returns an attempt number, not acceptance or loss. Capture/read its
  owning update receipt to verify acceptance. A saved receipt survives subsequent
  forwards, updates and workspace destruction, and is consumed once. It does not
  certify other attempts. Numeric rejection leaves weights intact; later finite
  work can recover without erasing the earlier failure evidence.
- Checkpoints contain weights and the inference plan, not optimizer state,
  submitted counters or policy. Re-select the policy explicitly on resume.

## Validation and Limits

The shared native/browser `resident_graph_training` fixture runs 64 updates in
12 conditions (three seed/shape pairs, both policies, fused/unfused plans) using
quadratic and quartic GPU cotangents. It retains five checkpoints and checks
predictions, both raw VJPs and updated weights against ordinary Rust NN; the
independent PyTorch replayer checks CPU and MPS. JSON resume checks the next
update for bit identity. Separate public Python/WASM clients exercise ownership,
batch limits, stale state and zero-weight invalid-source rollback.

These are bounded learning/correctness fixtures, not model-quality or throughput
claims. Composition uses prepared weight/flag/shape buffers and writes groups of
up to four contributions directly into the graph's gradient buffers. Each product
and ordered intermediate sum retains the shared finite checks; one unit-weight
source uses a direct copy. The derivative outputs are still owning GPU snapshots,
and composing updates still creates bind groups and dispatches. This is not
a fused optimizer or a general reverse-mode tape for arbitrary `WgpuTensor`
expressions. External objectives supply their cotangents. Cross-forward
microbatch accumulation, Adam/momentum and mixed precision are not provided.

The source-bound training benchmark has a separate `--graph --learner` workload.
It times full quadratic/quartic seed construction, both VJPs and weighted SGD.
Rust captures and reads every update acceptance receipt; Torch synchronizes
completion without claiming equivalent guards/rollback. Losses are observed only
in untimed initial/final probes. Ordinary MSE timings remain a separate workload.
