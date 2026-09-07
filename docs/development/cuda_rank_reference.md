# Canonical CUDA Rank Controls

The resident rank benchmark tools compare the shared Rust/WGPU implementation
with eager PyTorch CUDA controls. `tools/torch_rank_reference.py` is a benchmark
reference, not a second SpiralTorch runtime or a new CUDA backend. Inputs are
finite f32 matrices; non-finite inputs are rejected rather than timed under a
different contract. Native correctness checks include exact output bits and
canonical source indices, including the distinction between -0 and +0.

## Admission

Admission is outside timing, based on the immutable fixture after f32 rounding,
not on Python double uniqueness. A changed input requires new admission.

| Control | Condition | Work Included In Every Operation |
| --- | --- | --- |
| `topk` | TopK/BottomK; retained values and cutoff are untied | Sorted float32 topk |
| `topk_index_repair` | Internal ties, unique cutoff, no mixed signed zeros in the retained set | Unsorted topk, index sort, value gather, stable value sort, index gather |
| `stable_sort` | No mixed signed zeros within an input row | Full stable float32 sort; take the requested view |
| `packed_topk` | All admitted fixtures, including MidK and tied cutoffs | Exact integer key construction, integer topk, value gather |

Equal values outside the retained set need not force full sorting. Internal
ties can be repaired only when the cutoff is unique: then all tied selected
values are already present. Sorting their unique source indices first makes
the subsequent stable value sort canonical. A cutoff tie cannot be repaired
from an arbitrary topk subset; the omitted lower source index may be the right
answer. Plain PyTorch topk does not promise stable tied indices; see its
[API documentation](https://docs.pytorch.org/docs/stable/generated/torch.topk.html).

Packed keys use a signed 32-bit monotone transform of finite f32 bits as the
high word, and a source-index tie breaker as the low word. TopK reverses the
index word, not the float value. BottomK/MidK use ascending source indices.
The column count must be below 2^32, so distinct source indices cannot collide
or disturb the value-word ordering. MidK selects the first `start + k` keys
in ascending order and returns its requested window.

## Measurement

Output/scratch buffers and geometry-only index words are allocated before
timing. No value-dependent keys or corrected indices are cached across
operations. Sixteen complete operations plus a completion fence form each
batch. Controls rotate within the same process; two warmup batches precede
twelve retained batches. Exact value-bit/index checks run before measurement
and after every batch, with validation readbacks outside timing.

Both resident report schemas are v2. `torch_controls` retains all eligible
controls, their order and raw samples. `torch_operation` and the primary CUDA
timing select the hindsight lowest **sample mean** among those controls.
`legacy_torch_operation` records the old duplicate-anywhere rule; it is not the
new selection rule. This best-fixed comparison is not an online policy win,
a confidence interval or a claim to the fastest possible PyTorch implementation.
Native WGPU and CUDA remain separate process blocks, not interleaved GPU events.

The tests in `tests/test_resident_rank_bench.py` have a dependency-free admission
layer and opt-in live Torch tests. A requested CUDA test fails rather than
falling back to CPU:

```sh
SPIRALTORCH_RUN_TORCH_RANK_TESTS=1 SPIRALTORCH_TORCH_RANK_DEVICE=cpu python -I tests/test_resident_rank_bench.py -v
SPIRALTORCH_RUN_TORCH_RANK_TESTS=1 SPIRALTORCH_TORCH_RANK_DEVICE=cuda python tests/test_resident_rank_bench.py -v
```

The second recipe permits a user-site Torch installation on Furnace. Check
`spiral status` before launching GPU work, keep jobs non-overlapping and leave
other workloads untouched. Source/build identity and fixed-control versus real
Black Cat feedback receipts retain their existing Rust-owned validation gates.
