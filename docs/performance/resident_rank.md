# Resident Exact Rank

`st-backend-wgpu::rankk_exact_2ce::resident::ResidentRank` owns the input,
scratch, output buffers, bind group and pipelines for one fixed rank plan.
It uses the same exact two-stage shaders as `dispatch_host`; the existing host
API is unchanged. TopK is descending, BottomK ascending, and MidK selects the
center of the ascending finite candidates. Equal values prefer the lower input
index; float total ordering distinguishes signed zero. Non-finite candidates
are ignored and missing outputs are `(NaN, -1)`.

## WASM First

Build with the `webgpu` feature. Creation validates dimensions before narrowing
JS numbers and awaits WebGPU pipeline validation. No CPU fallback is used.

```js
const rank = await WgpuRank.create("topk", 1, 8, 3, 128);
rank.upload(new Float32Array([3, 1, 3, -2, 7, 4, 0, 1]));
rank.dispatch();
const {values, indices, generation} = await rank.readback();
// values: [7, 4, 3], indices: [4, 5, 0], generation: 1n
rank.free();
```

`dispatch(repetitions=1)` enqueues 1..1024 repetitions without reading back.
`synchronize()` waits using a four-byte map completion fence. `readback()`
immediately snapshots values and indices into one staging buffer, then returns
an asynchronous mapping promise. Later uploads, dispatches or `free()` do not
change that snapshot. Upload invalidates the current output; readback before a
new dispatch is rejected. Failed input-length validation preserves prior state.

The shared shader writes value bits as `u32`, avoiding browser rejection of a
constant NaN expression, and uses `workgroupUniformLoad` for the merge-loop bound.
Neither change disables shader validation or changes the CPU ordering contract.
For MidK with at most 32 tiles, candidates find their global rank through
parallel binary searches over the sorted tiles. This avoids serially discarding
half of each row. More fragmented geometries retain the existing GPU merge.

Tiles with a padded stride up to 1024 now sort in 8 KiB of workgroup memory,
publishing their sorted run to global scratch only once. Larger tiles keep the
storage-memory sorting network. Both paths use the same total-order comparator;
requested tile geometry, output ordering, padding and snapshot semantics are
unchanged. Device admission checks the workgroup storage requirement explicitly.

## Python And Rust

```python
import spiraltorch as st

rank = st.WgpuRank("topk", 1, 8, 3, tile_cols=128)
rank.upload(st.Tensor(1, 8, [3, 1, 3, -2, 7, 4, 0, 1]))
rank.dispatch()
result = rank.readback()  # flat values/indices lists plus integer generation
assert result["indices"] == [4, 5, 0]
```

`st.wgpu.WgpuRank` is the same class. A CPU-only wheel exposes a constructor
that raises `NotImplementedError`, rather than silently executing on CPU.
Rust callers construct `ResidentRank` from a `WgpuRuntime` and checked `Plan`,
then use `upload`, `dispatch`, `snapshot().read()` and `synchronize` directly.

## Projection To Rank Without Host Staging

`set_input_from_matmul` (`setInputFromMatmul` in JavaScript) copies current
matmul output into the rank input on the GPU. This is a device-to-device copy,
not zero-copy aliasing: later source updates or destruction cannot change the
copied input. The destination has a new input generation and requires dispatch.
Source shape and exact device/queue handles must match. Missing/stale source
output, mismatched shape/device, or generation overflow leave rank state intact.
Freshness means submitted logical output, not a completed GPU execution receipt.

```python
head = st.WgpuMatmul(1, 2, 4)
head.upload(st.Tensor(1, 2, [1, 0]), st.Tensor(2, 4, [2, 7, 3, 1, 0, 0, 0, 0]))
rank = st.WgpuRank("topk", 1, 4, 2)
head.dispatch()
rank.set_input_from_matmul(head)  # no logits readback or host upload
rank.dispatch()
assert rank.readback()["indices"] == [1, 2]
```

```js
const head = await WgpuMatmul.create(1, 2, 4);
head.upload(new Float32Array([1, 0]), new Float32Array([2, 7, 3, 1, 0, 0, 0, 0]));
const rank = await WgpuRank.create("topk", 1, 4, 2);
head.dispatch();
rank.setInputFromMatmul(head);
head.free();
rank.dispatch();
const result = await rank.readback(); // indices: [1, 2]
rank.free();
```

Rust exposes the same setter on `ResidentRank`. This composes a resident linear
projection with exact rank selection; it is not a complete language model or a
fused single-dispatch kernel. The copy and the following rank dispatch are
separate queue submissions.

```sh
cargo build --release --locked -p st-core --features wgpu-rt --example resident_matmul_rank_bench
python tools/bench_matmul_rank_vs_torch.py \
  --executable target/release/examples/resident_matmul_rank_bench \
  --output matmul-rank-comparison.json
```

This harness compares the intermediate host bridge against the GPU copy bridge,
including the same final rank readback. It also measures resident chains against
preallocated PyTorch CUDA matmul/rank with TF32 disabled. Inputs are bounded
integers to make fp32 projection and stable tie-order checks exact; these
fixtures do not establish arbitrary floating-point rank stability or model
quality. Source/image binding and foreign-GPU-process gates match the rank-only
harness; no exclusive GPU reservation or interleaved framework timing is claimed.

## Measurement Boundaries

The comparison separates the existing host API, a persistent-buffer
host-to-host call, and batched resident dispatch plus completion fence. The
PyTorch reference uses preallocated CUDA outputs, with the same 16 repetitions
and no upload/readback in the resident interval. MidK uses stable full sort in
PyTorch. This is an API throughput diagnostic, not GPU-event kernel timing or
an interleaved cross-framework experiment. Do not compare resident times with
PyTorch host-to-host times as a speedup claim.

```sh
cargo build --release --locked -p st-core --features wgpu-rt --example resident_rank_bench
python tools/bench_resident_rank_vs_torch.py \
  --executable target/release/examples/resident_rank_bench \
  --output resident-rank-comparison.json
```

The harness requires a clean source-bound build, preserves exact requests by
hash, verifies output against canonical PyTorch ranks, and checks source and
execution-image stability. It crosses three seeds, three tiles, three rank
kinds and two widths. Numerical failures are errors, not timing samples.
The runner rejects other CUDA compute PIDs before and after measurement. This
is a best-effort process check, not an exclusive GPU reservation; shared-load
measurements must still be treated as provisional.
