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
