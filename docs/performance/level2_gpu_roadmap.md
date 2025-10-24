# Level 2 GPU optimisation roadmap

SpiralTorch already routes high-level ops through the WGPU backend, but the next performance leap requires more aggressive shader specialisation and runtime adaptation. The following initiatives expand on the "もっと早くしろ選手権" brief by framing concrete engineering steps inside the existing Rust/WGSL stack.

## 6. Subgroup-aware shader primitives

Many compute passes inside `crates/st-backend-wgpu` still broadcast values across a full workgroup even when the reduction scope fits inside a hardware subgroup. Extending the shader code generators and reusable WGSL snippets to prefer `subgroupBroadcastFirst`, `subgroupAdd`, and `subgroupMax` unlocks the SIMD datapaths exposed on Metal, Vulkan, and DirectX 12.

Priority integration points:

- The softmax-style reductions that live in `st-backend-wgpu/src/topk_keepk.rs` and its helper WGSL fragments.
- Attention and transformer kernels under `st-backend-wgpu/src/midk_bottomk.rs`, which already stage partial sums inside shared memory.
- Batched normalisation / statistics pipelines that perform per-row reductions before scaling.

Implementation guidance:

1. Detect subgroup support at pipeline creation using `wgpu::Features::SUBGROUP_*` and encode it as part of the kernel dispatch key.
2. Provide dual WGSL paths: a subgroup-enhanced variant for GPUs that expose the feature and a fallback that matches the current workgroup-level synchronisation.
3. Update the shader authoring utilities in `st-backend-wgpu/src/shaders` to template the subgroup invocations so future ops can opt-in without copying boilerplate.

The expected gains mirror the performance we observe when moving from shared-memory reductions to warp-synchronous primitives in CUDA: softmax, layer normalisation, and attention score accumulation should benefit immediately on Apple M-series and RDNA3 hardware.

## 7. Chimera data layouts and Split-K batching

SpiralTorch’s allocator (`st-tensor`) currently honours PyTorch-like `contiguous()` semantics. Introducing a layout-aware allocator lets us keep intermediate tensors in a Chimera/Split-K form that matches the memory access pattern of downstream kernels instead of round-tripping through NCHW permutations.

Recommended steps:

1. Extend the tensor descriptor to track logical shapes separately from physical tiling (e.g., packed tiles, channel splits, Split-K stripes).
2. Teach the GEMM planners under `crates/st-backend-wgpu/src` and the CPU fallbacks to consume those tilings directly so we can launch matmuls without a `permute()` step.
3. Update fused pointwise operators in `crates/st-backend-wgpu/src/util.rs` to understand the same tiling metadata, allowing immediate execution after GEMM without re-layout.
4. Expose a public API hook (e.g., `Tensor::with_layout(Layout::Chimera)`) so advanced users can opt in while we migrate the standard modules (`st-nn`) incrementally.

Besides eliminating transient transposes, this approach prepares the backend for channel-split attention, block-sparse convolutions, and other irregular workloads where PyTorch’s single-contiguous contract becomes a bottleneck.

## 8. Fusion-first IR for composite ops

The current execution path still allocates intermediate buffers for pointwise follow-ups after GEMM/conv primitives. A lightweight fusion IR (think: Triton’s TTIR or TVM’s Tensor Expression graph) enables SpiralTorch to emit a single WGSL kernel for sequences such as `relu(matmul(a, b) + bias)`.

Development blueprint:

1. Introduce a per-backend IR node representation that captures tensor accessors, reductions, and simple arithmetic without committing to a final schedule.
2. Extend the operator registry (`crates/st-core`) so that op lowering can emit fusion candidates when the shapes and datatype constraints match.
3. Add a WGSL codegen path that realises the fused expression into one shader, emitting tiled loads from the existing allocator layout metadata.
4. Integrate with the autograd tape to ensure gradients can reuse the same fused IR (or fall back to eager ops when symbolic derivatives become complex).

With the IR in place, we can fold bias addition, activation, residual adds, and even small-scale normalisations into the same launch, keeping data in registers/shared memory instead of spilling to global buffers between passes.

## 9. Runtime-guided optimisation and adaptive codegen

Borrowing from TensorRT, XLA, and TVM, SpiralTorch can capture per-op latency/throughput metrics during execution and feed them back into the kernel selection logic. The telemetry hooks already flow through `st-core`’s span reporting; we can extend them into a runtime optimisation loop.

Roll-out strategy:

1. During the first few invocations of an op, record execution timing, memory bandwidth, and occupancy hints (e.g., workgroup sizes, subgroup availability).
2. Persist these metrics in the existing kernel cache so later dispatches can choose between specialised variants (subgroup vs. workgroup, tiled vs. split-K) based on observed winners.
3. Expose a background task that re-JITs or recompiles shaders when a clearly superior configuration is identified, replacing the cached pipeline handle transparently.
4. Surface debugging/profiling toggles via the Python bindings so users can inspect which kernels were auto-tuned.

Once wired, SpiralTorch becomes self-optimising: models run once to collect telemetry, and subsequent epochs adopt kernels tuned to the resident GPU. This closes the gap to PyTorch + cudnn heuristics while keeping the stack portable across WGPU targets.
