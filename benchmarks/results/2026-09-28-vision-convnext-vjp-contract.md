# Resident ConvNeXt block VJP contract

The Rust `ConvNeXtBlock::vjp_resident` combines the resident depthwise VJP
with the existing Rust graph-autograd LayerNorm, Linear, and GELU VJPs, then
adds the residual cotangent. It returns one input gradient and eight parameter
gradients in the same order and shapes as `Module::visit_parameters`. A
one-workspace cache rebuilds the frozen tail graph when its host parameter
values, logical input shape, or device change. No intermediate host readback,
parameter update, loss reduction, or CPU fallback occurs in the GPU route.

## Checks

- Native Apple M4 Metal: `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-vision --features wgpu --lib resident_block_vjp_matches_host_and_refreshes_changed_parameters`. The test compares all nine gradient tensors against CPU `ConvNeXtBlock::backward`, verifies host gradients are not mutated by the resident call, tests non-contiguous input and cotangent, checks cache reuse and rebuild after parameter edits, holds an old gradient across rebuild, rejects empty/mismatched shapes, and confirms invalid-source propagation and recovery.
- Browser: build the `st-vision` `convnext_block_vjp_browser` example for `wasm32-unknown-unknown --release`, generate web bindings with `wasm-bindgen 0.2.104`, then run `tools/test_resident_browser.cjs` with fixture `convnext-block-vjp-resident`. The [Chrome WebGPU report](2026-09-28-vision-convnext-vjp-browser.json) records all nine terminal CPU-parity checks, the separate non-fallback Apple Metal adapter probe, asset hashes, and no page errors. The probe does not attest the Rust runtime device.

This is a block-level VJP correctness result, not a full ConvNeXt training
step or a performance comparison. The backward recomputes depthwise and tail
activations. Stem, stage downsampling, and final-normalization VJPs, a
model-owned GPU parameter update, and matched real-data training evidence
remain open.
