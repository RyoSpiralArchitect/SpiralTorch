import { EllipticWarpKernel, EllipticCausalLearningBatch } from "spiraltorch-wasm";

const kernel = new EllipticWarpKernel(1, 2, 1, 64);
const snapshot: EllipticCausalLearningBatch = kernel.forwardCausal(
    new Float32Array([1, 0.2, 0.3, 1, -0.4, 0.5]), 1, 2, 4,
);
const features: Float32Array = snapshot.features;
const gradient: Float32Array = snapshot.vjp(new Float32Array(features.length));
void gradient;
snapshot.free();
kernel.free();

// @ts-expect-error The pair budget is explicit at the WASM boundary.
kernel.forwardCausal(new Float32Array(6), 1, 2);
// @ts-expect-error Snapshots are produced by the native forward, not constructed.
new EllipticCausalLearningBatch();
