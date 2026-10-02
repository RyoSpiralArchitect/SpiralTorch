import { EllipticWarpKernel, EllipticGatedCausalLearningBatch, EllipticGatedCausalGradients } from "spiraltorch-wasm";
const kernel = new EllipticWarpKernel(1, 3, 2, 8);
const snapshot: EllipticGatedCausalLearningBatch = kernel.forwardGatedCausal(new Float32Array(6), 1, 2, 0, 4);
const gradients: EllipticGatedCausalGradients = snapshot.vjp(new Float32Array(18));
const values: Float32Array = gradients.orientations;
const raw: number = gradients.rawMix;
const mix: number = snapshot.mix;
// @ts-expect-error Gate must be numeric.
kernel.forwardGatedCausal(values, 1, 2, "0", 4);
// @ts-expect-error Snapshots cannot be manually constructed.
new EllipticGatedCausalLearningBatch();
// @ts-expect-error Gradients cannot be manually constructed.
new EllipticGatedCausalGradients();
gradients.free(); snapshot.free(); kernel.free();
void raw; void mix;
