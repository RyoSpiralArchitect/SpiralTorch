import { EllipticWarpKernel, EllipticAnchoredLearningBatch, EllipticAnchoredGradients } from "spiraltorch-wasm";
const kernel = new EllipticWarpKernel(1, 3, 2, 8);
const snapshot: EllipticAnchoredLearningBatch = kernel.forwardAnchored(new Float32Array([1, 0, 0]), 0);
const gradients: EllipticAnchoredGradients = snapshot.vjp(new Float32Array(9));
const values: Float32Array = gradients.orientations;
const raw: number = gradients.rawMix;
const mix: number = snapshot.mix;
// @ts-expect-error Gate must be numeric.
kernel.forwardAnchored(values, "0");
// @ts-expect-error Snapshots cannot be manually constructed.
new EllipticAnchoredLearningBatch();
// @ts-expect-error Gradients cannot be manually constructed.
new EllipticAnchoredGradients();
gradients.free(); snapshot.free(); kernel.free();
void raw; void mix;
