import { EllipticWarpKernel } from 'spiraltorch-wasm';

const kernel = new EllipticWarpKernel(1.3, 3, 2, 2);
const x = new Float32Array([1, 0.3, -0.4]);
const local = kernel.forward(x);
const tangent: Float32Array = local.jvp(new Float32Array([0, 0.2, 0.3]));
const anchored = kernel.forwardAnchored(x, -0.4);
const joint: Float32Array = anchored.jvp(new Float32Array([0, 0.2, 0.3]), 0.7);
const pullback: Float32Array = local.vjp(tangent);
void joint; void pullback;
anchored.free(); local.free(); kernel.free();
