/// <reference path="../types/spiraltorch-wasm.d.ts" />
import { FractionalGlKernel, FractionalGlLearningBatch, FractionalGlGradients } from 'spiraltorch-wasm';
const kernel = new FractionalGlKernel(8, 1, 128, 1024);
const batch: FractionalGlLearningBatch = kernel.forward(new Float32Array(24), new Uint32Array([2,4,3]), 1, .5);
const derivative: FractionalGlGradients = batch.vjp(new Float32Array(24));
const alpha: number = derivative.alpha;
const dx: Float32Array = derivative.input;
const inputOnly: Float32Array = batch.vjp_input(dx);
const alphaOnly: number = batch.vjp_alpha(dx);
const dy: Float32Array = batch.jvp(dx, alpha);
const history: FractionalGlLearningBatch = kernel.forward_history(dx, new Uint32Array([2,4,3]), 1, .5);
const normalized: FractionalGlLearningBatch = kernel.forward_history_l2(dx, new Uint32Array([2,4,3]), 1, .5, 1);
normalized.free();
history.free();
void dy;
void inputOnly; void alphaOnly;
derivative.free(); batch.free(); kernel.free();
