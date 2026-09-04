const fs = require("node:fs");
const assert = require("node:assert/strict");
const path = require("node:path");

const modulePath = path.resolve(process.argv[2]);
const types = fs.readFileSync(modulePath.replace(/\.js$/, ".d.ts"), "utf8");
const declaration = types.match(/export class WgpuMatmul \{[\s\S]*?\n\}/)?.[0];
assert.ok(declaration, "WebGPU feature must export its workspace type");
assert.match(declaration, /create\(rows: number, inner: number, cols: number\): Promise<WgpuMatmul>/);
assert.match(declaration, /createWithTile\(rows: number, inner: number, cols: number, tile_m: number, tile_n: number, tile_k: number\): Promise<WgpuMatmul>/);
assert.match(declaration, /tileMNK\(\): Uint32Array/);
assert.match(declaration, /upload\(lhs: Float32Array, rhs: Float32Array\): void/);
assert.match(declaration, /dispatch\(repetitions\?: number(?: \| null)?\): bigint/);
assert.match(declaration, /readback\(\): Promise<Float32Array>/);
assert.match(declaration, /synchronize\(\): Promise<void>/);
assert.match(declaration, /readonly generation: bigint/);
console.log("resident WebGPU TypeScript contract passed");
