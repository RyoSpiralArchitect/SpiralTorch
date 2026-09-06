const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

const generated = path.resolve(process.argv[2]).replace(/\.js$/, ".d.ts");
for (const file of [generated, path.join(__dirname, "../types/spiraltorch-wasm.d.ts")]) {
  const types = fs.readFileSync(file, "utf8");
  const declaration = types.match(/^( *)export class TensorMeanBatch \{[\s\S]*?^\1\}/m)?.[0];
  assert.ok(declaration, "TensorMeanBatch export: " + file);
  assert.match(declaration, /constructor\(rows: number, cols: number, partial_count: number, data: Float32Array\)/);
  assert.match(declaration, /meanScaled\(scale: number\): Float32Array/);
  assert.match(declaration, /free\(\): void/);
  console.log("TensorMeanBatch TypeScript contract passed: " + file);
}
