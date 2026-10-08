const fs = require("node:fs");
const assert = require("node:assert/strict");
const path = require("node:path");

function checkAttentionContract(types, label) {
  const get = name => {
    const declaration = types.match(new RegExp("^( *)export class " + name + " \\{[\\s\\S]*?^\\1\\}", "m"))?.[0];
    assert.ok(declaration, label + " must export " + name);
    assert.match(declaration, /free\(\): void/);
    return declaration;
  };
  const tensor = get("WgpuTensor");
  const biases = get("WgpuAttentionBiases");
  const gradients = get("WgpuAttentionGradients");
  assert.match(tensor, /private constructor\(\)/);
  assert.match(biases, /^\s+constructor\(\);$/m);
  assert.match(gradients, /private constructor\(\)/);
  const tail = "scale: number, causal_offset: number \\| null \\| undefined, biases: WgpuAttentionBiases";
  assert.match(tensor, new RegExp("scaledDotAttention\\(keys: WgpuTensor, values: WgpuTensor, " + tail + "\\): WgpuTensor;"));
  assert.match(tensor, new RegExp("scaledDotAttentionVjp\\(keys: WgpuTensor, values: WgpuTensor, upstream: WgpuTensor, " + tail + "\\): WgpuAttentionGradients;"));
  for (const name of ["ZBias", "PairBias"]) {
    assert.match(biases, new RegExp("set" + name + "\\(bias: WgpuTensor\\): void;"));
    assert.match(biases, new RegExp("clear" + name + "\\(\\): void;"));
  }
  for (const name of ["query", "key", "value"])
    assert.match(gradients, new RegExp("readonly " + name + ": WgpuTensor;"));
  for (const name of ["zBias", "pairBias"])
    assert.match(gradients, new RegExp("readonly " + name + ": WgpuTensor \\| undefined;"));
  console.log(label + " resident attention TypeScript contract passed");
}

module.exports = {checkAttentionContract};

if (require.main === module) {
  assert.equal(process.argv.length, 3, "Pass the freshly generated spiraltorch_wasm.js path");
  const generated = path.resolve(process.argv[2]).replace(/\.js$/, ".d.ts");
  checkAttentionContract(fs.readFileSync(generated, "utf8"), "generated");
  const shipped = path.join(__dirname, "../types/spiraltorch-wasm.d.ts");
  checkAttentionContract(fs.readFileSync(shipped, "utf8"), "shipped");
}
