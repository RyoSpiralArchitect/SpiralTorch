// Execute actual wasm32 Rust under Node, replaying a native Python checkpoint.
// This is a CPU transform portability check, not a browser/WebGPU learning run.
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const wasm = require(path.resolve(process.argv[2]));
const fixture = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

async function main() {
  assert.equal(fixture.expected.length, 20);
  const pipeline = wasm.VisionTransformPipeline.createCpu(999);
  pipeline.addRandomHorizontalFlip(0.5);
  pipeline.addNormalize(new Float32Array([0.5]), new Float32Array([0.25]));
  pipeline.restoreCheckpointJson(fixture.state);
  for (const expected of fixture.expected) {
    const output = await pipeline.apply(1, 4, 4, new Float32Array(fixture.image));
    assert.deepEqual(Array.from(output.values), expected);
    output.free();
  }
  assert.equal(pipeline.checkpointJson(), fixture.final_state);
  const before = pipeline.checkpointJson();
  const bad = JSON.parse(fixture.state);
  bad.rng.word_position = (1n << 68n).toString();
  assert.throws(() => pipeline.restoreCheckpointJson(JSON.stringify(bad)));
  assert.equal(pipeline.checkpointJson(), before);
  pipeline.addRandomHorizontalFlip(0.5);
  assert.throws(() => pipeline.restoreCheckpointJson(fixture.state));
  pipeline.free();
  console.log("Python to wasm32 input checkpoint: 20 exact transforms and final RNG state passed");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
