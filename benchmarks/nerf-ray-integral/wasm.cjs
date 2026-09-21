// Actual st-vision/st-nn Rust code compiled for wasm32, not a JS compositor.
const {NerfCase, contract_report} = require(process.argv[2]);
const cases = [];
for (const batch of [1, 32, 256]) {
  for (const samples of [1, 8, 64]) {
    for (const varying of [false, true]) {
      const model = new NerfCase(batch, samples, varying, false);
      for (let i = 0; i < 5; i++) model.run().free();
      const elapsed_ns = [];
      for (let interval = 0; interval < 9; interval++) {
        const start = process.hrtime.bigint();
        for (let i = 0; i < 4; i++) model.run().free();
        elapsed_ns.push(Number(process.hrtime.bigint() - start) / 4);
      }
      const output = model.run();
      cases.push({metadata: JSON.parse(model.metadata()), values: Array.from(output.values()), elapsed_ns});
      output.free();
      model.free();
    }
  }
}
process.stdout.write(JSON.stringify({runtime: 'node-wasm-cpu', node: process.version,
  warmups: 5, intervals: 9, repetitions: 4, contract: JSON.parse(contract_report()), cases}));
