// The same scalar-WASM contract runs in Node and an actual browser.
export function checkToposSnapshotOwnership({ToposResonatorKernel}, memory) {
  let checks = 0, cases = 0, memoryGrowths = 0;
  const assert = (ok, label) => { checks++; if (!ok) throw Error(label); };
  const bits = values => new Uint32Array(values.buffer, values.byteOffset, values.length);
  const same = (a, b) => a.length === b.length && bits(a).every((v, i) => v === bits(b)[i]);
  assert(memory instanceof WebAssembly.Memory, "real WASM memory required");
  for (const shared of [false, true]) for (const [rows, features] of [[0, 3], [1, 1], [7, 17]]) {
    const input = Float32Array.from({length: rows * features}, (_, i) => i % 4 ? (i % 11 - 5) * .3 : -0);
    const gate = Float32Array.from({length: shared ? features : input.length}, (_, i) => (i % 7 - 3) * .4);
    const upstream = Float32Array.from(input, (_, i) => i % 3 ? .7 : -0);
    const kernel = new ToposResonatorKernel(.25, 5, 1, .2, 4096);
    let batch, pulled, retained, expected;
    try {
      batch = shared ? kernel.captureSharedRows(input, gate, rows, features)
        : kernel.capture(input, gate, rows, features);
      pulled = batch.vjp(upstream);
      const getters = [() => batch.output, () => pulled.grad_input, () => pulled.grad_gate];
      retained = getters.map(read => read());
      expected = retained.map(a => a.slice());
      for (let i = 0; i < getters.length; i++) {
        const first = retained[i], another = getters[i]();
        assert(first instanceof Float32Array && another instanceof Float32Array, "typed snapshot contract");
        assert(first.buffer !== memory.buffer && another.buffer !== first.buffer, "getter returned shared storage");
        if (another.length) another[0] = NaN;
        assert(same(first, expected[i]) && same(getters[i](), expected[i]), "JS mutation reached Rust or another snapshot");
      }
      memory.grow(1); memoryGrowths++;
      for (let i = 0; i < getters.length; i++) {
        assert(same(retained[i], expected[i]), "snapshot detached by WASM memory growth");
        assert(same(getters[i](), expected[i]), "getter failed after WASM memory growth");
      }
      let failed = false;
      try { batch.vjp(new Float32Array(input.length + 1)).free(); }
      catch { failed = true; }
      assert(failed, "invalid upstream was accepted");
      const retry = batch.vjp(upstream);
      try { assert(same(retry.grad_input, expected[1]) && same(retry.grad_gate, expected[2]), "failed VJP corrupted snapshots"); }
      finally { retry.free(); }
    } finally { pulled?.free(); batch?.free(); kernel.free(); }
    memory.grow(1); memoryGrowths++;
    for (let i = 0; i < retained.length; i++) {
      assert(same(retained[i], expected[i]), "snapshot changed after owner destruction and memory growth");
    }
    cases++;
  }
  return {status: "passed", cases, checks, memory_growths: memoryGrowths,
    contract: "Independent f32 snapshots remain valid after mutation, WASM memory growth, failure and owner destruction."};
}

export function checkSharedTopos({ToposResonatorKernel}) {
  let checks = 0, cases = 0;
  const assert = (ok, message) => { checks++; if (!ok) throw Error(message); };
  const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const rejected = call => {
    let failed = false;
    try { call()?.free?.(); } catch { failed = true; }
    assert(failed, "invalid request accepted");
  };
  const sumRows = (values, features) => {
    const sums = new Float64Array(features);
    values.forEach((v, i) => { sums[i % features] += v; });
    return Float32Array.from(sums);
  };
  for (const [rows, features] of [[0, 3], [7, 17], [64, 129]]) {
    for (const iterations of [1, 5, 64]) for (const porosity of [0, .3, 1]) {
      const x = Float32Array.from({length: rows * features}, (_, i) => (i * 37 % 127) / 53 - 1);
      const gate = Float32Array.from({length: features}, (_, i) => (i % 7 - 3) * .7);
      const expanded = Float32Array.from(x, (_, i) => gate[i % features]);
      const dy = Float32Array.from(x, (_, i) => (i * 13 % 31) / 31 - .5);
      const kernel = new ToposResonatorKernel(.25, iterations, 1, porosity, 16384);
      let batch, legacy, expected, output, audit;
      try {
        batch = kernel.captureSharedRows(x, gate, rows, features);
        legacy = kernel.capture(x, expanded, rows, features);
        const pulled = legacy.vjp(dy);
        try { expected = [pulled.grad_input, sumRows(pulled.grad_gate, features)]; }
        finally { pulled.free(); }
        output = legacy.output; audit = legacy.audit_json();
        assert(same(output, kernel.forwardSharedRows(x, gate, rows, features)), "shared forward differs");
        assert(batch.gateLayout === "shared_rows" && batch.gateValues === features, "gate was expanded");
        assert(legacy.gateLayout === "elementwise" && legacy.gateValues === x.length, "legacy gate shape changed");
      } finally { legacy?.free(); kernel.free(); }
      try {
        x.fill(NaN); gate.fill(NaN); batch.output.fill(NaN);
        assert(same(batch.output, output) && batch.audit_json() === audit, "captured state differs or aliases JS");
        rejected(() => batch.vjp(new Float32Array(dy.length + 1)));
        if (dy.length) rejected(() => batch.vjp(new Float32Array(dy.length).fill(NaN)));
        for (let repeat = 0; repeat < 2; repeat++) {
          const pulled = batch.vjp(dy);
          let retained;
          try { retained = [pulled.grad_input, pulled.grad_gate]; }
          finally { pulled.free(); }
          assert(same(retained[0], expected[0]) && same(retained[1], expected[1]), "shared pullback differs");
        }
      } finally { batch?.free(); }
      cases++;
    }
  }
  const kernel = new ToposResonatorKernel(0, 1, 1, 0, 8);
  try {
    for (const method of ["captureSharedRows", "forwardSharedRows"]) {
      for (const [x, gate, rows, features] of [
        [[], [], 0, 0], [[], Array(9).fill(1), 0, 9],
        [Array(9).fill(1), [1], 9, 1], [[1, 2], [1, 2], 2, 1],
        [[], [NaN], 0, 1], [[Infinity], [1], 1, 1],
        [[1], [1], 1.5, 1], [[1], [1], -1, 1], [[1], [1], 2 ** 32, 1],
      ]) rejected(() => kernel[method](Float32Array.from(x), Float32Array.from(gate), rows, features));
    }
    const max = Math.fround(3.4028234663852886e38);
    const batch = kernel.captureSharedRows(Float32Array.of(max, max, -max, -max, 1), Float32Array.of(0), 5, 1);
    try {
      rejected(() => batch.vjp(Float32Array.of(1, 1, 0, 0, 0)));
      rejected(() => batch.vjp(Float32Array.of(2, -2, 0, 0, 0)));
      const gradient = batch.vjp(new Float32Array(5).fill(1));
      try { assert(gradient.grad_gate[0] === 1, "finite cancellation lost"); }
      finally { gradient.free(); }
    } finally { batch.free(); }
  } finally { kernel.free(); }
  const learner = new ToposResonatorKernel(.4, 5, 1, .2, 64);
  let learning;
  try {
    const x = Float32Array.of(.2, -.3, .6, .5, -.7, .4);
    const target = learner.forwardSharedRows(x, Float32Array.of(.6, -.4), 3, 2);
    function step(gate, shared, update) {
      const batch = shared ? learner.captureSharedRows(x, gate, 3, 2)
        : learner.capture(x, Float32Array.from(x, (_, i) => gate[i % 2]), 3, 2);
      let pulled;
      try {
        const output = batch.output;
        const loss = output.reduce((s, v, i) => s + (v - target[i]) ** 2 / x.length, 0);
        if (update) {
          pulled = batch.vjp(Float32Array.from(output, (v, i) => 2 * (v - target[i]) / x.length));
          const gradient = shared ? pulled.grad_gate : sumRows(pulled.grad_gate, 2);
          gate.forEach((v, i) => { gate[i] = Math.fround(v - .3 * gradient[i]); });
        }
        return loss;
      } finally { pulled?.free(); batch.free(); }
    }
    const legacy = new Float32Array(2), shared = new Float32Array(2);
    const initial = step(shared, true, false);
    for (let i = 0; i < 240; i++) {
      step(legacy, false, true); step(shared, true, true);
      assert(same(legacy, shared), "matched wide-sum learning trajectory differs");
    }
    const final = step(shared, true, false), restored = shared.slice();
    step(shared, true, true); step(restored, true, true);
    assert(final < initial * .01 && same(shared, restored), "learning or continuation failed");
    learning = {updates: 240, initial_loss: initial, final_loss: final,
      matched_wide_sum_trajectory_exact: true, next_update_exact: true};
  } finally { learner.free(); }
  return {status: "passed", cases, checks, learning,
    scope: "Scalar WASM execution. Same Rust recurrence, not independent math or GPU/speed/LLM-quality evidence."};
}
