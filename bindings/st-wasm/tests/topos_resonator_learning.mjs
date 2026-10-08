// Shared scalar-WASM learning contract for the browser page and Node CI.
export function runToposLearning(ToposResonatorKernel, fixture) {
  const report = {schema: "spiraltorch.topos_browser_learning.v2", status: "running", guard_checks: 0,
    scope: "Scalar WASM learning and ownership parity; host recorded separately. Not WebGPU, timing or model quality."};
  function assert(ok, message) { if (!ok) throw Error(message); }
  function reject(fn) {
    let failed = false;
    let result;
    try { result = fn(); } catch { failed = true; }
    finally { result?.free?.(); }
    assert(failed, "expected rejection");
    report.guard_checks++;
  }
  function exact(actual, expected, label) {
    assert(actual.length === expected.length, label + " shape");
    for (let i = 0; i < actual.length; i++)
      assert(Number.isFinite(actual[i]) && Object.is(actual[i], expected[i]), label + " value " + i);
  }
  function compare(actual, expected, atol = 1e-7, rtol = 1e-6) {
    assert(actual.length === expected.length, "result shape");
    let maximum = 0;
    for (let i = 0; i < actual.length; i++) {
      assert(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `nonfinite comparison ${i}`);
      const delta = Math.abs(actual[i] - expected[i]);
      assert(delta <= atol + rtol * Math.abs(expected[i]), `value mismatch ${i}`);
      maximum = Math.max(maximum, delta);
    }
    return maximum;
  }
  let kernel;
  try {
    const c = fixture.config, f = fixture;
    kernel = new ToposResonatorKernel(c.coupling, c.iterations, c.saturation, c.porosity, c.max_values);
    assert(kernel.executionBackend === "rust_f32_wasm", "execution label");
    report.execution_backend = kernel.executionBackend;
    const x = new Float32Array(f.input), gate = new Float32Array(f.gate), upstream = new Float32Array(f.upstream);
    const forward = (a = x, b = gate) => kernel.forward(a, b, f.rows, f.features);
    const backward = kernel.backward(x, gate, upstream, f.rows, f.features);
    report.forward_max_error = compare(forward(), f.output);
    report.input_vjp_max_error = compare(backward.grad_input, f.grad_input);
    report.gate_vjp_max_error = compare(backward.grad_gate, f.grad_gate);
    const captured = kernel.capture(x, gate, f.rows, f.features);
    try {
      exact(captured.output, forward(), "captured forward");
      const pulled = captured.vjp(upstream);
      try {
        exact(pulled.grad_input, backward.grad_input, "captured input VJP");
        exact(pulled.grad_gate, backward.grad_gate, "captured gate VJP");
      } finally { pulled.free(); }
    } finally { captured.free(); }
    report.capture_matches_stateless_exactly = true;
    report.finite_difference_max_error = 0;
    for (const [values, gradient] of [[x, backward.grad_input], [gate, backward.grad_gate]]) {
      const numeric = [];
      for (let i = 0; i < values.length; i++) {
        const old = values[i], h = 1e-3;
        values[i] = old + h;
        const plus = forward().reduce((sum, value, j) => sum + value * upstream[j], 0);
        values[i] = old - h;
        const minus = forward().reduce((sum, value, j) => sum + value * upstream[j], 0);
        values[i] = old;
        numeric.push((plus - minus) / (2 * h));
      }
      report.finite_difference_max_error = Math.max(report.finite_difference_max_error,
        compare(gradient, numeric, 5e-5, 2e-3));
    }
    const learnedGate = new Float32Array(gate.length), legacyGate = learnedGate.slice();
    const target = new Float32Array(f.output);
    function learningStep(values, update, useCapture) {
      const batch = useCapture ? kernel.capture(x, values, f.rows, f.features) : null;
      let pulled;
      try {
        const output = batch ? batch.output : forward(x, values);
        const loss = output.reduce((sum, value, i) => sum + (value - target[i]) ** 2 / target.length, 0);
        if (update) {
          const seed = Float32Array.from(output, (value, i) => 2 * (value - target[i]) / target.length);
          pulled = batch ? batch.vjp(seed) : kernel.backward(x, values, seed, f.rows, f.features);
          const gradient = pulled.grad_gate;
          for (let i = 0; i < values.length; i++) values[i] -= 0.02 * gradient[i];
        }
        return loss;
      } finally { pulled?.free?.(); batch?.free(); }
    }
    const losses = [];
    for (let step = 0; step <= 100; step++) {
      const loss = learningStep(learnedGate, step < 100, true);
      const legacyLoss = learningStep(legacyGate, step < 100, false);
      assert(Object.is(loss, legacyLoss), "captured learning loss differs at " + step);
      exact(learnedGate, legacyGate, "captured learning gate at " + step);
      losses.push(loss);
    }
    assert(losses.at(-1) < losses[0] * 0.1, "browser gate did not learn");
    report.learning_losses = losses;
    report.learned_gate = Array.from(learnedGate);
    const restored = Float32Array.from(JSON.parse(JSON.stringify(report.learned_gate)));
    learningStep(learnedGate, true, true);
    learningStep(restored, true, true);
    exact(learnedGate, restored, "saved-gate next update");
    report.learning_updates = 100;
    report.learning_trajectory_exact = true;
    report.saved_gate_next_update_exact = true;
    report.next_gate = Array.from(restored);

    const source = x.slice(), sourceGate = gate.slice();
    const owner = new ToposResonatorKernel(c.coupling, c.iterations, c.saturation, c.porosity, c.max_values);
    let snapshot;
    try { snapshot = owner.capture(source, sourceGate, f.rows, f.features); }
    finally { owner.free(); }
    try {
      const output = snapshot.output, audit = snapshot.audit_json();
      source.fill(NaN); sourceGate.fill(NaN); snapshot.output.fill(NaN);
      exact(snapshot.output, output, "snapshot independence");
      assert(snapshot.audit_json() === audit, "snapshot audit changed");
      reject(() => snapshot.vjp(new Float32Array(upstream.length + 1)));
      reject(() => snapshot.vjp(new Float32Array(upstream.length).fill(NaN)));
      for (let repeat = 0; repeat < 2; repeat++) {
        const pulled = snapshot.vjp(upstream);
        try {
          exact(pulled.grad_input, backward.grad_input, "detached input VJP");
          exact(pulled.grad_gate, backward.grad_gate, "detached gate VJP");
        } finally { pulled.free(); }
      }
    } finally { snapshot.free(); }
    report.snapshot_independent_after_kernel_free = true;
    reject(() => new ToposResonatorKernel(1, 4, 1, 0, 16));
    reject(() => new ToposResonatorKernel(0.25, 0, 1, 0, 16));
    for (const bad of [-1, 0.5, NaN, Infinity, 4294967296, "1", null, true]) {
      reject(() => new ToposResonatorKernel(0.25, bad, 1, 0, 16));
      reject(() => new ToposResonatorKernel(0.25, 4, 1, 0, bad));
      reject(() => kernel.forward(x, gate, bad, f.features));
      reject(() => kernel.backward(x, gate, upstream, f.rows, bad));
      reject(() => kernel.capture(x, gate, bad, f.features));
      reject(() => kernel.capture(x, gate, f.rows, bad));
    }
    for (const bad of [NaN, Infinity, "0.2", null, true]) {
      reject(() => new ToposResonatorKernel(bad, 4, 1, 0, 16));
      reject(() => new ToposResonatorKernel(0.25, 4, bad, 0, 16));
      reject(() => new ToposResonatorKernel(0.25, 4, 1, bad, 16));
    }
    reject(() => kernel.forward(new Float32Array([NaN, 0, 0, 0]), gate, 2, 2));
    reject(() => kernel.backward(x, gate, new Float32Array([NaN, 0, 0, 0]), 2, 2));
    reject(() => kernel.forward(x, gate, 1, 2));
    const empty = new Float32Array();
    assert(kernel.forward(empty, empty, 0, 2).length === 0, "empty forward");
    assert(kernel.backward(empty, empty, empty, 0, 2).grad_gate.length === 0, "empty backward");
    const emptyBatch = kernel.capture(empty, empty, 0, 2);
    try {
      assert(emptyBatch.output.length === 0, "empty captured forward");
      const pulled = emptyBatch.vjp(empty);
      try { assert(pulled.grad_input.length === 0 && pulled.grad_gate.length === 0, "empty captured VJP"); }
      finally { pulled.free(); }
    } finally { emptyBatch.free(); }
    compare(forward(), f.output);
    report.status = "passed";
  } catch (error) {
    report.status = "error";
    report.error = String(error.stack || error);
  } finally {
    kernel?.free();
  }
  return report;
}
