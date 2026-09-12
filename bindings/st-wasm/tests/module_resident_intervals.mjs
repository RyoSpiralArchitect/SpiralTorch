// Fixed protocol: completed reads are deliberately not a final-read-only burst.
export const protocol = Object.freeze({matrices: 4, warmup: 2, samples: 9, forwards: 256});
export const routes = Object.freeze([
  "baseline_module_d2h", "candidate_module_d2h",
  "baseline_scalar_d2h", "candidate_scalar_d2h",
]);

export function routeOrder(matrix, seed, block) {
  const offset = (matrix + seed + block) % routes.length;
  return [...routes.slice(offset), ...routes.slice(0, offset)];
}

export async function completedReads({forward, read, release, now = () => performance.now(),
  forwards = protocol.forwards}) {
  if (!Number.isSafeInteger(forwards) || forwards < 1) throw Error("invalid forward count");
  const outputs = [];
  const start = now();
  for (let i = 0; i < forwards; i++) {
    const output = forward();
    try {
      outputs.push(await read(output));
    } finally {
      release(output);
    }
  }
  const elapsed_ms = now() - start;
  if (!Number.isFinite(elapsed_ms) || elapsed_ms <= 0) throw Error("invalid interval clock");
  return {elapsed_ms, forwards, completed_reads: outputs.length, outputs};
}

export function clockProbe(now = () => performance.now()) {
  let previous = now(), equal = 0;
  const positive = [];
  for (let i = 0; i < 250000 && positive.length < 256; i++) {
    const current = now(), delta = current - previous;
    if (!Number.isFinite(delta) || delta < 0) throw Error("invalid probe clock");
    if (delta > 0) positive.push(delta);
    else equal++;
    previous = current;
  }
  if (!positive.length) throw Error("probe clock never advanced");
  return {positive_deltas_ms: positive, equal_reads: equal,
    minimum_positive_delta_ms: Math.min(...positive)};
}
