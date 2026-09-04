// Reproduce aggregates and enforce that failed variants were never timed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const output = {};
for (const name of ["native-all-modes", "native-tiled-ten-seeds", "browser-all-modes", "browser-tiled-repeat"]) {
  const report = JSON.parse(gunzipSync(fs.readFileSync(path.join(__dirname, name + ".json.gz"))));
  const rows = [];
  for (const entry of report.cases) {
    assert.ok(["passed", "correctness_error"].includes(entry.status));
    if (name.startsWith("native")) {
      assert.equal(entry.correctness.torch.passed, true);
      for (const [variant, info] of Object.entries(entry.wgpu_variants)) {
        rows.push({shape: entry.shape_m_k_n, seed: entry.seed, kernel: info.kernel,
          accumulation: info.accumulation, passed: entry.correctness[variant].passed,
          max_abs_error: entry.correctness[variant].max_abs_error,
          timing: entry.timings[variant], torch: entry.timings.torch_resident});
      }
    } else {
      rows.push({shape: entry.shape_m_k_n, seed: entry.seed, kernel: entry.kernel,
        accumulation: entry.accumulation, passed: entry.status === "passed",
        max_abs_error: entry.max_abs_error, timing: entry.timing});
    }
  }
  for (const row of rows) {
    assert.equal(Boolean(row.timing), row.passed);
    if (row.passed) {
      assert.equal(row.timing.samples_ms.length, 20);
      assert.ok(row.timing.samples_ms.every(x => Number.isFinite(x) && x >= 0));
    }
  }
  const groups = {};
  for (const row of rows) {
    const key = [row.shape.join("x"), row.kernel, row.accumulation].join("/");
    (groups[key] ??= []).push(row);
  }
  output[name] = Object.fromEntries(Object.entries(groups).map(([key, group]) => {
    const passing = group.filter(x => x.passed);
    const pairedSequential = passing.flatMap(row => {
      const sequential = rows.find(other => other.kernel === row.kernel && other.seed === row.seed &&
        other.shape.join() === row.shape.join() && other.accumulation === "sequential" && other.passed);
      return sequential ? [row.timing.median_ms / sequential.timing.median_ms] : [];
    });
    return [key, {passed: passing.length, total: group.length,
      failed_seeds: group.filter(x => !x.passed).map(x => x.seed),
      median_of_passing_seed_medians_ms: median(passing.map(x => x.timing.median_ms)),
      max_abs_error_passing: passing.length ? Math.max(...passing.map(x => x.max_abs_error)) : null,
      paired_sequential_seeds: pairedSequential.length,
      median_paired_ratio_to_sequential: median(pairedSequential),
      median_paired_ratio_to_torch: median(passing.filter(x => x.torch)
        .map(x => x.timing.median_ms / x.torch.median_ms))}];
  }));
}
console.log(JSON.stringify(output, null, 2));
