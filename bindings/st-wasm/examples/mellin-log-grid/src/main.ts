import init, {
  WasmMellinEvalPlan,
  WasmMellinLogGrid,
  mellin_exp_decay_samples,
} from "../pkg/spiraltorch_wasm.js";

const form = document.querySelector<HTMLFormElement>("#mellin-form")!;
const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const supportEl = document.querySelector<HTMLElement>("#support")!;
const hilbertEl = document.querySelector<HTMLElement>("#hilbert-norm")!;
const trainLossEl = document.querySelector<HTMLElement>("#train-loss")!;
const timingsEl = document.querySelector<HTMLElement>("#timings")!;
const previewEl = document.querySelector<HTMLPreElement>("#preview")!;
const canvas = document.querySelector<HTMLCanvasElement>("#plot")!;
const ctx = canvas.getContext("2d")!;
const heatmapCanvas = document.createElement("canvas");
const heatmapCtx = heatmapCanvas.getContext("2d");
const initButton = document.querySelector<HTMLButtonElement>("#init")!;
const trainButton = document.querySelector<HTMLButtonElement>("#train")!;
const reportEl = document.querySelector<HTMLPreElement>("#learning-report")!;
const copyReportButton = document.querySelector<HTMLButtonElement>("#copy-report")!;
const downloadReportButton = document.querySelector<HTMLButtonElement>("#download-report")!;

let ready = false;
let targetGrid: WasmMellinLogGrid | null = null;
let learnGrid: WasmMellinLogGrid | null = null;
let gridKey = "";
let lastReportJson = "";

type NumericStats = {
  count: number;
  finiteCount: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  l1: number;
  linf: number;
};

type MellinTrainTraceRow = {
  step: number;
  loss: number;
};

function setStatus(message: string, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#ff9a9a" : "";
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function getValue(id: string): string {
  return (document.querySelector<HTMLInputElement>(`#${id}`)?.value ?? "").trim();
}

function parseNumber(id: string, fallback: number): number {
  const raw = getValue(id);
  if (!raw) return fallback;
  const parsed = Number.parseFloat(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${id} must be a finite number`);
  }
  return parsed;
}

function parseIntStrict(id: string, fallback: number, min: number): number {
  const raw = getValue(id);
  const parsed = raw ? Number.parseInt(raw, 10) : fallback;
  if (!Number.isFinite(parsed) || parsed < min) {
    throw new Error(`${id} must be an integer >= ${min}`);
  }
  return parsed;
}

function snapshotNumber(id: string, fallback: number): number | null {
  try {
    return parseNumber(id, fallback);
  } catch {
    return null;
  }
}

function snapshotInt(id: string, fallback: number, min: number): number | null {
  try {
    return parseIntStrict(id, fallback, min);
  } catch {
    return null;
  }
}

function summarize(values: Float32Array): NumericStats {
  let finiteCount = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let sumSq = 0;
  let l1 = 0;
  let linf = 0;

  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    finiteCount += 1;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    sumSq += value * value;
    const abs = Math.abs(value);
    l1 += abs;
    if (abs > linf) linf = abs;
  }

  if (finiteCount === 0) {
    return {
      count: values.length,
      finiteCount,
      min: 0,
      max: 0,
      mean: 0,
      rms: 0,
      l1: 0,
      linf: 0,
    };
  }

  return {
    count: values.length,
    finiteCount,
    min,
    max,
    mean: sum / finiteCount,
    rms: Math.sqrt(sumSq / finiteCount),
    l1,
    linf,
  };
}

function sampleValues(values: Float32Array, limit = 16): number[] {
  const out: number[] = [];
  const n = Math.min(values.length, limit);
  for (let i = 0; i < n; i++) {
    out.push(Number(values[i].toPrecision(7)));
  }
  return out;
}

function sampleComplexValues(values: Float32Array, limit = 8): Array<{ re: number; im: number }> {
  const out: Array<{ re: number; im: number }> = [];
  const n = Math.min(Math.floor(values.length / 2), limit);
  for (let i = 0; i < n; i++) {
    out.push({
      re: Number(values[i * 2].toPrecision(7)),
      im: Number(values[i * 2 + 1].toPrecision(7)),
    });
  }
  return out;
}

function differenceMagnitude(a: Float32Array, b: Float32Array): Float32Array {
  const n = Math.min(a.length, b.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.abs(a[i] - b[i]);
  }
  return out;
}

function runtimeReport() {
  return {
    wasm: true,
    webgpuAvailable: "gpu" in navigator,
    userAgent: navigator.userAgent,
  };
}

function currentConfig() {
  const mode = getValue("mode") || "vertical";
  return {
    logStart: snapshotNumber("log-start", -5.0),
    logStep: snapshotNumber("log-step", 0.01),
    len: snapshotInt("len", 2048, 2),
    mode,
    sReal: snapshotNumber("s-real", 2.5),
    imagMin: snapshotNumber("imag-min", -30),
    imagMax: snapshotNumber("imag-max", 30),
    imagCount: snapshotInt("imag-count", 256, 2),
    realMin: snapshotNumber("real-min", 0.8),
    realMax: snapshotNumber("real-max", 3.2),
    realCount: snapshotInt("real-count", 256, 2),
    initNoise: snapshotNumber("init-noise", 0.02),
    trainSteps: snapshotInt("train-steps", 40, 1),
    trainLr: snapshotNumber("train-lr", 0.08),
  };
}

function setReport(report: unknown) {
  lastReportJson = JSON.stringify(report, null, 2);
  reportEl.textContent = lastReportJson;
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function safeFileStamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function copyLatestReport() {
  if (!lastReportJson) {
    throw new Error("No report yet. Run evaluation or training first.");
  }
  await navigator.clipboard.writeText(lastReportJson);
}

function downloadLatestReport() {
  if (!lastReportJson) {
    throw new Error("No report yet. Run evaluation or training first.");
  }
  downloadText(`spiraltorch-mellin-wasm-${safeFileStamp()}.json`, lastReportJson);
}

function linspace(min: number, max: number, count: number): Float32Array {
  const out = new Float32Array(count);
  const denom = count > 1 ? count - 1 : 1;
  for (let i = 0; i < count; i++) {
    const t = denom === 0 ? 0 : i / denom;
    out[i] = min + (max - min) * t;
  }
  return out;
}

function buildVerticalLineS(real: number, imagMin: number, imagMax: number, count: number): Float32Array {
  const out = new Float32Array(count * 2);
  const denom = count > 1 ? count - 1 : 1;
  for (let i = 0; i < count; i++) {
    const t = denom === 0 ? 0 : i / denom;
    const imag = imagMin + (imagMax - imagMin) * t;
    out[i * 2] = real;
    out[i * 2 + 1] = imag;
  }
  return out;
}

function makeGridKey(logStart: number, logStep: number, len: number): string {
  return `${logStart.toFixed(6)}|${logStep.toFixed(6)}|${len}`;
}

function ensureTargetGrid(logStart: number, logStep: number, len: number): WasmMellinLogGrid {
  const key = makeGridKey(logStart, logStep, len);
  if (!targetGrid || gridKey !== key) {
    const samples = mellin_exp_decay_samples(logStart, logStep, len);
    targetGrid = new WasmMellinLogGrid(logStart, logStep, samples);
    learnGrid = null;
    gridKey = key;
    trainLossEl.textContent = "—";
  }
  return targetGrid;
}

function initLearnableGrid(logStart: number, logStep: number, len: number, noise: number): WasmMellinLogGrid {
  const target = ensureTargetGrid(logStart, logStep, len);
  const base = target.samples();
  const out = new Float32Array(base);
  const sigma = Number.isFinite(noise) ? noise : 0.02;
  if (sigma > 0) {
    for (let i = 0; i < out.length; i++) {
      out[i] += (Math.random() * 2 - 1) * sigma;
    }
  }
  learnGrid = new WasmMellinLogGrid(logStart, logStep, out);
  return learnGrid;
}

function drawMagnitude(magnitudes: Float32Array) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "rgba(4, 6, 14, 0.75)";
  ctx.fillRect(0, 0, w, h);

  const n = magnitudes.length;
  if (n < 2) return;

  let max = 0;
  for (let i = 0; i < n; i++) {
    const v = magnitudes[i];
    if (Number.isFinite(v) && v > max) max = v;
  }
  if (max <= 0) max = 1;

  ctx.strokeStyle = "rgba(120, 160, 255, 0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const v = Math.min(1, Math.max(0, magnitudes[i] / max));
    const y = (1 - v) * (h - 16) + 8;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(210, 218, 255, 0.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  ctx.fillText(`max |M(s)| ≈ ${max.toExponential(3)}`, 10, 18);
}

function drawMagnitudeCompare(target: Float32Array, learned: Float32Array, loss: number) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "rgba(4, 6, 14, 0.75)";
  ctx.fillRect(0, 0, w, h);

  const n = Math.min(target.length, learned.length);
  if (n < 2) return;

  let max = 0;
  for (let i = 0; i < n; i++) {
    const a = target[i];
    const b = learned[i];
    if (Number.isFinite(a) && a > max) max = a;
    if (Number.isFinite(b) && b > max) max = b;
  }
  if (max <= 0) max = 1;

  const drawLine = (values: Float32Array, style: string) => {
    ctx.strokeStyle = style;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * (w - 16) + 8;
      const v = Math.min(1, Math.max(0, values[i] / max));
      const y = (1 - v) * (h - 24) + 16;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  drawLine(target, "rgba(120, 160, 255, 0.92)");
  drawLine(learned, "rgba(255, 120, 220, 0.92)");

  ctx.fillStyle = "rgba(210, 218, 255, 0.82)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  const lossText = Number.isFinite(loss) ? loss.toExponential(3) : "—";
  ctx.fillText(`target vs learned · loss=${lossText}`, 10, 16);
  ctx.fillText(`max |M(s)| ≈ ${max.toExponential(3)}`, 10, 32);
}

function rampColor(t: number): [number, number, number] {
  const x = clamp01(t) * 4;
  const seg = Math.min(3, Math.floor(x));
  const f = x - seg;

  switch (seg) {
    case 0:
      return [0, Math.round(255 * f), 255];
    case 1:
      return [0, 255, Math.round(255 * (1 - f))];
    case 2:
      return [Math.round(255 * f), 255, 0];
    default:
      return [255, Math.round(255 * (1 - f)), 0];
  }
}

function drawHeatmap(magnitudes: Float32Array, rows: number, cols: number) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "rgba(4, 6, 14, 0.75)";
  ctx.fillRect(0, 0, w, h);

  if (magnitudes.length !== rows * cols) return;
  if (!heatmapCtx) return;

  let max = 0;
  for (let i = 0; i < magnitudes.length; i++) {
    const v = magnitudes[i];
    if (Number.isFinite(v) && v > max) max = v;
  }
  if (max <= 0) max = 1;

  heatmapCanvas.width = cols;
  heatmapCanvas.height = rows;
  const img = heatmapCtx.createImageData(cols, rows);

  const denom = Math.log1p(max);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const v = Math.max(0, magnitudes[idx]);
      const t = denom > 0 ? Math.log1p(v) / denom : 0;
      const [rr, gg, bb] = rampColor(t);
      const o = idx * 4;
      img.data[o] = rr;
      img.data[o + 1] = gg;
      img.data[o + 2] = bb;
      img.data[o + 3] = 255;
    }
  }

  heatmapCtx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(heatmapCanvas, 0, 0, w, h);
  ctx.imageSmoothingEnabled = true;

  ctx.fillStyle = "rgba(210, 218, 255, 0.85)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  ctx.fillText(`max |M(s)| ≈ ${max.toExponential(3)} (log scale)`, 10, 18);
}

function interleavedToMagnitudes(values: Float32Array): Float32Array {
  if (values.length % 2 !== 0) {
    return new Float32Array();
  }
  const n = values.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const re = values[i * 2];
    const im = values[i * 2 + 1];
    out[i] = Math.hypot(re, im);
  }
  return out;
}

function gridSnapshot(grid: WasmMellinLogGrid) {
  const support = grid.support();
  return {
    len: grid.len(),
    logStart: grid.logStart,
    logStep: grid.logStep,
    support: [support[0], support[1]],
    hilbertNorm: grid.hilbertNorm(),
    sampleStats: summarize(grid.samples()),
  };
}

function planSnapshot(plan: WasmMellinEvalPlan) {
  const shape = plan.shape();
  return {
    len: plan.len(),
    shape: Array.from(shape),
    logStart: plan.logStart,
    logStep: plan.logStep,
  };
}

async function runOnce(ev?: Event) {
  ev?.preventDefault();
  if (!ready) return;

  try {
    setStatus("Building grid…");
    previewEl.textContent = "";

    const logStart = parseNumber("log-start", -5.0);
    const logStep = parseNumber("log-step", 0.01);
    const len = parseIntStrict("len", 2048, 2);

    const mode = getValue("mode") || "vertical";
    const real = parseNumber("s-real", 2.5);
    const imagMin = parseNumber("imag-min", -30);
    const imagMax = parseNumber("imag-max", 30);
    const imagCount = parseIntStrict("imag-count", 256, 2);

    const realMin = parseNumber("real-min", 0.8);
    const realMax = parseNumber("real-max", 3.2);
    const realCount = parseIntStrict("real-count", 256, 2);

    const key = makeGridKey(logStart, logStep, len);
    const reuse = targetGrid && gridKey === key;
    const t0 = performance.now();
    let t1 = t0;
    let t2 = t0;
    let grid: WasmMellinLogGrid;
    if (reuse) {
      grid = targetGrid!;
    } else {
      const samples = mellin_exp_decay_samples(logStart, logStep, len);
      t1 = performance.now();
      grid = new WasmMellinLogGrid(logStart, logStep, samples);
      t2 = performance.now();
      targetGrid = grid;
      learnGrid = null;
      gridKey = key;
      trainLossEl.textContent = "—";
    }

    const support = grid.support();
    supportEl.textContent = `[${support[0].toExponential(3)}, ${support[1].toExponential(3)}]`;
    hilbertEl.textContent = `${grid.hilbertNorm().toExponential(6)}`;

    setStatus("Evaluating…");
    const t3 = performance.now();

    if (mode === "mesh") {
      const realValues = linspace(realMin, realMax, realCount);
      const imagValues = linspace(imagMin, imagMax, imagCount);
      const plan = grid.planMesh(realValues, imagValues);
      const mags = grid.evaluatePlanMagnitude(plan);
      const t4 = performance.now();

      drawHeatmap(mags, realCount, imagCount);
      const buildLabel = reuse
        ? "cached"
        : `samples ${(t1 - t0).toFixed(1)}ms · grid ${(t2 - t1).toFixed(1)}ms`;
      timingsEl.textContent = `${buildLabel} · mesh ${(t4 - t3).toFixed(1)}ms`;

      const head = Math.min(6, imagCount);
      const rows: string[] = [];
      const previewS = new Float32Array(head * 2);
      for (let i = 0; i < head; i++) {
        previewS[i * 2] = realValues[0];
        previewS[i * 2 + 1] = imagValues[i];
      }
      const previewPlan = grid.planMany(previewS);
      const previewValues = grid.evaluatePlan(previewPlan);
      for (let i = 0; i < head; i++) {
        const re = previewValues[i * 2];
        const im = previewValues[i * 2 + 1];
        rows.push(
          `re=${realValues[0].toFixed(3)}  t=${imagValues[i].toFixed(3)}  M(s)≈${re.toExponential(
            4,
          )} + ${im.toExponential(4)}i`,
        );
      }
      previewEl.textContent = rows.join("\n");
      setReport({
        schema: "spiraltorch.wasm.mellin_eval_report.v1",
        kind: "mellin-log-grid-evaluation",
        createdAt: new Date().toISOString(),
        runtime: runtimeReport(),
        config: currentConfig(),
        grid: gridSnapshot(grid),
        plan: planSnapshot(plan),
        timingsMs: {
          samples: reuse ? 0 : t1 - t0,
          grid: reuse ? 0 : t2 - t1,
          evaluate: t4 - t3,
        },
        inferenceProbe: {
          mode: "mesh",
          realHead: sampleValues(realValues, 8),
          imagHead: sampleValues(imagValues, 8),
          magnitudeStats: summarize(mags),
          magnitudeHead: sampleValues(mags, 16),
          previewValues: sampleComplexValues(previewValues, 6),
        },
      });
      setStatus(`OK (mesh ${realCount}×${imagCount})`);
    } else {
      const imagValues = linspace(imagMin, imagMax, imagCount);
      const plan = grid.planVerticalLine(real, imagValues);
      const values = grid.evaluatePlan(plan);
      const t4 = performance.now();

      const mags = interleavedToMagnitudes(values);
      drawMagnitude(mags);
      const buildLabel = reuse
        ? "cached"
        : `samples ${(t1 - t0).toFixed(1)}ms · grid ${(t2 - t1).toFixed(1)}ms`;
      timingsEl.textContent = `${buildLabel} · eval ${(t4 - t3).toFixed(1)}ms`;

      const head = Math.min(6, imagCount);
      const rows: string[] = [];
      const sValues = buildVerticalLineS(real, imagMin, imagMax, imagCount);
      for (let i = 0; i < head; i++) {
        const re = values[i * 2];
        const im = values[i * 2 + 1];
        rows.push(
          `t=${sValues[i * 2 + 1].toFixed(3)}  M(s)≈${re.toExponential(4)} + ${im.toExponential(
            4,
          )}i  |M|≈${mags[i].toExponential(4)}`,
        );
      }
      previewEl.textContent = rows.join("\n");
      setReport({
        schema: "spiraltorch.wasm.mellin_eval_report.v1",
        kind: "mellin-log-grid-evaluation",
        createdAt: new Date().toISOString(),
        runtime: runtimeReport(),
        config: currentConfig(),
        grid: gridSnapshot(grid),
        plan: planSnapshot(plan),
        timingsMs: {
          samples: reuse ? 0 : t1 - t0,
          grid: reuse ? 0 : t2 - t1,
          evaluate: t4 - t3,
        },
        inferenceProbe: {
          mode: "vertical",
          sReal: real,
          imagHead: sampleValues(imagValues, 8),
          valueHead: sampleComplexValues(values, 8),
          magnitudeStats: summarize(mags),
          magnitudeHead: sampleValues(mags, 16),
        },
      });
      setStatus(`OK (n=${imagCount})`);
    }
  } catch (err) {
    setStatus((err as Error).message, true);
    console.error(err);
  }
}

form.addEventListener("submit", runOnce);

initButton.addEventListener("click", () => {
  if (!ready) return;
  try {
    const logStart = parseNumber("log-start", -5.0);
    const logStep = parseNumber("log-step", 0.01);
    const len = parseIntStrict("len", 2048, 2);
    const noise = parseNumber("init-noise", 0.02);
    initLearnableGrid(logStart, logStep, len, noise);
    trainLossEl.textContent = "—";
    const mode = getValue("mode") || "vertical";
    if (mode !== "vertical" || !targetGrid || !learnGrid) {
      setStatus("Learnable grid initialised.");
      void runOnce();
      return;
    }

    const real = parseNumber("s-real", 2.5);
    const imagMin = parseNumber("imag-min", -30);
    const imagMax = parseNumber("imag-max", 30);
    const imagCount = parseIntStrict("imag-count", 256, 2);
    const imagValues = linspace(imagMin, imagMax, imagCount);
    const plan: WasmMellinEvalPlan = targetGrid.planVerticalLine(real, imagValues);

    const targetMags = targetGrid.evaluatePlanMagnitude(plan);
    const learnedMags = learnGrid.evaluatePlanMagnitude(plan);
    drawMagnitudeCompare(targetMags, learnedMags, Number.NaN);
    timingsEl.textContent = `learnable init · noise=${noise.toExponential(2)} · n=${imagCount}`;
    setStatus("OK (learnable grid initialised)");
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

trainButton.addEventListener("click", () => {
  if (!ready) return;
  try {
    const logStart = parseNumber("log-start", -5.0);
    const logStep = parseNumber("log-step", 0.01);
    const len = parseIntStrict("len", 2048, 2);
    const noise = parseNumber("init-noise", 0.02);
    const steps = parseIntStrict("train-steps", 40, 1);
    const lr = parseNumber("train-lr", 0.08);

    const target = ensureTargetGrid(logStart, logStep, len);
    if (!learnGrid) {
      initLearnableGrid(logStart, logStep, len, noise);
    }
    const learned = learnGrid!;

    const real = parseNumber("s-real", 2.5);
    const imagMin = parseNumber("imag-min", -30);
    const imagMax = parseNumber("imag-max", 30);
    const imagCount = parseIntStrict("imag-count", 256, 2);
    const imagValues = linspace(imagMin, imagMax, imagCount);
    const plan: WasmMellinEvalPlan = target.planVerticalLine(real, imagValues);

    setStatus("Training…");
    const t0 = performance.now();
    let lastLoss = Number.NaN;
    const trace: MellinTrainTraceRow[] = [];
    const traceStride = Math.max(1, Math.floor(steps / 128));
    for (let i = 0; i < steps; i++) {
      lastLoss = learned.trainStepMatchGridPlan(plan, target, lr);
      const step = i + 1;
      if (step === 1 || step === steps || step % traceStride === 0) {
        trace.push({ step, loss: lastLoss });
      }
    }
    const t1 = performance.now();

    trainLossEl.textContent = Number.isFinite(lastLoss) ? lastLoss.toExponential(6) : "—";

    const targetMags = target.evaluatePlanMagnitude(plan);
    const learnedMags = learned.evaluatePlanMagnitude(plan);
    const absDiff = differenceMagnitude(targetMags, learnedMags);
    drawMagnitudeCompare(targetMags, learnedMags, lastLoss);

    timingsEl.textContent = `train ${(t1 - t0).toFixed(1)}ms · steps=${steps} · lr=${lr.toExponential(2)}`;
    setReport({
      schema: "spiraltorch.wasm.mellin_learning_report.v1",
      kind: "mellin-log-grid-training",
      createdAt: new Date().toISOString(),
      runtime: runtimeReport(),
      config: currentConfig(),
      target: {
        grid: gridSnapshot(target),
        magnitudeStats: summarize(targetMags),
      },
      learned: {
        grid: gridSnapshot(learned),
        magnitudeStats: summarize(learnedMags),
      },
      plan: planSnapshot(plan),
      training: {
        steps,
        lr,
        durationMs: t1 - t0,
        finalLoss: lastLoss,
        traceStride,
        trace,
        traceTruncated: trace.length < steps,
      },
      inferenceProbe: {
        mode: "vertical",
        sReal: real,
        imagHead: sampleValues(imagValues, 8),
        targetMagnitudeHead: sampleValues(targetMags, 16),
        learnedMagnitudeHead: sampleValues(learnedMags, 16),
        absDiffStats: summarize(absDiff),
        absDiffHead: sampleValues(absDiff, 16),
      },
    });
    setStatus("OK (trained)");
  } catch (err) {
    setStatus((err as Error).message, true);
    console.error(err);
  }
});

copyReportButton.addEventListener("click", () => {
  void copyLatestReport()
    .then(() => setStatus("Copied report JSON."))
    .catch((err) => setStatus((err as Error).message, true));
});

downloadReportButton.addEventListener("click", () => {
  try {
    downloadLatestReport();
    setStatus("Downloaded report JSON.");
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

initWasm()
  .then(() => {
    ready = true;
    setStatus("WebAssembly ready. Click “Build + Evaluate”.");
  })
  .catch((err) => {
    setStatus(`WASM init failed: ${(err as Error).message}`, true);
  });

async function initWasm(): Promise<void> {
  const url = new URL("../pkg/spiraltorch_wasm_bg.wasm", import.meta.url);
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`failed to fetch wasm (${resp.status})`);
  }
  const bytes = await resp.arrayBuffer();
  await init(bytes);
}
