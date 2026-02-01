import init, {
  WasmMellinLogGrid,
  mellin_exp_decay_samples,
} from "../pkg/spiraltorch_wasm.js";

const form = document.querySelector<HTMLFormElement>("#mellin-form")!;
const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const supportEl = document.querySelector<HTMLElement>("#support")!;
const hilbertEl = document.querySelector<HTMLElement>("#hilbert-norm")!;
const timingsEl = document.querySelector<HTMLElement>("#timings")!;
const previewEl = document.querySelector<HTMLPreElement>("#preview")!;
const canvas = document.querySelector<HTMLCanvasElement>("#plot")!;
const ctx = canvas.getContext("2d")!;

let ready = false;

function setStatus(message: string, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#ff9a9a" : "";
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

async function runOnce(ev?: Event) {
  ev?.preventDefault();
  if (!ready) return;

  try {
    setStatus("Building grid…");
    previewEl.textContent = "";

    const logStart = parseNumber("log-start", -5.0);
    const logStep = parseNumber("log-step", 0.01);
    const len = parseIntStrict("len", 2048, 2);

    const real = parseNumber("s-real", 2.5);
    const imagMin = parseNumber("imag-min", -30);
    const imagMax = parseNumber("imag-max", 30);
    const count = parseIntStrict("count", 256, 2);

    const t0 = performance.now();
    const samples = mellin_exp_decay_samples(logStart, logStep, len);
    const t1 = performance.now();
    const grid = new WasmMellinLogGrid(logStart, logStep, samples);
    const t2 = performance.now();

    const support = grid.support();
    supportEl.textContent = `[${support[0].toExponential(3)}, ${support[1].toExponential(3)}]`;
    hilbertEl.textContent = `${grid.hilbertNorm().toExponential(6)}`;

    setStatus("Evaluating…");
    const sValues = buildVerticalLineS(real, imagMin, imagMax, count);
    const t3 = performance.now();
    const values = grid.evaluateMany(sValues);
    const t4 = performance.now();

    const mags = interleavedToMagnitudes(values);
    drawMagnitude(mags);

    timingsEl.textContent = `samples ${(t1 - t0).toFixed(1)}ms · grid ${(t2 - t1).toFixed(
      1,
    )}ms · eval ${(t4 - t3).toFixed(1)}ms`;

    const head = Math.min(6, count);
    const rows: string[] = [];
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

    setStatus(`OK (n=${count})`);
  } catch (err) {
    setStatus((err as Error).message, true);
    console.error(err);
  }
}

form.addEventListener("submit", runOnce);

init()
  .then(() => {
    ready = true;
    setStatus("WebAssembly ready. Click “Build + Evaluate”.");
  })
  .catch((err) => {
    setStatus(`WASM init failed: ${(err as Error).message}`, true);
  });
