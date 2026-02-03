import init, { FractalCanvas, available_palettes } from "../pkg/spiraltorch_wasm.js";

type Mode = "hyper" | "real" | "webgpu-hyperop";
type TrailRendererMode = "2d" | "webgpu";

const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const stepEl = document.querySelector<HTMLElement>("#train-step")!;
const lossEl = document.querySelector<HTMLElement>("#loss")!;
const gradEl = document.querySelector<HTMLElement>("#grad")!;
const desireEl = document.querySelector<HTMLElement>("#desire")!;
const lrEl = document.querySelector<HTMLElement>("#lr-effective")!;
const timingEl = document.querySelector<HTMLElement>("#timing")!;

const paletteSelect = document.querySelector<HTMLSelectElement>("#palette")!;
const modeSelect = document.querySelector<HTMLSelectElement>("#mode")!;

const runToggle = document.querySelector<HTMLInputElement>("#run")!;
const useDesireToggle = document.querySelector<HTMLInputElement>("#use-desire")!;

const rebuildButton = document.querySelector<HTMLButtonElement>("#rebuild")!;
const seedButton = document.querySelector<HTMLButtonElement>("#seed")!;
const stepButton = document.querySelector<HTMLButtonElement>("#step")!;
const resetNormButton = document.querySelector<HTMLButtonElement>("#reset-norm")!;
const resetMetricsButton = document.querySelector<HTMLButtonElement>("#reset-metrics")!;
const fftRunButton = document.querySelector<HTMLButtonElement>("#fft-run")!;

const trailRendererSelect = document.querySelector<HTMLSelectElement>("#trail-renderer")!;
const fftRowInput = document.querySelector<HTMLInputElement>("#fft-row")!;
const fftAutoToggle = document.querySelector<HTMLInputElement>("#fft-auto")!;

const fractalCanvasEl = document.querySelector<HTMLCanvasElement>("#fractal")!;
const fractalCtx = fractalCanvasEl.getContext("2d")!;
const trailCanvasEl = document.querySelector<HTMLCanvasElement>("#trail")!;
const metricsCanvasEl = document.querySelector<HTMLCanvasElement>("#metrics")!;
const metricsCtx = metricsCanvasEl.getContext("2d")!;
const spectrumCanvasEl = document.querySelector<HTMLCanvasElement>("#spectrum")!;
const spectrumCtx = spectrumCanvasEl.getContext("2d")!;
let trailCtx2d: CanvasRenderingContext2D | null = null;

let ready = false;
let canvas: FractalCanvas | null = null;
let gpuDevice: GPUDevice | null = null;
let gpuInitAttempted = false;
let gpuInitFailed = false;
let gpuInitPromise: Promise<void> | null = null;
let gpuTrail: TrailWebGpuRenderer | null = null;
let gpuTrailFailed = false;
let gpuFft: WebGpuFftRow | null = null;
let gpuTrainer: WebGpuHyperTrainer | null = null;
let stepInFlight: Promise<void> | null = null;
let lastFftAt = 0;

type TrainingMetric = {
  step: number;
  mode: Mode;
  lr: number;
  lrScale: number;
  hyperRms: number;
  realRms: number;
  loss: number;
  balance: number;
  stability: number;
  saturation: number;
};

const METRICS_CAPACITY = 512;
const metricsHistory: TrainingMetric[] = [];
let trainStep = 0;
let metricsDirty = false;

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

function clampFinite(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function metricLoss(packet: any, mode: Mode): number {
  if (mode === "real") return clampFinite(packet.realgradRms);
  return clampFinite(packet.hypergradRms);
}

function recordMetrics(packet: any, mode: Mode, lr: number, lrScale: number) {
  const metric: TrainingMetric = {
    step: trainStep,
    mode,
    lr: clampFinite(lr),
    lrScale: clampFinite(lrScale),
    hyperRms: clampFinite(packet.hypergradRms),
    realRms: clampFinite(packet.realgradRms),
    loss: metricLoss(packet, mode),
    balance: clampFinite(packet.balance),
    stability: clampFinite(packet.stability),
    saturation: clampFinite(packet.saturation),
  };
  metricsHistory.push(metric);
  if (metricsHistory.length > METRICS_CAPACITY) {
    metricsHistory.splice(0, metricsHistory.length - METRICS_CAPACITY);
  }
  trainStep += 1;
  metricsDirty = true;

  stepEl.textContent = `${trainStep}`;
  lossEl.textContent = `${metric.loss.toExponential(3)} (${mode})`;
}

function resetMetrics() {
  metricsHistory.length = 0;
  trainStep = 0;
  metricsDirty = true;
  stepEl.textContent = "0";
  lossEl.textContent = "—";
}

function parseTrailRenderer(raw: string): TrailRendererMode {
  return raw === "webgpu" ? "webgpu" : "2d";
}

function randomRelation(len: number, gain = 0.25): Float32Array {
  const out = new Float32Array(len);
  if (typeof crypto !== "undefined" && "getRandomValues" in crypto) {
    const buf = new Uint32Array(len);
    crypto.getRandomValues(buf);
    for (let i = 0; i < len; i++) {
      const unit = buf[i] / 0xffffffff;
      out[i] = (unit * 2 - 1) * gain;
    }
    return out;
  }
  for (let i = 0; i < len; i++) {
    out[i] = (Math.random() * 2 - 1) * gain;
  }
  return out;
}

function ensureCanvasDimensions(width: number, height: number) {
  if (fractalCanvasEl.width !== width) fractalCanvasEl.width = width;
  if (fractalCanvasEl.height !== height) fractalCanvasEl.height = height;
  if (trailCanvasEl.width !== width) trailCanvasEl.width = width;
  if (trailCanvasEl.height !== height) trailCanvasEl.height = height;
}

function drawPixels(pixels: Uint8Array, width: number, height: number) {
  const clamped = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  const image = new ImageData(clamped, width, height);
  fractalCtx.putImageData(image, 0, 0);
}

function drawTrail2d(buffer: Float32Array, strideStep: number) {
  const ctx = trailCtx2d ?? trailCanvasEl.getContext("2d");
  if (!ctx) return;
  trailCtx2d = ctx;

  const w = trailCanvasEl.width;
  const h = trailCanvasEl.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "rgba(6, 8, 18, 0.85)";
  ctx.fillRect(0, 0, w, h);

  const stride = 7;
  const count = Math.floor(buffer.length / stride);
  if (count === 0) return;

  const step = Math.max(1, Math.floor(strideStep));
  for (let i = 0; i < count; i += step) {
    const base = i * stride;
    const x = buffer[base];
    const y = buffer[base + 1];
    const z = buffer[base + 2];
    const energy = buffer[base + 3];
    // WASM trail chroma is emitted in Z-space friendly `[-1, 1]` coordinates.
    // Remap to `[0, 1]` for display.
    const r = buffer[base + 4] * 0.5 + 0.5;
    const g = buffer[base + 5] * 0.5 + 0.5;
    const b = buffer[base + 6] * 0.5 + 0.5;

    const sx = x * w;
    const sy = y * h;
    const radius = 0.65 + Math.min(3.5, Math.max(0, Math.abs(z) * 2.0));
    const alpha = Math.min(0.9, Math.max(0.06, energy * 0.25));

    ctx.fillStyle = `rgba(${Math.floor(255 * Math.min(1, Math.max(0, r)))}, ${Math.floor(
      255 * Math.min(1, Math.max(0, g)),
    )}, ${Math.floor(255 * Math.min(1, Math.max(0, b)))}, ${alpha})`;
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function applyPalette(name: string) {
  if (!canvas) return;
  if (!name) return;
  try {
    canvas.set_palette(name);
  } catch (err) {
    console.warn("palette error", err);
  }
}

function rebuildCanvas() {
  const width = parseIntStrict("width", 512, 64);
  const height = parseIntStrict("height", 320, 64);
  const capacity = parseIntStrict("capacity", 1, 1);

  ensureCanvasDimensions(width, height);

  canvas = new FractalCanvas(capacity, width, height);
  applyPalette(paletteSelect.value || "blue-magenta");
  seedRelation();

  if (gpuTrail) {
    gpuTrail.resize(width, height);
  }
  if (gpuTrainer) {
    gpuTrainer.resize(width, height);
  }

  setStatus("OK (canvas rebuilt)");
}

function seedRelation() {
  if (!canvas) return;
  const width = parseIntStrict("width", 512, 64);
  const height = parseIntStrict("height", 320, 64);
  const coherence = parseNumber("coherence", 1.0);
  const tension = parseNumber("tension", 1.0);
  const depth = parseIntStrict("depth", 0, 0);

  const relation = randomRelation(width * height);
  canvas.push_patch(relation, coherence, tension, depth);
}

function getMode(): Mode {
  const raw = modeSelect.value;
  if (raw === "real") return "real";
  if (raw === "webgpu-hyperop") return "webgpu-hyperop";
  return "hyper";
}

function getTrailStride(): number {
  return parseIntStrict("trail-stride", 4, 1);
}

function getPointSize(): number {
  return parseNumber("point-size", 2.0);
}

function getRotateSpeed(): number {
  return parseNumber("rotate-speed", 0.6);
}

function getZScale(): number {
  return parseNumber("z-scale", 1.2);
}

function stepOnceCpu(curvature: number) {
  if (!canvas) return;

  const coherence = parseNumber("coherence", 1.0);
  const tension = parseNumber("tension", 1.0);
  const depth = parseIntStrict("depth", 0, 0);

  const baseLr = parseNumber("lr", 0.02);
  const useDesire = useDesireToggle.checked;
  const mode = getMode();

  const t0 = performance.now();
  const packet = canvas.framePacket(curvature);
  const t1 = performance.now();

  const rel = packet.relation;
  const grad =
    mode === "hyper"
      ? canvas.hypergradWaveCurrent(curvature)
      : (canvas.realgradWaveCurrent() as Float32Array);
  const t2 = performance.now();

  const lrScale =
    mode === "hyper" ? packet.hyperLearningRateScale : packet.realLearningRateScale;
  const lr = useDesire ? baseLr * lrScale : baseLr;

  const next = new Float32Array(rel.length);
  for (let i = 0; i < rel.length; i++) {
    next[i] = clampFinite(rel[i] - lr * grad[i]);
  }
  canvas.push_patch(next, coherence, tension, depth);
  const t3 = performance.now();

  gradEl.textContent = `hyper_rms=${packet.hypergradRms.toExponential(3)} real_rms=${packet.realgradRms.toExponential(
    3,
  )} (n=${packet.hypergradCount})`;
  desireEl.textContent = `balance=${packet.balance.toFixed(3)} stability=${packet.stability.toFixed(
    3,
  )} saturation=${packet.saturation.toFixed(3)}`;
  lrEl.textContent = `base=${baseLr.toExponential(2)} scale=${lrScale.toFixed(3)} → lr=${lr.toExponential(
    2,
  )}`;

  timingEl.textContent = `frame ${(t1 - t0).toFixed(2)}ms · grad ${(t2 - t1).toFixed(
    2,
  )}ms · push ${(t3 - t2).toFixed(2)}ms`;

  recordMetrics(packet, mode, lr, lrScale);
}

async function stepOnceWebGpu(curvature: number) {
  if (!canvas) return;
  await ensureGpuDevice();
  if (!gpuDevice) {
    setStatus("WebGPU not available (trainer).", true);
    return;
  }
  if (!gpuTrainer) {
    gpuTrainer = new WebGpuHyperTrainer(gpuDevice);
  }

  const coherence = parseNumber("coherence", 1.0);
  const tension = parseNumber("tension", 1.0);
  const depth = parseIntStrict("depth", 0, 0);

  const baseLr = parseNumber("lr", 0.02);
  const useDesire = useDesireToggle.checked;

  const t0 = performance.now();
  const packet = canvas.framePacket(curvature);
  const t1 = performance.now();

  const lrScale = packet.hyperLearningRateScale;
  const lr = useDesire ? baseLr * lrScale : baseLr;

  const mix = packet.operatorMix;
  const gain = packet.operatorGain;

  gpuTrainer.resize(packet.width, packet.height);
  const next = await gpuTrainer.step(canvas, packet.relation, mix, gain, lr);
  const t2 = performance.now();

  canvas.push_patch(next, coherence, tension, depth);
  const t3 = performance.now();

  gradEl.textContent = `hyper_rms=${packet.hypergradRms.toExponential(3)} real_rms=${packet.realgradRms.toExponential(
    3,
  )} (n=${packet.hypergradCount})`;
  desireEl.textContent = `balance=${packet.balance.toFixed(3)} stability=${packet.stability.toFixed(
    3,
  )} saturation=${packet.saturation.toFixed(3)}`;
  lrEl.textContent = `base=${baseLr.toExponential(2)} scale=${lrScale.toFixed(3)} → lr=${lr.toExponential(
    2,
  )}`;
  timingEl.textContent = `frame ${(t1 - t0).toFixed(2)}ms · webgpu ${(t2 - t1).toFixed(
    2,
  )}ms · push ${(t3 - t2).toFixed(2)}ms`;

  recordMetrics(packet, "webgpu-hyperop", lr, lrScale);
}

function drawMetrics() {
  const w = metricsCanvasEl.width;
  const h = metricsCanvasEl.height;
  metricsCtx.clearRect(0, 0, w, h);
  metricsCtx.fillStyle = "rgba(6, 8, 18, 0.85)";
  metricsCtx.fillRect(0, 0, w, h);

  if (metricsHistory.length < 2) {
    metricsCtx.fillStyle = "rgba(210, 218, 255, 0.65)";
    metricsCtx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
    metricsCtx.fillText("training metrics (no data yet)", 10, 18);
    return;
  }

  let max = 0;
  for (const m of metricsHistory) {
    if (m.hyperRms > max) max = m.hyperRms;
    if (m.realRms > max) max = m.realRms;
  }
  if (max <= 0) max = 1;
  const denom = Math.log1p(max);

  const padX = 10;
  const padY = 22;
  const innerW = Math.max(1, w - padX * 2);
  const innerH = Math.max(1, h - padY * 2);

  const toY = (value: number): number => {
    const v = Math.max(0, value);
    const t = denom > 0 ? Math.log1p(v) / denom : 0;
    return padY + (1 - t) * innerH;
  };

  const drawLine = (selector: (m: TrainingMetric) => number, stroke: string) => {
    metricsCtx.strokeStyle = stroke;
    metricsCtx.lineWidth = 2;
    metricsCtx.beginPath();
    const n = metricsHistory.length;
    for (let i = 0; i < n; i++) {
      const x = padX + (i / (n - 1)) * innerW;
      const y = toY(selector(metricsHistory[i]));
      if (i === 0) metricsCtx.moveTo(x, y);
      else metricsCtx.lineTo(x, y);
    }
    metricsCtx.stroke();
  };

  drawLine((m) => m.hyperRms, "rgba(120, 160, 255, 0.95)");
  drawLine((m) => m.realRms, "rgba(255, 140, 210, 0.75)");

  const last = metricsHistory[metricsHistory.length - 1];
  metricsCtx.fillStyle = "rgba(210, 218, 255, 0.82)";
  metricsCtx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  metricsCtx.fillText(
    `grad rms (log1p): hyper ${last.hyperRms.toExponential(3)} · real ${last.realRms.toExponential(
      3,
    )} (max ${max.toExponential(3)})`,
    10,
    18,
  );
}

function drawSpectrumMagnitudes(magnitudes: Float32Array) {
  const w = spectrumCanvasEl.width;
  const h = spectrumCanvasEl.height;
  spectrumCtx.clearRect(0, 0, w, h);
  spectrumCtx.fillStyle = "rgba(6, 8, 18, 0.85)";
  spectrumCtx.fillRect(0, 0, w, h);

  const n = magnitudes.length;
  if (n < 2) return;

  let max = 0;
  for (let i = 0; i < n; i++) {
    const v = magnitudes[i];
    if (Number.isFinite(v) && v > max) max = v;
  }
  if (max <= 0) max = 1;

  spectrumCtx.strokeStyle = "rgba(120, 160, 255, 0.95)";
  spectrumCtx.lineWidth = 2;
  spectrumCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const v = Math.min(1, Math.max(0, magnitudes[i] / max));
    const y = (1 - v) * (h - 24) + 12;
    if (i === 0) spectrumCtx.moveTo(x, y);
    else spectrumCtx.lineTo(x, y);
  }
  spectrumCtx.stroke();

  spectrumCtx.fillStyle = "rgba(210, 218, 255, 0.82)";
  spectrumCtx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  spectrumCtx.fillText(`row DFT |energy| (max ≈ ${max.toExponential(3)})`, 10, 18);
}

async function computeFftFromPacket(packet: { width: number; field: Float32Array }) {
  if (!canvas) return;
  await ensureGpuDevice();
  if (!gpuDevice) return;
  if (!gpuFft) gpuFft = new WebGpuFftRow(gpuDevice);

  const width = packet.width;
  const row = Math.max(0, Math.min(parseIntStrict("fft-row", 0, 0), Number.MAX_SAFE_INTEGER));
  const safeRow = Math.min(row, Math.max(0, Math.floor(packet.field.length / (width * 4)) - 1));

  const start = safeRow * width * 4;
  const end = start + width * 4;
  const fieldRow = packet.field.slice(start, end);

  const spectrum = await gpuFft.computeRow(canvas, fieldRow, width, false);
  const mags = new Float32Array(width);
  for (let i = 0; i < width; i++) {
    const re = spectrum[i * 8];
    const im = spectrum[i * 8 + 1];
    mags[i] = Math.hypot(re, im);
  }
  drawSpectrumMagnitudes(mags);
}

function frame() {
  requestAnimationFrame(frame);
  if (!ready || !canvas) return;

  const curvature = parseNumber("curvature", -1.0);
  const steps = parseIntStrict("steps", 2, 1);
  const shouldRun = runToggle.checked;
  const mode = getMode();

  if (shouldRun) {
    if (mode === "webgpu-hyperop") {
      if (!stepInFlight) {
        stepInFlight = stepOnceWebGpu(curvature).finally(() => {
          stepInFlight = null;
        });
      }
    } else {
      for (let i = 0; i < steps; i++) stepOnceCpu(curvature);
    }
  }

  try {
    const packet = canvas.framePacket(curvature);
    drawPixels(packet.pixels, packet.width, packet.height);

    const renderer = parseTrailRenderer(trailRendererSelect.value);
    const strideStep = getTrailStride();
    if (renderer === "webgpu") {
      if (gpuTrail) {
        gpuTrail.draw(packet.trail, performance.now() / 1000, {
          pointSize: getPointSize(),
          rotateSpeed: getRotateSpeed(),
          zScale: getZScale(),
          strideStep,
        });
      } else if (gpuInitFailed || gpuTrailFailed) {
        drawTrail2d(packet.trail, strideStep);
      } else {
        void ensureTrailWebGpu();
      }
    } else {
      drawTrail2d(packet.trail, strideStep);
    }

    const now = performance.now();
    if (fftAutoToggle.checked && gpuDevice && now - lastFftAt >= 350) {
      lastFftAt = now;
      void computeFftFromPacket(packet);
    }

    if (metricsDirty) {
      metricsDirty = false;
      drawMetrics();
    }
  } catch (err) {
    setStatus((err as Error).message, true);
  }
}

function populatePalettes() {
  paletteSelect.innerHTML = "";
  const entries = available_palettes() as unknown as string[];
  for (const name of entries) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    paletteSelect.appendChild(option);
  }
  if (entries.length > 0) {
    paletteSelect.value = entries[0];
  } else {
    paletteSelect.value = "blue-magenta";
  }
}

rebuildButton.addEventListener("click", () => {
  if (!ready) return;
  try {
    rebuildCanvas();
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

seedButton.addEventListener("click", () => {
  if (!ready) return;
  try {
    seedRelation();
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

stepButton.addEventListener("click", () => {
  if (!ready || !canvas) return;
  const curvature = parseNumber("curvature", -1.0);
  const mode = getMode();
  if (mode === "webgpu-hyperop") {
    void (async () => {
      try {
        await ensureGpuDevice();
        await stepOnceWebGpu(curvature);
      } catch (err) {
        setStatus((err as Error).message, true);
      }
    })();
    return;
  }
  try {
    stepOnceCpu(curvature);
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

resetNormButton.addEventListener("click", () => {
  if (!ready || !canvas) return;
  canvas.reset_normalizer();
});

resetMetricsButton.addEventListener("click", () => {
  if (!ready) return;
  resetMetrics();
});

paletteSelect.addEventListener("change", () => applyPalette(paletteSelect.value));

trailRendererSelect.addEventListener("change", async () => {
  if (!ready) return;
  if (parseTrailRenderer(trailRendererSelect.value) !== "webgpu") return;
  await ensureTrailWebGpu();
});

fftRunButton.addEventListener("click", async () => {
  if (!ready || !canvas) return;
  try {
    await ensureGpuDevice();
    const curvature = parseNumber("curvature", -1.0);
    const packet = canvas.framePacket(curvature);
    await computeFftFromPacket(packet);
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

initWasm()
  .then(() => {
    ready = true;
    resetMetrics();
    populatePalettes();
    rebuildCanvas();
    setStatus("WebAssembly ready.");
    void ensureGpuDevice().then(() => {
      if (parseTrailRenderer(trailRendererSelect.value) === "webgpu") {
        return ensureTrailWebGpu();
      }
    });
    requestAnimationFrame(frame);
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

async function ensureGpuDevice(): Promise<void> {
  if (gpuDevice) return;
  if (gpuInitFailed) return;
  if (gpuInitPromise) return gpuInitPromise;

  gpuInitPromise = (async () => {
    gpuInitAttempted = true;
    try {
      if (!("gpu" in navigator)) {
        gpuInitFailed = true;
        return;
      }
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (!adapter) {
        gpuInitFailed = true;
        return;
      }
      gpuDevice = await adapter.requestDevice();
    } catch (err) {
      gpuInitFailed = true;
      console.warn("WebGPU init failed", err);
    }
  })().finally(() => {
    gpuInitPromise = null;
  });

  return gpuInitPromise;
}

async function ensureTrailWebGpu(): Promise<void> {
  if (gpuTrail) return;
  if (gpuTrailFailed) return;
  await ensureGpuDevice();
  if (!gpuDevice) return;

  const context = trailCanvasEl.getContext("webgpu") as GPUCanvasContext | null;
  if (!context) {
    setStatus("WebGPU trail context unavailable (already using 2D context).", true);
    gpuTrailFailed = true;
    return;
  }
  gpuTrail = new TrailWebGpuRenderer(gpuDevice, context);
  if (canvas) {
    gpuTrail.resize(canvas.width, canvas.height);
  }
}

class TrailWebGpuRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private pipeline: GPURenderPipeline;
  private bindGroup: GPUBindGroup;
  private uniform: GPUBuffer;
  private vertex: GPUBuffer;
  private vertexCapacityFloats: number;
  private depthTex: GPUTexture | null = null;

  constructor(device: GPUDevice, context: GPUCanvasContext) {
    this.device = device;
    this.context = context;
    this.format = (navigator as any).gpu.getPreferredCanvasFormat();

    this.uniform = device.createBuffer({
      size: 4 * 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shader = device.createShaderModule({
      code: `
struct Params {
  time: f32,
  aspect: f32,
  point_size: f32,
  z_scale: f32,
  rotate_speed: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};
@group(0) @binding(0) var<uniform> params: Params;

struct VsIn {
  @location(0) position: vec3<f32>,
  @location(1) energy: f32,
  @location(2) color: vec3<f32>,
};

struct VsOut {
  @builtin(position) position: vec4<f32>,
  @builtin(point_size) point_size: f32,
  @location(0) color: vec3<f32>,
  @location(1) energy: f32,
};

fn rotate_y(p: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
}

fn rotate_x(p: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

@vertex
fn vs_main(input: VsIn) -> VsOut {
  let x = input.position.x * 2.0 - 1.0;
  let y = (1.0 - input.position.y) * 2.0 - 1.0;
  let z = input.position.z * params.z_scale;
  var p = vec3<f32>(x, y, z);
  let a = params.time * params.rotate_speed;
  p = rotate_y(p, a);
  p = rotate_x(p, a * 0.73);

  let camera = 2.4;
  let depth = (camera - p.z);
  let inv = 1.0 / max(0.35, depth);
  let px = (p.x * inv) / max(0.5, params.aspect);
  let py = p.y * inv;

  var out: VsOut;
  out.position = vec4<f32>(px, py, 0.0, 1.0);
  out.point_size = clamp(params.point_size + input.energy * params.point_size, 1.0, 12.0);
  out.color = clamp(input.color, vec3<f32>(0.0), vec3<f32>(1.0));
  out.energy = input.energy;
  return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
  let alpha = clamp(0.06 + input.energy * 0.35, 0.05, 0.95);
  return vec4<f32>(input.color, alpha);
}
      `,
    });

    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shader,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: 7 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 3 * 4, format: "float32" },
              { shaderLocation: 2, offset: 4 * 4, format: "float32x3" },
            ],
          },
        ],
      },
      fragment: {
        module: shader,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "point-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniform } }],
    });

    this.vertexCapacityFloats = 7 * 4096;
    this.vertex = device.createBuffer({
      size: this.vertexCapacityFloats * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.context.configure({
      device,
      format: this.format,
      alphaMode: "premultiplied",
    });
  }

  resize(width: number, height: number) {
    this.depthTex?.destroy();
    this.depthTex = this.device.createTexture({
      size: { width, height },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  draw(
    trail: Float32Array,
    timeSec: number,
    opts: { pointSize: number; rotateSpeed: number; zScale: number; strideStep: number },
  ) {
    const stride = 7;
    const count = Math.floor(trail.length / stride);
    if (count === 0) return;

    const step = Math.max(1, Math.floor(opts.strideStep));
    const vertexCount = Math.floor(count / step);
    const neededFloats = vertexCount * stride;
    if (neededFloats > this.vertexCapacityFloats) {
      this.vertex.destroy();
      this.vertexCapacityFloats = Math.ceil(neededFloats * 1.15);
      this.vertex = this.device.createBuffer({
        size: this.vertexCapacityFloats * 4,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }

    const packed = new Float32Array(neededFloats);
    let out = 0;
    for (let i = 0; i < count; i += step) {
      const base = i * stride;
      packed[out++] = trail[base]; // x
      packed[out++] = trail[base + 1]; // y
      packed[out++] = trail[base + 2]; // z
      packed[out++] = trail[base + 3]; // energy
      // WASM trail chroma is emitted in Z-space `[-1, 1]` coordinates.
      // Convert to `[0, 1]` so the shader sees valid colors.
      packed[out++] = trail[base + 4] * 0.5 + 0.5;
      packed[out++] = trail[base + 5] * 0.5 + 0.5;
      packed[out++] = trail[base + 6] * 0.5 + 0.5;
    }

    this.device.queue.writeBuffer(this.vertex, 0, packed.buffer, packed.byteOffset, packed.byteLength);

    const aspect = trailCanvasEl.width / Math.max(1, trailCanvasEl.height);
    const uniforms = new Float32Array([
      timeSec,
      aspect,
      opts.pointSize,
      opts.zScale,
      opts.rotateSpeed,
      0,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.uniform, 0, uniforms.buffer, uniforms.byteOffset, uniforms.byteLength);

    const encoder = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.03, g: 0.04, b: 0.08, a: 1.0 },
        },
      ],
      depthStencilAttachment: this.depthTex
        ? {
            view: this.depthTex.createView(),
            depthLoadOp: "clear",
            depthStoreOp: "store",
            depthClearValue: 1.0,
          }
        : undefined,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertex);
    pass.draw(vertexCount);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

class WebGpuFftRow {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private uniform: GPUBuffer;

  constructor(device: GPUDevice) {
    this.device = device;
    this.uniform = device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private ensurePipeline(canvas: FractalCanvas) {
    if (this.pipeline && this.bindGroupLayout) return;
    const wgsl = canvas.vectorFieldFftKernel(false);
    const module = this.device.createShaderModule({ code: wgsl });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipeline = pipeline;
    this.bindGroupLayout = pipeline.getBindGroupLayout(0);
  }

  async computeRow(
    canvas: FractalCanvas,
    fieldRow: Float32Array,
    width: number,
    inverse: boolean,
  ): Promise<Float32Array> {
    this.ensurePipeline(canvas);
    if (!this.pipeline || !this.bindGroupLayout) {
      throw new Error("missing FFT pipeline");
    }

    const fieldBytes = width * 4 * 4;
    const spectrumBytes = width * 8 * 4;

    const fieldBuf = this.device.createBuffer({
      size: fieldBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const spectrumBuf = this.device.createBuffer({
      size: spectrumBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      size: spectrumBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const params = new Uint32Array([width, 1, inverse ? 1 : 0, 0]);
    this.device.queue.writeBuffer(this.uniform, 0, params.buffer, params.byteOffset, params.byteLength);
    this.device.queue.writeBuffer(fieldBuf, 0, fieldRow.buffer, fieldRow.byteOffset, fieldRow.byteLength);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: fieldBuf } },
        { binding: 1, resource: { buffer: spectrumBuf } },
        { binding: 2, resource: { buffer: this.uniform } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroup = 64;
    pass.dispatchWorkgroups(Math.ceil(width / workgroup), 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(spectrumBuf, 0, readback, 0, spectrumBytes);
    this.device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange();
    const out = new Float32Array(copy.slice(0));
    readback.unmap();

    fieldBuf.destroy();
    spectrumBuf.destroy();
    readback.destroy();
    return out;
  }
}

class WebGpuHyperTrainer {
  private device: GPUDevice;
  private pipelineHyper: GPUComputePipeline | null = null;
  private pipelineUpdate: GPUComputePipeline | null = null;
  private uniformHyper: GPUBuffer;
  private uniformUpdate: GPUBuffer;
  private len = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.uniformHyper = device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformUpdate = device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  resize(width: number, height: number) {
    this.len = width * height;
  }

  private ensurePipelines(canvas: FractalCanvas) {
    if (!this.pipelineHyper) {
      const hyperWgsl = canvas.hypergradOperatorKernel(false);
      const module = this.device.createShaderModule({ code: hyperWgsl });
      this.pipelineHyper = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }
    if (!this.pipelineUpdate) {
      const module = this.device.createShaderModule({
        code: `
struct UpdateParams {
  lr: f32,
  len: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(0) var<storage, read> relation: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: UpdateParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.len) { return; }
  out[idx] = relation[idx] - params.lr * grad[idx];
}
        `,
      });
      this.pipelineUpdate = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }
  }

  async step(
    canvas: FractalCanvas,
    relation: Float32Array,
    mix: number,
    gain: number,
    lr: number,
  ): Promise<Float32Array> {
    this.ensurePipelines(canvas);
    if (!this.pipelineHyper || !this.pipelineUpdate) {
      throw new Error("missing WebGPU pipelines");
    }
    const len = relation.length;
    if (len === 0) return relation;

    const bytes = len * 4;
    const relationBuf = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const hyperBuf = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outBuf = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(relationBuf, 0, relation.buffer, relation.byteOffset, relation.byteLength);
    this.device.queue.writeBuffer(hyperBuf, 0, new Float32Array(len).buffer);

    const hyperParams = new Float32Array([canvas.width, canvas.height, mix, gain]);
    this.device.queue.writeBuffer(this.uniformHyper, 0, hyperParams.buffer, hyperParams.byteOffset, hyperParams.byteLength);
    const updateParams = new Uint32Array(new Float32Array([lr]).buffer);
    const updateU32 = new Uint32Array([updateParams[0], len, 0, 0]);
    this.device.queue.writeBuffer(this.uniformUpdate, 0, updateU32.buffer, updateU32.byteOffset, updateU32.byteLength);

    const bindHyper = this.device.createBindGroup({
      layout: this.pipelineHyper.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: relationBuf } },
        { binding: 1, resource: { buffer: hyperBuf } },
        { binding: 2, resource: { buffer: this.uniformHyper } },
      ],
    });
    const bindUpdate = this.device.createBindGroup({
      layout: this.pipelineUpdate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: relationBuf } },
        { binding: 1, resource: { buffer: hyperBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: this.uniformUpdate } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelineHyper);
      pass.setBindGroup(0, bindHyper);
      const workgroup = 64;
      pass.dispatchWorkgroups(Math.ceil(canvas.width / workgroup), canvas.height, 1);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelineUpdate);
      pass.setBindGroup(0, bindUpdate);
      pass.dispatchWorkgroups(Math.ceil(len / 256), 1, 1);
      pass.end();
    }
    encoder.copyBufferToBuffer(outBuf, 0, readback, 0, bytes);
    this.device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const out = new Float32Array(mapped.slice(0));
    readback.unmap();

    relationBuf.destroy();
    hyperBuf.destroy();
    outBuf.destroy();
    readback.destroy();
    return out;
  }
}
