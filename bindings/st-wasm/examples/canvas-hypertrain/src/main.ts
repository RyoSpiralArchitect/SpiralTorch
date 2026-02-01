import init, { FractalCanvas, available_palettes } from "../pkg/spiraltorch_wasm.js";

type Mode = "hyper" | "real";

const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
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

const fractalCanvasEl = document.querySelector<HTMLCanvasElement>("#fractal")!;
const trailCanvasEl = document.querySelector<HTMLCanvasElement>("#trail")!;
const trailCtx = trailCanvasEl.getContext("2d")!;

let ready = false;
let canvas: FractalCanvas | null = null;

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

function drawTrail(buffer: Float32Array) {
  const w = trailCanvasEl.width;
  const h = trailCanvasEl.height;
  trailCtx.clearRect(0, 0, w, h);

  trailCtx.fillStyle = "rgba(6, 8, 18, 0.85)";
  trailCtx.fillRect(0, 0, w, h);

  const stride = 7;
  const count = Math.floor(buffer.length / stride);
  if (count === 0) return;

  for (let i = 0; i < count; i++) {
    const base = i * stride;
    const x = buffer[base];
    const y = buffer[base + 1];
    const z = buffer[base + 2];
    const energy = buffer[base + 3];
    const r = buffer[base + 4];
    const g = buffer[base + 5];
    const b = buffer[base + 6];

    const sx = (x * 0.5 + 0.5) * w;
    const sy = (1 - (y * 0.5 + 0.5)) * h;
    const radius = 0.75 + Math.min(3.5, Math.max(0, Math.abs(z) * 2.0));
    const alpha = Math.min(0.9, Math.max(0.08, energy * 0.25));

    trailCtx.fillStyle = `rgba(${Math.floor(255 * Math.min(1, Math.max(0, r)))}, ${Math.floor(
      255 * Math.min(1, Math.max(0, g)),
    )}, ${Math.floor(255 * Math.min(1, Math.max(0, b)))}, ${alpha})`;
    trailCtx.beginPath();
    trailCtx.arc(sx, sy, radius, 0, Math.PI * 2);
    trailCtx.fill();
  }
}

function applyPalette(name: string) {
  if (!canvas) return;
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
  applyPalette(paletteSelect.value || "midnight");
  seedRelation();
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

function stepOnce() {
  if (!canvas) return;

  const coherence = parseNumber("coherence", 1.0);
  const tension = parseNumber("tension", 1.0);
  const depth = parseIntStrict("depth", 0, 0);

  const curvature = parseNumber("curvature", -1.0);
  const baseLr = parseNumber("lr", 0.02);
  const useDesire = useDesireToggle.checked;
  const mode = (modeSelect.value as Mode) ?? "hyper";

  const t0 = performance.now();
  const rel = canvas.relation();
  const t1 = performance.now();

  const grad =
    mode === "hyper" ? canvas.hypergradWave(curvature) : (canvas.realgradWave() as Float32Array);
  const t2 = performance.now();

  const control = canvas.desireControl(curvature);
  const lrScale =
    mode === "hyper" ? control.hyperLearningRateScale : control.realLearningRateScale;
  const lr = useDesire ? baseLr * lrScale : baseLr;

  const next = new Float32Array(rel.length);
  for (let i = 0; i < rel.length; i++) {
    next[i] = clampFinite(rel[i] - lr * grad[i]);
  }
  canvas.push_patch(next, coherence, tension, depth);
  const t3 = performance.now();

  const summary = canvas.gradientSummary(curvature);
  gradEl.textContent = `hyper_rms=${summary.hypergradRms.toExponential(3)} real_rms=${summary.realgradRms.toExponential(
    3,
  )} (n=${summary.hypergradCount})`;

  const desire = canvas.desireInterpretation(curvature);
  desireEl.textContent = `balance=${desire.balance.toFixed(3)} stability=${desire.stability.toFixed(
    3,
  )} saturation=${desire.saturation.toFixed(3)}`;

  lrEl.textContent = `base=${baseLr.toExponential(2)} scale=${lrScale.toFixed(3)} → lr=${lr.toExponential(
    2,
  )}`;

  timingEl.textContent = `relation ${(t1 - t0).toFixed(2)}ms · grad ${(t2 - t1).toFixed(
    2,
  )}ms · push ${(t3 - t2).toFixed(2)}ms`;
}

function frame() {
  requestAnimationFrame(frame);
  if (!ready || !canvas) return;

  const curvature = parseNumber("curvature", -1.0);
  const steps = parseIntStrict("steps", 2, 1);
  const shouldRun = runToggle.checked;

  if (shouldRun) {
    for (let i = 0; i < steps; i++) stepOnce();
  }

  try {
    canvas.render_to_canvas(fractalCanvasEl);
    const trail = canvas.emitWasmTrail(curvature);
    drawTrail(trail);
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
  paletteSelect.value = "midnight";
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
  if (!ready) return;
  try {
    stepOnce();
  } catch (err) {
    setStatus((err as Error).message, true);
  }
});

resetNormButton.addEventListener("click", () => {
  if (!ready || !canvas) return;
  canvas.reset_normalizer();
});

paletteSelect.addEventListener("change", () => applyPalette(paletteSelect.value));

init()
  .then(() => {
    ready = true;
    populatePalettes();
    rebuildCanvas();
    setStatus("WebAssembly ready.");
    requestAnimationFrame(frame);
  })
  .catch((err) => {
    setStatus(`WASM init failed: ${(err as Error).message}`, true);
  });
