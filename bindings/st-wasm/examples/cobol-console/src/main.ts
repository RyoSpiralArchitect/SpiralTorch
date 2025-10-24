import init, { CobolDispatchPlanner } from "../../pkg/spiraltorch_wasm.js";

const form = document.querySelector<HTMLFormElement>("#planner-form")!;
const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const jsonPreview = document.querySelector<HTMLPreElement>("#json-preview")!;
const cobolPreview = document.querySelector<HTMLPreElement>("#cobol-preview")!;
const payloadBytes = document.querySelector<HTMLSpanElement>("#payload-bytes")!;
const sendButton = document.querySelector<HTMLButtonElement>("#send-envelope")!;
const seedButton = document.querySelector<HTMLButtonElement>("#seed-coeffs")!;

let ready = false;

function setStatus(message: string, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#ff9a9a" : "";
}

function parseCoefficients(raw: string): Float32Array {
  const tokens = raw
    .split(/[\s,]+/g)
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
  const floats = tokens.map((token) => {
    const parsed = Number.parseFloat(token);
    if (!Number.isFinite(parsed)) {
      throw new Error(`Invalid coefficient: ${token}`);
    }
    return parsed;
  });
  return new Float32Array(floats);
}

function parsePositiveInteger(raw: string, label: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseMetadata(raw: string): unknown | null {
  const trimmed = raw.trim();
  if (!trimmed) {
    return null;
  }
  try {
    return JSON.parse(trimmed);
  } catch (err) {
    throw new Error(`Metadata JSON error: ${(err as Error).message}`);
  }
}

function getInputValue(id: string): string {
  return (document.querySelector<HTMLInputElement>(`#${id}`)?.value ?? "").trim();
}

function getTextareaValue(id: string): string {
  return (document.querySelector<HTMLTextAreaElement>(`#${id}`)?.value ?? "").trim();
}

function buildPlanner(): CobolDispatchPlanner {
  const jobId = getInputValue("job-id") || "demo-lab";
  const channel = getInputValue("release-channel") || undefined;
  const planner = new CobolDispatchPlanner(jobId, channel ?? null);

  const curvature = Number.parseFloat(getInputValue("curvature") || "0.5");
  const temperature = Number.parseFloat(getInputValue("temperature") || "0.5");
  const encoder = getInputValue("encoder") || "st-language.wave";
  const locale = getInputValue("locale") || undefined;
  planner.setNarratorConfig(curvature, temperature, encoder, locale ?? null);

  const coefficientText = getTextareaValue("coefficients");
  if (coefficientText) {
    const buffer = parseCoefficients(coefficientText);
    planner.setCoefficients(buffer);
  }

  const humanName = getInputValue("human-name");
  if (humanName) {
    planner.addHumanInitiator(humanName, null, null, "browser curators");
  }

  const modelName = getInputValue("model-name");
  if (modelName) {
    planner.addModelInitiator(modelName, "2025-05", "lm-agent", "WASM UI");
  }

  const mqManager = getInputValue("mq-manager");
  const mqQueue = getInputValue("mq-queue");
  if (mqManager && mqQueue) {
    const commit = getInputValue("mq-commit") || undefined;
    planner.setMqRoute(mqManager, mqQueue, commit ?? null);
  }

  const cicsTran = getInputValue("cics-transaction");
  if (cicsTran) {
    const cicsProgram = getInputValue("cics-program") || undefined;
    const cicsChannel = getInputValue("cics-channel") || undefined;
    planner.setCicsRoute(cicsTran, cicsProgram ?? null, cicsChannel ?? null);
  }

  const dataset = getInputValue("dataset");
  planner.setDataset(dataset || undefined);
  const datasetMember = getInputValue("dataset-member");
  planner.setDatasetMember(datasetMember || undefined);
  const datasetDisposition = getInputValue("dataset-disposition");
  planner.setDatasetDisposition(datasetDisposition || undefined);
  const datasetVolume = getInputValue("dataset-volume");
  planner.setDatasetVolume(datasetVolume || undefined);
  const datasetRecordFormat = getInputValue("dataset-record-format");
  planner.setDatasetRecordFormat(datasetRecordFormat || undefined);
  const datasetRecordLength = parsePositiveInteger(
    getInputValue("dataset-record-length"),
    "Record length",
  );
  planner.setDatasetRecordLength(datasetRecordLength ?? undefined);
  const datasetBlockSize = parsePositiveInteger(
    getInputValue("dataset-block-size"),
    "Block size",
  );
  planner.setDatasetBlockSize(datasetBlockSize ?? undefined);
  const datasetDataClass = getInputValue("dataset-data-class");
  planner.setDatasetDataClass(datasetDataClass || undefined);
  const datasetManagementClass = getInputValue("dataset-management-class");
  planner.setDatasetManagementClass(datasetManagementClass || undefined);
  const datasetStorageClass = getInputValue("dataset-storage-class");
  planner.setDatasetStorageClass(datasetStorageClass || undefined);
  const datasetSpacePrimary = parsePositiveInteger(
    getInputValue("dataset-space-primary"),
    "Primary space",
  );
  planner.setDatasetSpacePrimary(datasetSpacePrimary ?? undefined);
  const datasetSpaceSecondary = parsePositiveInteger(
    getInputValue("dataset-space-secondary"),
    "Secondary space",
  );
  planner.setDatasetSpaceSecondary(datasetSpaceSecondary ?? undefined);
  const datasetSpaceUnit = getInputValue("dataset-space-unit");
  planner.setDatasetSpaceUnit(datasetSpaceUnit || undefined);
  const datasetDirectoryBlocks = parsePositiveInteger(
    getInputValue("dataset-directory-blocks"),
    "Directory blocks",
  );
  planner.setDatasetDirectoryBlocks(datasetDirectoryBlocks ?? undefined);
  const datasetType = getInputValue("dataset-type");
  planner.setDatasetType(datasetType || undefined);
  const datasetLike = getInputValue("dataset-like");
  planner.setDatasetLike(datasetLike || undefined);
  const datasetUnit = getInputValue("dataset-unit");
  planner.setDatasetUnit(datasetUnit || undefined);
  const datasetAverageRecordUnit = getInputValue("dataset-average-record-unit");
  planner.setDatasetAverageRecordUnit(datasetAverageRecordUnit || undefined);
  const retentionDays = parsePositiveInteger(
    getInputValue("dataset-retention-period"),
    "Retention days",
  );
  planner.setDatasetRetentionPeriod(retentionDays ?? undefined);
  const releaseCheckbox = document.querySelector<HTMLInputElement>(
    "#dataset-release-space",
  );
  planner.setDatasetReleaseSpace(releaseCheckbox?.checked ? true : undefined);
  const expirationDate = getInputValue("dataset-expiration-date");
  planner.setDatasetExpirationDate(expirationDate || undefined);

  const metadata = getTextareaValue("metadata");
  if (metadata) {
    const parsed = parseMetadata(metadata);
    if (parsed != null) {
      planner.mergeMetadata(parsed);
    }
  }

  planner.addTag("wasm-web-ui");
  planner.addAnnotation("generated-from-browser");

  return planner;
}

function updatePreview(planner: CobolDispatchPlanner) {
  try {
    const json = planner.toJson();
    jsonPreview.textContent = json;
  } catch (err) {
    jsonPreview.textContent = `Failed to render JSON: ${(err as Error).message}`;
  }

  try {
    const preview = planner.toCobolPreview();
    cobolPreview.textContent = JSON.stringify(preview, null, 2);
  } catch (err) {
    cobolPreview.textContent = `Failed to render COBOL preview: ${(err as Error).message}`;
  }

  try {
    const payload = planner.toUint8Array();
    payloadBytes.textContent = payload.length.toString();
  } catch (err) {
    payloadBytes.textContent = "0";
    throw err;
  }
}

async function handleSubmit(event: SubmitEvent) {
  event.preventDefault();
  try {
    const planner = buildPlanner();
    updatePreview(planner);
    setStatus("Envelope generated. Inspect the preview before dispatching.");
  } catch (err) {
    setStatus((err as Error).message, true);
  }
}

async function handleSend() {
  try {
    const planner = buildPlanner();
    updatePreview(planner);
    const endpoint = getInputValue("bridge-endpoint");
    if (!endpoint) {
      setStatus("Provide an HTTP bridge endpoint before sending.", true);
      return;
    }
    const payload = planner.toUint8Array();
    const envelope = planner.toObject() as { release_channel?: string };
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-SpiralTorch-Release": envelope.release_channel ?? "",
      },
      body: payload,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Bridge responded with ${response.status}: ${text}`);
    }
    setStatus("Envelope dispatched to bridge.");
  } catch (err) {
    setStatus((err as Error).message, true);
  }
}

function seedCoefficients() {
  const samples = Array.from({ length: 24 }, (_, index) => (
    Math.sin(index / 3.2) * 0.42 + Math.cos(index / 2.7) * 0.18
  ));
  const field = document.querySelector<HTMLTextAreaElement>("#coefficients");
  if (field) {
    field.value = samples.map((value) => value.toFixed(6)).join(", ");
  }
  setStatus("Loaded demo coefficient buffer. Adjust and generate when ready.");
}

async function bootstrap() {
  setStatus("Loading WebAssembly runtime…");
  await init();
  ready = true;
  setStatus("Runtime ready. Configure the planner and generate an envelope.");
}

form.addEventListener("submit", handleSubmit);
sendButton.addEventListener("click", () => {
  if (!ready) {
    setStatus("WASM runtime still loading…", true);
    return;
  }
  handleSend();
});
seedButton.addEventListener("click", seedCoefficients);

bootstrap().catch((err) => {
  setStatus(`Failed to initialize WASM: ${(err as Error).message}`, true);
});
