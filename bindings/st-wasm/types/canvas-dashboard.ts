import { available_palettes, type CanvasPaletteName } from "spiraltorch-wasm";
import type { CanvasPaletteCanonicalName } from "spiraltorch-wasm";
import {
    SpiralCanvasView,
    type CustomPalette,
    type CanvasStatsEvent,
    type CanvasFrameEvent,
    type CanvasPointerEvent,
} from "./canvas-view";
import { SpiralCanvasRecorder, type SpiralCanvasRecorderOptions } from "./canvas-recorder";
import type {
    SpiralCanvasCollabSession,
    CanvasCollabParticipantState,
} from "./canvas-collab";

/** Options used to configure {@link SpiralCanvasDashboard}. */
export interface SpiralCanvasDashboardOptions {
    /** Map of friendly palette names to custom palette definitions. */
    customPalettes?: Record<string, CustomPalette>;
    /** Whether to render the built-in palette selector. Defaults to `true`. */
    showPaletteSelector?: boolean;
    /** Whether to render the pointer navigation toggle. Defaults to `true`. */
    showPointerToggle?: boolean;
    /** Whether to render the reset view button. Defaults to `true`. */
    showResetView?: boolean;
    /** Whether to expose snapshot and recording buttons. Defaults to `true`. */
    showSnapshot?: boolean;
    /** Optional file name used when downloading snapshots. */
    snapshotFilename?: string;
    /** MIME type forwarded to {@link SpiralCanvasView#toBlob}. Defaults to PNG. */
    snapshotMimeType?: string;
    /** Quality forwarded to {@link SpiralCanvasView#toBlob}. */
    snapshotQuality?: number;
    /** Callback invoked with the captured snapshot {@link Blob}. */
    onSnapshot?: (blob: Blob) => void | Promise<void>;
    /** Callback invoked when snapshot capture fails. */
    onSnapshotError?: (error: unknown) => void;
    /** Whether to expose a recording toggle. Disabled when unsupported. */
    showRecorder?: boolean;
    /** Options forwarded to {@link SpiralCanvasRecorder}. */
    recorderOptions?: SpiralCanvasRecorderOptions;
    /** Callback invoked once a recording blob has been assembled. */
    onRecordingComplete?: (blob: Blob) => void | Promise<void>;
    /** Callback invoked when recording fails to start or stop. */
    onRecordingError?: (error: unknown) => void;
    /** Minimum curvature value exposed by the slider. Defaults to `0.1`. */
    curvatureMin?: number;
    /** Maximum curvature value exposed by the slider. Defaults to `4`. */
    curvatureMax?: number;
    /** Step value for the curvature slider. Defaults to `0.05`. */
    curvatureStep?: number;
    /** Number of decimal digits displayed for stats values. Defaults to `3`. */
    statsDigits?: number;
    /** Inject default dashboard styles into the document head. Defaults to `true`. */
    injectStyles?: boolean;
}

const DASHBOARD_STYLE_ID = "spiraltorch-dashboard-style";
const ROLE_COLORS: Record<string, string> = {
    trainer: "#6366f1",
    model: "#22d3ee",
    human: "#f97316",
};

const DEFAULT_STYLES = `
.spiraltorch-dashboard {
    font-family: system-ui, sans-serif;
    font-size: 12px;
    line-height: 1.4;
    color: #e5e7eb;
    background: rgba(13, 18, 32, 0.92);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
    width: 280px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.spiraltorch-dashboard button,
.spiraltorch-dashboard select,
.spiraltorch-dashboard input[type="range"],
.spiraltorch-dashboard label {
    font: inherit;
}

.spiraltorch-dashboard header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.spiraltorch-dashboard header h1 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
}

.spiraltorch-dashboard .stats-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 8px;
}

.spiraltorch-dashboard .stats-grid dt {
    opacity: 0.7;
}

.spiraltorch-dashboard .stats-grid dd {
    margin: 0;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.spiraltorch-dashboard .controls {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.spiraltorch-dashboard .control-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.spiraltorch-dashboard .control-row label {
    flex: 1 0 auto;
}

.spiraltorch-dashboard .control-row select,
.spiraltorch-dashboard .control-row input[type="checkbox"],
.spiraltorch-dashboard .control-row button {
    flex: 0 0 auto;
}

.spiraltorch-dashboard .actions-row {
    display: flex;
    gap: 8px;
}

.spiraltorch-dashboard .actions-row button {
    flex: 1 1 auto;
}

.spiraltorch-dashboard .actions-row button.recording-active {
    background: #ef4444;
    color: #fff;
}

.spiraltorch-dashboard .actions-row .recording-indicator {
    flex: 0 0 auto;
    font-variant-numeric: tabular-nums;
    opacity: 0.75;
}

.spiraltorch-dashboard .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
}

.spiraltorch-dashboard .slider-row span {
    font-variant-numeric: tabular-nums;
    min-width: 48px;
    text-align: right;
}

.spiraltorch-dashboard .participants {
    display: none;
    flex-direction: column;
    gap: 6px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 8px;
}

.spiraltorch-dashboard .participants h2 {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}

.spiraltorch-dashboard .participants-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.spiraltorch-dashboard .participant-row {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 4px 6px;
    border-radius: 6px;
    background: rgba(148, 163, 184, 0.08);
}

.spiraltorch-dashboard .participant-header {
    display: flex;
    align-items: center;
    gap: 6px;
}

.spiraltorch-dashboard .participant-indicator {
    width: 8px;
    height: 8px;
    border-radius: 9999px;
    background: var(--indicator, #22d3ee);
    box-shadow: 0 0 0 2px rgba(148, 163, 184, 0.18);
}

.spiraltorch-dashboard .participant-label {
    font-weight: 600;
}

.spiraltorch-dashboard .participant-meta,
.spiraltorch-dashboard .participant-pointer {
    font-variant-numeric: tabular-nums;
    opacity: 0.75;
}
`;

const DEFAULT_STATS: Array<{ key: string; label: string; read: (event: CanvasStatsEvent) => number }> = [
    {
        key: "hypergradRms",
        label: "Hyper RMS",
        read: (event) => event.summary.hypergradRms,
    },
    {
        key: "realgradRms",
        label: "Real RMS",
        read: (event) => event.summary.realgradRms,
    },
    {
        key: "penaltyGain",
        label: "Penalty gain",
        read: (event) => event.control.penaltyGain,
    },
    {
        key: "operatorMix",
        label: "Operator mix",
        read: (event) => event.control.operatorMix,
    },
    {
        key: "qualityGain",
        label: "Quality gain",
        read: (event) => event.control.qualityGain,
    },
];

function ensureStyles(doc: Document, inject: boolean): void {
    if (!inject) {
        return;
    }
    if (doc.getElementById(DASHBOARD_STYLE_ID)) {
        return;
    }
    const style = doc.createElement("style");
    style.id = DASHBOARD_STYLE_ID;
    style.textContent = DEFAULT_STYLES;
    if (doc.head) {
        doc.head.appendChild(style);
    }
}

function safeAvailablePalettes(): CanvasPaletteCanonicalName[] {
    try {
        return available_palettes();
    } catch {
        return ["blue-magenta", "turbo", "grayscale"];
    }
}

function formatNumber(value: number, digits: number): string {
    if (!Number.isFinite(value)) {
        return "–";
    }
    const abs = Math.abs(value);
    const precision = Math.max(0, Math.floor(digits));
    if (abs !== 0 && (abs < 0.001 || abs >= 1000)) {
        return value.toExponential(precision);
    }
    return value.toFixed(precision);
}

/**
 * Small vanilla UI helper that exposes real-time stats, palette controls and
 * navigation toggles for {@link SpiralCanvasView} instances.
 */
export class SpiralCanvasDashboard {
    readonly #view: SpiralCanvasView;
    readonly #root: HTMLElement;
    readonly #customPalettes = new Map<string, CustomPalette>();
    readonly #customPaletteLookup = new WeakMap<object, string>();
    readonly #statsDigits: number;
    readonly #options: SpiralCanvasDashboardOptions;

    #paletteSelect?: HTMLSelectElement;
    #curvatureSlider?: HTMLInputElement;
    #curvatureLabel?: HTMLSpanElement;
    #pointerToggle?: HTMLInputElement;
    #snapshotButton?: HTMLButtonElement;
    #recordButton?: HTMLButtonElement;
    #recordingIndicator?: HTMLElement;
    #fpsField: HTMLElement;
    #zoomField: HTMLElement;
    #offsetField: HTMLElement;
    #paletteField: HTMLElement;
    #statFields = new Map<string, HTMLElement>();

    #participantsContainer?: HTMLElement;
    #participantsList?: HTMLElement;
    #collabSession: SpiralCanvasCollabSession | null = null;
    #collabParticipantsHandler?: (event: { participants: CanvasCollabParticipantState[] }) => void;
    #collabStateHandler?: (event: {
        participant: CanvasCollabParticipantState;
        origin: "local" | "remote";
    }) => void;

    #frameHandler: (event: CanvasFrameEvent) => void;
    #statsHandler: (event: CanvasStatsEvent) => void;
    #pointerHandler: (event: CanvasPointerEvent) => void;
    #recorder: SpiralCanvasRecorder | null = null;
    #recordingStart = 0;

    constructor(container: HTMLElement, view: SpiralCanvasView, options: SpiralCanvasDashboardOptions = {}) {
        if (!container) {
            throw new Error("SpiralCanvasDashboard requires a container element");
        }
        if (!container.ownerDocument) {
            throw new Error("Dashboard container must be attached to a document");
        }
        this.#view = view;
        this.#options = options;
        const digits = options.statsDigits ?? 3;
        this.#statsDigits = Number.isFinite(digits) ? Math.max(0, Math.round(digits)) : 3;

        ensureStyles(container.ownerDocument, options.injectStyles ?? true);

        this.#root = container.ownerDocument.createElement("section");
        this.#root.className = "spiraltorch-dashboard";
        container.appendChild(this.#root);

        this.#ingestCustomPalettes(options.customPalettes);
        this.#buildHeader(options);
        this.#buildControls(options);
        this.#buildStats();
        this.#buildParticipantsPane();

        const palette = this.#view.palette;
        this.#updatePaletteField(palette);
        this.#syncPaletteSelector(palette);
        this.#syncPointerToggle();
        this.#syncCurvatureControls();
        this.#zoomField.textContent = formatNumber(this.#view.zoom, this.#statsDigits);
        const offset = this.#view.offset;
        this.#setOffsetReadout(offset.x, offset.y);

        this.#frameHandler = (event) => {
            this.#updateFrame(event);
        };
        this.#statsHandler = (event) => {
            this.#updateStats(event);
        };
        this.#pointerHandler = (event) => {
            this.#zoomField.textContent = formatNumber(event.zoom, this.#statsDigits);
            this.#setOffsetReadout(event.offset.x, event.offset.y);
        };

        this.#view.on("frame", this.#frameHandler);
        this.#view.on("stats", this.#statsHandler);
        this.#view.on("pointer", this.#pointerHandler);
    }

    /** Root DOM element created by the dashboard. */
    get element(): HTMLElement {
        return this.#root;
    }

    /** Removes event listeners and detaches the dashboard from the DOM. */
    destroy(): void {
        this.#view.off("frame", this.#frameHandler);
        this.#view.off("stats", this.#statsHandler);
        this.#view.off("pointer", this.#pointerHandler);
        this.detachCollaboration();
        if (this.#recorder?.recording) {
            this.#recorder.cancel();
        }
        this.#recorder = null;
        this.#recordButton?.classList.remove("recording-active");
        if (this.#recordingIndicator) {
            this.#recordingIndicator.textContent = "";
        }
        this.#root.remove();
    }

    /**
     * Connects the dashboard to a {@link SpiralCanvasCollabSession} so that
     * participants and remote activity are surfaced within the UI.
     */
    attachCollaboration(session: SpiralCanvasCollabSession): void {
        if (this.#collabSession === session) {
            this.#renderParticipants(session.participants);
            return;
        }
        this.detachCollaboration();
        this.#collabSession = session;
        if (!this.#participantsContainer || !this.#participantsList) {
            this.#buildParticipantsPane();
        }
        if (this.#participantsContainer) {
            this.#participantsContainer.style.display = "flex";
        }
        this.#collabParticipantsHandler = (event) => {
            this.#renderParticipants(event.participants);
        };
        this.#collabStateHandler = () => {
            this.#renderParticipants(session.participants);
        };
        session.on("participants", this.#collabParticipantsHandler);
        session.on("state", this.#collabStateHandler);
        this.#renderParticipants(session.participants);
    }

    /**
     * Detaches the dashboard from any active collaboration session.
     */
    detachCollaboration(): void {
        if (!this.#collabSession) {
            return;
        }
        if (this.#collabParticipantsHandler) {
            this.#collabSession.off("participants", this.#collabParticipantsHandler);
            this.#collabParticipantsHandler = undefined;
        }
        if (this.#collabStateHandler) {
            this.#collabSession.off("state", this.#collabStateHandler);
            this.#collabStateHandler = undefined;
        }
        this.#collabSession = null;
        if (this.#participantsContainer) {
            this.#participantsContainer.style.display = "none";
        }
        if (this.#participantsList) {
            this.#participantsList.innerHTML = "";
        }
    }

    #buildHeader(options: SpiralCanvasDashboardOptions): void {
        const header = this.#root.ownerDocument!.createElement("header");
        const title = this.#root.ownerDocument!.createElement("h1");
        title.textContent = "SpiralTorch";
        header.appendChild(title);

        if (options.showResetView ?? true) {
            const reset = this.#root.ownerDocument!.createElement("button");
            reset.type = "button";
            reset.textContent = "Reset view";
            reset.addEventListener("click", () => {
                this.#view.setZoom(1);
                this.#view.setOffset(0, 0);
            });
            header.appendChild(reset);
        }

        this.#root.appendChild(header);
    }

    #buildControls(options: SpiralCanvasDashboardOptions): void {
        const controls = this.#root.ownerDocument!.createElement("div");
        controls.className = "controls";

        if (options.showPaletteSelector ?? true) {
            const row = this.#root.ownerDocument!.createElement("div");
            row.className = "control-row";
            const label = this.#root.ownerDocument!.createElement("label");
            label.textContent = "Palette";
            const select = this.#root.ownerDocument!.createElement("select");
            select.addEventListener("change", () => {
                const value = select.value;
                if (this.#customPalettes.has(value)) {
                    const palette = this.#customPalettes.get(value)!;
                    this.#view.setPalette(palette);
                    this.#updatePaletteField(palette);
                } else {
                    this.#view.setPalette(value as CanvasPaletteName);
                    this.#updatePaletteField(value as CanvasPaletteCanonicalName);
                }
            });
            this.#paletteSelect = select;
            this.#populatePaletteOptions();
            row.appendChild(label);
            row.appendChild(select);
            controls.appendChild(row);
        }

        const sliderRow = this.#root.ownerDocument!.createElement("div");
        sliderRow.className = "slider-row";
        const sliderLabel = this.#root.ownerDocument!.createElement("label");
        sliderLabel.textContent = "Curvature";
        const slider = this.#root.ownerDocument!.createElement("input");
        slider.type = "range";
        slider.min = String(options.curvatureMin ?? 0.1);
        slider.max = String(options.curvatureMax ?? 4);
        slider.step = String(options.curvatureStep ?? 0.05);
        slider.addEventListener("input", () => {
            const value = Number.parseFloat(slider.value);
            this.#view.setStatsCurvature(value);
            this.#updateCurvatureLabel();
        });
        this.#curvatureSlider = slider;
        const curvatureValue = this.#root.ownerDocument!.createElement("span");
        this.#curvatureLabel = curvatureValue;

        sliderRow.appendChild(sliderLabel);
        sliderRow.appendChild(slider);
        sliderRow.appendChild(curvatureValue);
        controls.appendChild(sliderRow);

        if (options.showPointerToggle ?? true) {
            const row = this.#root.ownerDocument!.createElement("div");
            row.className = "control-row";
            const label = this.#root.ownerDocument!.createElement("label");
            label.textContent = "Pointer navigation";
            const toggle = this.#root.ownerDocument!.createElement("input");
            toggle.type = "checkbox";
            toggle.addEventListener("change", () => {
                this.#view.setPointerNavigation(toggle.checked);
            });
            this.#pointerToggle = toggle;
            row.appendChild(label);
            row.appendChild(toggle);
            controls.appendChild(row);
        }

        const snapshotEnabled = options.showSnapshot ?? true;
        const recorderRequested = options.showRecorder ?? false;
        const recorderSupported =
            recorderRequested &&
            SpiralCanvasRecorder.isSupported() &&
            typeof this.#view.element.captureStream === "function";

        if (snapshotEnabled || recorderRequested) {
            const row = this.#root.ownerDocument!.createElement("div");
            row.className = "control-row actions-row";

            if (snapshotEnabled) {
                const button = this.#root.ownerDocument!.createElement("button");
                button.type = "button";
                button.textContent = "Snapshot";
                button.addEventListener("click", () => {
                    void this.#handleSnapshot();
                });
                this.#snapshotButton = button;
                row.appendChild(button);
            }

            if (recorderRequested) {
                const button = this.#root.ownerDocument!.createElement("button");
                button.type = "button";
                button.textContent = "Start recording";
                if (recorderSupported) {
                    button.addEventListener("click", () => {
                        void this.#toggleRecording();
                    });
                } else {
                    button.disabled = true;
                    button.title = "MediaRecorder API is not available";
                }
                this.#recordButton = button;
                row.appendChild(button);

                const indicator = this.#root.ownerDocument!.createElement("span");
                indicator.className = "recording-indicator";
                this.#recordingIndicator = indicator;
                row.appendChild(indicator);
            }

            controls.appendChild(row);
        }

        this.#root.appendChild(controls);
    }

    #buildStats(): void {
        const statsSection = this.#root.ownerDocument!.createElement("dl");
        statsSection.className = "stats-grid";

        const fpsLabel = this.#root.ownerDocument!.createElement("dt");
        fpsLabel.textContent = "FPS";
        const fpsValue = this.#root.ownerDocument!.createElement("dd");
        fpsValue.textContent = "–";
        this.#fpsField = fpsValue;

        const zoomLabel = this.#root.ownerDocument!.createElement("dt");
        zoomLabel.textContent = "Zoom";
        const zoomValue = this.#root.ownerDocument!.createElement("dd");
        zoomValue.textContent = "–";
        this.#zoomField = zoomValue;

        const offsetLabel = this.#root.ownerDocument!.createElement("dt");
        offsetLabel.textContent = "Offset";
        const offsetValue = this.#root.ownerDocument!.createElement("dd");
        offsetValue.textContent = "–";
        this.#offsetField = offsetValue;

        const paletteLabel = this.#root.ownerDocument!.createElement("dt");
        paletteLabel.textContent = "Palette";
        const paletteValue = this.#root.ownerDocument!.createElement("dd");
        paletteValue.textContent = "–";
        this.#paletteField = paletteValue;

        statsSection.appendChild(fpsLabel);
        statsSection.appendChild(fpsValue);
        statsSection.appendChild(zoomLabel);
        statsSection.appendChild(zoomValue);
        statsSection.appendChild(offsetLabel);
        statsSection.appendChild(offsetValue);
        statsSection.appendChild(paletteLabel);
        statsSection.appendChild(paletteValue);

        for (const stat of DEFAULT_STATS) {
            const dt = this.#root.ownerDocument!.createElement("dt");
            dt.textContent = stat.label;
            const dd = this.#root.ownerDocument!.createElement("dd");
            dd.textContent = "–";
            statsSection.appendChild(dt);
            statsSection.appendChild(dd);
            this.#statFields.set(stat.key, dd);
        }

        this.#root.appendChild(statsSection);
    }

    #buildParticipantsPane(): void {
        if (this.#participantsContainer) {
            return;
        }
        const section = this.#root.ownerDocument!.createElement("section");
        section.className = "participants";
        section.style.display = "none";
        const title = this.#root.ownerDocument!.createElement("h2");
        title.textContent = "Participants";
        const list = this.#root.ownerDocument!.createElement("div");
        list.className = "participants-list";
        section.appendChild(title);
        section.appendChild(list);
        this.#participantsContainer = section;
        this.#participantsList = list;
        this.#root.appendChild(section);
    }

    #renderParticipants(participants: CanvasCollabParticipantState[]): void {
        if (!this.#participantsContainer || !this.#participantsList) {
            return;
        }
        if (!this.#collabSession) {
            this.#participantsContainer.style.display = "none";
            this.#participantsList.innerHTML = "";
            return;
        }
        this.#participantsContainer.style.display = "flex";
        this.#participantsList.innerHTML = "";
        const doc = this.#participantsList.ownerDocument!;
        for (const participant of participants) {
            const row = doc.createElement("div");
            row.className = "participant-row";

            const header = doc.createElement("div");
            header.className = "participant-header";

            const indicator = doc.createElement("span");
            indicator.className = "participant-indicator";
            indicator.style.setProperty("--indicator", this.#resolveParticipantColor(participant));
            header.appendChild(indicator);

            const label = doc.createElement("span");
            label.className = "participant-label";
            label.textContent = participant.label ?? participant.role;
            header.appendChild(label);

            if (participant.label && participant.label !== participant.role) {
                const role = doc.createElement("span");
                role.className = "participant-meta";
                role.textContent = `(${participant.role})`;
                header.appendChild(role);
            }

            row.appendChild(header);

            const stateLine = doc.createElement("div");
            stateLine.className = "participant-meta";
            const state = participant.state;
            stateLine.textContent = `zoom ${formatNumber(state.zoom, this.#statsDigits)} · offset ${formatNumber(
                state.offset.x,
                this.#statsDigits,
            )}, ${formatNumber(state.offset.y, this.#statsDigits)}`;
            row.appendChild(stateLine);

            if (participant.lastPointer) {
                const pointerLine = doc.createElement("div");
                pointerLine.className = "participant-pointer";
                pointerLine.textContent = `${participant.lastPointer.kind} via ${participant.lastPointer.source} · offset ${formatNumber(
                    participant.lastPointer.offset.x,
                    this.#statsDigits,
                )}, ${formatNumber(participant.lastPointer.offset.y, this.#statsDigits)}`;
                row.appendChild(pointerLine);
            }

            const lastSeen = doc.createElement("div");
            lastSeen.className = "participant-meta";
            const seconds = Math.max(0, Math.round((Date.now() - participant.lastSeen) / 1000));
            lastSeen.textContent = seconds === 0 ? "active now" : `last input ${seconds}s ago`;
            row.appendChild(lastSeen);

            this.#participantsList.appendChild(row);
        }
    }

    #resolveParticipantColor(participant: CanvasCollabParticipantState): string {
        if (participant.color) {
            return participant.color;
        }
        const preset = ROLE_COLORS[participant.role];
        if (preset) {
            return preset;
        }
        return "#38bdf8";
    }

    #populatePaletteOptions(): void {
        if (!this.#paletteSelect) {
            return;
        }
        this.#paletteSelect.innerHTML = "";
        const builtin = safeAvailablePalettes();
        for (const name of builtin) {
            const option = this.#root.ownerDocument!.createElement("option");
            option.value = name;
            option.textContent = name;
            this.#paletteSelect.appendChild(option);
        }
        if (this.#customPalettes.size > 0) {
            const group = this.#root.ownerDocument!.createElement("optgroup");
            group.label = "Custom";
            for (const [label, palette] of this.#customPalettes) {
                const option = this.#root.ownerDocument!.createElement("option");
                option.value = label;
                option.textContent = label;
                group.appendChild(option);
                this.#customPaletteLookup.set(palette as unknown as object, label);
            }
            this.#paletteSelect.appendChild(group);
        }
    }

    #ingestCustomPalettes(custom?: Record<string, CustomPalette>): void {
        if (!custom) {
            return;
        }
        for (const [label, palette] of Object.entries(custom)) {
            this.#customPalettes.set(label, palette);
            this.#customPaletteLookup.set(palette as unknown as object, label);
        }
    }

    #syncPaletteSelector(palette: CanvasPaletteCanonicalName | CustomPalette): void {
        if (!this.#paletteSelect) {
            return;
        }
        if (typeof palette === "string") {
            this.#paletteSelect.value = palette;
            return;
        }
        const label = this.#customPaletteLookup.get(palette as unknown as object);
        if (label) {
            this.#paletteSelect.value = label;
        } else {
            const customLabel = "Custom";
            if (!this.#customPalettes.has(customLabel)) {
                const option = this.#root.ownerDocument!.createElement("option");
                option.value = customLabel;
                option.textContent = customLabel;
                this.#paletteSelect.appendChild(option);
                this.#customPalettes.set(customLabel, palette);
                this.#customPaletteLookup.set(palette as unknown as object, customLabel);
            }
            this.#paletteSelect.value = customLabel;
        }
    }

    #syncPointerToggle(): void {
        if (!this.#pointerToggle) {
            return;
        }
        const enabled = this.#view.pointerNavigationEnabled;
        if (this.#pointerToggle.checked !== enabled) {
            this.#pointerToggle.checked = enabled;
        }
    }

    #syncCurvatureControls(): void {
        if (!this.#curvatureSlider || !this.#curvatureLabel) {
            return;
        }
        this.#curvatureSlider.value = String(this.#view.statsCurvature);
        this.#updateCurvatureLabel();
    }

    #updateCurvatureLabel(): void {
        if (!this.#curvatureLabel) {
            return;
        }
        this.#curvatureLabel.textContent = formatNumber(this.#view.statsCurvature, this.#statsDigits);
    }

    #updateFrame(event: CanvasFrameEvent): void {
        if (event.delta > 0) {
            const fps = 1000 / event.delta;
            this.#fpsField.textContent = formatNumber(fps, this.#statsDigits);
        } else {
            this.#fpsField.textContent = "–";
        }
        this.#zoomField.textContent = formatNumber(event.zoom, this.#statsDigits);
        this.#setOffsetReadout(event.offset.x, event.offset.y);
        this.#syncPointerToggle();
        if (this.#recorder?.recording) {
            this.#updateRecordingIndicator(event.time);
        }
    }

    #updateStats(event: CanvasStatsEvent): void {
        this.#updatePaletteField(event.palette);
        for (const stat of DEFAULT_STATS) {
            const target = this.#statFields.get(stat.key);
            if (!target) {
                continue;
            }
            const value = stat.read(event);
            target.textContent = formatNumber(value, this.#statsDigits);
        }
        if (!this.#paletteSelect || typeof event.palette !== "string") {
            this.#syncPaletteSelector(event.palette);
        }
        this.#syncCurvatureControls();
    }

    #updatePaletteField(palette: CanvasPaletteCanonicalName | CustomPalette): void {
        if (typeof palette === "string") {
            this.#paletteField.textContent = palette;
            return;
        }
        const label = this.#customPaletteLookup.get(palette as unknown as object);
        this.#paletteField.textContent = label ?? "custom";
    }

    #setOffsetReadout(x: number, y: number): void {
        this.#offsetField.textContent = `${formatNumber(x, this.#statsDigits)}, ${formatNumber(y, this.#statsDigits)}`;
    }

    async #handleSnapshot(): Promise<void> {
        if (!this.#snapshotButton) {
            return;
        }
        this.#snapshotButton.disabled = true;
        try {
            const blob = await this.#view.toBlob(
                this.#options.snapshotMimeType ?? "image/png",
                this.#options.snapshotQuality,
            );
            if (this.#options.onSnapshot) {
                await this.#options.onSnapshot(blob);
            } else {
                const name = this.#options.snapshotFilename ?? "spiraltorch-canvas.png";
                this.#downloadBlob(blob, name);
            }
        } catch (error) {
            this.#options.onSnapshotError?.(error);
            if (typeof console !== "undefined" && typeof console.error === "function") {
                console.error("Failed to capture snapshot", error);
            }
        } finally {
            this.#snapshotButton.disabled = false;
        }
    }

    async #toggleRecording(): Promise<void> {
        if (!this.#recordButton) {
            return;
        }
        if (!this.#recorder) {
            this.#recorder = new SpiralCanvasRecorder(
                this.#view,
                this.#options.recorderOptions ?? {},
            );
        }
        const button = this.#recordButton;
        const indicator = this.#recordingIndicator;

        if (!this.#recorder.recording) {
            try {
                this.#recorder.start();
                this.#recordingStart =
                    typeof performance !== "undefined" && typeof performance.now === "function"
                        ? performance.now()
                        : Date.now();
                button.textContent = "Stop recording";
                button.classList.add("recording-active");
                if (indicator) {
                    indicator.textContent = "Rec 0.0s";
                }
            } catch (error) {
                button.textContent = "Start recording";
                button.classList.remove("recording-active");
                if (indicator) {
                    indicator.textContent = "";
                }
                this.#options.onRecordingError?.(error);
                if (typeof console !== "undefined" && typeof console.error === "function") {
                    console.error("Failed to start recording", error);
                }
            }
            return;
        }

        button.disabled = true;
        try {
            const blob = await this.#recorder.stop();
            button.textContent = "Start recording";
            button.classList.remove("recording-active");
            button.disabled = false;
            this.#recordingStart = 0;
            if (indicator) {
                indicator.textContent = "";
            }
            if (this.#options.onRecordingComplete) {
                await this.#options.onRecordingComplete(blob);
            } else {
                const filename = this.#resolveRecordingFilename(blob.type);
                this.#downloadBlob(blob, filename);
            }
        } catch (error) {
            button.disabled = false;
            button.classList.remove("recording-active");
            if (indicator) {
                indicator.textContent = "";
            }
            this.#options.onRecordingError?.(error);
            if (typeof console !== "undefined" && typeof console.error === "function") {
                console.error("Failed to stop recording", error);
            }
        }
    }

    #updateRecordingIndicator(now?: number): void {
        if (!this.#recordingIndicator || !this.#recorder?.recording) {
            return;
        }
        const current =
            typeof now === "number"
                ? now
                : typeof performance !== "undefined" && typeof performance.now === "function"
                  ? performance.now()
                  : Date.now();
        const elapsed = Math.max(0, current - this.#recordingStart);
        this.#recordingIndicator.textContent = `Rec ${(elapsed / 1000).toFixed(1)}s`;
    }

    #downloadBlob(blob: Blob, filename: string): void {
        const doc = this.#root.ownerDocument!;
        const url = URL.createObjectURL(blob);
        const link = doc.createElement("a");
        link.href = url;
        link.download = filename;
        link.style.display = "none";
        const parent = doc.body ?? this.#root;
        parent.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    #resolveRecordingFilename(type?: string): string {
        const fallback = this.#options.recorderOptions?.mimeType ?? "video/webm";
        const mime = type && type.length > 0 ? type : fallback;
        const extension = mime.includes("/") ? mime.split("/").pop() ?? "webm" : mime;
        const clean = extension.split(";")[0] || "webm";
        return `spiraltorch-canvas.${clean}`;
    }
}
