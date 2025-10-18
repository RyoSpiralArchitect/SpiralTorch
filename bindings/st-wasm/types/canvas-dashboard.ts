import { available_palettes, type CanvasPaletteName } from "spiraltorch-wasm";
import type { CanvasPaletteCanonicalName } from "spiraltorch-wasm";
import {
    SpiralCanvasView,
    type CustomPalette,
    type CanvasStatsEvent,
    type CanvasFrameEvent,
    type CanvasPointerEvent,
} from "./canvas-view";

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

    #paletteSelect?: HTMLSelectElement;
    #curvatureSlider?: HTMLInputElement;
    #curvatureLabel?: HTMLSpanElement;
    #pointerToggle?: HTMLInputElement;
    #fpsField: HTMLElement;
    #zoomField: HTMLElement;
    #offsetField: HTMLElement;
    #paletteField: HTMLElement;
    #statFields = new Map<string, HTMLElement>();

    #frameHandler: (event: CanvasFrameEvent) => void;
    #statsHandler: (event: CanvasStatsEvent) => void;
    #pointerHandler: (event: CanvasPointerEvent) => void;

    constructor(container: HTMLElement, view: SpiralCanvasView, options: SpiralCanvasDashboardOptions = {}) {
        if (!container) {
            throw new Error("SpiralCanvasDashboard requires a container element");
        }
        if (!container.ownerDocument) {
            throw new Error("Dashboard container must be attached to a document");
        }
        this.#view = view;
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
        this.#root.remove();
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
}
