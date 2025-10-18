import type {
    CanvasDesireControl,
    CanvasGradientSummary,
    CanvasPaletteName,
    CanvasPaletteCanonicalName,
    FractalCanvas,
} from "spiraltorch-wasm";

/** Event payload emitted whenever the view paints a new frame. */
export interface CanvasFrameEvent {
    time: number;
    /**
     * Delta time in milliseconds between the current frame and the previously
     * rendered frame.
     */
    delta: number;
    zoom: number;
    offset: Readonly<{ x: number; y: number }>;
}

/** Event payload emitted whenever statistics are sampled from the canvas. */
export interface CanvasStatsEvent {
    curvature: number;
    control: CanvasDesireControl;
    summary: CanvasGradientSummary;
    palette: CanvasPaletteCanonicalName | CustomPalette;
}

/** Event payload emitted when the user manipulates the view transform. */
export interface CanvasPointerEvent {
    kind: "pan" | "zoom";
    zoom: number;
    offset: Readonly<{ x: number; y: number }>;
    source: "drag" | "wheel";
}

/** Palette stop definition used to construct a custom color ramp. */
export interface CustomPaletteStop {
    /** Stop position in the `[0, 1]` range. Values outside the range are clamped. */
    offset: number;
    /**
     * Color definition encoded as an RGB tuple in the `[0, 255]` range or a
     * CSS-compatible hex string.
     */
    color: readonly [number, number, number] | string;
}

/** Custom palette description used when overriding the shader palette. */
export interface CustomPalette {
    stops: CustomPaletteStop[];
    /** Optional gamma correction applied to the interpolated ramp. */
    gamma?: number;
}

/** Configuration supplied to {@link SpiralCanvasView}. */
export interface SpiralCanvasViewOptions {
    /** Palette name or custom palette description. Defaults to the current palette. */
    palette?: CanvasPaletteName | CustomPalette;
    /**
     * Curvature value used when sampling control and gradient statistics.
     * Defaults to `1.0`.
     */
    statsCurvature?: number;
    /** Interval in milliseconds between stats samples. Defaults to `250ms`. */
    statsInterval?: number;
    /**
     * Enables pointer based navigation (pan + zoom). Defaults to `true` when a
     * `window` global is present.
     */
    pointerNavigation?: boolean;
    /** Lower bound applied when zooming. Defaults to `0.25`. */
    minZoom?: number;
    /** Upper bound applied when zooming. Defaults to `8`. */
    maxZoom?: number;
    /** Device pixel ratio override. Uses `window.devicePixelRatio` by default. */
    devicePixelRatio?: number;
    /** Automatically start the render loop. Defaults to `true`. */
    autoStart?: boolean;
}

export interface CanvasViewEventMap {
    frame: CanvasFrameEvent;
    stats: CanvasStatsEvent;
    pointer: CanvasPointerEvent;
}

type EventHandler<T> = (event: T) => void;

type PaletteState =
    | { type: "builtin"; name: CanvasPaletteCanonicalName }
    | { type: "custom"; palette: CustomPalette; lut: Uint8ClampedArray };

interface NormalizedPaletteStop {
    offset: number;
    color: [number, number, number];
}

class EventDispatcher<Events extends Record<string, unknown>> {
    private readonly listeners = new Map<keyof Events, Set<EventHandler<any>>>();

    on<K extends keyof Events>(type: K, handler: EventHandler<Events[K]>): void {
        if (!this.listeners.has(type)) {
            this.listeners.set(type, new Set());
        }
        this.listeners.get(type)!.add(handler as EventHandler<any>);
    }

    off<K extends keyof Events>(type: K, handler: EventHandler<Events[K]>): void {
        this.listeners.get(type)?.delete(handler as EventHandler<any>);
    }

    emit<K extends keyof Events>(type: K, payload: Events[K]): void {
        const handlers = this.listeners.get(type);
        if (!handlers) {
            return;
        }
        handlers.forEach((handler) => handler(payload));
    }
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

function parseColor(value: string): [number, number, number] {
    const hex = value.trim().toLowerCase();
    const shorthand = /^#([0-9a-f]{3})$/i;
    const full = /^#([0-9a-f]{6})$/i;
    let normalized: string | null = null;

    if (shorthand.test(hex)) {
        normalized = hex.replace(shorthand, (_, digits: string) => {
            return `#${digits[0]}${digits[0]}${digits[1]}${digits[1]}${digits[2]}${digits[2]}`;
        });
    } else if (full.test(hex)) {
        normalized = hex;
    }

    if (!normalized) {
        throw new Error(`Unsupported color format: ${value}`);
    }

    const r = parseInt(normalized.slice(1, 3), 16);
    const g = parseInt(normalized.slice(3, 5), 16);
    const b = parseInt(normalized.slice(5, 7), 16);
    return [r, g, b];
}

function normalizeStops(stops: CustomPaletteStop[]): NormalizedPaletteStop[] {
    const sorted = stops
        .map<NormalizedPaletteStop>((stop) => ({
            offset: clamp(stop.offset, 0, 1),
            color: Array.isArray(stop.color)
                ? [stop.color[0], stop.color[1], stop.color[2]]
                : [...parseColor(stop.color)] as [number, number, number],
        }))
        .sort((a, b) => a.offset - b.offset);

    if (sorted.length === 0) {
        throw new Error("Custom palettes require at least one stop");
    }

    if (sorted.length === 1) {
        const [color] = sorted;
        return [
            { offset: 0, color: color.color },
            { offset: 1, color: color.color },
        ];
    }

    return sorted;
}

function buildPaletteLut(palette: CustomPalette, resolution = 256): Uint8ClampedArray {
    const stops = normalizeStops(palette.stops);
    const gamma = palette.gamma ?? 1;
    const lut = new Uint8ClampedArray(resolution * 3);

    let stopIndex = 0;
    for (let i = 0; i < resolution; i++) {
        const t = i / (resolution - 1);
        while (stopIndex + 1 < stops.length && t > stops[stopIndex + 1].offset) {
            stopIndex += 1;
        }

        const left = stops[stopIndex];
        const right = stops[Math.min(stopIndex + 1, stops.length - 1)];
        const span = Math.max(right.offset - left.offset, 1e-6);
        const localT = clamp((t - left.offset) / span, 0, 1);
        const mix = Math.pow(localT, gamma);

        for (let channel = 0; channel < 3; channel++) {
            const start = left.color[channel];
            const end = right.color[channel];
            lut[i * 3 + channel] = Math.round(start + (end - start) * mix);
        }
    }

    return lut;
}

function applyPalette(
    source: Uint8Array,
    target: Uint8ClampedArray,
    lut: Uint8ClampedArray,
): void {
    for (let i = 0; i < source.length; i += 4) {
        const shade = source[i];
        const lutIndex = shade * 3;
        target[i + 0] = lut[lutIndex + 0];
        target[i + 1] = lut[lutIndex + 1];
        target[i + 2] = lut[lutIndex + 2];
        target[i + 3] = source[i + 3];
    }
}

/**
 * High level orchestrator that manages rendering, palette application and
 * interaction for {@link FractalCanvas} instances.
 */
export class SpiralCanvasView {
    readonly #canvas: HTMLCanvasElement;
    readonly #fractal: FractalCanvas;
    readonly #ctx: CanvasRenderingContext2D;
    readonly #events = new EventDispatcher<CanvasViewEventMap>();

    readonly #offscreen: HTMLCanvasElement;
    readonly #offscreenCtx: CanvasRenderingContext2D;
    #imageData?: ImageData;

    #animationHandle: number | null = null;
    #running = false;
    #lastFrameTime = 0;

    #devicePixelRatio: number;
    #pointerNavigation: boolean;
    #minZoom: number;
    #maxZoom: number;

    #zoom = 1;
    #offset = { x: 0, y: 0 };

    #statsCurvature: number;
    #statsInterval: number;
    #lastStatsTime = 0;

    #palette: PaletteState;
    #destroyed = false;

    #pointerActive = false;

    constructor(canvas: HTMLCanvasElement, fractal: FractalCanvas, options: SpiralCanvasViewOptions = {}) {
        if (!canvas) {
            throw new Error("SpiralCanvasView requires a target <canvas> element");
        }
        this.#canvas = canvas;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            throw new Error("Failed to acquire 2D context for target canvas");
        }
        this.#ctx = ctx;
        this.#fractal = fractal;

        const doc =
            canvas.ownerDocument ?? (typeof document !== "undefined" ? document : undefined);
        if (!doc) {
            throw new Error("Cannot create an offscreen canvas without a DOM Document");
        }
        const offscreen = doc.createElement("canvas");
        offscreen.width = fractal.width;
        offscreen.height = fractal.height;
        const ctx = offscreen.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
            throw new Error("Failed to acquire 2D context for offscreen canvas");
        }
        this.#offscreen = offscreen;
        this.#offscreenCtx = ctx;

        this.#devicePixelRatio = options.devicePixelRatio ?? (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);
        this.#pointerNavigation = options.pointerNavigation ?? typeof window !== "undefined";
        this.#minZoom = options.minZoom ?? 0.25;
        this.#maxZoom = options.maxZoom ?? 8;
        this.#statsCurvature = options.statsCurvature ?? 1.0;
        this.#statsInterval = options.statsInterval ?? 250;

        this.#palette = this.#resolveInitialPalette(options.palette);

        this.#configureCanvasSizing();
        this.#attachPointerNavigation();

        if (options.autoStart ?? true) {
            this.start();
        }
    }

    /** Returns the underlying {@link FractalCanvas}. */
    get fractal(): FractalCanvas {
        return this.#fractal;
    }

    /** Returns the current palette configuration. */
    get palette(): CanvasPaletteCanonicalName | CustomPalette {
        if (this.#palette.type === "builtin") {
            return this.#palette.name;
        }
        return this.#palette.palette;
    }

    /** Indicates whether the render loop is currently active. */
    get running(): boolean {
        return this.#running;
    }

    /** Current zoom factor. */
    get zoom(): number {
        return this.#zoom;
    }

    /** Current offset expressed in CSS pixels. */
    get offset(): Readonly<{ x: number; y: number }> {
        return this.#offset;
    }

    on<K extends keyof CanvasViewEventMap>(type: K, handler: EventHandler<CanvasViewEventMap[K]>): void {
        this.#events.on(type, handler);
    }

    off<K extends keyof CanvasViewEventMap>(type: K, handler: EventHandler<CanvasViewEventMap[K]>): void {
        this.#events.off(type, handler);
    }

    /** Starts the requestAnimationFrame loop. */
    start(): void {
        if (this.#destroyed || this.#running) {
            return;
        }
        this.#running = true;
        this.#lastFrameTime = 0;
        this.#lastStatsTime = 0;
        const step = (timestamp: number) => {
            if (!this.#running) {
                return;
            }
            this.#animationHandle = (typeof window !== "undefined" ? window.requestAnimationFrame(step) : -1);
            this.#renderFrame(timestamp);
        };
        this.#animationHandle = (typeof window !== "undefined" ? window.requestAnimationFrame(step) : -1);
    }

    /** Stops the requestAnimationFrame loop. */
    stop(): void {
        if (!this.#running) {
            return;
        }
        this.#running = false;
        if (this.#animationHandle !== null && typeof window !== "undefined") {
            window.cancelAnimationFrame(this.#animationHandle);
        }
        this.#animationHandle = null;
        this.#lastFrameTime = 0;
        this.#lastStatsTime = 0;
    }

    /** Adjusts the zoom factor while clamping to the configured bounds. */
    setZoom(value: number): void {
        const clamped = clamp(value, this.#minZoom, this.#maxZoom);
        if (clamped === this.#zoom) {
            return;
        }
        this.#zoom = clamped;
        this.invalidate();
    }

    /** Sets the view offset (in CSS pixels). */
    setOffset(x: number, y: number): void {
        this.#offset = { x, y };
        this.invalidate();
    }

    /**
     * Sets a palette name. Built-in palettes are forwarded to the WebAssembly
     * side whereas custom palettes are applied client-side using the RGBA
     * buffer.
     */
    setPalette(palette: CanvasPaletteName | CustomPalette): void {
        this.#palette = this.#resolvePalette(palette);
        this.invalidate();
    }

    /**
     * Requests the view to repaint without waiting for the next animation
     * frame.
     */
    invalidate(): void {
        if (this.#running) {
            return;
        }
        const now =
            typeof performance !== "undefined" && typeof performance.now === "function"
                ? performance.now()
                : Date.now();
        this.#renderFrame(now);
    }

    /** Releases all event listeners and stops the animation loop. */
    destroy(): void {
        if (this.#destroyed) {
            return;
        }
        this.stop();
        this.#detachPointerNavigation();
        this.#destroyed = true;
    }

    #renderFrame(timestamp: number): void {
        const delta = this.#lastFrameTime === 0 ? 0 : timestamp - this.#lastFrameTime;
        this.#lastFrameTime = timestamp;

        this.#fractal.render_to_canvas(this.#offscreen);
        const pixels = this.#fractal.pixels();
        if (this.#palette.type === "builtin") {
            const clamped = new Uint8ClampedArray(
                pixels.buffer,
                pixels.byteOffset,
                pixels.byteLength,
            );
            this.#drawPixels(clamped);
        } else {
            const clamped = this.#getImageData();
            const source = new Uint8Array(
                pixels.buffer,
                pixels.byteOffset,
                pixels.byteLength,
            );
            applyPalette(source, clamped.data, this.#palette.lut);
            this.#drawPixels(clamped.data);
        }

        this.#emitFrame(delta, timestamp);
        this.#maybeEmitStats(timestamp);
    }

    #drawPixels(data: Uint8ClampedArray): void {
        const imageData = this.#ensureImageData();
        imageData.data.set(data);
        this.#offscreenCtx.putImageData(imageData, 0, 0);

        const ctx = this.#ctx;
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);
        const scale = this.#zoom * this.#devicePixelRatio;
        const offsetX = this.#offset.x * this.#devicePixelRatio;
        const offsetY = this.#offset.y * this.#devicePixelRatio;
        ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(this.#offscreen, 0, 0);
        ctx.restore();
    }

    #ensureImageData(): ImageData {
        if (!this.#imageData || this.#imageData.width !== this.#fractal.width || this.#imageData.height !== this.#fractal.height) {
            this.#imageData = new ImageData(this.#fractal.width, this.#fractal.height);
        }
        return this.#imageData;
    }

    #getImageData(): ImageData {
        const imageData = this.#ensureImageData();
        return imageData;
    }

    #emitFrame(delta: number, timestamp: number): void {
        this.#events.emit("frame", {
            time: timestamp,
            delta,
            zoom: this.#zoom,
            offset: { ...this.#offset },
        });
    }

    #maybeEmitStats(timestamp: number): void {
        if (this.#statsInterval <= 0) {
            return;
        }
        if (timestamp - this.#lastStatsTime < this.#statsInterval) {
            return;
        }
        this.#lastStatsTime = timestamp;
        const control = this.#fractal.desireControl(this.#statsCurvature);
        const summary = this.#fractal.gradientSummary(this.#statsCurvature);
        const palette = this.palette;
        this.#events.emit("stats", {
            curvature: this.#statsCurvature,
            control,
            summary,
            palette,
        });
    }

    #configureCanvasSizing(): void {
        const dpr = this.#devicePixelRatio;
        const width = Math.max(1, Math.round(this.#fractal.width * dpr));
        const height = Math.max(1, Math.round(this.#fractal.height * dpr));
        this.#canvas.width = width;
        this.#canvas.height = height;
        this.#ctx.imageSmoothingEnabled = false;
        if (!this.#canvas.style.width) {
            this.#canvas.style.width = `${this.#fractal.width}px`;
        }
        if (!this.#canvas.style.height) {
            this.#canvas.style.height = `${this.#fractal.height}px`;
        }
    }

    #resolveInitialPalette(value?: CanvasPaletteName | CustomPalette): PaletteState {
        if (!value) {
            const palette = this.#fractal.palette();
            return { type: "builtin", name: palette };
        }
        return this.#resolvePalette(value);
    }

    #resolvePalette(value: CanvasPaletteName | CustomPalette): PaletteState {
        if (typeof value === "string") {
            this.#fractal.set_palette(value);
            const canonical = this.#fractal.palette();
            return { type: "builtin", name: canonical };
        }
        this.#fractal.set_palette("grayscale");
        const lut = buildPaletteLut(value);
        return { type: "custom", palette: value, lut };
    }

    #attachPointerNavigation(): void {
        if (!this.#pointerNavigation || typeof window === "undefined") {
            return;
        }
        const onPointerDown = (event: PointerEvent) => {
            if (event.button !== 0) {
                return;
            }
            this.#pointerActive = true;
            (event.target as HTMLElement | null)?.setPointerCapture?.(event.pointerId);
        };
        const onPointerMove = (event: PointerEvent) => {
            if (!this.#pointerActive) {
                return;
            }
            this.#offset = {
                x: this.#offset.x + event.movementX,
                y: this.#offset.y + event.movementY,
            };
            if (!this.#running) {
                this.invalidate();
            }
            this.#events.emit("pointer", {
                kind: "pan",
                source: "drag",
                zoom: this.#zoom,
                offset: { ...this.#offset },
            });
        };
        const onPointerUp = (event: PointerEvent) => {
            this.#pointerActive = false;
            (event.target as HTMLElement | null)?.releasePointerCapture?.(event.pointerId);
        };
        const onWheel = (event: WheelEvent) => {
            event.preventDefault();
            const delta = event.deltaY;
            const zoomFactor = Math.pow(2, -delta * 0.001);
            const newZoom = clamp(this.#zoom * zoomFactor, this.#minZoom, this.#maxZoom);
            const rect = this.#canvas.getBoundingClientRect();
            const originX = event.clientX - rect.left;
            const originY = event.clientY - rect.top;
            const ratio = newZoom / this.#zoom;
            this.#offset = {
                x: originX - (originX - this.#offset.x) * ratio,
                y: originY - (originY - this.#offset.y) * ratio,
            };
            this.#zoom = newZoom;
            if (!this.#running) {
                this.invalidate();
            }
            this.#events.emit("pointer", {
                kind: "zoom",
                source: "wheel",
                zoom: this.#zoom,
                offset: { ...this.#offset },
            });
        };

        this.#canvas.addEventListener("pointerdown", onPointerDown);
        window.addEventListener("pointermove", onPointerMove);
        window.addEventListener("pointerup", onPointerUp);
        window.addEventListener("pointercancel", onPointerUp);
        this.#canvas.addEventListener("wheel", onWheel, { passive: false });

        this.#pointerCleanup = () => {
            this.#canvas.removeEventListener("pointerdown", onPointerDown);
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("pointercancel", onPointerUp);
            this.#canvas.removeEventListener("wheel", onWheel);
        };
    }

    #pointerCleanup: (() => void) | null = null;

    #detachPointerNavigation(): void {
        this.#pointerCleanup?.();
        this.#pointerCleanup = null;
    }
}

