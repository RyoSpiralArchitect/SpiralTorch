import type { CanvasPaletteCanonicalName } from "spiraltorch-wasm";
import {
    SpiralCanvasView,
    type CustomPalette,
    type CanvasPointerEvent,
} from "./canvas-view";

const COLLAB_SCHEMA_VERSION = 1;
const COLLAB_BATCH_INTERVAL_MS = 16;
const COLLAB_DEFAULT_POINTER_RATE_HZ = 30;
const COLLAB_DEFAULT_PATCH_RATE_HZ = 20;
const COLLAB_DEFAULT_PRESENCE_INTERVAL_MS = 1_000;
const COLLAB_MAX_MESSAGE_BYTES_DEFAULT = 256 * 1024;
const COLLAB_STORAGE_POLL_INTERVAL_MS = 750;

export type CanvasCollabRole = "trainer" | "model" | "human" | (string & {});

export interface CanvasCollabParticipant {
    id?: string;
    role: CanvasCollabRole;
    label?: string;
    color?: string;
}

interface SerializedPaletteBuiltin {
    kind: "builtin";
    name: CanvasPaletteCanonicalName;
}

interface SerializedPaletteCustom {
    kind: "custom";
    palette: CustomPalette;
}

export type SerializedPalette = SerializedPaletteBuiltin | SerializedPaletteCustom;

export interface CanvasCollabState {
    zoom: number;
    offset: { x: number; y: number };
    palette: SerializedPalette;
    pointerNavigation: boolean;
    statsCurvature: number;
    statsInterval: number;
    devicePixelRatio: number;
    running: boolean;
}

export interface CanvasCollabParticipantState extends CanvasCollabParticipant {
    id: string;
    state: CanvasCollabState;
    lastSeen: number;
    lastPointer?: CanvasPointerEvent;
}

export interface CanvasCollabEventMap {
    state: {
        participant: CanvasCollabParticipantState;
        origin: "local" | "remote";
    };
    participants: {
        participants: CanvasCollabParticipantState[];
    };
    pointer: {
        participant: CanvasCollabParticipantState;
        event: CanvasPointerEvent;
    };
}

export type CollabTelemetryEvent =
    | { type: "join"; participant: CanvasCollabParticipantState }
    | { type: "leave"; participantId: string }
    | { type: "transport-change"; transport: CollabTransportKind | "none" }
    | { type: "latency-suppressed"; participantId: string; clock: number }
    | { type: "budget-suppressed"; participantId: string; remainingBudget: number }
    | { type: "conflict-resolved"; participantId: string; field: keyof CanvasCollabState }
    | { type: "degraded"; reason: string }
    | { type: "presence"; participants: number }
    | { type: "pointer"; participantId: string }
    | { type: "unknown-schema"; version: number }
    | { type: "message-dropped"; reason: string };

export type CollabTelemetrySink = (event: CollabTelemetryEvent) => void;

export interface CollabPatchAttributionEvent {
    participant: CanvasCollabParticipantState;
    patch: Partial<CanvasCollabState>;
    origin: "local" | "remote";
    clock: number;
}

type CollabTransportKind = "broadcast" | "storage";

interface CollabTransport {
    readonly kind: CollabTransportKind;
    post(envelope: CollabEnvelope, serialized: string): void;
    subscribe(handler: (envelope: CollabEnvelope) => void): void;
    destroy(): void;
}

export interface SpiralCanvasCollabOptions {
    sessionId?: string;
    participant: CanvasCollabParticipant;
    sharePointer?: boolean;
    pointerRateHz?: number;
    patchRateHz?: number;
    presenceIntervalMs?: number;
    maxMessageBytes?: number;
    telemetry?: CollabTelemetrySink;
    attributionSink?: (event: CollabPatchAttributionEvent) => void;
}

interface CollabMessageBase {
    type: "hello" | "state" | "pointer" | "leave" | "presence";
    id: string;
    participant: {
        role: CanvasCollabRole;
        label?: string;
        color?: string;
    };
    clock: number;
}

interface CollabHelloMessage extends CollabMessageBase {
    type: "hello";
    state: CanvasCollabState;
}

interface CollabStateMessage extends CollabMessageBase {
    type: "state";
    state: Partial<CanvasCollabState>;
    full?: boolean;
}

interface CollabPointerMessage extends CollabMessageBase {
    type: "pointer";
    event: CanvasPointerEvent;
}

interface CollabLeaveMessage extends CollabMessageBase {
    type: "leave";
}

interface CollabPresenceMessage extends CollabMessageBase {
    type: "presence";
    roster: Array<{ id: string; role: CanvasCollabRole; lastSeen: number }>;
}

type CollabMessage =
    | CollabHelloMessage
    | CollabStateMessage
    | CollabPointerMessage
    | CollabLeaveMessage
    | CollabPresenceMessage;

interface CollabEnvelope {
    v: number;
    message: CollabMessage;
}

type EventHandler<T> = (event: T) => void;

class EventDispatcher<Events extends Record<string, unknown>> {
    readonly #listeners = new Map<keyof Events, Set<EventHandler<any>>>();

    on<K extends keyof Events>(type: K, handler: EventHandler<Events[K]>): void {
        if (!this.#listeners.has(type)) {
            this.#listeners.set(type, new Set());
        }
        this.#listeners.get(type)!.add(handler as EventHandler<any>);
    }

    off<K extends keyof Events>(type: K, handler: EventHandler<Events[K]>): void {
        this.#listeners.get(type)?.delete(handler as EventHandler<any>);
    }

    emit<K extends keyof Events>(type: K, event: Events[K]): void {
        const listeners = this.#listeners.get(type);
        if (!listeners) {
            return;
        }
        listeners.forEach((handler) => handler(event));
    }
}

interface ParticipantRecord {
    info: CanvasCollabParticipantState;
    clock: number;
}

function generateId(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `collab-${Math.random().toString(36).slice(2, 10)}`;
}

function clonePalette(palette: CustomPalette): CustomPalette {
    return {
        stops: palette.stops.map((stop) => ({
            offset: stop.offset,
            color: Array.isArray(stop.color)
                ? ([stop.color[0], stop.color[1], stop.color[2]] as [number, number, number])
                : stop.color,
        })),
        gamma: palette.gamma,
    };
}

function cloneState(state: CanvasCollabState): CanvasCollabState {
    return {
        zoom: state.zoom,
        offset: { x: state.offset.x, y: state.offset.y },
        palette:
            state.palette.kind === "builtin"
                ? { kind: "builtin", name: state.palette.name }
                : { kind: "custom", palette: clonePalette(state.palette.palette) },
        pointerNavigation: state.pointerNavigation,
        statsCurvature: state.statsCurvature,
        statsInterval: state.statsInterval,
        devicePixelRatio: state.devicePixelRatio,
        running: state.running,
    };
}

function mergeState(target: CanvasCollabState, patch: Partial<CanvasCollabState>): CanvasCollabState {
    const merged: CanvasCollabState = {
        zoom: patch.zoom ?? target.zoom,
        offset: patch.offset ? { x: patch.offset.x, y: patch.offset.y } : { ...target.offset },
        palette: patch.palette
            ? patch.palette.kind === "builtin"
                ? { kind: "builtin", name: patch.palette.name }
                : { kind: "custom", palette: clonePalette(patch.palette.palette) }
            : target.palette.kind === "builtin"
            ? { kind: "builtin", name: target.palette.name }
            : { kind: "custom", palette: clonePalette(target.palette.palette) },
        pointerNavigation: patch.pointerNavigation ?? target.pointerNavigation,
        statsCurvature: patch.statsCurvature ?? target.statsCurvature,
        statsInterval: patch.statsInterval ?? target.statsInterval,
        devicePixelRatio: patch.devicePixelRatio ?? target.devicePixelRatio,
        running: patch.running ?? target.running,
    };
    return merged;
}

function mergePatches(
    target: Partial<CanvasCollabState> | null,
    patch: Partial<CanvasCollabState>,
): Partial<CanvasCollabState> {
    const next: Partial<CanvasCollabState> = target ? { ...target } : {};
    if (patch.zoom !== undefined) {
        next.zoom = patch.zoom;
    }
    if (patch.offset) {
        next.offset = { x: patch.offset.x, y: patch.offset.y };
    }
    if (patch.palette) {
        next.palette =
            patch.palette.kind === "builtin"
                ? { kind: "builtin", name: patch.palette.name }
                : { kind: "custom", palette: clonePalette(patch.palette.palette) };
    }
    if (patch.pointerNavigation !== undefined) {
        next.pointerNavigation = patch.pointerNavigation;
    }
    if (patch.statsCurvature !== undefined) {
        next.statsCurvature = patch.statsCurvature;
    }
    if (patch.statsInterval !== undefined) {
        next.statsInterval = patch.statsInterval;
    }
    if (patch.devicePixelRatio !== undefined) {
        next.devicePixelRatio = patch.devicePixelRatio;
    }
    if (patch.running !== undefined) {
        next.running = patch.running;
    }
    return next;
}

function paletteEquals(a: SerializedPalette, b: SerializedPalette): boolean {
    if (a.kind !== b.kind) {
        return false;
    }
    if (a.kind === "builtin") {
        return a.name === (b as SerializedPaletteBuiltin).name;
    }
    const pa = a.palette;
    const pb = (b as SerializedPaletteCustom).palette;
    if ((pa.gamma ?? null) !== (pb.gamma ?? null)) {
        return false;
    }
    if (pa.stops.length !== pb.stops.length) {
        return false;
    }
    for (let i = 0; i < pa.stops.length; i += 1) {
        const sa = pa.stops[i];
        const sb = pb.stops[i];
        if (sa.offset !== sb.offset) {
            return false;
        }
        if (Array.isArray(sa.color) && Array.isArray(sb.color)) {
            if (sa.color[0] !== sb.color[0] || sa.color[1] !== sb.color[1] || sa.color[2] !== sb.color[2]) {
                return false;
            }
        } else if (sa.color !== sb.color) {
            return false;
        }
    }
    return true;
}

function statesEqual(a: CanvasCollabState | null, b: CanvasCollabState): boolean {
    if (!a) {
        return false;
    }
    return (
        a.zoom === b.zoom &&
        a.offset.x === b.offset.x &&
        a.offset.y === b.offset.y &&
        paletteEquals(a.palette, b.palette) &&
        a.pointerNavigation === b.pointerNavigation &&
        a.statsCurvature === b.statsCurvature &&
        a.statsInterval === b.statsInterval &&
        a.devicePixelRatio === b.devicePixelRatio &&
        a.running === b.running
    );
}

function deserializePalette(payload: SerializedPalette): CanvasPaletteCanonicalName | CustomPalette {
    if (payload.kind === "builtin") {
        return payload.name;
    }
    return clonePalette(payload.palette);
}

function serializeEnvelope(envelope: CollabEnvelope): { serialized: string; bytes: number } {
    const serialized = JSON.stringify(envelope);
    if (typeof TextEncoder !== "undefined") {
        const encoder = new TextEncoder();
        return { serialized, bytes: encoder.encode(serialized).byteLength };
    }
    return { serialized, bytes: serialized.length };
}

function nowMillis(): number {
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
        return performance.now();
    }
    return Date.now();
}

function rolePriority(role: CanvasCollabRole): number {
    switch (role) {
        case "trainer":
            return 3;
        case "model":
            return 2;
        case "human":
            return 1;
        default:
            return 0;
    }
}

function hasLocalStorage(): boolean {
    try {
        return typeof localStorage !== "undefined";
    } catch {
        return false;
    }
}

function randomNonce(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return Math.random().toString(36).slice(2, 10);
}

class BroadcastChannelTransport implements CollabTransport {
    readonly kind: CollabTransportKind = "broadcast";

    #channel: BroadcastChannel;

    constructor(name: string) {
        this.#channel = new BroadcastChannel(name);
    }

    post(envelope: CollabEnvelope): void;
    post(envelope: CollabEnvelope, serialized: string): void;
    post(envelope: CollabEnvelope, _serialized: string): void {
        this.#channel.postMessage(envelope);
    }

    subscribe(handler: (envelope: CollabEnvelope) => void): void {
        this.#channel.onmessage = (event) => {
            const data = event.data as CollabEnvelope;
            if (data && typeof data === "object") {
                handler(data);
            }
        };
    }

    destroy(): void {
        this.#channel.onmessage = null;
        this.#channel.close();
    }
}

class StorageEventTransport implements CollabTransport {
    readonly kind: CollabTransportKind = "storage";

    #key: string;
    #handler: ((envelope: CollabEnvelope) => void) | null = null;
    #seen = new Set<string>();
    #listener = (event: StorageEvent) => {
        if (event.key !== this.#key || !event.newValue) {
            return;
        }
        this.#dispatch(event.newValue, false);
    };
    #pollHandle: ReturnType<typeof setInterval> | null = null;

    constructor(key: string) {
        this.#key = key;
        if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
            window.addEventListener("storage", this.#listener);
        }
        if (typeof setInterval === "function") {
            this.#pollHandle = setInterval(() => {
                if (!hasLocalStorage()) {
                    return;
                }
                const value = localStorage.getItem(this.#key);
                if (value) {
                    this.#dispatch(value, false);
                }
            }, COLLAB_STORAGE_POLL_INTERVAL_MS);
        }
    }

    post(_envelope: CollabEnvelope, serialized: string): void {
        if (!hasLocalStorage()) {
            return;
        }
        const nonce = randomNonce();
        const payload = JSON.stringify({ nonce, envelope: serialized });
        this.#seen.add(nonce);
        try {
            localStorage.setItem(this.#key, payload);
        } catch {
            this.#seen.delete(nonce);
            return;
        }
        this.#dispatch(payload, true);
        if (this.#seen.size > 256) {
            const entries = Array.from(this.#seen);
            this.#seen = new Set(entries.slice(entries.length - 128));
        }
    }

    subscribe(handler: (envelope: CollabEnvelope) => void): void {
        this.#handler = handler;
    }

    destroy(): void {
        if (typeof window !== "undefined" && typeof window.removeEventListener === "function") {
            window.removeEventListener("storage", this.#listener);
        }
        if (this.#pollHandle) {
            clearInterval(this.#pollHandle);
            this.#pollHandle = null;
        }
        this.#handler = null;
    }

    #dispatch(raw: string, local: boolean): void {
        let parsed: { nonce?: string; envelope?: string };
        try {
            parsed = JSON.parse(raw);
        } catch {
            return;
        }
        if (!parsed || !parsed.envelope) {
            return;
        }
        const nonce = parsed.nonce;
        if (nonce && this.#seen.has(nonce)) {
            this.#seen.delete(nonce);
            if (!local) {
                return;
            }
        }
        try {
            const envelope = JSON.parse(parsed.envelope) as CollabEnvelope;
            this.#handler?.(envelope);
        } catch {
            // ignore malformed payloads
        }
    }
}

/**
 * Creates a peer-to-peer collaboration session that synchronizes a single
 * {@link SpiralCanvasView} across multiple browser contexts using
 * {@link BroadcastChannel}. Every participant is treated symmetrically and can
 * modify the canvas state (palette, zoom, navigation toggles, etc.).
 */
export class SpiralCanvasCollabSession {
    readonly #view: SpiralCanvasView;
    readonly #events = new EventDispatcher<CanvasCollabEventMap>();

    readonly #participantId: string;
    readonly #sessionId: string;
    readonly #sharePointer: boolean;

    #transport: CollabTransport | null = null;
    #transportKind: CollabTransportKind | "none" = "none";
    #clock = 0;
    #applyingRemote = false;
    #destroyed = false;

    readonly #participants = new Map<string, ParticipantRecord>();
    #lastState: CanvasCollabState | null = null;

    #pointerHandler: ((event: CanvasPointerEvent) => void) | null = null;
    readonly #restoreStack: Array<() => void> = [];
    #pointerMinInterval: number;
    #lastPointerSentAt = 0;

    #pendingPatch: Partial<CanvasCollabState> | null = null;
    #pendingFullSync = false;
    #batchTimer: ReturnType<typeof setTimeout> | null = null;

    #patchRateHz: number;
    #patchAllowance: number;
    #lastPatchRefill: number;

    #presenceIntervalMs: number;
    #presenceTimer: ReturnType<typeof setInterval> | null = null;

    #maxMessageBytes: number;
    #telemetry?: CollabTelemetrySink;
    #attributionSink?: (event: CollabPatchAttributionEvent) => void;

    constructor(view: SpiralCanvasView, options: SpiralCanvasCollabOptions) {
        this.#view = view;
        this.#sessionId = options.sessionId ?? "spiraltorch";
        this.#sharePointer = options.sharePointer ?? true;
        this.#telemetry = options.telemetry;
        this.#attributionSink = options.attributionSink;
        const pointerRateHz = Math.max(options.pointerRateHz ?? COLLAB_DEFAULT_POINTER_RATE_HZ, 1);
        this.#pointerMinInterval = 1_000 / pointerRateHz;
        this.#patchRateHz = Math.max(options.patchRateHz ?? COLLAB_DEFAULT_PATCH_RATE_HZ, 0);
        this.#patchAllowance = this.#patchRateHz;
        this.#lastPatchRefill = nowMillis();
        this.#presenceIntervalMs = Math.max(options.presenceIntervalMs ?? COLLAB_DEFAULT_PRESENCE_INTERVAL_MS, 250);
        this.#maxMessageBytes = options.maxMessageBytes ?? COLLAB_MAX_MESSAGE_BYTES_DEFAULT;

        const participantId = options.participant.id ?? generateId();
        this.#participantId = participantId;

        const initialState = this.#snapshotState();
        this.#lastState = initialState;
        const participantState: CanvasCollabParticipantState = {
            id: participantId,
            role: options.participant.role,
            label: options.participant.label,
            color: options.participant.color,
            state: initialState,
            lastSeen: Date.now(),
        };
        this.#participants.set(participantId, { info: participantState, clock: 0 });
        this.#emitTelemetry({ type: "join", participant: this.#cloneParticipant(participantState) });

        this.#wrapViewMethod("setZoom");
        this.#wrapViewMethod("setOffset");
        this.#wrapViewMethod("setPalette");
        this.#wrapViewMethod("setPointerNavigation");
        this.#wrapViewMethod("setStatsCurvature");
        this.#wrapViewMethod("setStatsInterval");
        this.#wrapViewMethod("setDevicePixelRatio");
        this.#wrapViewMethod("start");
        this.#wrapViewMethod("stop");

        this.#pointerHandler = (event) => {
            if (this.#applyingRemote) {
                return;
            }
            const now = nowMillis();
            const record = this.#participants.get(this.#participantId);
            if (record) {
                record.info.lastPointer = event;
                record.info.lastSeen = Date.now();
                record.info.state = this.#snapshotState();
                this.#emitState("local", record.info);
            }
            this.#enqueuePatch({ zoom: event.zoom, offset: event.offset });
            const allowPointer = now - this.#lastPointerSentAt >= this.#pointerMinInterval;
            if (this.#sharePointer && allowPointer) {
                this.#lastPointerSentAt = now;
                this.#postMessage({
                    type: "pointer",
                    id: this.#participantId,
                    participant: this.#participantEnvelope(),
                    clock: this.#tick(),
                    event,
                });
                this.#emitTelemetry({ type: "pointer", participantId: this.#participantId });
                if (record) {
                    this.#events.emit("pointer", { participant: this.#cloneParticipant(record.info), event });
                }
            }
        };
        this.#view.on("pointer", this.#pointerHandler);

        this.#initTransport();
        if (this.#transport && typeof setInterval === "function") {
            this.#presenceTimer = setInterval(() => this.#sendPresence(), this.#presenceIntervalMs);
        }

        this.#emitParticipants();
        this.#emitState("local", participantState);

        if (this.#transport) {
            this.#postMessage({
                type: "hello",
                id: this.#participantId,
                participant: this.#participantEnvelope(),
                clock: this.#tick(),
                state: this.#snapshotState(),
            });
            this.#sendPresence();
        }
    }

    /** Participant identifier for the local client. */
    get id(): string {
        return this.#participantId;
    }

    /** Broadcast channel identifier shared by all peers. */
    get sessionId(): string {
        return this.#sessionId;
    }

    /** Returns an immutable view of known participants. */
    get participants(): CanvasCollabParticipantState[] {
        return Array.from(this.#participants.values()).map((record) => this.#cloneParticipant(record.info));
    }

    on<K extends keyof CanvasCollabEventMap>(type: K, handler: EventHandler<CanvasCollabEventMap[K]>): void {
        this.#events.on(type, handler);
    }

    off<K extends keyof CanvasCollabEventMap>(type: K, handler: EventHandler<CanvasCollabEventMap[K]>): void {
        this.#events.off(type, handler);
    }

    /** Broadcasts an updated view state to collaborators. */
    sync(): void {
        this.#enqueueFullSync();
    }

    /** Tears down the collaboration session and restores patched methods. */
    destroy(): void {
        if (this.#destroyed) {
            return;
        }
        this.#destroyed = true;
        if (this.#transport) {
            this.#postMessage({
                type: "leave",
                id: this.#participantId,
                participant: this.#participantEnvelope(),
                clock: this.#tick(),
            });
            this.#transport.destroy();
            this.#transport = null;
        }
        if (this.#presenceTimer) {
            clearInterval(this.#presenceTimer);
            this.#presenceTimer = null;
        }
        if (this.#batchTimer) {
            clearTimeout(this.#batchTimer);
            this.#batchTimer = null;
        }
        if (this.#pointerHandler) {
            this.#view.off("pointer", this.#pointerHandler);
            this.#pointerHandler = null;
        }
        while (this.#restoreStack.length) {
            const restore = this.#restoreStack.pop();
            restore?.();
        }
    }

    #participantEnvelope(): { role: CanvasCollabRole; label?: string; color?: string } {
        const record = this.#participants.get(this.#participantId);
        if (record) {
            const { role, label, color } = record.info;
            return { role, label, color };
        }
        return {
            role: "human",
        };
    }

    #cloneParticipant(participant: CanvasCollabParticipantState): CanvasCollabParticipantState {
        return {
            id: participant.id,
            role: participant.role,
            label: participant.label,
            color: participant.color,
            state: cloneState(participant.state),
            lastSeen: participant.lastSeen,
            lastPointer: participant.lastPointer ? { ...participant.lastPointer, offset: { ...participant.lastPointer.offset } } : undefined,
        };
    }

    #channelName(): string {
        return `spiraltorch-canvas-${this.#sessionId}`;
    }

    #initTransport(): void {
        const name = this.#channelName();
        if (typeof BroadcastChannel !== "undefined") {
            try {
                this.#transport = new BroadcastChannelTransport(name);
                this.#transportKind = "broadcast";
            } catch {
                this.#transport = null;
            }
        }
        if (!this.#transport && hasLocalStorage()) {
            try {
                this.#transport = new StorageEventTransport(name);
                this.#transportKind = "storage";
            } catch {
                this.#transport = null;
            }
        }
        if (!this.#transport) {
            this.#transportKind = "none";
            this.#emitTelemetry({ type: "transport-change", transport: "none" });
            this.#emitTelemetry({ type: "degraded", reason: "no-transport" });
            return;
        }
        this.#transport.subscribe((envelope) => {
            this.#handleEnvelope(envelope);
        });
        this.#emitTelemetry({ type: "transport-change", transport: this.#transportKind });
        if (this.#transportKind === "storage") {
            this.#emitTelemetry({ type: "degraded", reason: "storage-fallback" });
        }
    }

    #tick(): number {
        this.#clock += 1;
        return this.#clock;
    }

    #snapshotState(): CanvasCollabState {
        const palette = this.#view.palette;
        const payload: SerializedPalette =
            typeof palette === "string"
                ? { kind: "builtin", name: palette }
                : { kind: "custom", palette: clonePalette(palette) };
        const offset = this.#view.offset;
        return {
            zoom: this.#view.zoom,
            offset: { x: offset.x, y: offset.y },
            palette: payload,
            pointerNavigation: this.#view.pointerNavigationEnabled,
            statsCurvature: this.#view.statsCurvature,
            statsInterval: this.#view.statsInterval,
            devicePixelRatio: this.#view.devicePixelRatio,
            running: this.#view.running,
        };
    }

    #wrapViewMethod(method: keyof SpiralCanvasView): void {
        const original = (this.#view as any)[method];
        if (typeof original !== "function") {
            return;
        }
        const session = this;
        (this.#view as any)[method] = function (...args: unknown[]) {
            const result = original.apply(this, args);
            if (!session.#applyingRemote) {
                session.#enqueueFullSync();
            }
            return result;
        };
        this.#restoreStack.push(() => {
            (this.#view as any)[method] = original;
        });
    }

    #broadcastState(patch?: Partial<CanvasCollabState>): void {
        if (this.#destroyed) {
            return;
        }
        const base = this.#lastState ?? this.#snapshotState();
        const next = patch ? mergeState(base, patch) : this.#snapshotState();
        if (statesEqual(this.#lastState, next)) {
            return;
        }
        const clonedNext = cloneState(next);
        this.#lastState = clonedNext;
        const record = this.#participants.get(this.#participantId);
        if (record) {
            record.info.state = cloneState(clonedNext);
            record.info.lastSeen = Date.now();
        }
        const clock = this.#tick();
        if (record) {
            record.clock = clock;
        }
        const diff: Partial<CanvasCollabState> = patch ? mergePatches(null, patch) : cloneState(clonedNext);
        if (this.#transport) {
            this.#postMessage({
                type: "state",
                id: this.#participantId,
                participant: this.#participantEnvelope(),
                clock,
                state: diff,
                full: !patch,
            });
        }
        if (record) {
            this.#recordAttribution("local", record.info, diff, clock);
        }
        if (record) {
            this.#emitState("local", record.info);
        }
        this.#emitParticipants();
    }

    #applyRemoteState(record: ParticipantRecord, state: CanvasCollabState): void {
        const resolved = this.#resolveConflicts(record.info, state);
        this.#applyingRemote = true;
        try {
            this.#view.setPointerNavigation(resolved.pointerNavigation);
            this.#view.setStatsCurvature(resolved.statsCurvature);
            this.#view.setStatsInterval(resolved.statsInterval);
            this.#view.setDevicePixelRatio(resolved.devicePixelRatio);
            this.#view.setPalette(deserializePalette(resolved.palette));
            this.#view.setZoom(resolved.zoom);
            this.#view.setOffset(resolved.offset.x, resolved.offset.y);
            const shouldRun = resolved.running;
            if (shouldRun && !this.#view.running) {
                this.#view.start();
            } else if (!shouldRun && this.#view.running) {
                this.#view.stop();
            }
            if (!this.#view.running) {
                this.#view.invalidate();
            }
        } finally {
            this.#applyingRemote = false;
        }
        const cloned = cloneState(resolved);
        this.#lastState = cloned;
        const selfRecord = this.#participants.get(this.#participantId);
        if (selfRecord) {
            selfRecord.info.state = cloneState(resolved);
            selfRecord.info.lastSeen = Date.now();
        }
        record.info.state = cloneState(resolved);
        record.info.lastSeen = Date.now();
        this.#emitParticipants();
    }

    #resolveConflicts(participant: CanvasCollabParticipantState, candidate: CanvasCollabState): CanvasCollabState {
        const localRecord = this.#participants.get(this.#participantId);
        const localPriority = localRecord ? rolePriority(localRecord.info.role) : rolePriority("human");
        const remotePriority = rolePriority(participant.role);
        const preferRemote =
            remotePriority > localPriority ||
            (remotePriority === localPriority && participant.id < this.#participantId);
        if (preferRemote) {
            return candidate;
        }
        const current = this.#snapshotState();
        const resolved = cloneState(candidate);
        const conflicts: (keyof CanvasCollabState)[] = [];
        if (candidate.zoom !== current.zoom) {
            resolved.zoom = current.zoom;
            conflicts.push("zoom");
        }
        if (candidate.offset.x !== current.offset.x || candidate.offset.y !== current.offset.y) {
            resolved.offset = { ...current.offset };
            conflicts.push("offset");
        }
        if (!paletteEquals(candidate.palette, current.palette)) {
            if (current.palette.kind === "builtin") {
                resolved.palette = { kind: "builtin", name: current.palette.name };
            } else {
                resolved.palette = { kind: "custom", palette: clonePalette(current.palette.palette) };
            }
            conflicts.push("palette");
        }
        if (candidate.pointerNavigation !== current.pointerNavigation) {
            resolved.pointerNavigation = current.pointerNavigation;
            conflicts.push("pointerNavigation");
        }
        if (candidate.statsCurvature !== current.statsCurvature) {
            resolved.statsCurvature = current.statsCurvature;
            conflicts.push("statsCurvature");
        }
        if (candidate.statsInterval !== current.statsInterval) {
            resolved.statsInterval = current.statsInterval;
            conflicts.push("statsInterval");
        }
        if (candidate.devicePixelRatio !== current.devicePixelRatio) {
            resolved.devicePixelRatio = current.devicePixelRatio;
            conflicts.push("devicePixelRatio");
        }
        if (candidate.running !== current.running) {
            resolved.running = current.running;
            conflicts.push("running");
        }
        conflicts.forEach((field) => {
            this.#emitTelemetry({ type: "conflict-resolved", participantId: participant.id, field });
        });
        return resolved;
    }

    #enqueuePatch(patch: Partial<CanvasCollabState>): void {
        if (this.#destroyed) {
            return;
        }
        this.#pendingPatch = mergePatches(this.#pendingPatch, patch);
        this.#scheduleBatch();
    }

    #enqueueFullSync(): void {
        if (this.#destroyed) {
            return;
        }
        this.#pendingFullSync = true;
        this.#scheduleBatch();
    }

    #scheduleBatch(delay: number = COLLAB_BATCH_INTERVAL_MS): void {
        if (this.#batchTimer) {
            return;
        }
        this.#batchTimer = setTimeout(() => {
            this.#batchTimer = null;
            this.#flushPendingState();
        }, delay);
    }

    #flushPendingState(): void {
        if (this.#destroyed) {
            return;
        }
        if (this.#pendingFullSync) {
            if (!this.#consumePatchBudget(1)) {
                this.#emitTelemetry({
                    type: "budget-suppressed",
                    participantId: this.#participantId,
                    remainingBudget: this.#patchAllowance,
                });
                this.#scheduleBatch(this.#timeUntilPatchBudget());
                return;
            }
            this.#pendingFullSync = false;
            this.#pendingPatch = null;
            this.#broadcastState();
            return;
        }
        if (this.#pendingPatch) {
            if (!this.#consumePatchBudget(1)) {
                this.#emitTelemetry({
                    type: "budget-suppressed",
                    participantId: this.#participantId,
                    remainingBudget: this.#patchAllowance,
                });
                this.#scheduleBatch(this.#timeUntilPatchBudget());
                return;
            }
            const patch = this.#pendingPatch;
            this.#pendingPatch = null;
            this.#broadcastState(patch);
        }
    }

    #consumePatchBudget(weight = 1): boolean {
        if (this.#patchRateHz <= 0) {
            return true;
        }
        this.#refillPatchBudget();
        if (this.#patchAllowance >= weight) {
            this.#patchAllowance -= weight;
            return true;
        }
        return false;
    }

    #refillPatchBudget(): void {
        if (this.#patchRateHz <= 0) {
            this.#patchAllowance = this.#patchRateHz;
            return;
        }
        const now = nowMillis();
        const elapsed = Math.max(0, now - this.#lastPatchRefill) / 1_000;
        if (elapsed > 0) {
            this.#patchAllowance = Math.min(
                this.#patchRateHz,
                this.#patchAllowance + elapsed * this.#patchRateHz,
            );
            this.#lastPatchRefill = now;
        }
    }

    #timeUntilPatchBudget(weight = 1): number {
        if (this.#patchRateHz <= 0) {
            return COLLAB_BATCH_INTERVAL_MS;
        }
        this.#refillPatchBudget();
        if (this.#patchAllowance >= weight) {
            return COLLAB_BATCH_INTERVAL_MS;
        }
        const deficit = weight - this.#patchAllowance;
        const seconds = deficit / this.#patchRateHz;
        return Math.max(COLLAB_BATCH_INTERVAL_MS, seconds * 1_000);
    }

    #recordAttribution(
        origin: "local" | "remote",
        participant: CanvasCollabParticipantState,
        patch: Partial<CanvasCollabState>,
        clock: number,
    ): void {
        if (!this.#attributionSink) {
            return;
        }
        const clonedPatch = mergePatches(null, patch);
        this.#attributionSink({
            participant: this.#cloneParticipant(participant),
            patch: clonedPatch,
            origin,
            clock,
        });
    }

    #handleEnvelope(envelope: CollabEnvelope): void {
        if (!envelope || typeof envelope !== "object") {
            return;
        }
        if (typeof envelope.v !== "number" || envelope.v !== COLLAB_SCHEMA_VERSION) {
            const version = typeof envelope.v === "number" ? envelope.v : -1;
            this.#emitTelemetry({ type: "unknown-schema", version });
            return;
        }
        const message = envelope.message;
        if (!message || typeof message !== "object") {
            return;
        }
        this.#handleMessage(message);
    }

    #handleMessage(message: CollabMessage): void {
        if (message.id === this.#participantId) {
            return;
        }
        const result = this.#upsertParticipant(message);
        if (!result) {
            return;
        }
        const { record } = result;
        switch (message.type) {
            case "hello": {
                if (message.clock > record.clock) {
                    record.clock = message.clock;
                    record.info.state = cloneState(message.state);
                    this.#applyRemoteState(record, message.state);
                    this.#recordAttribution("remote", record.info, message.state, message.clock);
                    this.#emitState("remote", record.info);
                }
                this.#enqueueFullSync();
                break;
            }
            case "state": {
                if (message.clock <= record.clock) {
                    this.#emitTelemetry({ type: "latency-suppressed", participantId: record.info.id, clock: message.clock });
                    break;
                }
                record.clock = message.clock;
                const merged = mergeState(record.info.state, message.state);
                record.info.state = cloneState(merged);
                record.info.lastSeen = Date.now();
                if (message.state.offset) {
                    record.info.lastPointer = {
                        kind: "pan",
                        source: "drag",
                        zoom: merged.zoom,
                        offset: { ...merged.offset },
                    };
                }
                this.#applyRemoteState(record, merged);
                this.#recordAttribution("remote", record.info, message.state, message.clock);
                this.#emitState("remote", record.info);
                break;
            }
            case "pointer": {
                record.info.lastPointer = {
                    kind: message.event.kind,
                    source: message.event.source,
                    zoom: message.event.zoom,
                    offset: { ...message.event.offset },
                };
                record.info.lastSeen = Date.now();
                this.#events.emit("pointer", {
                    participant: this.#cloneParticipant(record.info),
                    event: message.event,
                });
                this.#emitTelemetry({ type: "pointer", participantId: record.info.id });
                this.#emitParticipants();
                break;
            }
            case "leave": {
                this.#participants.delete(record.info.id);
                this.#emitTelemetry({ type: "leave", participantId: record.info.id });
                this.#emitParticipants();
                break;
            }
            case "presence": {
                this.#applyPresence(message);
                break;
            }
        }
    }

    #applyPresence(message: CollabPresenceMessage): void {
        const now = Date.now();
        message.roster.forEach((entry) => {
            if (entry.id === this.#participantId) {
                return;
            }
            const existing = this.#participants.get(entry.id);
            if (existing) {
                existing.info.role = entry.role;
                existing.info.lastSeen = entry.lastSeen ?? now;
            } else {
                const placeholderState = this.#lastState ? cloneState(this.#lastState) : this.#snapshotState();
                const info: CanvasCollabParticipantState = {
                    id: entry.id,
                    role: entry.role,
                    state: placeholderState,
                    lastSeen: entry.lastSeen ?? now,
                };
                this.#participants.set(entry.id, { info, clock: 0 });
                this.#emitTelemetry({ type: "join", participant: this.#cloneParticipant(info) });
            }
        });
        this.#emitTelemetry({ type: "presence", participants: this.#participants.size });
        this.#emitParticipants();
    }

    #sendPresence(): void {
        if (!this.#transport || this.#destroyed) {
            return;
        }
        const selfRecord = this.#participants.get(this.#participantId);
        if (selfRecord) {
            selfRecord.info.lastSeen = Date.now();
            this.#emitParticipants();
        }
        const roster = Array.from(this.#participants.values()).map((entry) => ({
            id: entry.info.id,
            role: entry.info.role,
            lastSeen: entry.info.lastSeen,
        }));
        this.#postMessage({
            type: "presence",
            id: this.#participantId,
            participant: this.#participantEnvelope(),
            clock: this.#tick(),
            roster,
        });
        this.#emitTelemetry({ type: "presence", participants: roster.length });
    }

    #upsertParticipant(message: CollabMessage): { record: ParticipantRecord; created: boolean } | null {
        const existing = this.#participants.get(message.id);
        const now = Date.now();
        if (existing) {
            existing.info.role = message.participant.role;
            existing.info.label = message.participant.label;
            existing.info.color = message.participant.color;
            existing.info.lastSeen = now;
            return { record: existing, created: false };
        }
        const placeholderState = this.#lastState ? cloneState(this.#lastState) : this.#snapshotState();
        const info: CanvasCollabParticipantState = {
            id: message.id,
            role: message.participant.role,
            label: message.participant.label,
            color: message.participant.color,
            state: placeholderState,
            lastSeen: now,
        };
        const record: ParticipantRecord = { info, clock: message.clock };
        this.#participants.set(message.id, record);
        this.#emitTelemetry({ type: "join", participant: this.#cloneParticipant(info) });
        return { record, created: true };
    }

    #postMessage(message: CollabMessage): void {
        if (!this.#transport) {
            return;
        }
        const envelope: CollabEnvelope = { v: COLLAB_SCHEMA_VERSION, message };
        const { serialized, bytes } = serializeEnvelope(envelope);
        if (bytes > this.#maxMessageBytes) {
            this.#emitTelemetry({ type: "message-dropped", reason: "oversize" });
            return;
        }
        try {
            this.#transport.post(envelope, serialized);
        } catch {
            this.#emitTelemetry({ type: "message-dropped", reason: "transport-error" });
        }
    }

    #emitParticipants(): void {
        this.#events.emit("participants", { participants: this.participants });
    }

    #emitState(origin: "local" | "remote", participant: CanvasCollabParticipantState): void {
        this.#events.emit("state", {
            participant: this.#cloneParticipant(participant),
            origin,
        });
    }

    #emitTelemetry(event: CollabTelemetryEvent): void {
        this.#telemetry?.(event);
    }
}

