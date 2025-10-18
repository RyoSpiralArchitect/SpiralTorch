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
const COLLAB_CAPABILITY_KEY_MAX_LENGTH = 64;
const COLLAB_CAPABILITY_MAX_ENTRIES_DEFAULT = 16;
const COLLAB_CAPABILITY_MAX_ENTRIES_HARD_LIMIT = 64;
const COLLAB_CAPABILITY_MAX_SCAN = 128;
const COLLAB_CAPABILITY_VALUE_MAX_BYTES_DEFAULT = 512;
const COLLAB_CAPABILITY_VALUE_MAX_BYTES_HARD_LIMIT = 4_096;

const TEXT_ENCODER =
    typeof TextEncoder !== "undefined" ? new TextEncoder() : null;

export type CanvasCollabRole = "trainer" | "model" | "human" | (string & {});

export type CollabCapabilityValue = boolean | number | string | null;

export type CollabCapabilities = Record<string, CollabCapabilityValue>;

export interface CanvasCollabParticipant {
    id?: string;
    role: CanvasCollabRole;
    label?: string;
    color?: string;
    capabilities?: CollabCapabilities;
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
    gain?: number | null;
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
    | { type: "policy-blocked"; participantId: string; reason: string }
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
    gain?: number | null;
}

export interface CollabRolePolicy {
    canPatch?: boolean;
    canState?: boolean;
    rateLimitHz?: number | null;
    gain?: number | null;
    allowedCapabilities?: ReadonlyArray<string> | null;
    blockedCapabilities?: ReadonlyArray<string> | null;
    maxCapabilityEntries?: number;
    maxCapabilityValueBytes?: number;
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
    rolePolicies?: Partial<Record<CanvasCollabRole, CollabRolePolicy>>;
    defaultRolePolicy?: CollabRolePolicy;
}

interface CollabMessageBase {
    type: "hello" | "state" | "pointer" | "leave" | "presence";
    id: string;
    participant: {
        role: CanvasCollabRole;
        label?: string;
        color?: string;
        capabilities?: CollabCapabilities | null;
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
    roster: Array<{
        id: string;
        role: CanvasCollabRole;
        lastSeen: number;
        capabilities?: CollabCapabilities | null;
    }>;
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

interface ResolvedRolePolicy {
    canPatch: boolean;
    canState: boolean;
    rateLimitHz: number | null;
    gain: number | null;
    allowedCapabilities: Set<string> | null;
    blockedCapabilities: Set<string>;
    maxCapabilityEntries: number;
    maxCapabilityValueBytes: number;
}

function createDefaultResolvedRolePolicy(): ResolvedRolePolicy {
    return {
        canPatch: true,
        canState: true,
        rateLimitHz: null,
        gain: null,
        allowedCapabilities: null,
        blockedCapabilities: new Set<string>(),
        maxCapabilityEntries: COLLAB_CAPABILITY_MAX_ENTRIES_DEFAULT,
        maxCapabilityValueBytes: COLLAB_CAPABILITY_VALUE_MAX_BYTES_DEFAULT,
    };
}

function cloneResolvedRolePolicy(policy: ResolvedRolePolicy): ResolvedRolePolicy {
    return {
        canPatch: policy.canPatch,
        canState: policy.canState,
        rateLimitHz: policy.rateLimitHz,
        gain: policy.gain,
        allowedCapabilities: policy.allowedCapabilities ? new Set(policy.allowedCapabilities) : null,
        blockedCapabilities: new Set(policy.blockedCapabilities),
        maxCapabilityEntries: policy.maxCapabilityEntries,
        maxCapabilityValueBytes: policy.maxCapabilityValueBytes,
    };
}

function normalizeRolePolicy(policy?: CollabRolePolicy | null): ResolvedRolePolicy {
    const resolved = createDefaultResolvedRolePolicy();
    if (!policy) {
        return resolved;
    }
    const rate =
        typeof policy.rateLimitHz === "number" && Number.isFinite(policy.rateLimitHz) && policy.rateLimitHz > 0
            ? policy.rateLimitHz
            : null;
    resolved.canPatch = policy.canPatch ?? true;
    resolved.canState = policy.canState ?? true;
    resolved.rateLimitHz = rate;
    resolved.gain = typeof policy.gain === "number" && Number.isFinite(policy.gain) ? policy.gain : null;
    resolved.allowedCapabilities = normalizeCapabilityKeySet(policy.allowedCapabilities);
    resolved.blockedCapabilities = normalizeCapabilityKeySet(policy.blockedCapabilities) ?? new Set<string>();
    resolved.maxCapabilityEntries = normalizeCapabilityBudget(
        policy.maxCapabilityEntries,
        COLLAB_CAPABILITY_MAX_ENTRIES_DEFAULT,
        COLLAB_CAPABILITY_MAX_ENTRIES_HARD_LIMIT,
    );
    resolved.maxCapabilityValueBytes = normalizeCapabilityBudget(
        policy.maxCapabilityValueBytes,
        COLLAB_CAPABILITY_VALUE_MAX_BYTES_DEFAULT,
        COLLAB_CAPABILITY_VALUE_MAX_BYTES_HARD_LIMIT,
    );
    return resolved;
}

function normalizeCapabilityKey(value: unknown): string | null {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    if (!trimmed || trimmed.length > COLLAB_CAPABILITY_KEY_MAX_LENGTH) {
        return null;
    }
    return trimmed;
}

function normalizeCapabilityKeySet(values?: ReadonlyArray<string> | null): Set<string> | null {
    if (!values) {
        return null;
    }
    const set = new Set<string>();
    for (const value of values) {
        const key = normalizeCapabilityKey(value);
        if (!key) {
            continue;
        }
        set.add(key);
        if (set.size >= COLLAB_CAPABILITY_MAX_ENTRIES_HARD_LIMIT) {
            break;
        }
    }
    return set;
}

function normalizeCapabilityBudget(
    value: number | null | undefined,
    fallback: number,
    hardLimit: number,
): number {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return fallback;
    }
    const clamped = Math.max(0, Math.floor(value));
    if (hardLimit >= 0) {
        return Math.min(clamped, hardLimit);
    }
    return clamped;
}

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
    policy: ResolvedRolePolicy;
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

function cloneCapabilities(capabilities?: CollabCapabilities | null): CollabCapabilities | undefined {
    if (!capabilities) {
        return undefined;
    }
    const clone: CollabCapabilities = {};
    for (const [key, value] of Object.entries(capabilities)) {
        clone[key] = value;
    }
    return clone;
}

function capabilitiesEqual(a?: CollabCapabilities | null, b?: CollabCapabilities | null): boolean {
    if (!a && !b) {
        return true;
    }
    if (!a || !b) {
        return false;
    }
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length) {
        return false;
    }
    for (const key of keysA) {
        if (!Object.prototype.hasOwnProperty.call(b, key)) {
            return false;
        }
        if (a[key] !== b[key]) {
            return false;
        }
    }
    return true;
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

interface CapabilityFilterResult {
    allowed: CollabCapabilities;
    blocked: Array<{ key: string; reason: string }>;
}

function measureByteLength(value: string): number {
    if (TEXT_ENCODER) {
        return TEXT_ENCODER.encode(value).byteLength;
    }
    return value.length;
}

function truncateStringByBytes(value: string, maxBytes: number): string {
    if (maxBytes <= 0) {
        return "";
    }
    if (measureByteLength(value) <= maxBytes) {
        return value;
    }
    if (!TEXT_ENCODER) {
        return value.slice(0, maxBytes);
    }
    let result = "";
    let total = 0;
    for (const char of value) {
        const encoded = TEXT_ENCODER.encode(char);
        if (total + encoded.byteLength > maxBytes) {
            break;
        }
        result += char;
        total += encoded.byteLength;
    }
    return result;
}

function sanitizeCapabilityValue(
    raw: unknown,
    maxBytes: number,
): CollabCapabilityValue | undefined {
    if (raw === null) {
        return null;
    }
    switch (typeof raw) {
        case "boolean":
            return raw;
        case "number":
            return Number.isFinite(raw) ? raw : undefined;
        case "string": {
            const limited = truncateStringByBytes(raw, maxBytes);
            return limited;
        }
        default:
            return undefined;
    }
}

function filterCapabilities(source: unknown, policy: ResolvedRolePolicy): CapabilityFilterResult {
    const allowed: CollabCapabilities = {};
    const blocked: Array<{ key: string; reason: string }> = [];
    if (source === undefined) {
        return { allowed, blocked };
    }
    if (source === null) {
        return { allowed, blocked };
    }
    if (typeof source !== "object") {
        blocked.push({ key: "*", reason: "invalid-structure" });
        return { allowed, blocked };
    }
    const entries = Object.entries(source as Record<string, unknown>);
    const inspectLimit = Math.min(entries.length, COLLAB_CAPABILITY_MAX_SCAN);
    const maxEntries = Math.max(0, Math.min(policy.maxCapabilityEntries, COLLAB_CAPABILITY_MAX_ENTRIES_HARD_LIMIT));
    let count = 0;
    for (let i = 0; i < inspectLimit; i += 1) {
        const [rawKey, rawValue] = entries[i];
        const key = normalizeCapabilityKey(rawKey);
        if (!key) {
            blocked.push({ key: String(rawKey), reason: "invalid-key" });
            continue;
        }
        if (policy.allowedCapabilities && !policy.allowedCapabilities.has(key)) {
            blocked.push({ key, reason: "not-allowed" });
            continue;
        }
        if (policy.blockedCapabilities.has(key)) {
            blocked.push({ key, reason: "blocked" });
            continue;
        }
        if (count >= maxEntries) {
            blocked.push({ key, reason: "budget-exhausted" });
            continue;
        }
        const value = sanitizeCapabilityValue(rawValue, policy.maxCapabilityValueBytes);
        if (value === undefined) {
            blocked.push({ key, reason: "invalid-value" });
            continue;
        }
        allowed[key] = value;
        count += 1;
    }
    if (entries.length > inspectLimit) {
        blocked.push({ key: "*", reason: "overflow" });
    }
    return { allowed, blocked };
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
        this.#rememberNonce(nonce);
        try {
            localStorage.setItem(this.#key, payload);
        } catch {
            this.#seen.delete(nonce);
            return;
        }
        this.#dispatch(payload, true);
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
        if (nonce) {
            if (this.#seen.has(nonce)) {
                if (!local) {
                    return;
                }
            } else if (!local) {
                this.#rememberNonce(nonce);
            }
        }
        try {
            const envelope = JSON.parse(parsed.envelope) as CollabEnvelope;
            this.#handler?.(envelope);
        } catch {
            // ignore malformed payloads
        }
    }

    #rememberNonce(nonce: string): void {
        this.#seen.add(nonce);
        if (this.#seen.size > 256) {
            const entries = Array.from(this.#seen);
            this.#seen = new Set(entries.slice(entries.length - 128));
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

    #basePatchRateHz: number;
    #patchRateHz: number;
    #patchAllowance: number;
    #lastPatchRefill: number;

    #presenceIntervalMs: number;
    #presenceTimer: ReturnType<typeof setInterval> | null = null;

    #maxMessageBytes: number;
    #telemetry?: CollabTelemetrySink;
    #attributionSink?: (event: CollabPatchAttributionEvent) => void;
    #rolePolicyByRole = new Map<string, ResolvedRolePolicy>();
    #defaultRolePolicy: ResolvedRolePolicy = createDefaultResolvedRolePolicy();

    constructor(view: SpiralCanvasView, options: SpiralCanvasCollabOptions) {
        this.#view = view;
        this.#sessionId = options.sessionId ?? "spiraltorch";
        this.#sharePointer = options.sharePointer ?? true;
        this.#telemetry = options.telemetry;
        this.#attributionSink = options.attributionSink;
        if (options.defaultRolePolicy) {
            this.#defaultRolePolicy = normalizeRolePolicy(options.defaultRolePolicy);
        }
        if (options.rolePolicies) {
            for (const [role, policy] of Object.entries(options.rolePolicies)) {
                if (!policy) {
                    continue;
                }
                this.#rolePolicyByRole.set(role, normalizeRolePolicy(policy));
            }
        }
        const pointerRateHz = Math.max(options.pointerRateHz ?? COLLAB_DEFAULT_POINTER_RATE_HZ, 1);
        this.#pointerMinInterval = 1_000 / pointerRateHz;
        this.#basePatchRateHz = Math.max(options.patchRateHz ?? COLLAB_DEFAULT_PATCH_RATE_HZ, 0);
        this.#patchRateHz = this.#basePatchRateHz;
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
        const localPolicy = this.#resolvePolicyForRole(participantState.role);
        participantState.gain = localPolicy.gain;
        const participantRecord: ParticipantRecord = {
            info: participantState,
            clock: 0,
            policy: localPolicy,
        };
        this.#participants.set(participantId, participantRecord);
        this.#applyCapabilities(participantRecord, options.participant.capabilities, "local-init");
        this.#applyLocalPolicy(localPolicy);
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

    /**
     * Updates the advertised capability surface for the local participant and
     * broadcasts a presence heartbeat so peers can observe the change without
     * waiting for the scheduled interval.
     */
    setCapabilities(capabilities: CollabCapabilities | null): void {
        if (this.#destroyed) {
            return;
        }
        const record = this.#participants.get(this.#participantId);
        if (!record) {
            return;
        }
        const changed = this.#applyCapabilities(record, capabilities, "local-update");
        if (changed) {
            this.#emitParticipants();
        }
        this.#sendPresence();
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

    #participantEnvelope(): {
        role: CanvasCollabRole;
        label?: string;
        color?: string;
        capabilities?: CollabCapabilities | null;
    } {
        const record = this.#participants.get(this.#participantId);
        if (record) {
            const { role, label, color, capabilities } = record.info;
            const envelope: {
                role: CanvasCollabRole;
                label?: string;
                color?: string;
                capabilities?: CollabCapabilities | null;
            } = { role, label, color };
            const clonedCapabilities = cloneCapabilities(capabilities);
            if (clonedCapabilities && Object.keys(clonedCapabilities).length > 0) {
                envelope.capabilities = clonedCapabilities;
            }
            return envelope;
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
            gain: participant.gain ?? null,
            capabilities: cloneCapabilities(participant.capabilities) ?? undefined,
            lastPointer: participant.lastPointer ? { ...participant.lastPointer, offset: { ...participant.lastPointer.offset } } : undefined,
        };
    }

    #resolvePolicyForRole(role: CanvasCollabRole): ResolvedRolePolicy {
        const template = this.#rolePolicyByRole.get(role) ?? this.#defaultRolePolicy;
        return cloneResolvedRolePolicy(template);
    }

    #refreshRecordPolicy(record: ParticipantRecord, role: CanvasCollabRole): void {
        record.policy = this.#resolvePolicyForRole(role);
        record.info.gain = record.policy.gain;
        if (record.info.id === this.#participantId) {
            this.#applyLocalPolicy(record.policy);
        } else {
            const currentCaps = cloneCapabilities(record.info.capabilities) ?? {};
            const capsChanged = this.#applyCapabilities(record, currentCaps, "policy-update");
            if (capsChanged) {
                this.#emitParticipants();
            }
        }
    }

    #applyLocalPolicy(policy: ResolvedRolePolicy): void {
        const limit = policy.rateLimitHz;
        const nextRate = limit !== null ? Math.max(0, Math.min(this.#basePatchRateHz, limit)) : this.#basePatchRateHz;
        this.#patchRateHz = nextRate;
        if (this.#patchRateHz <= 0) {
            this.#patchAllowance = 0;
        } else if (!Number.isFinite(this.#patchAllowance) || this.#patchAllowance > this.#patchRateHz) {
            this.#patchAllowance = this.#patchRateHz;
        }
        this.#lastPatchRefill = nowMillis();
        if (!policy.canPatch) {
            this.#pendingPatch = null;
        }
        if (!policy.canState) {
            this.#pendingFullSync = false;
        }
        const localRecord = this.#participants.get(this.#participantId);
        if (localRecord) {
            const currentCaps = cloneCapabilities(localRecord.info.capabilities) ?? {};
            const capsChanged = this.#applyCapabilities(localRecord, currentCaps, "local-policy");
            if (capsChanged) {
                this.#emitParticipants();
            }
        }
    }

    #applyCapabilities(record: ParticipantRecord, candidate: unknown, context: string): boolean {
        if (candidate === undefined) {
            return false;
        }
        const { allowed, blocked } = filterCapabilities(candidate, record.policy);
        const sanitized = Object.keys(allowed).length > 0 ? allowed : undefined;
        const changed = !capabilitiesEqual(record.info.capabilities, sanitized);
        if (sanitized) {
            record.info.capabilities = sanitized;
        } else {
            delete (record.info as Partial<CanvasCollabParticipantState>).capabilities;
        }
        if (blocked.length) {
            for (const entry of blocked) {
                const keyTag = entry.key === "*" ? "aggregate" : entry.key;
                const reason = `capability:${context}:${entry.reason}:${keyTag}`;
                const trimmedReason = reason.length > 128 ? `${reason.slice(0, 125)}...` : reason;
                this.#emitTelemetry({ type: "policy-blocked", participantId: record.info.id, reason: trimmedReason });
            }
        }
        return changed;
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
        const record = this.#participants.get(this.#participantId);
        if (!record) {
            return;
        }
        if (patch && !record.policy.canPatch) {
            this.#emitTelemetry({ type: "policy-blocked", participantId: this.#participantId, reason: "patch-disabled" });
            return;
        }
        if (!patch && !record.policy.canState) {
            this.#emitTelemetry({ type: "policy-blocked", participantId: this.#participantId, reason: "state-disabled" });
            return;
        }
        const base = this.#lastState ?? this.#snapshotState();
        const next = patch ? mergeState(base, patch) : this.#snapshotState();
        if (statesEqual(this.#lastState, next)) {
            return;
        }
        const clonedNext = cloneState(next);
        this.#lastState = clonedNext;
        record.info.state = cloneState(clonedNext);
        record.info.lastSeen = Date.now();
        const clock = this.#tick();
        record.clock = clock;
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
        this.#recordAttribution("local", record.info, diff, clock, record.policy);
        this.#emitState("local", record.info);
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
        const record = this.#participants.get(this.#participantId);
        if (!record || !record.policy.canPatch) {
            this.#emitTelemetry({ type: "policy-blocked", participantId: this.#participantId, reason: "patch-disabled" });
            return;
        }
        this.#pendingPatch = mergePatches(this.#pendingPatch, patch);
        this.#scheduleBatch();
    }

    #enqueueFullSync(): void {
        if (this.#destroyed) {
            return;
        }
        const record = this.#participants.get(this.#participantId);
        if (!record || !record.policy.canState) {
            this.#emitTelemetry({ type: "policy-blocked", participantId: this.#participantId, reason: "state-disabled" });
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
        policy: ResolvedRolePolicy,
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
            gain: policy.gain,
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
                if (!record.policy.canState) {
                    record.clock = Math.max(record.clock, message.clock);
                    record.info.lastSeen = Date.now();
                    this.#emitTelemetry({ type: "policy-blocked", participantId: record.info.id, reason: "state-disabled" });
                    this.#enqueueFullSync();
                    break;
                }
                if (message.clock > record.clock) {
                    record.clock = message.clock;
                    record.info.state = cloneState(message.state);
                    this.#applyRemoteState(record, message.state);
                    this.#recordAttribution("remote", record.info, message.state, message.clock, record.policy);
                    this.#emitState("remote", record.info);
                }
                this.#enqueueFullSync();
                break;
            }
            case "state": {
                if (!record.policy.canPatch) {
                    record.clock = Math.max(record.clock, message.clock);
                    record.info.lastSeen = Date.now();
                    this.#emitTelemetry({ type: "policy-blocked", participantId: record.info.id, reason: "patch-disabled" });
                    break;
                }
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
                this.#recordAttribution("remote", record.info, message.state, message.clock, record.policy);
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
                this.#refreshRecordPolicy(existing, entry.role);
                this.#applyCapabilities(existing, entry.capabilities, "presence");
            } else {
                const placeholderState = this.#lastState ? cloneState(this.#lastState) : this.#snapshotState();
                const info: CanvasCollabParticipantState = {
                    id: entry.id,
                    role: entry.role,
                    state: placeholderState,
                    lastSeen: entry.lastSeen ?? now,
                };
                const policy = this.#resolvePolicyForRole(entry.role);
                info.gain = policy.gain;
                const record: ParticipantRecord = { info, clock: 0, policy };
                this.#participants.set(entry.id, record);
                this.#applyCapabilities(record, entry.capabilities, "presence");
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
            capabilities: cloneCapabilities(entry.info.capabilities) ?? null,
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
            this.#refreshRecordPolicy(existing, message.participant.role);
            existing.info.gain = existing.policy.gain;
            this.#applyCapabilities(existing, message.participant.capabilities, "message");
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
        const policy = this.#resolvePolicyForRole(info.role);
        info.gain = policy.gain;
        const record: ParticipantRecord = { info, clock: message.clock, policy };
        this.#participants.set(message.id, record);
        this.#applyCapabilities(record, message.participant.capabilities, "message");
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

