import type { CanvasPaletteCanonicalName } from "spiraltorch-wasm";
import {
    SpiralCanvasView,
    type CustomPalette,
    type CanvasPointerEvent,
} from "./canvas-view";

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

export interface SpiralCanvasCollabOptions {
    sessionId?: string;
    participant: CanvasCollabParticipant;
    sharePointer?: boolean;
}

interface CollabMessageBase {
    type: "hello" | "state" | "pointer" | "leave";
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

type CollabMessage = CollabHelloMessage | CollabStateMessage | CollabPointerMessage | CollabLeaveMessage;

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

    #channel: BroadcastChannel | null = null;
    #clock = 0;
    #applyingRemote = false;
    #destroyed = false;

    readonly #participants = new Map<string, ParticipantRecord>();
    #lastState: CanvasCollabState | null = null;

    #pointerHandler: ((event: CanvasPointerEvent) => void) | null = null;
    readonly #restoreStack: Array<() => void> = [];

    constructor(view: SpiralCanvasView, options: SpiralCanvasCollabOptions) {
        this.#view = view;
        this.#sessionId = options.sessionId ?? "spiraltorch";
        this.#sharePointer = options.sharePointer ?? true;

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
            const record = this.#participants.get(this.#participantId);
            if (record) {
                record.info.lastPointer = event;
                record.info.lastSeen = Date.now();
                record.info.state = this.#snapshotState();
                this.#emitState("local", record.info);
            }
            this.#broadcastState({ zoom: event.zoom, offset: event.offset });
            if (this.#sharePointer) {
                this.#postMessage({
                    type: "pointer",
                    id: this.#participantId,
                    participant: this.#participantEnvelope(),
                    clock: this.#tick(),
                    event,
                });
                if (record) {
                    this.#events.emit("pointer", { participant: this.#cloneParticipant(record.info), event });
                }
            }
        };
        this.#view.on("pointer", this.#pointerHandler);

        if (typeof BroadcastChannel !== "undefined") {
            try {
                this.#channel = new BroadcastChannel(this.#channelName());
                this.#channel.onmessage = (event) => {
                    this.#handleMessage(event.data as CollabMessage);
                };
            } catch {
                this.#channel = null;
            }
        }

        this.#emitParticipants();
        this.#emitState("local", participantState);

        if (this.#channel) {
            this.#postMessage({
                type: "hello",
                id: this.#participantId,
                participant: this.#participantEnvelope(),
                clock: this.#tick(),
                state: this.#snapshotState(),
            });
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
        this.#broadcastState();
    }

    /** Tears down the collaboration session and restores patched methods. */
    destroy(): void {
        if (this.#destroyed) {
            return;
        }
        this.#destroyed = true;
        if (this.#channel) {
            this.#postMessage({
                type: "leave",
                id: this.#participantId,
                participant: this.#participantEnvelope(),
                clock: this.#tick(),
            });
            this.#channel.close();
            this.#channel = null;
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
                session.#broadcastState();
            }
            return result;
        };
        this.#restoreStack.push(() => {
            (this.#view as any)[method] = original;
        });
    }

    #broadcastState(patch?: Partial<CanvasCollabState>): void {
        const base = this.#lastState ?? this.#snapshotState();
        const next = patch ? mergeState(base, patch) : this.#snapshotState();
        if (statesEqual(this.#lastState, next)) {
            return;
        }
        this.#lastState = next;
        const record = this.#participants.get(this.#participantId);
        if (record) {
            record.info.state = cloneState(next);
            record.info.lastSeen = Date.now();
        }
        if (this.#channel) {
            this.#postMessage({
                type: "state",
                id: this.#participantId,
                participant: this.#participantEnvelope(),
                clock: this.#tick(),
                state: patch ?? next,
                full: !patch,
            });
        }
        if (record) {
            this.#emitState("local", record.info);
        }
        this.#emitParticipants();
    }

    #applyRemoteState(state: CanvasCollabState): void {
        this.#applyingRemote = true;
        try {
            this.#view.setPointerNavigation(state.pointerNavigation);
            this.#view.setStatsCurvature(state.statsCurvature);
            this.#view.setStatsInterval(state.statsInterval);
            this.#view.setDevicePixelRatio(state.devicePixelRatio);
            this.#view.setPalette(deserializePalette(state.palette));
            this.#view.setZoom(state.zoom);
            this.#view.setOffset(state.offset.x, state.offset.y);
            const shouldRun = state.running;
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
        const cloned = cloneState(state);
        this.#lastState = cloned;
        const selfRecord = this.#participants.get(this.#participantId);
        if (selfRecord) {
            selfRecord.info.state = cloneState(state);
            selfRecord.info.lastSeen = Date.now();
        }
        this.#emitParticipants();
    }

    #handleMessage(message: CollabMessage): void {
        if (message.id === this.#participantId) {
            return;
        }
        const record = this.#upsertParticipant(message);
        if (!record) {
            return;
        }
        switch (message.type) {
            case "hello": {
                if (message.clock > record.clock) {
                    record.clock = message.clock;
                    record.info.state = cloneState(message.state);
                    this.#applyRemoteState(message.state);
                    this.#emitState("remote", record.info);
                }
                this.#broadcastState();
                break;
            }
            case "state": {
                if (message.clock <= record.clock) {
                    break;
                }
                record.clock = message.clock;
                const merged = message.full ? mergeState(record.info.state, message.state) : mergeState(record.info.state, message.state);
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
                this.#applyRemoteState(merged);
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
                this.#emitParticipants();
                break;
            }
            case "leave": {
                this.#participants.delete(record.info.id);
                this.#emitParticipants();
                break;
            }
        }
    }

    #upsertParticipant(message: CollabMessage): ParticipantRecord | null {
        const existing = this.#participants.get(message.id);
        const now = Date.now();
        if (existing) {
            existing.info.role = message.participant.role;
            existing.info.label = message.participant.label;
            existing.info.color = message.participant.color;
            existing.info.lastSeen = now;
            return existing;
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
        return record;
    }

    #postMessage(message: CollabMessage): void {
        if (!this.#channel) {
            return;
        }
        try {
            this.#channel.postMessage(message);
        } catch {
            // Ignore transport errors.
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
}

