import { SpiralCanvasView } from "./canvas-view";

/** Options configuring {@link SpiralCanvasRecorder}. */
export interface SpiralCanvasRecorderOptions {
    /** Preferred MIME type passed to the {@link MediaRecorder} constructor. */
    mimeType?: string;
    /** Bits per second used when encoding the video track. */
    videoBitsPerSecond?: number;
    /** Bits per second used when encoding the audio track (usually unused). */
    audioBitsPerSecond?: number;
    /** Optional timeslice forwarded to {@link MediaRecorder#start}. */
    timeslice?: number;
    /** Frame rate supplied to {@link HTMLCanvasElement#captureStream}. */
    frameRate?: number;
    /** Callback fired for every chunk emitted by the recorder. */
    ondata?: (chunk: Blob) => void;
}

function ensureMediaRecorder(): typeof MediaRecorder {
    if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
        throw new Error("MediaRecorder API is not available in this environment");
    }
    return MediaRecorder;
}

/**
 * Helper that wires {@link SpiralCanvasView#createCaptureStream} to the
 * browser's {@link MediaRecorder} so callers can collect recordings.
 */
export class SpiralCanvasRecorder {
    readonly #view: SpiralCanvasView;
    readonly #options: SpiralCanvasRecorderOptions;
    #recorder: MediaRecorder | null = null;
    #stream: MediaStream | null = null;
    #chunks: Blob[] = [];
    #recording = false;

    constructor(view: SpiralCanvasView, options: SpiralCanvasRecorderOptions = {}) {
        this.#view = view;
        this.#options = options;
    }

    /** Indicates whether the browser supports the MediaRecorder API. */
    static isSupported(): boolean {
        return typeof window !== "undefined" && typeof MediaRecorder !== "undefined";
    }

    /** Whether a recording is currently in progress. */
    get recording(): boolean {
        return this.#recording;
    }

    /** Underlying {@link MediaStream} feeding the recorder. */
    get stream(): MediaStream | null {
        return this.#stream;
    }

    /** Starts capturing video frames from the canvas. */
    start(): void {
        if (this.#recording) {
            return;
        }
        const MediaRecorderCtor = ensureMediaRecorder();
        const stream = this.#view.createCaptureStream(this.#options.frameRate);
        const recorder = new MediaRecorderCtor(stream, {
            mimeType: this.#options.mimeType,
            videoBitsPerSecond: this.#options.videoBitsPerSecond,
            audioBitsPerSecond: this.#options.audioBitsPerSecond,
        });

        this.#chunks = [];
        recorder.addEventListener("dataavailable", (event) => {
            if (!event.data || event.data.size === 0) {
                return;
            }
            this.#chunks.push(event.data);
            this.#options.ondata?.(event.data);
        });

        recorder.start(this.#options.timeslice);
        this.#recorder = recorder;
        this.#stream = stream;
        this.#recording = true;
    }

    /** Stops the recording and resolves with the final {@link Blob}. */
    async stop(): Promise<Blob> {
        if (!this.#recording || !this.#recorder) {
            throw new Error("Recorder is not running");
        }
        const recorder = this.#recorder;
        return await new Promise<Blob>((resolve, reject) => {
            const cleanup = () => {
                recorder.removeEventListener("stop", handleStop);
                recorder.removeEventListener("error", handleError as EventListener);
                this.#stream?.getTracks().forEach((track) => track.stop());
                this.#stream = null;
                this.#recorder = null;
                this.#recording = false;
            };
            const handleStop = () => {
                try {
                    const type = recorder.mimeType || this.#options.mimeType || "video/webm";
                    const blob = new Blob(this.#chunks, { type });
                    cleanup();
                    resolve(blob);
                } catch (error) {
                    cleanup();
                    reject(error instanceof Error ? error : new Error(String(error)));
                }
            };
            const handleError = (event: MediaRecorderErrorEvent) => {
                cleanup();
                reject(event.error);
            };

            recorder.addEventListener("stop", handleStop, { once: true });
            recorder.addEventListener("error", handleError as EventListener, { once: true });
            recorder.stop();
        });
    }

    /** Aborts the current recording and discards buffered data. */
    cancel(): void {
        if (!this.#recording || !this.#recorder) {
            return;
        }
        const recorder = this.#recorder;
        const stream = this.#stream;
        this.#recorder = null;
        this.#stream = null;
        this.#recording = false;
        this.#chunks = [];
        recorder.stop();
        stream?.getTracks().forEach((track) => track.stop());
    }
}
