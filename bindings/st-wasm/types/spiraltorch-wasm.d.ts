/**
 * TypeScript declarations for the `spiraltorch-wasm` package.
 *
 * The bindings expose the raw WebAssembly surface implemented in `bindings/st-wasm` so
 * JavaScript and TypeScript callers receive editor completions and compile-time safety.
 *
 * These definitions mirror the wasm-bindgen exports and intentionally keep the naming
 * scheme used by the generated JavaScript glue (snake_case for low-level routines and
 * camelCase when explicitly configured with `js_name`).
 */
declare module "spiraltorch-wasm" {
    /** Palette identifiers accepted by {@link FractalCanvas.set_palette}. */
    export type CanvasPaletteName =
        | "blue-magenta"
        | "blue_magenta"
        | "blue"
        | "turbo"
        | "grayscale"
        | "grey"
        | "gray";

    /** Canonical palette names reported by {@link FractalCanvas.palette}. */
    export type CanvasPaletteCanonicalName = "blue-magenta" | "turbo" | "grayscale";

    /** Labels emitted by {@link CanvasDesireControl.eventLabels}. */
    export type DesireControlEventLabel =
        | "lr_increase"
        | "lr_decrease"
        | "lr_clipped"
        | "temperature_suppress"
        | "quality_weight"
        | "quality_suppress"
        | "z_suppress"
        | "lr_slew_limit";

    export class CanvasFftLayout {
        private constructor();
        readonly fieldBytes: number;
        readonly fieldStride: number;
        readonly spectrumBytes: number;
        readonly spectrumStride: number;
        readonly uniformBytes: number;
    }

    export class CanvasGradientSummary {
        private constructor();
        readonly hypergradL1: number;
        readonly hypergradL2: number;
        readonly hypergradLInf: number;
        readonly hypergradMeanAbs: number;
        readonly hypergradRms: number;
        readonly hypergradCount: number;
        readonly realgradL1: number;
        readonly realgradL2: number;
        readonly realgradLInf: number;
        readonly realgradMeanAbs: number;
        readonly realgradRms: number;
        readonly realgradCount: number;
    }

    export class CanvasDesireInterpretation {
        private constructor();
        readonly hyperPressure: number;
        readonly realPressure: number;
        readonly balance: number;
        readonly stability: number;
        readonly saturation: number;
        readonly penaltyGain: number;
        readonly biasMix: number;
        readonly observationGain: number;
    }

    export class CanvasDesireControl {
        private constructor();
        readonly penaltyGain: number;
        readonly biasMix: number;
        readonly observationGain: number;
        readonly damping: number;
        readonly hyperLearningRateScale: number;
        readonly realLearningRateScale: number;
        readonly operatorMix: number;
        readonly operatorGain: number;
        readonly tuningGain: number;
        readonly targetEntropy: number;
        readonly learningRateEta: number;
        readonly learningRateMin: number;
        readonly learningRateMax: number;
        readonly learningRateSlew: number;
        readonly clipNorm: number;
        readonly clipFloor: number;
        readonly clipCeiling: number;
        readonly clipEma: number;
        readonly temperatureKappa: number;
        readonly temperatureSlew: number;
        readonly qualityGain: number;
        readonly qualityBias: number;
        eventsMask(): number;
        eventLabels(): DesireControlEventLabel[];
    }

    export class CanvasFramePacket {
        private constructor();
        readonly width: number;
        readonly height: number;
        readonly pixels: Uint8Array;
        readonly relation: Float32Array;
        readonly field: Float32Array;
        readonly trail: Float32Array;
        readonly hypergradRms: number;
        readonly realgradRms: number;
        readonly hypergradCount: number;
        readonly realgradCount: number;
        readonly balance: number;
        readonly stability: number;
        readonly saturation: number;
        readonly hyperLearningRateScale: number;
        readonly realLearningRateScale: number;
        readonly operatorMix: number;
        readonly operatorGain: number;
        readonly eventsMask: number;
    }

    export class FractalCanvas {
        constructor(capacity: number, width: number, height: number);
        readonly width: number;
        readonly height: number;
        framePacket(curvature: number): CanvasFramePacket;
        push_patch(
            relation: Float32Array,
            coherence: number,
            tension: number,
            depth: number,
        ): void;
        render_to_canvas(canvas: HTMLCanvasElement): void;
        pixels(): Uint8Array;
        vector_field(): Float32Array;
        vectorFieldFft(inverse: boolean): Float32Array;
        emitWasmTrail(curvature: number): Float32Array;
        relation(): Float32Array;
        hypergradWave(curvature: number): Float32Array;
        hypergradWaveCurrent(curvature: number): Float32Array;
        realgradWave(): Float32Array;
        realgradWaveCurrent(): Float32Array;
        gradientSummary(curvature: number): CanvasGradientSummary;
        desireInterpretation(curvature: number): CanvasDesireInterpretation;
        desireControl(curvature: number): CanvasDesireControl;
        desireControlUniform(curvature: number): Uint32Array;
        vectorFieldFftKernel(subgroup: boolean): string;
        vectorFieldFftUniform(inverse: boolean): Uint32Array;
        vectorFieldFftDispatch(subgroup: boolean): Uint32Array;
        hypergradOperatorKernel(subgroup: boolean): string;
        hypergradOperatorUniform(mix: number, gain: number): Float32Array;
        hypergradOperatorUniformFromControl(control: CanvasDesireControl): Float32Array;
        hypergradOperatorUniformAuto(curvature: number): Float32Array;
        hypergradOperatorDispatch(subgroup: boolean): Uint32Array;
        vectorFieldFftLayout(): CanvasFftLayout;
        reset_normalizer(): void;
        set_palette(name: CanvasPaletteName): void;
        palette(): CanvasPaletteCanonicalName;
    }

    export function available_palettes(): CanvasPaletteCanonicalName[];

    export type WasmFftPlanObject = {
        radix: number;
        tile_cols: number;
        segments: number;
        subgroup: boolean;
    };

    export class WasmFftPlan {
        constructor(radix: number, tileCols: number, segments: number, subgroup: boolean);
        readonly radix: number;
        readonly tileCols: number;
        readonly segments: number;
        readonly subgroup: boolean;
        workgroupSize(): number;
        wgsl(): string;
        spiralkHint(): string;
        toJson(): string;
        toObject(): WasmFftPlanObject;
        static fromJson(json: string): WasmFftPlan;
        static fromObject(value: unknown): WasmFftPlan;
    }

    export function auto_plan_fft(
        rows: number,
        cols: number,
        k: number,
        subgroup: boolean,
    ): WasmFftPlan | undefined;

    export function auto_fft_wgsl(
        rows: number,
        cols: number,
        k: number,
        subgroup: boolean,
    ): string | undefined;

    export function auto_fft_spiralk(
        rows: number,
        cols: number,
        k: number,
        subgroup: boolean,
    ): string | undefined;

    export function auto_fft_plan_json(
        rows: number,
        cols: number,
        k: number,
        subgroup: boolean,
    ): string | undefined;

    export function auto_fft_plan_object(
        rows: number,
        cols: number,
        k: number,
        subgroup: boolean,
    ): WasmFftPlanObject | undefined;

    export function fft_forward(buffer: Float32Array): Float32Array;
    export function fft_inverse(buffer: Float32Array): Float32Array;
    export function fft_forward_in_place(buffer: Float32Array): void;
    export function fft_inverse_in_place(buffer: Float32Array): void;

    export class WasmMellinLogGrid {
        constructor(log_start: number, log_step: number, samples: Float32Array);
        readonly logStart: number;
        readonly logStep: number;
        len(): number;
        isEmpty(): boolean;
        samples(): Float32Array;
        weights(): Float32Array;
        support(): Float32Array;
        weightedSeries(): Float32Array;
        evaluate(s: Float32Array): Float32Array;
        evaluateMany(sValues: Float32Array): Float32Array;
        evaluateVerticalLine(real: number, imagValues: Float32Array): Float32Array;
        evaluateMesh(realValues: Float32Array, imagValues: Float32Array): Float32Array;
        evaluateMeshMagnitude(realValues: Float32Array, imagValues: Float32Array): Float32Array;
        evaluateMeshLogMagnitude(
            realValues: Float32Array,
            imagValues: Float32Array,
            epsilon: number,
        ): Float32Array;
        hilbertInnerProduct(other: WasmMellinLogGrid): Float32Array;
        hilbertNorm(): number;
    }

    export function mellin_exp_decay_samples(
        log_start: number,
        log_step: number,
        len: number,
    ): Float32Array;

    export function mellin_exp_decay_samples_scaled(
        log_start: number,
        log_step: number,
        len: number,
        rate: number,
    ): Float32Array;

    export interface WasmTunerRecord {
        rows_min?: number | null;
        rows_max?: number | null;
        cols_min?: number;
        cols_max?: number;
        k_min?: number;
        k_max?: number;
        subgroup?: boolean | null;
        algo_topk?: number;
        ctile?: number;
        wg?: number;
        kl?: number;
        ch?: number;
        mode_midk?: number;
        mode_bottomk?: number;
        tile_cols?: number;
        radix?: number;
        segments?: number;
        use_2ce?: boolean;
    }

    export interface WasmTunerChoice {
        use_2ce: boolean;
        wg: number;
        kl: number;
        ch: number;
        algo_topk: number;
        ctile: number;
        mode_midk: number;
        mode_bottomk: number;
        tile_cols: number;
        radix: number;
        segments: number;
    }

    export type ResolvedFftPlanReport = {
        plan: WasmFftPlanObject;
        overrideApplied: boolean;
        heuristicUsed: boolean;
        source: "override" | "heuristic" | "fallback";
    };

    export class WasmTuner {
        constructor(json?: string | null);
        static fromObject(value: unknown): WasmTuner;
        loadJson(json: string): void;
        loadObject(value: unknown): void;
        mergeJson(json: string): void;
        mergeObject(value: unknown): void;
        len(): number;
        recordAt(index: number): WasmTunerRecord | undefined;
        findRecord(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmTunerRecord | undefined;
        isEmpty(): boolean;
        clear(): void;
        push(record: WasmTunerRecord): void;
        replaceIndex(index: number, record: WasmTunerRecord): boolean;
        removeIndex(index: number): WasmTunerRecord | undefined;
        removeRecord(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmTunerRecord | undefined;
        toJson(): string;
        records(): WasmTunerRecord[];
        toObject(): WasmTunerRecord[];
        choose(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmTunerChoice | undefined;
        planFft(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmFftPlan | undefined;
        planFftJson(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): string | undefined;
        planFftObject(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmFftPlanObject | undefined;
        planFftWithFallback(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmFftPlan;
        planFftWithFallbackJson(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): string;
        planFftWithFallbackObject(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): WasmFftPlanObject;
        planFftResolution(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): ResolvedWasmFftPlan;
        planFftResolutionJson(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): string;
        planFftResolutionObject(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): ResolvedFftPlanReport;
        planFftReport(
            rows: number,
            cols: number,
            k: number,
            subgroup: boolean,
        ): ResolvedFftPlanReport;
    }

    export enum WasmFftPlanSource {
        Override = 0,
        Heuristic = 1,
        Fallback = 2,
    }

    export class ResolvedWasmFftPlan {
        private constructor();
        readonly plan: WasmFftPlan;
        readonly overrideApplied: boolean;
        readonly heuristicUsed: boolean;
        readonly source: WasmFftPlanSource;
        toJson(): string;
        toObject(): ResolvedFftPlanReport;
        static fromJson(json: string): ResolvedWasmFftPlan;
        static fromObject(value: unknown): ResolvedWasmFftPlan;
    }

    export function baseChoice(
        rows: number,
        cols: number,
        k: number,
        subgroup: boolean,
    ): WasmTunerChoice;

    export type CobolInitiatorType = "human" | "model" | "automation";

    export interface CobolInitiator {
        type: CobolInitiatorType;
        name: string;
        persona?: string;
        revision?: string;
        contact?: string;
        notes?: string[];
    }

    export interface CobolMqRoute {
        manager: string;
        queue: string;
        commit?: string;
    }

    export interface CobolCicsRoute {
        transaction: string;
        program?: string;
        channel?: string;
    }

    export interface CobolNarratorPayload {
        curvature: number;
        temperature: number;
        encoder: string;
        locale?: string;
        coefficients?: number[];
    }

    export interface CobolMetadata {
        tags?: string[];
        annotations?: string[];
        extra?: unknown;
    }

    export interface CobolDatasetRoute {
        dataset: string;
        member?: string;
        disposition?: string;
        volume?: string;
        record_format?: string;
        record_length?: number;
        block_size?: number;
        data_class?: string;
        management_class?: string;
        storage_class?: string;
        space_primary?: number;
        space_secondary?: number;
        space_unit?: string;
        directory_blocks?: number;
        dataset_type?: string;
        like_dataset?: string;
        organization?: string;
        key_length?: number;
        key_offset?: number;
        control_interval_size?: number;
        share_options_cross_region?: number;
        share_options_cross_system?: number;
        reuse?: boolean;
        log?: boolean;
        unit?: string;
        unit_count?: number;
        average_record_unit?: string;
        catalog_behavior?: string;
        retention_period?: number;
        release_space?: boolean;
        erase_on_delete?: boolean;
        expiration_date?: string;
    }

    export interface CobolRoutePlan {
        mq?: CobolMqRoute;
        cics?: CobolCicsRoute;
        dataset?: string | CobolDatasetRoute;
    }

    export interface CobolDispatchEnvelope {
        job_id: string;
        release_channel: string;
        created_at: string;
        initiators: CobolInitiator[];
        route: CobolRoutePlan;
        payload: CobolNarratorPayload;
        metadata: CobolMetadata;
    }

    export interface CobolPreviewDataset {
        dataset: string;
        member?: string;
        disposition?: string;
        volume?: string;
        record_format?: string;
        record_length?: number;
        block_size?: number;
        data_class?: string;
        management_class?: string;
        storage_class?: string;
        space_primary?: number;
        space_secondary?: number;
        space_unit?: string;
        directory_blocks?: number;
        dataset_type?: string;
        like_dataset?: string;
        organization?: string;
        key_length?: number;
        key_offset?: number;
        control_interval_size?: number;
        share_options_cross_region?: number;
        share_options_cross_system?: number;
        reuse?: boolean;
        log?: boolean;
        unit?: string;
        unit_count?: number;
        average_record_unit?: string;
        catalog_behavior?: string;
        retention_period?: number;
        release_space?: boolean;
        erase_on_delete?: boolean;
        expiration_date?: string;
    }

    export interface CobolPreviewEnvelope {
        job_id: string;
        curvature: number;
        temperature: number;
        coefficient_count: number;
        release_channel: string;
        dataset?: CobolPreviewDataset;
    }

    export class CobolDispatchPlanner {
        constructor(jobId: string, releaseChannel?: string | null);
        static fromJson(json: string): CobolDispatchPlanner;
        static fromObject(envelope: CobolDispatchEnvelope): CobolDispatchPlanner;
        setReleaseChannel(channel: string): void;
        setCreatedAt(timestamp: string): void;
        resetCreatedAt(): void;
        setNarratorConfig(
            curvature: number,
            temperature: number,
            encoder: string,
            locale?: string | null,
        ): void;
        setCoefficients(coefficients: Float32Array): void;
        addHumanInitiator(
            name: string,
            persona?: string | null,
            contact?: string | null,
            note?: string | null,
        ): void;
        addModelInitiator(
            name: string,
            revision?: string | null,
            persona?: string | null,
            note?: string | null,
        ): void;
        addAutomationInitiator(
            name: string,
            persona?: string | null,
            note?: string | null,
        ): void;
        clearInitiators(): void;
        setMqRoute(manager: string, queue: string, commitMode?: string | null): void;
        clearMqRoute(): void;
        setCicsRoute(
            transaction: string,
            program?: string | null,
            channel?: string | null,
        ): void;
        clearCicsRoute(): void;
        setDataset(dataset?: string | null): void;
        setDatasetMember(member?: string | null): void;
        setDatasetDisposition(disposition?: string | null): void;
        setDatasetVolume(volume?: string | null): void;
        setDatasetRecordFormat(recordFormat?: string | null): void;
        setDatasetRecordLength(recordLength?: number | null): void;
        setDatasetBlockSize(blockSize?: number | null): void;
        setDatasetDataClass(dataClass?: string | null): void;
        setDatasetManagementClass(managementClass?: string | null): void;
        setDatasetStorageClass(storageClass?: string | null): void;
        setDatasetSpacePrimary(spacePrimary?: number | null): void;
        setDatasetSpaceSecondary(spaceSecondary?: number | null): void;
        setDatasetSpaceUnit(spaceUnit?: string | null): void;
        setDatasetDirectoryBlocks(directoryBlocks?: number | null): void;
        setDatasetType(datasetType?: string | null): void;
        setDatasetLike(likeDataset?: string | null): void;
        setDatasetOrganization(organization?: string | null): void;
        setDatasetKeyLength(keyLength?: number | null): void;
        setDatasetKeyOffset(keyOffset?: number | null): void;
        setDatasetControlIntervalSize(controlIntervalSize?: number | null): void;
        setDatasetShareOptionsCrossRegion(shareOptionsCrossRegion?: number | null): void;
        setDatasetShareOptionsCrossSystem(shareOptionsCrossSystem?: number | null): void;
        setDatasetReuse(reuse?: boolean | null): void;
        setDatasetLog(log?: boolean | null): void;
        setDatasetUnit(unit?: string | null): void;
        setDatasetUnitCount(unitCount?: number | null): void;
        setDatasetAverageRecordUnit(averageRecordUnit?: string | null): void;
        setDatasetCatalogBehavior(catalogBehavior?: string | null): void;
        setDatasetRetentionPeriod(retentionPeriod?: number | null): void;
        setDatasetReleaseSpace(releaseSpace?: boolean | null): void;
        setDatasetEraseOnDelete(eraseOnDelete?: boolean | null): void;
        setDatasetExpirationDate(expirationDate?: string | null): void;
        clearDataset(): void;
        clearRoute(): void;
        addTag(tag: string): void;
        addAnnotation(annotation: string): void;
        mergeMetadata(metadata: unknown): void;
        clearMetadata(): void;
        isValid(): boolean;
        validationIssues(): string[];
        loadJson(json: string): void;
        loadObject(envelope: CobolDispatchEnvelope): void;
        toObject(): CobolDispatchEnvelope;
        toJson(): string;
        toUint8Array(): Uint8Array;
        coefficientsAsBytes(): Uint8Array;
        readonly createdAt: string;
        mqRoute(): CobolMqRoute | undefined;
        cicsRoute(): CobolCicsRoute | undefined;
        toCobolPreview(): CobolPreviewEnvelope;
    }
}
