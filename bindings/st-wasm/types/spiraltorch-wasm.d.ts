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

    export type WasmReportAuditStatus = "ready" | "usable" | "needs_attention";

    export type ToposRuntimeMode =
        | "balanced"
        | "guarded"
        | "exploratory"
        | "contextual"
        | "training_first"
        | "inference_first";

    export type ToposRuntimeProfileInput = {
        training_gain?: number;
        inference_gain?: number;
        closure_risk?: number;
        exploration_budget?: number;
        control_energy?: number;
        training_rate_scale?: number;
        training_gradient_bias_scale?: number;
        inference_temperature?: number;
        inference_top_p?: number;
        inference_context_weight?: number;
        learning_inference_balance?: number;
    };

    export type ToposRuntimeProfile = Required<ToposRuntimeProfileInput> & {
        vector: [number, number, number, number, number];
    };

    export type ToposRuntimeRouteScores = {
        training: number;
        inference: number;
        guard: number;
        exploration: number;
        context: number;
        vector: number[];
    };

    export type ToposRuntimeRoute = {
        kind: "spiraltorch.topos_runtime_route";
        contract_version: "spiraltorch.topos_runtime_route.v1";
        semantic_owner: "st-tensor::pure::topos";
        semantic_backend: "rust";
        execution_client: "wasm";
        mode: ToposRuntimeMode;
        mode_id: number;
        score: number;
        score_key: "training" | "inference" | "guard" | "exploration" | "context";
        learning_action: string;
        inference_action: string;
        scores: ToposRuntimeRouteScores;
        runtime_profile: ToposRuntimeProfile;
    };

    export type RuntimeDeviceRouteEvidence = {
        requested_backend: string;
        effective_backend?: string | null;
        runtime_ready?: boolean | null;
        requested_backend_runtime_ready?: boolean | null;
        effective_backend_runtime_ready?: boolean | null;
        available?: boolean | null;
        runtime_status?: string | null;
        requested_backend_runtime_status?: string | null;
        effective_backend_runtime_status?: string | null;
        status?: string | null;
        error?: string | null;
    };

    export type RuntimeDeviceRouteRequest = {
        reports?: RuntimeDeviceRouteEvidence[];
        requested_backends?: string[];
        required_available_backends?: string[];
        required_ready_backends?: string[];
    };

    export type RuntimeDeviceRouteRow = {
        requested_backend: string;
        effective_backend: string;
        report_available: boolean;
        native_ready: boolean | null;
        route_ready: boolean;
        fallback: boolean;
        route: "direct" | "surrogate" | "unavailable";
        route_status: "ready" | "surrogate_ready" | "not_ready" | "error";
        runtime_status: string;
        requested_backend_runtime_status: string | null;
        effective_backend_runtime_status: string | null;
        diagnostic: string | null;
    };

    export type RuntimeDeviceRoute = {
        kind: "spiraltorch.runtime_device_route";
        contract_version: "spiraltorch.runtime_device_route.v1";
        semantic_owner: "st-core::backend::runtime_route";
        semantic_backend: "rust";
        execution_client: "wasm";
        backends: string[];
        report_count: number;
        routes: RuntimeDeviceRouteRow[];
        available_backends: string[];
        native_ready_backends: string[];
        native_not_ready_backends: string[];
        native_readiness_unknown_backends: string[];
        ready_backends: string[];
        not_ready_backends: string[];
        fallback_backends: string[];
        error_backends: string[];
        missing_report_backends: string[];
        status_by_backend: Record<string, string>;
        all_ready: boolean;
        has_errors: boolean;
        required_available_backends: string[];
        required_available_backends_missing: string[];
        required_available_backends_passed: boolean | null;
        required_ready_backends: string[];
        required_ready_backends_missing: string[];
        required_ready_backends_passed: boolean | null;
        failures: string[];
        passed: boolean;
    };

    export type ToposRoutePolicyProfile =
        | "balanced"
        | "quality"
        | "grounded"
        | "efficiency"
        | "latency";

    export type ToposRoutePolicyRow = {
        label: string;
        count?: number;
        trace_route_score?: number | null;
        trace_quality_score?: number | null;
        trace_efficiency_score?: number | null;
        trace_text_quality_score?: number | null;
        response_text_quality_score?: number | null;
        response_prompt_coverage?: number | null;
        response_completion_rate?: number | null;
        response_incomplete_rate?: number | null;
        response_confidence?: number | null;
        latency_ms_mean?: number | null;
        total_tokens?: number | null;
        adapter_runtime_route_score?: number | null;
        adapter_guard_score?: number | null;
        adapter_exploration_score?: number | null;
        adapter_context_score?: number | null;
        closure_pressure?: number | null;
        openness?: number | null;
        context_weight?: number | null;
        request_temperature?: number | null;
        mode?: string | null;
        selection_scores?: Partial<Record<ToposRoutePolicyProfile, number>>;
    };

    export type ToposRoutePolicyWinner = {
        label: string | null;
        score: number;
        trace_route_score: number;
        trace_quality_score: number;
        trace_efficiency_score: number;
        response_text_quality_score: number;
        response_prompt_coverage: number;
        response_completion_rate: number;
        latency_ms_mean: number;
        total_tokens: number;
        adapter_runtime_route_score: number;
        adapter_guard_score: number;
        adapter_context_score: number;
        closure_pressure: number | null;
        openness: number | null;
        context_weight: number | null;
    };

    export type ToposRoutePolicyEvaluationRequest = {
        rows: ToposRoutePolicyRow[];
    };

    export type ToposRoutePolicyEvaluation = {
        kind: "spiraltorch.topos_route_policy";
        contract_version: "spiraltorch.topos_route_policy.v1";
        semantic_owner: "st-core::runtime::topos_route_policy";
        semantic_backend: "rust";
        execution_client: "wasm";
        row_count: number;
        active_row_count: number;
        rows: ToposRoutePolicyRow[];
        profiles: Record<ToposRoutePolicyProfile, ToposRoutePolicyWinner>;
    };

    export type ToposRouteRewardsRequest = {
        rows: ToposRoutePolicyRow[];
        profile?: ToposRoutePolicyProfile;
    };

    export type ToposRouteReward = {
        index: number;
        source_index: number;
        label: string;
        profile: ToposRoutePolicyProfile;
        reward: number;
        count: number;
        trace_route_score: number;
        response_text_quality_score: number;
        response_completion_rate: number;
        response_incomplete_rate: number;
        adapter_runtime_route_score: number;
    };

    export type ToposRouteRewards = {
        kind: "spiraltorch.topos_route_rewards";
        contract_version: "spiraltorch.topos_route_policy.v1";
        semantic_owner: "st-core::runtime::topos_route_policy";
        semantic_backend: "rust";
        execution_client: "wasm";
        profile: ToposRoutePolicyProfile;
        input_row_count: number;
        reward_count: number;
        rewards: ToposRouteReward[];
    };

    export type ToposRoutePolicyResolveRequest = {
        rewards: ToposRouteReward[];
        selected_label?: string | null;
        selected_index?: number;
    };

    export type ToposRoutePolicyResolution = {
        kind: "spiraltorch.topos_route_policy_resolution";
        contract_version: "spiraltorch.topos_route_policy.v1";
        semantic_owner: "st-core::runtime::topos_route_policy";
        semantic_backend: "rust";
        execution_client: "wasm";
        resolution: "label" | "index" | "none";
        selected_position: number | null;
        selected_label: string | null;
        selected_reward: number | null;
        route_reward: ToposRouteReward | null;
    };

    export type ToposControlSignalInput = {
        curvature?: number;
        tolerance?: number;
        saturation?: number;
        porosity?: number;
        max_depth?: number;
        max_volume?: number;
        observed_depth?: number;
        visited_volume?: number;
    };

    export type ToposTrainingHints = {
        learning_rate_scale: number;
        regularization_scale: number;
        step_damping: number;
        gradient_bias_scale: number;
        clip_scale: number;
        momentum_damping: number;
        vector: number[];
    };

    export type ToposTrainingPlan = ToposTrainingHints & {
        gain: number;
        raw_rate_scale: number;
        rate_scale: number;
        effective_gradient_bias_scale: number;
        effective_momentum_damping: number;
    };

    export type ToposInferenceHints = {
        temperature_scale: number;
        top_p_scale: number;
        sampling_focus: number;
        frequency_penalty_bias: number;
        presence_penalty_bias: number;
        context_weight: number;
        vector: number[];
    };

    export type ToposInferencePlan = {
        gain: number;
        temperature: number;
        top_p: number;
        frequency_penalty: number;
        presence_penalty: number;
        context_weight: number;
        temperature_scale: number;
        top_p_scale: number;
        sampling_focus: number;
        vector: number[];
    };

    export type ToposControlSignal = Required<ToposControlSignalInput> & {
        kind: "spiraltorch.topos_control_signal";
        contract_version: "spiraltorch.topos_control_signal.v1";
        semantic_owner: "st-tensor::pure::topos";
        semantic_backend: "rust";
        execution_client: "wasm";
        remaining_volume: number;
        depth_pressure: number;
        volume_pressure: number;
        closure_pressure: number;
        openness: number;
        guard_strength: number;
        stability_hint: number;
        exploration_hint: number;
        learning_rate_scale: number;
        temperature_scale: number;
        regularization_scale: number;
        step_damping: number;
        sampling_focus: number;
        runtime_hints: number[];
        gradient: number[];
        training_hints: ToposTrainingHints;
        training_plan: ToposTrainingPlan;
        inference_hints: ToposInferenceHints;
        inference_plan: ToposInferencePlan;
        runtime_profile: ToposRuntimeProfile;
        runtime_route: ToposRuntimeRoute;
    };

    export type ToposInferencePlanOptions = {
        gain?: number;
        base_temperature?: number;
        base_top_p?: number;
        min_temperature?: number;
        max_temperature?: number;
        min_top_p?: number;
        max_top_p?: number;
        base_frequency_penalty?: number;
        base_presence_penalty?: number;
    };

    export type ToposControlPlanOptions = {
        training_gain?: number;
        inference?: ToposInferencePlanOptions;
    };

    export type ToposOptimizerSnapshotRequest = {
        signal: ToposControlSignalInput;
        sequence: number;
        hyper_learning_rate: number;
        real_learning_rate: number;
        options?: ToposControlPlanOptions;
        training_hints?: Partial<ToposTrainingHints>;
        inference_hints?: Partial<ToposInferenceHints>;
    };

    export type ToposOptimizerApplication = {
        scope: "learning_rate_and_gradient_state";
        control_path: "control.training_plan";
        input_hyper_learning_rate: number;
        input_real_learning_rate: number;
        rate_scale: number;
        hyper_learning_rate: number;
        real_learning_rate: number;
        gradient_bias_rule: "g_biased[i]=g[i]+rms(g)*bias_scale*basis[i%10]";
        gradient_bias_normalization: "raw_gradient_rms";
        effective_gradient_bias_scale: number;
        gradient_bias_basis_dim: 10;
        gradient_bias_basis: number[];
        momentum_rule: "m_t=damping*m_(t-1)+(1-damping)*g_biased";
        effective_momentum_damping: number;
    };

    export type ToposOptimizerSnapshot = {
        kind: "spiraltorch.topos_optimizer_snapshot";
        contract_version: "spiraltorch.topos_optimizer_snapshot.v2";
        semantic_owner: "st-tensor::pure::topos";
        semantic_backend: "rust";
        execution_client: "wasm";
        sequence: number;
        control: Omit<ToposControlSignal, "execution_client">;
        optimizer_application: ToposOptimizerApplication;
    };

    export type ToposZSpaceProjection = {
        kind: "spiraltorch.topos_zspace_projection";
        contract_version: "spiraltorch.topos_zspace_projection.v1";
        semantic_owner: "st-tensor::pure::topos";
        semantic_backend: "rust";
        execution_client: "wasm";
        gradient_dim: number;
        base_gradient_dim: 6;
        speed: number;
        memory: number;
        stability: number;
        drs: number;
        frac: number;
        gradient: number[];
        vector: number[];
    };

    export type ZSpaceFusionStrategy = "mean" | "last" | "max" | "min";

    export type ZSpaceTelemetrySummary = {
        count: number;
        l1: number;
        l2: number;
        linf: number;
        mean: number;
        variance: number;
        energy: number;
        amplitude: number;
        positive: number;
        negative: number;
        balance: number;
        focus: number;
    };

    export type ZSpaceTelemetrySourceAudit = {
        index: number;
        flattened_count: number;
        ignored_value_count: number;
        conflict_count: number;
    };

    export type ZSpaceTelemetryFusion = {
        kind: "spiraltorch.zspace_telemetry_fusion";
        contract_version: "spiraltorch.zspace_telemetry_fusion.v1";
        semantic_owner: "st-core::telemetry::zspace_fusion";
        semantic_backend: "rust";
        execution_client: "wasm";
        payload: Record<string, number>;
        summary: ZSpaceTelemetrySummary;
        input_count: number;
        active_input_count: number;
        ignored_value_count: number;
        conflict_count: number;
        sources: ZSpaceTelemetrySourceAudit[];
    };

    export type ZSpacePartialInput = {
        metrics: Record<string, number | number[]>;
        weight?: number;
        origin?: string | null;
        telemetry?: Record<string, unknown> | null;
    };

    export type ZSpacePartialFusionRequest = {
        partials: Array<ZSpacePartialInput | null>;
        weights?: number[] | null;
        strategy?: ZSpaceFusionStrategy;
        telemetry?: Array<Record<string, unknown>>;
    };

    export type ZSpacePartialSourceAudit = {
        index: number;
        origin: string | null;
        weight: number | null;
        status: "active" | "suppressed" | "null";
        metric_count: number;
        gradient_dim: number;
        telemetry_entry_count: number;
    };

    export type ZSpacePartialFusion = {
        kind: "spiraltorch.zspace_partial_fusion";
        contract_version: "spiraltorch.zspace_partial_fusion.v1";
        semantic_owner: "st-core::telemetry::zspace_fusion";
        semantic_backend: "rust";
        execution_client: "wasm";
        strategy: ZSpaceFusionStrategy;
        metrics: Record<string, number>;
        gradient?: number[];
        telemetry: ZSpaceTelemetryFusion;
        input_count: number;
        active_count: number;
        suppressed_count: number;
        null_count: number;
        sources: ZSpacePartialSourceAudit[];
    };

    export type ZSpaceCoherenceDiagnosticsInput = {
        mean_coherence: number;
        coherence_entropy: number;
        energy_ratio: number;
        z_bias: number;
        fractional_order: number;
        normalized_weights?: number[];
        preserved_channels?: number | null;
        discarded_channels?: number | null;
        dominant_channel?: number | null;
    };

    export type ZSpaceCoherenceContourInput = {
        coherence_strength: number;
        prosody_index: number;
        articulation_bias: number;
        timbre_spread?: number | null;
    };

    export type ZSpaceCoherenceProjectionConfig = {
        speed_gain?: number;
        stability_gain?: number;
        frac_gain?: number;
        drs_gain?: number;
    };

    export type ZSpaceCoherenceProjectionRequest = {
        diagnostics: ZSpaceCoherenceDiagnosticsInput;
        coherence?: number[];
        contour?: ZSpaceCoherenceContourInput | null;
        config?: ZSpaceCoherenceProjectionConfig;
        classification_policy?: ZSpaceCoherenceClassificationPolicy;
    };

    export type ZSpaceCoherenceProjectionDerived = {
        channels: number;
        distribution_source: "normalized_weights" | "diagnostic_entropy";
        weight_mass: number;
        weight_entropy: number;
        normalized_entropy: number;
        concentration: number;
        effective_channels: number;
        response_peak: number;
        response_mean: number;
    };

    export type ZSpaceCoherenceLabel =
        | "background"
        | "symmetric_pulse"
        | "cascade_imbalance"
        | "diffuse_drift";

    export type ZSpaceCoherenceClassificationPolicy = {
        background_energy_ratio_max: number;
        cascade_energy_ratio_min: number;
    };

    export type ZSpaceCoherenceClassification = {
        kind: "spiraltorch.zspace_coherence_classification";
        contract_version: "spiraltorch.zspace_coherence_classification.v1";
        semantic_owner: "st-core::inference::zspace_coherence";
        semantic_backend: "rust";
        classification_formula: string;
        label: ZSpaceCoherenceLabel;
        reason: string;
        energy_ratio: number;
        swap_invariant: boolean;
        policy: ZSpaceCoherenceClassificationPolicy;
    };

    export type ZSpaceCoherenceControl = {
        kind: "spiraltorch.zspace_coherence_control";
        contract_version: "spiraltorch.zspace_coherence_control.v1";
        semantic_owner: "st-core::inference::zspace_coherence";
        semantic_backend: "rust";
        control_formula: string;
        channels: number;
        raw_mean_coherence: number;
        raw_coherence_entropy: number;
        spectral_radius: number;
        spectral_entropy: number;
        spectral_pressure: number;
        effective_channels: number;
        energy_ratio: number;
    };

    export type ZSpaceCoherenceProjection = {
        kind: "spiraltorch.zspace_coherence_projection";
        contract_version: "spiraltorch.zspace_coherence_projection.v1";
        semantic_owner: "st-core::inference::zspace_coherence";
        semantic_backend: "rust";
        execution_client: "wasm";
        projection_formula: string;
        summary_formula: string;
        diagnostics: ZSpaceCoherenceDiagnosticsInput;
        coherence: number[];
        contour?: ZSpaceCoherenceContourInput;
        config: Required<ZSpaceCoherenceProjectionConfig>;
        derived: ZSpaceCoherenceProjectionDerived;
        classification?: ZSpaceCoherenceClassification;
        control?: ZSpaceCoherenceControl;
        partial: Record<string, number>;
    };

    export type ZSpacePosteriorDecodeRequest = {
        z_state: number[];
        alpha?: number;
    };

    export type ZSpacePosteriorDecode = {
        kind: "spiraltorch.zspace_posterior_decode";
        contract_version: "spiraltorch.zspace_posterior.v1";
        semantic_owner: "st-core::inference::zspace_posterior";
        semantic_backend: "rust";
        execution_client: "wasm";
        metric_formula: string;
        fractional_formula: string;
        gradient_formula: string;
        barycentric_formula: string;
        z_state: number[];
        alpha: number;
        metrics: Record<string, number>;
        gradient: number[];
        barycentric: [number, number, number];
        energy: number;
        frac_energy: number;
        spectral_bins: number;
    };

    export type ZSpacePosteriorProjectionRequest = {
        z_state: number[];
        alpha?: number;
        partial?: Record<string, number | number[]>;
        smoothing?: number;
        telemetry?: Array<Record<string, unknown>>;
    };

    export type ZSpacePosteriorTelemetryAdjustment = {
        variance_damping: number;
        focus_gain: number;
        energy_gain: number;
        residual_before: number;
        confidence_before: number;
    };

    export type ZSpacePosteriorProjection = {
        kind: "spiraltorch.zspace_posterior_projection";
        contract_version: "spiraltorch.zspace_posterior.v1";
        semantic_owner: "st-core::inference::zspace_posterior";
        semantic_backend: "rust";
        execution_client: "wasm";
        projection_formula: string;
        telemetry_formula: string;
        smoothing: number;
        metrics: Record<string, number>;
        gradient: number[];
        barycentric: [number, number, number];
        residual: number;
        confidence: number;
        applied: Record<string, number | number[]>;
        prior: ZSpacePosteriorDecode;
        telemetry?: ZSpaceTelemetryFusion;
        telemetry_adjustment?: ZSpacePosteriorTelemetryAdjustment;
    };

    export type FreeEnergyBandInput = {
        above: number;
        here: number;
        beneath: number;
    };

    export type FreeEnergyBandPriorInput = Partial<FreeEnergyBandInput>;

    export type FreeEnergyConfigInput = {
        loss_scale?: number;
        step_time_scale_ms?: number;
        memory_scale_mb?: number;
        retry_scale?: number;
        observation_entropy_scale?: number;
        loss_weight?: number;
        speed_weight?: number;
        memory_weight?: number;
        retry_weight?: number;
        uncertainty_weight?: number;
        external_penalty_weight?: number;
        temperature?: number;
        prior?: FreeEnergyBandPriorInput;
        band_potentials?: Partial<FreeEnergyBandInput>;
    };

    export type FreeEnergyObservationInput = {
        reference_loss?: number;
        candidate_loss?: number;
        step_time_ms?: number;
        memory_mb?: number;
        retry_rate?: number;
        observation_entropy?: number;
        external_penalty?: number;
        band?: FreeEnergyBandInput;
    };

    export type FreeEnergyRequest = {
        observation?: FreeEnergyObservationInput;
        config?: FreeEnergyConfigInput;
    };

    export type FreeEnergyReport = {
        kind: "spiraltorch.variational_free_energy";
        contract_version: "spiraltorch.variational_free_energy.v1";
        semantic_owner: "st-core::heur::free_energy";
        semantic_backend: "rust";
        execution_client: "wasm";
        formula: "F(q)=E_observed+(E_q[V]-E_prior[V])+temperature*KL(q||prior)";
        acceptance_rule: "P(accept)=1/(1+exp(F_candidate-F_neutral)),F_neutral=0";
        config: Required<Omit<FreeEnergyConfigInput, "prior" | "band_potentials">> & {
            prior: FreeEnergyBandInput;
            band_potentials: FreeEnergyBandInput;
        };
        observation: Required<Omit<FreeEnergyObservationInput, "band">> & {
            band: FreeEnergyBandInput;
        };
        normalized: {
            loss_delta: number;
            step_time: number;
            memory: number;
            retry: number;
            observation_entropy: number;
            external_penalty: number;
        };
        distribution: {
            status: "normalized" | "prior_zero_mass";
            raw_total: number;
            zero_mass_threshold: number;
            above: number;
            here: number;
            beneath: number;
            prior_above: number;
            prior_here: number;
            prior_beneath: number;
            entropy: number;
            normalized_entropy: number;
            cross_entropy: number;
            kl_divergence: number;
            variational_identity_residual: number;
            dominant_band: "above" | "here" | "beneath";
        };
        components: {
            loss: number;
            speed: number;
            memory: number;
            retry: number;
            uncertainty: number;
            external_penalty: number;
            observed_energy: number;
            band_potential_expectation: number;
            prior_band_potential: number;
            band_potential: number;
            relative_entropy: number;
        };
        free_energy: number;
        utility: number;
        acceptance_probability: number;
        component_sum_residual: number;
    };

    export type ZSpaceMetaOptimizerGradientProjection =
        | "tile_or_truncate"
        | "exact";

    export type ZSpaceMetaOptimizerWeights = {
        speed?: number;
        memory?: number;
        stability?: number;
        fractional?: number;
        drift_response?: number;
    };

    export type ZSpaceMetaOptimizerConfigInput = {
        dimension?: number;
        fractional_order?: number;
        weights?: ZSpaceMetaOptimizerWeights;
        learning_rate?: number;
        first_moment_decay?: number;
        second_moment_decay?: number;
        epsilon?: number;
        topos_control_gain?: number;
        gradient_projection?: ZSpaceMetaOptimizerGradientProjection;
    };

    export type ZSpaceMetaOptimizerConfig = Required<
        Omit<ZSpaceMetaOptimizerConfigInput, "weights">
    > & {
        weights: Required<ZSpaceMetaOptimizerWeights>;
    };

    export type ZSpaceMetaOptimizerState = {
        z: number[];
        first_moment: number[];
        second_moment: number[];
        step: number;
    };

    export type ZSpaceMetaOptimizerObservation = {
        speed?: number;
        memory?: number;
        stability?: number;
        drift_response?: number;
        gradient?: number[];
        telemetry?: Record<string, number>;
    };

    export type ZSpaceMetaOptimizerRestoreRequest = {
        config: ZSpaceMetaOptimizerConfigInput;
        state: ZSpaceMetaOptimizerState;
        strict?: boolean;
    };

    export type ZSpaceMetaOptimizerStepRequest = {
        config: ZSpaceMetaOptimizerConfigInput;
        state: ZSpaceMetaOptimizerState;
        observation: ZSpaceMetaOptimizerObservation;
    };

    export type ZSpaceMetaOptimizerCheckpoint = {
        contract_version: "spiraltorch.zspace_meta_optimizer.v1";
        kind: "spiraltorch.zspace_meta_optimizer";
        semantic_owner: "st-core::runtime::zspace_optimizer";
        semantic_backend: "rust";
        execution_client: "wasm";
        config: ZSpaceMetaOptimizerConfig;
        state: ZSpaceMetaOptimizerState;
    };

    export type ZSpaceMetaOptimizerStepReport = {
        contract_version: "spiraltorch.zspace_meta_optimizer.v1";
        kind: "spiraltorch.zspace_meta_optimizer";
        semantic_owner: "st-core::runtime::zspace_optimizer";
        semantic_backend: "rust";
        execution_client: "wasm";
        objective_formula: "J_obs=sum_i(lambda_i*tanh(metric_i))+lambda_topos*tanh(topos_pressure)+lambda_frac_eff*R_alpha(z)";
        transition_validated: true;
        config: ZSpaceMetaOptimizerConfig;
        observation: Required<ZSpaceMetaOptimizerObservation>;
        objective: {
            normalized_speed: number;
            normalized_memory: number;
            normalized_stability: number;
            normalized_drift_response: number;
            speed_term: number;
            memory_term: number;
            stability_term: number;
            drift_response_term: number;
            topos_term: number;
            fractional_term: number;
            observed_resource_penalty: number;
            objective_before: number;
        };
        fractional_regularizer: {
            formula: string;
            order: number;
            signal_length: number;
            spectral_bins: number;
            energy: number;
            raw_gradient: number[];
            gradient_normalization_scale: number;
            normalized_gradient: number[];
        };
        topos_control: {
            present: boolean;
            active: boolean;
            closure_pressure: number;
            depth_pressure: number;
            volume_pressure: number;
            guard_strength: number;
            step_damping: number;
            sampling_focus: number;
            openness: number;
            exploration_hint: number;
            pressure: number;
            learning_rate_hint: number;
            learning_rate_scale: number;
            effective_learning_rate: number;
            clip_hint: number;
            clip_scale: number;
            gradient_clip_threshold: number | null;
            regularization_hint: number;
            regularization_scale: number;
            effective_fractional_weight: number;
            gradient_bias_scale: number;
            gradient_bias: number[];
        };
        gradient: {
            rule: string;
            source_dimension: number;
            target_dimension: number;
            projection: ZSpaceMetaOptimizerGradientProjection;
            observed: number[];
            projected_normalized: number[];
            before_clip: number[];
            applied: number[];
            clipped_values: number;
        };
        adam: {
            rule: string;
            step: number;
            first_moment_bias_correction: number;
            second_moment_bias_correction: number;
            effective_learning_rate: number;
            parameter_delta: number[];
        };
        state_before: ZSpaceMetaOptimizerState;
        state_after: ZSpaceMetaOptimizerState;
    };

    export type ZSpaceParameterControl = {
        contract_version: "spiraltorch.zspace_parameter_control.v1";
        kind: "spiraltorch.zspace_parameter_control";
        semantic_owner: "st-core::runtime::zspace_optimizer";
        semantic_backend: "rust";
        source_contract_version: "spiraltorch.zspace_meta_optimizer.v1";
        source_semantic_owner: "st-core::runtime::zspace_optimizer";
        source_step: number;
        absolute_learning_rate_scale: number;
        source_learning_rate: number;
        source_effective_learning_rate: number;
        execution_client: "wasm";
    };

    export type WasmReportRuntimeAudit = {
        status: "webgpu_ready" | "webgpu_available" | "wasm_only" | "missing_runtime";
        score: number;
        wasm: boolean;
        webgpu_available: boolean;
        webgpu_device_ready: boolean;
        webgpu_component_ready_count: number;
        webgpu_init_failed: boolean;
    };

    export type WasmReportLearningAudit = {
        source: "trace" | "loss_stats" | "missing";
        loss: number | null;
        first_loss: number | null;
        last_loss: number | null;
        absolute_improvement: number | null;
        relative_improvement: number | null;
        improved: boolean | null;
        progress_score: number;
    };

    export type WasmReportAudit = {
        kind: "spiraltorch.wasm_report_audit";
        schema: string;
        report_kind: string;
        family: "mellin" | "canvas" | "unknown";
        artifact_path: string | null;
        status: WasmReportAuditStatus;
        readiness_score: number;
        runtime: WasmReportRuntimeAudit;
        learning: WasmReportLearningAudit;
        loss_score: number;
        stability_score: number;
        risk_flags: string[];
        recommendations: string[];
    };

    export type WasmReportComparisonRow = {
        label: string;
        schema: string;
        kind: string;
        family: "mellin" | "canvas" | "unknown";
        loss: number | null;
        stability: number | null;
        readiness_score: number | null;
        audit_status: WasmReportAuditStatus;
        risk_flags: string[];
        audit: WasmReportAudit;
    };

    export type WasmReportComparison = {
        kind: "spiraltorch.wasm_report_comparison";
        count: number;
        families: Record<string, number>;
        best_loss: WasmReportComparisonRow | null;
        best_stability: WasmReportComparisonRow | null;
        best_readiness: WasmReportComparisonRow | null;
        reports: WasmReportComparisonRow[];
    };

    export type ScaleStackProbeMode = "scalar" | "semantic::euclidean" | "semantic::cosine";

    export type ScaleStackProbeSample = {
        scale: number;
        gate_mean: number;
    };

    export type ScaleStackPersistenceBin = {
        scale_low: number;
        scale_high: number;
        mass: number;
    };

    export type ScaleStackCoherenceBreak = {
        level: number;
        scale: number | null;
    };

    export type ScaleStackProbe = {
        kind: "spiraltorch.wasm_scale_stack_probe";
        source_crate: "st-frac::scale_stack";
        mode: ScaleStackProbeMode;
        threshold: number;
        sample_count: number;
        samples: ScaleStackProbeSample[];
        persistence: ScaleStackPersistenceBin[];
        interface_density: number | null;
        moment_0: number;
        moment_1: number;
        moment_2: number;
        boundary_dimension: number | null;
        coherence_profile: ScaleStackCoherenceBreak[];
    };

    export type FractalFieldGeneratorConfig = {
        octaves: number;
        lacunarity: number;
        gain: number;
        iterations: number;
    };

    export type FractalFieldLogLattice = {
        log_start: number;
        log_step: number;
        len: number;
        support: [number, number];
    };

    export type FractalFieldProbeSample = {
        index: number;
        log: number;
        re: number;
        im: number;
        abs: number;
        phase: number;
    };

    export type FractalFieldProbe = {
        kind: "spiraltorch.wasm_fractal_field_probe";
        source_crate: "st-frac::fractal_field";
        mode: "branching_field";
        generator: FractalFieldGeneratorConfig;
        log_lattice: FractalFieldLogLattice;
        sample_count: number;
        preview_count: number;
        energy: number;
        mean_abs: number;
        max_abs: number;
        mean_real: number;
        mean_imag: number;
        phase_drift: number;
        total_variation: number;
        coherence_score: number;
        samples: FractalFieldProbeSample[];
    };

    export type LogZSeriesWindow = "rectangular" | "hann";
    export type LogZSeriesNormalisation = "none" | "l1" | "l2";

    export type LogZSeriesLattice = {
        log_start: number;
        log_step: number;
        len: number;
        support: [number, number];
    };

    export type LogZSeriesScalarStats = {
        count: number;
        mean: number;
        min: number;
        max: number;
        energy: number;
    };

    export type LogZSeriesProjectionPreview = {
        index: number;
        re: number;
        im: number;
        abs: number;
        phase: number;
    };

    export type LogZSeriesProjectionStats = {
        count: number;
        mean_abs: number;
        max_abs: number;
        energy: number;
        phase_drift: number;
        stability_score: number;
        preview_count: number;
        preview: LogZSeriesProjectionPreview[];
    };

    export type LogZSeriesProbe = {
        kind: "spiraltorch.wasm_log_z_series_probe";
        source_crate: "st-frac::cosmology";
        mode: "log_z_series";
        log_lattice: LogZSeriesLattice;
        options: {
            window: LogZSeriesWindow;
            normalisation: LogZSeriesNormalisation;
        };
        sample_count: number;
        sample_stats: LogZSeriesScalarStats;
        weight_stats: LogZSeriesScalarStats;
        z_count: number;
        projection: LogZSeriesProjectionStats;
    };

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

    export function auditWasmReportJson(reportJson: string): string;
    export function auditWasmReportObject(report: unknown): WasmReportAudit;
    export function compareWasmReportsJson(reportsJson: string): string;
    export function compareWasmReportsObject(reports: unknown): WasmReportComparison;
    export function toposRuntimeRouteJson(profileJson: string): string;
    export function toposRuntimeRouteObject(
        profile: ToposRuntimeProfileInput,
    ): ToposRuntimeRoute;
    export function runtimeDeviceRouteJson(requestJson: string): string;
    export function runtimeDeviceRouteObject(
        request: RuntimeDeviceRouteRequest,
    ): RuntimeDeviceRoute;
    export function toposRoutePolicyEvaluateJson(requestJson: string): string;
    export function toposRoutePolicyEvaluateObject(
        request: ToposRoutePolicyEvaluationRequest,
    ): ToposRoutePolicyEvaluation;
    export function toposRoutePolicyRewardsJson(requestJson: string): string;
    export function toposRoutePolicyRewardsObject(
        request: ToposRouteRewardsRequest,
    ): ToposRouteRewards;
    export function toposRoutePolicyResolveJson(requestJson: string): string;
    export function toposRoutePolicyResolveObject(
        request: ToposRoutePolicyResolveRequest,
    ): ToposRoutePolicyResolution;
    export function toposControlSignalJson(inputJson: string): string;
    export function toposControlSignalObject(
        input: ToposControlSignalInput,
    ): ToposControlSignal;
    export function toposOptimizerSnapshotJson(inputJson: string): string;
    export function toposOptimizerSnapshotObject(
        input: ToposOptimizerSnapshotRequest,
    ): ToposOptimizerSnapshot;
    export function toposZSpaceProjectionJson(
        inputJson: string,
        gradientDim: number,
    ): string;
    export function toposZSpaceProjectionObject(
        input: ToposControlSignalInput,
        gradientDim: number,
    ): ToposZSpaceProjection;
    export function zspaceTelemetryFusionJson(payloadsJson: string): string;
    export function zspaceTelemetryFusionObject(
        payloads: Array<Record<string, unknown>>,
    ): ZSpaceTelemetryFusion;
    export function zspacePartialFusionJson(requestJson: string): string;
    export function zspacePartialFusionObject(
        request: ZSpacePartialFusionRequest,
    ): ZSpacePartialFusion;
    export function zspaceCoherenceProjectJson(requestJson: string): string;
    export function zspaceCoherenceProjectObject(
        request: ZSpaceCoherenceProjectionRequest,
    ): ZSpaceCoherenceProjection;
    export function zspacePosteriorDecodeJson(requestJson: string): string;
    export function zspacePosteriorDecodeObject(
        request: ZSpacePosteriorDecodeRequest,
    ): ZSpacePosteriorDecode;
    export function zspacePosteriorProjectJson(requestJson: string): string;
    export function zspacePosteriorProjectObject(
        request: ZSpacePosteriorProjectionRequest,
    ): ZSpacePosteriorProjection;
    export function zspaceFreeEnergyJson(requestJson: string): string;
    export function zspaceFreeEnergyObject(
        request: FreeEnergyRequest,
    ): FreeEnergyReport;
    export function zspaceMetaOptimizerInitJson(configJson: string): string;
    export function zspaceMetaOptimizerInitObject(
        config: ZSpaceMetaOptimizerConfigInput,
    ): ZSpaceMetaOptimizerCheckpoint;
    export function zspaceMetaOptimizerRestoreJson(requestJson: string): string;
    export function zspaceMetaOptimizerRestoreObject(
        request: ZSpaceMetaOptimizerRestoreRequest,
    ): ZSpaceMetaOptimizerCheckpoint;
    export function zspaceMetaOptimizerStepJson(requestJson: string): string;
    export function zspaceMetaOptimizerStepObject(
        request: ZSpaceMetaOptimizerStepRequest,
    ): ZSpaceMetaOptimizerStepReport;
    export function zspaceMetaOptimizerParameterControlJson(
        reportJson: string,
    ): string;
    export function zspaceMetaOptimizerParameterControlObject(
        report: ZSpaceMetaOptimizerStepReport,
    ): ZSpaceParameterControl;

    export function scalarScaleStackProbeJson(
        field: Float32Array,
        shape: Uint32Array,
        scales: Float32Array,
        threshold: number,
        ambientDim: number,
        dimensionWindow: number,
        levels: Float32Array,
    ): string;

    export function scalarScaleStackProbeObject(
        field: Float32Array,
        shape: Uint32Array,
        scales: Float32Array,
        threshold: number,
        ambientDim: number,
        dimensionWindow: number,
        levels: Float32Array,
    ): ScaleStackProbe;

    export function semanticScaleStackProbeJson(
        embeddings: Float32Array,
        rows: number,
        dims: number,
        scales: Float32Array,
        threshold: number,
        metric: "euclidean" | "cosine",
        ambientDim: number,
        dimensionWindow: number,
        levels: Float32Array,
    ): string;

    export function semanticScaleStackProbeObject(
        embeddings: Float32Array,
        rows: number,
        dims: number,
        scales: Float32Array,
        threshold: number,
        metric: "euclidean" | "cosine",
        ambientDim: number,
        dimensionWindow: number,
        levels: Float32Array,
    ): ScaleStackProbe;

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

    export type LogLatticeWindow =
        | "rect"
        | "rectangular"
        | "none"
        | "hann"
        | "tukey"
        | "blackman";

    export type ComplexArrayPair = [Float32Array, Float32Array];

    export class WasmMellinEvalPlan {
        private constructor();
        static many(
            log_start: number,
            log_step: number,
            sValues: Float32Array,
        ): WasmMellinEvalPlan;
        static verticalLine(
            log_start: number,
            log_step: number,
            real: number,
            imagValues: Float32Array,
        ): WasmMellinEvalPlan;
        static mesh(
            log_start: number,
            log_step: number,
            realValues: Float32Array,
            imagValues: Float32Array,
        ): WasmMellinEvalPlan;
        readonly logStart: number;
        readonly logStep: number;
        len(): number;
        shape(): Uint32Array;
    }

    export class WasmMellinLogGrid {
        constructor(log_start: number, log_step: number, samples: Float32Array);
        static newWithWindow(
            log_start: number,
            log_step: number,
            samples: Float32Array,
            window: LogLatticeWindow,
            preserve_sum: boolean,
        ): WasmMellinLogGrid;
        static expDecay(
            log_start: number,
            log_step: number,
            len: number,
            window: LogLatticeWindow,
            preserve_sum: boolean,
        ): WasmMellinLogGrid;
        static expDecayScaled(
            log_start: number,
            log_step: number,
            len: number,
            rate: number,
            window: LogLatticeWindow,
            preserve_sum: boolean,
        ): WasmMellinLogGrid;
        readonly logStart: number;
        readonly logStep: number;
        len(): number;
        isEmpty(): boolean;
        samples(): Float32Array;
        weights(): Float32Array;
        support(): Float32Array;
        weightedSeries(): Float32Array;
        planMany(sValues: Float32Array): WasmMellinEvalPlan;
        planVerticalLine(real: number, imagValues: Float32Array): WasmMellinEvalPlan;
        planMesh(realValues: Float32Array, imagValues: Float32Array): WasmMellinEvalPlan;
        evaluatePlan(plan: WasmMellinEvalPlan): Float32Array;
        evaluatePlanStable(plan: WasmMellinEvalPlan): Float32Array;
        evaluatePlanWithDerivative(plan: WasmMellinEvalPlan): ComplexArrayPair;
        evaluatePlanWithDerivativeStable(plan: WasmMellinEvalPlan): ComplexArrayPair;
        evaluatePlanMagnitude(plan: WasmMellinEvalPlan): Float32Array;
        evaluatePlanLogMagnitude(plan: WasmMellinEvalPlan, epsilon: number): Float32Array;
        trainStepMatchGridPlan(
            plan: WasmMellinEvalPlan,
            target: WasmMellinLogGrid,
            lr: number,
        ): number;
        evaluate(s: Float32Array): Float32Array;
        evaluateStable(s: Float32Array): Float32Array;
        evaluateWithDerivative(s: Float32Array): ComplexArrayPair;
        evaluateWithDerivativeStable(s: Float32Array): ComplexArrayPair;
        evaluateMany(sValues: Float32Array): Float32Array;
        evaluateManyStable(sValues: Float32Array): Float32Array;
        evaluateManyWithDerivative(sValues: Float32Array): ComplexArrayPair;
        evaluateManyWithDerivativeStable(sValues: Float32Array): ComplexArrayPair;
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

    export class WasmFractalFieldGenerator {
        constructor(octaves: number, lacunarity: number, gain: number, iterations: number);
        readonly octaves: number;
        readonly lacunarity: number;
        readonly gain: number;
        readonly iterations: number;
        branchingField(logStart: number, logStep: number, len: number): Float32Array;
        probeObject(
            logStart: number,
            logStep: number,
            len: number,
            previewLen: number,
        ): FractalFieldProbe;
        probeJson(logStart: number, logStep: number, len: number, previewLen: number): string;
    }

    export function fractalFieldProbeObject(
        octaves: number,
        lacunarity: number,
        gain: number,
        iterations: number,
        logStart: number,
        logStep: number,
        len: number,
        previewLen: number,
    ): FractalFieldProbe;

    export function fractalFieldProbeJson(
        octaves: number,
        lacunarity: number,
        gain: number,
        iterations: number,
        logStart: number,
        logStep: number,
        len: number,
        previewLen: number,
    ): string;

    export class WasmLogZSeries {
        constructor(
            logStart: number,
            logStep: number,
            samples: Float32Array,
            window: LogZSeriesWindow,
            normalisation: LogZSeriesNormalisation,
        );
        len(): number;
        isEmpty(): boolean;
        readonly logStart: number;
        readonly logStep: number;
        readonly window: LogZSeriesWindow;
        readonly normalisation: LogZSeriesNormalisation;
        samples(): Float32Array;
        weights(): Float32Array;
        evaluateZ(z: Float32Array): Float32Array;
        evaluateManyZ(zValues: Float32Array): Float32Array;
        probeObject(zValues: Float32Array, previewLen: number): LogZSeriesProbe;
        probeJson(zValues: Float32Array, previewLen: number): string;
    }

    export function logZSeriesProbeObject(
        logStart: number,
        logStep: number,
        samples: Float32Array,
        window: LogZSeriesWindow,
        normalisation: LogZSeriesNormalisation,
        zValues: Float32Array,
        previewLen: number,
    ): LogZSeriesProbe;

    export function logZSeriesProbeJson(
        logStart: number,
        logStep: number,
        samples: Float32Array,
        window: LogZSeriesWindow,
        normalisation: LogZSeriesNormalisation,
        zValues: Float32Array,
        previewLen: number,
    ): string;

    export class WasmScaleStack {
        private constructor();
        static scalar(
            field: Float32Array,
            shape: Uint32Array,
            scales: Float32Array,
            threshold: number,
        ): WasmScaleStack;
        static semantic(
            embeddings: Float32Array,
            rows: number,
            dims: number,
            scales: Float32Array,
            threshold: number,
            metric: "euclidean" | "cosine",
        ): WasmScaleStack;
        readonly threshold: number;
        readonly mode: ScaleStackProbeMode;
        readonly sampleCount: number;
        samples(): Float32Array;
        persistence(): Float32Array;
        interfaceDensity(): number | undefined;
        moment(order: number): number;
        boundaryDimension(ambientDim: number, window: number): number | undefined;
        coherenceBreakScale(level: number): number | undefined;
        coherenceProfile(levels: Float32Array): Float32Array;
        toObject(ambientDim: number, dimensionWindow: number, levels: Float32Array): ScaleStackProbe;
        toJson(ambientDim: number, dimensionWindow: number, levels: Float32Array): string;
    }

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
