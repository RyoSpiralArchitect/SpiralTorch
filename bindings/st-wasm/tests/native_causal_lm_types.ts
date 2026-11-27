import {CausalLmDeclaredBatch, GeometryAdapterArtifact, NativeCausalLm, NativeCausalLmContext, NativeCausalLmLineTrainingSession} from 'spiraltorch-wasm';

export function usesPortableHostMath(model: NativeCausalLm): boolean {
    return model.hostAttentionExp === 'portable_libm_exp_f64.v1'
        && model.hostGeluTanh === 'portable_libm_tanh_f32.v1';
}

export function chunkedPrefill(model: NativeCausalLm, ids: Uint32Array) {
    const plan: string = model.planChunkedPrefillJson(ids.length, 1024, 2);
    const baseline: NativeCausalLmContext = model.prefillChunked(ids, 1024);
    try {
        const optimized: NativeCausalLmContext = model.prefillChunkedLastOutput(ids, 1024, 2);
        try {
            return {plan, baseline: baseline.nextLogits(), optimized: optimized.nextLogits()};
        } finally { optimized.free(); }
    } finally { baseline.free(); }
}

export function jointLine(geometry: NativeCausalLm, prompt: Uint32Array, flat: Uint32Array, lengths: Uint32Array) {
    const m = geometry.withLora(0, 2, 2);
    try {
        const g = m.lineAdaptationGradients(prompt, flat, lengths, 0, '{"kind":"set"}', 'joint');
        try {
            const promptId: string | undefined = g.linePromptId;
            void promptId;
            const step = m.stepLineGradientAtDistance(g, 1/256);
            const report: string = step.reportJson();
            void report;
            const n = step.intoModel();
            try { return n.withLoraJson(n.loraJson(), n.loraArtifactId!); }
            finally { n.free(); }
        } finally { g.free(); }
    } finally { m.free(); }
}

export function tryScheduledStep(s: NativeCausalLmLineTrainingSession): string | undefined {
    return s.attemptStepJson();
}

export function frozenPlacement(m: NativeCausalLm, path: string) {
    const base = m.freezeLora();
    try {
        const id: string | undefined = base.frozenLoraArtifactId;
        const weights: string = base.frozenLoraJson();
        return {id, weights, model: base.withSingleGeometryAt(path)};
    } finally { base.free(); }
}

export function resumeLine(origin: NativeCausalLm, plan: string, checkpoint: string, pin: string) {
    const s = NativeCausalLmLineTrainingSession.fromJson(origin,plan,checkpoint,pin);
    try {
        const step: string | undefined = s.stepJson();
        const next = s.checkpoint();
        try { return {step,payload:next.payload,id:next.id,updates:s.completedUpdates}; }
        finally { next.free(); }
    } finally { s.free(); }
}

export function update(config: string, weights: Uint8Array, geometry: string, id: string, wire: string) {
    const base = NativeCausalLm.fromGpt2(config, weights);
    const art = GeometryAdapterArtifact.fromJson(geometry, id, base.baseModelIdentity);
    const model = base.withGeometry(art);
    const batch = CausalLmDeclaredBatch.fromJson(wire);
    try {
        const g = model.geometryLineGradients(Uint32Array.of(2),Uint32Array.of(1,3),Uint32Array.of(1,1),0,'{"kind":"set"}', 256);
        try {
            const work: string = g.trainingWorkJson();
            const named: string = g.parameterGradientsJson();
            const loss: string = g.reportJson();
            void [work,named,loss];
            model.stepLineGeometrySgd(g, 0.01).free();
        } finally { g.free(); }
        const step = model.stepDeclaredSetGeometryAtDistance(batch, 1/65536);
        const receipt: string = step.reportJson();
        const result = step.intoModel();
        try {
            const ctx: NativeCausalLmContext = result.prefill(Uint32Array.of(2), 1024);
            let logits: Float32Array;
            let scores: string;
            try {
                scores = ctx.scoreLinesJson(Uint32Array.of(1, 1, 2), Uint32Array.of(1, 2), 0, 256);
                const child = ctx.withSuffix(Uint32Array.of(1, 0));
                try { logits = child.nextLogits(); } finally { child.free(); }
            } finally { ctx.free(); }
            const saved: GeometryAdapterArtifact | undefined = result.geometryArtifact();
            try { return {receipt, logits, scores, artifact: saved?.toJson()}; }
            finally { saved?.free(); }
        } finally { result.free(); }
    } finally { batch.free(); model.free(); art.free(); base.free(); }
}
