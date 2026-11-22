import {CausalLmDeclaredBatch, GeometryAdapterArtifact, NativeCausalLm, NativeCausalLmContext, NativeCausalLmLineTrainingSession} from 'spiraltorch-wasm';

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
