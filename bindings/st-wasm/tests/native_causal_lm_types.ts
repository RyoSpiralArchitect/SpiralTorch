import {CausalLmDeclaredBatch, GeometryAdapterArtifact, NativeCausalLm} from 'spiraltorch-wasm';

export function update(config: string, weights: Uint8Array, geometry: string, id: string, wire: string) {
    const base = NativeCausalLm.fromGpt2(config, weights);
    const art = GeometryAdapterArtifact.fromJson(geometry, id, base.baseModelIdentity);
    const model = base.withGeometry(art);
    const batch = CausalLmDeclaredBatch.fromJson(wire);
    try {
        const step = model.stepDeclaredSetGeometryAtDistance(batch, 1/65536);
        const receipt: string = step.reportJson();
        const result = step.intoModel();
        try {
            const logits: Float32Array = result.nextLogits('[2]');
            const saved: GeometryAdapterArtifact | undefined = result.geometryArtifact();
            try { return {receipt, logits, artifact: saved?.toJson()}; }
            finally { saved?.free(); }
        } finally { result.free(); }
    } finally { batch.free(); model.free(); art.free(); base.free(); }
}
