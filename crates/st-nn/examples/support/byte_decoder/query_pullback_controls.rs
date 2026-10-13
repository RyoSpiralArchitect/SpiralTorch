//! Paired same-runtime control. Never changes the earlier mixed-runtime gate.
use super::*;
use st_nn::resident::ByteDecoderQueryPullback;

pub(super) async fn run(runtime: WgpuRuntime, fixture: &str) -> Result<Value> {
    let cases = projective_controls::cases(fixture)?;
    let mut checks = Vec::new();
    for case in &cases[4..] {
        let plan = plan(case, 4)?;
        let mut model = plan.compile_training_wgpu(runtime.clone())?;
        let windows: Vec<Vec<u8>> = case["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| {
                sizes(w)
                    .into_iter()
                    .map(|v| u8::try_from(v).unwrap())
                    .collect()
            })
            .collect();
        let resident = model.prepare_batch(&batch(&windows, 4)?)?;
        let external = biases(model.tensor_device(), case, 4)?;
        let before = read_many(model.tensor_device(), model.parameter_snapshot().values()).await?;
        let forward = model.forward_with_external_biases(&resident, &borrowed(&external))?;
        let output = read(forward.prediction()).await?;
        close(&output, &data(&case["output"]), "paired logits")?;
        let seed = model
            .tensor_device()
            .upload(model.output_layout().shape(), &data(&case["cotangent"]))?;
        let full = model.backward(&forward, &seed)?;
        // Freeze each observation before another backward can touch any buffers.
        let full_values = read(full.embedding_output_gradient()).await?;
        let full_parameters = read_many(model.tensor_device(), full.parameter_gradients()).await?;
        let stopped = model.backward_with_query_pullback(
            &forward,
            &seed,
            ByteDecoderQueryPullback::StopCoordinates,
        )?;
        let stopped_values = read(stopped.embedding_output_gradient()).await?;
        let stopped_parameters =
            read_many(model.tensor_device(), stopped.parameter_gradients()).await?;
        let full_retained = read_many(model.tensor_device(), full.parameter_gradients()).await?;
        if !same_bits(&full_values, &read(full.embedding_output_gradient()).await?)
            || !same_family(&full_parameters, &full_retained)
        {
            return Err("stopped pullback overwrote a retained full VJP".into());
        }
        let repeated = model.backward(&forward, &seed)?;
        if full.query_pullback() != ByteDecoderQueryPullback::Full
            || stopped.query_pullback() != ByteDecoderQueryPullback::StopCoordinates
        {
            return Err("query pullback provenance lost".into());
        }
        let stopped_retained =
            read_many(model.tensor_device(), stopped.parameter_gradients()).await?;
        if !same_bits(
            &full_values,
            &read(repeated.embedding_output_gradient()).await?,
        ) || !same_bits(&output, &read(forward.prediction()).await?)
            || !same_bits(
                &stopped_values,
                &read(stopped.embedding_output_gradient()).await?,
            )
            || !same_family(&stopped_parameters, &stopped_retained)
        {
            return Err("query control changed a retained forward or full backward".into());
        }
        close(
            &full_values,
            &data(&case["embedding_output_gradient"]),
            "paired full embedding VJP",
        )?;
        let reference_stopped = data(&case["query_controls"]["router_detached_embedding_gradient"]);
        close(
            &stopped_values,
            &reference_stopped,
            "paired stopped embedding VJP",
        )?;
        let repeated_parameters =
            read_many(model.tensor_device(), repeated.parameter_gradients()).await?;
        let layout = plan
            .parameter_layout()
            .geometry()
            .ok_or("missing query geometry")?;
        let changed_geometry = layout.projection().start..layout.raw_gains().start;
        let mut unchanged = Vec::new();
        if full_parameters.len() != plan.parameter_layout().len()
            || stopped_parameters.len() != full_parameters.len()
            || repeated_parameters.len() != full_parameters.len()
        {
            return Err("truncated query pullback parameter family".into());
        }
        for (slot, ((a, b), c)) in full_parameters
            .iter()
            .zip(&stopped_parameters)
            .zip(&repeated_parameters)
            .enumerate()
        {
            if !same_bits(a, c) {
                return Err("stopped pullback poisoned a later full pullback".into());
            }
            if slot >= 2 && !changed_geometry.contains(&slot) {
                if !same_bits(a, b) {
                    return Err("query stop-gradient changed an unrelated parameter VJP".into());
                }
                unchanged.push(slot);
            }
        }
        for range in layout.query_projections() {
            for slot in range.clone() {
                geometry_controls::relative_gradient(
                    &full_parameters[slot],
                    &data(&case["parameter_gradients"][slot]),
                )?;
            }
        }
        let now = read_many(model.tensor_device(), model.parameter_snapshot().values()).await?;
        if before.len() != now.len() || before.iter().zip(&now).any(|(a, b)| !same_bits(a, b)) {
            return Err("paired query observation changed model parameters".into());
        }
        let difference = |a: &[f32], b: &[f32]| -> Result<Vec<f32>> {
            if a.len() != b.len() || a.is_empty() {
                return Err("invalid paired contrast shape".into());
            }
            let values: Vec<_> = a.iter().zip(b).map(|(a, b)| a - b).collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite paired contrast".into());
            }
            Ok(values)
        };
        let reference = difference(
            &data(&case["embedding_output_gradient"]),
            &reference_stopped,
        )?;
        let delta = difference(&full_values, &stopped_values)?;
        let paired = geometry_controls::relative_gradient(&delta, &reference);
        let legacy = geometry_controls::relative_gradient(
            &difference(&full_values, &reference_stopped)?,
            &reference,
        );
        for bad in [
            vec![0.; reference.len()],
            reference.iter().map(|v| -v).collect(),
        ] {
            if geometry_controls::relative_gradient(&bad, &reference).is_ok() {
                return Err("paired query negative control was accepted".into());
            }
        }
        checks.push(json!({"name":case["name"], "passed":paired.is_ok(),
            "paired_relative_l2":paired.as_ref().ok(), "paired_error":paired.err().map(|e|e.to_string()),
            "legacy_mixed_runtime_passed":legacy.is_ok(), "legacy_mixed_runtime_error":legacy.err().map(|e|e.to_string()),
            "unchanged_parameter_slots":unchanged,"negative_controls_rejected":2,
            "forward_unchanged":true,"parameters_unchanged":true,"full_repeated_exact":true,
            "full_embedding_gradient":full_values,"stopped_embedding_gradient":stopped_values,
            "full_parameter_gradients":full_parameters,"stopped_parameter_gradients":stopped_parameters}));
    }
    Ok(
        json!({"schema":"spiraltorch.resident_byte_query_pullback.validation.v1",
        "passed":checks.iter().all(|c|c["passed"] == true),"checks":checks,
        "adapter":format!("{:?}",runtime.adapter_info()),
        "scope":"paired query-input stop-gradient correctness, not the old six-case qualification, corpus quality or speed"}),
    )
}

fn same_family(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_bits(a, b))
}
