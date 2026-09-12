//! Diagnostic pass attribution, never a replacement for end-to-end timing.
use super::*;
use st_backend_wgpu::resident_training::{
    graph::{GraphGpuProfile, GraphProfileReadback, ProfiledGraphTraining},
    TrainingError,
};

async fn profile_read(value: GraphProfileReadback) -> Result<GraphGpuProfile> {
    #[cfg(not(target_arch = "wasm32"))]
    let result = value.read();
    #[cfg(target_arch = "wasm32")]
    let result = value.read_async().await;
    Ok(result?)
}

fn compare(a: &Value, b: &Value) -> Result<f64> {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().ok_or("number")?, b.as_f64().ok_or("number")?);
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-5 + 2e-4 * b.abs() {
                return Err(format!("profile changed state: {a} != {b}").into());
            }
            Ok((a - b).abs())
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => a
            .iter()
            .zip(b)
            .try_fold(0f64, |e, (a, b)| Ok(e.max(compare(a, b)?))),
        (Value::Object(a), Value::Object(b)) if a.len() == b.len() => {
            a.iter().try_fold(0f64, |e, (key, a)| {
                Ok(e.max(compare(a, b.get(key).ok_or("key mismatch")?)?))
            })
        }
        _ => Err("incompatible state".into()),
    }
}

fn pending<T>(value: std::result::Result<T, TrainingError>) -> Result<()> {
    if !matches!(value, Err(TrainingError::PendingProfile)) {
        return Err("unread profile permitted workspace reuse".into());
    }
    Ok(())
}

fn check_pending(gpu: &mut ProfiledGraphTraining, x: &[f32], y: &[f32]) -> Result<()> {
    let before = (gpu.submitted_steps(), gpu.batch_generation());
    pending(gpu.step(0.))?;
    pending(gpu.step_profiled(0.))?;
    pending(gpu.upload_batch(x, y))?;
    pending(gpu.loss_snapshot())?;
    pending(gpu.state_snapshot())?;
    pending(gpu.parameter_snapshot())?;
    if before != (gpu.submitted_steps(), gpu.batch_generation()) {
        return Err("pending call mutated counters".into());
    }
    Ok(())
}

impl Benchmark {
    /// Same seeded workload as the matched benchmark. Three warm-up updates and
    /// nine retained updates; all results are checked, with no wall-clock fallback.
    pub async fn profile(&self, policy: GraphGradientPolicy) -> Result<Value> {
        if !self.config.graph || self.config.learner_optimizer.is_some() {
            return Err("profiling requires a graph without learner-only optimizer options".into());
        }
        let ordinary_features = self.runtime.context().device().features();
        if ordinary_features
            .iter_names()
            .any(|(name, _)| name == "TIMESTAMP_QUERY")
        {
            return Err("ordinary benchmark device unexpectedly enables timestamps".into());
        }
        let mut gpu = self
            .plan
            .profile_graph_training_wgpu(
                policy,
                Default::default(),
                MatmulKernel::Register2x2,
                MatmulAccumulation::Sequential,
            )
            .await?;
        let mut private_control = self
            .plan
            .profile_graph_training_wgpu(
                policy,
                Default::default(),
                MatmulKernel::Register2x2,
                MatmulAccumulation::Sequential,
            )
            .await?;
        let mut control = self.plan.compile_graph_training_wgpu_with_options(
            self.runtime.clone(),
            policy,
            Default::default(),
            MatmulKernel::Register2x2,
            MatmulAccumulation::Sequential,
        )?;
        for info in [
            gpu.adapter_info(),
            private_control.adapter_info(),
            control.adapter_info(),
        ] {
            if format!("{:?}", info.device_type) == "Cpu" {
                return Err("GPU required".into());
            }
        }
        if gpu.adapter_info() != private_control.adapter_info()
            || gpu.adapter_info() != control.adapter_info()
        {
            return Err("profile/control adapter metadata differ".into());
        }
        if !matches!(gpu.step_profiled(0.01), Err(TrainingError::MissingBatch))
            || gpu.submitted_steps() != 0
        {
            return Err("missing batch changed profiler".into());
        }
        let x = self.input.data();
        let y = self.target.data();
        gpu.upload_batch(x, y)?;
        private_control.upload_batch(x, y)?;
        control.upload_batch(x, y)?;
        for rate in [-1., f32::NAN, f32::INFINITY] {
            if !matches!(gpu.step_profiled(rate), Err(TrainingError::LearningRate)) {
                return Err("bad rate admitted".into());
            }
        }
        if gpu.upload_batch(x, &[]).is_ok()
            || gpu.submitted_steps() != 0
            || gpu.batch_generation() != 1
        {
            return Err("invalid profile input mutated counters".into());
        }
        let mut reports = Vec::new();
        let mut controls = Vec::new();
        let mut max_error = 0f64;
        for step in 0..12 {
            // The profiled workspace cannot enqueue the next step before validation.
            // Control steps use the identical encoder with timestamps disabled.
            let read = gpu.step_profiled(0.01)?;
            if read.submitted_step() != step + 1 || read.batch_generation() != 1 {
                return Err("profile receipt counter mismatch".into());
            }
            check_pending(&mut gpu, x, y)?;
            let profile = profile_read(read).await?;
            private_control.step(0.01)?;
            control.step(0.01)?;
            let p = loss(private_control.loss_snapshot()?).await?;
            let c = loss(control.loss_snapshot()?).await?;
            max_error = max_error
                .max(compare(&json!(profile.loss()), &json!(p))?)
                .max(compare(&json!(profile.loss()), &json!(c))?);
            let mut report = profile.report();
            report["warmup"] = json!(step < 3);
            reports.push(report);
            controls.push(json!({"private_loss":p,"ordinary_loss":c}));
        }
        let measured = graph_state(gpu.state_snapshot()?).await?;
        let private_state = graph_state(private_control.state_snapshot()?).await?;
        let ordinary = graph_state(control.state_snapshot()?).await?;
        max_error = max_error
            .max(compare(&measured, &private_state)?)
            .max(compare(&measured, &ordinary)?);

        // A numerical rejection proves rollback, not a successful profile. It
        // permits only an explicit subsequent attempt with a new valid batch.
        let before = measured["parameters"].clone();
        gpu.upload_batch(x, &vec![f32::MAX; y.len()])?;
        let read = gpu.step_profiled(0.01)?;
        #[cfg(not(target_arch = "wasm32"))]
        let rejected = read.read();
        #[cfg(target_arch = "wasm32")]
        let rejected = read.read_async().await;
        if !matches!(rejected, Err(TrainingError::Rejected { .. })) {
            return Err("nonfinite loss accepted or rollback not verified".into());
        }
        let snapshot = gpu.parameter_snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        let restored = snapshot.read()?;
        #[cfg(target_arch = "wasm32")]
        let restored = snapshot.read_async().await?;
        if json!(restored
            .parameters()
            .iter()
            .map(|p| &p.values)
            .collect::<Vec<_>>())
            != before
        {
            return Err("rejected profile partially updated weights".into());
        }
        gpu.upload_batch(x, y)?;
        profile_read(gpu.step_profiled(0.)?).await?;
        let adapter = json!({"name":gpu.adapter_info().name,"backend":format!("{:?}",gpu.adapter_info().backend),
            "device_type":format!("{:?}",gpu.adapter_info().device_type)});
        // An owning read remains valid after the profiler is dropped.
        let read = gpu.step_profiled(0.)?;
        drop(gpu);
        profile_read(read).await?;
        // Dropping a receipt must NOT act as acceptance or as a silent retry.
        let abandoned = private_control.step_profiled(0.)?;
        drop(abandoned);
        check_pending(&mut private_control, x, y)?;
        if self.runtime.context().device().features() != ordinary_features {
            return Err("profiling changed ordinary device features".into());
        }
        Ok(
            json!({"status":"passed","schema":"spiraltorch.graph_training_profile_fixture.v1",
                "config":self.config,"policy":policy.as_str(),"adapter":adapter,
                "profiles":reports,"controls":controls,"max_abs_error":max_error,
                "final_profile_state":measured,"final_private_control_state":private_state,"final_ordinary_state":ordinary,
                "guards":{"invalid_input":true,"pending_reuse":true,"numerical_rollback":true,
                    "explicit_recovery":true,"owning_read_after_drop":true,"abandoned_profile_quarantined":true,
                    "ordinary_device_features_unchanged":true},
                "boundary":"12 sequential SGD updates from identical initial weights on three private/ordinary workspaces; first 3 timestamp samples are warmups. Diagnostic pass timings only, not end-to-end speedup. All final gradients, predictions and parameters compared. Controls are uninstrumented; device timestamp feature differs only for the ordinary control."
            }),
        )
    }
}
