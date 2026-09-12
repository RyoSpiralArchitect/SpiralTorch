//! Custom-objective timing, distinct from the ordinary mean-MSE benchmark.
use super::*;
use st_backend_wgpu::{
    resident_tensor::pointwise::PointwisePlan,
    resident_tensor::{ResidentTensor, TensorReadback},
    resident_training::graph::{
        GraphForward, GraphGradients, GraphUpdateReadback, ResidentGraphLearner,
    },
};
use st_tensor::{PointwiseChain, PointwiseExecution, PointwiseStep};

const HOST_PHASES: [&str; 8] = [
    "forward_enqueue",
    "quadratic_seed_enqueue",
    "quadratic_vjp_enqueue",
    "quartic_seed_enqueue",
    "quartic_vjp_enqueue",
    "update_enqueue",
    "receipt_capture_enqueue",
    "receipt_wait",
];

// Compile out phase clocks on ordinary intervals. These are host wall times,
// not GPU timestamps: enqueued GPU work overlaps them and the eventual wait.
struct HostPhases<const ENABLED: bool> {
    now: fn() -> f64,
    totals: [f64; 8],
    counts: [usize; 8],
}

impl<const ENABLED: bool> HostPhases<ENABLED> {
    fn new(now: fn() -> f64) -> Self {
        Self {
            now,
            totals: [0.; 8],
            counts: [0; 8],
        }
    }
    fn start(&self) -> f64 {
        if ENABLED {
            (self.now)()
        } else {
            0.
        }
    }
    fn end(&mut self, phase: usize, start: f64) {
        if ENABLED {
            self.totals[phase] += (self.now)() - start;
            self.counts[phase] += 1;
        }
    }
    fn measure<T>(&mut self, phase: usize, action: impl FnOnce() -> T) -> T {
        let start = self.start();
        let result = action();
        self.end(phase, start);
        result
    }
    fn report(&self, elapsed_ms: f64) -> Value {
        json!({"schema":"spiraltorch.learner_host_phases.v1", "instrumented":true,
            "clock_domain":"host_wall", "gpu_phase_attribution":false,
            "phases":HOST_PHASES.iter().enumerate().map(|(i, name)|
                json!({"name":name,"ms":self.totals[i],"count":self.counts[i]})).collect::<Vec<_>>(),
            "unattributed_ms":elapsed_ms-self.totals.iter().sum::<f64>()})
    }
}

async fn accepted(receipt: GraphUpdateReadback) -> Result<u64> {
    #[cfg(not(target_arch = "wasm32"))]
    let value = receipt.read();
    #[cfg(target_arch = "wasm32")]
    let value = receipt.read_async().await;
    Ok(value?)
}

async fn values(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot: TensorReadback = tensor.snapshot()?;
    #[cfg(not(target_arch = "wasm32"))]
    let value = snapshot.read();
    #[cfg(target_arch = "wasm32")]
    let value = snapshot.read_async().await;
    Ok(value?)
}

fn vjps<const PROFILE: bool>(
    gpu: &mut ResidentGraphLearner,
    negative: &ResidentTensor,
    norm: &ResidentTensor,
    cube: Option<&PointwisePlan>,
    phases: &mut HostPhases<PROFILE>,
) -> Result<(GraphForward, [GraphGradients; 2])> {
    let f = phases.measure(0, || gpu.forward())?;
    let (e, seed) = phases.measure(1, || -> Result<_> {
        let e = f.prediction().add(negative)?;
        let seed = e.mul(norm)?;
        Ok((e, seed))
    })?;
    let first = phases.measure(2, || gpu.backward(&f, &seed))?;
    drop(seed);
    let seed = phases.measure(3, || -> Result<_> {
        Ok(match cube {
            Some(plan) => plan.run(&[&e, norm], PointwiseExecution::Fused)?,
            None => e.mul(&e)?.mul(&e)?.mul(norm)?,
        })
    })?;
    let second = phases.measure(4, || gpu.backward(&f, &seed))?;
    Ok((f, [first, second]))
}

fn objective(prediction: &[f32], target: &[f32]) -> f32 {
    prediction
        .iter()
        .zip(target)
        .map(|(p, t)| {
            let e = p - t;
            (0.75 * 0.5 * e * e + 0.25 * 0.25 * e * e * e * e) / target.len() as f32
        })
        .sum()
}

impl Benchmark {
    pub async fn learn(&self, cadence: Cadence, capture: bool, now: fn() -> f64) -> Result<Value> {
        self.learn_inner::<false>(cadence, capture, now).await
    }

    pub async fn learn_host_profile(
        &self,
        cadence: Cadence,
        capture: bool,
        now: fn() -> f64,
    ) -> Result<Value> {
        self.learn_inner::<true>(cadence, capture, now).await
    }

    async fn learn_inner<const PROFILE: bool>(
        &self,
        cadence: Cadence,
        capture: bool,
        now: fn() -> f64,
    ) -> Result<Value> {
        if !self.config.graph {
            return Err("learner benchmark requires a graph".into());
        }
        let setup = now();
        let mut gpu = self.plan.compile_graph_learner_wgpu_with_options(
            self.runtime.clone(),
            GraphGradientPolicy::Exact,
            Default::default(),
            MatmulKernel::Register2x2,
            MatmulAccumulation::Sequential,
        )?;
        if let Some(optimizer) = self.config.learner_optimizer {
            gpu.set_momentum_damping(0.5)?;
            if matches!(optimizer, LearnerOptimizer::ClippedToposEma) {
                gpu.set_grad_clip_max_norm(1. / 1024.)?;
            }
        }
        let d = gpu.tensor_device().clone();
        let negative = d.upload(
            &self.config.shape,
            &self.target.data().iter().map(|v| -v).collect::<Vec<_>>(),
        )?;
        let norm = d.upload(&[], &[1. / self.target.data().len() as f32])?;
        gpu.upload(self.input.data())?;
        let cube = if self.config.fuse_learner_seeds {
            Some(PointwisePlan::new(
                d.clone(),
                PointwiseChain::new(
                    2,
                    vec![
                        PointwiseStep::named("multiply", Some(0))?,
                        PointwiseStep::named("multiply", Some(0))?,
                        PointwiseStep::named("multiply", Some(1))?,
                    ],
                )?,
                vec![gpu.output_layout().clone(), norm.layout().clone()],
            )?)
        } else {
            None
        };
        let mut unprofiled = HostPhases::<false>::new(now);
        let (initial, g) = vjps(&mut gpu, &negative, &norm, cube.as_ref(), &mut unprofiled)?;
        gpu.sgd_weighted(&[(&g[0], 0.75), (&g[1], 0.25)], 0.)?;
        if accepted(gpu.update_snapshot()?).await? != 1 {
            return Err("initial update identity".into());
        }
        let initial_loss = objective(&values(initial.prediction()).await?, self.target.data());
        let setup_ms = now() - setup;
        drop((initial, g));
        let mut receipts = Vec::with_capacity(self.config.steps);
        let mut accepted_updates = Vec::with_capacity(self.config.steps);
        let mut phases = HostPhases::<PROFILE>::new(now);
        let start = now();
        for _ in 0..self.config.steps {
            let (_, g) = vjps(&mut gpu, &negative, &norm, cube.as_ref(), &mut phases)?;
            phases.measure(5, || {
                gpu.sgd_weighted(&[(&g[0], 0.75), (&g[1], 0.25)], 0.01)
            })?;
            let receipt = phases.measure(6, || gpu.update_snapshot())?;
            match cadence {
                Cadence::Immediate => {
                    let start = phases.start();
                    accepted_updates.push(accepted(receipt).await?);
                    phases.end(7, start);
                }
                Cadence::Deferred => receipts.push(receipt),
            }
        }
        for receipt in receipts {
            let start = phases.start();
            accepted_updates.push(accepted(receipt).await?);
            phases.end(7, start);
        }
        let elapsed_ms = now() - start;
        if !elapsed_ms.is_finite()
            || elapsed_ms <= 0.
            || accepted_updates != (2..self.config.steps as u64 + 2).collect::<Vec<_>>()
        {
            return Err("invalid learner interval or missing acceptance".into());
        }
        // Final full state observation is outside timing, including the host objective.
        let (f, g) = vjps(&mut gpu, &negative, &norm, cube.as_ref(), &mut unprofiled)?;
        let prediction = values(f.prediction()).await?;
        let final_loss = objective(&prediction, self.target.data());
        let mut input_gradients = Vec::new();
        let mut raw_gradients = Vec::new();
        for source in g {
            input_gradients.push(values(source.input_gradient()).await?);
            let mut raw = Vec::new();
            for p in source.parameter_gradients() {
                raw.push(values(p).await?);
            }
            raw_gradients.push(raw);
        }
        let saved = gpu.parameter_snapshot()?;
        #[cfg(not(target_arch = "wasm32"))]
        let parameters = saved.read()?;
        #[cfg(target_arch = "wasm32")]
        let parameters = saved.read_async().await?;
        let mut final_state = json!({"loss":final_loss,"prediction":prediction,"input_gradients":input_gradients,
            "raw_gradients":raw_gradients,"parameters":parameters.parameters().iter().map(|p|&p.values).collect::<Vec<_>>()});
        if self.config.learner_optimizer.is_some() {
            let mut history = Vec::new();
            for tensor in gpu.momentum_tensors()? {
                history.push(values(&tensor).await?);
            }
            final_state["momentum"] = json!(history);
        }
        let state_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&final_state)?));
        let mut result = json!({"status":"passed","learner":true,"cadence":cadence,"steps":self.config.steps,
            "fused_learner_seeds":self.config.fuse_learner_seeds,
            "learner_optimizer":self.config.learner_optimizer,
            "momentum_damping":gpu.momentum_damping(),"grad_clip_max_norm":gpu.grad_clip_max_norm(),
            "completed_updates":self.config.steps,"accepted_updates":accepted_updates,"acceptance":"guarded_receipts",
            "losses":[],"initial_loss":initial_loss,"final_loss":final_loss,"elapsed_ms":elapsed_ms,"setup_ms":setup_ms,
            "state_sha256":state_sha256,"state":if capture { final_state } else { Value::Null },
            "boundary":"quadratic/quartic GPU seeds and two exact VJPs; weighted SGD; every update acceptance captured/read; no per-step loss read; reset, zero-rate warmup and terminal state/host objective outside timing"});
        if PROFILE {
            result["host_profile"] = phases.report(elapsed_ms);
        }
        Ok(result)
    }
}
