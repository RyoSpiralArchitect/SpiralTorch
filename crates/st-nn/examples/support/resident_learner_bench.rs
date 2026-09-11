//! Custom-objective timing, distinct from the ordinary mean-MSE benchmark.
use super::*;
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorReadback},
    resident_training::graph::{
        GraphForward, GraphGradients, GraphUpdateReadback, ResidentGraphLearner,
    },
};

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

fn vjps(
    gpu: &mut ResidentGraphLearner,
    negative: &ResidentTensor,
    norm: &ResidentTensor,
) -> Result<(GraphForward, [GraphGradients; 2])> {
    let f = gpu.forward()?;
    let e = f.prediction().add(negative)?;
    let first = gpu.backward(&f, &e.mul(norm)?)?;
    let second = gpu.backward(&f, &e.mul(&e)?.mul(&e)?.mul(norm)?)?;
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
        let d = gpu.tensor_device().clone();
        let negative = d.upload(
            &self.config.shape,
            &self.target.data().iter().map(|v| -v).collect::<Vec<_>>(),
        )?;
        let norm = d.upload(&[], &[1. / self.target.data().len() as f32])?;
        gpu.upload(self.input.data())?;
        let (initial, g) = vjps(&mut gpu, &negative, &norm)?;
        gpu.sgd_weighted(&[(&g[0], 0.75), (&g[1], 0.25)], 0.)?;
        if accepted(gpu.update_snapshot()?).await? != 1 {
            return Err("initial update identity".into());
        }
        let initial_loss = objective(&values(initial.prediction()).await?, self.target.data());
        let setup_ms = now() - setup;
        drop((initial, g));
        let mut receipts = Vec::with_capacity(self.config.steps);
        let mut accepted_updates = Vec::with_capacity(self.config.steps);
        let start = now();
        for _ in 0..self.config.steps {
            let (_, g) = vjps(&mut gpu, &negative, &norm)?;
            gpu.sgd_weighted(&[(&g[0], 0.75), (&g[1], 0.25)], 0.01)?;
            let receipt = gpu.update_snapshot()?;
            match cadence {
                Cadence::Immediate => accepted_updates.push(accepted(receipt).await?),
                Cadence::Deferred => receipts.push(receipt),
            }
        }
        for receipt in receipts {
            accepted_updates.push(accepted(receipt).await?);
        }
        let elapsed_ms = now() - start;
        if !elapsed_ms.is_finite()
            || elapsed_ms <= 0.
            || accepted_updates != (2..self.config.steps as u64 + 2).collect::<Vec<_>>()
        {
            return Err("invalid learner interval or missing acceptance".into());
        }
        // Final full state observation is outside timing, including the host objective.
        let (f, g) = vjps(&mut gpu, &negative, &norm)?;
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
        let final_state = json!({"loss":final_loss,"prediction":prediction,"input_gradients":input_gradients,
            "raw_gradients":raw_gradients,"parameters":parameters.parameters().iter().map(|p|&p.values).collect::<Vec<_>>()});
        let state_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&final_state)?));
        Ok(
            json!({"status":"passed","learner":true,"cadence":cadence,"steps":self.config.steps,
            "completed_updates":self.config.steps,"accepted_updates":accepted_updates,"acceptance":"guarded_receipts",
            "losses":[],"initial_loss":initial_loss,"final_loss":final_loss,"elapsed_ms":elapsed_ms,"setup_ms":setup_ms,
            "state_sha256":state_sha256,"state":if capture { final_state } else { Value::Null },
            "boundary":"quadratic/quartic GPU seeds and two exact VJPs; weighted SGD; every update acceptance captured/read; no per-step loss read; reset, zero-rate warmup and terminal state/host objective outside timing"}),
        )
    }
}
