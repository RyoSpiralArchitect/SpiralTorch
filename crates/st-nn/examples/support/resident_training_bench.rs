//! Shared measured workload. Browser/native clients supply only a monotonic clock.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_backend_wgpu::{
    resident_matmul::{MatmulAccumulation, MatmulKernel},
    resident_training::{StepReadback, TrainingState, TrainingStateReadback},
    runtime::WgpuRuntime,
};
use st_nn::{layers::Gelu, module::Module, resident::InferencePlan, Linear, Sequential};
use st_tensor::{NdLayout, Tensor};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub shape: Vec<usize>,
    pub depth: usize,
    pub seed: u32,
    pub steps: usize,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Cadence {
    Immediate,
    Deferred,
}

pub struct Benchmark {
    runtime: WgpuRuntime,
    plan: InferencePlan,
    input: Tensor,
    target: Tensor,
    config: Config,
}

async fn loss(snapshot: StepReadback) -> Result<f32> {
    #[cfg(not(target_arch = "wasm32"))]
    let result = snapshot.read();
    #[cfg(target_arch = "wasm32")]
    let result = snapshot.read_async().await;
    Ok(result?)
}

async fn state(snapshot: TrainingStateReadback) -> Result<TrainingState> {
    #[cfg(not(target_arch = "wasm32"))]
    let result = snapshot.read();
    #[cfg(target_arch = "wasm32")]
    let result = snapshot.read_async().await;
    Ok(result?)
}

fn state_json(value: &TrainingState) -> Value {
    json!({"loss":value.loss,"prediction":value.prediction,"input_gradient":value.input_gradient,
        "parameters":value.parameters.iter().map(|p| json!({"weights":p.weights,"bias":p.bias})).collect::<Vec<_>>(),
        "parameter_gradients":value.parameter_gradients.iter().map(|p| json!({"weights":p.weights,"bias":p.bias})).collect::<Vec<_>>()})
}

impl Benchmark {
    pub fn new(runtime: WgpuRuntime, config: Config) -> Result<Self> {
        let layout = NdLayout::contiguous(&config.shape)?;
        let width = *config.shape.last().ok_or("feature axis required")?;
        if width == 0
            || width > 256
            || layout.is_empty()
            || layout.len() > 32768
            || config.depth == 0
            || config.depth > 16
            || config.steps == 0
            || config.steps > 32
            || config.seed == 0
        {
            return Err("benchmark exceeds bounded shape/depth/steps/seed".into());
        }
        let mut model = Sequential::new();
        for i in 0..config.depth {
            model.push(Linear::new(format!("linear_{i}"), width, width)?);
            if i + 1 < config.depth {
                model.push(Gelu::new());
            }
        }
        let mut seed = config.seed;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((seed % 65) as f32 - 32.) / 64.
        };
        let gain = 1.0 / (width as f32).sqrt();
        model.visit_parameters_mut(&mut |p| {
            let scale = if p.name().ends_with("weight") {
                gain
            } else {
                0.25
            };
            for value in p.value_mut().data_mut() {
                *value = next() * scale;
            }
            Ok(())
        })?;
        let input = Tensor::from_fn(layout.len() / width, width, |_, _| next())?;
        let target = Tensor::from_fn(layout.len() / width, width, |r, c| {
            0.4 * input.data()[r * width + c] - 0.2 * input.data()[r * width + (c + 1) % width]
        })?;
        let plan = InferencePlan::from_module(&model, layout)?;
        Ok(Self {
            runtime,
            plan,
            input,
            target,
            config,
        })
    }

    pub fn fixture(&self) -> Result<Value> {
        Ok(
            json!({"config":self.config,"plan_json":self.plan.to_json()?,"input":self.input.data(),"target":self.target.data(),
            "learning_rate":0.01,"kernel":"register_2x2","accumulation":"sequential",
            "adapter":{"name":self.runtime.adapter_info().name,"backend":format!("{:?}",self.runtime.adapter_info().backend),
                "device_type":format!("{:?}",self.runtime.adapter_info().device_type)},
            "build_manifest":serde_json::from_str::<Value>(st_core::build_manifest_json())?}),
        )
    }

    pub async fn sample(&self, cadence: Cadence, capture: bool, now: fn() -> f64) -> Result<Value> {
        let setup = now();
        let mut gpu = self.plan.compile_training_wgpu_with_options(
            self.runtime.clone(),
            Default::default(),
            MatmulKernel::Register2x2,
            MatmulAccumulation::Sequential,
        )?;
        gpu.upload_batch(self.input.data(), self.target.data())?;
        // Settle uploads and lazy pipeline compilation without changing the weights.
        gpu.step(0.)?;
        let initial_loss = loss(gpu.loss_snapshot()?).await?;
        let setup_ms = now() - setup;
        let mut receipts = Vec::with_capacity(self.config.steps);
        let mut losses = Vec::with_capacity(self.config.steps);
        let start = now();
        for _ in 0..self.config.steps {
            gpu.step(0.01)?;
            let snapshot = gpu.loss_snapshot()?;
            match cadence {
                Cadence::Immediate => losses.push(loss(snapshot).await?),
                Cadence::Deferred => receipts.push(snapshot),
            }
        }
        for snapshot in receipts {
            losses.push(loss(snapshot).await?);
        }
        let elapsed_ms = now() - start;
        if !elapsed_ms.is_finite() || elapsed_ms <= 0. || losses.len() != self.config.steps {
            return Err("invalid timed interval".into());
        }
        // The final zero-rate VJP and serialization are deliberately outside timing.
        gpu.step(0.)?;
        let final_state = state(gpu.state_snapshot()?).await?;
        let final_state = state_json(&final_state);
        let state_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&final_state)?));
        Ok(
            json!({"status":"passed","cadence":cadence,"elapsed_ms":elapsed_ms,"setup_ms":setup_ms,
            "steps":self.config.steps,"losses":losses,"initial_loss":initial_loss,"state_sha256":state_sha256,
            "state":if capture { final_state } else { Value::Null },
            "boundary":"device-persistent parameters and batch; every step's loss/finite flags captured and read; setup, initial/final zero-rate probes and state serialization excluded; elapsed includes all requested loss maps"}),
        )
    }
}
