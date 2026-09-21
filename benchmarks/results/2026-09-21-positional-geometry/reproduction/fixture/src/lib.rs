use st_tensor::{Layout, Tensor};
use st_vision::nerf::PositionalEncoding;
use wasm_bindgen::prelude::*;
use st_nn::Module;
use st_vision::nerf::{NerfField, NerfFieldConfig};

#[wasm_bindgen]
pub struct EncodingCase {
    encoder: PositionalEncoding,
    input: Tensor,
}

#[wasm_bindgen]
impl EncodingCase {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize, bands: usize, residual: bool, layout: u32) -> Result<Self, String> {
        let values = (0..rows * cols).map(|i| (i % 257) as f32 / 64.0 - 2.0).collect();
        Self::with_values(rows, cols, bands, residual, layout, values)
    }
    pub fn with_values(rows: usize, cols: usize, bands: usize, residual: bool, layout: u32, values: Vec<f32>) -> Result<Self, String> {
        let mut encoder = PositionalEncoding::new(cols, bands).map_err(|e| e.to_string())?;
        if !residual { encoder = encoder.without_input(); }
        let input = Tensor::from_vec(rows, cols, values).map_err(|e| e.to_string())?;
        let layout = match layout {
            0 => Layout::RowMajor, 1 => Layout::ColMajor,
            _ => Layout::Chimera { stripes: 3, tile: (cols / 3) as u32 },
        };
        Ok(Self { encoder, input: input.to_layout(layout).map_err(|e| e.to_string())? })
    }
    pub fn run(&self) -> Result<Encoded, String> {
        Ok(Encoded { tensor: self.encoder.encode(&self.input).map_err(|e| e.to_string())? })
    }
}

fn field() -> NerfField {
    let mut field = NerfField::new(NerfFieldConfig {
        position_frequencies: 1, direction_frequencies: 1,
        hidden_layers: 1, hidden_width: 4, feature_dim: 3,
        color_layers: 1, color_hidden_width: 4, ..NerfFieldConfig::default()
    }).unwrap();
    field.visit_parameters_mut(&mut |p| {
        for (i, v) in p.value_mut().data_mut().iter_mut().enumerate() {
            *v = 0.01 + (i % 11) as f32 / 100.0;
        }
        Ok(())
    }).unwrap();
    field
}
fn parameters(field: &NerfField, gradient: bool) -> Vec<u32> {
    let mut bits = Vec::new();
    field.visit_parameters(&mut |p| {
        let tensor = if gradient { p.gradient().unwrap() } else { p.value() };
        bits.extend(tensor.data().iter().map(|v| v.to_bits()));
        Ok(())
    }).unwrap();
    bits
}

#[wasm_bindgen]
pub fn contract_report() -> String {
    use st_core::util::rope_lru::{RopeKey, RopeLRU};
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::execution::{push_backend_policy, BackendPolicy};
    let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
    let key = |theta| RopeKey { theta, t: 8, dtype: "f64", device: "cpu" };
    let mut cached = RopeLRU::new(2);
    cached.get(key(1.0));
    let actual = cached.get(key(1.0+5e-10)).0.to_vec();
    let mut fresh = RopeLRU::new(2);
    let history_independent = actual == fresh.get(key(1.0+5e-10)).0;
    let input = Tensor::from_vec(2,6,(0..12).map(|i|0.05+i as f32/100.0).collect()).unwrap();
    let seed = Tensor::from_vec(2,4,vec![0.5,-0.25,0.75,0.125,-0.5,0.25,-0.125,1.0]).unwrap();
    let mut reference = field();
    let y = reference.forward(&input).unwrap();
    reference.backward(&input,&seed).unwrap();
    let g = parameters(&reference,true);
    reference.apply_step(1e-3).unwrap();
    let next = parameters(&reference,false);
    let mut fields = Vec::new();
    for (layout_id, layout) in [Layout::RowMajor,Layout::ColMajor,Layout::Chimera{stripes:2,tile:3}].into_iter().enumerate() {
        for (seed_id, seed_layout) in [Layout::RowMajor,Layout::ColMajor,Layout::Chimera{stripes:2,tile:2}].into_iter().enumerate() {
            let mut model = field();
            let x = input.to_layout(layout).unwrap(); let s = seed.to_layout(seed_layout).unwrap();
            let forward = model.forward(&x).unwrap();
            let dx = model.backward(&x,&s).unwrap();
            let gradient = parameters(&model,true);
            model.apply_step(1e-3).unwrap();
            let update = parameters(&model,false);
            fields.push(serde_json::json!({"layout":layout_id,"seed_layout":seed_id,
                "forward_equal":forward.data()==y.data(),"gradient_equal":gradient==g,"update_equal":update==next,
                "input_gradient_zero":dx.data().iter().all(|x|*x==0.0)}));
        }
    }
    serde_json::json!({"rope_history_independent":history_independent,"fields":fields}).to_string()
}

#[wasm_bindgen]
pub struct Encoded { tensor: Tensor }

#[wasm_bindgen]
impl Encoded {
    pub fn values(&self) -> Vec<f32> { self.tensor.data().to_vec() }
}

impl Encoded {
    pub fn data(&self) -> &[f32] { self.tensor.data() }
}

pub fn reference(rows: usize, cols: usize, bands: usize, residual: bool) -> Vec<f64> {
    let mut expected = Vec::new();
    for row in 0..rows {
        let values: Vec<_> = (0..cols).map(|col| ((row * cols + col) % 257) as f64 / 64.0 - 2.0).collect();
        if residual { expected.extend_from_slice(&values); }
        for band in 0..bands {
            for &value in &values {
                let phase = value * 2f64.powi(band as i32);
                expected.extend([phase.sin(), phase.cos()]);
            }
        }
    }
    expected
}
