#![cfg(target_arch = "wasm32")]

use st_nn::layers::spiral_rnn::SpiralRnn;
use st_nn::layers::wave_scan::{WaveScan, WaveScanStack};
use st_nn::{
    Dropout, GradientBands, Lstm, Module, Sequential, Tensor, ToposResonator, WaveRnn,
    ZSpaceCoherenceScan, ZSpaceCoherenceWaveBlock,
};
use wasm_bindgen::prelude::*;

fn consuming_layers() -> Vec<Box<dyn Module>> {
    vec![
        Box::new(Lstm::new("lstm", 6, 2).unwrap()),
        Box::new(SpiralRnn::new("spiral", 2, 2, 3).unwrap()),
        Box::new(WaveRnn::new("rnn", 2, 2, 1, 1, 0, -1., 1.).unwrap()),
        Box::new(WaveScan::new("scan", 2, 2, 1, 1, 0, 1, -1., 1.).unwrap()),
        Box::new(
            WaveScanStack::new(vec![
                WaveScan::new("a", 2, 2, 1, 1, 0, 1, -1., 1.).unwrap(),
                WaveScan::new("b", 2, 2, 1, 1, 0, 2, -1., 1.).unwrap(),
            ])
            .unwrap(),
        ),
        Box::new(ZSpaceCoherenceScan::new(2, 3, 3, -1., 1.).unwrap()),
        Box::new(ZSpaceCoherenceWaveBlock::new(2, 3, 3, -1., 1., 1, vec![1, 2]).unwrap()),
    ]
}

/// Execute the host-Tensor NN path compiled to scalar WASM, not WebGPU.
#[wasm_bindgen]
pub fn run_sequential_forward_tape_contract() -> String {
    let mut dropout = Sequential::new();
    dropout.push(Dropout::with_seed(0.5, Some(47)).unwrap());
    let ones = Tensor::from_vec(4, 32, vec![1.; 128]).unwrap();
    let mask = dropout.forward(&ones).unwrap();
    for _ in 0..2 {
        assert_eq!(dropout.backward(&ones, &ones).unwrap(), mask);
    }
    dropout.forward_untracked(&ones).unwrap();
    assert!(dropout.backward(&ones, &ones).is_err());

    let input = Tensor::from_fn(2, 6, |r, c| (r * 6 + c) as f32 * 0.03 - 0.1).unwrap();
    let seed = Tensor::from_vec(2, 2, vec![0.25, -0.2, 0.125, 0.3]).unwrap();
    let bands = GradientBands::from_components(seed.clone(), seed.clone(), seed.clone()).unwrap();
    for (mut reference, layer) in consuming_layers().into_iter().zip(consuming_layers()) {
        let mut child = Sequential::new();
        child.push_boxed(layer);
        let mut sequence = Sequential::new();
        sequence.push(child);
        sequence
            .load_state_dict(&reference.state_dict().unwrap())
            .unwrap();
        assert_eq!(
            sequence.forward(&input).unwrap(),
            reference.forward(&input).unwrap()
        );
        let expected = reference.backward(&input, &seed).unwrap();
        assert!(reference.backward(&input, &seed).is_err());
        for _ in 0..2 {
            sequence.zero_accumulators().unwrap();
            assert_eq!(sequence.backward(&input, &seed).unwrap(), expected);
        }
        sequence.zero_accumulators().unwrap();
        let actual = sequence.backward_bands(&input, &bands).unwrap();
        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - 3. * expected).abs() < 1e-5);
        }
    }

    let mut child = Sequential::new();
    child.push(ToposResonator::new("topos", 2, 3).unwrap());
    let mut topos = Sequential::new();
    topos.push(child);
    topos.attach_realgrad(0.01).unwrap();
    let input = Tensor::from_vec(2, 3, vec![0.25; 6]).unwrap();
    topos.forward(&input).unwrap();
    assert!(topos.scale_learning_rates(f32::NAN).is_err());
    topos.scale_learning_rates(0.5).unwrap();
    topos.zero_accumulators().unwrap();
    let expected = topos.backward(&input, &input).unwrap();
    topos.zero_accumulators().unwrap();
    assert_eq!(topos.backward(&input, &input).unwrap(), expected);
    topos.apply_step(0.01).unwrap();
    assert!(topos.backward(&input, &input).is_err());

    "passed: scalar WASM; Dropout mask, 7 consuming layer/stack types, nested Topos; no speed claim"
        .into()
}
