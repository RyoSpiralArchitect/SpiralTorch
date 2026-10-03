//! Local diagnostic of a learning snapshot plus requested alpha VJP.
//! No bindings, model execution, output inspection or snapshot drop is timed.

use st_frac::learning::FractionalGlKernel;
use std::hint::black_box;
use std::time::Instant;

fn fingerprint(values: impl Iterator<Item = f32>) -> u64 {
    values.fold(0xcbf29ce484222325, |hash, value| {
        value
            .to_bits()
            .to_le_bytes()
            .into_iter()
            .fold(hash, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
    })
}

fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("iterations must be an integer"))
        .unwrap_or(12);
    assert!((1..=1000).contains(&iterations));
    for (shape, axis) in [
        (vec![2, 128, 768], 1),
        (vec![2, 768, 128], 2),
        (vec![196608], 0),
    ] {
        let len: usize = shape.iter().product();
        let input: Vec<f32> = (0..len)
            .map(|i| ((i * 73 % 997) as f32 / 997.) - 0.5)
            .collect();
        let upstream: Vec<f32> = (0..len)
            .map(|i| ((i * 17 % 991) as f32 / 991.) - 0.5)
            .collect();
        let kernel = FractionalGlKernel::new(32, 1., len, len * 32).unwrap();
        for history in [false, true] {
            let mut times = Vec::with_capacity(iterations);
            let mut identity = None;
            for iteration in 0..iterations + 2 {
                let start = Instant::now();
                let saved = if history {
                    kernel.forward_history(black_box(&input), &shape, axis, 0.9)
                } else {
                    kernel.forward(black_box(&input), &shape, axis, 0.9)
                }
                .unwrap();
                let alpha = saved.vjp_alpha(black_box(&upstream)).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                let current = (fingerprint(saved.output().iter().copied()), alpha.to_bits());
                if let Some(expected) = identity {
                    assert_eq!(current, expected);
                } else {
                    identity = Some(current);
                }
                if iteration >= 2 {
                    times.push(elapsed);
                }
            }
            let (output_hash, alpha_bits) = identity.unwrap();
            let mut ordered = times.clone();
            ordered.sort_by(f64::total_cmp);
            let median = if iterations.is_multiple_of(2) {
                (ordered[iterations / 2 - 1] + ordered[iterations / 2]) / 2.
            } else {
                ordered[iterations / 2]
            };
            println!("{{\"shape\":{shape:?},\"axis\":{axis},\"history\":{history},\"kernel_len\":32,\"alpha\":0.9,\"step\":1,\"output_fnv1a64\":\"{output_hash:016x}\",\"alpha_gradient_f32_bits\":{alpha_bits},\"median_ms\":{median},\"measurements_ms\":{times:?}}}");
        }
    }
}
