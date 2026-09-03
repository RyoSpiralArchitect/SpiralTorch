// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complex-state extension of the same Strang-split equation. Unlike the real
//! projection interface, both output quadratures can be fed into the next step.

use super::*;

pub const STOCHASTIC_SCHRODINGER_COMPLEX_CONTRACT_VERSION: &str =
    "spiraltorch.stochastic_complex_schrodinger.v1";

#[derive(Clone, Copy, Debug)]
pub struct StochasticSchrodingerComplexInput<'a> {
    pub real: &'a [f32],
    pub imaginary: &'a [f32],
    pub potential: &'a [f32],
    pub standard_normal: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: StochasticSchrodingerConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StochasticSchrodingerComplexStep {
    pub contract_version: &'static str,
    pub arithmetic: &'static str,
    pub output_real: Vec<f32>,
    pub output_imaginary: Vec<f32>,
    pub phase: Vec<f32>,
    pub initial_norm_squared: f64,
    pub final_norm_squared: f64,
    pub expected_norm_ratio: f64,
    pub max_row_norm_error: f64,
}

/// Real Euclidean VJP of both quadratures, not a holomorphic derivative of a loss.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StochasticSchrodingerComplexBackward {
    pub grad_input_real: Vec<f32>,
    pub grad_input_imaginary: Vec<f32>,
    pub grad_potential: Vec<f32>,
}

type Complex = (f64, f64);

fn multiply(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn phase_cotangent(g: Complex, z: Complex) -> f64 {
    g.0 * z.1 - g.1 * z.0
}

fn value(real: &[f32], imaginary: &[f32], index: usize) -> Complex {
    (f64::from(real[index]), f64::from(imaginary[index]))
}

fn phases(
    input: StochasticSchrodingerComplexInput<'_>,
) -> Result<Vec<f32>, StochasticSchrodingerError> {
    validate_stochastic_schrodinger_state(
        input.real,
        input.potential,
        input.rows,
        input.features,
        input.config,
    )?;
    validate_length("input_imaginary", input.imaginary, input.real.len())?;
    validate_length("standard_normal", input.standard_normal, input.real.len())?;
    let noise_scale = input.config.noise_scale() * libm::sqrtf(input.config.time_step());
    input
        .standard_normal
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            require_derived_finite(
                "phase",
                input.potential[i % input.features] * input.config.time_step()
                    + sample * noise_scale,
            )
        })
        .collect()
}

fn damping_factor(config: StochasticSchrodingerConfig) -> f32 {
    // Host expf implementations can differ by an ULP, invalidating cross-client replay.
    libm::expf(-0.5 * config.loss_rate() * config.time_step())
}

// The pair block is symmetric: a_i = d*c*exp(-i*p_i),
// b_ij = -i*d*s*exp(-i*(p_i+p_j)/2). Odd terminal features have no hopping.
fn coefficients(
    phase: &[f32],
    index: usize,
    partner: Option<usize>,
    config: StochasticSchrodingerConfig,
) -> (Complex, Complex) {
    let p = f64::from(phase[index]);
    let damping = f64::from(damping_factor(config));
    let (own_scale, cross) = if let Some(j) = partner {
        let angle = f64::from(config.hopping_angle());
        let mean = f64::from(phase[index]) * 0.5 + f64::from(phase[j]) * 0.5;
        (
            damping * libm::cos(angle),
            (
                -damping * libm::sin(angle) * libm::sin(mean),
                -damping * libm::sin(angle) * libm::cos(mean),
            ),
        )
    } else {
        (damping, (0.0, 0.0))
    };
    ((own_scale * libm::cos(p), -own_scale * libm::sin(p)), cross)
}

pub fn apply_stochastic_schrodinger_complex_step(
    input: StochasticSchrodingerComplexInput<'_>,
) -> Result<StochasticSchrodingerComplexStep, StochasticSchrodingerError> {
    let phase = phases(input)?;
    let volume = input.real.len();
    let mut output_real = vec![0.0; volume];
    let mut output_imaginary = vec![0.0; volume];
    let mut initial_norm_squared = 0.0;
    let mut final_norm_squared = 0.0;
    let mut max_row_norm_error = 0.0f64;
    let expected_norm_ratio = f64::from(damping_factor(input.config)).powi(2);
    for row in 0..input.rows {
        let start = row * input.features;
        let mut initial = 0.0;
        let mut final_norm = 0.0;
        for col in 0..input.features {
            let i = start + col;
            let partner = adjacent_partner(col, input.features).map(|j| start + j);
            let (a, b) = coefficients(&phase, i, partner, input.config);
            let x = value(input.real, input.imaginary, i);
            let own = multiply(a, x);
            let cross = partner
                .map(|j| multiply(b, value(input.real, input.imaginary, j)))
                .unwrap_or((0.0, 0.0));
            output_real[i] = require_derived_finite("output_real", (own.0 + cross.0) as f32)?;
            output_imaginary[i] =
                require_derived_finite("output_imaginary", (own.1 + cross.1) as f32)?;
            initial += x.0 * x.0 + x.1 * x.1;
            final_norm +=
                f64::from(output_real[i]).powi(2) + f64::from(output_imaginary[i]).powi(2);
        }
        let expected = initial * expected_norm_ratio;
        let error = (final_norm - expected).abs();
        let tolerance = norm_tolerance(input.features, expected);
        if error > tolerance {
            return Err(StochasticSchrodingerError::NormInvariant {
                row,
                error,
                tolerance,
            });
        }
        initial_norm_squared += initial;
        final_norm_squared += final_norm;
        max_row_norm_error = max_row_norm_error.max(error);
    }
    Ok(StochasticSchrodingerComplexStep {
        contract_version: STOCHASTIC_SCHRODINGER_COMPLEX_CONTRACT_VERSION,
        arithmetic: "libm_0.2.16_f32_phase_f64_complex",
        output_real,
        output_imaginary,
        phase,
        initial_norm_squared,
        final_norm_squared,
        expected_norm_ratio,
        max_row_norm_error,
    })
}

/// Differentiate one discrete step with fixed config and Gaussian witnesses.
/// Feed both returned input cotangents into the preceding step for a trajectory.
pub fn backward_stochastic_schrodinger_complex_step(
    input: StochasticSchrodingerComplexInput<'_>,
    grad_output_real: &[f32],
    grad_output_imaginary: &[f32],
) -> Result<StochasticSchrodingerComplexBackward, StochasticSchrodingerError> {
    let phase = phases(input)?;
    let volume = input.real.len();
    validate_length("grad_output_real", grad_output_real, volume)?;
    validate_length("grad_output_imaginary", grad_output_imaginary, volume)?;
    let mut grad_input_real = vec![0.0; volume];
    let mut grad_input_imaginary = vec![0.0; volume];
    let mut potential_sum = vec![0.0f64; input.features];
    for row in 0..input.rows {
        let start = row * input.features;
        for (col, potential_grad) in potential_sum.iter_mut().enumerate() {
            let i = start + col;
            let partner = adjacent_partner(col, input.features).map(|j| start + j);
            let (a, b) = coefficients(&phase, i, partner, input.config);
            let x = value(input.real, input.imaginary, i);
            let g = value(grad_output_real, grad_output_imaginary, i);
            let mut adjoint = multiply((a.0, -a.1), g);
            let mut grad_phase = phase_cotangent(g, multiply(a, x));
            if let Some(j) = partner {
                let partner_g = value(grad_output_real, grad_output_imaginary, j);
                let cross_adjoint = multiply((b.0, -b.1), partner_g);
                adjoint.0 += cross_adjoint.0;
                adjoint.1 += cross_adjoint.1;
                grad_phase += 0.5
                    * (phase_cotangent(g, multiply(b, value(input.real, input.imaginary, j)))
                        + phase_cotangent(partner_g, multiply(b, x)));
            }
            grad_input_real[i] = require_derived_finite("grad_input_real", adjoint.0 as f32)?;
            grad_input_imaginary[i] =
                require_derived_finite("grad_input_imaginary", adjoint.1 as f32)?;
            *potential_grad += grad_phase * f64::from(input.config.time_step());
        }
    }
    let grad_potential = potential_sum
        .into_iter()
        .map(|g| require_derived_finite("grad_potential", g as f32))
        .collect::<Result<_, _>>()?;
    Ok(StochasticSchrodingerComplexBackward {
        grad_input_real,
        grad_input_imaginary,
        grad_potential,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> StochasticSchrodingerConfig {
        StochasticSchrodingerConfig::new(0.07, 0.2)
            .unwrap()
            .with_time_step(0.3)
            .unwrap()
            .with_hopping_rate(-0.8)
            .unwrap()
    }

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())),
            "{a} != {b}"
        );
    }

    #[test]
    fn damping_uses_the_cross_client_golden_bits_not_host_expf() {
        // Native macOS expf previously rounded this witness one ULP below WASM.
        let config = StochasticSchrodingerConfig::new(0.20157552, 0.0)
            .unwrap()
            .with_time_step(0.22900859)
            .unwrap();
        let step = apply_stochastic_schrodinger_complex_step(StochasticSchrodingerComplexInput {
            real: &[1.0],
            imaginary: &[0.0],
            potential: &[0.0],
            standard_normal: &[0.0],
            rows: 1,
            features: 1,
            config,
        })
        .unwrap();
        assert_eq!(step.output_real[0].to_bits(), 0x3f7a28ac);
        assert_eq!(step.expected_norm_ratio, 0.9548868178858925);
        assert_eq!(step.arithmetic, "libm_0.2.16_f32_phase_f64_complex");
    }

    #[test]
    fn complex_identity_and_real_projection_compatibility() {
        let real = [0.2, -0.8, 0.5, 0.3, 0.7, -0.4];
        let imaginary = [0.5, 0.1, -0.2, 0.6, -0.9, 0.2];
        let potential = [0.3, -0.6, 0.2];
        let noise = [0.5; 6];
        let mut input = StochasticSchrodingerComplexInput {
            real: &real,
            imaginary: &imaginary,
            potential: &potential,
            standard_normal: &noise,
            rows: 2,
            features: 3,
            config: config().with_time_step(0.0).unwrap(),
        };
        let step = apply_stochastic_schrodinger_complex_step(input).unwrap();
        assert_eq!(step.output_real, real);
        assert_eq!(step.output_imaginary, imaginary);
        input.imaginary = &[0.0; 6];
        input.config = config();
        let complex = apply_stochastic_schrodinger_complex_step(input).unwrap();
        let legacy =
            apply_stochastic_schrodinger_step(&real, &potential, &noise, 2, 3, config()).unwrap();
        for (actual, expected) in complex
            .output_real
            .iter()
            .chain(&complex.output_imaginary)
            .zip(legacy.output_real.iter().chain(&legacy.output_imaginary))
        {
            close(f64::from(*actual), f64::from(*expected), 2e-7);
        }
        let g = [0.1, -0.2, 0.4, 0.3, -0.5, 0.7];
        let complex_grad =
            backward_stochastic_schrodinger_complex_step(input, &g, &[0.0; 6]).unwrap();
        let legacy_grad =
            backward_stochastic_schrodinger_step(&real, &legacy.phase, &g, 2, 3, config()).unwrap();
        for (a, b) in complex_grad
            .grad_input_real
            .iter()
            .chain(&complex_grad.grad_potential)
            .zip(
                legacy_grad
                    .grad_input
                    .iter()
                    .chain(&legacy_grad.grad_potential),
            )
        {
            close(f64::from(*a), f64::from(*b), 2e-7);
        }
    }

    #[test]
    fn both_quadratures_and_shared_potential_match_finite_differences() {
        let real = [0.2, -0.8, 0.5, 0.3, 0.7, -0.4];
        let imaginary = [0.5, 0.1, -0.2, 0.6, -0.9, 0.2];
        let potential = [0.3, -0.6, 0.2];
        let noise = [0.3, -0.5, 0.7, -0.2, 0.1, 0.4];
        let gr = [0.6, -0.2, 0.8, -0.3, 0.5, 0.1];
        let gi = [-0.4, 0.7, -0.6, 0.2, 0.8, -0.5];
        let input = StochasticSchrodingerComplexInput {
            real: &real,
            imaginary: &imaginary,
            potential: &potential,
            standard_normal: &noise,
            rows: 2,
            features: 3,
            config: config(),
        };
        let gradient = backward_stochastic_schrodinger_complex_step(input, &gr, &gi).unwrap();
        let loss = |r: &[f32], im: &[f32], p: &[f32]| {
            let output =
                apply_stochastic_schrodinger_complex_step(StochasticSchrodingerComplexInput {
                    real: r,
                    imaginary: im,
                    potential: p,
                    ..input
                })
                .unwrap();
            output
                .output_real
                .iter()
                .zip(gr)
                .chain(output.output_imaginary.iter().zip(gi))
                .map(|(&x, g)| f64::from(x) * f64::from(g))
                .sum::<f64>()
        };
        for field in 0..3 {
            let (source, expected) = match field {
                0 => (&real[..], &gradient.grad_input_real),
                1 => (&imaginary[..], &gradient.grad_input_imaginary),
                _ => (&potential[..], &gradient.grad_potential),
            };
            for i in 0..source.len() {
                let mut plus = source.to_vec();
                let mut minus = source.to_vec();
                plus[i] += 1e-3;
                minus[i] -= 1e-3;
                let (a, b) = match field {
                    0 => (
                        loss(&plus, &imaginary, &potential),
                        loss(&minus, &imaginary, &potential),
                    ),
                    1 => (
                        loss(&real, &plus, &potential),
                        loss(&real, &minus, &potential),
                    ),
                    _ => (
                        loss(&real, &imaginary, &plus),
                        loss(&real, &imaginary, &minus),
                    ),
                };
                close((a - b) / 2e-3, f64::from(expected[i]), 6e-5);
            }
        }
    }

    #[test]
    fn retained_imaginary_state_composes_and_is_reversible() {
        let real = [0.8, -0.3];
        let imaginary = [0.2, 0.5];
        let input = StochasticSchrodingerComplexInput {
            real: &real,
            imaginary: &imaginary,
            potential: &[0.0; 2],
            standard_normal: &[0.0; 2],
            rows: 1,
            features: 2,
            config: StochasticSchrodingerConfig::default(),
        };
        let first = apply_stochastic_schrodinger_complex_step(input).unwrap();
        let next_input = StochasticSchrodingerComplexInput {
            real: &first.output_real,
            imaginary: &first.output_imaginary,
            ..input
        };
        let second = apply_stochastic_schrodinger_complex_step(next_input).unwrap();
        let full = apply_stochastic_schrodinger_complex_step(StochasticSchrodingerComplexInput {
            config: input.config.with_time_step(0.2).unwrap(),
            ..input
        })
        .unwrap();
        for (a, b) in second
            .output_real
            .iter()
            .chain(&second.output_imaginary)
            .zip(full.output_real.iter().chain(&full.output_imaginary))
        {
            close(f64::from(*a), f64::from(*b), 1e-7);
        }
        let recovered =
            apply_stochastic_schrodinger_complex_step(StochasticSchrodingerComplexInput {
                config: input.config.with_hopping_rate(-1.0).unwrap(),
                ..next_input
            })
            .unwrap();
        for (a, b) in recovered
            .output_real
            .iter()
            .chain(&recovered.output_imaginary)
            .zip(real.iter().chain(&imaginary))
        {
            close(f64::from(*a), f64::from(*b), 1e-7);
        }
        close(first.initial_norm_squared, first.final_norm_squared, 1e-7);
    }

    #[test]
    fn malformed_and_overflowing_complex_states_fail_closed() {
        let input = StochasticSchrodingerComplexInput {
            real: &[1.0, 2.0],
            imaginary: &[0.1, 0.2],
            potential: &[0.0, 0.0],
            standard_normal: &[0.0, 0.0],
            rows: 1,
            features: 2,
            config: config(),
        };
        assert!(
            apply_stochastic_schrodinger_complex_step(StochasticSchrodingerComplexInput {
                imaginary: &[0.0],
                ..input
            })
            .is_err()
        );
        assert!(
            apply_stochastic_schrodinger_complex_step(StochasticSchrodingerComplexInput {
                imaginary: &[f32::NAN, 0.0],
                ..input
            })
            .is_err()
        );
        assert!(backward_stochastic_schrodinger_complex_step(
            input,
            &[0.0; 2],
            &[f32::INFINITY; 2]
        )
        .is_err());
        assert!(backward_stochastic_schrodinger_complex_step(input, &[0.0; 2], &[]).is_err());
        let huge = StochasticSchrodingerComplexInput {
            real: &[f32::MAX; 2],
            imaginary: &[-f32::MAX; 2],
            config: StochasticSchrodingerConfig::default()
                .with_time_step(std::f32::consts::FRAC_PI_4)
                .unwrap(),
            ..input
        };
        assert!(apply_stochastic_schrodinger_complex_step(huge).is_err());
    }
}
