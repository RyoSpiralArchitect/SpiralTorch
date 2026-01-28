// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use st_core::ops::zspace_round::SpectralFeatureSample;
use st_nn::{SimpleZFrame, Tensor, ZIndex, ZMetricWeights, ZRBAConfig, ZTensor, ZRBA};

fn main() -> st_nn::PureResult<()> {
    let frame = SimpleZFrame::new(3, 3, 4);
    let indices = vec![
        ZIndex {
            band: 0,
            sheet: 0,
            echo: 0,
        },
        ZIndex {
            band: 1,
            sheet: 1,
            echo: 2,
        },
        ZIndex {
            band: 2,
            sheet: 2,
            echo: 3,
        },
    ];

    let mu = Tensor::from_vec(
        3,
        6,
        vec![
            0.12, -0.05, 0.08, -0.02, 0.10, 0.07, 0.03, 0.02, -0.06, 0.01, 0.05, -0.04,
            0.02, -0.03, 0.06, 0.09, -0.01, 0.04,
        ],
    )?;
    let sigma = Tensor::from_vec(
        3,
        6,
        vec![
            0.05, 0.04, 0.03, 0.02, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.02,
        ],
    )?;
    let input = ZTensor::new(mu, sigma, indices)?;

    let stats = SpectralFeatureSample {
        sheet_index: 1,
        sheet_confidence: 0.7,
        curvature: 0.2,
        spin: 0.1,
        energy: 0.5,
    };

    let config = ZRBAConfig {
        d_model: 6,
        n_heads: 2,
        metric: ZMetricWeights::default(),
        ard: true,
        cov_rank: 3,
        gate_momentum: 0.05,
        gate_seed: 7,
        gate_use_expected: true,
    };
    let zrba = ZRBA::new(config)?;
    let (output, _cov, telemetry) = zrba.forward(&input, &frame, &stats)?;

    let targets = vec![0.10, -0.02, 0.05];
    let metrics = telemetry.metrics(&output.mu, &output.sigma, &targets, 0.9, &output.indices)?;
    let bundle = telemetry.bundle_metrics(&metrics);
    println!("{}", bundle.to_json());

    Ok(())
}

