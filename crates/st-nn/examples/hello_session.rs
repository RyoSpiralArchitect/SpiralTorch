// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

// Enable ψ telemetry and CollapseDrive automation via the roundtable schedule:
// `cargo run -p st-nn --example hello_session --features "psi collapse"`

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    dataset_from_vec, Linear, MeanSquaredError, RoundtableConfig, Sequential, SpiralSession, Tensor,
};
use st_tensor::PureResult;

fn main() -> PureResult<()> {
    let caps = DeviceCaps::wgpu(32, true, 256);
    let session = SpiralSession::builder(caps)
        .with_curvature(-1.0)
        .with_hyper_learning_rate(0.05)
        .with_fallback_learning_rate(0.01)
        .build()?;

    let densities = vec![
        Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
        Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
    ];
    let weights = [0.5, 0.5];
    let barycenter = session.barycenter(&weights, &densities)?;

    let mut hypergrad = session.hypergrad(1, 2)?;
    session.align_hypergrad(&mut hypergrad, &barycenter)?;

    let mut model = Sequential::new();
    model.push(Linear::new("layer", 2, 2)?);
    session.prepare_module(&mut model)?;

    let mut trainer = session.trainer();
    #[allow(unused_mut)]
    let mut config = RoundtableConfig::default();
    #[cfg(feature = "psi")]
    {
        config = config.enable_psi();
    }
    #[cfg(feature = "collapse")]
    {
        config = config.enable_collapse();
    }
    let schedule = trainer.roundtable(1, 2, config);

    let mut loss = MeanSquaredError::new();
    let dataset = dataset_from_vec(vec![
        (
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
        ),
        (
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
        ),
    ])
    .shuffle(0xC0FFEE)
    .batched(2)
    .prefetch(2);

    let stats = trainer.train_epoch(&mut model, &mut loss, dataset, &schedule)?;
    println!(
        "roundtable avg loss {:.6} over {} batches",
        stats.average_loss, stats.batches
    );

    Ok(())
}
