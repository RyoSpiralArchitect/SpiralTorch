// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Coherence model-zoo: ZSpaceVae reconstruction + Mellin basis projection.

use nalgebra::DVector;
use st_nn::{MellinBasis, ZSpaceVae};

fn main() -> st_nn::PureResult<()> {
    let input_dim = 8usize;
    let latent_dim = 3usize;
    let mut vae = ZSpaceVae::new(input_dim, latent_dim, 42);

    let basis = MellinBasis::new(vec![1.0, 0.5, 2.0, 1.25, 0.75, 1.5, 1.0, 0.9]);
    let input = DVector::from_vec(vec![0.35, -0.12, 0.77, 0.05, -0.28, 0.44, 0.10, -0.06]);
    let projected = basis.project(&input);

    println!("input_dim={input_dim} latent_dim={latent_dim}");
    println!("input_norm={:.6} projected_norm={:.6}", input.norm(), projected.norm());

    let mut last_recon = None;
    for step in 0..12usize {
        let state = vae.forward(&projected);
        let recon = state.stats.recon_loss;
        let kl = state.stats.kl_loss;
        let elbo = state.stats.evidence_lower_bound;
        let delta = last_recon.map(|prev| recon - prev);
        println!(
            "step={step:02} recon_loss={recon:.6} kl_loss={kl:.6} elbo={elbo:.6} delta={}",
            delta.map(|v| format!("{v:+.6}")).unwrap_or_else(|| "—".to_string())
        );
        last_recon = Some(recon);
        vae.refine_decoder(&state, 1e-2);
    }

    let final_state = vae.forward(&projected);
    let error = (&final_state.reconstruction - &projected).norm();
    println!("final_error_norm={error:.6}");

    Ok(())
}

