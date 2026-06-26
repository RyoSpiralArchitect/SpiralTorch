// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Coherence model-zoo: ZSpaceVae reconstruction + Mellin basis projection.

use nalgebra::DVector;
use st_nn::{MellinBasis, ZSpaceVae, ZSpaceVaeOptimizerKind};

fn main() -> st_nn::PureResult<()> {
    let input_dim = 8usize;
    let latent_dim = 3usize;
    let mut vae = ZSpaceVae::new(input_dim, latent_dim, 42);
    vae.configure_optimizer(
        ZSpaceVaeOptimizerKind::Adam,
        0.9,
        0.999,
        1e-8,
        0.99,
        Some(5.0),
    )?;

    let basis = MellinBasis::new(vec![1.0, 0.5, 2.0, 1.25, 0.75, 1.5, 1.0, 0.9]);
    let input = DVector::from_vec(vec![0.35, -0.12, 0.77, 0.05, -0.28, 0.44, 0.10, -0.06]);
    let projected = basis.project(&input);

    println!("input_dim={input_dim} latent_dim={latent_dim}");
    println!(
        "input_norm={:.6} projected_norm={:.6}",
        input.norm(),
        projected.norm()
    );

    let mut last_recon = None;
    let batch = vec![projected.clone(); 4];
    for step in 0..12usize {
        let stats = vae.train_batch(&batch, 1e-2, 1e-3)?;
        let recon = stats.recon_loss;
        let kl = stats.kl_loss;
        let delta = last_recon.map(|prev| recon - prev);
        println!(
            "step={step:02} recon_loss={recon:.6} kl_loss={kl:.6} weighted_loss={:.6} grad_l2={:.6} update_l2={:.6} delta={}",
            stats.weighted_loss,
            stats.gradient_l2,
            stats.update_l2,
            delta
                .map(|v| format!("{v:+.6}"))
                .unwrap_or_else(|| "—".to_string())
        );
        last_recon = Some(recon);
    }

    let final_state = vae.forward_mean(&projected)?;
    let error = (&final_state.reconstruction - &projected).norm();
    println!("final_error_norm={error:.6}");

    Ok(())
}
