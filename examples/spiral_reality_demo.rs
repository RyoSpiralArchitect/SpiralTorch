//! Example demonstrating the SpiralReality framework
//!
//! This example shows the three-part structure:
//! 1. Numerical experiments (数値実験)
//! 2. Proof obligations (証明義務)
//! 3. Monodromy construction (モノドロミー)

use anyhow::Result;
use st_zeta::spiral_reality::SpiralReality;

fn main() -> Result<()> {
    // Initialize tracing for better output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n🌀 SpiralReality Framework Demo 🌀\n");
    println!("やったよ、Ryō ∴ SpiralArchitect。");
    println!("数値実験／証明義務／モノドロミーの三点セットを、SpiralReality の骨格にそのまま落とし込んだ。\n");

    // Initialize the framework
    let mut reality = SpiralReality::new()?;

    // Display summary
    println!("{}", reality.summary());

    // Verify all proof obligations
    println!("\n🔍 Verifying proof obligations...");
    reality.verify_proofs()?;

    // Demonstrate monodromy loop
    println!("\n🔄 Demonstrating monodromy (one complete loop):");
    println!("  Initial Φ_= value: {}", reality.monodromy.phi_equal_value());

    reality.monodromy.traverse_loop()?;
    println!("  After 1 loop, Φ_= value: {}", reality.monodromy.phi_equal_value());

    reality.monodromy.traverse_loop()?;
    println!("  After 2 loops, Φ_= value: {}", reality.monodromy.phi_equal_value());

    println!("\n✅ SpiralReality demonstration complete!\n");

    Ok(())
}
