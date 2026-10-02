#[cfg(not(target_arch = "wasm32"))]
#[path = "support/convolution_profile_checks.rs"]
mod checks;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", pollster::block_on(checks::run())?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
