#[cfg(not(target_arch = "wasm32"))]
#[path = "support/attention_chain.rs"]
mod support;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&pollster::block_on(support::run())?)?
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
