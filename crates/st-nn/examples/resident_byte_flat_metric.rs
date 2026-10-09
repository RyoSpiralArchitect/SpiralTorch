//! Local synthetic correctness harness, not a general model import API.
#[cfg(not(target_arch = "wasm32"))]
#[path = "support/byte_decoder.rs"]
mod support;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: resident_byte_flat_metric <local-torch-fixture.json>")?;
    if args.next().is_some() {
        return Err("expected exactly one local fixture".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    const LIMIT: u64 = 64 * 1024 * 1024;
    if !file.metadata()?.is_file() {
        return Err("fixture must be a regular file".into());
    }
    let mut input = String::new();
    file.take(LIMIT + 1).read_to_string(&mut input)?;
    if input.len() as u64 > LIMIT {
        return Err("fixture exceeds 64 MiB".into());
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.flat.metric")?;
    let report = pollster::block_on(support::run_flat_metric(runtime, &input))?;
    println!("{report}");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
