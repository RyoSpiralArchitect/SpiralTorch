fn main() {
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.starts_with("wasm32") {
        return;
    }

    let rustflags = match std::env::var("CARGO_ENCODED_RUSTFLAGS") {
        Ok(encoded) if !encoded.is_empty() => encoded.split('\x1f').collect::<Vec<_>>().join(" "),
        _ => std::env::var("RUSTFLAGS").unwrap_or_default(),
    };
    let looks_like_vcpkg = rustflags.contains("vcpkg/installed") || rustflags.contains("vcpkg\\installed");
    let links_host_archives =
        rustflags.contains("-l archive") || rustflags.contains("-larchive") || rustflags.contains("libarchive");
    if !(looks_like_vcpkg || links_host_archives) {
        return;
    }

    println!("cargo:warning=Detected host linker flags while building a wasm32 target.");
    println!("cargo:warning=These flags will break wasm linking (rust-lld will try to consume native .a archives).");
    println!("cargo:warning=Fix: unset the flags for wasm builds, e.g.:");
    println!(
        "cargo:warning=  env -u RUSTFLAGS -u LIBRARY_PATH -u PKG_CONFIG_PATH wasm-pack build bindings/st-wasm --target web"
    );
    println!("cargo:warning=  rustflags={rustflags}");

    if let Ok(value) = std::env::var("LIBRARY_PATH") {
        if value.contains("vcpkg/installed") || value.contains("vcpkg\\installed") {
            println!("cargo:warning=  LIBRARY_PATH={value}");
        }
    }

    if let Ok(value) = std::env::var("PKG_CONFIG_PATH") {
        if value.contains("vcpkg/installed") || value.contains("vcpkg\\installed") {
            println!("cargo:warning=  PKG_CONFIG_PATH={value}");
        }
    }
}
