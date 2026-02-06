use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    // Rely on PyO3's extension-module configuration for platform-appropriate linking.
    // Explicitly linking `libpython` breaks manylinux compliance on Linux.
    pyo3_build_config::add_extension_module_link_args();

    // Rust unit tests compile to executables, which need `libpython` at link time.
    //
    // When we build the extension module for distribution we intentionally *don't* link
    // `libpython`. For local development (including `cargo test`) we add a macOS-only
    // framework link so Rust test binaries can resolve the Python symbols.
    //
    // Maturin sets `PYO3_BUILD_EXTENSION_MODULE=1` when building the wheel / extension,
    // so we use that to avoid embedding a hard dependency on the Python framework in
    // the distributed artefact.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && !env_var_truthy("PYO3_BUILD_EXTENSION_MODULE")
    {
        if let Some((framework_prefix, framework_name)) = python_framework_config() {
            println!("cargo:rustc-link-search=framework={framework_prefix}");
            println!("cargo:rustc-link-lib=framework={framework_name}");
        }
    }
}

fn python_framework_config() -> Option<(String, String)> {
    let python = std::env::var("PYO3_PYTHON")
        .or_else(|_| std::env::var("PYTHON_SYS_EXECUTABLE"))
        .unwrap_or_else(|_| "python3".to_string());

    let framework_prefix = python_sysconfig_var(&python, "PYTHONFRAMEWORKPREFIX")?;
    let framework_name = python_sysconfig_var(&python, "PYTHONFRAMEWORK")?;
    if framework_prefix.trim().is_empty() || framework_name.trim().is_empty() {
        return None;
    }
    Some((framework_prefix, framework_name))
}

fn env_var_truthy(key: &str) -> bool {
    match std::env::var(key) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn python_sysconfig_var(python: &str, key: &str) -> Option<String> {
    let output = Command::new(python)
        .env("PYTHONNOUSERSITE", "1")
        .args([
            "-S",
            "-s",
            "-c",
            &format!(
                "import sysconfig; value = sysconfig.get_config_var({key:?}) or ''; print(value)"
            ),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
