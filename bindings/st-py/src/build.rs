fn main() {
    // Cross-platform: adds the right linker args for Python extension modules.
    // - macOS: -undefined dynamic_lookup
    // - Linux/Windows: no-ops or the correct defaults for abi3/extension-module
    pyo3_build_config::add_extension_module_link_args();
}
