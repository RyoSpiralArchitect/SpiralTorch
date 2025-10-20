fn main() {
    // Ensure PyO3 injects the appropriate linker arguments for extension modules.
    pyo3_build_config::add_extension_module_link_args();
}
