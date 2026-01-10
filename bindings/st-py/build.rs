fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    // Rely on PyO3's extension-module configuration for platform-appropriate linking.
    // Explicitly linking `libpython` breaks manylinux compliance on Linux.
    pyo3_build_config::add_extension_module_link_args();
}
